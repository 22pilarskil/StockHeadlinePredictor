import torch
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader
from DataLoader.dataset import FinancialDataset
from sklearn.metrics import accuracy_score, f1_score
from model import StockPredictor
import argparse

BATCH_SIZE = None
NEUTRAL_WINDOW = None
PRINT_EVERY = None
EARLY_EXIT = None
EPOCHS = None

def convert_to_class_indices(labels, neutral_window):
    class_labels = torch.empty(labels.size(0), device=labels.device, dtype=torch.long)
    class_labels[labels < -neutral_window] = 0  # Negative class
    class_labels[(labels >= -neutral_window) & (labels <= neutral_window)] = 1  # Neutral class
    class_labels[labels > neutral_window] = 2  # Positive class

    return class_labels


def save_model(epoch, model, optimizer, neutral_window, model_save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'neutral_window': neutral_window
    }

    torch.save(checkpoint, model_save_path)

def load_model(model_save_path, model, optimizer):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    neutral_window = checkpoint['neutral_window']
    return model, optimizer, current_epoch, neutral_window



def train_epoch(model, data_loader, loss_function, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_examples = 0
    all_preds = []
    all_labels = []

    for batch_num, batch in enumerate(data_loader):
        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        financial_data = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        labels = convert_to_class_indices(labels, NEUTRAL_WINDOW)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)

        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        if batch_num % PRINT_EVERY == 0:
            print("EPOCH {}: BATCH {}/{}, LOSS: {:.4f}".format(epoch, batch_num, len(data_loader), loss.item() / input_ids.size(0)))
            print("LOGITS MEAN", logits.mean(dim=0))

        total_loss += loss.item()
        total_examples += input_ids.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(true_labels)
        if EARLY_EXIT is not None and batch_num > EARLY_EXIT:
            break

    avg_loss = total_loss / total_examples
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, data_loader, loss_function, device, epoch):
    model.eval()
    total_loss = 0
    total_examples = 0
    all_preds = []
    all_labels = []
    positive_samples = 0
    negative_samples = 0
    neutral_samples = 0

    with torch.no_grad(): 
        for batch_num, batch in enumerate(data_loader):
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            financial_data = batch['numerical_features'].to(device)

            labels = batch['label'].to(device)
            labels = convert_to_class_indices(labels, NEUTRAL_WINDOW)
            counts = torch.bincount(labels, minlength=3) 
            negative_samples += counts[0]
            neutral_samples += counts[1]
            positive_samples += counts[2]

            logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)

            loss = loss_function(logits, labels)
            total_loss += loss.item()
            total_examples += input_ids.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true_labels)
            if EARLY_EXIT is not None and batch_num > EARLY_EXIT:
                break

    avg_loss = total_loss / total_examples
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  
    print("EPOCH {}:\nAverage Loss: {}\nAccuracy: {}\nF1 Score: {}".format(epoch, avg_loss, accuracy, f1))
    print("Data distribution:\nNegative Samples: {}\nNeutral Samples: {}\nNegative Samples: {}".format(negative_samples, neutral_samples, positive_samples))
    print("Negative %: {:.4f}\nNeutral %: {:.4f}\nPositive %: {:.4f}".format(negative_samples / total_examples, neutral_samples / total_examples, positive_samples / total_examples))
    print("-----------------------------------\n")

    return avg_loss, accuracy, f1



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate a stock predictor model.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="analyst_ratings_processed_filtered_combined.csv",
        help="Path to the dataset file."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--neutral_window", type=float, default=0.005, help="Neutral window for label classification.")
    parser.add_argument("--print_every", type=int, default=1, help="Print loss every n batches.")
    parser.add_argument("--early_exit", type=int, default=None, help="Exit training after n batches for debugging.")
    parser.add_argument("--epochs", type=int, default=0, help="Number of epochs")
    args = parser.parse_args()

    file_path = args.file_path
    BATCH_SIZE = args.batch_size
    NEUTRAL_WINDOW = args.neutral_window
    PRINT_EVERY = args.print_every
    EARLY_EXIT = args.early_exit
    EPOCHS = args.epochs

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FinancialDataset(file_path, tokenizer)

    dataset_size = len(dataset)
    test_ratio = 0.2
    train_size = int((1 - test_ratio) * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = StockPredictor()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epoch = 0
    avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch)
    for i in range(EPOCHS):
        epoch += 1
        train_epoch(model, train_dataloader, loss_fn, optimizer, device, epoch)
        avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch)
        save_model(epoch, model, optimizer, NEUTRAL_WINDOW, 'model_checkpoint.pth')

    model, optimizer, epoch, neutral_window = load_model('model_checkpoint.pth', model, optimizer)
    print(neutral_window)

# Example usage:
# python3 train.py --print_every 100 --epochs 5 --batch_size 32