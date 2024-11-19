import torch
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader
from DataLoader.dataset import FinancialDataset
from sklearn.metrics import accuracy_score
from model import StockPredictor


BATCH_SIZE=16
NEUTRAL_WINDOW=0.005

def convert_to_class_indices(labels, neutral_window):
    class_labels = torch.empty(labels.size(0), device=labels.device, dtype=torch.long)
    class_labels[labels < -neutral_window] = 0  # Negative class
    class_labels[(labels >= -neutral_window) & (labels <= neutral_window)] = 1  # Neutral class
    class_labels[labels > neutral_window] = 2  # Positive class

    return class_labels


def train_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    total_examples = 0
    all_preds = []
    all_labels = []

    for batch_num, batch in enumerate(data_loader):
        print("BATCH {}".format(batch_num))
        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        financial_data = batch['numerical_features'].to(device)
        labels = batch['label'].to(device)
        labels = convert_to_class_indices(labels, NEUTRAL_WINDOW)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)

        print(labels.shape)

        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_examples += input_ids.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(true_labels)

    avg_loss = total_loss / total_examples
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


if __name__ == "__main__":
    file_path = "analyst_ratings_processed_filtered_combined.csv"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FinancialDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = StockPredictor()
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_epoch(model, dataloader, loss_fn, optimizer, device)