import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import pickle

from DataLoader.dataset import FinancialDataset
from model import StockPredictor
from model_baseline import BaselineModel
from util import *

def plot_confusion_matrix(predictions, true_labels, title="Confusion Matrix", ax=None, cm_output_file="cm_output.pkl"):

    cm = confusion_matrix(true_labels, predictions)
    with open(cm_output_file, "wb") as file:
        pickle.dump(cm, file)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

def plot_confidence_distribution(correct_preds, incorrect_preds, confidence, ax):
    correct_confidence = confidence[correct_preds]
    incorrect_confidence = confidence[incorrect_preds]

    ax.hist(correct_confidence, bins=20, alpha=0.7, label='Correct Predictions')
    ax.hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect Predictions')
    ax.legend(loc='best')
    ax.set_title("Confidence Distribution of Correct vs. Incorrect Predictions")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of Predictions")


def evaluate_model(model, data_loader, loss_function, device, figure_path, cm_output_file):
    model.eval()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    all_confidences = []  # Track confidence scores
    correct_preds = []
    incorrect_preds = []

    with torch.no_grad(): 
        for batch_num, batch in enumerate(data_loader):
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            financial_data = batch['numerical_features'].to(device)
            headlines = batch['headlines']

            labels = batch['label'].to(device)
            labels = convert_to_class_indices(labels, NEUTRAL_WINDOW)

            if IS_BASELINE:
                logits = model(headlines=headlines, financial_data=financial_data, device=device)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)

            loss = loss_function(logits, labels)
            total_loss += loss.item()
            total_batches += 1

            # Compute probabilities and confidence (max probability of the predicted class)
            probabilities = F.softmax(logits, dim=1)
            confidences, _ = torch.max(probabilities, dim=1)

            # Collect predictions, true labels, and confidence scores
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)
            all_confidences.extend(confidences.cpu().numpy())  # Store confidence for weighted metrics

            # Track correct and incorrect predictions
            correct_preds.extend(np.where(preds == true_labels)[0])
            incorrect_preds.extend(np.where(preds != true_labels)[0])

            if EARLY_EXIT is not None and batch_num > EARLY_EXIT:
                break

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    plot_confusion_matrix(all_preds, all_labels, title=f'Confusion Matrix {"(baseline)" if IS_BASELINE else "(proposed)"}', ax=ax[0], cm_output_file=cm_output_file)

    plot_confidence_distribution(correct_preds, incorrect_preds, np.array(all_confidences), ax=ax[1])

    weighted_accuracy = np.sum((np.array(all_preds) == np.array(all_labels)) * np.array(all_confidences)) / np.sum(all_confidences)

    weighted_f1 = weighted_f1_score(np.array(all_preds), np.array(all_labels), np.array(all_confidences))

    avg_loss = total_loss / total_batches

    # Print metrics
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Weighted Accuracy: {weighted_accuracy:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("-----------------------------------\n")

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.show()

    return avg_loss, weighted_accuracy, weighted_f1


# Function to calculate weighted F1 score
def weighted_f1_score(predictions, true_labels, confidence):
    f1_scores = []

    # Calculate F1 score for each sample (this is the per-sample F1 score)
    for i in range(len(predictions)):
        f1 = f1_score([true_labels[i]], [predictions[i]], average='weighted')
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)

    # Weight the F1 scores by confidence and compute the weighted average
    weighted_f1 = np.sum(f1_scores * confidence) / np.sum(confidence)
    return weighted_f1



# Main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate a stock predictor model.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="analyst_ratings_processed_filtered_combined.csv",
        help="Path to the dataset file."
    )
    parser.add_argument("--early_exit", type=int, default=None, help="Exit training after n batches for debugging.")
    parser.add_argument("--checkpoint_path", type=str, default="model_checkpoint.pth")
    parser.add_argument("--figure_path", type=str, default="figure.png")
    parser.add_argument("--use_baseline", action='store_true')
    parser.add_argument("--cm_output_file", default="cm_output.pkl")
    args = parser.parse_args()

    file_path = args.file_path
    IS_BASELINE = args.use_baseline
    EARLY_EXIT = args.early_exit
    figure_path = args.figure_path
    checkpoint_path = args.checkpoint_path
    cm_output_file = args.cm_output_file

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FinancialDataset(file_path, tokenizer)

    dataset_size = len(dataset)
    test_ratio = 0.2
    train_size = int((1 - test_ratio) * dataset_size)
    test_size = dataset_size - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if IS_BASELINE:
        model = BaselineModel()
    else:
        model = StockPredictor()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = AdamW(model.parameters(), lr=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _, NEUTRAL_WINDOW = load_model(checkpoint_path, model, optimizer)
    model = model.to(device)

    # Evaluate the model and plot both confusion matrix and confidence distribution
    avg_loss, weighted_accuracy, weighted_f1 = evaluate_model(model, test_dataloader, loss_fn, device, figure_path, cm_output_file)