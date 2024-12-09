import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt


from DataLoader.dataset import FinancialDataset
from util import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and evaluate a stock predictor model.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="analyst_ratings_processed_filtered_combined.csv",
        help="Path to the dataset file."
    )
    args = parser.parse_args()

    file_path = args.file_path

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

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)


    # Initialize accumulators for financial data
    financial_data_sum = None
    financial_data_squared_sum = None
    financial_data_min = None
    financial_data_max = None
    total_samples = 0

    all_financial_data = []

    for batch_num, batch in enumerate(train_dataloader):
        print("BATCH: {}/{}".format(batch_num, len(train_dataloader)))
        financial_data = batch["numerical_features"]  # Shape: (batch_size, num_features, window_len)
        batch_size, num_features, window_len = financial_data.shape
        
        # Reshape to combine samples from batch and window length
        reshaped_data = financial_data.view(-1, num_features)  # Shape: (batch_size * window_len, num_features)
        all_financial_data.append(reshaped_data)

    # Concatenate all the data
    all_financial_data = torch.cat(all_financial_data, dim=0)  # Shape: (total_samples, num_features)

    # Plot histograms for each feature
    num_features = all_financial_data.shape[1]
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))

    for i in range(num_features):
        feature_data = all_financial_data[:, i].numpy()  # Convert to NumPy for easier plotting
        ax = axes[i] if num_features > 1 else axes  # Handle single plot case
        ax.hist(feature_data, bins=30, color='blue', alpha=0.7)
        ax.set_title(f"Feature {i + 1} Histogram")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()