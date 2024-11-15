import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from utils import generate_ticker_encoding

BATCH_SIZE=16
MAX_TICKER_LENGTH=4
STRING_HEADERS = {"Date", "Title", "Ticker"}

class FinancialDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.numerical_headers_past = [header for header in list(self.data) if header not in STRING_HEADERS and "+" not in header]
        self.label_header = "D+1 High"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Numerical features
        numerical_features = torch.tensor(
            row[self.numerical_headers_past].values.astype(float), dtype=torch.float64
        )
        
        # Text features (headline)
        headline = row['Title']
        encoded_text = self.tokenizer(
            headline,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        ticker = row['Ticker']
        date = pd.to_datetime(row['Date'])

        date_features = [date.year, date.month, date.day]
        ticker_feature = generate_ticker_encoding(ticker, MAX_TICKER_LENGTH)
        
        # Labels (if any, optional)
        label = row.get(self.label_header, None)  # Replace 'label' with your actual label column
        if label is not None:
            label = torch.tensor(label, dtype=torch.float64)

        return {
            "numerical_features": numerical_features,
            "text_input_ids": encoded_text['input_ids'].squeeze(0),
            "text_attention_mask": encoded_text['attention_mask'].squeeze(0),
            "label": label if label is not None else None,
            "date": torch.tensor(date_features),
            "ticker": torch.tensor(ticker_feature)
        }

# How to use:
"""
if __name__ == "__main__":
    file_path = "../Data/analyst_ratings_processed_filtered_combined.csv"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = FinancialDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for (batch_idx, data) in enumerate(dataloader):
        print(data)
        break
""" 
