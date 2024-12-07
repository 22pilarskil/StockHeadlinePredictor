import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from DataLoader.utils import generate_ticker_encoding

BATCH_SIZE=16
MAX_TICKER_LENGTH=4
STRING_HEADERS = {"Date", "Title", "Ticker"}
NUM_FEATURES=8
WINDOW_SIZE=10

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
        ).reshape(NUM_FEATURES, WINDOW_SIZE)
        
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
        ticker_feature = ticker #generate_ticker_encoding(ticker, MAX_TICKER_LENGTH)
        
        # Labels (if any, optional)
        label = row.get(self.label_header)  # Replace 'label' with your actual label column

        # print(
        #     f"numerical_features: {numerical_features.shape}, "
        #     f"text_input_ids: {encoded_text['input_ids'].squeeze(0).shape}, "
        #     f"text_attention_mask: {encoded_text['attention_mask'].squeeze(0).shape}, "
        #     f"label: {label}, "
        #     f"date: {torch.tensor(date_features).shape}, "
        #     f"ticker: {ticker_feature}"
        # )

        return {
            "numerical_features": numerical_features,
            "text_input_ids": encoded_text['input_ids'].squeeze(0),
            "text_attention_mask": encoded_text['attention_mask'].squeeze(0),
            "label": label if label is not None else None,
            "date": torch.tensor(date_features),
            "ticker": ticker_feature,
            "headlines": headline
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
