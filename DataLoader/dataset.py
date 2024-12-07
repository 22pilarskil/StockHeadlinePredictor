import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from DataLoader.utils import generate_ticker_encoding
from sklearn.preprocessing import StandardScaler
import pickle


BATCH_SIZE=16
MAX_TICKER_LENGTH=4
STRING_HEADERS = {"Date", "Title", "Ticker"}
NUM_FEATURES=8
WINDOW_SIZE=10

class FinancialDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128, scaler_dict=None):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.numerical_headers_past = [header for header in list(self.data) if header not in STRING_HEADERS and "+" not in header]
        self.label_header = "D+1 High"

        self.feature_groups = {}
        for header in self.numerical_headers_past:
            feature_type = header.split()[1]  # e.g., "Open" from "D-1 Open"
            if feature_type not in self.feature_groups:
                self.feature_groups[feature_type] = []
            self.feature_groups[feature_type].append(header)

        self.scaler_dict = {}
        for feature_type, columns in self.feature_groups.items():
            if scaler_dict and feature_type in scaler_dict:
                self.scaler_dict[feature_type] = scaler_dict[feature_type]
            else:
                feature_data = self.data[columns].values.flatten().reshape(-1, 1)
                scaler = StandardScaler()
                scaler.fit(feature_data)
                self.scaler_dict[feature_type] = scaler

        with open('scaler_dict.pkl', 'wb') as f:
            pickle.dump(scaler_dict, f)
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Scale numerical features
        numerical_features = []
        for feature_type, columns in self.feature_groups.items():
            raw_data = row[columns].values.astype(float).reshape(-1, 1)  # Extract raw data for this feature type
            scaled_data = self.scaler_dict[feature_type].transform(raw_data).flatten()  # Scale and flatten
            numerical_features.append(scaled_data)

        # Stack scaled features into (NUM_FEATURES, WINDOW_SIZE)
        numerical_features = np.stack(numerical_features, axis=0)
        numerical_features = torch.tensor(numerical_features, dtype=torch.float64)
        
        # Text features (headline)
        headline = row['Title']
        encoded_text = self.tokenizer(
            headline,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Other features
        ticker = row['Ticker']
        date = pd.to_datetime(row['Date'])
        date_features = [date.year, date.month, date.day]

        # Label (if applicable)
        label = row.get(self.label_header)

        return {
            "numerical_features": numerical_features,
            "text_input_ids": encoded_text['input_ids'].squeeze(0),
            "text_attention_mask": encoded_text['attention_mask'].squeeze(0),
            "label": label if label is not None else None,
            "date": torch.tensor(date_features),
            "ticker": ticker,
            "headlines": headline,
        }

    # def __getitem__(self, idx):
    #     row = self.data.iloc[idx]
        
    #     # Numerical features
    #     numerical_features = torch.tensor(
    #         row[self.numerical_headers_past].values.astype(float), dtype=torch.float64
    #     ).reshape(NUM_FEATURES, WINDOW_SIZE)
        
    #     # Text features (headline)
    #     headline = row['Title']
    #     encoded_text = self.tokenizer(
    #         headline,
    #         max_length=self.max_len,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors="pt"
    #     )

    #     ticker = row['Ticker']
    #     date = pd.to_datetime(row['Date'])

    #     date_features = [date.year, date.month, date.day]
    #     ticker_feature = ticker #generate_ticker_encoding(ticker, MAX_TICKER_LENGTH)
        
    #     # Labels (if any, optional)
    #     label = row.get(self.label_header)  # Replace 'label' with your actual label column

    #     # print(
    #     #     f"numerical_features: {numerical_features.shape}, "
    #     #     f"text_input_ids: {encoded_text['input_ids'].squeeze(0).shape}, "
    #     #     f"text_attention_mask: {encoded_text['attention_mask'].squeeze(0).shape}, "
    #     #     f"label: {label}, "
    #     #     f"date: {torch.tensor(date_features).shape}, "
    #     #     f"ticker: {ticker_feature}"
    #     # )

    #     return {
    #         "numerical_features": numerical_features,
    #         "text_input_ids": encoded_text['input_ids'].squeeze(0),
    #         "text_attention_mask": encoded_text['attention_mask'].squeeze(0),
    #         "label": label if label is not None else None,
    #         "date": torch.tensor(date_features),
    #         "ticker": ticker_feature,
    #         "headlines": headline
    #     }

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
