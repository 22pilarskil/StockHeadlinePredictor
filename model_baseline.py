import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer

class BaselineModel(nn.Module):
    def __init__(self, modelname='yiyanghkust/finbert-tone', num_labels=3, hidden_dim=32, num_financial_features=8, lstm_layers=2):
        super(BaselineModel, self).__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained(modelname)

        self.finbert = BertForSequenceClassification.from_pretrained(modelname, num_labels=3)
        for param in self.finbert.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=num_financial_features + 3,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, num_labels)

    def forward(self, headlines, financial_data, device):
        financial_data = financial_data.float()

        encoded_text = self.tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt" 
        )
        input_ids = encoded_text['input_ids'].squeeze(0).to(device)
        attention_mask = encoded_text['attention_mask'].squeeze(0).to(device)

        sentiment_scores = self.finbert(input_ids=input_ids, attention_mask=attention_mask)[0]

        financial_data = financial_data.permute(0, 2, 1)
        _, seq_length, _ = financial_data.size()
        repeated_sentiments = sentiment_scores.unsqueeze(1).repeat(1, seq_length, 1)
        combined_input = torch.cat([financial_data, repeated_sentiments], dim=-1)
        lstm_out, _ = self.lstm(combined_input)

        final_output = lstm_out[:, -1, :]
        logits = self.linear(final_output)

        return logits