import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer
import numpy as np
from DataLoader.dataset import WINDOW_SIZE



class BaselineModel(nn.Module):
    def __init__(self, modelname='yiyanghkust/finbert-tone', num_labels=3, hidden_dim=64):
        self.tokenizer = BertTokenizer(modelname)
        self.finbert = BertForSequenceClassification(modelname, num_labels=3)
        self.financial_projection = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.linear(hidden_dim + 1, num_labels)

    # input_ids are the tokenized sentence, or input can be title and then use tokenizer in the forward method.
    def forward(self, sentences, financial_data):
        financial_data = financial_data.float()
        inputs_ids = self.tokenizer(sentences, return_tensors='pt', padding=True)
        sentiment_scores = self.finbert(**inputs_ids)[0]
        sentiment_scores = np.argmax(sentiment_scores.detach().numpy(), axis=-1) # 1 x batch_size
        sentiment_scores = sentiment_scores.transpose(0,1) # batch_size x 1
        financial_data = self.financial_projection(financial_data) # batch_size x seq_len x 64
        seq_len = financial_data.size(1)
        sentiment_scores= sentiment_scores.unsqueeze(1).expand(-1, seq_len, -1)
        concatenated = torch.cat((financial_data, sentiment_scores), dim=-1)
        logits = self.linear(concatenated)