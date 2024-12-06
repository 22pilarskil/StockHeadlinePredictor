import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoTokenizer
import numpy as np
from DataLoader.dataset import WINDOW_SIZE



class BaselineModel(nn.Module):
    def __init__(self, modelname='yiyanghkust/finbert-tone', num_labels=3, hidden_dim=64):
        super(BaselineModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.finbert = BertForSequenceClassification.from_pretrained(modelname, num_labels=3)
        self.financial_projection = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.linear = nn.Linear(hidden_dim + 1, num_labels)

    # input_ids are the tokenized sentence, or input can be title and then use tokenizer in the forward method.
    def forward(self, input_ids, financial_data):
        financial_data = financial_data.float()
        sentiment_scores = self.finbert(input_ids)[0]
        sentiment_scores = np.argmax(sentiment_scores.detach().numpy(), axis=-1) # (batch_size,)
        sentiment_scores = torch.from_numpy(sentiment_scores)
        sentiment_scores = sentiment_scores.unsqueeze(-1)  # batch_size x 1
        financial_data = self.financial_projection(financial_data) # batch_size x seq_len x 64
        seq_len = financial_data.size(1)
        sentiment_scores= sentiment_scores.unsqueeze(1).expand(-1, seq_len, -1)
        concatenated = torch.cat((financial_data, sentiment_scores), dim=-1)
        cls_output = concatenated[:,0,:]
        logits = self.linear(cls_output)
        return logits