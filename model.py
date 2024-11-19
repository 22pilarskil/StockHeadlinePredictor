import torch
from torch import nn
from transformers import BertModel
from torch.nn import MultiheadAttention, Linear, TransformerEncoder, TransformerEncoderLayer
from DataLoader.dataset import WINDOW_SIZE

class StockPredictor(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=3, hidden_dim=64):
        super(StockPredictor, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        self.bert_projection = Linear(768, hidden_dim)
        
        self.financial_projection = nn.Sequential(
            Linear(WINDOW_SIZE, 32),
            nn.ReLU(),
            Linear(32, hidden_dim)
        )

        self.cross_attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=hidden_dim * 2),
            num_layers=2
        )

        self.classifier = Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, financial_data):

        financial_data = financial_data.float()

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_length, 768)

        bert_sequence_output = self.bert_projection(bert_sequence_output)  # Shape: (batch_size, seq_length, 64)

        financial_data = self.financial_projection(financial_data)  # Shape: (batch_size, seq_length, 64)

        attn_output, _ = self.cross_attention(bert_sequence_output, financial_data, financial_data)  # Shape: (batch_size, seq_length, 64)

        transformer_output = self.transformer_encoder(attn_output)  # Shape: (batch_size, seq_length, 64)

        cls_token_output = transformer_output[:, 0, :]  # Shape: (batch_size, 64)
        logits = self.classifier(cls_token_output)
                
        return logits
