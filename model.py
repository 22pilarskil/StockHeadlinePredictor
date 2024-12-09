import torch
from torch import nn
from transformers import BertModel
from torch.nn import MultiheadAttention, Linear, TransformerEncoder, TransformerEncoderLayer
from DataLoader.dataset import NUM_FEATURES

class StockPredictor(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=3):
        super(StockPredictor, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_projection = Linear(768, NUM_FEATURES)

        self.cross_attention = MultiheadAttention(embed_dim=NUM_FEATURES, num_heads=4, batch_first=True)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=NUM_FEATURES, nhead=4, batch_first=True, dim_feedforward=NUM_FEATURES * 2),
            num_layers=2
        )

        self.classifier = Linear(NUM_FEATURES, num_labels)

    def forward(self, input_ids, attention_mask, financial_data):

        financial_data = financial_data.float().permute(0, 2, 1)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence_output = bert_outputs.last_hidden_state[:, 0, :].unsqueeze(1) 

        bert_sequence_output = self.bert_projection(bert_sequence_output)


        attn_output, _ = self.cross_attention(bert_sequence_output, financial_data, financial_data)

        transformer_output = self.transformer_encoder(attn_output)

        cls_token_output = transformer_output[:, 0, :]
        logits = self.classifier(cls_token_output)
                
        return logits
    

