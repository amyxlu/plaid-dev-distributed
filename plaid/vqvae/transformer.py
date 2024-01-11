# import torch
# import torch.nn as nn


# class TransformerEncoder(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers, dropout=0.1, activation="gelu", dtype=torch.float32):
#         super().__init__()
#         layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, dtype)
#         self.transformer_encoder = nn.TransformerEncoder(layer, num_encoder_layers)

#     def forward(self, src, is_causal=True):
#         return self.transformer_encoder(src, is_causal)


