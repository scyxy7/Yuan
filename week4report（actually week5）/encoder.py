"""
encoder.py
Transformer-based Encoder for EVRP-TW
Maps raw node features to contextual embeddings.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding
    (helps the model understand node order / layout)
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=6,        # x, y, demand, tw_start, tw_end, node_type
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.hidden_dim = hidden_dim

    def forward(self, node_features):
        """
        Args:
            node_features: Tensor[batch, num_nodes, input_dim]
        Returns:
            embeddings: Tensor[batch, num_nodes, hidden_dim]
        """
        x = self.input_proj(node_features)      # projection
        x = self.pos_encoder(x)                 # positional encoding
        embeddings = self.transformer(x)        # self-attention
        return embeddings