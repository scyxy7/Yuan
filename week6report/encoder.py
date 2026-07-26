"""
encoder.py
Transformer-based Encoder for EVRPTW
Compatible with PPO + AttentionDecoder
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    Standard practice for Transformer with non-sequential inputs
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
        """
        x: [batch, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=6,        # [x, y, demand, e, l, node_type]
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
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.hidden_dim = hidden_dim

    def forward(self, node_features):
        """
        node_features:
            - single:  [num_nodes, input_dim]
            - batch:   [batch, num_nodes, input_dim]

        Returns:
            embeddings: same shape as input, but encoded
        """
        # ---- Add batch dimension if needed ----
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)

        x = self.input_proj(node_features)
        x = self.pos_encoder(x)
        embeddings = self.transformer(x)

        return embeddings


# =========================
# Sanity check (optional)
# =========================
if __name__ == "__main__":
    from env import EVRPTWEnv, NODES

    env = EVRPTWEnv(NODES)
    encoder = TransformerEncoder()

    node_list = [env.nodes[i] for i in range(env.num_nodes)]
    node_features = torch.tensor(node_list, dtype=torch.float32)

    embeddings = encoder(node_features)
    print("✅ Encoder output shape:", embeddings.shape)
    print("✅ Hidden dim:", encoder.hidden_dim)