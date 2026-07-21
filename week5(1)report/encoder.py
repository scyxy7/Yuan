"""
encoder.py
Transformer-based Encoder for EVRP-TW
Maps raw node features → contextual embeddings
Compatible with AttentionDecoder + FFP
"""

import torch
import torch.nn as nn
import math


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    Helps the model understand spatial / sequential layout
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


# =========================
# Transformer Encoder
# =========================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=6,        # x, y, demand, e, l, node_type
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
        x = self.input_proj(node_features)   # projection
        x = self.pos_encoder(x)               # add position info
        embeddings = self.transformer(x)      # self-attention
        return embeddings


# =========================
# Helper: Build node features from env
# =========================
def build_node_features(env):
    """
    Convert env.NODES into a tensor suitable for TransformerEncoder

    Feature order:
    [x, y, demand, tw_start, tw_end, node_type]

    Returns:
        features: Tensor[1, num_nodes, 6]
    """
    features = []
    for i in sorted(env.nodes.keys()):
        x, y, demand, e, l, ntype = env.nodes[i]
        features.append([x, y, demand, e, l, ntype])

    features = torch.tensor(features, dtype=torch.float32)
    return features.unsqueeze(0)  # [1, num_nodes, 6]


# =========================
# Sanity check
# =========================
if __name__ == "__main__":
    from env import EVRPTWEnv

    env = EVRPTWEnv()
    encoder = TransformerEncoder()

    node_features = build_node_features(env)
    embeddings = encoder(node_features)

    print("Node features shape:", node_features.shape)
    print("Embeddings shape:", embeddings.shape)
    print("Embedding norm:", embeddings.norm(dim=-1))