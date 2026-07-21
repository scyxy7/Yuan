# decoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Query: current node embedding + SoC
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        # Key: all node embeddings
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

        # ✅ Value Head (Critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        # Use normalized query/key similarity for stable logits
        self.eps = 1e-6

    def forward(self, node_embeddings, cur_node_idx, soc, ffp_logits_mask):
        # ---------- Query ----------
        cur_emb = node_embeddings[cur_node_idx]
        soc_tensor = torch.tensor([soc], dtype=torch.float32, device=node_embeddings.device)
        query_raw = self.query_fc(torch.cat([cur_emb, soc_tensor]))
        query = F.normalize(query_raw, dim=-1, eps=self.eps).unsqueeze(0)

        # ---------- Keys ----------
        keys_raw = self.key_fc(node_embeddings)  # [num_nodes, hidden_dim]
        keys = F.normalize(keys_raw, dim=-1, eps=self.eps)

        # ---------- Attention Logits ----------
        logits = (query @ keys.T).squeeze(0)

        # ---------- FFP Hard Mask ----------
        if isinstance(ffp_logits_mask, np.ndarray):
            ffp_mask_tensor = torch.from_numpy(ffp_logits_mask).to(logits.device)
        else:
            ffp_mask_tensor = ffp_logits_mask.to(logits.device)

        invalid_mask = ffp_mask_tensor == float('-inf')
        logits = logits.masked_fill(invalid_mask, float('-inf'))

        # ---------- Policy (Action Probabilities) ----------
        logits = logits - logits.max()
        probs = F.softmax(logits, dim=-1)

        # ---------- Critic (State Value) ----------
        # 用当前节点的 embedding 来估计 V(s)
        value_emb = node_embeddings[cur_node_idx]
        value = self.value_head(value_emb).squeeze(-1)  # scalar

        return probs, logits, value