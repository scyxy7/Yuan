# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # ✅ 修复：新增这一行，解决 NameError

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_embeddings, cur_node_idx, soc, ffp_logits_mask):
        # Query construction
        cur_emb = node_embeddings[cur_node_idx]
        query = self.query_fc(
            torch.cat([cur_emb, torch.tensor([soc], dtype=torch.float32)])
        ).unsqueeze(0)  # [1, hidden_dim]

        # Keys
        keys = self.key_fc(node_embeddings)  # [num_nodes, hidden_dim]

        # Dot-product attention -> logits
        logits = (query @ keys.T).squeeze(0)  # [num_nodes]

        # FFP Hard Mask
        if isinstance(ffp_logits_mask, np.ndarray):
            ffp_mask_tensor = torch.from_numpy(ffp_logits_mask).to(logits.device)
        else:
            ffp_mask_tensor = ffp_logits_mask.to(logits.device)

        invalid_mask = ffp_mask_tensor == float('-inf')
        logits = logits.masked_fill(invalid_mask, float('-inf'))

        # Softmax
        probs = F.softmax(logits, dim=-1)

        return probs, logits