import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.eps = 1e-6

    def forward(self, node_embeddings, cur_node_idx, soc, ffp_logits_mask):
        batched = node_embeddings.dim() == 3

        # =========================
        # Batch / Single mode dispatch
        # =========================
        if batched:
            batch_size = node_embeddings.size(0)

            # ---- Current node index ----
            if isinstance(cur_node_idx, int):
                idx = torch.full(
                    (batch_size,), cur_node_idx,
                    device=node_embeddings.device, dtype=torch.long
                )
            elif isinstance(cur_node_idx, np.ndarray):
                idx = torch.from_numpy(cur_node_idx).to(node_embeddings.device)
            else:
                idx = cur_node_idx.to(node_embeddings.device)

            cur_emb = node_embeddings[
                torch.arange(batch_size, device=node_embeddings.device), idx
            ]

            # ---- SoC tensor ----
            soc_tensor = torch.as_tensor(
                soc, dtype=torch.float32, device=node_embeddings.device
            )
            if soc_tensor.dim() == 0:
                soc_tensor = soc_tensor.unsqueeze(0).expand(batch_size)
            soc_tensor = soc_tensor.view(batch_size, 1)

            # ---- Attention ----
            query_raw = self.query_fc(torch.cat([cur_emb, soc_tensor], dim=-1))
            query = F.normalize(query_raw, dim=-1, eps=self.eps).unsqueeze(1)

            keys_raw = self.key_fc(node_embeddings)
            keys = F.normalize(keys_raw, dim=-1, eps=self.eps)

            logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)

        else:
            # =========================
            # Single sample mode
            # =========================
            cur_emb = node_embeddings[cur_node_idx]

            soc_tensor = torch.as_tensor(
                soc, dtype=torch.float32, device=node_embeddings.device
            ).view(1)

            query_raw = self.query_fc(torch.cat([cur_emb, soc_tensor]))
            query = F.normalize(query_raw, dim=-1, eps=self.eps).unsqueeze(0)

            keys_raw = self.key_fc(node_embeddings)
            keys = F.normalize(keys_raw, dim=-1, eps=self.eps)

            logits = (query @ keys.T).squeeze(0)

        # =========================
        # FFP Mask (NumPy → Tensor)
        # =========================
        if isinstance(ffp_logits_mask, np.ndarray):
            ffp_mask_tensor = torch.from_numpy(ffp_logits_mask).to(logits.device)
        else:
            ffp_mask_tensor = ffp_logits_mask.to(logits.device)

        if batched and ffp_mask_tensor.dim() == 1:
            ffp_mask_tensor = ffp_mask_tensor.unsqueeze(0).expand_as(logits)

        invalid_mask = ffp_mask_tensor == float('-inf')
        logits = logits.masked_fill(invalid_mask, float('-inf'))

        # =========================
        # Safe Softmax (PPO Critical)
        # =========================
        if batched:
            all_inf = torch.isinf(logits).all(dim=1)

            if all_inf.any():
                # ---- Fallback: stay at current node ----
                probs = torch.zeros_like(logits)
                logits = torch.zeros_like(logits)
                
                rows = torch.nonzero(all_inf, as_tuple=False).squeeze(1)
                if rows.numel() > 0:
                    probs[rows, idx[rows]] = 1.0
            else:
                logits = logits - logits.max(dim=1, keepdim=True).values
                probs = F.softmax(logits.clamp(min=-30.0), dim=-1)

            value_emb = node_embeddings[
                torch.arange(batch_size, device=node_embeddings.device), idx
            ]

        else:
            if torch.isinf(logits).all():
                # ---- Fallback: stay at current node ----
                probs = torch.zeros_like(logits)
                logits = torch.zeros_like(logits)
                probs[cur_node_idx] = 1.0
            else:
                logits = logits - logits.max()
                probs = F.softmax(logits.clamp(min=-30.0), dim=-1)

            value_emb = node_embeddings[cur_node_idx]

        # =========================
        # Value Head
        # CRITICAL: Do NOT detach here.
        # The critic must receive gradients from the value loss.
        # Detaching here is a common bug that prevents value convergence.
        # =========================
        value = self.value_head(value_emb).squeeze(-1)

        return probs, logits, value