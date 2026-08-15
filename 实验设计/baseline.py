"""
baseline.py
提供：
- `LinearEncoder`（轻量线性编码器）
- `PolicyNetworkLinear`（基于线性编码器的策略网络）
- `greedy_baseline`：贪心启发式基线，用于与 PPO 策略比较

此文件设计为轻量且可导入到 `experiments.py` 中
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder
from encoder import TransformerEncoder


class LinearEncoder(nn.Module):
    """简单的线性编码器：Linear -> LayerNorm -> GELU
    输入支持 [num_nodes, input_dim] 或 [batch, num_nodes, input_dim]
    返回 embedding 同形状 [batch?, num_nodes, hidden_dim]
    """
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        h = self.fc(x)
        h = F.gelu(self.ln(h))
        if not batched:
            h = h.squeeze(0)
        return h


class PolicyNetworkLinear(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = LinearEncoder(input_dim=6, hidden_dim=hidden_dim)
        self.decoder = AttentionDecoder(hidden_dim=hidden_dim)

    def forward(self, node_features, cur_node_idx, soc, ffp_mask):
        node_embeddings = self.encoder(node_features)
        probs, logits, value = self.decoder(node_embeddings, cur_node_idx, soc, ffp_mask)
        return probs, logits, value


def greedy_baseline(env: EVRPTWEnv, max_steps=30):
    """贪心基线：每步选择距当前节点最近的可行节点（按照 mask）"""
    state = env.reset()
    route = [0]
    total_reward = 0.0
    step_info = []

    for t in range(max_steps):
        mask = env.get_ffp_logits_mask(state)
        # mask == 0 -> feasible
        feasible = [i for i, m in enumerate(mask) if m == 0.0 and i != state.cur_node]

        if len(feasible) == 0:
            # 若没有可行节点，则停留（保持在当前）
            action = state.cur_node
        else:
            # 选择最短距离
            dists = [(env.dist[state.cur_node][i], i) for i in feasible]
            dists.sort()
            action = dists[0][1]

        next_state = env.step(state, action)
        reward = env.compute_reward(state, action, next_state)

        step_info.append({
            "step": t,
            "from": state.cur_node,
            "to": action,
            "reward": reward,
            "soc": next_state.soc,
            "clock": next_state.clock,
            "visited": sorted(list(next_state.visited))
        })

        route.append(action)
        total_reward += reward
        state = next_state

        if state.done or state.soc <= 0:
            break

    return {
        "route": route,
        "total_reward": total_reward,
        "visited": sorted(list(state.visited)),
        "final_soc": state.soc,
        "final_clock": state.clock,
        "steps": len(step_info),
        "step_info": step_info
    }


if __name__ == "__main__":
    env = EVRPTWEnv(NODES)
    res = greedy_baseline(env, max_steps=30)
    print("Greedy baseline result:")
    print(f"Route: {' -> '.join(map(str,res['route']))}")
    print(f"Reward: {res['total_reward']:.2f} | Visited: {res['visited']} | SoC: {res['final_soc']:.2f}")
