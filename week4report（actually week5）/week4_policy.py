"""
Week 4: Policy Network + FFP Integration
- Env encapsulation
- FFP as logits mask (differentiable)
- Attention-based Decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 全局参数（与 TERRAN 对齐）
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100
CAPACITY = 10
SERVICE_TIME = 5

# =========================
# 节点定义
# (x, y, demand, e, l, node_type)
# node_type: 0=depot, 1=customer, 2=charging station
# =========================
NODES = {
    0: (0, 0, 0, 0, 200, 0),
    1: (2, 0, 5, 20, 80, 1),
    2: (4, 0, 5, 40, 120, 1),
    3: (6, 0, 0, 0, 200, 2),
}


# =========================
# 状态定义
# =========================
class State:
    def __init__(self):
        self.cur_node = 0
        self.clock = 0.0
        self.soc = BATTERY
        self.load = CAPACITY
        self.visited = set()

    def copy(self):
        s = State()
        s.cur_node = self.cur_node
        s.clock = self.clock
        s.soc = self.soc
        s.load = self.load
        s.visited = self.visited.copy()
        return s


# =========================
# 环境封装（B：打通 FFP）
# =========================
class EVRPTWEnv:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)

        # 距离 & 时间矩阵
        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in nodes:
            for j in nodes:
                xi, yi = nodes[i][:2]
                xj, yj = nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])
        self.time_mat = self.dist / SPEED

        # 充电站索引
        self.charging_stations = [
            i for i in nodes if nodes[i][5] == 2
        ]

    def reset(self):
        return State()

    def step(self, state, action):
        s = state.copy()
        i = action

        s.clock += self.time_mat[s.cur_node][i]

        if self.nodes[i][5] == 1:  # customer
            s.clock += SERVICE_TIME
            s.load -= self.nodes[i][2]
            s.soc -= ETA * self.dist[s.cur_node][i]
            s.visited.add(i)
        elif self.nodes[i][5] == 2:  # charging station
            s.soc = BATTERY

        s.cur_node = i
        return s

    # =======================
    # ✅ FFP：返回 logits mask
    # =======================
    def get_ffp_logits_mask(self, state):
        """
        返回 logits mask:
            0       → 合法
            -inf    → 非法
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)

        cur = state.cur_node

        for i in range(self.num_nodes):
            # 1. 充电站：永远放行
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                continue

            # 2. 已服务客户
            if i in state.visited and self.nodes[i][5] == 1:
                mask[i] = -np.inf
                continue

            # 3. 单步能量检查
            travel_energy = ETA * self.dist[cur][i]
            if state.soc < travel_energy:
                mask[i] = -np.inf
                continue

            soc_after = state.soc - travel_energy

            # 4. 能否到达任意充电站
            min_dist = min(
                self.dist[i][cs]
                for cs in self.charging_stations
            )
            if soc_after < ETA * min_dist:
                mask[i] = -np.inf
                continue

            # 5. 合法
            mask[i] = 0.0

        return mask


# =========================
# Decoder（A：Attention）
# =========================
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Query: 当前节点嵌入 + SoC
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        # Key: 节点嵌入
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_embeddings, cur_node_idx, soc, ffp_logits_mask):
        """
        node_embeddings: [num_nodes, hidden_dim]
        cur_node_idx: int
        soc: float
        ffp_logits_mask: [num_nodes] (numpy array)
        """
        # 1. Query
        cur_emb = node_embeddings[cur_node_idx]
        query_input = torch.cat([cur_emb, torch.tensor([soc], dtype=torch.float32)])
        query = self.query_fc(query_input).unsqueeze(0)  # [1, hidden_dim]

        # 2. Keys
        keys = self.key_fc(node_embeddings)  # [num_nodes, hidden_dim]

        # 3. 计算原始 Logits (点积注意力)
        logits = (query @ keys.T).squeeze(0)  # [num_nodes]

        # 4. ✅ 终极修复：使用 PyTorch 原生 masked_fill 强制屏蔽
        # 将 numpy 掩码转为 boolean tensor
        # 假设 ffp_logits_mask 中 0.0 表示合法，-np.inf 表示非法
        # 我们需要把非法位置（值为 -inf）找出来并填充
        mask_tensor = torch.from_numpy(ffp_logits_mask).to(logits.device)
        
        # 找出需要屏蔽的位置（mask_tensor 中为 -inf 的地方）
        # 注意：如果掩码是用 0 和 -inf 表示的，我们需要反转逻辑
        # 更安全的方式：假设 mask_tensor 中 0 是允许，-inf 是禁止
        # 我们创建一个 boolean mask：True 表示需要屏蔽
        invalid_mask = mask_tensor == float('-inf')
        
        # 强制将非法位置的 logits 设为 -inf
        logits = logits.masked_fill(invalid_mask, float('-inf'))

        # 5. Softmax —— 此时 -inf 在 softmax 中会自动变为 0
        probs = F.softmax(logits, dim=-1)

        return probs, logits


# =========================
# Demo：前向传播验证
# =========================
def demo():
    print("=== Week 4: Policy Network + FFP Demo ===\n")

    env = EVRPTWEnv(NODES)
    state = env.reset()

    # 假装 Encoder 输出（后续会被 Transformer 替换）
    hidden_dim = 64
    num_nodes = env.num_nodes
    node_embeddings = torch.randn(num_nodes, hidden_dim)

    decoder = AttentionDecoder(hidden_dim)

    for step_id in range(4):
        ffp_mask = env.get_ffp_logits_mask(state)

        probs, logits = decoder(
            node_embeddings,
            state.cur_node,
            state.soc,
            ffp_mask
        )

        print(f"Step {step_id}")
        print(f"  Cur Node: {state.cur_node}")
        print(f"  SoC: {state.soc:.1f}")
        print(f"  Logits: {logits.detach().numpy().round(2)}")
        print(f"  Probs:  {probs.detach().numpy().round(3)}")
        print(f"  Sum(Probs): {probs.sum().item():.6f}")

        # 采样动作（避免 -inf 干扰）
        valid_indices = torch.where(probs > 0)[0]
        if len(valid_indices) == 0:
            print("  ❌ No valid action\n")
            break

        action = valid_indices[torch.multinomial(probs[valid_indices], 1)].item()
        print(f"  Action: {action}\n")

        state = env.step(state, action)


if __name__ == "__main__":
    demo()