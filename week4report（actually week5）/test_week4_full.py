"""
Full Self-Contained Test Script
Policy Network + FFP Integration Validation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 全局参数
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100

# =========================
# 节点定义
# =========================
NODES = {
    0: (0, 0, 0, 0, 200, 0),   # Depot
    1: (2, 0, 5, 20, 80, 1),   # Customer C1
    2: (4, 0, 5, 40, 120, 1),  # Customer C2
    3: (6, 0, 0, 0, 200, 2),   # Charging Station
}

# =========================
# 状态定义
# =========================
class State:
    def __init__(self):
        self.cur_node = 0
        self.soc = BATTERY
        self.visited = set()

# =========================
# 环境封装（含 FFP logits mask）
# =========================
class EVRPTWEnv:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)

        # 距离矩阵
        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in nodes:
            for j in nodes:
                xi, yi = nodes[i][:2]
                xj, yj = nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])

        self.charging_stations = [
            i for i in nodes if nodes[i][5] == 2
        ]

    def get_ffp_logits_mask(self, state):
        """
        返回 logits mask:
            0      → 合法
            -inf   → 非法
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
# Attention Decoder（Policy Network）
# =========================
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, embeddings, cur_idx, soc, ffp_mask):
        """
        embeddings: [num_nodes, hidden_dim]
        cur_idx: int
        soc: float
        ffp_mask: Tensor[num_nodes]
        """
        cur_emb = embeddings[cur_idx]
        query = self.query_fc(
            torch.cat([cur_emb, torch.tensor([soc], dtype=torch.float32)])
        ).unsqueeze(0)

        keys = self.key_fc(embeddings)
        logits = (query @ keys.T).squeeze(0)

        # ✅ FFP 硬掩码
        logits = logits + ffp_mask
        probs = F.softmax(logits, dim=-1)

        return probs, logits

# =========================
# 测试函数
# =========================
def test_ffp_blocks_invalid_actions():
    print("🔹 Test 1: FFP blocks energy-stranding actions")
    env = EVRPTWEnv(NODES)
    state = State()
    state.cur_node = 1
    state.soc = 5  # 极低电量

    mask = env.get_ffp_logits_mask(state)
    for i in range(env.num_nodes):
        if NODES[i][5] != 2 and mask[i] != -np.inf:
            print(f"❌ FAIL: Node {i} should be blocked")
            return False

    print("✅ PASS: All non-CS nodes blocked")
    return True


def test_probability_distribution():
    print("\n🔹 Test 2: Probability distribution sums to 1")
    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)
    ffp_mask = torch.from_numpy(env.get_ffp_logits_mask(state))

    probs, _ = decoder(embeddings, state.cur_node, state.soc, ffp_mask)
    prob_sum = probs.sum().item()

    if abs(prob_sum - 1.0) > 1e-6:
        print(f"❌ FAIL: Sum = {prob_sum}")
        return False

    print(f"✅ PASS: Sum(probs) = {prob_sum:.6f}")
    return True


def test_gradient_flow():
    print("\n🔹 Test 3: Gradient flows through masked softmax")
    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)
    ffp_mask = torch.from_numpy(env.get_ffp_logits_mask(state))

    probs, _ = decoder(embeddings, state.cur_node, state.soc, ffp_mask)
    loss = -torch.log(probs[1] + 1e-8)
    loss.backward()

    if embeddings.grad is None or torch.isnan(embeddings.grad).any():
        print("❌ FAIL: Gradient broken")
        return False

    print("✅ PASS: Gradient exists and clean")
    return True


def test_charging_station_always_allowed():
    print("\n🔹 Test 4: Charging station always allowed")
    env = EVRPTWEnv(NODES)
    state = State()
    state.soc = 0.1  # 濒死电量

    mask = env.get_ffp_logits_mask(state)
    cs_idx = env.charging_stations[0]

    if mask[cs_idx] != 0.0:
        print(f"❌ FAIL: CS {cs_idx} blocked")
        return False

    print("✅ PASS: CS accessible at near-zero SoC")
    return True

# =========================
# 主入口
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Policy + FFP Validation (Self-Contained)")
    print("=" * 60)

    tests = [
        test_ffp_blocks_invalid_actions,
        test_probability_distribution,
        test_gradient_flow,
        test_charging_station_always_allowed,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    if all(results):
        print("🎉 ALL TESTS PASSED. Ready for PPO training.")
    else:
        print("❌ SOME TESTS FAILED. Check logic above.")
    print("=" * 60)