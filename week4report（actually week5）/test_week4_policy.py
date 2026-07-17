"""
Test Script: Validate Policy Network + FFP Integration
Objectives:
1. FFP blocks energy-stranding actions
2. Probability distribution is valid (sum = 1)
3. Gradient can flow through masked softmax
"""

import numpy as np
import torch
import torch.nn.functional as F

# =========================
# 复用 week4_policy.py 的核心组件
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100

NODES = {
    0: (0, 0, 0, 0, 200, 0),
    1: (2, 0, 5, 20, 80, 1),
    2: (4, 0, 5, 40, 120, 1),
    3: (6, 0, 0, 0, 200, 2),
}

class State:
    def __init__(self):
        self.cur_node = 0
        self.soc = BATTERY
        self.visited = set()

class EVRPTWEnv:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in nodes:
            for j in nodes:
                xi, yi = nodes[i][:2]
                xj, yj = nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])
        self.charging_stations = [i for i in nodes if nodes[i][5] == 2]

    def get_ffp_logits_mask(self, state):
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        cur = state.cur_node

        for i in range(self.num_nodes):
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                continue
            if i in state.visited and self.nodes[i][5] == 1:
                mask[i] = -np.inf
                continue

            travel_energy = ETA * self.dist[cur][i]
            if state.soc < travel_energy:
                mask[i] = -np.inf
                continue

            soc_after = state.soc - travel_energy
            min_dist = min(self.dist[i][cs] for cs in self.charging_stations)
            if soc_after < ETA * min_dist:
                mask[i] = -np.inf
                continue

            mask[i] = 0.0
        return mask

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, embeddings, cur_idx, soc, ffp_mask):
        cur_emb = embeddings[cur_idx]
        query = self.query_fc(
            torch.cat([cur_emb, torch.tensor([soc])])
        ).unsqueeze(0)

        keys = self.key_fc(embeddings)
        logits = (query @ keys.T).squeeze(0)

        logits = logits + ffp_mask
        probs = F.softmax(logits, dim=-1)
        return probs, logits

# =========================
# 测试函数
# =========================
def test_ffp_blocks_invalid_actions():
    """Test 1: FFP 必须挡掉会导致电量耗尽的节点"""
    print("🔹 Test 1: FFP blocks energy-stranding actions")

    env = EVRPTWEnv(NODES)
    state = State()
    state.cur_node = 1  # 从 C1 出发
    state.soc = 5        # 极低电量（故意制造危险）

    ffp_mask = env.get_ffp_logits_mask(state)

    # 检查：非充电站节点是否被正确屏蔽
    for i in range(env.num_nodes):
        if NODES[i][5] != 2:  # 非充电站
            if ffp_mask[i] != -np.inf:
                print(f"❌ FAIL: Node {i} should be blocked (low SoC)")
                return False

    print("✅ PASS: All non-CS nodes blocked")
    return True


def test_probability_distribution():
    """Test 2: Softmax 后概率和为 1"""
    print("\n🔹 Test 2: Probability distribution sums to 1")

    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)

    ffp_mask = env.get_ffp_logits_mask(state)
    ffp_mask_tensor = torch.from_numpy(ffp_mask)

    probs, logits = decoder(embeddings, state.cur_node, state.soc, ffp_mask_tensor)

    prob_sum = probs.sum().item()

    if abs(prob_sum - 1.0) > 1e-6:
        print(f"❌ FAIL: Sum = {prob_sum}")
        return False

    print(f"✅ PASS: Sum(probs) = {prob_sum:.6f}")
    return True


def test_gradient_flow():
    """Test 3: 梯度可以穿过 masked softmax"""
    print("\n🔹 Test 3: Gradient flows through masked softmax")

    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)

    ffp_mask = env.get_ffp_logits_mask(state)
    ffp_mask_tensor = torch.from_numpy(ffp_mask)

    probs, logits = decoder(embeddings, state.cur_node, state.soc, ffp_mask_tensor)

    # 构造一个假损失：最大化选中节点 1 的概率
    loss = -torch.log(probs[1] + 1e-8)
    loss.backward()

    if embeddings.grad is None:
        print("❌ FAIL: No gradient")
        return False

    if torch.isnan(embeddings.grad).any():
        print("❌ FAIL: Gradient contains NaN")
        return False

    print("✅ PASS: Gradient exists and is clean")
    return True


def test_charging_station_always_allowed():
    """Test 4: 充电站永远可达"""
    print("\n🔹 Test 4: Charging station always allowed")

    env = EVRPTWEnv(NODES)
    state = State()
    state.soc = 0.1  # 濒死电量

    ffp_mask = env.get_ffp_logits_mask(state)

    cs_idx = env.charging_stations[0]
    if ffp_mask[cs_idx] != 0.0:
        print(f"❌ FAIL: CS {cs_idx} is blocked")
        return False

    print("✅ PASS: Charging station accessible even at near-zero SoC")
    return True


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Policy Network + FFP Validation")
    print("=" * 60)

    results = []
    results.append(test_ffp_blocks_invalid_actions())
    results.append(test_probability_distribution())
    results.append(test_gradient_flow())
    results.append(test_charging_station_always_allowed())

    print("\n" + "=" * 60)
    if all(results):
        print("🎉 ALL TESTS PASSED. Ready for PPO training.")
    else:
        print("❌ SOME TESTS FAILED. Please check the logic.")
    print("=" * 60)