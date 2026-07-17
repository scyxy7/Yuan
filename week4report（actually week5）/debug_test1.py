"""
Diagnostic Script: Why is Test 1 failing?
Focus: Print the FFP mask BEFORE it reaches PyTorch
"""

import numpy as np

# =========================
# 节点定义（和你的一模一样）
# =========================
NODES = {
    0: (0, 0, 0, 0, 200, 0),
    1: (2, 0, 5, 20, 80, 1),
    2: (4, 0, 5, 40, 120, 1),
    3: (6, 0, 0, 0, 200, 2),
}

SPEED = 1.0
ETA = 1.0
BATTERY = 100

# =========================
# 状态
# =========================
class State:
    def __init__(self):
        self.cur_node = 0
        self.soc = BATTERY
        self.visited = set()

# =========================
# 环境（只保留 FFP）
# =========================
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

        print("\n🐞 DEBUG: FFP internal checks")
        print(f"Current Node: {cur}")
        print(f"Current SoC: {state.soc}")

        for i in range(self.num_nodes):
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                print(f"  Node {i}: CS → ALLOW (0.0)")
                continue

            if i in state.visited and self.nodes[i][5] == 1:
                mask[i] = -np.inf
                print(f"  Node {i}: Visited → BLOCK (-inf)")
                continue

            travel_energy = ETA * self.dist[cur][i]
            if state.soc < travel_energy:
                mask[i] = -np.inf
                print(f"  Node {i}: Not enough energy to reach → BLOCK (-inf)")
                continue

            soc_after = state.soc - travel_energy
            min_dist = min(self.dist[i][cs] for cs in self.charging_stations)
            if soc_after < ETA * min_dist:
                mask[i] = -np.inf
                print(f"  Node {i}: Cannot reach CS afterwards → BLOCK (-inf)")
                continue

            mask[i] = 0.0
            print(f"  Node {i}: Feasible → ALLOW (0.0)")

        return mask

# =========================
# 运行诊断
# =========================
env = EVRPTWEnv(NODES)
state = State()
state.cur_node = 1
state.soc = 5  # 极低电量

mask = env.get_ffp_logits_mask(state)

print("\n📊 Final FFP Mask:")
for i in range(env.num_nodes):
    val = mask[i]
    if val == -np.inf:
        print(f"  Node {i}: -inf ✅ BLOCKED")
    else:
        print(f"  Node {i}: {val} ❌ ALLOWED")

print("\n✅ Expected: Node 1, 2 should be -inf")
print("❌ If they are not -inf, FFP logic is wrong, not PyTorch.")