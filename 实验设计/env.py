"""
env.py
EVRPTW Environment (10 Nodes)
Week 5 · Final No-Revisit Version
"""

import numpy as np

# =========================
# Global Parameters
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100
CAPACITY = 10
SERVICE_TIME = 5

# =========================
# 10 Nodes Definition
# =========================
NODES = {
    0: (0, 0, 0, 0, 200, 0),   # Depot
    1: (1, 0, 5, 10, 100, 1),  # C1
    2: (2, 1, 5, 20, 120, 1),  # C2
    3: (2, 1, 0, 0, 200, 2),   # Charging Station
    4: (3, 0, 4, 30, 90, 1),   # C3
    5: (3, 2, 6, 40, 110, 1),  # C4
    6: (4, 1, 3, 50, 130, 1),  # C5
    7: (4, 3, 5, 60, 140, 1),  # C6
    8: (5, 2, 4, 70, 150, 1),  # C7
    9: (5, 4, 5, 80, 160, 1),  # C8
}

# =========================
# State Definition
# =========================
class State:
    def __init__(self):
        self.cur_node = 0
        self.clock = 0.0
        self.soc = BATTERY
        self.load = CAPACITY
        self.visited = set()
        self.done = False

    def copy(self):
        s = State()
        s.cur_node = self.cur_node
        s.clock = self.clock
        s.soc = self.soc
        s.load = self.load
        s.visited = self.visited.copy()
        s.done = self.done
        return s

# =========================
# Environment Definition
# =========================
class EVRPTWEnv:
    def __init__(self, nodes=None):
        self.nodes = nodes if nodes is not None else NODES
        self.num_nodes = len(self.nodes)

        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in self.nodes:
            for j in self.nodes:
                xi, yi = self.nodes[i][:2]
                xj, yj = self.nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])

        self.time_mat = self.dist / SPEED
        self.charging_stations = [
            i for i in self.nodes if self.nodes[i][5] == 2
        ]
        self.customer_nodes = [
            i for i in self.nodes if self.nodes[i][5] == 1
        ]

    # =========================
    # FFP Mask
    # =========================
    def get_ffp_logits_mask(self, state):
        mask = np.full(self.num_nodes, -np.inf, dtype=np.float32)
        cur = state.cur_node

        for i in range(self.num_nodes):
            if i == cur:
                continue

            # Charging stations always feasible
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                continue

            # ❌ Never revisit customers
            if self.nodes[i][5] == 1 and i in state.visited:
                continue

            # Energy feasibility
            if state.soc < ETA * self.dist[cur][i]:
                continue

            # Future feasibility
            soc_after = state.soc - ETA * self.dist[cur][i]
            min_dist_to_cs = min(
                self.dist[i][cs] for cs in self.charging_stations
            )
            if soc_after < ETA * min_dist_to_cs:
                continue

            mask[i] = 0.0
        return mask

    # =========================
    # Step Function
    # =========================
    def step(self, state, action):
        s = state.copy()
        i = action

        s.clock += self.time_mat[s.cur_node][i]
        s.soc = max(0.0, s.soc - ETA * self.dist[s.cur_node][i])

        if self.nodes[i][5] == 1:
            s.clock += SERVICE_TIME
            s.load -= self.nodes[i][2]
            s.visited.add(i)
        elif self.nodes[i][5] == 2:
            s.soc = BATTERY

        s.cur_node = i

        if i == 0 and len(s.visited) == len(self.customer_nodes):
            s.done = True

        return s

    # =========================
    # Reward Function (Final No-Revisit Version)
    # =========================
    def compute_reward(self, state, action, next_state):
        cur = state.cur_node
        nxt = action
        reward = -0.5 * self.dist[cur][nxt]

        # ❌ Heavy punishment for revisiting a customer
        if self.nodes[nxt][5] == 1 and nxt in state.visited:
            reward -= 150.0   # Increased from -120
            # ❌ DO NOT RETURN HERE
            # Let other penalties accumulate

        # ✅ Reward for new customer
        if self.nodes[nxt][5] == 1 and nxt not in state.visited:
            reward += 30.0
            reward += 5.0 * len(next_state.visited)

        # ✅ Depot logic
        if nxt == 0:
            if len(state.visited) == len(self.customer_nodes):
                reward += 200.0
            else:
                reward -= 80.0

        # ✅ Charging station (discouraged but allowed)
        if self.nodes[nxt][5] == 2:
            reward -= 4.0

        # ✅ Time window
        if next_state.clock > self.nodes[nxt][4]:
            reward -= 8.0

        # ✅ Low battery
        if next_state.soc < 20:
            reward -= 5.0

        return reward

    def reset(self):
        return State()