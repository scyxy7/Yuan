"""
env.py
EVRPTW Environment, State, and FFP (Future-Feasibility Pruning)
Compatible with Week 4 Policy Network integration.
"""

import numpy as np

# =========================
# Global Parameters (Toy Instance)
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100
CAPACITY = 10
SERVICE_TIME = 5

NODES = {
    0: (0, 0, 0, 0, 200, 0),   # Depot
    1: (1, 0, 5, 20, 80, 1),   # Customer C1
    2: (2, 1, 5, 40, 120, 1),  # Customer C2
    3: (2, 1, 0, 0, 200, 2),   # Charging Station
    4: (3, 0, 5, 10, 150, 1),  # 新增客户 C4
    5: (3, 1, 5, 10, 150, 1),  # 新增客户 C5
}


# =========================
# State Definition
# =========================
class State:
    def __init__(self):
        self.cur_node = 0          # current node index
        self.clock = 0.0           # current time
        self.soc = BATTERY         # remaining battery
        self.load = CAPACITY       # remaining capacity
        self.visited = set()       # visited customers
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

        # Distance matrix
        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in self.nodes:
            for j in self.nodes:
                xi, yi = self.nodes[i][:2]
                xj, yj = self.nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])

        self.time_mat = self.dist / SPEED

        # Charging stations
        self.charging_stations = [
            i for i in self.nodes if self.nodes[i][5] == 2
        ]
        # Customer nodes
        self.customer_nodes = [
            i for i in self.nodes if self.nodes[i][5] == 1
        ]

    # =========================
    # FFP Logits Mask (Hard Mask)
    # =========================
    def get_ffp_logits_mask(self, state):
        """
        Returns:
            mask: np.ndarray[num_nodes]
                  0.0   -> feasible
                  -np.inf -> infeasible
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        cur = state.cur_node

        for i in range(self.num_nodes):
            # 1. Never allow staying at the current node
            if i == cur:
                mask[i] = -np.inf
                continue

            # 2. Charging stations always allowed
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                continue

            # 3. Already visited customers
            if i in state.visited and self.nodes[i][5] == 1:
                mask[i] = -np.inf
                continue

            # 4. Single-step energy check
            travel_energy = ETA * self.dist[cur][i]
            if state.soc < travel_energy:
                mask[i] = -np.inf
                continue

            soc_after = state.soc - travel_energy

            # 4. Future feasibility: can reach a CS afterwards?
            min_dist_to_cs = min(
                self.dist[i][cs] for cs in self.charging_stations
            )
            if soc_after < ETA * min_dist_to_cs:
                mask[i] = -np.inf
                continue

            # 5. Feasible
            mask[i] = 0.0

        return mask

    # =========================
    # Step function
    # =========================
    def step(self, state, action):
        s = state.copy()
        i = action

        s.clock += self.time_mat[s.cur_node][i]

        if self.nodes[i][5] == 1:   # customer
            s.clock += SERVICE_TIME
            s.load -= self.nodes[i][2]
            s.soc -= ETA * self.dist[s.cur_node][i]
            s.visited.add(i)
        elif self.nodes[i][5] == 2: # charging station
            s.soc = BATTERY

        s.cur_node = i
        return s

    # =========================
    # ✅ NEW: Reward Function (Key to make Loss move)
    # =========================
    def compute_reward(self, state, action, next_state):
        """
        Strong reward signal for small-scale EVRPTW.
        Designed to make policy learn quickly.
        """
        cur = state.cur_node
        nxt = action

        # 1. Base cost: negative travel distance
        reward = -self.dist[cur][nxt]

        # 2. Time window violation (tardiness penalty)
        arrival_time = next_state.clock
        due_time = self.nodes[nxt][4]
        if arrival_time > due_time:
            reward -= 10.0

        # 3. Visit customer bonus
        if self.nodes[nxt][5] == 1 and nxt not in state.visited:
            reward += 20.0

        # 4. Return to depot bonus (task completion)
        if nxt == 0 and len(state.visited) > 0:
            reward += 50.0

        # 5. Low battery penalty (encourage charging)
        if next_state.soc < 20:
            reward -= 5.0

        return reward

    def reset(self):
        return State()