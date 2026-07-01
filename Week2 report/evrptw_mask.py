"""
Constraint-Aware Hard Action Masking for EVRP-TW
Based on: TERRAN (Tang et al., IEEE T-ASE 2026), Section IV-C
"""

import numpy as np




SPEED = 1.0           # 恒定速度 (unit distance / min)
ETA = 1.0             # 能耗率：1 unit distance -> 1 unit energy
CAPACITY = 10         # 车辆最大载重
BATTERY = 100         # 电池容量 (SoC_max)
SERVICE_TIME = 5      # 客户服务时间 (min)


nodes = {
    0: (0, 0, 0, 0, 200, 0),    # Depot
    1: (2, 0, 5, 20, 80, 1),    # Customer C1
    2: (4, 0, 5, 40, 120, 1),   # Customer C2
    3: (6, 0, 0, 0, 200, 2),    # Charging Station
}

num_nodes = len(nodes)


dist = np.zeros((num_nodes, num_nodes))
for i in nodes:
    for j in nodes:
        dist[i][j] = np.linalg.norm(
            np.array(nodes[i][:2]) - np.array(nodes[j][:2])
        )


time_mat = dist / SPEED


class State:
    def __init__(self):
        self.cur_node = 0          # 当前所在节点
        self.clock = 0              # 当前时间
        self.soc = BATTERY          # 当前电量
        self.load = CAPACITY        # 当前剩余载重
        self.visited = set()        # 已服务客户

state = State()

# =========================
# 三个基础掩码（TERRAN Eq.12–16）
# =========================

def time_window_mask(state, next_node):
    """
    TERRAN Eq. (12)
    """
    i = next_node
    arrival = state.clock + time_mat[state.cur_node][i]

    if nodes[i][5] == 1:  # customer
        e, l = nodes[i][3], nodes[i][4]
        return 1 if arrival <= l else 0
    else:
        return 1  # depot / CS 无时间窗限制


def capacity_mask(state, next_node):
    """
    TERRAN Eq. (13)
    """
    i = next_node
    if nodes[i][5] == 1:  # customer
        demand = nodes[i][2]
        return 1 if state.load >= demand else 0
    else:
        return 1


def energy_mask(state, next_node):
    """
    TERRAN Eq. (16) 简化版（单步检查，不含FFP）
    """
    travel_energy = ETA * time_mat[state.cur_node][next_node]
    return 1 if state.soc >= travel_energy else 0


def feasibility_mask(state):
    """
    综合硬约束掩码
    M_t(i) = M_time · M_cap · M_energy
    """
    mask = np.zeros(num_nodes, dtype=int)

    for i in range(num_nodes):
        # 已服务客户不能再访问
        if i in state.visited and nodes[i][5] == 1:
            mask[i] = 0
            continue

        m_time = time_window_mask(state, i)
        m_cap = capacity_mask(state, i)
        m_energy = energy_mask(state, i)

        mask[i] = m_time * m_cap * m_energy

    return mask



def step(state, action):
    """环境状态更新（简化版）"""
    i = action
    state.clock += time_mat[state.cur_node][i]

    if nodes[i][5] == 1:  # customer
        state.clock += SERVICE_TIME
        state.load -= nodes[i][2]
        state.soc -= ETA * time_mat[state.cur_node][i]
        state.visited.add(i)
    elif nodes[i][5] == 2:  # charging station
        state.soc = BATTERY
    else:  # depot
        pass

    state.cur_node = i


def demo():
    print("=== Hard Action Masking Demo (TERRAN Style) ===\n")
    for t in range(4):
        mask = feasibility_mask(state)
        print(f"Step {t}")
        print(f"Current Node: {state.cur_node}")
        print(f"Clock: {state.clock:.1f}, SoC: {state.soc:.1f}, Load: {state.load}")
        print("Feasibility Mask:", mask)
        print("Allowed Nodes:", [i for i in range(num_nodes) if mask[i] == 1])

        # 简单贪婪：选第一个允许的节点
        allowed = [i for i in range(num_nodes) if mask[i] == 1]
        if len(allowed) == 0:
            print("No feasible actions!\n")
            break

        action = allowed[0]
        print(f"Selected Action: {action}\n")
        step(state, action)


if __name__ == "__main__":
    demo()