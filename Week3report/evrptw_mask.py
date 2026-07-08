"""
Constraint-Aware Hard Action Masking for EVRP-TW
Based on: TERRAN (Tang et al., IEEE T-ASE 2026), Section IV-C
Extended with Future-Feasibility Pruning (FFP)
"""

import numpy as np

# =========================
# 问题参数（Toy Instance）
# =========================
SPEED = 1.0           # 恒定速度 (unit distance / min)
ETA = 1.0             # 能耗率：1 unit distance -> 1 unit energy
CAPACITY = 10         # 车辆最大载重
BATTERY = 100         # 电池容量 (SoC_max)
SERVICE_TIME = 5      # 客户服务时间 (min)

# =========================
# 节点定义
# 节点格式: (x, y, demand, e, l, node_type)
# node_type: 0=depot, 1=customer, 2=charging station
# =========================
nodes = {
    0: (0, 0, 0, 0, 200, 0),    # Depot
    1: (2, 0, 5, 20, 80, 1),    # Customer C1
    2: (4, 0, 5, 40, 120, 1),   # Customer C2
    3: (6, 0, 0, 0, 200, 2),    # Charging Station
}

num_nodes = len(nodes)

# =========================
# 距离 & 时间矩阵
# =========================
dist = np.zeros((num_nodes, num_nodes))
for i in nodes:
    for j in nodes:
        dist[i][j] = np.linalg.norm(
            np.array(nodes[i][:2]) - np.array(nodes[j][:2])
        )

time_mat = dist / SPEED

# =========================
# 充电站索引集合（FFP 需要）
# =========================
CHARGING_STATIONS = [i for i in nodes if nodes[i][5] == 2]

# =========================
# 状态定义
# =========================
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
    时间窗约束：到达时间不能晚于最晚时间 l_i
    """
    i = next_node
    arrival = state.clock + time_mat[state.cur_node][i]

    if nodes[i][5] == 1:  # customer
        e, l = nodes[i][3], nodes[i][4]
        return 1 if arrival <= l else 0
    return 1  # depot / CS 无时间窗限制


def capacity_mask(state, next_node):
    """
    TERRAN Eq. (13)
    载重约束：剩余载重必须不小于客户需求
    """
    i = next_node
    if nodes[i][5] == 1:  # customer
        demand = nodes[i][2]
        return 1 if state.load >= demand else 0
    return 1


def energy_mask(state, next_node):
    """
    TERRAN Eq. (16) 简化版（单步检查）
    能量约束：当前电量足够行驶到下一节点
    """
    travel_energy = ETA * time_mat[state.cur_node][next_node]
    return 1 if state.soc >= travel_energy else 0

# ==================================
# ✅ 新增：FFP（未来可行性剪枝）
# ==================================

def ffp_mask(state, next_node):
    """
    Future-Feasibility Pruning (FFP)
    Returns 1 if future feasible, 0 otherwise.
    """
    # 1. 如果是充电站，直接放行（安全兜底）
    if nodes[next_node][5] == 2:
        # print("[FFP Debug] -> ALLOWED: Charging Station (Safety Override).")
        return 1

    # 2. 计算到达候选节点后的剩余电量
    travel_energy = ETA * time_mat[state.cur_node][next_node]
    soc_after_arrival = state.soc - travel_energy

    # 3. 如果连候选节点都到不了，直接阻止
    if soc_after_arrival < 0:
        # print("[FFP Debug] -> Blocked: Cannot even reach candidate node.")
        return 0

    # 4. 查找最近的充电站
    min_dist_to_cs = float('inf')
    for cs in CHARGING_STATIONS:
        dist_to_cs = dist[next_node][cs]
        if dist_to_cs < min_dist_to_cs:
            min_dist_to_cs = dist_to_cs

    # 5. 计算从候选节点到最近充电站所需能耗
    energy_to_cs = ETA * min_dist_to_cs

    # 6. FFP 判定条件
    if soc_after_arrival >= energy_to_cs:
        # print("[FFP Debug] -> ALLOWED: Future feasible.")
        return 1
    else:
        # print("[FFP Debug] -> Blocked: Risk of stranding.")
        return 0


def feasibility_mask(state):
    """
    综合硬约束掩码（加入 FFP）
    M_t(i) = M_time · M_cap · M_energy · M_ffp
    """
    mask = np.zeros(num_nodes, dtype=int)

    for i in range(num_nodes):
        # 已服务客户不能再访问
        if i in state.visited and nodes[i][5] == 1:
            mask[i] = 0
            continue

        # 基础约束
        m_time = time_window_mask(state, i)
        m_cap = capacity_mask(state, i)
        m_energy = energy_mask(state, i)

        # ✅ 新增：FFP 约束
        m_ffp = ffp_mask(state, i)

        # 综合掩码
        mask[i] = m_time * m_cap * m_energy * m_ffp

    return mask


# =========================
# 环境步进逻辑
# =========================
def step(state, action):
    """环境状态更新"""
    i = action
    state.clock += time_mat[state.cur_node][i]

    if nodes[i][5] == 1:  # customer
        state.clock += SERVICE_TIME
        state.load -= nodes[i][2]
        state.soc -= ETA * time_mat[state.cur_node][i]
        state.visited.add(i)
    elif nodes[i][5] == 2:  # charging station
        state.soc = BATTERY
    # depot: do nothing

    state.cur_node = i


def demo():
    print("=== Hard Action Masking Demo with FFP (TERRAN Style) ===\n")
    for t in range(4):
        mask = feasibility_mask(state)
        print(f"Step {t}")
        print(f"Current Node: {state.cur_node}")
        print(f"Clock: {state.clock:.1f}, SoC: {state.soc:.1f}, Load: {state.load}")
        print("Feasibility Mask:", mask)
        print("Allowed Nodes:", [i for i in range(num_nodes) if mask[i] == 1])

        allowed = [i for i in range(num_nodes) if mask[i] == 1]
        if len(allowed) == 0:
            print("No feasible actions!\n")
            break

        action = allowed[0]
        print(f"Selected Action: {action}\n")
        step(state, action)


if __name__ == "__main__":
    demo()