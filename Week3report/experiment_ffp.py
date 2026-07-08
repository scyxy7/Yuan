import numpy as np
import pandas as pd
import csv
import os
import random
from collections import deque

# ==========================================
# 1. 直接在这里定义所有需要的辅助函数
#    (这样就不需要从 evrptw_mask 导入了)
# ==========================================
def build_matrices(nodes):
    """构建距离矩阵和时间矩阵"""
    n = len(nodes)
    dist = np.zeros((n, n))
    time_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = nodes[i][0] - nodes[j][0]
                dy = nodes[i][1] - nodes[j][1]
                dist[i][j] = np.sqrt(dx**2 + dy**2)
                time_mat[i][j] = dist[i][j] / 1.0  # 假设速度 1.0
    return dist, time_mat

def get_charging_stations(nodes):
    """找出所有充电站节点 (假设节点 0 是 Depot，其他是客户或充电站)"""
    # 简单假设：节点 0 是 Depot，节点 1~N 中，某些是充电站
    # 这里为了演示，假设所有非 Depot 且坐标特殊的都是充电站
    cs = []
    for i, node in enumerate(nodes):
        if i == 0:
            continue  # Depot 不是充电站
        # 假设：如果 x 坐标是整数且 y 坐标是 0，则是充电站 (根据你的实际数据调整)
        if node[0].is_integer() and node[1] == 0:
            cs.append(i)
    return cs

def num_customers(nodes):
    """计算客户数量 (排除 Depot 和充电站)"""
    count = 0
    for i, node in enumerate(nodes):
        if i == 0:
            continue  # Depot
        # 假设：不是充电站的都是客户
        if i not in get_charging_stations(nodes):
            count += 1
    return count

# ==========================================
# 2. 定义 State 类和动作选择逻辑
# ==========================================
class State:
    def __init__(self, soc=100.0, node=0, visited=None):
        self.soc = soc
        self.node = node
        self.visited = visited if visited else set()
        self.path = [node]

    def copy(self):
        return State(self.soc, self.node, self.visited.copy())

def choose_action(mask, custom_nodes, state):
    """根据 Mask 选择一个合法动作 (简化版：随机选)"""
    allowed = [i for i, m in enumerate(mask) if m == 1]
    if not allowed:
        return 0  # 默认回 Depot
    return random.choice(allowed)

def step(state, action, custom_nodes, dist):
    """执行一步动作，更新状态"""
    n = len(custom_nodes)
    current_node = state.node
    next_node = action

    # 计算距离和能耗 (简化：能耗 = 距离)
    d = dist[current_node][next_node]
    state.soc -= d

    # 更新状态
    state.node = next_node
    state.visited.add(next_node)
    state.path.append(next_node)

    # 如果到达 Depot (节点 0)，重置一些状态 (可选)
    if next_node == 0:
        state.soc = 100.0  # 假设回到 Depot 充满电

# ==========================================
# 3. 定义实验场景和主逻辑
# ==========================================
SCENARIOS = {
    "A": [(0,0), (1,1), (2,2), (3,0)],  # Depot, C1, C2, CS
    "B": [(0,0), (1,2), (2,1), (3,3), (4,0)],  # 更多客户
    "C": [(0,0), (1,1), (2,3), (3,2), (4,4), (5,0)]
}

NUM_EPISODES = 100  # 每个场景每个方法跑 100 次

def run_episode(scenario_name, custom_nodes, dist, time_mat, cs_list, use_ffp):
    """运行单次实验"""
    try:
        state = State()
        steps = 0
        stranding = False

        while steps < 100:  # 防止死循环
            # 构建 Mask (简化版：禁止访问已访问节点)
            mask = [1] * len(custom_nodes)
            mask[0] = 1  # Depot 始终允许 (或者根据策略禁用)

            # 禁用已访问节点
            for v in state.visited:
                if v < len(mask):
                    mask[v] = 0

            # FFP 逻辑：如果电量低，优先去充电站
            if use_ffp and state.soc < 20:
                for cs in cs_list:
                    if cs < len(mask):
                        mask[cs] = 1  # 强制允许去充电站

            allowed = [i for i, m in enumerate(mask) if m == 1]
            if not allowed:
                stranding = True
                break

            action = choose_action(mask, custom_nodes, state)
            step(state, action, custom_nodes, dist)
            steps += 1

            if len(state.visited) >= num_customers(custom_nodes):
                break

        return {
            "scenario": scenario_name,
            "method": "FFP" if use_ffp else "Baseline",
            "stranding": int(stranding),
            "final_soc": round(state.soc, 2),
            "steps": steps
        }

    except Exception as e:
        return {
            "scenario": scenario_name,
            "method": "FFP" if use_ffp else "Baseline",
            "stranding": 1,
            "final_soc": -1,
            "steps": -1,
            "error": str(e)
        }

# ==========================================
# 4. 主函数：运行所有实验并保存 CSV
# ==========================================
def main():
    results = []
    
    # 🚨 指定你要求的绝对路径 🚨
    target_folder = r"C:\Users\58083\OneDrive - The University of Nottingham Ningbo China\桌面\Week3科研成果 - 副本"
    os.makedirs(target_folder, exist_ok=True)
    csv_path = os.path.join(target_folder, "experiment_results.csv")

    print("🚀 Starting FFP Experiments...")
    print(f"📁 Results will be saved to: {csv_path}")
    print("=" * 60)

    for scenario_name, custom_nodes in SCENARIOS.items():
        dist, time_mat = build_matrices(custom_nodes)
        cs_list = get_charging_stations(custom_nodes)

        for method in ["Baseline", "FFP"]:
            print(f"Running {scenario_name} - {method} ... ", end="", flush=True)
            use_ffp = (method == "FFP")

            for ep in range(NUM_EPISODES):
                res = run_episode(
                    scenario_name,
                    custom_nodes,
                    dist,
                    time_mat,
                    cs_list,
                    use_ffp
                )
                results.append(res)

            print("✅ Done")

    # 保存 CSV
    fieldnames = ["scenario", "method", "stranding", "final_soc", "steps", "error"]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ Experiment finished!")
    print(f"📄 Results saved to: {csv_path}")
    print(f"📊 Total episodes recorded: {len(results)}")

    # 打印 Summary
    print("\n📈 Summary Table")
    print("=" * 60)
    print(f"{'Scenario':<10} {'Method':<10} {'Stranding':<12} {'Avg SoC':<12}")
    print("-" * 60)

    for scenario in SCENARIOS:
        for method in ["Baseline", "FFP"]:
            data = [r for r in results if r["scenario"] == scenario and r["method"] == method]
            if not data:
                continue
            sr = sum(r["stranding"] for r in data) / len(data)
            valid_soc = [r["final_soc"] for r in data if r["final_soc"] >= 0]
            avg_soc = np.mean(valid_soc) if valid_soc else -1
            print(f"{scenario:<10} {method:<10} {sr:<12.3f} {avg_soc:<12.2f}")

if __name__ == "__main__":
    main()