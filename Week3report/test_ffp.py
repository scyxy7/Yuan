"""
Unit Test for Future-Feasibility Pruning (FFP)
Purpose:
    Verify that FFP correctly blocks actions that are
    single-step feasible but lead to energy stranding.
"""

import sys
import os

# 将主程序路径加入，方便直接 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evrptw_mask import (
    State,
    BATTERY,
    ETA,
    nodes,
    dist,
    CHARGING_STATIONS,
    ffp_mask,
    energy_mask
)

def test_ffp_blocks_stranding_risk():
    """
    测试场景：
    车辆当前在 C1，电量15。
    前往 C2：消耗2，剩余13。
    C2 到充电站 C3：距离2，需要2。
    13 >= 2 → FFP 应该允许（返回1）。
    所以这个测试应该期望 ffp_ok == 1。
    """
    print("=== Test Case 1: FFP Allows Safe Action (Should NOT Block) ===")

    state = State()
    state.cur_node = 1      # 当前在 C1
    state.soc = 15           # 电量15

    candidate = 2            # 前往 C2

    single_step_ok = energy_mask(state, candidate)
    ffp_ok = ffp_mask(state, candidate)

    print(f"Current Node: {state.cur_node}")
    print(f"SoC: {state.soc}")
    print(f"Single-step Feasible: {bool(single_step_ok)}")
    print(f"FFP Feasible: {bool(ffp_ok)}")

    assert single_step_ok == 1, "Single-step energy check should pass"
    assert ffp_ok == 1, "FFP should ALLOW this safe action (no stranding risk)"  # ✅ 修正断言

    print("✅ Test 1 PASSED: FFP correctly allowed safe action.\n")


def test_ffp_blocks_real_stranding():
    """
    测试场景：
    车辆当前在 C2，电量3。
    前往 C1：消耗2，剩余1。
    C1 到充电站 C3：距离4，需要4。
    1 < 4 → FFP 应该阻止（返回0）。
    """
    print("=== Test Case 1b: FFP Blocks Real Stranding ===")

    state = State()
    state.cur_node = 2      # 当前在 C2
    state.soc = 3            # 电量3

    candidate = 1            # 前往 C1

    single_step_ok = energy_mask(state, candidate)
    ffp_ok = ffp_mask(state, candidate)

    print(f"Current Node: {state.cur_node}")
    print(f"SoC: {state.soc}")
    print(f"Single-step Feasible: {bool(single_step_ok)}")
    print(f"FFP Feasible: {bool(ffp_ok)}")

    assert single_step_ok == 1, "Single-step energy check should pass"
    assert ffp_ok == 0, "FFP should block this action (risk of stranding)"

    print("✅ Test 1b PASSED: FFP correctly prevented energy stranding.\n")


def test_ffp_allows_safe_actions():
    """
    测试场景：
    车辆当前在 C1，
    电量充足，足以到达 C2 并继续前往充电站。
    → FFP 应返回 1（允许）
    """
    print("=== Test Case 2: FFP Allows Safe Actions ===")

    state = State()
    state.cur_node = 1      # 当前在 C1
    state.soc = 50           # 足够到 C2 再回 CS

    candidate = 2            # 前往 C2

    single_step_ok = energy_mask(state, candidate)
    ffp_ok = ffp_mask(state, candidate)

    print(f"Current Node: {state.cur_node}")
    print(f"SoC: {state.soc}")
    print(f"Single-step Feasible: {bool(single_step_ok)}")
    print(f"FFP Feasible: {bool(ffp_ok)}")

    assert single_step_ok == 1, "Single-step energy check should pass"
    assert ffp_ok == 1, "FFP should allow this safe action"

    print("✅ Test 2 PASSED: FFP correctly allowed safe action.\n")


def test_ffp_ignores_charging_stations():
    """
    测试场景：
    候选节点本身就是充电站。
    → FFP 应始终返回 1
    """
    print("=== Test Case 3: FFP Always Allows Charging Stations ===")

    state = State()
    state.cur_node = 2      # 当前在 C2
    state.soc = 1            # 极低电量

    candidate = 3            # 前往 CS

    ffp_ok = ffp_mask(state, candidate)

    print(f"Current Node: {state.cur_node}")
    print(f"SoC: {state.soc}")
    print(f"Candidate: Charging Station")
    print(f"FFP Feasible: {bool(ffp_ok)}")

    assert ffp_ok == 1, "FFP should never block access to a charging station"

    print("✅ Test 3 PASSED: Charging station always accessible.\n")


if __name__ == "__main__":
    print("Running FFP Unit Tests...\n")

    test_ffp_blocks_stranding_risk()
    test_ffp_allows_safe_actions()
    test_ffp_ignores_charging_stations()

    print("🎉 All FFP tests passed successfully!")