"""
Test Script for Hard Action Masking
"""

from evrptw_mask import *

def test_time_window_violation():
    """故意制造时间窗违规"""
    s = State()
    s.clock = 100  # 把时间拨得很晚
    mask = feasibility_mask(s)
    assert mask[1] == 0, "Customer 1 should be infeasible due to TW"
    print("✅ Time window mask test passed")


def test_capacity_violation():
    """故意制造超载"""
    s = State()
    s.load = 2  # 载重不足
    mask = feasibility_mask(s)
    assert mask[1] == 0, "Customer 1 should be infeasible due to capacity"
    print("✅ Capacity mask test passed")


def test_energy_violation():
    """故意制造电量不足"""
    s = State()
    s.soc = 1  # 电量极低
    mask = feasibility_mask(s)
    assert mask[2] == 0, "Customer 2 should be infeasible due to energy"
    print("✅ Energy mask test passed")


def test_cs_always_allowed():
    """充电站不应被常规约束屏蔽"""
    s = State()
    mask = feasibility_mask(s)
    assert mask[3] == 1, "Charging station should always be allowed"
    print("✅ Charging station accessibility test passed")


if __name__ == "__main__":
    test_time_window_violation()
    test_capacity_violation()
    test_energy_violation()
    test_cs_always_allowed()
    print("\n🎉 All tests passed! Masking logic is correct.")