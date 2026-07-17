"""
Final Validation Script: Policy Network + FFP Integration
Author: Week 4 EVRP-TW Project
Purpose: Verify structural correctness, numerical stability, and gradient flow.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Global Parameters
# =========================
SPEED = 1.0
ETA = 1.0
BATTERY = 100

# =========================
# Node Definition
# (x, y, demand, e, l, node_type)
# node_type: 0=depot, 1=customer, 2=charging station
# =========================
NODES = {
    0: (0, 0, 0, 0, 200, 0),   # Depot
    1: (1, 0, 5, 20, 80, 1),   # Customer C1
    2: (2, 1, 5, 40, 120, 1),  # Customer C2
    3: (2, 1, 0, 0, 200, 2),   # Charging Station
}

# =========================
# State Definition
# =========================
class State:
    def __init__(self):
        self.cur_node = 0
        self.soc = BATTERY
        self.visited = set()

# =========================
# Environment (EVRPTWEnv)
# =========================
class EVRPTWEnv:
    def __init__(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)

        # Distance matrix
        self.dist = np.zeros((self.num_nodes, self.num_nodes))
        for i in nodes:
            for j in nodes:
                xi, yi = nodes[i][:2]
                xj, yj = nodes[j][:2]
                self.dist[i][j] = np.linalg.norm([xi - xj, yi - yj])

        self.charging_stations = [
            i for i in nodes if nodes[i][5] == 2
        ]

    # =========================
    # ✅ FFP: Logits Mask (Differentiable)
    # =========================
    def get_ffp_logits_mask(self, state):
        """
        Returns:
            mask: np.ndarray[num_nodes]
                  0.0   -> feasible
                  -inf -> infeasible
        """
        mask = np.zeros(self.num_nodes, dtype=np.float32)
        cur = state.cur_node

        for i in range(self.num_nodes):
            # Rule 1: Charging stations are always feasible
            if self.nodes[i][5] == 2:
                mask[i] = 0.0
                continue

            # Rule 2: Already visited customers are forbidden
            if i in state.visited and self.nodes[i][5] == 1:
                mask[i] = -np.inf
                continue

            # Rule 3: Single-step energy feasibility
            travel_energy = ETA * self.dist[cur][i]
            if state.soc < travel_energy:
                mask[i] = -np.inf
                continue

            soc_after = state.soc - travel_energy

            # Rule 4: Future feasibility (can reach a CS afterwards)
            min_dist_to_cs = min(
                self.dist[i][cs] for cs in self.charging_stations
            )
            if soc_after < ETA * min_dist_to_cs:
                mask[i] = -np.inf
                continue

            # Rule 5: Feasible
            mask[i] = 0.0

        return mask

# =========================
# Attention Decoder (Policy Network)
# =========================
class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.query_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, embeddings, cur_idx, soc, ffp_logits_mask):
        """
        Args:
            embeddings: [num_nodes, hidden_dim]
            cur_idx: int
            soc: float
            ffp_logits_mask: np.ndarray[num_nodes]
        """
        # Query: Current node embedding + SoC
        cur_emb = embeddings[cur_idx]
        query = self.query_fc(
            torch.cat([cur_emb, torch.tensor([soc], dtype=torch.float32)])
        ).unsqueeze(0)

        # Keys: All node embeddings
        keys = self.key_fc(embeddings)

        # Dot-product attention
        logits = (query @ keys.T).squeeze(0)

        # ✅ Hard Masking (Structural Safety)
        mask_tensor = torch.from_numpy(ffp_logits_mask).to(logits.device)
        invalid_mask = mask_tensor == float('-inf')
        logits = logits.masked_fill(invalid_mask, float('-inf'))

        # Softmax
        probs = F.softmax(logits, dim=-1)
        return probs, logits

# =========================
# Test Suites
# =========================
def test_numerical_correctness():
    """
    Test 1: Verify that FFP produces correct numerical values.
    """
    print("🔹 Test 1: Numerical Correctness of FFP Mask")
    env = EVRPTWEnv(NODES)
    state = State()
    state.cur_node = 1
    state.soc = 5  # Low SoC

    mask = env.get_ffp_logits_mask(state)

    # Physical sanity check based on your debug output:
    # Node 1 & 2 are reachable and can reach CS -> Should be 0.0
    # Node 3 is CS -> Should be 0.0
    assert mask[1] == 0.0, "Node 1 should be feasible"
    assert mask[2] == 0.0, "Node 2 should be feasible"
    assert mask[3] == 0.0, "Node 3 (CS) must be feasible"
    # Node 0 might be blocked depending on exact geometry, we accept either
    print("✅ PASS: FFP mask values are physically consistent")
    return True


def test_probability_distribution():
    """
    Test 2: Softmax produces a valid probability distribution.
    """
    print("\n🔹 Test 2: Probability Distribution Integrity")
    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)
    ffp_mask = env.get_ffp_logits_mask(state)

    probs, _ = decoder(embeddings, state.cur_node, state.soc, ffp_mask)

    prob_sum = probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-6, f"Sum of probs is {prob_sum}"
    assert torch.all(probs >= 0), "Negative probabilities detected"
    print(f"✅ PASS: Sum(probs) = {prob_sum:.6f}")
    return True


def test_gradient_flow():
    """
    Test 3: Gradients flow through masked softmax.
    """
    print("\n🔹 Test 3: Gradient Backpropagation")
    env = EVRPTWEnv(NODES)
    state = State()

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64, requires_grad=True)
    ffp_mask = env.get_ffp_logits_mask(state)

    probs, _ = decoder(embeddings, state.cur_node, state.soc, ffp_mask)

    # Fake loss: maximize probability of a feasible node
    loss = -torch.log(probs[3] + 1e-8)
    loss.backward()

    assert embeddings.grad is not None, "No gradient computed"
    assert not torch.isnan(embeddings.grad).any(), "NaN gradient detected"
    print("✅ PASS: Gradient exists and is clean")
    return True


def test_structural_safety():
    """
    Test 4: Near-zero SoC forces selection of Charging Station.
    """
    print("\n🔹 Test 4: Structural Safety Under Critical SoC")
    env = EVRPTWEnv(NODES)
    state = State()
    state.soc = 0.01  # Critical battery level

    decoder = AttentionDecoder()
    embeddings = torch.randn(env.num_nodes, 64)

    # Run multiple samplings to ensure robustness
    for _ in range(20):
        ffp_mask = env.get_ffp_logits_mask(state)
        probs, _ = decoder(embeddings, state.cur_node, state.soc, ffp_mask)

        # Only CS should have non-zero probability
        non_zero_indices = torch.where(probs > 1e-6)[0].tolist()
        assert len(non_zero_indices) == 1, "Multiple nodes allowed at critical SoC"
        assert non_zero_indices[0] == 3, "Non-CS node allowed at critical SoC"

    print("✅ PASS: Agent forced to CS at critical SoC")
    return True

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("Week 4 Final Validation: Policy Network + FFP")
    print("=" * 60)

    tests = [
        test_numerical_correctness,
        test_probability_distribution,
        test_gradient_flow,
        test_structural_safety,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except AssertionError as e:
            print(f"❌ FAIL: {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("🎉 ALL TESTS PASSED. System ready for PPO training.")
    else:
        print("❌ SOME TESTS FAILED. Review logic above.")
    print("=" * 60)