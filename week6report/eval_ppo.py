"""
eval_ppo.py
Deterministic Evaluation for PPO (Week 5 Final)
Purpose: Verify policy correctness, not performance
"""

import torch
import numpy as np
import sys

from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder

# =========================
# Hyperparameters (must match train_ppo.py)
# =========================
HIDDEN_DIM = 64
MAX_STEPS = 20

# =========================
# Policy Network Definition (must match train_ppo.py)
# =========================
class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding = torch.nn.Linear(6, hidden_dim)
        self.decoder = AttentionDecoder(hidden_dim=hidden_dim)

    def forward(self, node_features, cur_node_idx, soc, ffp_mask):
        embeddings = self.embedding(node_features)
        probs, logits, value = self.decoder(
            embeddings, cur_node_idx, soc, ffp_mask
        )
        return probs, logits, value

# =========================
# Initialization
# =========================
env = EVRPTWEnv(NODES)
policy = PolicyNetwork(env.num_nodes, HIDDEN_DIM)

# Load trained weights
try:
    policy.load_state_dict(torch.load("ppo_policy.pth"))
except FileNotFoundError:
    print("❌ Error: ppo_policy.pth not found.")
    print("Please run 'python train_ppo.py' first to generate the model file.")
    sys.exit(1)

policy.eval()
print("✅ PPO Deterministic Evaluation")
print(f"   Nodes: {env.num_nodes}")

# =========================
# Deterministic Rollout
# =========================
state = env.reset()
total_reward = 0
step_info = []

for t in range(MAX_STEPS):
    node_list = [env.nodes[i] for i in range(env.num_nodes)]
    node_features = torch.tensor(node_list, dtype=torch.float32)
    ffp_mask = env.get_ffp_logits_mask(state)

    with torch.no_grad():
        probs, logits, value = policy(
            node_features, state.cur_node, state.soc, ffp_mask
        )

    # ✅ CRITICAL: Apply mask before argmax
    # Convert numpy mask to boolean tensor
    mask_np = ffp_mask == -np.inf
    mask_tensor = torch.from_numpy(mask_np).to(probs.device)
    
    # Mask out invalid actions
    probs_masked = probs.clone()
    probs_masked[mask_tensor] = 0.0
    
    # Fallback: if all masked, allow staying at current node (should not happen)
    if probs_masked.sum() == 0:
        print(f"⚠️ Warning: All actions masked at step {t}. Forcing stay.")
        probs_masked[state.cur_node] = 1.0

    # Deterministic choice (Argmax)
    action = torch.argmax(probs_masked, dim=-1).item()
    
    dist = torch.distributions.Categorical(probs_masked)
    log_prob = dist.log_prob(torch.tensor(action)).item()

    next_state = env.step(state, action)
    reward = env.compute_reward(state, action, next_state)

    step_info.append({
        "step": t,
        "from": state.cur_node,
        "to": action,
        "prob": probs_masked[action].item(),
        "log_prob": log_prob,
        "reward": reward,
        "soc": next_state.soc,
        "clock": next_state.clock,
        "visited": sorted(list(next_state.visited)),
        "value": value.item()
    })

    total_reward += reward
    state = next_state

    if state.done or state.soc <= 0:
        break

# =========================
# Print Route Summary
# =========================
print("\n=== Route Summary ===")
route_str = " → ".join([str(s["to"]) for s in step_info])
print(f"Route: {route_str}")
print(f"Total Reward: {total_reward:.2f}")
print(f"Visited Customers: {sorted(list(state.visited))}")
print(f"Final SoC: {state.soc:.2f}")
print(f"Final Time: {state.clock:.2f}")

# =========================
# Print Step Details
# =========================
print("\n=== Step Details ===")
print(f"{'Step':<4} {'From':<5} {'To':<5} {'Prob':<8} {'Reward':<8} {'SoC':<8} {'Clock':<8}")
for s in step_info:
    print(
        f"{s['step']:<4} "
        f"{s['from']:<5} "
        f"{s['to']:<5} "
        f"{s['prob']:<8.4f} "
        f"{s['reward']:<8.2f} "
        f"{s['soc']:<8.2f} "
        f"{s['clock']:<8.2f}"
    )

# =========================
# Constraint Verification (Audit)
# =========================
print("\n=== Constraint Audit ===")
violations = []

for s in step_info:
    node_type = env.nodes[s["to"]][5]
    due_time = env.nodes[s["to"]][4]

    # 1. Check revisit
    if node_type == 1 and s["to"] in s["visited"][:-1]:
        violations.append(f"Step {s['step']}: Revisit customer {s['to']}")

    # 2. Check time window
    if s["clock"] > due_time:
        violations.append(
            f"Step {s['step']}: Late arrival at node {s['to']} "
            f"(clock={s['clock']:.1f}, due={due_time})"
        )

    # 3. Check SoC
    if s["soc"] < 0:
        violations.append(f"Step {s['step']}: SoC negative ({s['soc']:.2f})")

    # 4. Check FFP compliance (Sanity check)
    if s["prob"] < 1e-6 and s["to"] != s["from"]:
        violations.append(f"Step {s['step']}: Selected low-prob action ({s['prob']:.6f})")

if violations:
    print("❌ Violations Detected:")
    for v in violations:
        print(f"  - {v}")
else:
    print("✅ No constraint violations detected")

# =========================
# Policy Analysis
# =========================
print("\n=== Policy Analysis ===")
confidences = [s["prob"] for s in step_info]
avg_confidence = np.mean(confidences)
print(f"Average Decision Confidence: {avg_confidence:.4f}")

if avg_confidence > 0.8:
    print("Interpretation: Policy is decisive (high confidence).")
elif avg_confidence > 0.5:
    print("Interpretation: Policy is moderately confident.")
else:
    print("Interpretation: Policy is uncertain (may need more training).")

print("\n🎉 Evaluation Complete")