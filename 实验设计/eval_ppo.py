"""
eval_ppo.py
Deterministic Evaluation for PPO (Week 6 · Transformer Encoder)
Purpose: Verify policy correctness with Transformer-based embeddings
"""

import torch
import numpy as np
import sys

from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder
from encoder import TransformerEncoder  # ✅ 启用 Transformer Encoder

# =========================
# Hyperparameters (MUST match train_ppo.py)
# =========================
HIDDEN_DIM = 64
MAX_STEPS = 30

# =========================
# Policy Network Definition (Transformer Version)
# =========================
class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes

        # ✅ Transformer Encoder (same as train_ppo.py)
        self.encoder = TransformerEncoder(
            input_dim=6,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

        self.decoder = AttentionDecoder(hidden_dim=hidden_dim)

    def forward(self, node_features, cur_node_idx, soc, ffp_mask):
        # ✅ Encode node features with Transformer
        node_embeddings = self.encoder(node_features)

        # ✅ Decoder remains unchanged
        probs, logits, value = self.decoder(
            node_embeddings, cur_node_idx, soc, ffp_mask
        )
        return probs, logits, value

# =========================
# Initialization
# =========================
env = EVRPTWEnv(NODES)
policy = PolicyNetwork(env.num_nodes, HIDDEN_DIM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained weights (Transformer version) with proper map_location
try:
    policy.load_state_dict(torch.load("ppo_policy_transformer.pth", map_location=device))
except FileNotFoundError:
    print("❌ Error: ppo_policy_transformer.pth not found.")
    print("Please run 'python train_ppo.py' first to train the Transformer model.")
    sys.exit(1)

policy.eval()
policy.to(device)

print("✅ PPO Deterministic Evaluation (Transformer Encoder Enabled)")
print(f"   Nodes: {env.num_nodes}")

# =========================
# Deterministic Rollout
# =========================
state = env.reset()
total_reward = 0
step_info = []
route = [0]  # Start from depot

for t in range(MAX_STEPS):
    node_list = [env.nodes[i] for i in range(env.num_nodes)]
    node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
    ffp_mask = env.get_ffp_logits_mask(state)

    with torch.no_grad():
        probs, logits, value = policy(
            node_features, state.cur_node, state.soc, ffp_mask
        )

    # ✅ Apply mask before argmax
    mask_np = np.isneginf(ffp_mask)
    mask_tensor = torch.from_numpy(mask_np).to(device)

    probs_masked = probs.clone()
    if probs_masked.dim() == 2 and mask_tensor.dim() == 1:
        mask_tensor = mask_tensor.unsqueeze(0).expand_as(probs_masked)

    probs_masked[mask_tensor] = 0.0

    # Fallback: if all masked, force stay
    if probs_masked.sum() == 0:
        print(f"⚠️ Warning: All actions masked at step {t}. Forcing stay.")
        if probs_masked.dim() == 2:
            probs_masked[0, state.cur_node] = 1.0
        else:
            probs_masked[state.cur_node] = 1.0

    if probs_masked.dim() == 2:
        action = torch.argmax(probs_masked, dim=-1).item()
        logits_for_dist = probs_masked[0]
    else:
        action = torch.argmax(probs_masked, dim=-1).item()
        logits_for_dist = probs_masked

    dist = torch.distributions.Categorical(logits_for_dist)
    log_prob = dist.log_prob(torch.tensor(action, device=device)).item()

    next_state = env.step(state, action)
    reward = env.compute_reward(state, action, next_state)

    prob_value = logits_for_dist[action].item()

    step_info.append({
        "step": t,
        "from": state.cur_node,
        "to": action,
        "prob": prob_value,
        "log_prob": log_prob,
        "reward": reward,
        "soc": next_state.soc,
        "clock": next_state.clock,
        "visited": sorted(list(next_state.visited)),
        "value": value.item()
    })

    route.append(action)
    total_reward += reward
    state = next_state

    if state.done or state.soc <= 0:
        break

# =========================
# Print Route Summary
# =========================
print("\n=== Route Summary ===")
route_str = " → ".join(map(str, route))
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
# Constraint Verification (True Revisit Detection)
# =========================
print("\n=== Constraint Audit ===")
violations = []

full_route = [s["to"] for s in step_info]
full_route.insert(0, 0)

customer_nodes = [n for n in full_route if env.nodes[n][5] == 1]
if len(customer_nodes) != len(set(customer_nodes)):
    seen = set()
    for c in customer_nodes:
        if c in seen:
            violations.append(f"True revisit detected: customer {c}")
            break
        seen.add(c)

for i in range(len(full_route) - 1):
    if full_route[i] == full_route[i + 1]:
        violations.append(f"Step {i}: Self-loop at node {full_route[i]}")

for s in step_info:
    due_time = env.nodes[s["to"]][4]
    if s["clock"] > due_time:
        violations.append(
            f"Step {s['step']}: Late arrival at node {s['to']} "
            f"(clock={s['clock']:.1f}, due={due_time})"
        )

for s in step_info:
    if s["soc"] < 0:
        violations.append(f"Step {s['step']}: SoC negative ({s['soc']:.2f})")

for s in step_info:
    if s["prob"] < 1e-6 and s["to"] != s["from"]:
        violations.append(
            f"Step {s['step']}: Selected low-prob action ({s['prob']:.6f})"
        )

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

# =========================
# Success Check
# =========================
print("\n=== Success Check ===")
if len(state.visited) == len(env.customer_nodes):
    print("✅ Mission Accomplished: All customers served!")
else:
    print(f"❌ Incomplete: Only {len(state.visited)}/{len(env.customer_nodes)} customers served.")

print("\n🎉 Transformer Evaluation Complete")