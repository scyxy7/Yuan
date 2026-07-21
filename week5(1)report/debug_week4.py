"""
debug_week4.py
End-to-End Debugger for Week 4 Training Issues
Run this BEFORE touching any code.
"""

import torch
import numpy as np
from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder
from encoder import TransformerEncoder, build_node_features

# =========================
# 1. 环境 & 模型初始化
# =========================
print("🔧 Step 1: Initializing Env & Models...")
env = EVRPTWEnv(NODES)
encoder = TransformerEncoder(input_dim=6, hidden_dim=64)
decoder = AttentionDecoder(hidden_dim=64)

print("✅ Env & Models loaded.")

# =========================
# 2. 检查 Reward 函数是否存在 & 是否生效
# =========================
print("\n🔧 Step 2: Checking Reward Function...")
state = env.reset()
state.cur_node = 1
state.soc = 80.0

try:
    reward = env.compute_reward(state, 2, env.step(state, 2))
    print(f"✅ compute_reward() exists.")
    print(f"   Sample reward (1→2): {reward:.4f}")

    if abs(reward + env.dist[1][2]) < 1e-3:
        print("⚠️ WARNING: Reward looks like '-distance' only.")
        print("   You may be missing time window / completion bonuses.")
    elif reward > 0:
        print("✅ Reward contains positive signals (good!).")
    else:
        print("⚠️ Reward is still negative. Check bonus weights.")

except AttributeError:
    print("❌ CRITICAL: env.compute_reward() does NOT exist!")
    print("   This explains why Reward never moves.")
    exit(1)

# =========================
# 3. 检查 Encoder → Decoder 数据流
# =========================
print("\n🔧 Step 3: Checking Encoder → Decoder Flow...")
node_features = build_node_features(env)
embeddings = encoder(node_features).squeeze(0)

print(f"   Node features shape: {node_features.shape}")
print(f"   Embeddings shape: {embeddings.shape}")

if embeddings.shape != (env.num_nodes, 64):
    print("❌ CRITICAL: Embedding shape mismatch!")
    exit(1)

# =========================
# 4. 检查 Decoder 输出（Value Head）
# =========================
print("\n🔧 Step 4: Checking Decoder Outputs...")
ffp_mask = env.get_ffp_logits_mask(state)
probs, logits, value = decoder(embeddings, state.cur_node, state.soc, ffp_mask)

print(f"   Probs shape: {probs.shape}")
print(f"   Logits shape: {logits.shape}")
print(f"   Value: {value.item():.6f}")

if torch.isnan(value) or torch.isinf(value):
    print("❌ CRITICAL: Value is NaN or Inf!")
    print("   This will kill your training.")
    exit(1)

if abs(value.item()) < 1e-6:
    print("⚠️ WARNING: Value is near zero. May cause Advantage collapse.")

# =========================
# 5. 模拟一个完整的训练 Step（检查梯度）
# =========================
print("\n🔧 Step 5: Simulating One Training Step...")

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=3e-4
)

# Fake trajectory
log_probs = []
values = []
returns = torch.tensor([-10.0, -5.0, 0.0], dtype=torch.float32)

for i in range(3):
    probs, _, value = decoder(embeddings, state.cur_node, state.soc, ffp_mask)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    log_probs.append(log_prob)
    values.append(value)

# Loss calculation
policy_loss = 0
value_loss = 0
for log_prob, value, ret in zip(log_probs, values, returns):
    advantage = ret - value.detach()
    policy_loss -= log_prob * advantage
    value_loss += torch.nn.functional.mse_loss(value, ret)

total_loss = policy_loss + 0.5 * value_loss

print(f"   Policy Loss: {policy_loss.item():.4f}")
print(f"   Value Loss: {value_loss.item():.4f}")
print(f"   Total Loss: {total_loss.item():.4f}")

# Backward pass
optimizer.zero_grad()
total_loss.backward()

# Check gradients
enc_grad_norm = sum(
    p.grad.norm().item() ** 2
    for p in encoder.parameters() if p.grad is not None
) ** 0.5

dec_grad_norm = sum(
    p.grad.norm().item() ** 2
    for p in decoder.parameters() if p.grad is not None
) ** 0.5

print(f"   Encoder Grad Norm: {enc_grad_norm:.6f}")
print(f"   Decoder Grad Norm: {dec_grad_norm:.6f}")

if enc_grad_norm < 1e-6 or dec_grad_norm < 1e-6:
    print("❌ CRITICAL: Gradient vanished!")
    print("   Training will not progress.")
    exit(1)

print("\n✅ Gradient flow is healthy.")

# =========================
# 6. 最终诊断结论
# =========================
print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

issues = []

if reward <= 0:
    issues.append("Reward is negative or zero.")
if abs(value.item()) < 1e-3:
    issues.append("Value prediction is too small.")
if enc_grad_norm < 0.001:
    issues.append("Encoder gradient is vanishing.")
if dec_grad_norm < 0.001:
    issues.append("Decoder gradient is vanishing.")

if not issues:
    print("🎉 NO CRITICAL ISSUES FOUND.")
    print("Your code should train normally.")
    print("If Reward still doesn't move, increase training episodes.")
else:
    print("⚠️ ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nRecommended fixes:")
    print("  1. Ensure env.compute_reward() is used in train_minimal.py")
    print("  2. Increase reward bonus for visiting customers")
    print("  3. Normalize returns before loss calculation")

print("=" * 60)