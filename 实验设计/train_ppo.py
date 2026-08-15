"""
train_ppo.py
Week 6 · PPO + Transformer Encoder
Final Stable Version · No-Revisit Aligned
Compatible with: env.py / decoder.py / encoder.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import time

from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder
from encoder import TransformerEncoder  # ✅ 启用 Encoder

# =========================
# PPO Hyperparameters
# =========================
NUM_EPISODES = 400  # ✅ 稍微加一点，Encoder 收敛稍慢
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
HIDDEN_DIM = 64
MAX_STEPS = 30

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
CLIP_EPS = 0.2
UPDATE_EPOCHS = 4
MINI_BATCH_SIZE = 64
CRITIC_LR = 1e-4

# =========================
# Policy Network (Actor-Critic + Transformer Encoder)
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes

        # ✅ Transformer Encoder (replaces linear embedding)
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PolicyNetwork(env.num_nodes, HIDDEN_DIM).to(device)

critic_params = list(policy.decoder.value_head.parameters())
actor_params = [p for n, p in policy.named_parameters() if 'decoder.value_head' not in n]

optimizer = optim.Adam([
    {'params': actor_params, 'lr': LR},
    {'params': critic_params, 'lr': CRITIC_LR}
])

print("✅ Week 6 · PPO + Transformer Encoder Training")
print(f"   Nodes: {env.num_nodes} | Hidden Dim: {HIDDEN_DIM}")
print(f"   Episodes: {NUM_EPISODES} | Max Steps: {MAX_STEPS}")
print(f"   Encoder: Transformer | Decoder: Attention")
print(f"   Clip Eps: {CLIP_EPS} | Update Epochs: {UPDATE_EPOCHS}")
print(f"   LR: {LR} | Entropy Coef: {ENTROPY_COEF}")

# =========================
# Rollout Buffer
# =========================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def clear(self):
        self.__init__()

buffer = RolloutBuffer()

# =========================
# Training Loop
# =========================
for ep in range(NUM_EPISODES):
    state = env.reset()
    ep_reward = 0
    ep_len = 0
    start_time = time.time()

    # ---------- Collect Trajectory ----------
    for t in range(MAX_STEPS):
        node_list = [env.nodes[i] for i in range(env.num_nodes)]
        node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
        ffp_mask = env.get_ffp_logits_mask(state)

        with torch.no_grad():
            probs, _, value = policy(
                node_features, state.cur_node, state.soc, ffp_mask
            )

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state = env.step(state, action.item())
        reward = env.compute_reward(state, action.item(), next_state)

        buffer.states.append(
            (
                node_features.clone().detach(),
                state.cur_node,
                state.soc,
                ffp_mask.copy()
            )
        )
        buffer.actions.append(action.cpu())
        buffer.log_probs.append(log_prob.cpu().detach())
        buffer.values.append(value.item())
        buffer.rewards.append(reward)
        buffer.dones.append(next_state.done)

        state = next_state
        ep_reward += reward
        ep_len += 1

        if state.done or state.soc <= 0:
            break

    # ---------- GAE Computation ----------
    advantages, returns = [], []
    gae = 0.0
    next_value = 0.0

    if len(buffer.dones) > 0 and not buffer.dones[-1]:
        last_state = buffer.states[-1]
        last_nf = last_state[0]
        last_cn = last_state[1]
        last_sc = last_state[2]
        last_fm = last_state[3]
        if isinstance(last_fm, np.ndarray):
            last_fm_t = torch.from_numpy(last_fm).to(device)
        else:
            last_fm_t = last_fm.to(device)

        with torch.no_grad():
            _, _, nv = policy(last_nf.to(device), last_cn, last_sc, last_fm_t)
            next_value = float(nv.item())

    for r, v, done in zip(
        reversed(buffer.rewards),
        reversed(buffer.values),
        reversed(buffer.dones)
    ):
        if done:
            next_value = 0.0
            gae = 0.0

        delta = r + GAMMA * next_value - v
        gae = delta + GAMMA * LAMBDA * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + v)
        next_value = v

    buffer.advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    buffer.returns = torch.tensor(returns, dtype=torch.float32, device=device)

    if len(buffer.advantages) > 1:
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / \
                           (buffer.advantages.std() + 1e-8)

    if len(buffer.returns) > 1:
        buffer.returns = (buffer.returns - buffer.returns.mean()) / \
                         (buffer.returns.std() + 1e-8)

    # ---------- PPO Update ----------
    epoch_policy_loss_sum = 0.0
    epoch_value_loss_sum = 0.0
    epoch_entropy_sum = 0.0
    epoch_value_mean_sum = 0.0
    epoch_value_std_sum = 0.0
    epoch_batches = 0

    for _ in range(UPDATE_EPOCHS):
        perm = np.random.permutation(len(buffer.advantages))
        for start in range(0, len(perm), MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            batch_idx = perm[start:end]

            states_batch = [buffer.states[i] for i in batch_idx]
            nf = torch.stack([s[0] for s in states_batch]).to(device)
            cn = torch.tensor([s[1] for s in states_batch], device=device)
            sc = torch.tensor([s[2] for s in states_batch], dtype=torch.float32, device=device)
            fm = torch.stack([
                torch.from_numpy(s[3]) if isinstance(s[3], np.ndarray)
                else s[3] for s in states_batch
            ]).to(device)

            probs, _, value = policy(nf, cn, sc, fm)
            dist = Categorical(probs)

            actions_tensor = torch.tensor(
                [int(buffer.actions[i].item()) for i in batch_idx],
                dtype=torch.long, device=device
            )
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            old_log_probs = torch.stack(
                [buffer.log_probs[i] for i in batch_idx]
            ).to(device).detach()

            batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=device)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * buffer.advantages[batch_idx_t]
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * buffer.advantages[batch_idx_t]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(
                value,
                buffer.returns[batch_idx_t]
            )

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_policy_loss_sum += policy_loss.item()
            epoch_value_loss_sum += value_loss.item()
            epoch_entropy_sum += entropy.item()
            epoch_value_mean_sum += value.mean().item()
            epoch_value_std_sum += value.std().item()
            epoch_batches += 1

    avg_policy_loss = epoch_policy_loss_sum / epoch_batches if epoch_batches > 0 else 0.0
    avg_value_loss = epoch_value_loss_sum / epoch_batches if epoch_batches > 0 else 0.0
    avg_entropy = epoch_entropy_sum / epoch_batches if epoch_batches > 0 else 0.0
    avg_value_mean = epoch_value_mean_sum / epoch_batches if epoch_batches > 0 else 0.0
    avg_value_std = epoch_value_std_sum / epoch_batches if epoch_batches > 0 else 0.0

    if 'rewards_history' not in globals():
        rewards_history = []
        success_history = []
        ep_length_history = []

    rewards_history.append(ep_reward)
    success_history.append(1 if state.done else 0)
    ep_length_history.append(ep_len)

    if (ep + 1) % 10 == 0 or ep == NUM_EPISODES - 1:
        recent_mean = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else 0.0
        recent_std = np.std(rewards_history[-10:]) if len(rewards_history) >= 10 else 0.0
        recent_success = np.mean(success_history[-10:]) if len(success_history) >= 10 else 0.0

        print(
            f"Ep {ep+1:03d} | "
            f"Pol: {avg_policy_loss:.3f} | Val: {avg_value_loss:.3f} | "
            f"Vμ: {avg_value_mean:.2f} | Vσ: {avg_value_std:.2f} | "
            f"R: {ep_reward:.2f} (μ10={recent_mean:.1f},σ10={recent_std:.1f}) | "
            f"Ent: {avg_entropy:.3f} | Len: {ep_len} | "
            f"Succ10: {recent_success:.2f} | "
            f"Time: {time.time()-start_time:.2f}s"
        )

    buffer.clear()

# =========================
# Save Model
# =========================
torch.save(policy.state_dict(), "ppo_policy_transformer.pth")
print("\n🎉 PPO + Transformer Encoder Training Complete")
print("✅ Model saved as 'ppo_policy_transformer.pth'")