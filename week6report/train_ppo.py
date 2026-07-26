"""
train_ppo.py
Week 5 · PPO (Proximal Policy Optimization)
Final Stable Version
Compatible with: env.py / decoder.py / encoder.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from env import EVRPTWEnv, NODES
from decoder import AttentionDecoder
# from encoder import TransformerEncoder  # Optional: uncomment to use Transformer

# =========================
# PPO Hyperparameters
# =========================
NUM_EPISODES = 1000
GAMMA = 0.99
LAMBDA = 0.95
LR = 3e-4
HIDDEN_DIM = 64
MAX_STEPS = 20

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
CLIP_EPS = 0.2                 # PPO clip range
UPDATE_EPOCHS = 4              # K epochs per rollout
MINI_BATCH_SIZE = 64           # Mini-batch size

# =========================
# Policy Network (Actor-Critic)
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        # self.encoder = TransformerEncoder(input_dim=6, hidden_dim=hidden_dim)
        self.embedding = nn.Linear(6, hidden_dim)
        self.decoder = AttentionDecoder(hidden_dim=hidden_dim)

    def forward(self, node_features, cur_node_idx, soc, ffp_mask):
        # node_embeddings = self.encoder(node_features)
        node_embeddings = self.embedding(node_features)
        probs, logits, value = self.decoder(
            node_embeddings, cur_node_idx, soc, ffp_mask
        )
        return probs, logits, value

# =========================
# Initialization
# =========================
env = EVRPTWEnv(NODES)
policy = PolicyNetwork(env.num_nodes, HIDDEN_DIM)
optimizer = optim.Adam(policy.parameters(), lr=LR)

print("✅ Week 5 · PPO Training Start")
print(f"   Nodes: {env.num_nodes} | Hidden Dim: {HIDDEN_DIM}")
print(f"   Episodes: {NUM_EPISODES} | Max Steps: {MAX_STEPS}")
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

    # ---------- Collect Trajectory ----------
    for t in range(MAX_STEPS):
        node_list = [env.nodes[i] for i in range(env.num_nodes)]
        node_features = torch.tensor(node_list, dtype=torch.float32)
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

        buffer.states.append((
            node_features.clone(),
            state.cur_node,
            state.soc,
            ffp_mask.copy()
        ))
        buffer.actions.append(action)
        buffer.log_probs.append(log_prob)
        buffer.values.append(value)
        buffer.rewards.append(reward)
        buffer.dones.append(next_state.done)

        state = next_state
        ep_reward += reward

        if state.done or state.soc <= 0:
            break

    # ---------- GAE Computation ----------
    advantages, returns = [], []
    gae, next_value = 0.0, 0.0

    for r, v, done in zip(
        reversed(buffer.rewards),
        reversed(buffer.values),
        reversed(buffer.dones)
    ):
        if done:
            next_value = 0.0
            gae = 0.0

        delta = r + GAMMA * next_value - v.item()
        gae = delta + GAMMA * LAMBDA * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + v.item())
        next_value = v.item()

    buffer.advantages = torch.tensor(advantages, dtype=torch.float32)
    buffer.returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize advantages (critical for stability)
    if len(buffer.advantages) > 1:
        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / \
                           (buffer.advantages.std() + 1e-8)

    # ---------- PPO Update (Multi-Epoch) ----------
    for _ in range(UPDATE_EPOCHS):
        idx = np.random.permutation(len(buffer.advantages))
        for start in range(0, len(idx), MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            batch_idx = idx[start:end]

            # Reconstruct batch
            states_batch = [buffer.states[i] for i in batch_idx]
            nf = torch.stack([s[0] for s in states_batch])
            cn = torch.tensor([s[1] for s in states_batch])
            sc = torch.tensor([s[2] for s in states_batch], dtype=torch.float32)
            fm = torch.stack([
                torch.from_numpy(s[3]) if isinstance(s[3], np.ndarray)
                else s[3] for s in states_batch
            ])

            probs, _, value = policy(nf, cn, sc, fm)
            dist = Categorical(probs)

            new_log_probs = dist.log_prob(
                torch.stack([buffer.actions[i] for i in batch_idx])
            )
            entropy = dist.entropy().mean()

            old_log_probs = torch.stack(
                [buffer.log_probs[i] for i in batch_idx]
            ).detach()

            # PPO Clip Objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * buffer.advantages[batch_idx]
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * buffer.advantages[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(
                value,
                buffer.returns[batch_idx]
            )

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---------- Logging ----------
    if (ep + 1) % 20 == 0:
        print(
            f"Ep {ep+1:03d} | "
            f"Loss: {loss.item():.3f} | "
            f"R: {ep_reward:.2f} | "
            f"Ent: {entropy.item():.3f}"
        )

    buffer.clear()

# =========================
# Save Model
# =========================
torch.save(policy.state_dict(), "ppo_policy.pth")
print("\n🎉 PPO Training Complete")
print("✅ Model saved as 'ppo_policy.pth'")