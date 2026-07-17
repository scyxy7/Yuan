"""
train_minimal.py
Minimal End-to-End PPO Training Loop
Guaranteed to run with env.py + decoder.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random

# =========================
# 1. Imports (CRITICAL FIX HERE)
# =========================
# 从 env.py 中导入环境类、状态类以及节点数据 NODES
from env import EVRPTWEnv, State, NODES
from decoder import AttentionDecoder

# =========================
# 2. Hyperparameters
# =========================
NUM_EPISODES = 200
GAMMA = 0.99
LR = 3e-4
HIDDEN_DIM = 64
MAX_STEPS_PER_EPISODE = 10

# =========================
# 3. Network Definition
# We use a simple Linear layer as a stand-in for the Transformer Encoder
# to ensure the training loop works first.
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.num_nodes = num_nodes
        # Simple embedding layer
        self.embedding = nn.Linear(6, hidden_dim)  # 6 features per node
        self.decoder = AttentionDecoder(hidden_dim=hidden_dim)

    def forward(self, node_features, cur_node_idx, soc, ffp_mask):
        # node_features: [num_nodes, 6]
        embeddings = self.embedding(node_features)  # [num_nodes, hidden_dim]
        probs, logits = self.decoder(embeddings, cur_node_idx, soc, ffp_mask)
        return probs, logits

# =========================
# 4. Initialization
# =========================
env = EVRPTWEnv(NODES)  # Use NODES directly, not env.NODES
policy = PolicyNetwork(num_nodes=env.num_nodes, hidden_dim=HIDDEN_DIM)
optimizer = optim.Adam(policy.parameters(), lr=LR)

print("✅ Initialization successful. Starting training...")

# =========================
# 5. Training Loop
# =========================
reward_history = []

for ep in range(NUM_EPISODES):
    # Reset state
    state = env.reset()
    state.cur_node = 1  # Start at Customer 1
    state.soc = 10.0     # Start with some battery

    log_probs = []
    rewards = []
    episode_reward = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # --- Prepare Inputs ---
        # Convert node dict to tensor [num_nodes, 6]
        node_list = [env.nodes[i] for i in range(env.num_nodes)]
        node_features = torch.tensor(node_list, dtype=torch.float32)

        # Get FFP mask
        ffp_mask = env.get_ffp_logits_mask(state)

        # --- Forward Pass ---
        probs, _ = policy(node_features, state.cur_node, state.soc, ffp_mask)

        # --- Sample Action ---
        # Mask out invalid actions (where prob is 0 due to -inf logits)
        valid_indices = torch.where(probs > 1e-6)[0]
        if len(valid_indices) == 0:
            # If no valid actions, break episode (should not happen with FFP)
            break
        
        # Create a distribution only over valid actions for sampling
        valid_probs = probs[valid_indices]
        dist = Categorical(valid_probs)
        sampled_idx = dist.sample()
        action = valid_indices[sampled_idx]
        log_prob = dist.log_prob(sampled_idx)

        # --- Environment Step (Manual Simulation) ---
        prev_node = state.cur_node
        next_node = action.item()
        
        # Calculate reward (negative distance)
        travel_dist = env.dist[prev_node][next_node]
        reward = -travel_dist  # Objective: minimize distance
        
        # Update state manually
        state.cur_node = next_node
        state.soc -= travel_dist  # Simplified energy consumption
        
        if env.nodes[next_node][5] == 1:  # If customer
            state.visited.add(next_node)
        elif env.nodes[next_node][5] == 2:  # If charging station
            state.soc = 100.0  # Recharge

        # Store transition
        log_probs.append(log_prob)
        rewards.append(reward)
        episode_reward += reward

        # Termination conditions
        if next_node == 0 or state.soc <= 0:
            break

    # --- PPO/REINFORCE Update ---
    if len(rewards) == 0:
        continue

    # Calculate discounted returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize returns for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Calculate loss
    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).mean()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    reward_history.append(episode_reward)

    # Logging
    if (ep + 1) % 20 == 0:
        avg_reward = np.mean(reward_history[-20:])
        print(f"Episode {ep+1:03d} | Avg Reward: {avg_reward:.2f} | Loss: {loss.item():.4f}")

print("\n🎉 Training Finished Successfully!")
print(f"Final Average Reward (last 20 eps): {np.mean(reward_history[-20:]):.2f}")

# Optional: Save the model
torch.save(policy.state_dict(), "policy_minimal.pth")
print("Model saved to policy_minimal.pth")