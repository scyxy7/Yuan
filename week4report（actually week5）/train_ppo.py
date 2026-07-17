import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import time

# 假设你已有以下模块（根据你的项目结构调整导入）
from env import EVRPTWEnv, State
from policy import AttentionDecoder  # 你刚修复的那个

# =========================
# PPO Hyperparameters
# =========================
LR = 3e-4
GAMMA = 0.99
CLIP_RATIO = 0.2
EPOCHS = 4
BATCH_SIZE = 32
MAX_EPISODES = 500
HIDDEN_DIM = 64
NUM_NODES = 4  # 你的 Toy Instance

# =========================
# PPO Agent
# =========================
class PPOAgent:
    def __init__(self, num_nodes, hidden_dim):
        self.policy = AttentionDecoder(hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.memory = []

    def select_action(self, state, ffp_mask):
        with torch.no_grad():
            embeddings = torch.randn(NUM_NODES, HIDDEN_DIM)
            probs, _ = self.policy(embeddings, state.cur_node, state.soc, ffp_mask)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, probs

    def store_transition(self, state, action, log_prob, reward, done, ffp_mask):
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'done': done,
            'ffp_mask': ffp_mask
        })

    def update(self):
        states = [m['state'] for m in self.memory]
        actions = torch.tensor([m['action'] for m in self.memory])
        old_log_probs = torch.stack([m['log_prob'] for m in self.memory])
        rewards = [m['reward'] for m in self.memory]
        dones = [m['done'] for m in self.memory]
        ffp_masks = [m['ffp_mask'] for m in self.memory]

        # Compute returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Optimize policy
        for _ in range(EPOCHS):
            indices = np.random.permutation(len(self.memory))
            for i in range(0, len(indices), BATCH_SIZE):
                batch_idx = indices[i:i+BATCH_SIZE]
                batch_states = [states[idx] for idx in batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_ffp_masks = [ffp_masks[idx] for idx in batch_idx]

                # Recompute probs and log_probs
                new_log_probs = []
                for s, m in zip(batch_states, batch_ffp_masks):
                    emb = torch.randn(NUM_NODES, HIDDEN_DIM)
                    probs, _ = self.policy(emb, s.cur_node, s.soc, m)
                    dist = Categorical(probs)
                    new_log_prob = dist.log_prob(batch_actions[len(new_log_probs)])
                    new_log_probs.append(new_log_prob)
                new_log_probs = torch.stack(new_log_probs)

                # Ratio and clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_returns
                surr2 = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * batch_returns
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropies = []
                for s, m in zip(batch_states, batch_ffp_masks):
                    emb = torch.randn(NUM_NODES, HIDDEN_DIM)
                    probs, _ = self.policy(emb, s.cur_node, s.soc, m)
                    dist = Categorical(probs)
                    entropies.append(dist.entropy())
                entropy = torch.mean(torch.stack(entropies))

                loss = policy_loss - 0.01 * entropy  # entropy coefficient

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()

# =========================
# Training Loop
# =========================
def train():
    env = EVRPTWEnv(NODES)
    agent = PPOAgent(num_nodes=NUM_NODES, hidden_dim=HIDDEN_DIM)

    episode_rewards = []
    episode_lengths = []

    print("🚀 Starting PPO Training...")
    start_time = time.time()

    for episode in range(MAX_EPISODES):
        state = State()
        state.cur_node = 1  # 固定起点
        state.soc = 10.0     # 初始电量
        total_reward = 0
        steps = 0

        while not state.done:  # 你需要实现 state.done 逻辑，或用最大步数限制
            ffp_mask = env.get_ffp_logits_mask(state)
            action, log_prob, _ = agent.select_action(state, ffp_mask)

            # 模拟环境步进（你需要实现 step 函数）
            # 这里简化为：奖励 = -distance, 电量减少
            next_node = action
            travel_cost = env.dist[state.cur_node][next_node]
            state.soc -= travel_cost
            state.cur_node = next_node
            state.visited.add(next_node)

            reward = -travel_cost  # 简单奖励设计
            done = (next_node == 0) or (state.soc <= 0)  # 到达 depot 或没电

            agent.store_transition(state, action, log_prob, reward, done, ffp_mask)

            total_reward += reward
            steps += 1

            if done or steps > 50:  # 防止无限循环
                break

        agent.update()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode+1:03d} | Avg Reward: {avg_reward:.2f} | Steps: {np.mean(episode_lengths[-50:]):.1f}")

    print(f"\n✅ Training completed in {time.time() - start_time:.2f}s")

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("PPO Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("ppo_rewards.png")
    plt.show()

if __name__ == "__main__":
    train()