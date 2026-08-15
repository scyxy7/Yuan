"""
train_linear.py
训练使用 `baseline.PolicyNetworkLinear` 的 PPO（简化版本，适合小规模实验）
保存 checkpoint 到 `checkpoints/linear_seed{seed}.pth`
"""

import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import EVRPTWEnv, NODES
from baseline import PolicyNetworkLinear


def train(seed=0, episodes=200, max_steps=30, hidden_dim=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = EVRPTWEnv(NODES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetworkLinear(env.num_nodes, hidden_dim).to(device)

    critic_params = list(policy.decoder.value_head.parameters())
    actor_params = [p for n, p in policy.named_parameters() if 'decoder.value_head' not in n]

    optimizer = optim.Adam([
        {'params': actor_params, 'lr': 3e-4},
        {'params': critic_params, 'lr': 1e-4}
    ])

    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            node_list = [env.nodes[i] for i in range(env.num_nodes)]
            node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
            ffp_mask = env.get_ffp_logits_mask(state)


            # Compute policy and value (no torch.no_grad so value has grad for value_loss)
            probs, _, value = policy(node_features, state.cur_node, state.soc, ffp_mask)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state = env.step(state, action.item())
            reward = env.compute_reward(state, action.item(), next_state)

            # Advantage: use detached value so gradients don't flow through the target
            value_det = value.detach()
            advantage = (torch.tensor(reward, dtype=torch.float32, device=device) - value_det).detach()

            policy_loss = -log_prob * advantage

            # Value loss: make sure shapes match (use 1-d tensors)
            value_target = torch.tensor([reward], dtype=torch.float32, device=device)
            value_est = value.view(1)
            value_loss = nn.functional.mse_loss(value_est, value_target)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            ep_reward += reward
            state = next_state

            if state.done or state.soc <= 0:
                break

        rewards_history.append(ep_reward)

        if (ep+1) % 50 == 0 or ep == episodes-1:
            print(f"Seed {seed} Ep {ep+1}/{episodes} | Reward {ep_reward:.2f}")

    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = os.path.join('checkpoints', f'linear_seed{seed}.pth')
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return rewards_history, ckpt_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--hidden-dim', type=int, default=64)
    args = parser.parse_args()

    train(args.seed, episodes=args.episodes, max_steps=args.max_steps, hidden_dim=args.hidden_dim)
"""
train_linear.py
对 baseline.PolicyNetworkLinear 进行 PPO 训练的轻量脚本。

用法示例：
python train_linear.py --seed 0 --episodes 200
"""

import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import EVRPTWEnv, NODES
from baseline import PolicyNetworkLinear
from train_ppo import GAMMA, LAMBDA, LR, HIDDEN_DIM, MAX_STEPS, ENTROPY_COEF, VALUE_COEF, CLIP_EPS, UPDATE_EPOCHS, MINI_BATCH_SIZE


def train(seed=0, num_episodes=200):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = EVRPTWEnv(NODES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNetworkLinear(env.num_nodes, HIDDEN_DIM).to(device)

    critic_params = list(policy.decoder.value_head.parameters())
    actor_params = [p for n, p in policy.named_parameters() if 'decoder.value_head' not in n]

    optimizer = optim.Adam([
        {'params': actor_params, 'lr': LR},
        {'params': critic_params, 'lr': 1e-4}
    ])

    # simple rollout buffer
    class Buffer:
        def __init__(self):
            self.states = []
            self.actions = []
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.dones = []

        def clear(self):
            self.__init__()

    buffer = Buffer()

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        for t in range(MAX_STEPS):
            node_list = [env.nodes[i] for i in range(env.num_nodes)]
            node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
            ffp_mask = env.get_ffp_logits_mask(state)

            with torch.no_grad():
                probs, _, value = policy(node_features, state.cur_node, state.soc, ffp_mask)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state = env.step(state, action.item())
            reward = env.compute_reward(state, action.item(), next_state)

            buffer.states.append((node_features.clone().detach(), state.cur_node, state.soc, ffp_mask.copy()))
            buffer.actions.append(action.cpu())
            buffer.log_probs.append(log_prob.cpu().detach())
            buffer.values.append(value.item())
            buffer.rewards.append(reward)
            buffer.dones.append(next_state.done)

            state = next_state
            ep_reward += reward

            if state.done or state.soc <= 0:
                break

        # GAE
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0
        if len(buffer.dones) > 0 and not buffer.dones[-1]:
            last = buffer.states[-1]
            last_nf = last[0]
            last_cn = last[1]
            last_sc = last[2]
            last_fm = last[3]
            if isinstance(last_fm, np.ndarray):
                last_fm_t = torch.from_numpy(last_fm).to(device)
            else:
                last_fm_t = last_fm.to(device)
            with torch.no_grad():
                _, _, nv = policy(last_nf.to(device), last_cn, last_sc, last_fm_t)
                next_value = float(nv.item())

        for r, v, done in zip(reversed(buffer.rewards), reversed(buffer.values), reversed(buffer.dones)):
            if done:
                next_value = 0.0
                gae = 0.0
            delta = r + GAMMA * next_value - v
            gae = delta + GAMMA * LAMBDA * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        if len(ret_t) > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

        # PPO update simplified (single epoch to keep fast)
        for _ in range(UPDATE_EPOCHS):
            nf = torch.stack([s[0] for s in buffer.states]).to(device)
            cn = torch.tensor([s[1] for s in buffer.states], device=device)
            sc = torch.tensor([s[2] for s in buffer.states], dtype=torch.float32, device=device)
            fm = torch.stack([
                torch.from_numpy(s[3]) if isinstance(s[3], np.ndarray) else s[3] for s in buffer.states
            ]).to(device)

            probs, _, values = policy(nf, cn, sc, fm)
            dist = torch.distributions.Categorical(probs)

            actions_tensor = torch.tensor([int(a.item()) for a in buffer.actions], dtype=torch.long, device=device)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()

            old_log_probs = torch.stack([lp for lp in buffer.log_probs]).to(device).detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, ret_t)

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

        buffer.clear()

        if (ep+1) % 50 == 0 or ep == num_episodes-1:
            print(f"Seed {seed} Ep {ep+1} | Reward {ep_reward:.2f}")

    ckpt = f"checkpoints/linear_seed{seed}.pth"
    torch.save(policy.state_dict(), ckpt)
    print(f"Saved {ckpt}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=200)
    args = p.parse_args()
    train(seed=args.seed, num_episodes=args.episodes)
