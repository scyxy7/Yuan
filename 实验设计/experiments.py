"""
experiments.py
统一的训练/评估脚本：
- action: train/eval/eval_greedy
- method: transformer/linear/greedy

用法：
python experiments.py --action eval --method transformer --model ppo_policy_transformer.pth
python experiments.py --action eval_greedy
"""

import argparse
import os
import csv
import random
from pathlib import Path

import numpy as np
import torch

from env import EVRPTWEnv, NODES
from train_ppo import PolicyNetwork as PolicyNetworkTransformer, NUM_EPISODES, MAX_STEPS, HIDDEN_DIM
from baseline import PolicyNetworkLinear, greedy_baseline


RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def evaluate_policy(policy, env, device, max_steps=30):
    policy.eval()
    state = env.reset()
    total_reward = 0.0

    for t in range(max_steps):
        node_list = [env.nodes[i] for i in range(env.num_nodes)]
        node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
        ffp_mask = env.get_ffp_logits_mask(state)

        with torch.no_grad():
            probs, logits, value = policy(node_features, state.cur_node, state.soc, ffp_mask)

        # mask
        mask_np = np.isneginf(ffp_mask)
        mask_tensor = torch.from_numpy(mask_np).to(device)
        probs_masked = probs.clone()
        if probs_masked.dim() == 2 and mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(probs_masked)
        probs_masked[mask_tensor] = 0.0

        if probs_masked.sum() == 0:
            action = state.cur_node
        else:
            if probs_masked.dim() == 2:
                action = torch.argmax(probs_masked, dim=-1).item()
            else:
                action = torch.argmax(probs_masked, dim=-1).item()

        next_state = env.step(state, action)
        reward = env.compute_reward(state, action, next_state)

        total_reward += reward
        state = next_state

        if state.done or state.soc <= 0:
            break

    return {
        "total_reward": total_reward,
        "visited": sorted(list(state.visited)),
        "final_soc": state.soc,
        "final_clock": state.clock,
        "steps": t+1
    }


def run_evaluation(method, checkpoint_path=None, n_runs=20):
    env = EVRPTWEnv(NODES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []

    for run in range(n_runs):
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)

        if method == 'transformer':
            policy = PolicyNetworkTransformer(env.num_nodes, HIDDEN_DIM).to(device)
        elif method == 'linear':
            policy = PolicyNetworkLinear(env.num_nodes, HIDDEN_DIM).to(device)
        elif method == 'greedy':
            res = greedy_baseline(env, max_steps=MAX_STEPS)
            records.append({
                'run': run,
                'total_reward': res['total_reward'],
                'visited_count': len(res['visited']),
                'final_soc': res['final_soc'],
                'final_clock': res['final_clock'],
                'steps': res['steps']
            })
            continue
        else:
            raise ValueError('Unknown method')

        # load checkpoint
        if checkpoint_path is None:
            # try default saved file
            default = Path('ppo_policy_transformer.pth') if method == 'transformer' else None
            if default and default.exists():
                checkpoint_path = default
            else:
                print(f"No checkpoint provided for {method}, skipping run {run}.")
                continue

        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

        res = evaluate_policy(policy, env, device, max_steps=MAX_STEPS)
        records.append({
            'run': run,
            'total_reward': res['total_reward'],
            'visited_count': len(res['visited']),
            'final_soc': res['final_soc'],
            'final_clock': res['final_clock'],
            'steps': res['steps']
        })

    # save CSV
    out = Path('results') / f"results_{method}.csv"
    out.parent.mkdir(exist_ok=True)
    if len(records) > 0:
        keys = records[0].keys()
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"Saved evaluation to {out}")
    else:
        print("No records to save.")

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['train','eval','eval_greedy'], required=True)
    parser.add_argument('--method', choices=['transformer','linear','greedy'], default='transformer')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--n_runs', type=int, default=20)
    args = parser.parse_args()

    if args.action == 'eval_greedy':
        run_evaluation('greedy', n_runs=args.n_runs)
    elif args.action == 'eval':
        run_evaluation(args.method, checkpoint_path=args.checkpoint, n_runs=args.n_runs)
    else:
        print('Training via experiments.py is not yet implemented in this unified script. Use train_ppo.py to train.')


if __name__ == '__main__':
    main()
"""
experiments.py
- 支持三种方法：Transformer PPO（现有）、Linear PPO（使用 LinearEncoder）、Greedy Baseline
- 支持多次随机种子训练/评估并保存 CSV 日志与模型检查点
- 如果找不到按-seed 保存的 checkpoint，会尝试回退到 'ppo_policy_transformer.pth'
"""

import os
import csv
import random
import argparse
from pathlib import Path

import numpy as np
import torch

from env import EVRPTWEnv, NODES
from train_ppo import PolicyNetwork as PolicyNetworkTransformer, HIDDEN_DIM, MAX_STEPS
from baseline import PolicyNetworkLinear, greedy_baseline


RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def evaluate_policy(policy, env, device, max_steps=30):
    policy.eval()
    state = env.reset()
    total_reward = 0.0

    for t in range(max_steps):
        node_list = [env.nodes[i] for i in range(env.num_nodes)]
        node_features = torch.tensor(node_list, dtype=torch.float32, device=device)
        ffp_mask = env.get_ffp_logits_mask(state)

        with torch.no_grad():
            probs, logits, value = policy(node_features, state.cur_node, state.soc, ffp_mask)

        # mask
        mask_np = np.isneginf(ffp_mask)
        mask_tensor = torch.from_numpy(mask_np).to(device)
        probs_masked = probs.clone()
        if probs_masked.dim() == 2 and mask_tensor.dim() == 1:
            mask_tensor = mask_tensor.unsqueeze(0).expand_as(probs_masked)
        probs_masked[mask_tensor] = 0.0

        if probs_masked.sum() == 0:
            action = state.cur_node
        else:
            if probs_masked.dim() == 2:
                action = torch.argmax(probs_masked, dim=-1).item()
            else:
                action = torch.argmax(probs_masked, dim=-1).item()

        next_state = env.step(state, action)
        reward = env.compute_reward(state, action, next_state)

        total_reward += reward
        state = next_state

        if state.done or state.soc <= 0:
            break

    return {
        "route": None,
        "total_reward": total_reward,
        "visited": sorted(list(state.visited)),
        "final_soc": state.soc,
        "final_clock": state.clock,
        "steps": None
    }


def run_single(method, seed, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = EVRPTWEnv(NODES)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if method == 'transformer':
        policy = PolicyNetworkTransformer(env.num_nodes, args.hidden_dim).to(device)
        # try seed-specific checkpoint first
        ckpt_seed = CHECKPOINT_DIR / f"transformer_seed{seed}.pth"
        ckpt_fallback = Path("ppo_policy_transformer.pth")

        if ckpt_seed.exists():
            policy.load_state_dict(torch.load(ckpt_seed, map_location=device))
        elif ckpt_fallback.exists():
            print(f"Info: using fallback checkpoint {ckpt_fallback}")
            policy.load_state_dict(torch.load(ckpt_fallback, map_location=device))
        else:
            print(f"Warning: no checkpoint found for transformer (tried {ckpt_seed} and {ckpt_fallback}). Skipping.")
            return None

        res = evaluate_policy(policy, env, device, max_steps=args.max_steps)

    elif method == 'linear':
        policy = PolicyNetworkLinear(env.num_nodes, args.hidden_dim).to(device)
        ckpt_seed = CHECKPOINT_DIR / f"linear_seed{seed}.pth"
        if ckpt_seed.exists():
            policy.load_state_dict(torch.load(ckpt_seed, map_location=device))
            res = evaluate_policy(policy, env, device, max_steps=args.max_steps)
        else:
            print(f"Warning: checkpoint {ckpt_seed} not found for linear. Skipping.")
            return None

    elif method == 'greedy':
        res = greedy_baseline(env, max_steps=args.max_steps)

    else:
        raise ValueError("Unknown method")

    # 保存 CSV 行
    csv_path = RESULTS_DIR / f"results_{method}.csv"
    header = ["seed", "total_reward", "visited_count", "final_soc", "final_clock"]
    row = [seed, res['total_reward'], len(res['visited']), res['final_soc'], res['final_clock']]

    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['transformer', 'linear', 'greedy', 'all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--max-steps', type=int, default=30)
    args = parser.parse_args()

    methods = [args.method] if args.method != 'all' else ['transformer', 'linear', 'greedy']

    for m in methods:
        for s in args.seeds:
            print(f"Running {m} seed={s} ...")
            res = run_single(m, s, args)
            if res is not None:
                print(f"  -> Reward: {res['total_reward']:.2f} | Visited: {len(res['visited'])}")


if __name__ == '__main__':
    main()
