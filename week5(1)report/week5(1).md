# Week 5 Progress Report: Stabilizing RL Training for EVRPTW via Reward Reshaping

## 1. Executive Summary
This week builds upon the architectural foundation established in Week 4. The primary objective was to resolve the critical issue of **stagnant rewards and frozen loss values**. By refactoring the environment API, implementing dense reward shaping, and scaling the problem instance, we successfully transitioned the training loop from a "false dead" state to active learning. The Policy Network is now receiving valid gradient signals, evidenced by dynamic Loss convergence and fluctuating Average Rewards.

## 2. Analysis of Week 4 Bottlenecks
Despite a verified gradient flow in Week 4, the training metrics remained frozen (`Avg Reward: -0.22`). Root cause analysis identified three major bottlenecks:

1.  **Broken Environment API**: The `env.step()` method failed to return the `reward` value. Consequently, the training loop optimized based on null values, resulting in "blind training."
2.  **Sparse Reward Signal**: The naive distance penalty ($-\Delta d$) lacked sufficient magnitude. The Policy Network could not distinguish between sub-optimal and optimal actions, leading to vanishing gradients.
3.  **Insufficient Task Complexity**: The 4-node toy instance allowed episodes to terminate prematurely, capping the model's learning potential.

## 3. Key Improvements Implemented

### 3.1 Environment Interface Refactoring (Critical Fix)
We corrected the `EVRPTWEnv.step()` method to comply with standard RL interfaces (Gym-style), ensuring the reward calculated by `compute_reward()` is propagated back to the agent.

**Before:**
```python
# Only returned state
return s
```

**After:**
```python
# Calculates reward and returns (state, reward, done, info)
reward = self.compute_reward(state, action, s)
done = self._check_done(s)
return s, reward, done, {}
```

### 3.2 Dense Reward Shaping
To combat sparse rewards, we implemented a heuristic-driven reward function. This significantly increased the dynamic range of the feedback signal, allowing the network to differentiate between actions more effectively.

| Component | Logic / Formula | Weight | Purpose |
| :--- | :--- | :--- | :--- |
| **Travel Cost** | $-\text{distance}$ | -1.0 | Encourage shorter routes |
| **Time Window Violation** | Penalty if $\text{arrival} > \text{due\_time}$ | -10.0 | Enforce temporal constraints |
| **Service Completion** | Bonus for visiting new customers | +20.0 | Promote task fulfillment |
| **Return to Depot** | Bonus for completing the tour | +50.0 | Reward successful termination |
| **Low Battery Penalty** | Penalty if $SoC < 20\%$ | -5.0 | Encourage proactive charging |

### 3.3 Instance Scaling
The problem instance was scaled from 4 nodes to **6 nodes** (4 Customers, 1 Depot, 1 Charging Station). This increased the search space, forcing the Attention Decoder to learn complex dependencies rather than memorizing trivial paths.

## 4. Experimental Results & Analysis

### 4.1 Quantitative Metrics Comparison
The following tables illustrate the transformation in training dynamics after applying the fixes.

**Table 1: Training Dynamics Comparison (Week 4 vs. Week 5)**

| Metric | Week 4 Status (Pre-fix) | Week 5 Status (Post-fix) | Observation |
| :--- | :--- | :--- | :--- |
| **Loss Trend** | Static (~2000) | Dynamic (0.2 ~ 66.7) | Indicates active gradient descent |
| **Avg Reward** | Fixed (-0.22) | Fluctuating (15.17 ~ 32.76) | Validates reward signal efficacy |
| **Gradient Flow** | Vanishing | Healthy | Confirmed via backward pass |
| **Convergence** | Failed | In Progress | Model is exploring the state space |

**Table 2: Detailed Training Log Snippet (Week 5)**  
*(Data extracted from the latest training run)*

| Episode | Loss | Avg Reward | Notes |
| :--- | :--- | :--- | :--- |
| 20 | 66.7521 | 32.76 | Initial high variance as policy explores |
| 60 | 7.1945 | 28.45 | Loss begins to stabilize |
| 100 | 0.2045 | 32.76 | Policy finds high-reward local patterns |
| 140 | 3.2641 | 20.40 | Exploration increases variance |
| 180 | 1.8199 | 25.79 | Recovery towards higher rewards |
| 220 | 0.6188 | 27.87 | Convergence迹象 (Signs of convergence) |
| 260 | 3.7744 | 16.78 | Temporary dip due to exploration |
| 300 | 6.3012 | 17.07 | Stabilizing around a positive mean |

### 4.2 Analysis
1.  **Active Learning**: The shift from static to fluctuating metrics confirms that the Policy Network is no longer "blind." It is actively adjusting weights based on the new reward signals.
2.  **Reward-Value Correlation**: The Average Reward remains consistently positive (above 15), indicating the model prioritizes service completion and depot returns over random wandering.
3.  **Variance Management**: While the Loss shows healthy movement, the fluctuations suggest the Critic (Value Head) requires further tuning to reduce the Advantage estimate variance.

## 5. Conclusion & Next Steps

### 5.1 Conclusion
Week 5 successfully resolved the "training deadlock" encountered previously. The root cause was identified as a broken data pipeline between the environment and the agent. With the API fixed and a robust reward mechanism in place, the Actor-Critic architecture is now functioning as intended.