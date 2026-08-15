# EVRPTW Experiment Report (Detailed)

**Date**: 2026-08-14

## Overview
This report summarizes a small-scale comparison between three policies on the Electric Vehicle Routing Problem with Time Windows (EVRPTW):
- **Transformer PPO** (checkpoint: [ppo_policy_transformer.pth](ppo_policy_transformer.pth))
- **Linear PPO** (policy implemented as `PolicyNetworkLinear`, checkpoints in `checkpoints/`)
- **Greedy baseline** (deterministic greedy policy)

Evaluation protocol used for these runs:
- Transformer and Linear policies: 5 deterministic evaluation runs each (saved in CSVs).
- Greedy baseline: single run (saved in CSV).

For reproducibility, see the result CSVs: [results/results_transformer.csv](results/results_transformer.csv), [results/results_linear.csv](results/results_linear.csv), [results/results_greedy.csv](results/results_greedy.csv).

---

## Numerical Summary

| Method | n_runs | Mean Total Reward | Std Total Reward | Mean Visited | Mean Final SoC | Mean Final Clock | Mean Steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 5 | 606.9724 | 0.0000 | 8 | 97.7639 | 58.0552 | 10 |
| Linear      | 5 | 609.8623 | 0.0000 | 8 | 79.7246 | 60.2754 | 9 |
| Greedy      | 1 | 512.6929 | - | 8 | 97.7639 | 62.6143 | 14 |

Notes:
- The Standard Deviation for the small evaluation sets is zero because the saved deterministic runs returned identical metrics across seeds (this indicates either deterministic policy behavior under the given seeds or identical evaluation trajectories). Increase the number of stochastic trials to estimate variance.

---

## Additional Analysis and Observations

- **Total reward vs. energy (SoC)**: Linear PPO achieves the highest mean total reward (609.86) while finishing with a lower mean final SoC (79.72) than Transformer (97.76). This implies the Linear policy uses more battery energy to accrue more reward—plausible if its routes are more time-efficient or accept higher travel costs to reduce waiting/penalties. Investigate reward terms to confirm whether time penalties or served-customer bonuses dominate the score.

- **Finish time (final_clock) and steps**: Transformer finishes earlier on average (final_clock ≈ 58.06) than Linear (≈ 60.28) and Greedy (≈ 62.61). However, Linear uses fewer recorded steps (9) than Transformer (10) and Greedy (14) while still achieving a higher reward. This mismatch between steps and final_clock suggests that the environment's `steps` counter measures discrete decision points (e.g., node visits) while `final_clock` measures accumulated time; Linear may choose longer travel legs that result in fewer decisions but longer raw time, or different charging/waiting behavior—inspect sampled trajectories to confirm.

- **Visited_count**: All three methods visit the same number of customers (8) in this scenario, indicating differences in reward stem from route quality and timing rather than coverage.

- **Determinism and evaluation protocol**: The present small-sample results show near-deterministic outputs for the policy checkpoints under the evaluation procedure used. For statistical validity you should:
  - Run at least 30 independent evaluations per method (different environment seeds) to estimate variance.
  - Report 95% confidence intervals for mean total reward (use bootstrap or t-based CI depending on sample size).
  - Use paired tests when comparing two policies on the same episode seeds (paired t-test or Wilcoxon signed-rank if non-normal).

- **Potential reward-function bias**: Because Linear achieves higher reward but lower final SoC, check whether the reward heavily favors service completion / time reduction over residual energy. If energy conservation is a design objective, include explicit energy penalties or multi-objective evaluation.

---

## Visualizations
The following figures were generated from the CSVs and are saved in `results/`.

- **Total reward distribution and boxplot**: [results/total_reward_boxplot.png](results/total_reward_boxplot.png)
- **Mean visited count per method**: [results/visited_count_bar.png](results/visited_count_bar.png)
- **Mean final SoC per method**: [results/final_soc_bar.png](results/final_soc_bar.png)
- **Visited count distribution**: [results/visited_count_dist.png](results/visited_count_dist.png)

Open the images from the `results/` folder or view them inline in this Markdown file if your viewer supports local images.

---

## Recommendations and Next Steps

1. **Increase evaluation runs**: Run N>=30 deterministic/stochastic evaluations per method and compute mean ± 95% CI for total reward and other metrics.
2. **Paired comparisons**: Evaluate each policy on the same set of environment seeds and compute paired statistical tests (paired t-test or Wilcoxon) to control for episode variance.
3. **Trajectory inspection**: Save sample trajectories (node sequence, timestamps, SoC over time) for qualitative comparison—plot 2–5 exemplar routes per policy to explain differences.
4. **Ablation on reward terms**: If reward is composite (service reward, time penalty, energy penalty), run ablation experiments to measure sensitivity.
5. **Hyperparameter search**: Run small grid search over learning rate, value loss coefficient, and hidden dimension for both Linear and Transformer policies.
6. **Report generation**: After expanded runs, regenerate summary tables and plots; include statistical test output and sample route visualizations in the final report.

---

## Appendix: files

- Results CSVs: [results/results_transformer.csv](results/results_transformer.csv), [results/results_linear.csv](results/results_linear.csv), [results/results_greedy.csv](results/results_greedy.csv)
- Checkpoints: [ppo_policy_transformer.pth](ppo_policy_transformer.pth), `checkpoints/` (linear checkpoints)
- Plots: [results/total_reward_boxplot.png](results/total_reward_boxplot.png), [results/visited_count_bar.png](results/visited_count_bar.png), [results/final_soc_bar.png](results/final_soc_bar.png), [results/visited_count_dist.png](results/visited_count_dist.png)

