# EVRPTW PPO Experiments

包含：
- `encoder.py`, `decoder.py`, `env.py`, `train_ppo.py`, `eval_ppo.py`（原始文件）
- `baseline.py`：Linear encoder + 贪心基线
- `experiments.py`：训练/评估脚本，保存 CSV
- `plot_results.py`：生成图表

快速开始：

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 训练（示例）：

```bash
python experiments.py --method transformer --seeds 0
```

3. 评估并保存结果：

```bash
python experiments.py --method transformer --seeds 0
python experiments.py --method linear --seeds 0
python experiments.py --method greedy --seeds 0
```

4. 画图：

```bash
python plot_results.py
```
