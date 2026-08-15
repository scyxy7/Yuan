"""
plot_results.py
- 从 `results/results_*.csv` 读取结果
- 生成训练/评估对比图：条形图/箱线图

用法：
python plot_results.py
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("results")

def plot_summary():
    files = list(RESULTS_DIR.glob('results_*.csv'))
    if len(files) == 0:
        print('No results CSV found in results/ — run experiments first.')
        return

    df_list = []
    for f in files:
        method = f.stem.replace('results_','')
        df = pd.read_csv(f)
        df['method'] = method
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)

    plt.figure(figsize=(8,5))
    sns.boxplot(x='method', y='total_reward', data=df_all)
    plt.title('Total Reward by Method')
    plt.savefig('results/total_reward_boxplot.png', dpi=200)
    print('Saved results/total_reward_boxplot.png')

    plt.figure(figsize=(8,5))
    sns.barplot(x='method', y='visited_count', data=df_all, estimator='mean', ci=95)
    plt.title('Visited Count Mean by Method')
    plt.savefig('results/visited_count_bar.png', dpi=200)
    print('Saved results/visited_count_bar.png')

    plt.figure(figsize=(8,5))
    sns.barplot(x='method', y='final_soc', data=df_all, estimator='mean', ci=95)
    plt.title('Final SoC Mean by Method')
    plt.savefig('results/final_soc_bar.png', dpi=200)
    print('Saved results/final_soc_bar.png')

    plt.figure(figsize=(8,5))
    sns.countplot(x='method', hue='visited_count', data=df_all)
    plt.title('Visited Count Distribution')
    plt.savefig('results/visited_count_dist.png', dpi=200)
    print('Saved results/visited_count_dist.png')


if __name__ == '__main__':
    plot_summary()
