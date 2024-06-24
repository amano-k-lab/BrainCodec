import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({"font.family": "serif"})

parser = argparse.ArgumentParser()
parser.add_argument("--paths", type=str, nargs="+", help="List of paths of npy", required=True)
parser.add_argument("--labels", type=str, nargs="+", help="List of labels", required=True)
parser.add_argument("--output-path", type=str, help="output path", required=True)
args = parser.parse_args()

n_q = 8
bins = 1

paths = []
for p, l in zip(args.paths, args.labels):
    paths.append([Path(p), l])

plt.figure(figsize=(8, 6))
for p, label in paths:
    data = np.load(p)
    plt.plot(data, linewidth=2, linestyle="-", marker="o", markersize=6, label=label)

# 軸の設定
plt.xlabel("Codebook number", fontsize=30)
plt.ylabel("Sinkhorn distance", fontsize=30)
plt.xticks(range(0, n_q, bins), range(0, n_q, bins), fontsize=20)
plt.yticks(fontsize=20)
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))  # 補助目盛りを設定

# 凡例の設定
plt.legend(loc="lower right", fontsize=20)

plt.tight_layout()  # レイアウトを調整
plt.savefig(args.output_path, dpi=300)  # 解像度を設定して保存
plt.show()
