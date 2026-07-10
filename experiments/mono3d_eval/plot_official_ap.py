"""绘制 KITTI 官方风格 AP40 结果。

输入是 evaluate_kitti_official.py 生成的 official_kitti_eval.json。
图中分别展示 strict(3D IoU 0.7) 和 loose(3D IoU 0.5) 的 Easy/Moderate/Hard。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EVAL_DIR = Path(__file__).resolve().parent / "results" / "monocon_current_env_500" / "official_eval"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KITTI AP40 metrics from official_kitti_eval.json.")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def metric(metrics: dict[str, float], kind: str, difficulty: str, overlap: str) -> float:
    return float(metrics[f"KITTI/Car_{kind}_AP40_{difficulty}_{overlap}"])


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output) if args.output else eval_dir / "official_ap40_car.png"

    payload = json.loads((eval_dir / "official_kitti_eval.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    difficulties = ["easy", "moderate", "hard"]
    kinds = ["2D", "BEV", "3D"]
    colors = {"2D": "#4f73b8", "BEV": "#d87540", "3D": "#2f9c65"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, overlap in zip(axes, ["strict", "loose"]):
        x = np.arange(len(difficulties))
        width = 0.24
        for offset, kind in zip([-width, 0.0, width], kinds):
            values = [metric(metrics, kind, difficulty, overlap) for difficulty in difficulties]
            ax.bar(x + offset, values, width=width, label=f"{kind} AP40", color=colors[kind])
            for xi, value in zip(x + offset, values):
                ax.text(xi, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

        title = "strict: Car AP40@0.70/0.70/0.70" if overlap == "strict" else "loose: Car AP40@0.70/0.50/0.50"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Easy", "Moderate", "Hard"])
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="lower left")

    axes[0].set_ylabel("AP40")
    fig.suptitle(f"KITTI official-style evaluation ({payload['frame_count']} frames)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
