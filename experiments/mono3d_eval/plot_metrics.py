"""MonoCon/KITTI 3D 检测评估图表。

这个脚本只消费评估阶段已经写出的 CSV，不重新跑模型。
目的是把数字指标变成更直观的图，方便快速判断距离、朝向和置信度的误差形态。
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
import pandas as pd


DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results" / "monocon_current_env_500"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mono 3D detection evaluation charts from CSV files.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR), help="Directory containing matches.csv and predictions.csv.")
    parser.add_argument("--output-dir", default=None, help="Optional chart output directory. Defaults to result-dir/visuals/stats.")
    parser.add_argument("--distance-bins", default="0,10,20,30,40,50,60,80,120", help="Comma-separated GT distance bin edges in meters.")
    return parser.parse_args(argv)


def load_inputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取预测和匹配结果；空文件也返回带列名的 DataFrame。"""

    matches_path = result_dir / "matches.csv"
    predictions_path = result_dir / "predictions.csv"
    if not matches_path.exists():
        raise FileNotFoundError(f"matches.csv not found: {matches_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.csv not found: {predictions_path}")

    matches = pd.read_csv(matches_path)
    predictions = pd.read_csv(predictions_path)
    return matches, predictions


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_bins(raw: str) -> list[float]:
    bins = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(bins) < 2:
        raise ValueError("Need at least two distance bin edges.")
    return bins


def plot_histogram(values: pd.Series, title: str, xlabel: str, output_path: Path, bins: int = 30) -> None:
    """画单变量直方图，并在图中标出均值和中位数。"""

    clean = pd.to_numeric(values, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    if clean.empty:
        ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(clean, bins=bins, color="#4f73b8", alpha=0.85, edgecolor="white")
        ax.axvline(clean.mean(), color="#d4553f", linewidth=2, label=f"mean={clean.mean():.2f}")
        ax.axvline(clean.median(), color="#2f9c65", linewidth=2, label=f"median={clean.median():.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_distance_scatter(matches: pd.DataFrame, output_path: Path) -> None:
    """画预测距离 vs 真值距离，越贴近 y=x 说明越准。"""

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if matches.empty:
        ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
    else:
        x = matches["label_distance_m"].astype(float)
        y = matches["pred_distance_m"].astype(float)
        color = matches["abs_distance_error_m"].astype(float)
        scatter = ax.scatter(x, y, c=color, s=18, cmap="viridis", alpha=0.8)
        limit = max(float(x.max()), float(y.max()), 1.0)
        ax.plot([0, limit], [0, limit], color="#333333", linewidth=1.5, linestyle="--")
        fig.colorbar(scatter, ax=ax, label="abs distance error (m)")
        ax.set_xlim(0, limit * 1.05)
        ax.set_ylim(0, limit * 1.05)
    ax.set_title("Predicted distance vs KITTI label distance")
    ax.set_xlabel("KITTI label distance (m)")
    ax.set_ylabel("MonoCon predicted distance (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_distance_bin_metrics(matches: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """按真值距离分段统计误差，便于观察远车是否明显变差。"""

    if matches.empty:
        return pd.DataFrame(
            columns=[
                "distance_bin",
                "count",
                "mae_distance_m",
                "median_distance_error_m",
                "rmse_distance_m",
                "mae_depth_m",
                "mae_yaw_deg",
            ]
        )

    work = matches.copy()
    work["distance_bin"] = pd.cut(work["label_distance_m"], bins=bins, right=False, include_lowest=True)
    rows = []
    for bin_label, group in work.groupby("distance_bin", observed=False):
        if group.empty:
            continue
        distance_errors = group["abs_distance_error_m"].astype(float)
        rows.append(
            {
                "distance_bin": str(bin_label),
                "count": int(len(group)),
                "mae_distance_m": float(distance_errors.mean()),
                "median_distance_error_m": float(distance_errors.median()),
                "rmse_distance_m": float(np.sqrt(np.mean(np.square(distance_errors)))),
                "mae_depth_m": float(group["abs_depth_error_m"].astype(float).mean()),
                "mae_yaw_deg": float(group["abs_yaw_error_deg"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_distance_bins(bin_metrics: pd.DataFrame, output_path: Path) -> None:
    """画不同距离段的 MAE/RMSE 和样本数。"""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if bin_metrics.empty:
        ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
    else:
        x = np.arange(len(bin_metrics))
        ax.bar(x - 0.18, bin_metrics["mae_distance_m"], width=0.36, label="MAE distance (m)", color="#4f73b8")
        ax.bar(x + 0.18, bin_metrics["rmse_distance_m"], width=0.36, label="RMSE distance (m)", color="#d87540")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_metrics["distance_bin"], rotation=35, ha="right")
        ax.set_ylabel("error (m)")
        ax.legend(loc="upper left")

        ax_count = ax.twinx()
        ax_count.plot(x, bin_metrics["count"], color="#2f9c65", marker="o", linewidth=2, label="matched count")
        ax_count.set_ylabel("matched count")
        ax_count.legend(loc="upper right")
    ax.set_title("Distance error by KITTI label distance bin")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_score_error(matches: pd.DataFrame, output_path: Path) -> None:
    """看置信度和距离误差的关系，用来判断低分预测是否更不可靠。"""

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if matches.empty:
        ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(matches["score"], matches["abs_distance_error_m"], s=18, alpha=0.75, color="#6b5fb5")
    ax.set_title("Score vs absolute distance error")
    ax.set_xlabel("MonoCon score")
    ax.set_ylabel("abs distance error (m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(matches: pd.DataFrame, predictions: pd.DataFrame, bin_metrics: pd.DataFrame, output_path: Path) -> None:
    """写一个机器可读的小报告，方便主程序或后续 notebook 复用。"""

    report = {
        "prediction_count": int(len(predictions)),
        "matched_count": int(len(matches)),
        "mean_abs_distance_error_m": float(matches["abs_distance_error_m"].mean()) if not matches.empty else None,
        "median_abs_distance_error_m": float(matches["abs_distance_error_m"].median()) if not matches.empty else None,
        "mean_abs_depth_error_m": float(matches["abs_depth_error_m"].mean()) if not matches.empty else None,
        "mean_abs_yaw_error_deg": float(matches["abs_yaw_error_deg"].mean()) if not matches.empty else None,
        "distance_bins": bin_metrics.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / "visuals" / "stats"
    ensure_output_dir(output_dir)

    matches, predictions = load_inputs(result_dir)
    bins = parse_bins(args.distance_bins)
    bin_metrics = build_distance_bin_metrics(matches, bins)
    bin_metrics.to_csv(output_dir / "distance_bin_metrics.csv", index=False)

    plot_histogram(matches.get("abs_distance_error_m", pd.Series(dtype=float)), "Absolute distance error", "abs distance error (m)", output_dir / "distance_error_hist.png")
    plot_histogram(matches.get("abs_depth_error_m", pd.Series(dtype=float)), "Absolute depth error", "abs depth error (m)", output_dir / "depth_error_hist.png")
    plot_histogram(matches.get("abs_yaw_error_deg", pd.Series(dtype=float)), "Absolute yaw error", "abs yaw error (deg)", output_dir / "yaw_error_hist.png")
    plot_distance_scatter(matches, output_dir / "pred_vs_gt_distance_scatter.png")
    plot_distance_bins(bin_metrics, output_dir / "error_by_distance_bin.png")
    plot_score_error(matches, output_dir / "score_vs_distance_error.png")
    write_report(matches, predictions, bin_metrics, output_dir / "metrics_report.json")

    print(f"wrote charts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
