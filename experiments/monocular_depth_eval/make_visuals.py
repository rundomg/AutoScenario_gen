"""Create visual summaries for KITTI monocular depth evaluation results.

Examples:
    python experiments/monocular_depth_eval/make_visuals.py
    python experiments/monocular_depth_eval/make_visuals.py --run results/runs/depth-anything-v2-hf/yolo-seg/500_masked
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = [
    SCRIPT_DIR / "results" / "runs" / "depth-anything-v2-hf" / "yolo-seg" / "500_masked",
    SCRIPT_DIR / "results" / "runs" / "metric3d" / "yolo-seg" / "500_masked",
    SCRIPT_DIR / "results" / "runs" / "unidepth" / "yolo-seg" / "500_masked",
]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "visuals" / "yolo-seg"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8-sig"))


def as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_depth_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if as_float(row.get("pred_depth_m")) is None:
            continue
        if as_float(row.get("label_depth_m")) is None:
            continue
        if as_float(row.get("abs_error_label_m")) is None:
            continue
        output.append(row)
    return output


def plot_pred_vs_gt(rows: list[dict[str, str]], output_path: Path, *, title: str) -> None:
    valid = valid_depth_rows(rows)
    gt = np.array([as_float(row["label_depth_m"]) for row in valid], dtype=np.float64)
    pred = np.array([as_float(row["pred_depth_m"]) for row in valid], dtype=np.float64)
    err = np.array([as_float(row["abs_error_label_m"]) for row in valid], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=160)
    scatter = ax.scatter(gt, pred, c=err, s=12, cmap="viridis", alpha=0.7, edgecolors="none")
    limit = float(max(np.nanmax(gt), np.nanmax(pred), 1.0))
    ax.plot([0, limit], [0, limit], color="#111111", linewidth=1.4, linestyle="--", label="ideal")
    ax.set_xlim(0, min(limit * 1.03, 120))
    ax.set_ylim(0, min(limit * 1.03, 120))
    ax.set_xlabel("KITTI label depth (m)")
    ax.set_ylabel("Predicted median depth (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("absolute error (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_error_bins(summary: dict, output_path: Path) -> None:
    bins = summary["label_distance_bins"]
    labels = [item["range"] for item in bins]
    mae = [float(item.get("mae_m", 0.0) or 0.0) for item in bins]
    med = [float(item.get("median_abs_error_m", 0.0) or 0.0) for item in bins]
    counts = [int(item.get("count", 0) or 0) for item in bins]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=160)
    ax.bar(x - width / 2, mae, width, label="MAE", color="#3d7ea6")
    ax.bar(x + width / 2, med, width, label="Median AE", color="#f08a4b")
    for index, count in enumerate(counts):
        top = max(mae[index], med[index])
        ax.text(index, top + 0.6, f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("absolute error (m)")
    ax.set_xlabel("KITTI label depth range")
    ax.set_title("Vehicle distance error by range")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_label(summary: dict, run_dir: Path) -> str:
    model = str(summary.get("model") or run_dir.parent.parent.name)
    if model == "depth-anything-v2-hf":
        return "Depth Anything V2"
    if model == "metric3d":
        variant = summary.get("metric3d_model")
        return f"Metric3D ({variant})" if variant else "Metric3D"
    if model == "unidepth":
        version = str(summary.get("unidepth_version") or "").upper()
        backbone = summary.get("unidepth_backbone")
        suffix = " ".join(value for value in (version, str(backbone or "")) if value)
        return f"UniDepth {suffix}".strip()
    return model


def plot_model_comparison(run_summaries: list[tuple[Path, dict]], output_path: Path) -> None:
    """Compare depth models while holding the target source fixed to YOLO masks."""

    metrics = [
        ("MAE (m)", "mae_m"),
        ("RMSE (m)", "rmse_m"),
        ("AbsRel", "absrel"),
    ]
    colors = ["#4169a8", "#d96b43", "#4f9d69", "#8a5fbf", "#607d3b"]
    series = [
        (
            f"{run_label(summary, run_dir)} ({summary.get('selected_frames', '?')} frames, "
            f"n={summary.get('label_depth_error', {}).get('count', '?')})",
            [float(summary["label_depth_error"].get(key, 0.0) or 0.0) for _, key in metrics],
            colors[index % len(colors)],
        )
        for index, (run_dir, summary) in enumerate(run_summaries)
    ]
    x = np.arange(len(metrics))
    width = 0.78 / len(series)

    fig, ax = plt.subplots(figsize=(8.7, 5.2), dpi=160)
    max_value = max(value for _name, values, _color in series for value in values)
    bar_groups = []
    start = -width * (len(series) - 1) / 2
    for index, (name, values, color) in enumerate(series):
        bars = ax.bar(x + start + index * width, values, width, label=name, color=color)
        bar_groups.append(bars)
    for bars in bar_groups:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max_value * 0.025,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, [name for name, _ in metrics])
    ax.set_title("Depth model metrics on YOLO instance-mask targets")
    ax.set_ylabel("metric value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def overlay_montage(
    overlay_dir: Path,
    output_path: Path,
    *,
    cols: int = 3,
    rows: int = 3,
    cell_width: int = 520,
    title: str = "Depth overlay examples",
) -> None:
    paths = sorted(overlay_dir.glob("*.jpg"))[: cols * rows]
    if not paths:
        raise FileNotFoundError(f"No overlay images found in {overlay_dir}")
    cell_height = int(cell_width * 0.38)
    title_height = 54
    pad = 12
    canvas = np.full(
        (title_height + rows * cell_height + (rows + 1) * pad, cols * cell_width + (cols + 1) * pad, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(canvas, title, (pad, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (25, 25, 25), 2)

    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = resize_and_crop(image, cell_width, cell_height)
        row = index // cols
        col = index % cols
        y = title_height + pad + row * (cell_height + pad)
        x = pad + col * (cell_width + pad)
        canvas[y : y + cell_height, x : x + cell_width] = image
        cv2.putText(canvas, path.stem, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(canvas, path.stem, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), canvas)


def resize_and_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (int(round(source_width * scale)), int(round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y0 = max(0, (resized.shape[0] - height) // 2)
    x0 = max(0, (resized.shape[1] - width) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual summaries for monocular depth results.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Result directory for one YOLO instance-mask depth-model run. Repeat for comparisons.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated figures.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = [Path(value) for value in args.run] if args.run else DEFAULT_RUNS
    run_summaries = [(run_dir, load_summary(run_dir)) for run_dir in run_dirs]
    primary_run = run_dirs[0]
    primary_summary = run_summaries[0][1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows = load_rows(primary_run / "metrics.csv")

    outputs = {
        "overlay_montage": output_dir / "overlay_montage_yolo_instance_masks.jpg",
        "pred_vs_gt": output_dir / "pred_vs_gt_yolo_instance_masks.png",
        "error_bins": output_dir / "error_by_distance_bin.png",
        "model_compare": output_dir / "depth_model_metric_comparison.png",
    }
    overlay_montage(
        primary_run / "overlays",
        outputs["overlay_montage"],
        title="YOLO instance masks: final sampling regions",
    )
    plot_pred_vs_gt(
        primary_rows,
        outputs["pred_vs_gt"],
        title=f"{run_label(primary_summary, primary_run)}: YOLO-mask distance vs KITTI label",
    )
    plot_error_bins(primary_summary, outputs["error_bins"])
    plot_model_comparison(run_summaries, outputs["model_compare"])

    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
