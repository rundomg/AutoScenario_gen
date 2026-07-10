"""Run MonoCon calibration ablations on KITTI.

The script keeps inference calibration and evaluation truth separate:
generated fake P2 files are used only by MonoCon during prediction decoding,
while KITTI `label_2` remains the ground truth for metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .calibration_modes import CALIBRATION_MODES, parse_calibration_mode
    from .monocon_paths import DEFAULT_KITTI_ROOT
except ImportError:
    from calibration_modes import CALIBRATION_MODES, parse_calibration_mode
    from monocon_paths import DEFAULT_KITTI_ROOT


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "monocon_pretrained" / "best.pth"
DEFAULT_CONFIG = EXPERIMENT_DIR / "checkpoints" / "monocon_pretrained" / "config.yaml"
DEFAULT_FRAME_LIST = EXPERIMENT_DIR / "ImageSets" / "training_eval_500.txt"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "results" / "calibration_ablation"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run true/fake calibration ablations for MonoCon on KITTI.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subprocesses.")
    parser.add_argument("--checkpoint-file", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--frame-list", default=str(DEFAULT_FRAME_LIST))
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--modes", default="true_k,fake_fov90,fake_ratio058")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--model-score-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max-objs", type=int, default=30)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--save-visuals", type=int, default=12)
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing prediction/evaluation files.")
    parser.add_argument("--skip-official", action="store_true", help="Skip KITTI official-style AP40 evaluation.")
    return parser.parse_args(argv)


def selected_modes(raw_modes: str) -> list[str]:
    modes = [item.strip() for item in raw_modes.split(",") if item.strip()]
    for mode in modes:
        parse_calibration_mode(mode)
    return modes


def frame_count_label(frame_list: Path, max_frames: int | None) -> int:
    frame_ids = [line.strip() for line in frame_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return min(len(frame_ids), max_frames) if max_frames is not None else len(frame_ids)


def run_dir_for(output_root: Path, mode: str, frame_count: int) -> Path:
    return output_root / f"{mode}_{frame_count}"


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def run_one_mode(args: argparse.Namespace, mode: str, run_dir: Path) -> None:
    prediction_dir = run_dir / "prediction_txt"
    generated_calib_dir = run_dir / "generated_calib"
    export_script = EXPERIMENT_DIR / "export_monocon_predictions.py"
    plot_metrics_script = EXPERIMENT_DIR / "plot_metrics.py"
    official_eval_script = EXPERIMENT_DIR / "evaluate_kitti_official.py"
    plot_official_script = EXPERIMENT_DIR / "plot_official_ap.py"

    if not args.skip_inference:
        run_command(
            [
                args.python,
                str(export_script),
                "--checkpoint-file",
                args.checkpoint_file,
                "--config-file",
                args.config_file,
                "--kitti-root",
                args.kitti_root,
                "--split",
                "training",
                "--frame-list",
                args.frame_list,
                "--max-frames",
                str(args.max_frames),
                "--prediction-dir",
                str(prediction_dir),
                "--eval-output-dir",
                str(run_dir),
                "--calibration-mode",
                mode,
                "--generated-calib-dir",
                str(generated_calib_dir),
                "--device",
                args.device,
                "--gpu-id",
                str(args.gpu_id),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--score-threshold",
                str(args.score_threshold),
                "--model-score-threshold",
                str(args.model_score_threshold),
                "--topk",
                str(args.topk),
                "--max-objs",
                str(args.max_objs),
                "--match-iou",
                str(args.match_iou),
                "--save-visuals",
                str(args.save_visuals),
            ]
        )

    run_command([args.python, str(plot_metrics_script), "--result-dir", str(run_dir)])

    if not args.skip_official:
        official_dir = run_dir / "official_eval"
        run_command(
            [
                args.python,
                str(official_eval_script),
                "--prediction-dir",
                str(prediction_dir),
                "--kitti-root",
                args.kitti_root,
                "--split",
                "training",
                "--frame-list",
                args.frame_list,
                "--max-frames",
                str(args.max_frames),
                "--output-dir",
                str(official_dir),
                "--classes",
                "Car",
                "--eval-types",
                "bbox,bev,3d",
            ]
        )
        run_command([args.python, str(plot_official_script), "--eval-dir", str(official_dir)])


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_or_none(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def build_comparison_rows(output_root: Path, modes: list[str], frame_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in modes:
        run_dir = run_dir_for(output_root, mode, frame_count)
        summary = read_json(run_dir / "summary.json")
        official_path = run_dir / "official_eval" / "official_kitti_eval.json"
        official = read_json(official_path) if official_path.exists() else {"metrics": {}}
        metrics = official.get("metrics", {})
        rows.append(
            {
                "run": run_dir.name,
                "calibration_mode": mode,
                "calibration_description": CALIBRATION_MODES[mode].description,
                "frame_count": summary.get("frame_count"),
                "prediction_count": summary.get("prediction_count"),
                "matched_count": summary.get("matched_count"),
                "mean_abs_depth_error_m": summary.get("mean_abs_depth_error_m"),
                "mean_abs_distance_error_m": summary.get("mean_abs_distance_error_m"),
                "mean_abs_yaw_error_deg": summary.get("mean_abs_yaw_error_deg"),
                "ap40_2d_moderate_strict": metric_or_none(metrics, "KITTI/Car_2D_AP40_moderate_strict"),
                "ap40_bev_moderate_strict": metric_or_none(metrics, "KITTI/Car_BEV_AP40_moderate_strict"),
                "ap40_3d_moderate_strict": metric_or_none(metrics, "KITTI/Car_3D_AP40_moderate_strict"),
                "ap40_bev_moderate_loose": metric_or_none(metrics, "KITTI/Car_BEV_AP40_moderate_loose"),
                "ap40_3d_moderate_loose": metric_or_none(metrics, "KITTI/Car_3D_AP40_moderate_loose"),
            }
        )
    return rows


def add_relative_changes(rows: list[dict[str, object]]) -> None:
    true_row = next((row for row in rows if row["calibration_mode"] == "true_k"), None)
    if true_row is None:
        return

    lower_is_better = [
        "mean_abs_depth_error_m",
        "mean_abs_distance_error_m",
        "mean_abs_yaw_error_deg",
    ]
    higher_is_better = [
        "ap40_2d_moderate_strict",
        "ap40_bev_moderate_strict",
        "ap40_3d_moderate_strict",
        "ap40_bev_moderate_loose",
        "ap40_3d_moderate_loose",
    ]
    for row in rows:
        for key in lower_is_better:
            row[f"{key}_increase_pct_vs_true"] = relative_pct(row.get(key), true_row.get(key), lower_is_better=True)
        for key in higher_is_better:
            row[f"{key}_drop_pct_vs_true"] = relative_pct(row.get(key), true_row.get(key), lower_is_better=False)


def relative_pct(value: object, baseline: object, *, lower_is_better: bool) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    value_f = float(value)
    baseline_f = float(baseline)
    if lower_is_better:
        return (value_f - baseline_f) / baseline_f * 100.0
    return (baseline_f - value_f) / baseline_f * 100.0


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_array(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) if row.get(key) is not None else np.nan for row in rows], dtype=np.float64)


def plot_grouped(rows: list[dict[str, object]], output_dir: Path) -> None:
    names = [str(row["calibration_mode"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.26
    series = [
        ("Depth MAE (m)", "mean_abs_depth_error_m", "#4f73b8", -width),
        ("Distance MAE (m)", "mean_abs_distance_error_m", "#d87540", 0.0),
        ("Yaw MAE (deg)", "mean_abs_yaw_error_deg", "#2f9c65", width),
    ]
    for label, key, color, offset in series:
        values = value_array(rows, key)
        ax.bar(x + offset, values, width=width, label=label, color=color)
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xi, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Lightweight matched error by calibration mode")
    ax.set_ylabel("metric value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lightweight_error_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    series = [
        ("2D AP40 moderate strict", "ap40_2d_moderate_strict", "#4f73b8", -width),
        ("BEV AP40 moderate strict", "ap40_bev_moderate_strict", "#d87540", 0.0),
        ("3D AP40 moderate strict", "ap40_3d_moderate_strict", "#2f9c65", width),
    ]
    for label, key, color, offset in series:
        values = value_array(rows, key)
        ax.bar(x + offset, values, width=width, label=label, color=color)
        for xi, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xi, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_title("KITTI official-style AP40 by calibration mode")
    ax.set_ylabel("AP40")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "official_ap40_comparison.png", dpi=160)
    plt.close(fig)

    fake_rows = [row for row in rows if row["calibration_mode"] != "true_k"]
    if fake_rows:
        fake_names = [str(row["calibration_mode"]) for row in fake_rows]
        x_fake = np.arange(len(fake_rows))
        fig, ax = plt.subplots(figsize=(10, 5.2))
        width = 0.22
        series = [
            ("Distance MAE increase %", "mean_abs_distance_error_m_increase_pct_vs_true", "#d87540", -1.5 * width),
            ("Depth MAE increase %", "mean_abs_depth_error_m_increase_pct_vs_true", "#4f73b8", -0.5 * width),
            ("BEV AP drop %", "ap40_bev_moderate_strict_drop_pct_vs_true", "#8b5fbf", 0.5 * width),
            ("3D AP drop %", "ap40_3d_moderate_strict_drop_pct_vs_true", "#2f9c65", 1.5 * width),
        ]
        for label, key, color, offset in series:
            values = value_array(fake_rows, key)
            ax.bar(x_fake + offset, values, width=width, label=label, color=color)
            for xi, value in zip(x_fake + offset, values):
                if np.isfinite(value):
                    ax.text(xi, value, f"{value:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(x_fake)
        ax.set_xticklabels(fake_names)
        ax.set_title("Relative degradation vs true KITTI calibration")
        ax.set_ylabel("change vs true calibration (%)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "relative_degradation_vs_true.png", dpi=160)
        plt.close(fig)


def write_comparison(output_root: Path, modes: list[str], frame_count: int) -> list[dict[str, object]]:
    rows = build_comparison_rows(output_root, modes, frame_count)
    add_relative_changes(rows)
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "calibration_ablation_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(rows, comparison_dir / "calibration_ablation_summary.csv")
    plot_grouped(rows, comparison_dir)
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    modes = selected_modes(args.modes)
    frame_count = frame_count_label(Path(args.frame_list), args.max_frames)

    for mode in modes:
        run_dir = run_dir_for(output_root, mode, frame_count)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_one_mode(args, mode, run_dir)

    rows = write_comparison(output_root, modes, frame_count)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote calibration ablation comparison to {output_root / 'comparison'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
