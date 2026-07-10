"""解析、评估并可视化 MonoDETR KITTI 格式预测结果。

该脚本不运行 MonoDETR 模型本身；它消费 MonoDETR 生成的 per-frame txt
结果，并输出 JSON/CSV、轻量误差统计和可视化图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

try:
    from .kitti_3d import (
        Kitti3DObject,
        angle_error_rad,
        finite_mean,
        match_predictions_to_labels,
        object_to_record,
        read_kitti_objects,
    )
    from .visualize import draw_bev, draw_frame_overlay
except ImportError:  # 允许直接执行脚本
    from kitti_3d import Kitti3DObject, angle_error_rad, finite_mean, match_predictions_to_labels, object_to_record, read_kitti_objects
    from visualize import draw_bev, draw_frame_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "KITTI" / "kitti"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "monodetr_kitti_smoke"


def discover_frame_ids(prediction_dir: Path, frame_list: Path | None, max_frames: int | None) -> list[str]:
    if frame_list is not None:
        frame_ids = [line.strip() for line in frame_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        frame_ids = sorted(path.stem for path in prediction_dir.glob("*.txt"))
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


def filter_by_score(objects: list[Kitti3DObject], score_threshold: float) -> list[Kitti3DObject]:
    return [obj for obj in objects if obj.score is None or obj.score >= score_threshold]


def write_predictions_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "frame_id",
        "class",
        "score",
        "bbox_2d",
        "dimensions_hwl",
        "location_xyz",
        "depth_m",
        "distance_m",
        "rotation_y",
        "alpha",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def evaluate_frames(
    *,
    prediction_dir: Path,
    kitti_root: Path,
    split: str,
    frame_ids: list[str],
    output_dir: Path,
    score_threshold: float,
    match_iou: float,
    save_visuals: int,
    visual_calib_dir: Path | None = None,
) -> dict[str, object]:
    prediction_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    all_predictions_by_frame: dict[str, list[Kitti3DObject]] = {}

    for frame_id in frame_ids:
        pred_path = prediction_dir / f"{frame_id}.txt"
        label_path = kitti_root / split / "label_2" / f"{frame_id}.txt"
        predictions = filter_by_score(read_kitti_objects(pred_path, frame_id), score_threshold)
        labels = read_kitti_objects(label_path, frame_id)
        all_predictions_by_frame[frame_id] = predictions

        for obj in predictions:
            prediction_rows.append(object_to_record(obj))

        for match in match_predictions_to_labels(predictions, labels, iou_threshold=match_iou):
            pred = match.pred
            label = match.label
            match_rows.append(
                {
                    "frame_id": frame_id,
                    "class": pred.object_type,
                    "score": pred.score,
                    "iou_2d": match.iou_2d,
                    "pred_depth_m": pred.depth_m,
                    "label_depth_m": label.depth_m,
                    "abs_depth_error_m": abs(pred.depth_m - label.depth_m),
                    "pred_distance_m": pred.distance_m,
                    "label_distance_m": label.distance_m,
                    "abs_distance_error_m": abs(pred.distance_m - label.distance_m),
                    "pred_rotation_y": pred.rotation_y,
                    "label_rotation_y": label.rotation_y,
                    "abs_yaw_error_deg": math.degrees(angle_error_rad(pred.rotation_y, label.rotation_y)),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    (output_dir / "predictions.json").write_text(json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_matches_csv(match_rows, output_dir / "matches.csv")

    if save_visuals > 0:
        write_visuals(
            kitti_root=kitti_root,
            split=split,
            frame_ids=frame_ids[:save_visuals],
            predictions_by_frame=all_predictions_by_frame,
            output_dir=output_dir / "visuals",
            score_threshold=score_threshold,
            visual_calib_dir=visual_calib_dir,
        )

    summary = summarize(prediction_rows, match_rows, frame_count=len(frame_ids), score_threshold=score_threshold, match_iou=match_iou)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_matches_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "frame_id",
        "class",
        "score",
        "iou_2d",
        "pred_depth_m",
        "label_depth_m",
        "abs_depth_error_m",
        "pred_distance_m",
        "label_distance_m",
        "abs_distance_error_m",
        "pred_rotation_y",
        "label_rotation_y",
        "abs_yaw_error_deg",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_visuals(
    *,
    kitti_root: Path,
    split: str,
    frame_ids: list[str],
    predictions_by_frame: dict[str, list[Kitti3DObject]],
    output_dir: Path,
    score_threshold: float,
    visual_calib_dir: Path | None = None,
) -> None:
    for frame_id in frame_ids:
        image_path = kitti_root / split / "image_2" / f"{frame_id}.png"
        calib_path = (visual_calib_dir / f"{frame_id}.txt") if visual_calib_dir else kitti_root / split / "calib" / f"{frame_id}.txt"
        predictions = predictions_by_frame.get(frame_id, [])
        if not predictions:
            continue
        draw_frame_overlay(image_path, calib_path, predictions, output_dir / "image_overlay" / f"{frame_id}.jpg", score_threshold=score_threshold)
        draw_bev(predictions, output_dir / "bev" / f"{frame_id}.jpg", score_threshold=score_threshold)


def summarize(
    prediction_rows: list[dict[str, object]],
    match_rows: list[dict[str, object]],
    *,
    frame_count: int,
    score_threshold: float,
    match_iou: float,
) -> dict[str, object]:
    return {
        "frame_count": frame_count,
        "prediction_count": len(prediction_rows),
        "matched_count": len(match_rows),
        "score_threshold": score_threshold,
        "match_iou": match_iou,
        "mean_abs_depth_error_m": finite_mean(row["abs_depth_error_m"] for row in match_rows),
        "mean_abs_distance_error_m": finite_mean(row["abs_distance_error_m"] for row in match_rows),
        "mean_abs_yaw_error_deg": finite_mean(row["abs_yaw_error_deg"] for row in match_rows),
        "note": "Lightweight smoke metrics: matched by 2D IoU; depth/distance/yaw errors are not official KITTI AP.",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI-format mono 3D prediction txt files.")
    parser.add_argument("--prediction-dir", required=True, help="Directory containing per-frame KITTI-format prediction txt files.")
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--split", default="training", choices=("training", "testing"))
    parser.add_argument("--frame-list", default=None, help="Optional txt file with one frame id per line.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--save-visuals", type=int, default=12)
    parser.add_argument("--visual-calib-dir", default=None, help="Optional calib directory for 3D overlay projection.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_dir = Path(args.prediction_dir)
    frame_list = Path(args.frame_list) if args.frame_list else None
    frame_ids = discover_frame_ids(prediction_dir, frame_list, args.max_frames)
    if not frame_ids:
        raise RuntimeError(f"No prediction frame ids found in {prediction_dir}")

    summary = evaluate_frames(
        prediction_dir=prediction_dir,
        kitti_root=Path(args.kitti_root),
        split=args.split,
        frame_ids=frame_ids,
        output_dir=Path(args.output_dir),
        score_threshold=args.score_threshold,
        match_iou=args.match_iou,
        save_visuals=args.save_visuals,
        visual_calib_dir=Path(args.visual_calib_dir) if args.visual_calib_dir else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
