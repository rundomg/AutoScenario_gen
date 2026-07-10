"""Evaluate monocular metric depth for KITTI vehicle distance estimates.

The script compares per-vehicle predicted metric depth against KITTI 3D label
depth and projected LiDAR depth. It is intentionally standalone so it can be
run without touching the main AutoScenario generation pipeline.

Examples:
    python experiments/monocular_depth_eval/evaluate_kitti_depth.py --model none --max-images 20
    python experiments/monocular_depth_eval/evaluate_kitti_depth.py --model depth-anything-v2-hf --max-images 500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .depth_models import DEFAULT_HF_MODEL, build_depth_model
except ImportError:  # Direct script execution: python experiments/.../evaluate_kitti_depth.py
    from depth_models import DEFAULT_HF_MODEL, build_depth_model

try:
    from .roi_sampling import build_sampling_region, depth_stats_in_region, lidar_depth_in_region
except ImportError:  # Direct script execution: python experiments/.../evaluate_kitti_depth.py
    from roi_sampling import build_sampling_region, depth_stats_in_region, lidar_depth_in_region


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
YOLO_CONFIG_DIR = RESULTS_ROOT / "config" / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "KITTI" / "kitti"
DEFAULT_VEHICLE_CLASSES = ("Car", "Van", "Truck", "Tram")


@dataclass(frozen=True)
class KittiObject:
    frame_id: str
    class_name: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float

    @property
    def label_depth_m(self) -> float:
        return float(self.location_xyz[2])

    @property
    def label_distance_m(self) -> float:
        x, y, z = self.location_xyz
        return float(math.sqrt(x * x + y * y + z * z))

    @property
    def bbox_area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class EvalTarget:
    frame_id: str
    object_index: int
    box_source: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    label: KittiObject
    det_conf: float | None = None
    match_iou: float | None = None
    instance_mask: np.ndarray | None = None
    mask_source: str = "bbox"
    seg_conf: float | None = None
    instance_mask_pixels: int = 0

    @property
    def bbox_area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_kitti_label_file(label_path: Path, frame_id: str, classes: set[str]) -> list[KittiObject]:
    objects: list[KittiObject] = []
    if not label_path.exists():
        return objects

    for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 15:
            raise ValueError(f"Malformed label line {label_path}:{line_number}: {line}")
        class_name = parts[0]
        if class_name not in classes:
            continue
        try:
            truncation = float(parts[1])
            occlusion = int(parts[2])
            alpha = float(parts[3])
            bbox = tuple(float(v) for v in parts[4:8])
            dimensions = tuple(float(v) for v in parts[8:11])
            location = tuple(float(v) for v in parts[11:14])
            rotation_y = float(parts[14])
        except ValueError as exc:
            raise ValueError(f"Failed to parse label line {label_path}:{line_number}") from exc
        objects.append(
            KittiObject(
                frame_id=frame_id,
                class_name=class_name,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                dimensions_hwl=dimensions,  # type: ignore[arg-type]
                location_xyz=location,  # type: ignore[arg-type]
                rotation_y=rotation_y,
            )
        )
    return objects


def parse_kitti_calib(calib_path: Path) -> dict[str, np.ndarray]:
    calib: dict[str, np.ndarray] = {}
    for line_number, raw_line in enumerate(calib_path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        key, value = raw_line.split(":", 1)
        try:
            calib[key] = np.array([float(v) for v in value.split()], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Failed to parse calibration line {calib_path}:{line_number}") from exc

    required = {"P2", "R0_rect", "Tr_velo_to_cam"}
    missing = required - set(calib)
    if missing:
        raise ValueError(f"Missing calibration fields in {calib_path}: {sorted(missing)}")

    p2 = calib["P2"].reshape(3, 4)
    r0 = np.eye(4, dtype=np.float64)
    r0[:3, :3] = calib["R0_rect"].reshape(3, 3)
    tr_velo_to_cam = np.eye(4, dtype=np.float64)
    tr_velo_to_cam[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)
    return {"P2": p2, "R0_rect_4x4": r0, "Tr_velo_to_cam_4x4": tr_velo_to_cam}


def project_velodyne_to_image(
    velodyne_path: Path,
    calib: dict[str, np.ndarray],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return projected LiDAR u, v, and rectified camera z depth arrays."""

    if not velodyne_path.exists() or velodyne_path.stat().st_size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    points = np.fromfile(str(velodyne_path), dtype=np.float32).reshape(-1, 4)
    points_h = np.ones((points.shape[0], 4), dtype=np.float64)
    points_h[:, :3] = points[:, :3]

    rect_points = (calib["R0_rect_4x4"] @ calib["Tr_velo_to_cam_4x4"] @ points_h.T).T
    z = rect_points[:, 2]
    front_mask = z > 0.1
    rect_points = rect_points[front_mask]
    z = z[front_mask]
    if rect_points.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    image_points = (calib["P2"] @ rect_points.T).T
    u = image_points[:, 0] / image_points[:, 2]
    v = image_points[:, 1] / image_points[:, 2]
    height, width = image_shape
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_image], v[in_image], z[in_image]


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_yolo_seg_targets(
    image_path: Path,
    labels: list[KittiObject],
    *,
    conf: float,
    model_path: str,
    imgsz: int | None,
    max_count: int | None,
    match_iou_threshold: float,
) -> list[EvalTarget]:
    if not labels:
        return []
    try:
        from instance_segmenter import segment_vehicles
    except ImportError:
        from .instance_segmenter import segment_vehicles

    # 评估脚本默认保留历史上限；需要查远车召回时可显式调大 max_count/imgsz。
    instances = segment_vehicles(image_path, conf=conf, model_path=model_path, imgsz=imgsz, max_count=max_count)
    targets: list[EvalTarget] = []
    used_label_indices: set[int] = set()
    for instance in instances:
        bbox_values = instance.get("bbox_xyxy")
        if not bbox_values or len(bbox_values) != 4:
            continue
        seg_bbox = tuple(float(v) for v in bbox_values)
        best_index: int | None = None
        best_iou = 0.0
        # Keep the deployment source as the predicted instance, but attach one
        # KITTI label so the distance error and occlusion ordering use truth.
        for label_index, label in enumerate(labels):
            if label_index in used_label_indices:
                continue
            iou = bbox_iou(seg_bbox, label.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_index = label_index
        if best_index is None or best_iou < match_iou_threshold:
            continue
        mask = instance.get("mask")
        if mask is None:
            continue
        used_label_indices.add(best_index)
        label = labels[best_index]
        conf_value = finite_float(instance.get("conf"))
        # The bbox comes from the instance model; the mask is carried through to
        # roi_sampling.py where it removes background before depth statistics.
        targets.append(
            EvalTarget(
                frame_id=label.frame_id,
                object_index=len(targets),
                box_source="yolo-seg",
                class_name=str(instance.get("label", label.class_name)),
                bbox_xyxy=seg_bbox,  # type: ignore[arg-type]
                label=label,
                det_conf=conf_value,
                match_iou=best_iou,
                instance_mask=np.asarray(mask, dtype=bool),
                mask_source="yolo-seg",
                seg_conf=conf_value,
                instance_mask_pixels=int(instance.get("mask_pixels", 0) or 0),
            )
        )
    return targets


def discover_frame_ids(split_dir: Path, frame_list_path: Path | None) -> list[str]:
    if frame_list_path is not None:
        return [
            line.strip()
            for line in frame_list_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    image_dir = split_dir / "image_2"
    return sorted(path.stem for path in image_dir.glob("*.png"))


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def compute_error(prediction: float | None, target: float | None) -> tuple[float | None, float | None]:
    pred = finite_float(prediction)
    tgt = finite_float(target)
    if pred is None or tgt is None or tgt <= 0:
        return None, None
    abs_error = abs(pred - tgt)
    return abs_error, abs_error / tgt


def fmt(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def draw_overlay(
    image_bgr: np.ndarray,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    canvas = image_bgr.copy()
    for row in rows:
        x1 = int(float(row["bbox_x1"]))
        y1 = int(float(row["bbox_y1"]))
        x2 = int(float(row["bbox_x2"]))
        y2 = int(float(row["bbox_y2"]))
        pred = finite_float(row.get("pred_depth_m"))
        label = finite_float(row.get("label_depth_m"))
        abs_error = finite_float(row.get("abs_error_label_m"))
        visible_ratio = finite_float(row.get("visible_roi_ratio"))
        overlap_ratio = finite_float(row.get("occluder_overlap_ratio"))
        foreground_ratio = finite_float(row.get("foreground_roi_ratio"))
        color = (0, 200, 255) if pred is not None else (180, 180, 180)
        _draw_instance_mask(canvas, row)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _draw_sampling_mask(canvas, row)
        if pred is None:
            text = f"{row['class_name']} gt={fmt(label, 1)}m"
        else:
            text = f"{row['class_name']} pred={fmt(pred, 1)} gt={fmt(label, 1)} err={fmt(abs_error, 1)}"
        if visible_ratio is not None and overlap_ratio is not None:
            text += f" vis={visible_ratio * 100:.0f}% occ={overlap_ratio * 100:.0f}%"
        if foreground_ratio is not None and row.get("mask_source") == "yolo-seg":
            text += f" fg={foreground_ratio * 100:.0f}%"
        y_text = max(18, y1 - 6)
        cv2.putText(canvas, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(canvas, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _draw_instance_mask(canvas: np.ndarray, row: dict[str, object]) -> None:
    mask = row.get("_instance_mask")
    if mask is None:
        return
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape[:2] != canvas.shape[:2] or int(mask_array.sum()) == 0:
        return
    # Amber fill is the predicted instance foreground before ROI/inset/occluder cuts.
    overlay = canvas.copy()
    overlay[mask_array] = (0, 165, 255)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)


def _draw_sampling_mask(canvas: np.ndarray, row: dict[str, object]) -> None:
    roi = row.get("_roi_xyxy")
    mask = row.get("_visible_mask")
    if not roi or mask is None:
        return
    x1, y1, _x2, _y2 = (int(v) for v in roi)
    mask_array = np.asarray(mask, dtype=np.uint8)
    if mask_array.size == 0 or int(mask_array.sum()) == 0:
        return
    contours, _ = cv2.findContours(mask_array * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour[:, 0, 0] += x1
        contour[:, 0, 1] += y1
    # Cyan contours show the final pixels actually used for distance sampling.
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1)


def summarize_errors(rows: list[dict[str, object]], key_abs: str, key_rel: str) -> dict[str, object]:
    abs_values = np.array([finite_float(row.get(key_abs)) for row in rows], dtype=object)
    rel_values = np.array([finite_float(row.get(key_rel)) for row in rows], dtype=object)
    abs_clean = np.array([v for v in abs_values if v is not None], dtype=np.float64)
    rel_clean = np.array([v for v in rel_values if v is not None], dtype=np.float64)
    if abs_clean.size == 0:
        return {"count": 0}
    return {
        "count": int(abs_clean.size),
        "mae_m": float(np.mean(abs_clean)),
        "median_abs_error_m": float(np.median(abs_clean)),
        "rmse_m": float(np.sqrt(np.mean(abs_clean * abs_clean))),
        "absrel": float(np.mean(rel_clean)) if rel_clean.size else None,
    }


def summarize_distance_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bins = [(0, 10), (10, 20), (20, 40), (40, 80), (80, math.inf)]
    output: list[dict[str, object]] = []
    for low, high in bins:
        bucket = []
        for row in rows:
            depth = finite_float(row.get("label_depth_m"))
            error = finite_float(row.get("abs_error_label_m"))
            if depth is None or error is None:
                continue
            if low <= depth < high:
                bucket.append(error)
        label = f"{low}-{high}m" if math.isfinite(high) else f"{low}+m"
        if bucket:
            values = np.array(bucket, dtype=np.float64)
            output.append(
                {
                    "range": label,
                    "count": int(values.size),
                    "mae_m": float(np.mean(values)),
                    "median_abs_error_m": float(np.median(values)),
                }
            )
        else:
            output.append({"range": label, "count": 0})
    return output


def summarize_by_field(rows: list[dict[str, object]], field: str) -> dict[str, dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        value = row.get(field)
        key = str(value) if value not in (None, "") else "unknown"
        groups.setdefault(key, []).append(row)
    return {
        key: summarize_errors(group_rows, "abs_error_label_m", "rel_error_label")
        for key, group_rows in sorted(groups.items(), key=lambda item: item[0])
    }


def summarize_overlap_bins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bins = [
        ("0%", 0.0, 0.0),
        ("0-10%", 0.0, 0.10),
        ("10-25%", 0.10, 0.25),
        ("25-50%", 0.25, 0.50),
        ("50%+", 0.50, math.inf),
    ]
    output: list[dict[str, object]] = []
    for label, low, high in bins:
        bucket = []
        for row in rows:
            ratio = finite_float(row.get("occluder_overlap_ratio"))
            if ratio is None:
                continue
            if label == "0%":
                in_bucket = ratio == 0.0
            elif math.isinf(high):
                in_bucket = ratio >= low
            else:
                in_bucket = low < ratio <= high
            if in_bucket:
                bucket.append(row)
        summary = summarize_errors(bucket, "abs_error_label_m", "rel_error_label")
        summary["range"] = label
        output.append(summary)
    return output


def count_by_field(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field)
        key = str(value) if value not in (None, "") else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "object_index",
        "box_source",
        "class_name",
        "matched_label_class",
        "det_conf",
        "seg_conf",
        "match_iou",
        "mask_source",
        "truncation",
        "occlusion",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_area_px",
        "label_x_m",
        "label_y_m",
        "label_z_m",
        "label_depth_m",
        "label_distance_m",
        "lidar_depth_m",
        "lidar_points_in_roi",
        "pred_depth_m",
        "pred_depth_p10_m",
        "pred_depth_p50_m",
        "pred_depth_p90_m",
        "abs_error_label_m",
        "rel_error_label",
        "abs_error_lidar_m",
        "rel_error_lidar",
        "visible_roi_ratio",
        "occluder_overlap_ratio",
        "foreground_roi_ratio",
        "background_removed_ratio",
        "instance_mask_pixels",
        "roi_pixels",
        "visible_pixels",
        "sampling_status",
        "roi_mode",
        "model",
        "image_path",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Place results by depth model and target source unless the caller overrides.

    The only active target source is YOLO instance segmentation. Keeping it in
    the path makes old bbox/label baselines easy to archive separately.
    """

    if args.output_dir:
        return Path(args.output_dir)
    frame_part = "all" if args.max_images is None else str(args.max_images)
    if args.model == "none":
        return RESULTS_ROOT / "smoke" / "yolo-seg" / f"{frame_part}_geometry"
    return RESULTS_ROOT / "runs" / safe_path_part(args.model) / "yolo-seg" / f"{frame_part}_masked"


def safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-") or "unknown"


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    kitti_root = Path(args.kitti_root)
    split_dir = kitti_root / args.split
    image_dir = split_dir / "image_2"
    calib_dir = split_dir / "calib"
    label_dir = split_dir / "label_2"
    velodyne_dir = split_dir / "velodyne"

    for required_dir in (image_dir, calib_dir, label_dir, velodyne_dir):
        if not required_dir.exists():
            raise FileNotFoundError(f"Missing KITTI directory: {required_dir}")

    frame_ids = discover_frame_ids(split_dir, Path(args.frame_list) if args.frame_list else None)
    frame_ids = frame_ids[args.start_index :]
    if args.max_images is not None:
        frame_ids = frame_ids[: args.max_images]
    if not frame_ids:
        raise RuntimeError("No KITTI frames selected.")

    classes = {value.strip() for value in args.classes.split(",") if value.strip()}
    model = build_depth_model(args)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    rows: list[dict[str, object]] = []
    processed_frames = 0
    frames_with_objects = 0

    for frame_offset, frame_id in enumerate(frame_ids, start=1):
        image_path = image_dir / f"{frame_id}.png"
        calib_path = calib_dir / f"{frame_id}.txt"
        label_path = label_dir / f"{frame_id}.txt"
        velodyne_path = velodyne_dir / f"{frame_id}.bin"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] failed to read image: {image_path}", file=sys.stderr)
            continue
        height, width = image.shape[:2]
        objects = parse_kitti_label_file(label_path, frame_id, classes)
        if args.min_bbox_height > 0:
            objects = [
                obj for obj in objects if (obj.bbox_xyxy[3] - obj.bbox_xyxy[1]) >= args.min_bbox_height
            ]
        targets = build_yolo_seg_targets(
            image_path,
            objects,
            conf=args.yolo_seg_conf,
            model_path=args.yolo_seg_model,
            imgsz=args.yolo_seg_imgsz,
            max_count=normalize_max_count(args.yolo_seg_max_count),
            match_iou_threshold=args.match_iou,
        )
        if args.min_bbox_height > 0:
            targets = [
                target
                for target in targets
                if (target.bbox_xyxy[3] - target.bbox_xyxy[1]) >= args.min_bbox_height
            ]
        if not targets:
            processed_frames += 1
            continue
        frames_with_objects += 1

        calib = parse_kitti_calib(calib_path)
        projected_lidar = project_velodyne_to_image(velodyne_path, calib, (height, width))
        depth_map = model.predict_depth(image_path, image, calib) if model is not None else None
        if depth_map is not None and depth_map.shape[:2] != (height, width):
            raise RuntimeError(
                f"Depth map shape mismatch for {frame_id}: "
                f"{depth_map.shape[:2]} vs image {(height, width)}"
            )

        frame_rows: list[dict[str, object]] = []
        for target in targets:
            obj = target.label
            region = build_sampling_region(
                target,
                targets,
                (height, width),
                mode=args.roi,
                center_ratio=args.center_ratio,
                inset_ratio=args.roi_inset_ratio,
                min_visible_ratio=args.min_visible_ratio,
                disable_overlap_mask=args.disable_overlap_mask,
            )
            if args.skip_low_visible and region.sampling_status != "ok":
                continue
            lidar_depth, lidar_points = lidar_depth_in_region(projected_lidar, region)
            depth_stats = depth_stats_in_region(
                depth_map,
                region,
                quantile_low=args.depth_quantile_low,
                quantile_high=args.depth_quantile_high,
            )
            pred_depth = depth_stats.median_m
            abs_label, rel_label = compute_error(pred_depth, obj.label_depth_m)
            abs_lidar, rel_lidar = compute_error(pred_depth, lidar_depth)
            x, y, z = obj.location_xyz
            x1, y1, x2, y2 = target.bbox_xyxy
            row: dict[str, object] = {
                "frame_id": frame_id,
                "object_index": target.object_index,
                "box_source": target.box_source,
                "class_name": target.class_name,
                "matched_label_class": obj.class_name,
                "det_conf": fmt(target.det_conf, 4),
                "seg_conf": fmt(target.seg_conf, 4),
                "match_iou": fmt(target.match_iou, 4),
                "mask_source": target.mask_source,
                "truncation": fmt(obj.truncation, 2),
                "occlusion": obj.occlusion,
                "bbox_x1": fmt(x1, 2),
                "bbox_y1": fmt(y1, 2),
                "bbox_x2": fmt(x2, 2),
                "bbox_y2": fmt(y2, 2),
                "bbox_area_px": fmt(target.bbox_area_px, 2),
                "label_x_m": fmt(x, 4),
                "label_y_m": fmt(y, 4),
                "label_z_m": fmt(z, 4),
                "label_depth_m": fmt(obj.label_depth_m, 4),
                "label_distance_m": fmt(obj.label_distance_m, 4),
                "lidar_depth_m": fmt(lidar_depth, 4),
                "lidar_points_in_roi": lidar_points,
                "pred_depth_m": fmt(pred_depth, 4),
                "pred_depth_p10_m": fmt(depth_stats.p10_m, 4),
                "pred_depth_p50_m": fmt(depth_stats.p50_m, 4),
                "pred_depth_p90_m": fmt(depth_stats.p90_m, 4),
                "abs_error_label_m": fmt(abs_label, 4),
                "rel_error_label": fmt(rel_label, 6),
                "abs_error_lidar_m": fmt(abs_lidar, 4),
                "rel_error_lidar": fmt(rel_lidar, 6),
                "visible_roi_ratio": fmt(region.visible_roi_ratio, 6),
                "occluder_overlap_ratio": fmt(region.occluder_overlap_ratio, 6),
                "foreground_roi_ratio": fmt(region.foreground_roi_ratio, 6),
                "background_removed_ratio": fmt(region.background_removed_ratio, 6),
                "instance_mask_pixels": target.instance_mask_pixels,
                "roi_pixels": region.roi_pixels,
                "visible_pixels": region.visible_pixels,
                "sampling_status": region.sampling_status,
                "roi_mode": args.roi,
                "model": args.model,
                "image_path": str(image_path),
                "_roi_xyxy": region.roi_xyxy,
                "_visible_mask": region.visible_mask,
                "_instance_mask": target.instance_mask,
            }
            frame_rows.append(row)
        rows.extend(frame_rows)

        if args.save_overlays > 0 and frames_with_objects <= args.save_overlays:
            draw_overlay(image, frame_rows, overlay_dir / f"{frame_id}.jpg")

        processed_frames += 1
        if args.progress_every > 0 and frame_offset % args.progress_every == 0:
            print(f"processed {frame_offset}/{len(frame_ids)} frames, rows={len(rows)}")

    csv_path = output_dir / "metrics.csv"
    write_csv(rows, csv_path)

    summary = {
        "model": args.model,
        "hf_model": args.hf_model if args.model == "depth-anything-v2-hf" else None,
        "metric3d_model": args.metric3d_model if args.model == "metric3d" else None,
        "unidepth_version": args.unidepth_version if args.model == "unidepth" else None,
        "unidepth_backbone": args.unidepth_backbone if args.model == "unidepth" else None,
        "kitti_root": str(kitti_root),
        "split": args.split,
        "selected_frames": len(frame_ids),
        "processed_frames": processed_frames,
        "frames_with_vehicle_objects": frames_with_objects,
        "vehicle_objects": len(rows),
        "objects_with_lidar_depth": sum(1 for row in rows if finite_float(row.get("lidar_depth_m")) is not None),
        "objects_with_pred_depth": sum(1 for row in rows if finite_float(row.get("pred_depth_m")) is not None),
        "target_source": "yolo-seg",
        "box_source": "yolo-seg",
        "yolo_seg_model": args.yolo_seg_model,
        "yolo_seg_conf": args.yolo_seg_conf,
        "yolo_seg_imgsz": args.yolo_seg_imgsz,
        "yolo_seg_max_count": normalize_max_count(args.yolo_seg_max_count),
        "match_iou": args.match_iou,
        "label_depth_error": summarize_errors(rows, "abs_error_label_m", "rel_error_label"),
        "lidar_depth_error": summarize_errors(rows, "abs_error_lidar_m", "rel_error_lidar"),
        "label_distance_bins": summarize_distance_bins(rows),
        "occlusion_groups": summarize_by_field(rows, "occlusion"),
        "overlap_ratio_bins": summarize_overlap_bins(rows),
        "sampling_status_counts": count_by_field(rows, "sampling_status"),
        "mask_source_counts": count_by_field(rows, "mask_source"),
        "roi_mode": args.roi,
        "roi_inset_ratio": args.roi_inset_ratio,
        "depth_quantile_low": args.depth_quantile_low,
        "depth_quantile_high": args.depth_quantile_high,
        "min_visible_ratio": args.min_visible_ratio,
        "disable_overlap_mask": args.disable_overlap_mask,
        "skip_low_visible": args.skip_low_visible,
        "classes": sorted(classes),
        "output_dir": str(output_dir),
        "metrics_csv": str(csv_path),
        "overlays_dir": str(overlay_dir) if args.save_overlays > 0 else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def normalize_max_count(value: int | None) -> int | None:
    """把命令行里的 0 转成无限制，正整数则作为最大保留实例数。"""

    if value is None or value <= 0:
        return None
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate monocular depth model vehicle distance accuracy on KITTI Object."
    )
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT), help="Path to KITTI/kitti root.")
    parser.add_argument("--split", default="training", choices=("training",), help="KITTI split to evaluate.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for metrics and overlays. Defaults to results/runs/<model>/yolo-seg/<N>_masked.",
    )
    parser.add_argument("--model", default="none", choices=("none", "depth-anything-v2-hf", "metric3d", "unidepth"))
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HuggingFace model id for Depth Anything V2.")
    parser.add_argument("--hf-offline", action="store_true", help="Load HuggingFace assets from the local cache only.")
    parser.add_argument("--metric3d-model", default="metric3d_vit_small", help="Metric3D Torch Hub entrypoint.")
    parser.add_argument("--metric3d-offline", action="store_true", help="Use only the locally cached Metric3D repo/checkpoint.")
    parser.add_argument("--unidepth-version", default="v2", choices=("v1", "v2", "v2old"), help="UniDepth Torch Hub version.")
    parser.add_argument("--unidepth-backbone", default="vits14", help="UniDepth backbone, for example vits14, vitb14, or vitl14.")
    parser.add_argument("--unidepth-offline", action="store_true", help="Use only the locally cached UniDepth repo/checkpoint.")
    parser.add_argument("--device", default="auto", help="Depth model device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of frames to evaluate.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset in sorted KITTI frame ids.")
    parser.add_argument("--frame-list", default=None, help="Optional text file with one frame id per line.")
    parser.add_argument("--classes", default=",".join(DEFAULT_VEHICLE_CLASSES), help="Comma-separated label classes.")
    parser.add_argument("--roi", default="lower-half", choices=("lower-half", "center", "full"))
    parser.add_argument("--center-ratio", type=float, default=0.5, help="ROI side ratio for --roi center.")
    parser.add_argument("--roi-inset-ratio", type=float, default=0.06, help="Shrink ROI inward to reduce bbox-edge background.")
    parser.add_argument("--depth-quantile-low", type=float, default=10.0, help="Lower percentile kept for robust depth sampling.")
    parser.add_argument("--depth-quantile-high", type=float, default=90.0, help="Upper percentile kept for robust depth sampling.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.05, help="Visible-mask ratio below which samples are flagged.")
    parser.add_argument("--disable-overlap-mask", action="store_true", help="Do not subtract closer overlapping vehicles from ROI.")
    parser.add_argument("--skip-low-visible", action="store_true", help="Drop samples whose visible ROI is empty or too small.")
    parser.add_argument("--min-bbox-height", type=float, default=8.0, help="Skip tiny objects below this box height.")
    parser.add_argument("--yolo-seg-model", default="yolo11m-seg.pt", help="YOLO segmentation model for vehicle instances.")
    parser.add_argument("--yolo-seg-conf", type=float, default=0.25, help="YOLO segmentation confidence.")
    parser.add_argument("--yolo-seg-imgsz", type=int, default=None, help="YOLO segmentation inference size; larger helps small distant vehicles.")
    parser.add_argument("--yolo-seg-max-count", type=int, default=12, help="Max kept instances after dedupe; 0 means no limit.")
    parser.add_argument("--match-iou", type=float, default=0.5, help="Min YOLO-seg-to-KITTI label IoU.")
    parser.add_argument("--save-overlays", type=int, default=12, help="Number of object-containing frames to visualize.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N selected frames.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
