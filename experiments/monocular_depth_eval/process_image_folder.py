"""Run vehicle distance inference on an unlabeled image folder.

The script is for deployment-style images without KITTI labels. It detects
vehicle instances with YOLO segmentation, predicts a metric depth map, samples
depth inside each vehicle mask, and writes results under `<image_dir>/output`
by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    from .depth_models import DEFAULT_HF_MODEL, build_depth_model
    from .instance_segmenter import segment_vehicles
    from .roi_sampling import build_sampling_region, depth_stats_in_region
except ImportError:  # Direct script execution: python experiments/.../process_image_folder.py
    from depth_models import DEFAULT_HF_MODEL, build_depth_model
    from instance_segmenter import segment_vehicles
    from roi_sampling import build_sampling_region, depth_stats_in_region


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_VEHICLE_CLASS_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")


@dataclass(frozen=True)
class PseudoDepthLabel:
    """Tiny label-like object used only for heuristic occlusion ordering."""

    label_depth_m: float | None


@dataclass(frozen=True)
class FolderTarget:
    """Vehicle target predicted from an unlabeled image."""

    object_index: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    instance_mask: np.ndarray
    det_conf: float | None
    seg_conf: float | None
    instance_mask_pixels: int
    label: PseudoDepthLabel
    box_source: str = "yolo-seg"
    mask_source: str = "yolo-seg"

    @property
    def bbox_area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def discover_images(input_dir: Path, *, recursive: bool, output_dir: Path) -> list[Path]:
    pattern_iter = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images: list[Path] = []
    output_dir = output_dir.resolve()
    for path in pattern_iter:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Avoid recursively feeding generated overlays/depth images back into inference.
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved == output_dir or output_dir in resolved.parents:
            continue
        images.append(path)
    return sorted(images)


def build_folder_targets(
    image_path: Path,
    image_shape: tuple[int, int],
    *,
    conf: float,
    model_path: str,
    imgsz: int | None,
    max_count: int | None,
    allowed_classes: set[str],
    use_heuristic_occlusion: bool,
) -> list[FolderTarget]:
    # 远处小车通常需要更大的 imgsz 和更高的 max_count 才不容易被筛掉。
    instances = segment_vehicles(image_path, conf=conf, model_path=model_path, imgsz=imgsz, max_count=max_count)
    height, width = image_shape
    targets: list[FolderTarget] = []
    for index, instance in enumerate(instances):
        label_name = str(instance.get("label", "vehicle")).lower()
        if label_name not in allowed_classes:
            continue
        bbox_values = instance.get("bbox_xyxy")
        mask = instance.get("mask")
        if not bbox_values or len(bbox_values) != 4 or mask is None:
            continue
        bbox = tuple(float(v) for v in bbox_values)
        pseudo_depth = heuristic_foreground_depth(bbox, (height, width)) if use_heuristic_occlusion else None
        conf_value = finite_float(instance.get("conf"))
        targets.append(
            FolderTarget(
                object_index=index,
                class_name=label_name,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                instance_mask=np.asarray(mask, dtype=bool),
                det_conf=conf_value,
                seg_conf=conf_value,
                instance_mask_pixels=int(instance.get("mask_pixels", 0) or 0),
                label=PseudoDepthLabel(pseudo_depth),
            )
        )
    return targets


def heuristic_foreground_depth(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> float:
    """Return a pseudo-depth rank for unlabeled occlusion handling.

    Smaller values mean "probably closer". With no labels or LiDAR, we only use
    image geometry: larger boxes and boxes lower in the image are treated as
    likely foreground occluders. This rank is never reported as a distance.
    """

    height, width = image_shape
    x1, y1, x2, y2 = bbox
    area_ratio = max(1.0, (x2 - x1) * (y2 - y1)) / max(1.0, width * height)
    bottom_ratio = max(0.0, min(1.0, y2 / max(1.0, height)))
    foreground_score = area_ratio * (0.5 + bottom_ratio)
    return 1.0 / max(foreground_score, 1e-6)


def synthetic_calib(
    image_shape: tuple[int, int],
    *,
    focal_length_px: float | None,
    focal_ratio: float,
    cx: float | None,
    cy: float | None,
) -> dict[str, np.ndarray]:
    """Create a KITTI-like P2 matrix for models that need intrinsics.

    For unlabeled folders there is usually no calibration file. Metric3D and
    UniDepth can still run with an approximate focal length, but absolute metric
    scale will only be as good as this camera assumption.
    """

    height, width = image_shape
    fx = focal_length_px if focal_length_px is not None else width * focal_ratio
    fy = fx
    px = width * 0.5 if cx is None else cx
    py = height * 0.5 if cy is None else cy
    p2 = np.array([[fx, 0.0, px, 0.0], [0.0, fy, py, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    return {"P2": p2}


def process_image(
    image_path: Path,
    *,
    model,
    args: argparse.Namespace,
    output_dir: Path,
    relative_name: str,
) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[warn] failed to read image: {image_path}")
        return []

    height, width = image.shape[:2]
    calib = synthetic_calib(
        (height, width),
        focal_length_px=args.focal_length_px,
        focal_ratio=args.focal_ratio,
        cx=args.cx,
        cy=args.cy,
    )
    depth_map = model.predict_depth(image_path, image, calib)
    targets = build_folder_targets(
        image_path,
        (height, width),
        conf=args.yolo_seg_conf,
        model_path=args.yolo_seg_model,
        imgsz=args.yolo_seg_imgsz,
        max_count=normalize_max_count(args.yolo_seg_max_count),
        allowed_classes=set(args.vehicle_classes),
        use_heuristic_occlusion=not args.disable_heuristic_occlusion,
    )

    rows: list[dict[str, object]] = []
    for target in targets:
        region = build_sampling_region(
            target,
            targets,
            (height, width),
            mode=args.roi,
            center_ratio=args.center_ratio,
            inset_ratio=args.roi_inset_ratio,
            min_visible_ratio=args.min_visible_ratio,
            disable_overlap_mask=args.disable_heuristic_occlusion,
        )
        stats = depth_stats_in_region(
            depth_map,
            region,
            quantile_low=args.depth_quantile_low,
            quantile_high=args.depth_quantile_high,
        )
        x1, y1, x2, y2 = target.bbox_xyxy
        rows.append(
            {
                "image": relative_name,
                "object_index": target.object_index,
                "class_name": target.class_name,
                "seg_conf": fmt(target.seg_conf, 4),
                "pred_depth_m": fmt(stats.median_m, 4),
                "pred_depth_p10_m": fmt(stats.p10_m, 4),
                "pred_depth_p50_m": fmt(stats.p50_m, 4),
                "pred_depth_p90_m": fmt(stats.p90_m, 4),
                "bbox_x1": fmt(x1, 2),
                "bbox_y1": fmt(y1, 2),
                "bbox_x2": fmt(x2, 2),
                "bbox_y2": fmt(y2, 2),
                "visible_roi_ratio": fmt(region.visible_roi_ratio, 6),
                "occluder_overlap_ratio": fmt(region.occluder_overlap_ratio, 6),
                "foreground_roi_ratio": fmt(region.foreground_roi_ratio, 6),
                "background_removed_ratio": fmt(region.background_removed_ratio, 6),
                "instance_mask_pixels": target.instance_mask_pixels,
                "roi_pixels": region.roi_pixels,
                "visible_pixels": region.visible_pixels,
                "sampling_status": region.sampling_status,
                "model": args.model,
                "_roi_xyxy": region.roi_xyxy,
                "_visible_mask": region.visible_mask,
                "_instance_mask": target.instance_mask,
            }
        )

    overlay_path = output_dir / "overlays" / f"{safe_stem(relative_name)}_overlay.jpg"
    draw_overlay(image, rows, overlay_path)
    if args.save_depth_vis:
        depth_vis_path = output_dir / "depth_vis" / f"{safe_stem(relative_name)}_depth.jpg"
        save_depth_visualization(depth_map, depth_vis_path)
    if args.save_depth_npy:
        npy_path = output_dir / "depth_npy" / f"{safe_stem(relative_name)}_depth.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, depth_map.astype(np.float32))
    return rows


def draw_overlay(image_bgr: np.ndarray, rows: list[dict[str, object]], output_path: Path) -> None:
    canvas = image_bgr.copy()
    for row in rows:
        mask = row.get("_instance_mask")
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            overlay = canvas.copy()
            overlay[mask_array] = (0, 165, 255)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)

        x1 = int(float(row["bbox_x1"]))
        y1 = int(float(row["bbox_y1"]))
        x2 = int(float(row["bbox_x2"]))
        y2 = int(float(row["bbox_y2"]))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 255), 2)
        draw_sampling_contour(canvas, row)

        # Keep deployment overlays visually clean: only show the estimated distance.
        depth_text = row.get("pred_depth_m") or ""
        text = f"{depth_text}m" if depth_text else ""
        if not text:
            continue
        y_text = max(18, y1 - 6)
        cv2.putText(canvas, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
        cv2.putText(canvas, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def draw_sampling_contour(canvas: np.ndarray, row: dict[str, object]) -> None:
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
    cv2.drawContours(canvas, contours, -1, (255, 255, 0), 1)


def save_depth_visualization(depth_map: np.ndarray, output_path: Path) -> None:
    valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid.size == 0:
        return
    low, high = np.percentile(valid, [2, 98])
    normalized = np.clip((depth_map - low) / max(high - low, 1e-6), 0, 1)
    image = (255 * (1.0 - normalized)).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "image",
        "object_index",
        "class_name",
        "seg_conf",
        "pred_depth_m",
        "pred_depth_p10_m",
        "pred_depth_p50_m",
        "pred_depth_p90_m",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "visible_roi_ratio",
        "occluder_overlap_ratio",
        "foreground_roi_ratio",
        "background_removed_ratio",
        "instance_mask_pixels",
        "roi_pixels",
        "visible_pixels",
        "sampling_status",
        "model",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(rows: list[dict[str, object]], image_count: int, args: argparse.Namespace) -> dict[str, object]:
    valid_depths = [finite_float(row.get("pred_depth_m")) for row in rows]
    valid_depths = [value for value in valid_depths if value is not None]
    return {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(resolve_output_dir(args).resolve()),
        "image_count": image_count,
        "vehicle_count": len(rows),
        "vehicles_with_depth": len(valid_depths),
        "model": args.model,
        "vehicle_classes": list(args.vehicle_classes),
        "yolo_seg_model": args.yolo_seg_model,
        "yolo_seg_conf": args.yolo_seg_conf,
        "yolo_seg_imgsz": args.yolo_seg_imgsz,
        "yolo_seg_max_count": normalize_max_count(args.yolo_seg_max_count),
        "roi_mode": args.roi,
        "focal_length_px": args.focal_length_px,
        "focal_ratio": args.focal_ratio,
        "depth_min_m": min(valid_depths) if valid_depths else None,
        "depth_median_m": float(np.median(valid_depths)) if valid_depths else None,
        "depth_max_m": max(valid_depths) if valid_depths else None,
        "note": "Unlabeled inference only; no ground-truth accuracy metrics are computed.",
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.input_dir) / "output"


def safe_stem(relative_name: str) -> str:
    return Path(relative_name).with_suffix("").as_posix().replace("/", "__").replace("\\", "__")


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: object, digits: int = 4) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def parse_vehicle_classes(value: str) -> tuple[str, ...]:
    """Parse a comma-separated COCO vehicle class allowlist."""

    selected = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(COCO_VEHICLE_CLASS_NAMES))
    if unknown:
        valid = ",".join(COCO_VEHICLE_CLASS_NAMES)
        raise argparse.ArgumentTypeError(f"unknown vehicle class {unknown}; valid values: {valid}")
    if not selected:
        raise argparse.ArgumentTypeError("at least one vehicle class is required")
    return selected


def normalize_max_count(value: int | None) -> int | None:
    """把命令行里的 0 转成无限制，正整数则作为最大保留实例数。"""

    if value is None or value <= 0:
        return None
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate vehicle distances for an unlabeled image folder.")
    parser.add_argument("input_dir", help="Folder containing images.")
    parser.add_argument("--output-dir", default=None, help="Output folder. Defaults to <input_dir>/output.")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit for smoke tests.")
    parser.add_argument("--model", default="metric3d", choices=("depth-anything-v2-hf", "metric3d", "unidepth"))
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL, help="HuggingFace model id for Depth Anything V2.")
    parser.add_argument("--hf-offline", action="store_true", help="Load HuggingFace assets from local cache only.")
    parser.add_argument("--metric3d-model", default="metric3d_vit_small", help="Metric3D Torch Hub entrypoint.")
    parser.add_argument("--metric3d-offline", action="store_true", help="Use only cached Metric3D assets.")
    parser.add_argument("--unidepth-version", default="v2", choices=("v1", "v2", "v2old"))
    parser.add_argument("--unidepth-backbone", default="vits14")
    parser.add_argument("--unidepth-offline", action="store_true", help="Use only cached UniDepth assets.")
    parser.add_argument("--device", default="auto", help="Depth model device: auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--yolo-seg-model", default="yolo11m-seg.pt")
    parser.add_argument("--yolo-seg-conf", type=float, default=0.25)
    parser.add_argument("--yolo-seg-imgsz", type=int, default=None, help="YOLO segmentation inference size; larger helps small distant vehicles.")
    parser.add_argument("--yolo-seg-max-count", type=int, default=50, help="Max kept instances after dedupe; 0 means no limit.")
    parser.add_argument(
        "--vehicle-classes",
        type=parse_vehicle_classes,
        default=COCO_VEHICLE_CLASS_NAMES,
        help="Comma-separated COCO vehicle classes to keep: bicycle,car,motorcycle,bus,truck.",
    )
    parser.add_argument("--roi", default="lower-half", choices=("lower-half", "center", "full"))
    parser.add_argument("--center-ratio", type=float, default=0.5)
    parser.add_argument("--roi-inset-ratio", type=float, default=0.06)
    parser.add_argument("--depth-quantile-low", type=float, default=10.0)
    parser.add_argument("--depth-quantile-high", type=float, default=90.0)
    parser.add_argument("--min-visible-ratio", type=float, default=0.05)
    parser.add_argument("--disable-heuristic-occlusion", action="store_true")
    parser.add_argument("--focal-length-px", type=float, default=None, help="Known focal length in pixels.")
    parser.add_argument("--focal-ratio", type=float, default=0.58, help="Fallback focal length as ratio of image width.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x; defaults to image center.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y; defaults to image center.")
    parser.add_argument("--save-depth-vis", action="store_true", help="Save colored depth maps.")
    parser.add_argument("--save-depth-npy", action="store_true", help="Save raw depth maps as .npy.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = resolve_output_dir(args)
    images = discover_images(input_dir, recursive=args.recursive, output_dir=output_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    model = build_depth_model(args)
    all_rows: list[dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        relative_name = image_path.relative_to(input_dir).as_posix()
        print(f"[{index}/{len(images)}] {relative_name}")
        all_rows.extend(
            process_image(
                image_path,
                model=model,
                args=args,
                output_dir=output_dir,
                relative_name=relative_name,
            )
        )

    write_csv(all_rows, output_dir / "predictions.csv")
    summary = summarize(all_rows, len(images), args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
