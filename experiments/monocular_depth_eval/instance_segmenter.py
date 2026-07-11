"""Vehicle instance segmentation for KITTI distance evaluation.

The evaluator needs per-vehicle masks, not a semantic "all cars" mask. This
module wraps Ultralytics YOLO segmentation and returns full-image boolean masks
aligned to the input image, keeping model-specific details out of the evaluator.
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_CONFIG_DIR = Path(__file__).resolve().parent / "results" / "config" / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

# COCO vehicle classes supported by YOLO segmentation models.
_COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_DEFAULT_MODEL = os.environ.get("YOLO_SEG_MODEL", "yolo11m-seg.pt")
_DEFAULT_CONF = float(os.environ.get("YOLO_SEG_CONF", "0.25"))
_DEFAULT_IMGSZ = int(os.environ["YOLO_SEG_IMGSZ"]) if os.environ.get("YOLO_SEG_IMGSZ") else None
_DEFAULT_MAX_COUNT = int(os.environ.get("YOLO_SEG_MAX_COUNT", "12"))

_model = None
_model_lock = threading.Lock()
_load_failed = False


def _get_model(model_path: str = _DEFAULT_MODEL):
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from ultralytics import YOLO

            _model = YOLO(model_path)
        except Exception as exc:  # noqa: BLE001 - caller can fall back cleanly
            print(
                "[instance_segmenter] YOLO segmentation unavailable "
                f"(model={model_path}). Reason: {exc}"
            )
            _load_failed = True
            return None
    return _model


def segment_vehicles(
    image_path: str | Path,
    *,
    conf: float = _DEFAULT_CONF,
    model_path: str = _DEFAULT_MODEL,
    imgsz: int | None = _DEFAULT_IMGSZ,
    max_count: int | None = _DEFAULT_MAX_COUNT,
) -> list[dict]:
    """Return COCO vehicle instances with full-image boolean masks.

    Each result dict contains `bbox_xyxy`, `mask`, `mask_pixels`, `label`, and
    `conf`. Returning masks in image coordinates keeps ROI sampling simple and
    avoids leaking Ultralytics tensor layout assumptions into other modules.
    """

    model = _get_model(model_path)
    if model is None:
        return []

    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    height, width = image.shape[:2]

    try:
        # imgsz 为空时沿用 Ultralytics 默认尺寸；调大 imgsz 有利于远处小车召回。
        predict_kwargs = {
            "conf": conf,
            "classes": list(_COCO_VEHICLE_CLASSES.keys()),
            "verbose": False,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        results = model.predict(str(image_path), **predict_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[instance_segmenter] segmentation failed, skipping: {exc}")
        return []

    instances: list[dict] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or getattr(masks, "data", None) is None:
            continue
        mask_data = masks.data
        for index, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = _COCO_VEHICLE_CLASSES.get(cls_id)
            if label is None or index >= len(mask_data):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            mask = _to_image_mask(mask_data[index], height, width)
            if mask is None or int(mask.sum()) == 0:
                continue
            confidence = float(box.conf[0])
            instances.append(
                {
                    "label": label,
                    "conf": round(confidence, 3),
                    "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "mask": mask,
                    "mask_pixels": int(mask.sum()),
                }
            )

    # max_count 只限制去重后的代表实例；传 None 可保留全部实例。
    selected = select_representative_instances(instances, max_count=max_count)
    for index, instance in enumerate(selected, start=1):
        instance["id"] = f"seg_{index}"
    return selected


def select_representative_instances(
    instances: list[dict],
    *,
    max_count: int | None = _DEFAULT_MAX_COUNT,
    iou_threshold: float = 0.55,
    containment_threshold: float = 0.80,
) -> list[dict]:
    """Deduplicate segmentation instances using bbox overlap.

    YOLO's NMS usually handles this already, but this extra pass keeps the
    downstream KITTI matching stable when multiple masks describe the same car.
    """

    ranked = []
    for instance in instances:
        bbox = instance.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        area = _bbox_area(bbox)
        try:
            conf = float(instance.get("conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        ranked.append((area * max(conf, 0.01), area, instance))
    # 代表实例按面积和置信度排序；max_count 过小时会偏向近处大车。
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[dict] = []
    for _score, _area, instance in ranked:
        bbox = instance["bbox_xyxy"]
        duplicate = False
        for kept in selected:
            kept_bbox = kept["bbox_xyxy"]
            inter = _bbox_intersection(bbox, kept_bbox)
            smaller = min(_bbox_area(bbox), _bbox_area(kept_bbox))
            containment = inter / smaller if smaller > 0.0 else 0.0
            if _bbox_iou(bbox, kept_bbox) >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(deepcopy(instance))
        if max_count is not None and max_count > 0 and len(selected) >= max_count:
            break
    return selected


def _to_image_mask(mask_tensor, height: int, width: int) -> np.ndarray | None:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim != 2:
        return None
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def _bbox_area(box: list | tuple) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection(a: list | tuple, b: list | tuple) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_iou(a: list | tuple, b: list | tuple) -> float:
    inter = _bbox_intersection(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0
