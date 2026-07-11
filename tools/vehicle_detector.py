"""Vehicle detector that feeds reliable 2D bounding boxes into the VLM scene
understanding stage.

Uses a standard COCO-pretrained ultralytics YOLO model (default ``yolo11m.pt``).
All target categories — car, truck, bus, motorcycle, bicycle — live in COCO, so
open-vocabulary detection (YOLO-World) is not required here.

The detector is optional: if ultralytics / the model weights are unavailable the
``detect_vehicles`` call returns an empty list and the scene understanding stage
falls back to the original image-only flow instead of crashing.
"""

import os
import threading
from copy import deepcopy

import cv2


# COCO class id -> our DSL category. Matches `traffic_subjects[].category`.
_COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolo11m.pt")
_DEFAULT_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
_FALLBACK_CONF = float(os.environ.get("YOLO_FALLBACK_CONF", "0.15"))

_model = None
_model_lock = threading.Lock()
_load_failed = False


def _get_model():
    """Lazy-load a single shared YOLO model. Returns None if unavailable."""
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

            _model = YOLO(_DEFAULT_MODEL)
        except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
            print(
                f"[vehicle_detector] YOLO unavailable, skipping detection "
                f"(install with `pip install ultralytics`). Reason: {exc}"
            )
            _load_failed = True
            return None
    return _model


def detect_vehicles(image_path: str, conf: float = _DEFAULT_CONF) -> list:
    """Detect vehicles in an image.

    Returns a list of dicts with normalized [0,1] coordinates (origin = top-left):
        {
          "id": "det_1",
          "label": "car",
          "conf": 0.91,
          "bbox_xyxy": [x1, y1, x2, y2],      # pixels
          "bbox_norm": [x1, y1, x2, y2],      # normalized [0,1]
          "center_norm": [cx, cy],            # normalized [0,1]
        }
    Returns [] when the detector is unavailable or finds nothing.
    """
    model = _get_model()
    if model is None:
        return []

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    height, width = image.shape[:2]

    try:
        results = model.predict(
            image_path, conf=conf, classes=list(_COCO_VEHICLE_CLASSES.keys()),
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[vehicle_detector] detection failed, skipping: {exc}")
        return []

    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label = _COCO_VEHICLE_CLASSES.get(cls_id)
            if label is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            detections.append(
                {
                    "label": label,
                    "conf": round(confidence, 3),
                    "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "bbox_norm": [
                        round(x1 / width, 4),
                        round(y1 / height, 4),
                        round(x2 / width, 4),
                        round(y2 / height, 4),
                    ],
                    "center_norm": [
                        round((x1 + x2) / 2 / width, 4),
                        round((y1 + y2) / 2 / height, 4),
                    ],
                }
            )

    # Stable ids, largest (closest/most salient) boxes first.
    detections.sort(
        key=lambda d: (d["bbox_norm"][2] - d["bbox_norm"][0])
        * (d["bbox_norm"][3] - d["bbox_norm"][1]),
        reverse=True,
    )
    for index, det in enumerate(detections, start=1):
        det["id"] = f"det_{index}"
    return detections


def detect_vehicles_balanced(
    image_path: str,
    *,
    primary_conf: float = _DEFAULT_CONF,
    fallback_conf: float = _FALLBACK_CONF,
) -> tuple[list, dict]:
    """Run a single lower-threshold retry only when the primary pass is empty."""
    detections = detect_vehicles(image_path, primary_conf)
    used_conf = float(primary_conf)
    fallback_used = False
    if not detections and fallback_conf < primary_conf:
        detections = detect_vehicles(image_path, fallback_conf)
        used_conf = float(fallback_conf)
        fallback_used = True
    return detections, {
        "primary_conf": float(primary_conf),
        "fallback_conf": float(fallback_conf),
        "fallback_used": fallback_used,
        "used_conf": used_conf,
        "raw_detection_count": len(detections),
        "status": "detected" if detections else "empty_after_fallback",
    }


def _bbox_area(box: list) -> float:
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_iou(a: list, b: list) -> float:
    inter = _bbox_intersection(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def select_representative_detections(
    detections: list,
    *,
    max_count: int = 8,
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.75,
) -> list:
    """Deduplicate raw detector boxes and keep representative vehicles.

    Two boxes are considered the same vehicle when their IoU is high or when
    the smaller box is mostly contained in the larger one. Representatives are
    selected by ``area * confidence`` and then re-numbered for the VLM prompt.
    """
    if not detections:
        return []

    ranked = []
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        area = _bbox_area(bbox)
        try:
            conf = float(det.get("conf", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        ranked.append((_bbox_area(bbox) * max(conf, 0.01), area, det))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = []
    for _score, _area, det in ranked:
        bbox = det["bbox_xyxy"]
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
        selected.append(deepcopy(det))
        if len(selected) >= max_count:
            break

    selected.sort(
        key=lambda d: (d["bbox_norm"][2] - d["bbox_norm"][0])
        * (d["bbox_norm"][3] - d["bbox_norm"][1]),
        reverse=True,
    )
    for index, det in enumerate(selected, start=1):
        det["id"] = f"det_{index}"
    return selected


def write_detection_crops(
    image_path: str,
    detections: list,
    *,
    output_dir: str | None = None,
    base_name: str | None = None,
    padding_ratio: float = 0.35,
    min_short_side: int = 384,
) -> list:
    """Write padded vehicle crops and return detections with ``crop_path``."""
    if not detections:
        return []
    image = cv2.imread(image_path)
    if image is None:
        return []
    height, width = image.shape[:2]
    if output_dir is None:
        output_dir = os.path.dirname(image_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]

    updated = []
    for det in detections:
        bbox = det.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x, pad_y = bw * padding_ratio, bh * padding_ratio
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(width, int(round(x2 + pad_x)))
        cy2 = min(height, int(round(y2 + pad_y)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = image[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        short_side = min(crop_w, crop_h)
        if short_side > 0 and short_side < min_short_side:
            scale = float(min_short_side) / float(short_side)
            crop = cv2.resize(
                crop,
                (int(round(crop_w * scale)), int(round(crop_h * scale))),
                interpolation=getattr(cv2, "INTER_CUBIC", 2),
            )
        crop_path = os.path.join(output_dir, f"{base_name}_{det['id']}_crop.jpg")
        if not cv2.imwrite(crop_path, crop):
            continue
        copied = deepcopy(det)
        copied["crop_path"] = crop_path
        copied["crop_bbox_xyxy"] = [cx1, cy1, cx2, cy2]
        updated.append(copied)
    return updated


def infer_row_group_hints(detections: list) -> list:
    """Return lightweight detector-only row candidates for the VLM.

    These are hints, not final semantics. The SU VLM decides whether a row is a
    traffic queue, parking row, or just unrelated nearby vehicles.
    """
    if len(detections) < 2:
        return []
    items = []
    for det in detections:
        bbox = det.get("bbox_norm") or []
        center = det.get("center_norm") or []
        if len(bbox) != 4 or len(center) != 2:
            continue
        items.append({"id": det.get("id"), "bbox": bbox, "cx": center[0], "cy": center[1]})
    if len(items) < 2:
        return []

    hints = []
    by_y = sorted(items, key=lambda item: item["cy"])
    horizontal_groups = []
    current = [by_y[0]]
    for item in by_y[1:]:
        if abs(item["cy"] - current[-1]["cy"]) <= 0.12:
            current.append(item)
        else:
            if len(current) >= 2:
                horizontal_groups.append(current)
            current = [item]
    if len(current) >= 2:
        horizontal_groups.append(current)
    for index, group in enumerate(horizontal_groups, start=1):
        ordered = sorted(group, key=lambda item: item["cx"])
        hints.append(
            {
                "id": f"row_hint_{index}",
                "det_ids": [item["id"] for item in ordered],
                "alignment": "horizontal_image_row",
                "order_rule_hint": "image_left_to_right",
            }
        )

    by_x = sorted(items, key=lambda item: item["cx"])
    vertical_groups = []
    current = [by_x[0]]
    for item in by_x[1:]:
        if abs(item["cx"] - current[-1]["cx"]) <= 0.10:
            current.append(item)
        else:
            if len(current) >= 2:
                vertical_groups.append(current)
            current = [item]
    if len(current) >= 2:
        vertical_groups.append(current)
    offset = len(hints)
    for index, group in enumerate(vertical_groups, start=1):
        ordered = sorted(group, key=lambda item: item["cy"], reverse=True)
        hints.append(
            {
                "id": f"row_hint_{offset + index}",
                "det_ids": [item["id"] for item in ordered],
                "alignment": "same_image_column",
                "order_rule_hint": "image_bottom_to_top",
            }
        )
    return hints


def annotate_image(image_path: str, detections: list, out_path: str) -> str | None:
    """Draw labeled detection boxes onto a copy of the image. Returns out_path,
    or None if nothing was drawn / the image could not be loaded."""
    if not detections:
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None

    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["bbox_xyxy"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = f"{det['id']} {det['label']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 2, y1), (0, 255, 0), -1)
        cv2.putText(
            image, tag, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, image)
    return out_path


def format_detections_for_prompt(detections: list) -> str:
    """Render detections as a compact text block for the VLM prompt."""
    if not detections:
        return ""
    lines = []
    for d in detections:
        line = (
            f"- {d['id']} {d['label']} conf={d['conf']} "
            f"bbox={d['bbox_norm']} center={d['center_norm']}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_row_hints_for_prompt(row_hints: list) -> str:
    if not row_hints:
        return ""
    lines = []
    for hint in row_hints:
        lines.append(
            f"- {hint['id']} det_ids={hint['det_ids']} "
            f"alignment={hint['alignment']} order_hint={hint['order_rule_hint']}"
        )
    return "\n".join(lines)
