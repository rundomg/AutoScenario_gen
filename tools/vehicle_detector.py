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
        if "depth_m" in det:
            tag += f" {det['depth_m']}m"
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
        if "depth_m" in d:
            line += f" depth≈{d['depth_m']}m"
        lines.append(line)
    return "\n".join(lines)
