"""Monocular metric-depth estimation to enrich vehicle detections with a coarse
distance, so the VLM can reason about longitudinal (front/back) ordering from a
real depth value instead of "how low the box sits in the image".

Backed by Depth Anything V2 (metric, outdoor/driving variant) via the
HuggingFace `transformers` depth-estimation pipeline. Like `vehicle_detector`,
this is fully optional: if transformers / weights are unavailable, every call
degrades gracefully (depth is simply omitted) instead of breaking the pipeline.

Convention: returned depth maps and `depth_m` are in metres, larger = farther.
"""

import os
import threading


# Outdoor/driving metric model (metres). Small variant is CPU-friendly; switch
# to `...-Outdoor-Base-hf` or `...-Outdoor-Large-hf` via DEPTH_MODEL for accuracy.
_DEFAULT_MODEL = os.environ.get(
    "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
)

_pipe = None
_pipe_lock = threading.Lock()
_load_failed = False


def _get_pipe():
    """Lazy-load a single shared depth pipeline. Returns None if unavailable."""
    global _pipe, _load_failed
    if _pipe is not None:
        return _pipe
    if _load_failed:
        return None
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if _load_failed:
            return None
        try:
            from transformers import pipeline

            _pipe = pipeline("depth-estimation", model=_DEFAULT_MODEL)
        except Exception as exc:  # noqa: BLE001 - any failure -> graceful fallback
            print(
                f"[depth_estimator] depth model unavailable, skipping depth "
                f"(install with `pip install transformers`, or set DEPTH_MODEL). "
                f"Reason: {exc}"
            )
            _load_failed = True
            return None
    return _pipe


def estimate_depth(image_path: str):
    """Return a metric depth map (numpy HxW, metres, larger = farther) aligned to
    the original image size, or None if depth is unavailable."""
    pipe = _get_pipe()
    if pipe is None:
        return None
    try:
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        out = pipe(image)
        pred = out.get("predicted_depth")
        if pred is None:
            return None
        tensor = pred.float()
        while tensor.ndim < 4:  # -> (1, 1, h, w)
            tensor = tensor.unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(height, width), mode="bilinear", align_corners=False
        )
        return tensor[0, 0].cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"[depth_estimator] depth estimation failed, skipping: {exc}")
        return None


def _sample_depth(depth_map, cx: int, cy: int, half: int = 3):
    """Median depth in a small window around (cx, cy), robust to outliers."""
    import numpy as np

    height, width = depth_map.shape
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    x0, x1 = max(0, cx - half), min(width, cx + half + 1)
    y0, y1 = max(0, cy - half), min(height, cy + half + 1)
    window = depth_map[y0:y1, x0:x1]
    if window.size == 0:
        return None
    return float(np.median(window))


def attach_depth(image_path: str, detections: list) -> list:
    """Add a coarse `depth_m` (and `depth_rank`, 1 = nearest) to each detection,
    sampled at the vehicle's ground-contact point (bbox bottom-centre).

    Mutates and returns `detections`. On any failure the detections are returned
    unchanged (no depth fields), so callers can stay depth-agnostic.
    """
    if not detections:
        return detections
    depth_map = estimate_depth(image_path)
    if depth_map is None:
        return detections

    for det in detections:
        x1, _, x2, y2 = det["bbox_xyxy"]
        # Ground-contact point: bottom-centre of the box, more depth-stable than
        # the box centre for a vehicle resting on the road surface.
        contact_x = int(round((x1 + x2) / 2))
        contact_y = int(round(y2))
        value = _sample_depth(depth_map, contact_x, contact_y)
        if value is not None:
            det["depth_m"] = round(value, 1)

    ranked = sorted(
        (d for d in detections if "depth_m" in d), key=lambda d: d["depth_m"]
    )
    for rank, det in enumerate(ranked, start=1):
        det["depth_rank"] = rank
    return detections
