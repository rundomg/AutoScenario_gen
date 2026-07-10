"""ROI mask construction and robust depth sampling.

This module keeps the messy geometry out of the evaluator. The important rule
is that occlusion decisions use annotation geometry and label depth, never the
model prediction being evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamplingRegion:
    """A local boolean mask inside an image-space ROI.

    `roi_pixels` is measured after the configured ROI mode and inward inset.
    `visible_pixels` is the remaining area after subtracting closer overlapping
    vehicles. The ratio fields make low-quality samples visible in the CSV
    instead of silently dropping them.
    """

    roi_xyxy: tuple[int, int, int, int] | None
    visible_mask: np.ndarray | None
    roi_pixels: int
    visible_pixels: int
    visible_roi_ratio: float | None
    occluder_overlap_ratio: float | None
    foreground_roi_ratio: float | None
    background_removed_ratio: float | None
    sampling_status: str


@dataclass(frozen=True)
class DepthStats:
    """Robust depth statistics sampled from a visible vehicle mask."""

    median_m: float | None
    p10_m: float | None
    p50_m: float | None
    p90_m: float | None


def clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    height, width = image_shape
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2


def make_base_roi(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    *,
    mode: str,
    center_ratio: float,
    inset_ratio: float,
) -> tuple[int, int, int, int] | None:
    """Create the initial rectangular ROI before overlap masking.

    The inward inset is deliberately applied after the ROI mode so that it
    trims background from the actual sampling region, not from unused parts of
    the vehicle box.
    """

    clamped = clamp_bbox(bbox, image_shape)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    width = x2 - x1
    height = y2 - y1

    if mode == "lower-half":
        y1 = y1 + height * 0.5
    elif mode == "center":
        ratio = max(0.05, min(1.0, center_ratio))
        pad_x = width * (1.0 - ratio) * 0.5
        pad_y = height * (1.0 - ratio) * 0.5
        x1 += pad_x
        x2 -= pad_x
        y1 += pad_y
        y2 -= pad_y
    elif mode != "full":
        raise ValueError(f"Unsupported ROI mode: {mode}")

    roi_width = x2 - x1
    roi_height = y2 - y1
    inset = max(0.0, min(0.45, inset_ratio))
    x1 += roi_width * inset
    x2 -= roi_width * inset
    y1 += roi_height * inset
    y2 -= roi_height * inset

    ix1 = int(math.floor(x1))
    iy1 = int(math.floor(y1))
    ix2 = int(math.ceil(x2))
    iy2 = int(math.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def build_sampling_region(
    target: Any,
    targets: list[Any],
    image_shape: tuple[int, int],
    *,
    mode: str,
    center_ratio: float,
    inset_ratio: float,
    min_visible_ratio: float,
    disable_overlap_mask: bool,
) -> SamplingRegion:
    """Build a visible sampling mask for one target.

    A vehicle in front can contaminate the target box with nearer pixels. We
    subtract only boxes whose matched KITTI label depth is closer than the
    current target, which prevents the model prediction from influencing the
    evaluation protocol.
    """

    roi = make_base_roi(
        target.bbox_xyxy,
        image_shape,
        mode=mode,
        center_ratio=center_ratio,
        inset_ratio=inset_ratio,
    )
    if roi is None:
        return SamplingRegion(None, None, 0, 0, None, None, None, None, "empty_roi")

    x1, y1, x2, y2 = roi
    mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
    roi_pixels = int(mask.size)
    foreground_removed_pixels = 0
    target_depth = _target_depth(target)

    target_mask = _instance_mask(target)
    if target_mask is not None:
        roi_instance_mask = _mask_roi(target_mask, roi)
        # Instance masks remove background pixels inside the detection box before
        # any occluder subtraction. This is the main improvement over bbox-only
        # sampling for parked cars, thin vehicles, and loose detector boxes.
        mask &= roi_instance_mask
        foreground_removed_pixels = roi_pixels - int(mask.sum())

    if not disable_overlap_mask and target_depth is not None:
        for other in targets:
            if other is target:
                continue
            other_depth = _target_depth(other)
            if other_depth is None or other_depth >= target_depth - 1e-3:
                continue
            other_mask = _instance_mask(other)
            if other_mask is not None:
                _subtract_mask_overlap(mask, roi, other_mask)
            else:
                _subtract_bbox_overlap(mask, roi, other.bbox_xyxy)

    visible_pixels = int(mask.sum())
    overlap_removed_pixels = max(0, roi_pixels - foreground_removed_pixels - visible_pixels)
    visible_ratio = visible_pixels / roi_pixels if roi_pixels else None
    overlap_ratio = overlap_removed_pixels / roi_pixels if roi_pixels else None
    foreground_ratio = (roi_pixels - foreground_removed_pixels) / roi_pixels if roi_pixels else None
    background_removed_ratio = foreground_removed_pixels / roi_pixels if roi_pixels else None
    if visible_pixels == 0:
        status = "no_visible_pixels"
    elif visible_ratio is not None and visible_ratio < min_visible_ratio:
        status = "low_visible_area"
    else:
        status = "ok"

    return SamplingRegion(
        roi,
        mask,
        roi_pixels,
        visible_pixels,
        visible_ratio,
        overlap_ratio,
        foreground_ratio,
        background_removed_ratio,
        status,
    )


def depth_stats_in_region(
    depth_map: np.ndarray | None,
    region: SamplingRegion,
    *,
    quantile_low: float,
    quantile_high: float,
) -> DepthStats:
    """Sample prediction depth from the visible mask with quantile trimming."""

    if depth_map is None or region.roi_xyxy is None or region.visible_mask is None:
        return DepthStats(None, None, None, None)
    if region.visible_pixels <= 0:
        return DepthStats(None, None, None, None)

    x1, y1, x2, y2 = region.roi_xyxy
    values = depth_map[y1:y2, x1:x2][region.visible_mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return DepthStats(None, None, None, None)

    low_q = max(0.0, min(100.0, quantile_low))
    high_q = max(low_q, min(100.0, quantile_high))
    p10 = float(np.percentile(values, low_q))
    p90 = float(np.percentile(values, high_q))
    clipped = values[(values >= p10) & (values <= p90)]
    if clipped.size == 0:
        clipped = values
    median = float(np.median(clipped))
    return DepthStats(median, p10, median, p90)


def lidar_depth_in_region(
    projected_lidar: tuple[np.ndarray, np.ndarray, np.ndarray],
    region: SamplingRegion,
) -> tuple[float | None, int]:
    """Sample projected LiDAR points only where the visible mask remains true."""

    if region.roi_xyxy is None or region.visible_mask is None or region.visible_pixels <= 0:
        return None, 0
    u, v, z = projected_lidar
    if z.size == 0:
        return None, 0

    x1, y1, x2, y2 = region.roi_xyxy
    in_roi = (u >= x1) & (u < x2) & (v >= y1) & (v < y2)
    if not np.any(in_roi):
        return None, 0

    local_x = np.floor(u[in_roi] - x1).astype(np.int64)
    local_y = np.floor(v[in_roi] - y1).astype(np.int64)
    local_x = np.clip(local_x, 0, region.visible_mask.shape[1] - 1)
    local_y = np.clip(local_y, 0, region.visible_mask.shape[0] - 1)
    visible = region.visible_mask[local_y, local_x]
    values = z[in_roi][visible]
    if values.size == 0:
        return None, 0
    return float(np.median(values)), int(values.size)


def _target_depth(target: Any) -> float | None:
    depth = getattr(getattr(target, "label", None), "label_depth_m", None)
    if depth is None:
        return None
    try:
        number = float(depth)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _instance_mask(target: Any) -> np.ndarray | None:
    mask = getattr(target, "instance_mask", None)
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or mask_array.size == 0:
        return None
    return mask_array


def _mask_roi(mask: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi
    height, width = mask.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(width, x2), min(height, y2)
    output = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    if sx2 <= sx1 or sy2 <= sy1:
        return output
    output[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = mask[sy1:sy2, sx1:sx2]
    return output


def _subtract_mask_overlap(mask: np.ndarray, roi: tuple[int, int, int, int], occluder_mask: np.ndarray) -> None:
    overlap = _mask_roi(occluder_mask, roi)
    if overlap.size == 0:
        return
    mask[overlap] = False


def _subtract_bbox_overlap(
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
    occluder_bbox: tuple[float, float, float, float],
) -> None:
    rx1, ry1, rx2, ry2 = roi
    ox1, oy1, ox2, oy2 = occluder_bbox
    ix1 = max(rx1, int(math.floor(ox1)))
    iy1 = max(ry1, int(math.floor(oy1)))
    ix2 = min(rx2, int(math.ceil(ox2)))
    iy2 = min(ry2, int(math.ceil(oy2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return
    mask[iy1 - ry1 : iy2 - ry1, ix1 - rx1 : ix2 - rx1] = False
