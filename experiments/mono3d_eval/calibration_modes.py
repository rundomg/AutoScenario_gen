"""Calibration helpers for MonoCon ablation experiments.

Generated intrinsics live in each run directory, so the original KITTI
`training/calib` files stay untouched.  This lets us isolate how much MonoCon
depends on the camera matrix during inference and 3D decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationMode:
    """One reproducible camera-intrinsic assumption."""

    name: str
    description: str
    uses_real_kitti_calib: bool
    horizontal_fov_deg: float | None = None
    focal_ratio: float | None = None


CALIBRATION_MODES: dict[str, CalibrationMode] = {
    "true_k": CalibrationMode(
        name="true_k",
        description="Use KITTI training/calib P2 as provided by the dataset.",
        uses_real_kitti_calib=True,
    ),
    "fake_fov90": CalibrationMode(
        name="fake_fov90",
        description="Generated pinhole intrinsics with 90 degree horizontal FOV and centered principal point.",
        uses_real_kitti_calib=False,
        horizontal_fov_deg=90.0,
    ),
    "fake_ratio058": CalibrationMode(
        name="fake_ratio058",
        description="Generated pinhole intrinsics with fx=fy=0.58*image_width and centered principal point.",
        uses_real_kitti_calib=False,
        focal_ratio=0.58,
    ),
}


def parse_calibration_mode(name: str) -> CalibrationMode:
    """Resolve a calibration mode name used by command-line tools."""

    try:
        return CALIBRATION_MODES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(CALIBRATION_MODES))
        raise ValueError(f"Unsupported calibration mode {name!r}; valid modes: {valid}") from exc


def generated_focal_px(width: int, mode: CalibrationMode) -> float:
    """Compute fake focal length in pixels for one image width."""

    if mode.focal_ratio is not None:
        return float(width) * float(mode.focal_ratio)
    if mode.horizontal_fov_deg is None:
        raise ValueError(f"Calibration mode {mode.name} does not define fake intrinsics.")

    # fx = width / (2 * tan(hfov / 2)); clamping avoids pathological values.
    fov_rad = math.radians(max(20.0, min(140.0, float(mode.horizontal_fov_deg))))
    return float(width) / (2.0 * math.tan(fov_rad / 2.0))


def make_calibration_dict(width: int, height: int, focal_px: float) -> dict[str, np.ndarray]:
    """Create the minimal KITTI-style calibration layout MonoCon expects."""

    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    projection = np.array(
        [
            [focal_px, 0.0, cx, 0.0],
            [0.0, focal_px, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    identity_3x4 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return {
        "P0": projection.copy(),
        "P1": projection.copy(),
        "P2": projection.copy(),
        "P3": projection.copy(),
        "R0": np.eye(3, dtype=np.float32),
        "Tr_velo2cam": identity_3x4.copy(),
        "Tr_imu2velo": identity_3x4.copy(),
    }


def write_calibration_txt(calib: dict[str, np.ndarray], output_path: Path) -> None:
    """Write one KITTI calibration text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("P0", calib["P0"].reshape(-1)),
        ("P1", calib["P1"].reshape(-1)),
        ("P2", calib["P2"].reshape(-1)),
        ("P3", calib["P3"].reshape(-1)),
        ("R0_rect", calib["R0"].reshape(-1)),
        ("Tr_velo_to_cam", calib["Tr_velo2cam"].reshape(-1)),
        ("Tr_imu_to_velo", calib["Tr_imu2velo"].reshape(-1)),
    ]
    lines = [f"{name}: " + " ".join(f"{value:.12e}" for value in values) for name, values in rows]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_generated_calibrations(
    *,
    kitti_root: Path,
    split: str,
    frame_ids: Iterable[str],
    mode: CalibrationMode,
    output_dir: Path,
) -> dict[str, object]:
    """Generate fake calibration files and return metadata for summary.json."""

    if mode.uses_real_kitti_calib:
        return {
            "calibration_mode": mode.name,
            "calibration_description": mode.description,
            "generated_calib_dir": None,
            "generated_count": 0,
            "focal_px_min": None,
            "focal_px_max": None,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = kitti_root / split / "image_2"
    focal_values: list[float] = []
    count = 0
    for frame_id in frame_ids:
        image_path = image_dir / f"{frame_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read KITTI image for fake calibration: {image_path}")
        height, width = image.shape[:2]
        focal_px = generated_focal_px(width, mode)
        write_calibration_txt(make_calibration_dict(width, height, focal_px), output_dir / f"{frame_id}.txt")
        focal_values.append(focal_px)
        count += 1

    return {
        "calibration_mode": mode.name,
        "calibration_description": mode.description,
        "generated_calib_dir": str(output_dir),
        "generated_count": count,
        "focal_px_min": min(focal_values) if focal_values else None,
        "focal_px_max": max(focal_values) if focal_values else None,
    }
