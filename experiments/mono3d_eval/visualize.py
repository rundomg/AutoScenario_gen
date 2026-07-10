"""MonoDETR 预测结果可视化。

输出两类图：
1. 原图上的 2D bbox 和投影 3D bbox；
2. BEV 俯视图，展示车辆位置和朝向箭头。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .kitti_3d import Kitti3DObject, object_corners_camera, project_corners_to_image, read_calib_p2
except ImportError:  # 允许直接 python experiments/.../visualize.py 调试
    from kitti_3d import Kitti3DObject, object_corners_camera, project_corners_to_image, read_calib_p2


BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def draw_frame_overlay(
    image_path: Path,
    calib_path: Path,
    predictions: list[Kitti3DObject],
    output_path: Path,
    *,
    score_threshold: float,
) -> None:
    """在原图上画 2D 框、3D 投影框、朝向和距离。"""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    p2 = read_calib_p2(calib_path)

    for obj in predictions:
        if obj.score is not None and obj.score < score_threshold:
            continue
        _draw_2d_box(image, obj)
        _draw_projected_3d_box(image, obj, p2)
        _draw_label(image, obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def draw_bev(
    predictions: list[Kitti3DObject],
    output_path: Path,
    *,
    score_threshold: float,
    width: int = 900,
    height: int = 900,
    x_range: tuple[float, float] = (-35.0, 35.0),
    z_range: tuple[float, float] = (0.0, 90.0),
) -> None:
    """画简化 BEV：横轴为相机 x，纵轴为相机 z，箭头表示 rotation_y。"""

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    _draw_bev_grid(canvas, x_range, z_range)

    for obj in predictions:
        if obj.score is not None and obj.score < score_threshold:
            continue
        corners = object_corners_camera(obj)
        # 底面四个角点是 0..3，只取 x/z 画俯视框。
        bottom = corners[:4, :]
        points = np.array([_bev_to_pixel(x, z, width, height, x_range, z_range) for x, _y, z in bottom], dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 120, 255), thickness=2)
        _draw_bev_heading(canvas, obj, width, height, x_range, z_range)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def _draw_2d_box(image: np.ndarray, obj: Kitti3DObject) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox_xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)


def _draw_projected_3d_box(image: np.ndarray, obj: Kitti3DObject, p2: np.ndarray) -> None:
    corners = object_corners_camera(obj)
    points = project_corners_to_image(corners, p2)
    if points is None:
        return
    points_i = np.round(points).astype(np.int32)
    for start, end in BOX_EDGES:
        cv2.line(image, tuple(points_i[start]), tuple(points_i[end]), (255, 170, 0), 2)


def _draw_label(image: np.ndarray, obj: Kitti3DObject) -> None:
    x1, y1, _x2, _y2 = (int(round(value)) for value in obj.bbox_xyxy)
    score = "" if obj.score is None else f" {obj.score:.2f}"
    text = f"{obj.object_type}{score} z={obj.depth_m:.1f} yaw={obj.rotation_y:.2f}"
    y = max(18, y1 - 6)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def _draw_bev_grid(canvas: np.ndarray, x_range: tuple[float, float], z_range: tuple[float, float]) -> None:
    height, width = canvas.shape[:2]
    for z in range(int(z_range[0]), int(z_range[1]) + 1, 10):
        y = _bev_to_pixel(0.0, float(z), width, height, x_range, z_range)[1]
        cv2.line(canvas, (0, y), (width, y), (220, 220, 220), 1)
        cv2.putText(canvas, f"{z}m", (8, max(16, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    x_zero, _ = _bev_to_pixel(0.0, 0.0, width, height, x_range, z_range)
    cv2.line(canvas, (x_zero, 0), (x_zero, height), (170, 170, 170), 1)
    cv2.putText(canvas, "camera x/z BEV", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)


def _draw_bev_heading(
    canvas: np.ndarray,
    obj: Kitti3DObject,
    width: int,
    height: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    x, _y, z = obj.location_xyz
    start = _bev_to_pixel(x, z, width, height, x_range, z_range)
    # KITTI rotation_y 绕相机 y 轴，方向向量可用 sin/cos 投到 x-z 平面。
    arrow_len = max(2.0, obj.dimensions_hwl[2] * 0.8)
    end_x = x + math.sin(obj.rotation_y) * arrow_len
    end_z = z + math.cos(obj.rotation_y) * arrow_len
    end = _bev_to_pixel(end_x, end_z, width, height, x_range, z_range)
    cv2.arrowedLine(canvas, start, end, (20, 80, 220), 2, tipLength=0.25)
    cv2.circle(canvas, start, 3, (20, 80, 220), -1)


def _bev_to_pixel(
    x: float,
    z: float,
    width: int,
    height: int,
    x_range: tuple[float, float],
    z_range: tuple[float, float],
) -> tuple[int, int]:
    x_min, x_max = x_range
    z_min, z_max = z_range
    px = int(round((x - x_min) / max(x_max - x_min, 1e-6) * (width - 1)))
    py = int(round((1.0 - (z - z_min) / max(z_max - z_min, 1e-6)) * (height - 1)))
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))

