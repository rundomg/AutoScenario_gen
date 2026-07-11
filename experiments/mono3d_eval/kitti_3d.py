"""KITTI 3D 标注/预测解析与几何工具。

MonoDETR 的输出通常沿用 KITTI detection 文本格式：
type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom
height width length x y z rotation_y score

这个模块只处理通用格式和几何计算，不依赖 MonoDETR 代码本体。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VEHICLE_CLASSES = {"Car", "Van", "Truck", "Tram"}


@dataclass(frozen=True)
class Kitti3DObject:
    """一条 KITTI 3D object 记录，既可表示真值，也可表示模型预测。"""

    frame_id: str
    object_type: str
    truncation: float
    occlusion: int
    alpha: float
    bbox_xyxy: tuple[float, float, float, float]
    dimensions_hwl: tuple[float, float, float]
    location_xyz: tuple[float, float, float]
    rotation_y: float
    score: float | None = None

    @property
    def is_vehicle(self) -> bool:
        return self.object_type in VEHICLE_CLASSES

    @property
    def depth_m(self) -> float:
        return float(self.location_xyz[2])

    @property
    def distance_m(self) -> float:
        x, y, z = self.location_xyz
        return float(math.sqrt(x * x + y * y + z * z))


@dataclass(frozen=True)
class MatchedObject:
    """模型预测与 KITTI 真值的一个匹配对。"""

    frame_id: str
    pred: Kitti3DObject
    label: Kitti3DObject
    iou_2d: float


def read_kitti_objects(path: Path, frame_id: str, *, keep_non_vehicle: bool = False) -> list[Kitti3DObject]:
    """读取 KITTI label 或 prediction 文件。

    预测文件最后可能带 score，真值文件通常没有 score；两种格式都支持。
    """

    if not path.exists():
        return []

    objects: list[Kitti3DObject] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 15:
            raise ValueError(f"KITTI line has fewer than 15 columns: {path}:{line_number}")

        object_type = parts[0]
        if not keep_non_vehicle and object_type not in VEHICLE_CLASSES:
            continue

        try:
            truncation = float(parts[1])
            occlusion = int(float(parts[2]))
            alpha = float(parts[3])
            bbox = tuple(float(value) for value in parts[4:8])
            dimensions = tuple(float(value) for value in parts[8:11])
            location = tuple(float(value) for value in parts[11:14])
            rotation_y = float(parts[14])
            score = float(parts[15]) if len(parts) > 15 else None
        except ValueError as exc:
            raise ValueError(f"Failed to parse KITTI line: {path}:{line_number}") from exc

        objects.append(
            Kitti3DObject(
                frame_id=frame_id,
                object_type=object_type,
                truncation=truncation,
                occlusion=occlusion,
                alpha=alpha,
                bbox_xyxy=bbox,  # type: ignore[arg-type]
                dimensions_hwl=dimensions,  # type: ignore[arg-type]
                location_xyz=location,  # type: ignore[arg-type]
                rotation_y=rotation_y,
                score=score,
            )
        )
    return objects


def read_calib_p2(calib_path: Path) -> np.ndarray:
    """读取 KITTI P2 投影矩阵，返回 3x4 numpy 数组。"""

    for raw_line in calib_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.startswith("P2:"):
            continue
        values = [float(value) for value in raw_line.split()[1:]]
        if len(values) != 12:
            raise ValueError(f"Invalid P2 calibration line: {calib_path}")
        return np.asarray(values, dtype=np.float64).reshape(3, 4)
    raise KeyError(f"P2 not found in calibration file: {calib_path}")


def object_corners_camera(obj: Kitti3DObject) -> np.ndarray:
    """计算 3D box 的 8 个角点，坐标系为 KITTI rect camera。

    KITTI 的 location 是 3D box 底面中心；y 轴向下，所以顶部角点 y=-h。
    """

    height, width, length = obj.dimensions_hwl
    x_corners = np.array([length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2])
    y_corners = np.array([0, 0, 0, 0, -height, -height, -height, -height])
    z_corners = np.array([width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2])

    cos_y = math.cos(obj.rotation_y)
    sin_y = math.sin(obj.rotation_y)
    rotation = np.array([[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]], dtype=np.float64)
    corners = rotation @ np.vstack([x_corners, y_corners, z_corners])
    location = np.asarray(obj.location_xyz, dtype=np.float64).reshape(3, 1)
    return (corners + location).T


def project_corners_to_image(corners_camera: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """把 3D box 角点投影到图像平面；若有角点在相机后方则返回 None。"""

    if np.any(corners_camera[:, 2] <= 0.1):
        return None
    homogeneous = np.hstack([corners_camera, np.ones((corners_camera.shape[0], 1), dtype=np.float64)])
    projected = (p2 @ homogeneous.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2]


def bbox_iou_2d(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """计算两个 2D xyxy 框的 IoU。"""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def angle_error_rad(pred: float, label: float) -> float:
    """返回 [-pi, pi] wrap 后的绝对角度误差。"""

    diff = (pred - label + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def match_predictions_to_labels(
    predictions: list[Kitti3DObject],
    labels: list[Kitti3DObject],
    *,
    iou_threshold: float,
) -> list[MatchedObject]:
    """按 2D IoU 贪心匹配预测与真值。

    第一轮只做轻量评估，先不用 3D IoU；这样即使 MonoDETR 输出只有 KITTI
    文本，也能快速检查距离和朝向质量。
    """

    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(predictions):
        for label_index, label in enumerate(labels):
            if pred.object_type != label.object_type and pred.object_type == "Car" and label.object_type not in VEHICLE_CLASSES:
                continue
            iou = bbox_iou_2d(pred.bbox_xyxy, label.bbox_xyxy)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, label_index))

    candidates.sort(reverse=True)
    used_preds: set[int] = set()
    used_labels: set[int] = set()
    matches: list[MatchedObject] = []
    for iou, pred_index, label_index in candidates:
        if pred_index in used_preds or label_index in used_labels:
            continue
        pred = predictions[pred_index]
        label = labels[label_index]
        used_preds.add(pred_index)
        used_labels.add(label_index)
        matches.append(MatchedObject(pred.frame_id, pred, label, iou))
    return matches


def object_to_record(obj: Kitti3DObject) -> dict[str, object]:
    """转换成 JSON/CSV 友好的字段。"""

    h, w, length = obj.dimensions_hwl
    x, y, z = obj.location_xyz
    x1, y1, x2, y2 = obj.bbox_xyxy
    return {
        "frame_id": obj.frame_id,
        "class": obj.object_type,
        "score": obj.score,
        "bbox_2d": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "dimensions_hwl": [round(h, 4), round(w, 4), round(length, 4)],
        "location_xyz": [round(x, 4), round(y, 4), round(z, 4)],
        "depth_m": round(obj.depth_m, 4),
        "distance_m": round(obj.distance_m, 4),
        "rotation_y": round(obj.rotation_y, 6),
        "alpha": round(obj.alpha, 6),
    }


def finite_mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None

