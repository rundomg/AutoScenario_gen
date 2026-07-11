"""MonoCon KITTI 子集推理 Dataset。

MonoCon 原仓库的 Dataset 会顺手导入 KITTI AP 评估代码，而那部分依赖
numba.cuda/nvvm。当前主环境能跑 PyTorch 推理，但没有 nvvm.dll，所以这里
实现一个只用于推理的轻量 Dataset：读取 RGB 图、相机标定，复用 MonoCon 的
测试预处理，不碰官方 AP 评估 kernel。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import torch
from torch.utils.data import Dataset

try:
    from .monocon_paths import add_monocon_to_path, require_monocon_repo
except ImportError:  # 允许直接执行调用
    from monocon_paths import add_monocon_to_path, require_monocon_repo


add_monocon_to_path()
require_monocon_repo()

from transforms import Compose, Normalize, Pad, ToTensor  # noqa: E402
from utils.data_classes import KITTICalibration  # noqa: E402


def read_frame_ids(frame_list: Optional[Path], kitti_root: Path, split: str, max_frames: Optional[int]) -> List[str]:
    """读取 frame id；没有 frame_list 时从 image_2 目录自动枚举。"""

    if frame_list is not None:
        frame_ids = [line.strip() for line in frame_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        image_dir = kitti_root / split / "image_2"
        frame_ids = sorted(path.stem for path in image_dir.glob("*.png"))

    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return frame_ids


class MonoConInferenceDataset(Dataset):
    """只服务 MonoCon 推理的数据集。

    返回字段保持为 MonoCon `batch_eval` 需要的 `img/img_metas/calib`，
    但不加载 label，因此不会触发训练目标生成或官方 AP 评估依赖。
    """

    def __init__(
        self,
        kitti_root: Path,
        split: str,
        frame_ids: Iterable[str],
        calib_dir_override: Optional[Path] = None,
    ):
        if split not in {"training", "testing"}:
            raise ValueError(f"Unsupported KITTI split: {split}")

        self.kitti_root = kitti_root
        self.split = split
        self.frame_ids = list(frame_ids)
        self.sub_root = "testing" if split == "testing" else "training"
        self.image_dir = kitti_root / self.sub_root / "image_2"
        # For calibration ablations, point MonoCon at generated KITTI-style
        # calibration files while keeping the original dataset untouched.
        self.calib_dir = calib_dir_override if calib_dir_override is not None else kitti_root / self.sub_root / "calib"
        self.transforms = Compose(
            [
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                Pad(size_divisor=32),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame_id = self.frame_ids[index]
        image_path = self.image_dir / f"{frame_id}.png"
        calib_path = self.calib_dir / f"{frame_id}.txt"

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read KITTI image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        data_dict = {
            "img": image_rgb,
            "img_metas": {
                "idx": index,
                "split": "test" if self.split == "testing" else "val",
                "sample_idx": int(frame_id),
                "image_path": str(image_path),
                "ori_shape": image_rgb.shape[:2],
            },
            "calib": KITTICalibration(str(calib_path)),
        }
        return self.transforms(data_dict)

    @staticmethod
    def collate_fn(batched: List[Dict[str, Any]]) -> Dict[str, Any]:
        """复刻 MonoCon 推理所需的 batch 结构。"""

        merged_image = torch.cat([item["img"].unsqueeze(0) for item in batched], dim=0)
        img_metas_list = [item["img_metas"] for item in batched]
        merged_metas = {key: [] for key in img_metas_list[0].keys()}
        for img_metas in img_metas_list:
            for key, value in img_metas.items():
                merged_metas[key].append(value)

        return {
            "img": merged_image,
            "img_metas": merged_metas,
            "calib": [item["calib"] for item in batched],
        }


def build_monocon_subset(
    *,
    kitti_root: Path,
    split: str,
    frame_ids: Iterable[str],
    max_objs: int,
    filter_configs: Optional[dict] = None,
    calib_dir_override: Optional[Path] = None,
) -> MonoConInferenceDataset:
    """保留旧调用签名；max_objs/filter_configs 对纯推理 Dataset 不生效。"""

    _ = max_objs, filter_configs
    return MonoConInferenceDataset(
        kitti_root=kitti_root,
        split=split,
        frame_ids=frame_ids,
        calib_dir_override=calib_dir_override,
    )
