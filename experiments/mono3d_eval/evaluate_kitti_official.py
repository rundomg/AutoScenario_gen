"""运行 MonoCon/KITTI 官方风格 AP40 评估。

这个脚本消费已经导出的 KITTI txt 预测结果，不重新跑模型。
它调用 MonoCon 仓库自带的 KITTI evaluator，输出 bbox/BEV/3D AP40。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .evaluate_monodetr_predictions import discover_frame_ids
    from .kitti_3d import Kitti3DObject, read_kitti_objects
    from .monocon_paths import DEFAULT_KITTI_ROOT, EXPERIMENT_DIR, MONOCON_ROOT, VENDOR_DIR
except ImportError:  # 允许直接 python experiments/mono3d_eval/evaluate_kitti_official.py
    from evaluate_monodetr_predictions import discover_frame_ids
    from kitti_3d import Kitti3DObject, read_kitti_objects
    from monocon_paths import DEFAULT_KITTI_ROOT, EXPERIMENT_DIR, MONOCON_ROOT, VENDOR_DIR


DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "results" / "monocon_current_env_500"
DEFAULT_PREDICTION_DIR = DEFAULT_RESULT_DIR / "prediction_txt"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "official_eval"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MonoCon/KITTI official-style AP40 evaluation.")
    parser.add_argument("--prediction-dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--split", default="training", choices=("training", "testing"))
    parser.add_argument("--frame-list", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--classes", default="Car", help="Comma-separated KITTI classes, e.g. Car or Car,Pedestrian,Cyclist.")
    parser.add_argument("--eval-types", default="bbox,bev,3d", help="Comma-separated eval types: bbox,bev,3d.")
    parser.add_argument("--force-cuda-cc", default="8.9", help="Numba CUDA CC override for new GPUs such as RTX 5080.")
    return parser.parse_args(argv)


def normalize_eval_types(raw: str) -> list[str]:
    """Normalize CLI eval type text before passing it to MonoCon's evaluator."""

    aliases = {"2d": "bbox", "3": "3d"}
    valid = {"bbox", "bev", "3d", "aos"}
    eval_types: list[str] = []
    for item in raw.split(","):
        normalized = aliases.get(item.strip().lower(), item.strip().lower())
        if not normalized:
            continue
        if normalized not in valid:
            raise ValueError(f"Unsupported eval type {item!r}; valid values: bbox,bev,3d,aos")
        if normalized not in eval_types:
            eval_types.append(normalized)
    return eval_types


def configure_monocon_eval_environment(force_cuda_cc: str) -> None:
    """配置官方 evaluator 所需的 import path 和 Numba CUDA 环境。

    当前 RTX 5080 的算力较新，Numba 还不认识 12.0；这里强制生成兼容 PTX。
    cuda_shim 只包含 libNVVM 和 libdevice，避免修改系统 CUDA 安装。
    """

    cuda_shim = EXPERIMENT_DIR / "cuda_shim"
    if cuda_shim.exists():
        os.environ.setdefault("CUDA_HOME", str(cuda_shim))
        os.environ.setdefault("CUDA_PATH", str(cuda_shim))
        nvvm_bin = cuda_shim / "nvvm" / "bin"
        os.environ["PATH"] = f"{nvvm_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if force_cuda_cc:
        os.environ.setdefault("NUMBA_FORCE_CUDA_CC", force_cuda_cc)

    for path in (VENDOR_DIR, MONOCON_ROOT, MONOCON_ROOT / "engine"):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def objects_to_anno(objects: list[Kitti3DObject], *, is_prediction: bool) -> dict[str, np.ndarray]:
    """把通用 KITTI object 转成 MonoCon 官方 evaluator 的 annos dict。

    重要：MonoCon evaluator 期望 dimensions 顺序是 l/h/w，不是 KITTI txt 的 h/w/l。
    """

    count = len(objects)
    anno = {
        "name": np.array([obj.object_type for obj in objects]),
        "truncated": np.array([obj.truncation for obj in objects], dtype=np.float64),
        "occluded": np.array([obj.occlusion for obj in objects], dtype=np.int64),
        "alpha": np.array([obj.alpha for obj in objects], dtype=np.float64),
        "bbox": np.zeros((count, 4), dtype=np.float64),
        "dimensions": np.zeros((count, 3), dtype=np.float64),
        "location": np.zeros((count, 3), dtype=np.float64),
        "rotation_y": np.array([obj.rotation_y for obj in objects], dtype=np.float64),
    }

    for index, obj in enumerate(objects):
        h, w, length = obj.dimensions_hwl
        anno["bbox"][index] = np.array(obj.bbox_xyxy, dtype=np.float64)
        anno["dimensions"][index] = np.array([length, h, w], dtype=np.float64)
        anno["location"][index] = np.array(obj.location_xyz, dtype=np.float64)

    if is_prediction:
        anno["score"] = np.array([obj.score if obj.score is not None else 0.0 for obj in objects], dtype=np.float64)
    return anno


def load_annos(
    *,
    prediction_dir: Path,
    kitti_root: Path,
    split: str,
    frame_ids: list[str],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """读取每帧 GT 和 prediction；GT 保留 DontCare，prediction 保留所有预测类别。"""

    gt_annos: list[dict[str, np.ndarray]] = []
    dt_annos: list[dict[str, np.ndarray]] = []
    for frame_id in frame_ids:
        label_path = kitti_root / split / "label_2" / f"{frame_id}.txt"
        pred_path = prediction_dir / f"{frame_id}.txt"
        gt_annos.append(objects_to_anno(read_kitti_objects(label_path, frame_id, keep_non_vehicle=True), is_prediction=False))
        dt_annos.append(objects_to_anno(read_kitti_objects(pred_path, frame_id, keep_non_vehicle=True), is_prediction=True))
    return gt_annos, dt_annos


def json_ready(metrics: dict[str, object]) -> dict[str, float]:
    """把 numpy scalar 转成普通 float，方便写 JSON。"""

    return {key: float(value) for key, value in metrics.items()}


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configure_monocon_eval_environment(args.force_cuda_cc)

    from engine.kitti_eval import kitti_eval

    prediction_dir = Path(args.prediction_dir)
    frame_list = Path(args.frame_list) if args.frame_list else None
    frame_ids = discover_frame_ids(prediction_dir, frame_list, args.max_frames)
    if not frame_ids:
        raise RuntimeError(f"No frames found for official eval in {prediction_dir}")

    eval_classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    eval_types = normalize_eval_types(args.eval_types)
    gt_annos, dt_annos = load_annos(
        prediction_dir=prediction_dir,
        kitti_root=Path(args.kitti_root),
        split=args.split,
        frame_ids=frame_ids,
    )

    result_text, metrics = kitti_eval(gt_annos, dt_annos, current_classes=eval_classes, eval_types=eval_types)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_kitti_eval.txt").write_text(result_text + "\n", encoding="utf-8")
    (output_dir / "official_kitti_eval.json").write_text(
        json.dumps(
            {
                "frame_count": len(frame_ids),
                "classes": eval_classes,
                "eval_types": eval_types,
                "metrics": json_ready(metrics),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(result_text)
    print(f"wrote official eval to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
