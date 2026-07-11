"""用当前 Python 环境运行 MonoCon，并导出 KITTI txt + 项目评估结果。

输出分两层：
1. prediction txt：每帧一个 KITTI detection 格式文件，便于复用外部工具。
2. eval output：复用本目录的通用评估脚本，生成 JSON/CSV、2D+3D overlay、BEV 图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .calibration_modes import parse_calibration_mode, prepare_generated_calibrations
    from .evaluate_monodetr_predictions import evaluate_frames
    from .monocon_dataset import build_monocon_subset, read_frame_ids
    from .monocon_paths import DEFAULT_KITTI_ROOT, add_monocon_to_path, require_monocon_repo
except ImportError:  # 允许直接 python experiments/mono3d_eval/export_monocon_predictions.py
    from calibration_modes import parse_calibration_mode, prepare_generated_calibrations
    from evaluate_monodetr_predictions import evaluate_frames
    from monocon_dataset import build_monocon_subset, read_frame_ids
    from monocon_paths import DEFAULT_KITTI_ROOT, add_monocon_to_path, require_monocon_repo


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FRAME_LIST = EXPERIMENT_DIR / "ImageSets" / "training_smoke_5.txt"
DEFAULT_PREDICTION_DIR = EXPERIMENT_DIR / "results" / "monocon_current_env_smoke" / "prediction_txt"
DEFAULT_EVAL_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "monocon_current_env_smoke"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MonoCon on a KITTI subset and export KITTI-format predictions.")
    parser.add_argument("--checkpoint-file", required=True, help="MonoCon pretrained checkpoint (.pth).")
    parser.add_argument("--config-file", default=None, help="Optional MonoCon yaml config. Defaults to repo config.")
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--split", default="training", choices=("training", "testing"))
    parser.add_argument("--frame-list", default=str(DEFAULT_FRAME_LIST))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--prediction-dir", default=str(DEFAULT_PREDICTION_DIR))
    parser.add_argument("--eval-output-dir", default=str(DEFAULT_EVAL_OUTPUT_DIR))
    parser.add_argument("--calibration-mode", default="true_k", choices=("true_k", "fake_fov90", "fake_ratio058"))
    parser.add_argument("--generated-calib-dir", default=None, help="Where fake KITTI calibration files are written/read.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--classes", default="Car", help="Comma-separated classes to export, or 'all'.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Evaluation/export score threshold.")
    parser.add_argument("--model-score-threshold", type=float, default=0.25, help="Threshold used by MonoCon decoder.")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--max-objs", type=int, default=30)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--save-visuals", type=int, default=5)
    parser.add_argument("--skip-eval", action="store_true", help="Only write prediction txt files.")
    return parser.parse_args(argv)


def choose_device(device_arg: str, gpu_id: int) -> torch.device:
    """根据参数和 CUDA 状态选择设备。"""

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def load_cfg(config_file: Optional[Path], kitti_root: Path, batch_size: int, num_workers: int):
    """读取 MonoCon 配置，并覆盖本实验需要的路径与 batch 参数。"""

    add_monocon_to_path()
    from utils.engine_utils import get_default_cfg, load_cfg as load_monocon_cfg

    cfg = load_monocon_cfg(str(config_file)) if config_file is not None else get_default_cfg()
    cfg.DATA.ROOT = str(kitti_root)
    cfg.DATA.BATCH_SIZE = batch_size
    cfg.DATA.NUM_WORKERS = num_workers
    cfg.DATA.TEST_SPLIT = "val"
    cfg.MODEL.BACKBONE.IMAGENET_PRETRAINED = False
    cfg.USE_BENCHMARK = False
    return cfg


def load_model(cfg, checkpoint_file: Path, device: torch.device, model_score_threshold: float, topk: int):
    """加载 MonoCon 权重；兼容原仓库 engine checkpoint 和常见 state_dict checkpoint。"""

    add_monocon_to_path()
    from model import MonoConDetector

    model = MonoConDetector(
        num_dla_layers=cfg.MODEL.BACKBONE.NUM_LAYERS,
        pretrained_backbone=False,
    )

    try:
        checkpoint = torch.load(str(checkpoint_file), map_location=device, weights_only=False)
    except TypeError:  # 兼容旧 PyTorch
        checkpoint = torch.load(str(checkpoint_file), map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "model" in checkpoint["state_dict"]:
        state_dict = checkpoint["state_dict"]["model"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 兼容 DataParallel 保存出来的 module.xxx key。
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model.head.test_thres = model_score_threshold
    model.head.topk = topk
    model.head.max_per_img = topk
    model.to(device)
    model.eval()
    return model


def parse_allowed_classes(classes_arg: str) -> Optional[Set[str]]:
    if classes_arg.strip().lower() == "all":
        return None
    return {item.strip() for item in classes_arg.split(",") if item.strip()}


def write_one_frame_txt(anno: dict, output_path: Path, allowed_classes: Optional[Set[str]]) -> int:
    """把 MonoCon 内部 annotation 写成标准 KITTI txt。

    MonoCon 内部 dimensions 顺序是 length/height/width；
    KITTI 文本顺序是 height/width/length，所以这里必须显式转换。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = anno.get("name", np.array([]))
    lines: List[str] = []

    for idx, name in enumerate(names):
        class_name = str(name)
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        bbox = anno["bbox"][idx]
        length, height, width = anno["dimensions"][idx]
        x, y, z = anno["location"][idx]
        alpha = float(anno["alpha"][idx])
        rotation_y = float(anno["rotation_y"][idx])
        score = float(anno["score"][idx])

        line = (
            f"{class_name} 0.00 0 {alpha:.6f} "
            f"{bbox[0]:.3f} {bbox[1]:.3f} {bbox[2]:.3f} {bbox[3]:.3f} "
            f"{height:.4f} {width:.4f} {length:.4f} "
            f"{x:.4f} {y:.4f} {z:.4f} {rotation_y:.6f} {score:.6f}"
        )
        lines.append(line)

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def export_predictions(
    *,
    model,
    loader: DataLoader,
    prediction_dir: Path,
    device: torch.device,
    allowed_classes: Optional[Set[str]],
) -> dict:
    """执行推理并写出每帧 KITTI txt。"""

    add_monocon_to_path()
    from utils.engine_utils import move_data_device

    prediction_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    prediction_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="MonoCon inference"):
            sample_ids = [f"{int(sample_id):06d}" for sample_id in batch["img_metas"]["sample_idx"]]
            batch = move_data_device(batch, device)
            results = model.batch_eval(batch)

            for frame_id, anno in zip(sample_ids, results["img_bbox"]):
                count = write_one_frame_txt(anno, prediction_dir / f"{frame_id}.txt", allowed_classes)
                prediction_count += count
                frame_count += 1

    return {"frame_count": frame_count, "prediction_count": prediction_count}


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    add_monocon_to_path()
    require_monocon_repo()

    checkpoint_file = Path(args.checkpoint_file)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    kitti_root = Path(args.kitti_root)
    frame_list = Path(args.frame_list) if args.frame_list else None
    frame_ids = read_frame_ids(frame_list, kitti_root, args.split, args.max_frames)
    if not frame_ids:
        raise RuntimeError("No KITTI frames were selected.")

    calibration_mode = parse_calibration_mode(args.calibration_mode)
    generated_calib_dir = Path(args.generated_calib_dir) if args.generated_calib_dir else Path(args.eval_output_dir) / "generated_calib"
    calibration_summary = prepare_generated_calibrations(
        kitti_root=kitti_root,
        split=args.split,
        frame_ids=frame_ids,
        mode=calibration_mode,
        output_dir=generated_calib_dir,
    )
    calib_dir_override = Path(calibration_summary["generated_calib_dir"]) if calibration_summary["generated_calib_dir"] else None

    device = choose_device(args.device, args.gpu_id)
    cfg = load_cfg(Path(args.config_file) if args.config_file else None, kitti_root, args.batch_size, args.num_workers)
    dataset = build_monocon_subset(
        kitti_root=kitti_root,
        split=args.split,
        frame_ids=frame_ids,
        max_objs=args.max_objs,
        filter_configs={key.lower(): value for key, value in dict(cfg.DATA.FILTER).items()},
        calib_dir_override=calib_dir_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    model = load_model(
        cfg,
        checkpoint_file=checkpoint_file,
        device=device,
        model_score_threshold=args.model_score_threshold,
        topk=args.topk,
    )

    prediction_dir = Path(args.prediction_dir)
    export_summary = export_predictions(
        model=model,
        loader=loader,
        prediction_dir=prediction_dir,
        device=device,
        allowed_classes=parse_allowed_classes(args.classes),
    )

    summary = {
        "device": str(device),
        "prediction_dir": str(prediction_dir),
        **calibration_summary,
        **export_summary,
    }
    if not args.skip_eval and args.split == "training":
        eval_summary = evaluate_frames(
            prediction_dir=prediction_dir,
            kitti_root=kitti_root,
            split=args.split,
            frame_ids=frame_ids,
            output_dir=Path(args.eval_output_dir),
            score_threshold=args.score_threshold,
            match_iou=args.match_iou,
            save_visuals=args.save_visuals,
            visual_calib_dir=calib_dir_override,
        )
        summary["evaluation"] = eval_summary

    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    (eval_output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
