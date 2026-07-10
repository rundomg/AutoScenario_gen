"""对普通图片文件夹运行 MonoCon 单目 3D 检测。

这个入口用于“没有 KITTI 标签/标定”的图片快速试跑：它会为每张图片生成一个
临时 KITTI 风格相机标定，跑 MonoCon，然后导出 JSON/CSV、KITTI txt、2D+3D
overlay 和 BEV 图。

注意：没有真实相机内参时，3D 位置和距离只能作为粗略可视化参考；真正部署时应传入
车辆相机的实际标定。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from .evaluate_monodetr_predictions import filter_by_score
    from .export_monocon_predictions import choose_device, load_cfg, load_model, write_one_frame_txt
    from .kitti_3d import Kitti3DObject, object_to_record, read_kitti_objects
    from .monocon_paths import add_monocon_to_path, require_monocon_repo
    from .visualize import draw_bev, draw_frame_overlay
except ImportError:  # 允许直接执行：python experiments/mono3d_eval/run_monocon_image_folder.py
    from evaluate_monodetr_predictions import filter_by_score
    from export_monocon_predictions import choose_device, load_cfg, load_model, write_one_frame_txt
    from kitti_3d import Kitti3DObject, object_to_record, read_kitti_objects
    from monocon_paths import add_monocon_to_path, require_monocon_repo
    from visualize import draw_bev, draw_frame_overlay


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_IMAGE_DIR / "output" / "mono3d_monocon_data6"
DEFAULT_CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "monocon_pretrained" / "best.pth"
DEFAULT_CONFIG = EXPERIMENT_DIR / "checkpoints" / "monocon_pretrained" / "config.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


add_monocon_to_path()
require_monocon_repo()

from transforms import Compose, Normalize, Pad, ToTensor  # noqa: E402
from utils.data_classes import KITTICalibration  # noqa: E402
from utils.engine_utils import move_data_device  # noqa: E402


@dataclass(frozen=True)
class ImageRecord:
    """保存一张输入图在推理流程中的所有路径和标定假设。"""

    sample_idx: int
    frame_id: str
    source_path: Path
    prepared_path: Path
    calib_path: Path
    original_size_wh: tuple[int, int]
    inference_size_wh: tuple[int, int]
    focal_px: float


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MonoCon 3D detection on an unlabeled image folder.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Folder containing input images.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for predictions and visuals.")
    parser.add_argument("--checkpoint-file", default=str(DEFAULT_CHECKPOINT), help="MonoCon pretrained checkpoint.")
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG), help="MonoCon config yaml.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit after sorting images by name.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Export/visualization score threshold.")
    parser.add_argument("--model-score-threshold", type=float, default=0.25, help="Threshold used by MonoCon decoder.")
    parser.add_argument("--topk", type=int, default=30, help="Maximum decoded objects per image before class filtering.")
    parser.add_argument("--classes", default="Car", help="Comma-separated classes to keep. MonoCon supports Car/Pedestrian/Cyclist.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=0,
        help="Resize images so the long edge is at most this value. 0 keeps original size.",
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=0.0,
        help="Generated calibration focal length in pixels. Positive value overrides horizontal FOV.",
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=90.0,
        help="Approximate horizontal FOV used when --focal-px is 0. Wider FOV usually fits CODA/ONCE images better.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output folder before writing new results.",
    )
    return parser.parse_args(argv)


def discover_images(image_dir: Path, max_images: Optional[int]) -> list[Path]:
    """按文件名稳定枚举输入图片，只取目录第一层的常见图片格式。"""

    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images is not None:
        images = images[:max_images]
    return images


def generated_focal_px(width: int, explicit_focal_px: float, horizontal_fov_deg: float) -> float:
    """生成无标定图片的近似焦距。

    默认按水平视场角换算焦距：fx = width / (2 * tan(hfov / 2))。
    CODA sample 中的 ONCE 风格图片更像前视广角相机，90 度比 KITTI 焦距缩放更贴近。
    """

    if explicit_focal_px > 0.0:
        return float(explicit_focal_px)
    fov_rad = np.deg2rad(max(20.0, min(140.0, horizontal_fov_deg)))
    return float(width / (2.0 * np.tan(fov_rad / 2.0)))


def make_calibration_dict(width: int, height: int, focal_px: float) -> dict[str, np.ndarray]:
    """构造 MonoCon/KITTI 兼容的最小相机标定。

    P0/P1/P2/P3 都使用同一组 pinhole 内参；外参设为单位变换，仅用于让模型完成
    相机坐标系下的 3D box 解码和投影。
    """

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
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
    """写出 KITTI 文本标定，供 overlay 投影函数复用。"""

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


def prepare_image(source_path: Path, output_path: Path, max_long_edge: int) -> tuple[int, int, int, int]:
    """读取图片并写入推理目录；必要时按长边缩放，返回原图和推理图尺寸。"""

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read input image: {source_path}")

    original_h, original_w = image.shape[:2]
    prepared = image
    if max_long_edge > 0:
        long_edge = max(original_w, original_h)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            new_size = (max(1, round(original_w * scale)), max(1, round(original_h * scale)))
            prepared = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), prepared)
    prepared_h, prepared_w = prepared.shape[:2]
    return original_w, original_h, prepared_w, prepared_h


def prepare_records(
    *,
    image_paths: list[Path],
    output_dir: Path,
    max_long_edge: int,
    focal_px_arg: float,
    horizontal_fov_deg: float,
) -> list[ImageRecord]:
    """生成自包含的推理输入：图片副本/缩放图 + 对应临时标定。"""

    records: list[ImageRecord] = []
    image_output_dir = output_dir / "prepared_images"
    calib_output_dir = output_dir / "generated_calib"

    for sample_idx, source_path in enumerate(image_paths):
        frame_id = source_path.stem
        prepared_path = image_output_dir / f"{frame_id}.jpg"
        calib_path = calib_output_dir / f"{frame_id}.txt"
        original_w, original_h, inference_w, inference_h = prepare_image(source_path, prepared_path, max_long_edge)
        focal_px = generated_focal_px(inference_w, focal_px_arg, horizontal_fov_deg)
        calib = make_calibration_dict(inference_w, inference_h, focal_px)
        write_calibration_txt(calib, calib_path)
        records.append(
            ImageRecord(
                sample_idx=sample_idx,
                frame_id=frame_id,
                source_path=source_path,
                prepared_path=prepared_path,
                calib_path=calib_path,
                original_size_wh=(original_w, original_h),
                inference_size_wh=(inference_w, inference_h),
                focal_px=focal_px,
            )
        )
    return records


class ImageFolderMonoConDataset(Dataset):
    """普通图片文件夹的 MonoCon 推理 Dataset。

    只提供 `img/img_metas/calib`，不加载标签，因此不会触发训练或 KITTI 评估逻辑。
    """

    def __init__(self, records: list[ImageRecord]):
        self.records = records
        self.transforms = Compose(
            [
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
                Pad(size_divisor=32),
                ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_bgr = cv2.imread(str(record.prepared_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read prepared image: {record.prepared_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        data_dict = {
            "img": image_rgb,
            "img_metas": {
                "idx": index,
                "split": "external",
                "sample_idx": record.sample_idx,
                "image_path": str(record.prepared_path),
                "ori_shape": image_rgb.shape[:2],
            },
            "calib": KITTICalibration(str(record.calib_path)),
        }
        return self.transforms(data_dict)

    @staticmethod
    def collate_fn(batched: list[dict[str, Any]]) -> dict[str, Any]:
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


def parse_allowed_classes(classes_arg: str) -> set[str] | None:
    """解析导出类别；传 all 时保留 MonoCon 的全部类别。"""

    if classes_arg.strip().lower() == "all":
        return None
    return {item.strip() for item in classes_arg.split(",") if item.strip()}


def export_predictions(
    *,
    model,
    loader: DataLoader,
    records: list[ImageRecord],
    prediction_dir: Path,
    prediction_2d_dir: Path,
    device: torch.device,
    allowed_classes: set[str] | None,
) -> dict[str, int]:
    """执行推理，并把每张图的 3D 投影结果和原始 2D 框分别写出。"""

    records_by_idx = {record.sample_idx: record for record in records}
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_2d_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    prediction_count = 0
    prediction_2d_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="MonoCon image-folder inference"):
            sample_indices = [int(sample_idx) for sample_idx in batch["img_metas"]["sample_idx"]]
            batch = move_data_device(batch, device)
            results = model.batch_eval(batch)

            for sample_idx, anno_3d, anno_2d in zip(sample_indices, results["img_bbox"], results["img_bbox2d"]):
                record = records_by_idx[sample_idx]
                output_path = prediction_dir / f"{record.frame_id}.txt"
                output_2d_path = prediction_2d_dir / f"{record.frame_id}.txt"
                prediction_count += write_one_frame_txt(anno_3d, output_path, allowed_classes)
                prediction_2d_count += write_one_frame_2d_txt(anno_2d, output_2d_path, allowed_classes)
                frame_count += 1

    return {"frame_count": frame_count, "prediction_count": prediction_count, "prediction_2d_count": prediction_2d_count}


def write_one_frame_2d_txt(anno: dict, output_path: Path, allowed_classes: set[str] | None) -> int:
    """把 MonoCon 原始 2D head 输出写成 KITTI-like txt。

    这些框不带真实 3D 尺寸/位置，只用于诊断“2D 检测是否准确”；因此尺寸和位置字段填入
    占位值，避免和 3D 预测结果混在一起。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = anno.get("name", np.array([]))
    lines: list[str] = []

    for idx, name in enumerate(names):
        class_name = str(name)
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        bbox = anno["bbox"][idx]
        score = float(anno["score"][idx])
        line = (
            f"{class_name} 0.00 0 -10.000000 "
            f"{bbox[0]:.3f} {bbox[1]:.3f} {bbox[2]:.3f} {bbox[3]:.3f} "
            f"0.0000 0.0000 0.0000 -1000.0000 -1000.0000 -1000.0000 0.000000 {score:.6f}"
        )
        lines.append(line)

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_predictions_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "frame_id",
        "class",
        "score",
        "bbox_2d",
        "dimensions_hwl",
        "location_xyz",
        "depth_m",
        "distance_m",
        "rotation_y",
        "alpha",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def raw_2d_objects_to_rows(objects: list[Kitti3DObject]) -> list[dict[str, object]]:
    """把原始 2D 检测框转成轻量 JSON/CSV 记录。"""

    rows: list[dict[str, object]] = []
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox_xyxy
        rows.append(
            {
                "frame_id": obj.frame_id,
                "class": obj.object_type,
                "score": obj.score,
                "bbox_2d": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
            }
        )
    return rows


def write_predictions_2d_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = ["frame_id", "class", "score", "bbox_2d"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), list) else row.get(key) for key in fieldnames})


def draw_raw_2d_boxes(base_overlay_path: Path, raw_2d_objects: list[Kitti3DObject], output_path: Path) -> None:
    """在已有 3D overlay 上叠加原始 2D 检测框。

    紫色框表示 MonoCon 2D head；黄色/青色仍然是 3D box 投影。这样能快速判断问题
    来自检测框本身，还是来自无标定 3D 投影。
    """

    image = cv2.imread(str(base_overlay_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read overlay image: {base_overlay_path}")

    for obj in raw_2d_objects:
        x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = "2D" if obj.score is None else f"2D {obj.score:.2f}"
        y = max(18, y1 - 6)
        cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def export_json_csv_and_visuals(
    *,
    records: list[ImageRecord],
    prediction_dir: Path,
    prediction_2d_dir: Path,
    output_dir: Path,
    score_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """把 KITTI txt 汇总成 JSON/CSV，并画出 overlay/BEV。"""

    prediction_rows: list[dict[str, object]] = []
    prediction_2d_rows: list[dict[str, object]] = []
    overlay_dir = output_dir / "visuals" / "image_overlay"
    overlay_compare_dir = output_dir / "visuals" / "image_overlay_2d_vs_3d"
    bev_dir = output_dir / "visuals" / "bev"

    for record in records:
        predictions = filter_by_score(read_kitti_objects(prediction_dir / f"{record.frame_id}.txt", record.frame_id), score_threshold)
        raw_2d_predictions = filter_by_score(read_kitti_objects(prediction_2d_dir / f"{record.frame_id}.txt", record.frame_id), score_threshold)
        for obj in predictions:
            row = object_to_record(obj)
            row["source_image"] = str(record.source_path)
            row["prepared_image"] = str(record.prepared_path)
            prediction_rows.append(row)
        for row in raw_2d_objects_to_rows(raw_2d_predictions):
            row["source_image"] = str(record.source_path)
            prediction_2d_rows.append(row)

        overlay_path = overlay_dir / f"{record.frame_id}.jpg"
        draw_frame_overlay(
            record.prepared_path,
            record.calib_path,
            predictions,
            overlay_path,
            score_threshold=score_threshold,
        )
        draw_raw_2d_boxes(overlay_path, raw_2d_predictions, overlay_compare_dir / f"{record.frame_id}.jpg")
        draw_bev(predictions, bev_dir / f"{record.frame_id}.jpg", score_threshold=score_threshold)

    write_predictions_csv(prediction_rows, output_dir / "predictions.csv")
    (output_dir / "predictions.json").write_text(json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions_2d_csv(prediction_2d_rows, output_dir / "predictions_2d.csv")
    (output_dir / "predictions_2d.json").write_text(json.dumps(prediction_2d_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return prediction_rows, prediction_2d_rows


def write_run_summary(
    *,
    records: list[ImageRecord],
    prediction_rows: list[dict[str, object]],
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """记录本次推理的输入、输出和无真实标定假设。"""

    summary = {
        "image_count": len(records),
        "prediction_count": len(prediction_rows),
        "device": str(device),
        "score_threshold": args.score_threshold,
        "classes": args.classes,
        "checkpoint_file": str(Path(args.checkpoint_file)),
        "config_file": str(Path(args.config_file)),
        "calibration_note": (
            "Generated pinhole calibration was used because these images do not include real camera intrinsics. "
            "3D distance/location should be treated as rough visualization only."
        ),
        "max_long_edge": args.max_long_edge,
        "horizontal_fov_deg": args.horizontal_fov_deg,
        "focal_px_override": args.focal_px if args.focal_px > 0.0 else None,
        "images": [
            {
                "frame_id": record.frame_id,
                "source_path": str(record.source_path),
                "prepared_path": str(record.prepared_path),
                "calib_path": str(record.calib_path),
                "original_size_wh": list(record.original_size_wh),
                "inference_size_wh": list(record.inference_size_wh),
                "generated_focal_px": round(record.focal_px, 4),
            }
            for record in records
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(image_dir, args.max_images)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    checkpoint_file = Path(args.checkpoint_file)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    records = prepare_records(
        image_paths=image_paths,
        output_dir=output_dir,
        max_long_edge=args.max_long_edge,
        focal_px_arg=args.focal_px,
        horizontal_fov_deg=args.horizontal_fov_deg,
    )

    device = choose_device(args.device, args.gpu_id)
    cfg = load_cfg(Path(args.config_file), PROJECT_ROOT / "KITTI" / "kitti", args.batch_size, args.num_workers)
    model = load_model(
        cfg,
        checkpoint_file=checkpoint_file,
        device=device,
        model_score_threshold=args.model_score_threshold,
        topk=args.topk,
    )

    dataset = ImageFolderMonoConDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    prediction_dir = output_dir / "prediction_txt"
    prediction_2d_dir = output_dir / "prediction_2d_txt"
    export_summary = export_predictions(
        model=model,
        loader=loader,
        records=records,
        prediction_dir=prediction_dir,
        prediction_2d_dir=prediction_2d_dir,
        device=device,
        allowed_classes=parse_allowed_classes(args.classes),
    )
    prediction_rows, prediction_2d_rows = export_json_csv_and_visuals(
        records=records,
        prediction_dir=prediction_dir,
        prediction_2d_dir=prediction_2d_dir,
        output_dir=output_dir,
        score_threshold=args.score_threshold,
    )
    summary = write_run_summary(records=records, prediction_rows=prediction_rows, output_dir=output_dir, args=args, device=device)
    summary.update(export_summary)
    summary["prediction_2d_count_after_threshold"] = len(prediction_2d_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
