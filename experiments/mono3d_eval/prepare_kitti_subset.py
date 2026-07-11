"""为 MonoDETR smoke/eval 准备 KITTI frame id 列表。

MonoDETR 训练/测试脚本通常读取 ImageSets/*.txt；这里生成很小的子集，
便于先确认环境、checkpoint、输出格式和可视化链路。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "KITTI" / "kitti"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ImageSets"


def collect_frame_ids(kitti_root: Path, split: str) -> list[str]:
    image_dir = kitti_root / split / "image_2"
    return sorted(path.stem for path in image_dir.glob("*.png"))


def write_frame_list(frame_ids: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(frame_ids) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small KITTI frame lists for MonoDETR tests.")
    parser.add_argument("--kitti-root", default=str(DEFAULT_KITTI_ROOT))
    parser.add_argument("--split", default="training", choices=("training", "testing"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke-count", type=int, default=5)
    parser.add_argument("--eval-count", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    frame_ids = collect_frame_ids(Path(args.kitti_root), args.split)
    if not frame_ids:
        raise RuntimeError(f"No KITTI frames found under {args.kitti_root}/{args.split}/image_2")

    output_dir = Path(args.output_dir)
    write_frame_list(frame_ids[: args.smoke_count], output_dir / f"{args.split}_smoke_{args.smoke_count}.txt")
    write_frame_list(frame_ids[: args.eval_count], output_dir / f"{args.split}_eval_{args.eval_count}.txt")
    print(f"wrote frame lists to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

