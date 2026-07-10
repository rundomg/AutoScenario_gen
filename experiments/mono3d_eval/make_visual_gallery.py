"""把 MonoCon overlay/BEV 抽样结果拼成总览图。

评估脚本已经逐帧保存了可视化图；这个脚本只是把前若干张拼成 gallery，
方便快速浏览，不影响任何指标计算。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results" / "monocon_current_env_500"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for MonoCon visual outputs.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=520)
    return parser.parse_args(argv)


def collect_images(image_dir: Path, max_images: int) -> list[Path]:
    return sorted(image_dir.glob("*.jpg"))[:max_images]


def resize_with_label(image_path: Path, tile_width: int) -> np.ndarray:
    """按宽度缩放图片，并在顶部加 frame id 标签。"""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    scale = tile_width / max(image.shape[1], 1)
    resized = cv2.resize(image, (tile_width, max(1, int(round(image.shape[0] * scale)))))
    label_h = 30
    canvas = np.full((resized.shape[0] + label_h, tile_width, 3), 245, dtype=np.uint8)
    canvas[label_h:, :, :] = resized
    cv2.putText(canvas, image_path.stem, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2, cv2.LINE_AA)
    return canvas


def make_gallery(image_paths: list[Path], output_path: Path, columns: int, tile_width: int) -> None:
    """把多张图补齐高度后拼成网格。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        blank = np.full((180, tile_width, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "No visual images", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.imwrite(str(output_path), blank)
        return

    tiles = [resize_with_label(path, tile_width) for path in image_paths]
    rows = math.ceil(len(tiles) / columns)
    tile_h = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        canvas = np.full((tile_h, tile_width, 3), 245, dtype=np.uint8)
        canvas[: tile.shape[0], :, :] = tile
        padded_tiles.append(canvas)

    blank_tile = np.full((tile_h, tile_width, 3), 245, dtype=np.uint8)
    while len(padded_tiles) < rows * columns:
        padded_tiles.append(blank_tile.copy())

    row_images = []
    for row_index in range(rows):
        start = row_index * columns
        row_images.append(np.hstack(padded_tiles[start : start + columns]))
    gallery = np.vstack(row_images)
    cv2.imwrite(str(output_path), gallery)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = Path(args.result_dir)
    visuals_dir = result_dir / "visuals"

    overlay_paths = collect_images(visuals_dir / "image_overlay", args.max_images)
    bev_paths = collect_images(visuals_dir / "bev", args.max_images)
    make_gallery(overlay_paths, visuals_dir / "overlay_gallery.jpg", args.columns, args.tile_width)
    make_gallery(bev_paths, visuals_dir / "bev_gallery.jpg", args.columns, args.tile_width)
    print(f"wrote galleries to {visuals_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
