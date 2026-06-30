"""Standalone preview for the vehicle detector.

Runs `tools.vehicle_detector` on one image, prints every detected vehicle
(class / confidence / normalized bbox + center), and writes an annotated image
so you can eyeball detection quality before wiring it into the VLM pipeline.

Usage:
    python experiments/run_vehicle_detector.py                 # auto-pick a data/ image
    python experiments/run_vehicle_detector.py path/to/img.png
    python experiments/run_vehicle_detector.py img.png --conf 0.4 --out preview.jpg
"""

import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import vehicle_detector
from tools import depth_estimator


def _auto_pick_image() -> str | None:
    candidates = ["data/001.png", *sorted(glob.glob("data/*.png")),
                  *sorted(glob.glob("data/*.jpg"))]
    return next((p for p in candidates if os.path.exists(p)), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview vehicle detection on one image.")
    parser.add_argument("image", nargs="?", default=None,
                        help="Image path. If omitted, auto-picks an image under data/.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (default: detector default / YOLO_CONF).")
    parser.add_argument("--out", default=None,
                        help="Annotated image output path (default: <image>_detections.jpg).")
    args = parser.parse_args()

    image_path = args.image or _auto_pick_image()
    if not image_path:
        print("No image found. Pass one explicitly: "
              "python experiments/run_vehicle_detector.py path/to/img.png")
        return 1
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1

    print(f"image: {image_path}")
    conf_kwargs = {} if args.conf is None else {"conf": args.conf}
    detections = vehicle_detector.detect_vehicles(image_path, **conf_kwargs)

    if not detections:
        print("No vehicles detected (or the detector is unavailable — "
              "install with `pip install ultralytics`).")
        return 0

    depth_estimator.attach_depth(image_path, detections)
    has_depth = any("depth_m" in d for d in detections)

    print(f"detected {len(detections)} vehicles"
          f"{' (with depth)' if has_depth else ' (no depth)'}:")
    print(f"  {'id':<8} {'label':<11} {'conf':>5} {'depth':>8}  {'center (cx,cy)':<16} bbox_norm")
    for d in detections:
        cx, cy = d["center_norm"]
        depth = f"{d['depth_m']}m" if "depth_m" in d else "-"
        print(f"  {d['id']:<8} {d['label']:<11} {d['conf']:>5.2f} {depth:>8}  "
              f"({cx:.3f}, {cy:.3f})    {d['bbox_norm']}")

    out_path = args.out or f"{os.path.splitext(image_path)[0]}_detections.jpg"
    written = vehicle_detector.annotate_image(image_path, detections, out_path)
    print(f"\nannotated image -> {written}" if written
          else "\nannotated image not written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
