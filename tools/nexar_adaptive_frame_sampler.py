"""Adaptive frame sampling for Nexar collision-prediction videos.

The positive-video metadata defines an alert-to-event window.  This sampler
keeps the whole window, samples sparsely near ``time_of_alert``, and increases
temporal density toward ``time_of_event``.  It also bounds the image count so a
downstream VLM receives enough temporal evidence without an excessive image
token bill.

Default policy (tuned for the Nexar positive training windows):

* desired frames = ceil(window_seconds * 3) + 4;
* clamp to 6--12 unique source frames;
* use a quadratic time warp, so gaps shrink toward the event;
* never duplicate a source frame; very short windows may therefore contain
  fewer than six frames and are marked ``limited_by_available_frames``;
* downscale the long image edge to at most 960 pixels by default.

Example:

    conda run -n autoscenario python tools/nexar_adaptive_frame_sampler.py

Use ``--plan-only`` to inspect the frame budget without writing images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only without OpenCV installed.
    cv2 = None  # type: ignore


DEFAULT_VIDEO_DIR = Path(
    "data/nexar_collision_prediction/train/positive/00000_00099"
)
DEFAULT_METADATA = Path(
    "data/nexar_collision_prediction/train/positive/metadata.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/00000_00099_frames"
)
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def _cap_prop(name: str, fallback: int) -> int:
    return int(getattr(cv2, name, fallback)) if cv2 is not None else fallback


def load_metadata(metadata_path: os.PathLike[str] | str) -> Dict[str, Dict[str, str]]:
    """Load metadata rows keyed by video basename."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"file_name", "time_of_alert", "time_of_event"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Metadata is missing columns: {sorted(missing_columns)}"
            )
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            filename = Path(str(row.get("file_name") or "")).name
            if filename:
                rows[filename] = dict(row)
    return rows


def desired_frame_count(
    window_duration_s: float,
    *,
    min_frames: int = 6,
    max_frames: int = 12,
    samples_per_second: float = 3.0,
    base_frames: int = 4,
) -> int:
    """Return a bounded per-video image budget for an alert/event window."""
    if window_duration_s < 0.0:
        raise ValueError("window_duration_s must not be negative")
    if min_frames <= 0 or max_frames < min_frames:
        raise ValueError("Require 0 < min_frames <= max_frames")
    if samples_per_second <= 0.0 or base_frames < 0:
        raise ValueError("samples_per_second must be positive and base_frames nonnegative")
    raw_count = int(math.ceil(window_duration_s * samples_per_second)) + base_frames
    return max(min_frames, min(max_frames, raw_count))


def plan_adaptive_frames(
    *,
    time_of_alert_s: float,
    time_of_event_s: float,
    fps: float,
    total_frames: int,
    min_frames: int = 6,
    max_frames: int = 12,
    samples_per_second: float = 3.0,
    base_frames: int = 4,
    density_power: float = 2.0,
) -> Dict[str, Any]:
    """Plan unique source frames with increasing density toward the event.

    For normalized sample position ``u``, the time progress is
    ``1 - (1 - u) ** density_power``.  A power greater than one makes the first
    temporal gap largest and the final gap smallest.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if density_power <= 1.0:
        raise ValueError("density_power must be greater than 1.0")
    if time_of_event_s < time_of_alert_s:
        raise ValueError(
            "time_of_event must be greater than or equal to time_of_alert"
        )

    duration_s = total_frames / fps
    alert_s = min(max(0.0, float(time_of_alert_s)), duration_s)
    event_s = min(max(alert_s, float(time_of_event_s)), duration_s)
    start_index = min(total_frames - 1, max(0, int(round(alert_s * fps))))
    end_index = min(total_frames - 1, max(start_index, int(round(event_s * fps))))
    available_frames = end_index - start_index + 1
    requested_frames = desired_frame_count(
        event_s - alert_s,
        min_frames=min_frames,
        max_frames=max_frames,
        samples_per_second=samples_per_second,
        base_frames=base_frames,
    )
    actual_count = min(requested_frames, available_frames)

    if actual_count == 1:
        frame_indices = [start_index]
    else:
        frame_indices: List[int] = []
        for sample_index in range(actual_count):
            u = sample_index / (actual_count - 1)
            progress = 1.0 - (1.0 - u) ** density_power
            ideal_index = start_index + (end_index - start_index) * progress
            min_allowed = start_index if sample_index == 0 else frame_indices[-1] + 1
            remaining = actual_count - sample_index - 1
            max_allowed = end_index - remaining
            frame_index = max(min_allowed, min(max_allowed, int(round(ideal_index))))
            frame_indices.append(frame_index)

    frames = []
    for sample_index, frame_index in enumerate(frame_indices):
        progress = (
            1.0
            if end_index == start_index
            else (frame_index - start_index) / (end_index - start_index)
        )
        phase = "early" if progress < 0.5 else "middle" if progress < 0.8 else "critical"
        frames.append(
            {
                "sample_index": sample_index,
                "frame_index": frame_index,
                "timestamp_s": round(frame_index / fps, 3),
                "window_progress": round(progress, 4),
                "phase": phase,
            }
        )

    return {
        "time_of_alert_s": round(alert_s, 3),
        "time_of_event_s": round(event_s, 3),
        "window_duration_s": round(event_s - alert_s, 3),
        "requested_frame_count": requested_frames,
        "actual_frame_count": len(frames),
        "available_unique_frames": available_frames,
        "limited_by_available_frames": available_frames < requested_frames,
        "density_power": density_power,
        "frames": frames,
    }


def sample_video(
    video_path: os.PathLike[str] | str,
    metadata_row: Mapping[str, Any],
    output_dir: os.PathLike[str] | str,
    *,
    min_frames: int = 6,
    max_frames: int = 12,
    samples_per_second: float = 3.0,
    base_frames: int = 4,
    density_power: float = 2.0,
    max_edge: int = 960,
    jpeg_quality: int = 90,
    plan_only: bool = False,
    capture_factory: Optional[Callable[[str], Any]] = None,
    image_writer: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Sample one video and write its images plus ``frames_manifest.json``."""
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    factory = capture_factory or (cv2.VideoCapture if cv2 is not None else None)
    writer = image_writer or (cv2.imwrite if cv2 is not None else None)
    if factory is None or (writer is None and not plan_only):
        raise RuntimeError("OpenCV (cv2) is required to sample video frames")

    capture = factory(str(video))
    try:
        is_opened = getattr(capture, "isOpened", None)
        if callable(is_opened) and not is_opened():
            raise FileNotFoundError(f"Could not open video: {video}")
        fps = float(capture.get(_cap_prop("CAP_PROP_FPS", 5)) or 0.0)
        total_frames = int(capture.get(_cap_prop("CAP_PROP_FRAME_COUNT", 7)) or 0)
        plan = plan_adaptive_frames(
            time_of_alert_s=float(metadata_row["time_of_alert"]),
            time_of_event_s=float(metadata_row["time_of_event"]),
            fps=fps,
            total_frames=total_frames,
            min_frames=min_frames,
            max_frames=max_frames,
            samples_per_second=samples_per_second,
            base_frames=base_frames,
            density_power=density_power,
        )

        scene_dir = Path(output_dir)
        frames_dir = scene_dir / "frames"
        if not plan_only:
            frames_dir.mkdir(parents=True, exist_ok=True)
            for frame_info in plan["frames"]:
                frame_index = int(frame_info["frame_index"])
                capture.set(_cap_prop("CAP_PROP_POS_FRAMES", 1), frame_index)
                success, frame = capture.read()
                if not success or frame is None:
                    raise ValueError(
                        f"Failed reading {video.name} at frame {frame_index}"
                    )
                frame = _resize_for_token_budget(frame, max_edge=max_edge)
                timestamp_s = float(frame_info["timestamp_s"])
                filename = (
                    f"{video.stem}_{int(frame_info['sample_index']):02d}_"
                    f"t{timestamp_s:07.3f}s_f{frame_index:06d}.jpg"
                )
                image_path = (frames_dir / filename).resolve()
                params: Sequence[int] = []
                if cv2 is not None:
                    params = [int(getattr(cv2, "IMWRITE_JPEG_QUALITY", 1)), jpeg_quality]
                written = writer(str(image_path), frame, params)
                if written is False:
                    raise ValueError(f"Failed writing sampled frame: {image_path}")
                frame_info["path"] = str(image_path)
    finally:
        release = getattr(capture, "release", None)
        if callable(release):
            release()

    manifest = {
        "scene_id": video.stem,
        "video_path": str(video.resolve()),
        "fps": round(fps, 6),
        "total_frames": total_frames,
        "video_duration_s": round(total_frames / fps, 3),
        "metadata": dict(metadata_row),
        "sampling_strategy": {
            "name": "alert_to_event_quadratic_density_v1",
            "min_frames": min_frames,
            "max_frames": max_frames,
            "samples_per_second": samples_per_second,
            "base_frames": base_frames,
            "density_power": density_power,
            "max_image_edge": max_edge,
            "jpeg_quality": jpeg_quality,
        },
        **plan,
    }
    if not plan_only:
        manifest_path = Path(output_dir) / "frames_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    return manifest


def _resize_for_token_budget(frame: Any, *, max_edge: int) -> Any:
    if cv2 is None or not hasattr(frame, "shape"):
        return frame
    height, width = frame.shape[:2]
    longest = max(int(height), int(width))
    if longest <= max_edge:
        return frame
    scale = max_edge / longest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def iter_videos(video_dir: os.PathLike[str] | str) -> Iterable[Path]:
    directory = Path(video_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Video directory not found: {directory}")
    yield from sorted(
        path for path in directory.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptively sample Nexar frames from alert to collision event."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-frames", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--samples-per-second", type=float, default=3.0)
    parser.add_argument("--base-frames", type=int, default=4)
    parser.add_argument("--density-power", type=float, default=2.0)
    parser.add_argument("--max-edge", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Read video metadata and print/write plans without saving images.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metadata = load_metadata(args.metadata)
    videos = list(iter_videos(args.video_dir))
    if not videos:
        raise FileNotFoundError(f"No videos found under {args.video_dir}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for video in videos:
        row = metadata.get(video.name)
        if row is None:
            errors.append({"video": video.name, "error": "metadata row missing"})
            continue
        try:
            manifest = sample_video(
                video,
                row,
                output_root / video.stem,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                samples_per_second=args.samples_per_second,
                base_frames=args.base_frames,
                density_power=args.density_power,
                max_edge=args.max_edge,
                jpeg_quality=args.jpeg_quality,
                plan_only=args.plan_only,
            )
            summaries.append(
                {
                    "scene_id": manifest["scene_id"],
                    "window_duration_s": manifest["window_duration_s"],
                    "requested_frame_count": manifest["requested_frame_count"],
                    "actual_frame_count": manifest["actual_frame_count"],
                    "limited_by_available_frames": manifest[
                        "limited_by_available_frames"
                    ],
                }
            )
            print(
                f"[{video.stem}] {manifest['window_duration_s']:.3f}s: "
                f"{manifest['actual_frame_count']} frames"
            )
        except Exception as exc:  # keep a large batch diagnosable.
            errors.append({"video": video.name, "error": str(exc)})
            print(f"[{video.stem}] ERROR: {exc}")

    batch_manifest = {
        "video_dir": str(Path(args.video_dir).resolve()),
        "metadata_path": str(Path(args.metadata).resolve()),
        "plan_only": bool(args.plan_only),
        "discovered_video_count": len(videos),
        "sampled_video_count": len(summaries),
        "total_sampled_frames": sum(
            int(item["actual_frame_count"]) for item in summaries
        ),
        "videos": summaries,
        "errors": errors,
    }
    summary_path = output_root / "sampling_summary.json"
    summary_path.write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Done: {len(summaries)}/{len(videos)} videos, "
        f"{batch_manifest['total_sampled_frames']} frames planned/saved. "
        f"Summary: {summary_path}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
