"""Frame sampling utilities for accident keyframe selection.

This module samples a fixed number of frames per second from an accident video
and writes a manifest that a VLM selector can consume. It intentionally keeps
the sampling step deterministic and simple: for 2 fps, it saves frames at
0.0s, 0.5s, 1.0s, 1.5s, ...
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - only used when OpenCV is unavailable.
    cv2 = None  # type: ignore

from tools.utils import write_to_file


def _cap_prop(name: str, fallback: int) -> int:
    return int(getattr(cv2, name, fallback)) if cv2 is not None else fallback


def sample_video_frames(
    video_path: str,
    output_dir: str,
    scene_id: str,
    samples_per_second: float = 2.0,
    image_ext: str = ".jpg",
    capture_factory: Optional[Callable[[str], Any]] = None,
    imwrite: Optional[Callable[[str, Any], Any]] = None,
) -> Dict[str, Any]:
    """Sample video frames at a fixed rate and write ``frames_manifest.json``.

    Args:
        video_path: Input video path.
        output_dir: Directory where frame images and manifest are saved.
        scene_id: Prefix used in saved frame names.
        samples_per_second: Sampling frequency. Use 2.0 for two frames/second.
        image_ext: Saved frame extension, normally ``.jpg``.
        capture_factory/imwrite: Injectable OpenCV hooks for tests.
    """
    if samples_per_second <= 0.0:
        raise ValueError("samples_per_second must be positive.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    factory = capture_factory or (cv2.VideoCapture if cv2 is not None else None)
    writer = imwrite or (cv2.imwrite if cv2 is not None else None)
    if factory is None or writer is None:
        raise RuntimeError("OpenCV (cv2) is required to sample frames.")

    os.makedirs(output_dir, exist_ok=True)
    capture = factory(video_path)
    try:
        if not _is_opened(capture):
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = float(capture.get(_cap_prop("CAP_PROP_FPS", 5))) or 0.0
        total_frames = int(capture.get(_cap_prop("CAP_PROP_FRAME_COUNT", 7)) or 0)
        if fps <= 0.0:
            raise ValueError(f"Could not read a positive fps from {video_path}.")
        if total_frames <= 0:
            raise ValueError(f"Could not read total frame count from {video_path}.")

        duration_s = total_frames / fps
        step_s = 1.0 / samples_per_second
        frames: List[Dict[str, Any]] = []

        sample_index = 0
        timestamp_s = 0.0
        last_frame_index = -1
        while timestamp_s < duration_s:
            frame_index = min(int(round(timestamp_s * fps)), total_frames - 1)
            if frame_index == last_frame_index:
                timestamp_s += step_s
                continue
            last_frame_index = frame_index

            capture.set(_cap_prop("CAP_PROP_POS_FRAMES", 1), frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError(
                    f"Failed to read frame at {timestamp_s:.3f}s "
                    f"(index {frame_index})."
                )

            filename = (
                f"{scene_id}_t{timestamp_s:07.3f}s_"
                f"f{frame_index:06d}{image_ext}"
            )
            path = os.path.abspath(os.path.join(output_dir, filename))
            writer(path, frame)
            frames.append(
                {
                    "sample_index": sample_index,
                    "timestamp_s": round(timestamp_s, 3),
                    "second": int(timestamp_s),
                    "frame_index": frame_index,
                    "path": path,
                }
            )
            sample_index += 1
            timestamp_s += step_s
    finally:
        _release(capture)

    manifest = {
        "scene_id": scene_id,
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 3),
        "samples_per_second": samples_per_second,
        "frames": frames,
    }
    write_to_file(
        os.path.join(output_dir, "frames_manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return manifest


def _is_opened(capture: Any) -> bool:
    is_opened = getattr(capture, "isOpened", None)
    if callable(is_opened):
        return bool(is_opened())
    return True


def _release(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        try:
            release()
        except Exception:
            pass
