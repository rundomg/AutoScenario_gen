"""Anchor frame extraction for the accident dynamic-reconstruction pipeline.

Per the plan's Assumptions (and the user's decision), v1 does NOT auto-detect the
start/end frames. The user supplies the two anchor *timestamps in seconds*; this
module turns them into saved frame images plus a ``frames.json`` manifest that
records, for each saved frame, its ``frame_index``, ``timestamp_s``, ``fps`` and
role flags (``is_start`` / ``is_end_anchor``). Optional intermediate context
frames can be sampled between the two anchors for the structured video
understanding agent.

cv2 access is injected (``capture_factory`` / ``imwrite``) so the module is unit
testable without a real video or OpenCV build.
"""

import json
import os
from typing import Any, Callable, Dict, List, Optional

try:  # cv2 is mocked as a SimpleNamespace in the test suite.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only when cv2 is absent.
    cv2 = None  # type: ignore

from tools.utils import write_to_file


# OpenCV property ids, read with fallbacks so a mocked cv2 module never crashes.
def _cap_prop(name: str, fallback: int) -> int:
    return int(getattr(cv2, name, fallback)) if cv2 is not None else fallback


def extract_anchor_frames(
    video_path: str,
    start_s: float,
    end_s: float,
    output_dir: str,
    scene_id: str,
    context_sample_rate_s: Optional[float] = None,
    capture_factory: Optional[Callable[[str], Any]] = None,
    imwrite: Optional[Callable[[str, Any], Any]] = None,
) -> Dict[str, Any]:
    """Extract the start + end anchor frames (and optional context frames).

    Returns a manifest dict (also written to ``frames.json``). Raises
    ``ValueError`` for out-of-range / inverted timestamps and ``FileNotFoundError``
    if the video cannot be opened.
    """
    if start_s < 0.0:
        raise ValueError(f"start_s must be >= 0, got {start_s}.")
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s}).")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    factory = capture_factory or (cv2.VideoCapture if cv2 is not None else None)
    writer = imwrite or (cv2.imwrite if cv2 is not None else None)
    if factory is None or writer is None:
        raise RuntimeError("OpenCV (cv2) is required to extract frames.")

    os.makedirs(output_dir, exist_ok=True)
    capture = factory(video_path)
    try:
        if not _is_opened(capture):
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = float(capture.get(_cap_prop("CAP_PROP_FPS", 5))) or 0.0
        total_frames = int(capture.get(_cap_prop("CAP_PROP_FRAME_COUNT", 7)) or 0)
        if fps <= 0.0:
            raise ValueError(f"Could not read a positive fps from {video_path}.")
        duration_s = total_frames / fps if total_frames > 0 else None
        if duration_s is not None and end_s > duration_s + 1.0 / fps:
            raise ValueError(
                f"end_s ({end_s}) exceeds video duration ({duration_s:.3f}s)."
            )

        frames: List[Dict[str, Any]] = []

        start_meta = _grab(
            capture, writer, fps, start_s, output_dir,
            f"{scene_id}_start_frame.jpg", total_frames,
            roles={"is_start": True, "is_end_anchor": False},
        )
        frames.append(start_meta)

        if context_sample_rate_s and context_sample_rate_s > 0.0:
            t = start_s + context_sample_rate_s
            ctx_index = 0
            while t < end_s - 1e-6:
                frames.append(
                    _grab(
                        capture, writer, fps, t, output_dir,
                        f"{scene_id}_context_{ctx_index:03d}.jpg", total_frames,
                        roles={"is_start": False, "is_end_anchor": False},
                    )
                )
                ctx_index += 1
                t += context_sample_rate_s

        end_meta = _grab(
            capture, writer, fps, end_s, output_dir,
            f"{scene_id}_end_anchor_frame.jpg", total_frames,
            roles={"is_start": False, "is_end_anchor": True},
        )
        frames.append(end_meta)
    finally:
        _release(capture)

    manifest = {
        "scene_id": scene_id,
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
        "start_s": start_s,
        "end_s": end_s,
        "start_frame_path": start_meta["path"],
        "end_anchor_frame_path": end_meta["path"],
        "frames": frames,
    }
    write_to_file(
        os.path.join(output_dir, "frames.json"),
        json.dumps(manifest, indent=2, sort_keys=True),
    )
    return manifest


def _grab(
    capture: Any,
    writer: Callable[[str, Any], Any],
    fps: float,
    timestamp_s: float,
    output_dir: str,
    filename: str,
    total_frames: int,
    roles: Dict[str, bool],
) -> Dict[str, Any]:
    frame_index = int(round(timestamp_s * fps))
    if total_frames > 0:
        frame_index = min(frame_index, total_frames - 1)
    capture.set(_cap_prop("CAP_PROP_POS_FRAMES", 1), frame_index)
    success, frame = capture.read()
    if not success or frame is None:
        raise ValueError(
            f"Failed to read frame at {timestamp_s}s (index {frame_index})."
        )
    path = os.path.join(output_dir, filename)
    writer(path, frame)
    meta = {
        "frame_index": frame_index,
        "timestamp_s": round(timestamp_s, 3),
        "fps": fps,
        "path": os.path.abspath(path),
    }
    meta.update(roles)
    return meta


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
