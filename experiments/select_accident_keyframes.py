"""Sample accident videos and use a VLM to estimate accident timing.

Example:
    python experiments/select_accident_keyframes.py \
        --video-dir data/accid_1_scenes \
        --output-dir data/accid_1_keyframes \
        --samples-per-second 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    args = parse_args()
    videos = list(resolve_videos(args))
    if not videos:
        raise FileNotFoundError("No input videos found.")

    from agents.accident_keyframe_selector import AccidentKeyframeSelector
    from tools.accident_keyframe_sampler import sample_video_frames
    from tools.utils import write_to_file

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selector = AccidentKeyframeSelector()
    summaries: List[Dict[str, object]] = []
    for video_path in videos:
        scene_id = video_path.stem
        scene_dir = output_root / scene_id
        frames_dir = scene_dir / "sampled_frames"
        scene_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{scene_id}] Sampling {args.samples_per_second} fps from {video_path}")
        manifest = sample_video_frames(
            str(video_path),
            str(frames_dir),
            scene_id=scene_id,
            samples_per_second=args.samples_per_second,
        )

        output_fn = scene_dir / "keyframe_selection.json"
        print(f"[{scene_id}] Asking VLM to estimate collision second")
        result = selector.call_agent(
            args.user_request or "",
            {
                "scene_id": scene_id,
                "frames": manifest["frames"],
                "output_fn": str(output_fn),
            },
        )

        write_to_file(str(output_fn), json.dumps(result, indent=2, sort_keys=True))
        summaries.append(
            {
                "scene_id": scene_id,
                "video_path": str(video_path),
                "output_dir": str(scene_dir),
                "collision_second": result.get("collision_second"),
            }
        )
        print(f"[{scene_id}] collision_second={result.get('collision_second')}")

    summary_path = output_root / "keyframe_selection_summary.json"
    write_to_file(str(summary_path), json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Done. Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample accident videos at a fixed rate and use a VLM to estimate "
            "the collision second."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to one accident video.")
    group.add_argument("--video-dir", help="Directory containing accident videos.")
    parser.add_argument(
        "--output-dir",
        default="data/accident_keyframes",
        help="Root directory for sampled frames and VLM JSON outputs.",
    )
    parser.add_argument(
        "--samples-per-second",
        type=float,
        default=2.0,
        help="Frame sampling rate. Default: 2 frames per second.",
    )
    parser.add_argument(
        "--user-request",
        default="",
        help="Optional extra instruction appended to the VLM prompt.",
    )
    return parser.parse_args()


def resolve_videos(args: argparse.Namespace) -> Iterable[Path]:
    if args.video:
        yield Path(args.video).resolve()
        return

    video_dir = Path(args.video_dir).resolve()
    for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
        for path in sorted(video_dir.glob(suffix)):
            yield path.resolve()


if __name__ == "__main__":
    main()
