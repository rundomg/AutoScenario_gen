"""Accident video -> CARLA dynamic reconstruction (Layer-2 dynamic branch).

Reuses the static reconstruction (``{scene_id}_actors.json`` / ``_match.json``)
produced by Layer-1 on the *start frame*, then turns an accident video into an
executable CARLA Python script via a structured video understanding and a
``video-trajectory-dsl-v1`` document (lowered to ``risk-dsl-v1``).

The user supplies the two anchor timestamps (seconds); v1 does NOT auto-select
frames. Step 1 below extracts and saves the start frame so it can be fed to
Layer-1 if it has not been reconstructed yet.

Example:
    # 1) extract the start frame (writes <folder>/frames/<scene_id>_start_frame.jpg)
    # 2) run Layer-1 on that start frame to produce {scene_id}_actors.json + _match.json
    # 3) then:
    python experiments/auto_generate_all_video_reconstruction.py \
        --video-path data/accident_001.mp4 \
        --output-folder results/auto_result_20260618_192858 \
        --scene-id s0000_c0 --start-frame 4 --end-frame 7
"""

import argparse
import json
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.video_reconstruction_runner import VideoReconstructionRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct an accident video as an executable CARLA dynamic scenario.",
    )
    parser.add_argument("--video-path", required=True, help="Accident video file.")
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Static reconstruction output folder (source of {scene_id}_actors.json).",
    )
    parser.add_argument(
        "--risk-output-folder",
        help="Folder for generated dynamic artifacts. Defaults to --output-folder.",
    )
    parser.add_argument("--scene-id", required=True, help="Static candidate scene id, e.g. s0000_c0.")
    parser.add_argument(
        "--start-frame",
        type=float,
        required=True,
        help="Start anchor timestamp in SECONDS (spawn anchor; before the accident).",
    )
    parser.add_argument(
        "--end-frame",
        type=float,
        required=True,
        help="End anchor timestamp in SECONDS (soft-constraint; accident outcome).",
    )
    parser.add_argument(
        "--context-sample-rate",
        type=float,
        default=1.0,
        help=(
            "Sample one context frame every N seconds strictly between the anchors "
            "(default 1.0). With a 3 s start->end gap this yields 4 frames "
            "(start + 2 context + end). Pass 0 to use only the start/end frames."
        ),
    )
    parser.add_argument("--ego-speed-mps", type=float, default=10.0, help="Fixed ego speed (m/s).")
    parser.add_argument("--max-retries", type=int, default=2, help="Max trajectory-DSL schema-repair retries.")
    parser.add_argument("--no-compile", action="store_true", help="Skip py_compile of the generated script.")
    parser.add_argument("--carla-host", default="localhost", help="CARLA RPC host for the generated script.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC port for the generated script.")
    parser.add_argument("--user-request", default="", help="Optional extra instruction passed to the VLM.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = VideoReconstructionRunner(
        output_folder=args.output_folder,
        scene_id=args.scene_id,
        video_path=args.video_path,
        start_s=args.start_frame,
        end_s=args.end_frame,
        risk_output_folder=args.risk_output_folder,
        ego_speed_mps=args.ego_speed_mps,
        context_sample_rate_s=args.context_sample_rate,
        max_retries=args.max_retries,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        enable_compile=not args.no_compile,
    )
    summary = runner.run(user_request=args.user_request)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
