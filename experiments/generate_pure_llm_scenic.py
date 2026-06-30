"""CLI driver for the pure-LLM (no template library) Scenic experiment.

Reuses an existing static reconstruction folder (which provides
``{scene_id}_actors.json`` and ``{scene_id}_match.json``) and the original camera
image, then runs the two-stage pure-LLM Scenic pipeline.

Example:
    python experiments/generate_pure_llm_scenic.py \
        --output-folder results/auto_result_20260604_112849 \
        --scene-id s0000_c0 --image-path data/001.png --max-retries 2
"""

import argparse
import json
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.pure_llm_scenic_runner import PureLlmScenicRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure-LLM (no template library) accident-to-Scenic experiment.",
    )
    parser.add_argument("--output-folder", required=True, help="Static reconstruction output folder (provides actors.json/match.json).")
    parser.add_argument("--risk-output-folder", help="Folder for generated Scenic artifacts. Defaults to --output-folder.")
    parser.add_argument("--scene-id", required=True, help="Static candidate scene id, e.g. s0000_c0.")
    parser.add_argument("--image-path", help="Original camera image. Required unless --candidates is provided.")
    parser.add_argument("--candidates", help="Existing {scene_id}_candidates.json. Skips stage-1 VLM prediction.")
    parser.add_argument("--ego-speed-mps", type=float, default=10.0, help="Fixed ego speed (m/s).")
    parser.add_argument("--max-retries", type=int, default=2, help="Max compile-repair retries per candidate.")
    parser.add_argument("--carla-map", help="Override CARLA map name (else taken from match.json world_name).")
    parser.add_argument("--carla-maps-dir", help="Directory holding TownXX.xodr files (else $CARLA_MAPS_DIR or built-in default).")
    parser.add_argument("--scenic-conda-env", default="scenicNL", help="Conda env that has scenic installed.")
    parser.add_argument("--no-compile", action="store_true", help="Generate .scenic only, skip compile checks.")
    parser.add_argument("--user-request", default="", help="Optional extra instruction passed to the stage-1 VLM.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.candidates and not args.image_path:
        raise SystemExit("--image-path is required unless --candidates is provided.")
    runner = PureLlmScenicRunner(
        output_folder=args.output_folder,
        scene_id=args.scene_id,
        risk_output_folder=args.risk_output_folder,
        ego_speed_mps=args.ego_speed_mps,
        carla_maps_dir=args.carla_maps_dir,
        max_retries=args.max_retries,
        carla_map=args.carla_map,
        scenic_conda_env=args.scenic_conda_env,
        enable_compile=not args.no_compile,
    )
    summary = runner.run(
        candidates_path=args.candidates,
        user_request=args.user_request,
        image_path=args.image_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
