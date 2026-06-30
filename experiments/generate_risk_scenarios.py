"""Generate ego-scripted risk scenarios from completed static reconstruction output.

By default this runs the pure-LLM two-stage pipeline (no accident template
library): a VLM predicts an accident from the original image, then an LLM emits a
structured DSL that is deterministically turned into an executable CARLA Python
script (with a DSL schema validation + repair loop). Pass ``--use-template`` to
fall back to the legacy template-library pipeline (``RiskScenarioRunner``).

Example (pure-LLM, default):
    python experiments/generate_risk_scenarios.py \
        --output-folder results/auto_result_20260617_145909 \
        --scene-id s0000_c0 --image-path data/001.png --num-accidents 1
"""

import argparse
import json
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.pure_llm_risk_runner import PureLlmRiskRunner
from tools.risk_scenario_runner import RiskScenarioRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ego-scripted risk scenarios from completed static CARLA reconstruction output.",
    )
    parser.add_argument("--output-folder", required=True, help="Static reconstruction output folder.")
    parser.add_argument(
        "--risk-output-folder",
        help=(
            "Folder for generated risk artifacts. Defaults to --output-folder "
            "for backward compatibility."
        ),
    )
    parser.add_argument("--scene-id", required=True, help="Static candidate scene id, e.g. s0000_c0.")
    parser.add_argument(
        "--image-path",
        help="Original camera image. Required unless --risk-spec / --candidates is provided.",
    )
    parser.add_argument(
        "--use-template",
        action="store_true",
        help="Use the legacy template-library pipeline (RiskScenarioRunner) instead of the pure-LLM pipeline.",
    )
    # Pure-LLM pipeline options (default).
    parser.add_argument("--ego-speed-mps", type=float, default=10.0, help="Fixed ego speed (m/s) for the pure-LLM pipeline.")
    parser.add_argument("--num-accidents", type=int, default=1, help="Number of predicted accidents to turn into scenarios (pure-LLM).")
    parser.add_argument("--max-retries", type=int, default=2, help="Max DSL schema-repair retries per accident (pure-LLM).")
    parser.add_argument("--candidates", help="Existing {scene_id}_candidates.json. Skips stage-1 VLM prediction (pure-LLM).")
    parser.add_argument("--no-compile", action="store_true", help="Skip py_compile syntax check of generated scripts (pure-LLM).")
    # Legacy template pipeline options (only used with --use-template).
    parser.add_argument("--risk-spec", help="Existing risk_scenario_spec JSON. Skips VLM interpretation (template pipeline).")
    parser.add_argument("--samples-per-candidate", type=int, default=1, help="Samples per risk candidate (template pipeline only).")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for deterministic sampling (template pipeline only).")
    # Shared CARLA options.
    parser.add_argument("--carla-host", default="localhost", help="CARLA RPC host for generated scripts.")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC port for generated scripts.")
    parser.add_argument("--carla-map", help="Override CARLA map for generated scripts.")
    parser.add_argument("--user-request", default="", help="Optional extra instruction passed to the stage-1 VLM.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.use_template:
        if not args.risk_spec and not args.image_path:
            raise SystemExit("--image-path is required unless --risk-spec is provided.")
        runner = RiskScenarioRunner(
            output_folder=args.output_folder,
            scene_id=args.scene_id,
            risk_output_folder=args.risk_output_folder,
            samples_per_candidate=args.samples_per_candidate,
            seed=args.seed,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_map=args.carla_map,
        )
        summary = runner.run(
            risk_spec_path=args.risk_spec,
            user_request=args.user_request,
            image_path=args.image_path,
        )
    else:
        if not args.candidates and not args.image_path:
            raise SystemExit("--image-path is required unless --candidates is provided.")
        runner = PureLlmRiskRunner(
            output_folder=args.output_folder,
            scene_id=args.scene_id,
            risk_output_folder=args.risk_output_folder,
            ego_speed_mps=args.ego_speed_mps,
            num_accidents=args.num_accidents,
            max_retries=args.max_retries,
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            carla_map=args.carla_map,
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
