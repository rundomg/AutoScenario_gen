"""Fit an existing video understanding with collision-anchored parameters.

This command is CARLA-free by default: it runs analytical initialisation plus
the OBB/path kinematic surrogate and emits a reproducible configuration.  The
same ``BehaviorParameterSolver`` accepts a CARLA rollout evaluator for final
simulation-guided refinement.
"""

import argparse
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.collision_fitting import (
    CollisionAnchoredFittingPipeline,
    build_accident_specification,
    build_scene_state,
)
from tools.collision_fitting.integration import (
    apply_start_offsets_to_spawn_payload,
    build_fitted_trajectory_dsl,
    build_scenario_configuration,
)
from tools.collision_fitting.solver import BehaviorParameterSolver
from tools.collision_fitting.paths import MapConstrainedPathBuilder
from tools.video_trajectory_dsl import lower_to_risk_dsl, validate_video_trajectory_dsl


def parse_args():
    parser = argparse.ArgumentParser(description="Collision-anchored behavior fitting")
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--understanding", help="Defaults to <folder>/<scene>_video_understanding.json")
    parser.add_argument("--base-trajectory-dsl", help="Optional VLM DSL; non-target actor timelines are preserved")
    parser.add_argument("--risk-output-folder", help="Defaults to --output-folder")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--tick-dt", type=float, default=0.05)
    parser.add_argument("--max-rollouts", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--carla-rollouts",
        action="store_true",
        help="Use resettable CARLA subprocess rollouts instead of the kinematic surrogate.",
    )
    parser.add_argument("--carla-host", default="localhost")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument(
        "--carla-python",
        help="Python executable whose environment contains the CARLA 0.9.16 API.",
    )
    parser.add_argument("--rollout-timeout", type=float, default=120.0)
    return parser.parse_args()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    folder = Path(args.output_folder)
    risk_folder = Path(args.risk_output_folder or args.output_folder)
    risk_folder.mkdir(parents=True, exist_ok=True)
    actors_path = folder / f"{args.scene_id}_actors.json"
    match_path = folder / f"{args.scene_id}_match.json"
    understanding_path = Path(
        args.understanding or risk_folder / f"{args.scene_id}_video_understanding.json"
    )
    spawn_payload = _load(actors_path)
    understanding = _load(understanding_path)
    match = _load(match_path) if match_path.exists() else {}
    base_dsl = _load(args.base_trajectory_dsl) if args.base_trajectory_dsl else {}
    duration = float(base_dsl.get("duration_s", args.duration))
    scene = build_scene_state(
        spawn_payload,
        duration_s=duration,
        fixed_delta_seconds=args.tick_dt,
        map_name=match.get("world_name"),
    )
    spec = build_accident_specification(
        understanding, scene.actors.keys(), duration_s=duration, top_k=args.top_k
    )
    evaluator_factory = None
    path_builder = None
    if args.carla_rollouts:
        import carla

        from tools.collision_fitting.carla_rollout import CarlaEvaluatorFactory

        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(min(30.0, args.rollout_timeout))
        world = client.get_world()
        target_map = str(match.get("world_name") or "").split("/")[-1]
        current_map = world.get_map().name.split("/")[-1]
        if target_map and current_map != target_map:
            world = client.load_world(target_map)
        path_builder = MapConstrainedPathBuilder(carla_map=world.get_map())

        evaluator_factory = CarlaEvaluatorFactory(
            spawn_payload=spawn_payload,
            spawn_payload_path=str(actors_path),
            output_dir=str(risk_folder / "collision_fit_rollouts"),
            scene_id=args.scene_id,
            base_trajectory_dsl=base_dsl,
            map_name=match.get("world_name"),
            map_match_status=match.get("status"),
            map_match_reason=match.get("reason"),
            carla_host=args.carla_host,
            carla_port=args.carla_port,
            python_executable=args.carla_python,
            timeout_s=args.rollout_timeout,
        )
    pipeline = CollisionAnchoredFittingPipeline(
        path_builder=path_builder,
        solver=BehaviorParameterSolver(max_rollouts=args.max_rollouts),
        top_k=args.top_k,
    )
    result = pipeline.fit(scene, spec, evaluator_factory=evaluator_factory)
    trajectory = build_fitted_trajectory_dsl(
        result, scene_id=args.scene_id, duration_s=duration, base_dsl=base_dsl
    )
    normalized, error = validate_video_trajectory_dsl(trajectory, spawn_payload)
    if error:
        raise SystemExit(f"Fitted trajectory DSL failed validation: {error}")
    config = build_scenario_configuration(
        result, scene_id=args.scene_id, map_name=match.get("world_name")
    )
    config_path = risk_folder / f"{args.scene_id}_collision_fitted_config.json"
    trajectory_path = risk_folder / f"{args.scene_id}_collision_fitted_trajectory_dsl.json"
    risk_path = risk_folder / f"{args.scene_id}_collision_fitted_risk_dsl.json"
    fitted_actors_path = risk_folder / f"{args.scene_id}_collision_fitted_actors.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    trajectory_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    risk_path.write_text(
        json.dumps(lower_to_risk_dsl(normalized), indent=2, sort_keys=True), encoding="utf-8"
    )
    fitted_actors_path.write_text(
        json.dumps(
            apply_start_offsets_to_spawn_payload(spawn_payload, result),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config_path": str(config_path),
                "trajectory_dsl_path": str(trajectory_path),
                "risk_dsl_path": str(risk_path),
                "fitted_actors_path": str(fitted_actors_path),
                "converged": result.solver_result.converged,
                "loss": result.solver_result.loss,
                "rollouts": result.solver_result.evaluations,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
