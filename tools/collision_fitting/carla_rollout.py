"""Subprocess CARLA rollout adapter for simulation-guided refinement.

Each candidate gets immutable DSL/script/metrics artifacts.  The generated
script already performs a deterministic fixed-step reset/spawn/run/cleanup
cycle, so the optimiser only depends on the generic callable contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from tools.video_trajectory_dsl import lower_to_risk_dsl, validate_video_trajectory_dsl

from .integration import apply_start_offsets_to_spawn_payload, build_fitted_trajectory_dsl
from .models import (
    ActorBehaviorParameters,
    PipelineResult,
    RolloutResult,
    SolverResult,
    pair_equal,
)


class CarlaSubprocessRolloutEvaluator:
    """Execute one generated CARLA script per candidate parameter set."""

    def __init__(
        self,
        *,
        scene_state,
        accident_spec,
        paths,
        collision_anchor,
        spawn_payload: Dict[str, Any],
        spawn_payload_path: str,
        output_dir: str,
        scene_id: str,
        base_trajectory_dsl: Optional[Dict[str, Any]] = None,
        map_name: Optional[str] = None,
        map_match_status: Optional[str] = None,
        map_match_reason: Optional[str] = None,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        python_executable: Optional[str] = None,
        timeout_s: float = 120.0,
        codegen: Any = None,
    ) -> None:
        self.scene_state = scene_state
        self.accident_spec = accident_spec
        self.paths = paths
        self.collision_anchor = collision_anchor
        self.spawn_payload = spawn_payload
        self.spawn_payload_path = os.path.abspath(spawn_payload_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_id = scene_id
        self.base_dsl = deepcopy(base_trajectory_dsl or {})
        self.map_name = str(map_name).split("/")[-1] if map_name else None
        self.map_match_status = map_match_status
        self.map_match_reason = map_match_reason
        self.carla_host = carla_host
        self.carla_port = int(carla_port)
        self.python_executable = python_executable or sys.executable
        self.timeout_s = max(10.0, float(timeout_s))
        if codegen is None:
            # Keep importing the LLM/CARLA generator (and its optional dotenv
            # dependency) outside the offline geometry/test path.
            from agents.existing_world_scenario_generator import (
                ExistingWorldScenarioGenerator,
            )

            codegen = ExistingWorldScenarioGenerator()
        self.codegen = codegen
        self.candidate_index = 0

    def __call__(
        self, parameters: Dict[str, ActorBehaviorParameters]
    ) -> RolloutResult:
        index = self.candidate_index
        self.candidate_index += 1
        stem = f"{self.scene_id}_fit_rollout_{index:04d}"
        trajectory_path = self.output_dir / f"{stem}_trajectory.json"
        risk_path = self.output_dir / f"{stem}_risk.json"
        metrics_path = self.output_dir / f"{stem}_metrics.json"
        script_path = self.output_dir / f"{stem}.py"
        spawn_path = self.output_dir / f"{stem}_actors.json"

        placeholder_rollout = RolloutResult(collided=False)
        placeholder_solver = SolverResult(
            parameters=deepcopy(parameters),
            rollout=placeholder_rollout,
            loss=0.0,
            evaluations=0,
            converged=False,
        )
        fitting_result = PipelineResult(
            specification=self.accident_spec,
            paths=self.paths,
            anchor=self.collision_anchor,
            solver_result=placeholder_solver,
        )
        trajectory = build_fitted_trajectory_dsl(
            fitting_result,
            scene_id=self.scene_id,
            duration_s=self.scene_state.duration_s,
            base_dsl=self.base_dsl,
        )
        normalized, error = validate_video_trajectory_dsl(
            trajectory, self.spawn_payload
        )
        if error is not None or normalized is None:
            raise ValueError(f"CARLA candidate DSL validation failed: {error}")
        risk_dsl = lower_to_risk_dsl(normalized)
        fitted_spawn_payload = apply_start_offsets_to_spawn_payload(
            self.spawn_payload, fitting_result
        )
        spawn_path.write_text(
            json.dumps(fitted_spawn_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        trajectory_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8"
        )
        risk_path.write_text(
            json.dumps(risk_dsl, indent=2, sort_keys=True), encoding="utf-8"
        )
        script = self.codegen.build_dsl_risk_scene_script(
            spawn_payload_filename=str(spawn_path.resolve()),
            dsl_filename=str(risk_path.resolve()),
            risk_metrics_filename=str(metrics_path.resolve()),
            carla_host=self.carla_host,
            carla_port=self.carla_port,
            carla_map=self.map_name,
            scene_match_status=self.map_match_status,
            scene_match_reason=self.map_match_reason,
        )
        script_path.write_text(script, encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "AUTOSCENARIO_CLEAR_EXISTING": "1",
                "AUTOSCENARIO_RISK_DURATION": str(self.scene_state.duration_s),
                "AUTOSCENARIO_RISK_TICK_DT": str(
                    self.scene_state.fixed_delta_seconds
                ),
                "AUTOSCENARIO_DISABLE_FLYING_START": "0",
            }
        )
        completed = subprocess.run(
            [self.python_executable, str(script_path)],
            cwd=str(self.output_dir),
            env=environment,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0 or not metrics_path.exists():
            message = (completed.stderr or completed.stdout or "")[-3000:]
            raise RuntimeError(
                f"CARLA rollout {index} failed with code {completed.returncode}: {message}"
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return _rollout_from_metrics(
            metrics,
            target_pair=(
                self.accident_spec.striking_actor_id,
                self.accident_spec.struck_actor_id,
            ),
            candidate_index=index,
            accident_type=self.accident_spec.accident_type,
        )


def _rollout_from_metrics(
    metrics, *, target_pair, candidate_index, accident_type=None
):
    events = list(metrics.get("collision_events") or [])
    first = events[0] if events else {}
    observed_pair = first.get("collision_pair") or metrics.get("collision_pair")
    non_target = []
    seen_non_target = set()
    for event in events:
        pair = event.get("collision_pair")
        if pair and not pair_equal(pair, target_pair):
            normalized_pair = tuple(sorted(str(item) for item in pair[:2]))
            if normalized_pair not in seen_non_target:
                seen_non_target.add(normalized_pair)
                non_target.append(normalized_pair)
    location = first.get("location") or metrics.get("collision_location")
    if isinstance(location, dict):
        collision_location = (float(location.get("x", 0.0)), float(location.get("y", 0.0)))
    else:
        collision_location = None
    arrival_times = dict(metrics.get("arrival_times") or {})
    arrivals = [
        float(value) for value in arrival_times.values() if value is not None
    ]
    relation_errors = {}
    if accident_type != "rear_end" and len(arrivals) == 2:
        relation_errors["arrival_time_delta_s"] = abs(arrivals[0] - arrivals[1])
    return RolloutResult(
        collided=bool(events),
        collision_pair=(
            tuple(str(item) for item in observed_pair[:2])
            if isinstance(observed_pair, (list, tuple)) and len(observed_pair) >= 2
            else None
        ),
        collision_time=(
            float(first.get("time_s"))
            if first.get("time_s") is not None
            else metrics.get("collision_time_s")
        ),
        collision_location=collision_location,
        relative_impact_speed=metrics.get("relative_impact_speed_mps"),
        trajectory_logs=dict(metrics.get("trajectory_logs") or {}),
        minimum_pair_distance=float(
            metrics.get("minimum_pair_bbox_distance_m")
            if metrics.get("minimum_pair_bbox_distance_m") is not None
            else metrics.get("min_distance_m")
            if metrics.get("min_distance_m") is not None
            else float("inf")
        ),
        relation_errors=relation_errors,
        arrival_times=arrival_times,
        non_target_collisions=non_target,
        contact_sides=dict(first.get("contact_sides") or {}),
        metadata={
            "backend": "carla_subprocess",
            "candidate_index": candidate_index,
            "raw_metrics": metrics,
        },
    )


class CarlaEvaluatorFactory:
    """Bind static artifacts once and create a pair-specific evaluator."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(self, scene_state, accident_spec, paths, collision_anchor):
        return CarlaSubprocessRolloutEvaluator(
            scene_state=scene_state,
            accident_spec=accident_spec,
            paths=paths,
            collision_anchor=collision_anchor,
            **self.kwargs,
        )
