"""Coarse-to-fine low-dimensional black-box parameter search."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .kinematics import KinematicRolloutEvaluator, analytical_initial_estimate
from .models import (
    AccidentSpecification,
    ActorBehaviorParameters,
    CollisionAnchor,
    ReferencePath,
    RolloutResult,
    SceneState,
    SolverResult,
    clamp,
    pair_equal,
)
from .objective import CollisionObjective
from .repair import DirectedRepairPlanner


RolloutEvaluator = Callable[[Dict[str, ActorBehaviorParameters]], RolloutResult]


@dataclass
class _Axis:
    actor_id: str
    field: str
    step: float
    low: float
    high: float


class BehaviorParameterSolver:
    """Search only interpretable behavior parameters, never per-frame controls."""

    def __init__(
        self,
        *,
        max_rollouts: int = 48,
        refinement_levels: int = 2,
        objective: Optional[CollisionObjective] = None,
        repair_planner: Optional[DirectedRepairPlanner] = None,
    ) -> None:
        self.max_rollouts = max(1, int(max_rollouts))
        self.refinement_levels = max(1, int(refinement_levels))
        self.objective = objective or CollisionObjective()
        self.repair_planner = repair_planner or DirectedRepairPlanner()

    def solve(
        self,
        scene_state: SceneState,
        accident_spec: AccidentSpecification,
        paths: Dict[str, ReferencePath],
        collision_anchor: CollisionAnchor,
        evaluator: Optional[RolloutEvaluator] = None,
    ) -> SolverResult:
        parameters = analytical_initial_estimate(scene_state, accident_spec, collision_anchor)
        rollout_evaluator = evaluator or KinematicRolloutEvaluator(
            scene_state, accident_spec, paths, collision_anchor
        )
        best_params = deepcopy(parameters)
        best_rollout, best_loss, best_components = self._evaluate(
            rollout_evaluator, accident_spec, collision_anchor, best_params
        )
        evaluations = 1
        history: List[Dict[str, object]] = []
        axes = self._axes(scene_state, accident_spec)

        for level in range(self.refinement_levels):
            if evaluations >= self.max_rollouts or best_loss <= 1e-9:
                break
            scale = 0.5 ** level
            candidates = list(self._grid(best_params, axes, scale))
            # Test close neighbours first; the cap therefore remains useful when
            # a four-dimensional 3^N grid would exceed the rollout budget.
            candidates.sort(key=lambda item: item[0])
            for _radius, candidate in candidates:
                if evaluations >= self.max_rollouts:
                    break
                rollout, loss, components = self._evaluate(
                    rollout_evaluator, accident_spec, collision_anchor, candidate
                )
                evaluations += 1
                if loss < best_loss:
                    best_params, best_rollout = candidate, rollout
                    best_loss, best_components = loss, components
            actions = self.repair_planner.propose(
                accident_spec, collision_anchor, best_rollout
            )
            history.append(
                {
                    "level": level,
                    "loss": best_loss,
                    "collided": best_rollout.collided,
                    "actions": actions,
                }
            )
            if actions and evaluations < self.max_rollouts:
                repaired = self.repair_planner.apply(best_params, accident_spec, actions)
                rollout, loss, components = self._evaluate(
                    rollout_evaluator, accident_spec, collision_anchor, repaired
                )
                evaluations += 1
                if loss < best_loss:
                    best_params, best_rollout = repaired, rollout
                    best_loss, best_components = loss, components

        converged = (
            best_rollout.collided
            and pair_equal(
                best_rollout.collision_pair,
                (accident_spec.striking_actor_id, accident_spec.struck_actor_id),
            )
            and best_rollout.collision_time is not None
            and collision_anchor.target_time_range[0]
            <= best_rollout.collision_time
            <= collision_anchor.target_time_range[1]
        )
        return SolverResult(
            parameters=best_params,
            rollout=best_rollout,
            loss=best_loss,
            evaluations=evaluations,
            converged=converged,
            loss_components=best_components,
            repair_history=history,
        )

    def _evaluate(self, evaluator, spec, anchor, params):
        rollout = evaluator(deepcopy(params))
        loss, components = self.objective.evaluate(spec, anchor, rollout, params)
        return rollout, loss, components

    def _axes(self, scene, spec) -> List[_Axis]:
        striker, struck = spec.striking_actor_id, spec.struck_actor_id
        axes = [
            _Axis(
                striker,
                "initial_speed",
                2.0,
                0.0,
                (scene.actors[striker].speed_limit_mps or scene.default_speed_limit_mps) * 1.25,
            )
        ]
        if spec.accident_type == "rear_end":
            axes.extend(
                [
                    _Axis(struck, "deceleration", 1.5, 0.0, 10.0),
                    _Axis(striker, "brake_start_time", 0.6, 0.0, scene.duration_s),
                ]
            )
        elif spec.accident_type == "intersection":
            axes.extend(
                [
                    _Axis(
                        struck,
                        "initial_speed",
                        2.0,
                        0.0,
                        (scene.actors[struck].speed_limit_mps or scene.default_speed_limit_mps) * 1.25,
                    ),
                    _Axis(striker, "start_delay", 0.45, 0.0, scene.duration_s * 0.7),
                    _Axis(struck, "start_delay", 0.45, 0.0, scene.duration_s * 0.7),
                ]
            )
        else:
            changing_id = next(
                (
                    actor_id
                    for actor_id, maneuver in spec.actor_maneuvers.items()
                    if str(maneuver).startswith("lane_change")
                ),
                striker,
            )
            axes.extend(
                [
                    _Axis(struck, "initial_speed", 1.5, 0.0, 30.0),
                    _Axis(changing_id, "lane_change_start_time", 0.5, 0.0, scene.duration_s),
                    _Axis(changing_id, "lane_change_duration", 0.5, 0.8, 5.0),
                ]
            )
        return axes

    def _grid(
        self,
        center: Dict[str, ActorBehaviorParameters],
        axes: List[_Axis],
        scale: float,
    ) -> Iterable[Tuple[float, Dict[str, ActorBehaviorParameters]]]:
        seen = set()
        for offsets in itertools.product((-1, 0, 1), repeat=len(axes)):
            if all(offset == 0 for offset in offsets):
                continue
            candidate = deepcopy(center)
            radius = 0.0
            for axis, offset in zip(axes, offsets):
                current = getattr(candidate[axis.actor_id], axis.field)
                if current is None or not math.isfinite(float(current)):
                    current = axis.high if axis.field == "brake_start_time" else axis.low
                value = clamp(float(current) + offset * axis.step * scale, axis.low, axis.high)
                setattr(candidate[axis.actor_id], axis.field, value)
                if axis.field == "initial_speed":
                    candidate[axis.actor_id].target_speed = max(
                        candidate[axis.actor_id].target_speed, value
                    )
                radius += abs(offset)
            signature = tuple(
                (axis.actor_id, axis.field, round(float(getattr(candidate[axis.actor_id], axis.field)), 4))
                for axis in axes
            )
            if signature in seen:
                continue
            seen.add(signature)
            yield radius, candidate
