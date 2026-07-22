"""End-to-end collision anchor -> search -> reproducible configuration pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from .anchors import CollisionAnchorBuilder
from .models import (
    AccidentSpecification,
    PipelineResult,
    SceneState,
)
from .paths import MapConstrainedPathBuilder
from .solver import BehaviorParameterSolver, RolloutEvaluator


EvaluatorFactory = Callable[
    [SceneState, AccidentSpecification, Dict[str, Any], Any], Optional[RolloutEvaluator]
]


class CollisionAnchoredFittingPipeline:
    def __init__(
        self,
        *,
        path_builder: Optional[MapConstrainedPathBuilder] = None,
        anchor_builder: Optional[CollisionAnchorBuilder] = None,
        solver: Optional[BehaviorParameterSolver] = None,
        top_k: int = 3,
    ) -> None:
        self.path_builder = path_builder or MapConstrainedPathBuilder()
        self.anchor_builder = anchor_builder or CollisionAnchorBuilder()
        self.solver = solver or BehaviorParameterSolver()
        self.top_k = max(1, int(top_k))

    def fit(
        self,
        scene_state: SceneState,
        accident_spec: AccidentSpecification,
        *,
        evaluator_factory: Optional[EvaluatorFactory] = None,
    ) -> PipelineResult:
        hypotheses = accident_spec.pair_hypotheses[: self.top_k]
        if not hypotheses:
            hypotheses = [None]
        best_result = None
        best_rank = float("inf")
        summaries = []
        for hypothesis in hypotheses:
            spec = deepcopy(accident_spec)
            confidence = spec.confidence.get("collision_pair", 0.5)
            if hypothesis is not None:
                previous_striking = spec.striking_actor_id
                previous_struck = spec.struck_actor_id
                spec.striking_actor_id = str(
                    hypothesis.striking_actor_id or hypothesis.actor_a
                )
                spec.struck_actor_id = str(hypothesis.struck_actor_id or hypothesis.actor_b)
                for new_id, previous_id in (
                    (spec.striking_actor_id, previous_striking),
                    (spec.struck_actor_id, previous_struck),
                ):
                    if new_id not in spec.actor_maneuvers:
                        spec.actor_maneuvers[new_id] = spec.actor_maneuvers.get(
                            previous_id, "keep_lane"
                        )
                    if new_id not in spec.behavior_sequences:
                        spec.behavior_sequences[new_id] = list(
                            spec.behavior_sequences.get(previous_id) or ["keep"]
                        )
                confidence = hypothesis.confidence
            if (
                spec.striking_actor_id not in scene_state.actors
                or spec.struck_actor_id not in scene_state.actors
            ):
                continue
            paths = {}
            for actor_id in {spec.striking_actor_id, spec.struck_actor_id}:
                maneuver = spec.actor_maneuvers.get(actor_id, "keep_lane")
                paths[actor_id] = self.path_builder.build(
                    scene_state.actors[actor_id], maneuver
                )
            anchor = self.anchor_builder.build(scene_state, spec, paths)
            evaluator = (
                evaluator_factory(scene_state, spec, paths, anchor)
                if evaluator_factory is not None
                else None
            )
            solved = self.solver.solve(scene_state, spec, paths, anchor, evaluator)
            # Pair confidence is a tie breaker only.  It is intentionally much
            # smaller than every hard collision-correctness tier.
            rank = solved.loss + (1.0 - confidence) * 50.0
            summaries.append(
                {
                    "striking_actor_id": spec.striking_actor_id,
                    "struck_actor_id": spec.struck_actor_id,
                    "confidence": confidence,
                    "anchor_feasible": anchor.feasible,
                    "loss": solved.loss,
                    "rank": rank,
                    "converged": solved.converged,
                }
            )
            if rank < best_rank:
                best_rank = rank
                best_result = PipelineResult(
                    specification=spec,
                    paths=paths,
                    anchor=anchor,
                    solver_result=solved,
                )
        if best_result is None:
            raise ValueError("No collision-pair hypothesis references available actors.")
        best_result.hypotheses_evaluated = len(summaries)
        best_result.hypothesis_summaries = summaries
        return best_result
