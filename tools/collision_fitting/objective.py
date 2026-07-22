"""Priority-preserving objective for accident reconstruction rollouts."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from .models import (
    AccidentSpecification,
    ActorBehaviorParameters,
    CollisionAnchor,
    RolloutResult,
    pair_equal,
)
from .paths import distance


class CollisionObjective:
    """Large separated tiers prevent a well-timed wrong crash from winning."""

    def evaluate(
        self,
        spec: AccidentSpecification,
        anchor: CollisionAnchor,
        rollout: RolloutResult,
        parameters: Dict[str, ActorBehaviorParameters],
    ) -> Tuple[float, Dict[str, float]]:
        target_pair = (spec.striking_actor_id, spec.struck_actor_id)
        wrong_pair = rollout.collided and not pair_equal(rollout.collision_pair, target_pair)
        components = {
            "wrong_collision_pair": 1_000_000.0 if wrong_pair else 0.0,
            "non_target_collision": 500_000.0 * len(rollout.non_target_collisions),
            "wrong_accident_type": 0.0,
            "target_collision_missing": 0.0,
            "collision_time": 0.0,
            "collision_location": 0.0,
            "relative_relations": 0.0,
            "physical_plausibility": self._physical_penalty(parameters),
            "control_smoothness": max(0.0, rollout.control_smoothness),
        }
        expected_sides = anchor.expected_contact_sides
        observed_sides = rollout.contact_sides
        if rollout.collided and expected_sides and observed_sides:
            mismatches = sum(
                1
                for actor_id, expected in expected_sides.items()
                if observed_sides.get(actor_id) is not None
                and str(observed_sides.get(actor_id)) != str(expected)
            )
            components["wrong_accident_type"] = 200_000.0 * mismatches
        if not rollout.collided:
            miss = rollout.minimum_pair_distance
            if not math.isfinite(miss):
                miss = 100.0
            components["target_collision_missing"] = 100_000.0 + min(100.0, miss) * 100.0
        elif not wrong_pair:
            if rollout.collision_time is not None:
                low, high = anchor.target_time_range
                if rollout.collision_time < low:
                    time_error = low - rollout.collision_time
                elif rollout.collision_time > high:
                    time_error = rollout.collision_time - high
                else:
                    time_error = 0.0
                components["collision_time"] = time_error * 2_000.0
            if rollout.collision_location is not None and spec.accident_type != "rear_end":
                components["collision_location"] = min(
                    100.0, distance(rollout.collision_location, anchor.conflict_position)
                ) * 300.0
        relation_error = sum(
            abs(float(value))
            for value in rollout.relation_errors.values()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        )
        components["relative_relations"] = relation_error * 100.0
        return sum(components.values()), components

    @staticmethod
    def _physical_penalty(parameters: Dict[str, ActorBehaviorParameters]) -> float:
        penalty = 0.0
        for params in parameters.values():
            penalty += max(0.0, -params.initial_speed) * 5_000.0
            penalty += max(0.0, params.initial_speed - 35.0) ** 2 * 20.0
            penalty += max(0.0, params.acceleration - 5.0) ** 2 * 20.0
            penalty += max(0.0, params.deceleration - 10.0) ** 2 * 20.0
            penalty += max(0.0, -params.start_delay) * 5_000.0
        return penalty
