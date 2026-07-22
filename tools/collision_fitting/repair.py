"""Directed repair actions derived from rollout failure modes."""

from __future__ import annotations

import math
from copy import deepcopy
from enum import Enum
from typing import Dict, List

from .models import (
    AccidentSpecification,
    ActorBehaviorParameters,
    CollisionAnchor,
    RolloutResult,
    clamp,
    pair_equal,
)


class RepairAction(str, Enum):
    INCREASE_STRIKER_SPEED = "increase_striker_speed"
    DECREASE_STRIKER_SPEED = "decrease_striker_speed"
    INCREASE_STRUCK_DECELERATION = "increase_struck_deceleration"
    DECREASE_STRUCK_DECELERATION = "decrease_struck_deceleration"
    DELAY_STRIKER_BRAKE = "delay_striker_brake"
    ADVANCE_STRIKER_BRAKE = "advance_striker_brake"
    DELAY_ACTOR_START = "delay_actor_start"
    ADVANCE_ACTOR_START = "advance_actor_start"
    ADVANCE_LANE_CHANGE = "advance_lane_change"
    LENGTHEN_LANE_CHANGE = "lengthen_lane_change"
    REBUILD_PATH = "rebuild_path"
    ISOLATE_BACKGROUND = "isolate_background"
    TRY_NEXT_COLLISION_PAIR = "try_next_collision_pair"


class DirectedRepairPlanner:
    def propose(
        self,
        spec: AccidentSpecification,
        anchor: CollisionAnchor,
        rollout: RolloutResult,
    ) -> List[Dict[str, object]]:
        target_pair = (spec.striking_actor_id, spec.struck_actor_id)
        if rollout.collided and not pair_equal(rollout.collision_pair, target_pair):
            return [
                {"action": RepairAction.ISOLATE_BACKGROUND.value},
                {"action": RepairAction.TRY_NEXT_COLLISION_PAIR.value},
            ]
        if rollout.collided and rollout.collision_time is not None:
            if rollout.collision_time < anchor.target_time_range[0]:
                return [
                    {"action": RepairAction.DECREASE_STRIKER_SPEED.value, "scale": 0.9},
                    {"action": RepairAction.ADVANCE_STRIKER_BRAKE.value, "delta_s": 0.35},
                ]
            if rollout.collision_time > anchor.target_time_range[1]:
                return [
                    {"action": RepairAction.INCREASE_STRIKER_SPEED.value, "scale": 1.1},
                    {"action": RepairAction.DELAY_STRIKER_BRAKE.value, "delta_s": 0.35},
                ]
            return []
        if spec.accident_type == "rear_end":
            return [
                {"action": RepairAction.INCREASE_STRIKER_SPEED.value, "scale": 1.12},
                {"action": RepairAction.DELAY_STRIKER_BRAKE.value, "delta_s": 0.4},
                {"action": RepairAction.INCREASE_STRUCK_DECELERATION.value, "delta_mps2": 1.0},
            ]
        arrivals = rollout.arrival_times
        time_a, time_b = arrivals.get(spec.striking_actor_id), arrivals.get(spec.struck_actor_id)
        if time_a is not None and time_b is not None:
            early = spec.striking_actor_id if time_a < time_b else spec.struck_actor_id
            return [{"action": RepairAction.DELAY_ACTOR_START.value, "actor_id": early, "delta_s": abs(time_a - time_b)}]
        if spec.accident_type == "lane_change":
            return [
                {"action": RepairAction.ADVANCE_LANE_CHANGE.value, "delta_s": 0.4},
                {"action": RepairAction.LENGTHEN_LANE_CHANGE.value, "delta_s": 0.5},
            ]
        return [{"action": RepairAction.REBUILD_PATH.value}]

    def apply(
        self,
        parameters: Dict[str, ActorBehaviorParameters],
        spec: AccidentSpecification,
        actions: List[Dict[str, object]],
    ) -> Dict[str, ActorBehaviorParameters]:
        updated = deepcopy(parameters)
        striker = updated[spec.striking_actor_id]
        struck = updated[spec.struck_actor_id]
        for item in actions:
            action = str(item.get("action") or "")
            if action == RepairAction.INCREASE_STRIKER_SPEED.value:
                striker.initial_speed = clamp(striker.initial_speed * float(item.get("scale", 1.1)), 0.0, 35.0)
                striker.target_speed = max(striker.target_speed, striker.initial_speed)
            elif action == RepairAction.DECREASE_STRIKER_SPEED.value:
                striker.initial_speed = clamp(striker.initial_speed * float(item.get("scale", 0.9)), 0.0, 35.0)
                striker.target_speed = min(striker.target_speed, striker.initial_speed)
            elif action == RepairAction.INCREASE_STRUCK_DECELERATION.value:
                struck.deceleration = clamp(struck.deceleration + float(item.get("delta_mps2", 1.0)), 0.0, 10.0)
            elif action == RepairAction.DECREASE_STRUCK_DECELERATION.value:
                struck.deceleration = clamp(struck.deceleration - float(item.get("delta_mps2", 1.0)), 0.0, 10.0)
            elif action == RepairAction.DELAY_STRIKER_BRAKE.value:
                if math.isfinite(striker.brake_start_time):
                    striker.brake_start_time += float(item.get("delta_s", 0.3))
            elif action == RepairAction.ADVANCE_STRIKER_BRAKE.value:
                if math.isfinite(striker.brake_start_time):
                    striker.brake_start_time = max(0.0, striker.brake_start_time - float(item.get("delta_s", 0.3)))
            elif action in {RepairAction.DELAY_ACTOR_START.value, RepairAction.ADVANCE_ACTOR_START.value}:
                actor_id = str(item.get("actor_id") or spec.striking_actor_id)
                if actor_id in updated:
                    sign = 1.0 if action == RepairAction.DELAY_ACTOR_START.value else -1.0
                    updated[actor_id].start_delay = max(0.0, updated[actor_id].start_delay + sign * float(item.get("delta_s", 0.3)))
            elif action == RepairAction.ADVANCE_LANE_CHANGE.value:
                if striker.lane_change_start_time is not None:
                    striker.lane_change_start_time = max(0.0, striker.lane_change_start_time - float(item.get("delta_s", 0.3)))
            elif action == RepairAction.LENGTHEN_LANE_CHANGE.value:
                if striker.lane_change_duration is not None:
                    striker.lane_change_duration += float(item.get("delta_s", 0.5))
        return updated
