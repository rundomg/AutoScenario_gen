"""Lower fitted parameters to the existing atomic trajectory DSL."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List

from .models import ActorBehaviorParameters, PipelineResult, to_jsonable


CONFIG_SCHEMA_VERSION = "collision-anchored-scenario-v1"


def build_fitted_trajectory_dsl(
    result: PipelineResult,
    *,
    scene_id: str,
    duration_s: float,
    base_dsl: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Replace target actors' VLM numbers while preserving background semantics."""

    base = deepcopy(base_dsl or {})
    target_ids = {
        result.specification.striking_actor_id,
        result.specification.struck_actor_id,
    }
    trajectories = [
        trajectory
        for trajectory in (base.get("trajectories") or [])
        if str(trajectory.get("actor_id")) not in target_ids
    ]
    for actor_id in sorted(target_ids):
        params = result.solver_result.parameters[actor_id]
        maneuver = result.specification.actor_maneuvers.get(actor_id, "keep_lane")
        sequence = result.specification.behavior_sequences.get(actor_id) or ["keep"]
        trajectories.append(
            {
                "actor_id": actor_id,
                "role": (
                    "striking_actor"
                    if actor_id == result.specification.striking_actor_id
                    else "struck_actor"
                ),
                "segments": _segments(params, maneuver, sequence, duration_s),
            }
        )
    ego_params = result.solver_result.parameters.get("ego")
    ego_target_speed = (
        ego_params.initial_speed
        if ego_params is not None
        else float((base.get("ego") or {}).get("target_speed_mps", 10.0))
    )
    return {
        "schema_version": "video-trajectory-dsl-v2",
        "scene_id": scene_id,
        "accident_type": result.specification.accident_type,
        "ego": {"actor_id": "ego", "target_speed_mps": round(ego_target_speed, 3)},
        "duration_s": round(float(duration_s), 3),
        "trajectories": trajectories,
        "metrics": list(base.get("metrics") or ["collision", "min_ttc_s", "min_distance_m"]),
        "metadata": {
            "parameter_source": "collision_anchored_fitting",
            "solver_loss": result.solver_result.loss,
            "solver_converged": result.solver_result.converged,
            "target_collision_pair": sorted(target_ids),
            "collision_anchor": {
                "conflict_position": list(result.anchor.conflict_position),
                "actor_path_distances": dict(result.anchor.actor_path_distances),
                "target_time_range": list(result.anchor.target_time_range),
                "expected_contact_sides": dict(result.anchor.expected_contact_sides),
            },
        },
    }


def build_scenario_configuration(
    result: PipelineResult,
    *,
    scene_id: str,
    map_name: str = None,
) -> Dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "scene_id": scene_id,
        "map_name": map_name,
        "specification": to_jsonable(result.specification),
        "reference_paths": to_jsonable(result.paths),
        "collision_anchor": to_jsonable(result.anchor),
        "solver": to_jsonable(result.solver_result),
        "hypotheses_evaluated": result.hypotheses_evaluated,
        "hypothesis_summaries": to_jsonable(result.hypothesis_summaries),
    }


def apply_start_offsets_to_spawn_payload(
    spawn_payload: Dict[str, Any], result: PipelineResult
) -> Dict[str, Any]:
    """Apply explicit, minimal along-heading offsets needed for valid OBB spawns."""

    payload = deepcopy(spawn_payload)
    offsets = result.anchor.actor_start_offsets
    if not offsets:
        return payload
    for entity in payload.get("entities") or []:
        actor_id = str(entity.get("id") or "")
        offset = offsets.get(actor_id)
        if offset is None:
            continue
        yaw = math.radians(float((entity.get("rotation") or {}).get("yaw", 0.0)))
        location = entity.setdefault("location", {})
        location["x"] = float(location.get("x", 0.0)) - math.cos(yaw) * float(offset)
        location["y"] = float(location.get("y", 0.0)) - math.sin(yaw) * float(offset)
        entity.setdefault("dynamic_fitting", {})["path_start_offset_m"] = float(offset)
    payload.setdefault("dynamic_fitting", {})["actor_start_offsets"] = dict(offsets)
    return payload


def _segments(
    params: ActorBehaviorParameters,
    maneuver: str,
    behavior_sequence: List[str],
    duration_s: float,
) -> List[Dict[str, Any]]:
    duration_s = max(0.1, float(duration_s))
    segments: List[Dict[str, Any]] = []
    cursor = 0.0
    if params.start_delay > 1e-3:
        end = min(duration_s, params.start_delay)
        segments.append(
            {"start_s": 0.0, "end_s": round(end, 3), "action": "hold_position", "confidence": 1.0}
        )
        cursor = end

    lane_start = params.lane_change_start_time
    if maneuver.startswith("lane_change") and lane_start is not None:
        lane_start = clamp_time(lane_start, cursor, duration_s)
        if lane_start > cursor + 1e-3:
            segments.append(_speed_segment(cursor, lane_start, params))
        lane_end = clamp_time(
            lane_start + max(0.5, params.lane_change_duration or 2.0),
            lane_start,
            duration_s,
        )
        segments.append(
            {
                "start_s": round(lane_start, 3),
                "end_s": round(lane_end, 3),
                "action": "lane_change",
                "direction": "left" if "left" in maneuver else "right",
                "lane_count": 1,
                "target_speed_mps": round(max(0.0, params.target_speed), 3),
                "transition_distance_m": round(
                    max(6.0, params.target_speed * max(0.5, lane_end - lane_start)), 3
                ),
                "confidence": 1.0,
            }
        )
        cursor = lane_end
    elif maneuver in {"turn_left", "turn_right", "straight", "u_turn"}:
        segments.append(
            {
                "start_s": round(cursor, 3),
                "end_s": round(duration_s, 3),
                "action": "junction_maneuver",
                "direction": maneuver.replace("turn_", ""),
                "target_speed_mps": round(max(0.0, params.target_speed), 3),
                "confidence": 1.0,
            }
        )
        return segments

    brake_time = params.brake_start_time
    has_deceleration = any(action in {"brake", "stop", "decelerate"} for action in behavior_sequence)
    if has_deceleration and math.isfinite(brake_time):
        brake_time = clamp_time(brake_time, cursor, duration_s)
        if brake_time > cursor + 1e-3:
            segments.append(_speed_segment(cursor, brake_time, params))
        if brake_time < duration_s - 1e-3:
            segments.append(
                {
                    "start_s": round(brake_time, 3),
                    "end_s": round(duration_s, 3),
                    "action": "decelerate_to_speed",
                    "target_speed_mps": round(max(0.0, params.target_speed), 3),
                    "max_decel_mps2": round(max(0.1, params.deceleration), 3),
                    "confidence": 1.0,
                }
            )
        cursor = duration_s
    if cursor < duration_s - 1e-3:
        segments.append(_speed_segment(cursor, duration_s, params))
    if not segments:
        segments.append(_speed_segment(0.0, duration_s, params))
    return _remove_zero_duration(segments)


def _speed_segment(start_s, end_s, params):
    if params.acceleration > 0.0 and params.target_speed > params.initial_speed + 0.1:
        return {
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "action": "accelerate_to_speed",
            "target_speed_mps": round(params.target_speed, 3),
            "max_accel_mps2": round(max(0.1, params.acceleration), 3),
            "confidence": 1.0,
        }
    return {
        "start_s": round(start_s, 3),
        "end_s": round(end_s, 3),
        "action": "lane_follow_speed",
        "target_speed_mps": round(max(0.0, params.initial_speed), 3),
        "confidence": 1.0,
    }


def _remove_zero_duration(segments):
    return [
        segment
        for segment in segments
        if float(segment["end_s"]) > float(segment["start_s"]) + 1e-6
    ]


def clamp_time(value, low, high):
    return max(float(low), min(float(high), float(value)))
