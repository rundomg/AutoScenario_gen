"""Schema + validator + lowering for the video trajectory atomic-behavior DSL.

This is the dynamic counterpart of ``tools/risk_dsl.py``. Where ``risk-dsl-v1``
expresses ONE predicted accident as a flat list of ``{actor_id, trigger, action}``
events, the video trajectory DSL expresses an *observed* accident video as, for
each actor, a time-ordered list of control *segments* (its timeline).

Design decision (see docs/video_accident_dynamic_reconstruction_plan.md review
item P0#2/#3): we do NOT add a second CARLA executor. A validated
``video-trajectory-dsl-v1`` document is *lowered* to ``risk-dsl-v1`` via
``lower_to_risk_dsl`` so the existing
``ExistingWorldScenarioGenerator.build_dsl_risk_scene_script`` /
``DslEventController`` runs it unchanged. ``DslEventController`` already supports
multiple latched events per actor, which is exactly a per-actor timeline.

Version 2 exposes a canonical catalog of bounded execution atoms covering
longitudinal control, lane/route control, closed-loop actor interaction, reverse,
lights, and a strictly shaped ``VehicleControl`` escape hatch. Complex accident
behavior is a time-ordered composition of atoms, never arbitrary Python. Legacy
v1 documents remain valid.

``validate_video_trajectory_dsl`` mirrors the ``(normalized, None)`` /
``(None, error)`` convention of ``validate_risk_dsl``.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tools.vehicle_atomic_behaviors import (
    SUPPORTED_ATOMIC_BEHAVIORS,
    normalize_atomic_behavior,
)


VIDEO_TRAJECTORY_SCHEMA_VERSION = "video-trajectory-dsl-v2"
LEGACY_VIDEO_TRAJECTORY_SCHEMA_VERSION = "video-trajectory-dsl-v1"
RISK_DSL_SCHEMA_VERSION = "risk-dsl-v1"

DEFAULT_EGO_TARGET_SPEED_MPS = 10.0
DEFAULT_DURATION_S = 12.0
DEFAULT_METRICS = ["collision", "min_ttc_s", "min_distance_m"]

# Time tolerance (s) for treating a segment as starting at t=0 (-> immediate).
START_EPS_S = 1e-3

SUPPORTED_ACTIONS = SUPPORTED_ATOMIC_BEHAVIORS


def validate_video_trajectory_dsl(
    dsl: Dict[str, Any],
    spawn_payload: Dict[str, Any],
    *,
    require_all_vehicle_actors: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize a v1/v2 video trajectory document.

    Returns ``(normalized, None)`` on success, otherwise ``(None, error)``.
    """
    if not isinstance(dsl, dict):
        return None, "Video trajectory DSL must be a JSON object."

    normalized = deepcopy(dsl)
    normalized.setdefault("schema_version", VIDEO_TRAJECTORY_SCHEMA_VERSION)
    if normalized.get("schema_version") not in {
        LEGACY_VIDEO_TRAJECTORY_SCHEMA_VERSION,
        VIDEO_TRAJECTORY_SCHEMA_VERSION,
    }:
        return None, (
            f"Unsupported schema: {normalized.get('schema_version')} "
            f"(expected {VIDEO_TRAJECTORY_SCHEMA_VERSION})."
        )

    actor_ids = _collect_actor_ids(spawn_payload)
    if "ego" not in actor_ids:
        return None, "Spawn payload must contain actor id `ego`."

    ego = normalized.setdefault("ego", {})
    if not isinstance(ego, dict):
        return None, "`ego` must be an object."
    ego.setdefault("actor_id", "ego")
    if ego.get("actor_id") != "ego":
        return None, "Video trajectory DSL only supports ego.actor_id=`ego`."
    ego["target_speed_mps"] = _to_float(
        ego.get("target_speed_mps"), DEFAULT_EGO_TARGET_SPEED_MPS
    )
    if ego["target_speed_mps"] < 0.0:
        return None, "ego.target_speed_mps must be non-negative."

    duration_s = _to_float(normalized.get("duration_s"), DEFAULT_DURATION_S)
    if duration_s <= 0.0:
        return None, "duration_s must be positive."
    normalized["duration_s"] = duration_s

    trajectories = normalized.get("trajectories")
    if trajectories is None:
        trajectories = []
    if not isinstance(trajectories, list):
        return None, "`trajectories` must be a list when present."

    normalized_trajectories: List[Dict[str, Any]] = []
    trajectory_actor_ids = set()
    for t_index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            return None, f"trajectories[{t_index}] must be an object."
        actor_id = str(trajectory.get("actor_id") or "")
        if not actor_id:
            return None, f"trajectories[{t_index}] is missing actor_id."
        if actor_id not in actor_ids:
            return None, (
                f"trajectories[{t_index}] references unknown actor id: {actor_id}"
            )
        if actor_id in trajectory_actor_ids:
            return None, f"Duplicate trajectory for actor id: {actor_id}."
        trajectory_actor_ids.add(actor_id)

        segments = trajectory.get("segments")
        if not isinstance(segments, list) or not segments:
            return None, (
                f"trajectories[{t_index}] ({actor_id}) must have a non-empty "
                "`segments` list."
            )

        normalized_segments, seg_error = _normalize_segments(
            segments, actor_id, duration_s, actor_ids
        )
        if seg_error is not None:
            return None, seg_error

        normalized_trajectories.append(
            {
                "actor_id": actor_id,
                "role": str(trajectory.get("role") or "participant"),
                "segments": normalized_segments,
            }
        )

    if require_all_vehicle_actors:
        required_vehicle_ids = _collect_non_ego_vehicle_ids(spawn_payload)
        missing_actor_ids = sorted(required_vehicle_ids - trajectory_actor_ids)
        if missing_actor_ids:
            return None, (
                "Missing trajectories for non-ego vehicle actors: "
                + ", ".join(missing_actor_ids)
                + ". Classify every surrounding vehicle as moving "
                "(lane_follow_speed) or stationary (hold_position)."
            )

    normalized["trajectories"] = normalized_trajectories
    normalized.setdefault("metrics", list(DEFAULT_METRICS))
    return normalized, None


def _collect_non_ego_vehicle_ids(spawn_payload: Dict[str, Any]) -> set:
    vehicle_ids = set()
    for entity in spawn_payload.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        actor_id = str(entity.get("id") or "")
        if not actor_id or actor_id == "ego":
            continue
        spawn_kind = str(entity.get("spawn_kind") or "vehicle").lower()
        category = str(entity.get("category") or "").lower()
        if spawn_kind == "vehicle" or category in {
            "car",
            "truck",
            "bus",
            "van",
            "motorcycle",
            "bicycle",
            "vehicle",
        }:
            vehicle_ids.add(actor_id)
    return vehicle_ids


def _normalize_segments(
    segments: List[Any],
    actor_id: str,
    duration_s: float,
    actor_ids: set,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    normalized: List[Dict[str, Any]] = []
    for s_index, segment in enumerate(segments):
        where = f"trajectories[{actor_id}].segments[{s_index}]"
        if not isinstance(segment, dict):
            return None, f"{where} must be an object."

        start_s = _to_float(segment.get("start_s"), None)
        end_s = _to_float(segment.get("end_s"), None)
        if start_s is None or end_s is None:
            return None, f"{where} must define numeric start_s and end_s."
        if start_s < 0.0:
            return None, f"{where} start_s must be >= 0."
        if end_s < start_s:
            return None, f"{where} end_s must be >= start_s."
        if start_s > duration_s:
            return None, (
                f"{where} start_s={start_s} exceeds duration_s={duration_s}."
            )

        normalized_segment: Dict[str, Any] = {
            "start_s": round(start_s, 3),
            "end_s": round(min(end_s, duration_s), 3),
            "confidence": _to_float(segment.get("confidence"), 0.5),
        }
        source_frames = segment.get("source_frames")
        if isinstance(source_frames, list):
            normalized_segment["source_frames"] = source_frames

        atomic, param_error = normalize_atomic_behavior(
            segment,
            where=where,
            actor_ids=actor_ids,
        )
        if param_error is not None or atomic is None:
            return None, param_error
        normalized_segment.update(atomic)

        normalized.append(normalized_segment)

    normalized.sort(key=lambda seg: seg["start_s"])
    # Reject strict temporal overlap within one actor (touching boundaries ok).
    for previous, current in zip(normalized, normalized[1:]):
        if current["start_s"] < previous["end_s"] - START_EPS_S:
            return None, (
                f"trajectories[{actor_id}] has overlapping segments: "
                f"[{previous['start_s']},{previous['end_s']}] and "
                f"[{current['start_s']},{current['end_s']}]."
            )
    return normalized, None


def lower_to_risk_dsl(
    trajectory_dsl: Dict[str, Any],
    ego_speed_mps: Optional[float] = None,
) -> Dict[str, Any]:
    """Lower a validated video trajectory program to ``risk-dsl-v1`` events.

    Each actor segment becomes one ``risk-dsl-v1`` event whose trigger is
    ``immediate`` (segment starting at t=0) or ``time_elapsed_above`` at the
    segment ``start_s``. Events are emitted in ascending ``start_s`` order so
    that when several of an actor's events are latched at once, the most recent
    segment is applied last and therefore wins (matching ``DslEventController``
    semantics).
    """
    ego_spec = trajectory_dsl.get("ego") or {}
    target_speed = (
        float(ego_speed_mps)
        if ego_speed_mps is not None
        else _to_float(ego_spec.get("target_speed_mps"), DEFAULT_EGO_TARGET_SPEED_MPS)
    )

    actors: List[Dict[str, Any]] = []
    indexed_events: List[Tuple[float, Dict[str, Any]]] = []
    for trajectory in trajectory_dsl.get("trajectories") or []:
        actor_id = str(trajectory.get("actor_id"))
        actors.append({"actor_id": actor_id, "role": trajectory.get("role", "participant")})
        for segment in trajectory.get("segments") or []:
            start_s = _to_float(segment.get("start_s"), 0.0)
            if start_s <= START_EPS_S:
                trigger = {"type": "immediate"}
            else:
                trigger = {"type": "time_elapsed_above", "value_s": round(start_s, 3)}
            indexed_events.append(
                (
                    start_s,
                    {
                        "actor_id": actor_id,
                        "trigger": trigger,
                        "active_duration_s": round(
                            max(
                                0.0,
                                _to_float(segment.get("end_s"), start_s) - start_s,
                            ),
                            3,
                        ),
                        "action": _lower_action(segment),
                    },
                )
            )

    indexed_events.sort(key=lambda item: item[0])
    events = [event for _start, event in indexed_events]

    risk_dsl: Dict[str, Any] = {
        "schema_version": RISK_DSL_SCHEMA_VERSION,
        "scene_id": trajectory_dsl.get("scene_id"),
        "accident_type": trajectory_dsl.get("accident_type", "video_reconstruction"),
        "ego": {
            "actor_id": "ego",
            "target_speed_mps": target_speed,
            "behavior": "drive_forward",
        },
        "actors": actors,
        "events": events,
        "duration_s": _to_float(trajectory_dsl.get("duration_s"), DEFAULT_DURATION_S),
        "metrics": list(trajectory_dsl.get("metrics") or DEFAULT_METRICS),
        "metadata": {
            **dict(trajectory_dsl.get("metadata") or {}),
            "lowered_from": trajectory_dsl.get(
                "schema_version", VIDEO_TRAJECTORY_SCHEMA_VERSION
            ),
        },
    }
    return risk_dsl


def _lower_action(segment: Dict[str, Any]) -> Dict[str, Any]:
    action = segment.get("action")
    if action == "lane_follow_speed":
        lowered = {
            "type": "set_speed",
            "speed_mps": _to_float(segment.get("target_speed_mps"), 6.0),
        }
    elif action == "brake":
        lowered = {
            "type": "brake",
            "intensity": _to_float(segment.get("intensity"), 0.8),
        }
    elif action in {"stop", "hold_position"}:
        lowered = {"type": "stop"}
        if bool(segment.get("hand_brake", action == "hold_position")):
            lowered["hand_brake"] = True
    elif action == "steer_offset":
        duration_s = max(0.1, _to_float(segment.get("end_s"), 0.0) - _to_float(segment.get("start_s"), 0.0))
        lowered = {
            "type": "steer",
            "steer": _to_float(segment.get("steer"), 0.25),
            "throttle": _to_float(segment.get("throttle"), 0.35),
            "brake": _to_float(segment.get("brake"), 0.0),
            "duration_s": round(duration_s, 3),
        }
    else:
        # v2 risk events use the same canonical atom names/parameters. Timing and
        # evidence fields belong to the trajectory layer and are intentionally
        # omitted from the executable action.
        lowered = {
            key: deepcopy(value)
            for key, value in segment.items()
            if key
            not in {
                "start_s",
                "end_s",
                "confidence",
                "source_frames",
                "action",
            }
        }
        lowered["type"] = str(action)
    if segment.get("lights") is not None:
        lowered["lights"] = list(segment.get("lights") or [])
    return lowered


def _collect_actor_ids(spawn_payload: Dict[str, Any]) -> set:
    return {
        str(entity.get("id"))
        for entity in (spawn_payload.get("entities", []) if isinstance(spawn_payload, dict) else [])
        if isinstance(entity, dict) and entity.get("id") is not None
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _to_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
