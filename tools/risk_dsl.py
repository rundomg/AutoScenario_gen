"""Schema + validator for the pure-LLM risk DSL (``risk-dsl-v1``).

Stage 2a of the pure-LLM risk pipeline asks an LLM to turn one predicted accident
candidate plus the reconstructed actor table into a small, structured scenario
DSL. Stage 2b then deterministically turns that DSL into an executable CARLA
Python script. Unlike ``tools/risk_scenario_pipeline.py`` this validator is
intentionally NOT backed by ``tools/accident_template_library.py`` -- the DSL is
free-form (event-driven), so the only constraints are structural: referenced
actor ids must exist in the spawn payload and trigger/action types must be ones
the deterministic generator knows how to execute.

``validate_risk_dsl`` mirrors the return convention of
``validate_risk_scenario_spec``: ``(normalized_dsl, None)`` on success or
``(None, error_message)`` on failure.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tools.vehicle_atomic_behaviors import (
    SUPPORTED_ATOMIC_BEHAVIORS,
    normalize_atomic_behavior,
)


RISK_DSL_SCHEMA_VERSION = "risk-dsl-v1"
DEFAULT_EGO_TARGET_SPEED_MPS = 10.0
DEFAULT_DURATION_S = 12.0
DEFAULT_METRICS = ["collision", "min_ttc_s", "min_distance_m"]

EGO_BEHAVIORS = {"drive_forward"}

TRIGGER_TYPES = {"immediate", "time_elapsed_above", "distance_to_ego_below"}
LEGACY_ACTION_TYPES = {"brake", "set_speed", "accelerate", "steer", "cross", "stop"}
ACTION_TYPES = LEGACY_ACTION_TYPES | set(SUPPORTED_ATOMIC_BEHAVIORS)


def validate_risk_dsl(
    dsl: Dict[str, Any],
    spawn_payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize a ``risk-dsl-v1`` document.

    Returns ``(normalized, None)`` on success, otherwise ``(None, error)``.
    """
    if not isinstance(dsl, dict):
        return None, "Risk DSL must be a JSON object."

    normalized = deepcopy(dsl)
    normalized.setdefault("schema_version", RISK_DSL_SCHEMA_VERSION)
    if normalized.get("schema_version") != RISK_DSL_SCHEMA_VERSION:
        return None, f"Unsupported risk DSL schema: {normalized.get('schema_version')}"

    actor_ids = _collect_actor_ids(spawn_payload)
    if "ego" not in actor_ids:
        return None, "Spawn payload must contain actor id `ego`."

    ego = normalized.setdefault("ego", {})
    if not isinstance(ego, dict):
        return None, "`ego` must be an object."
    ego.setdefault("actor_id", "ego")
    if ego.get("actor_id") != "ego":
        return None, "v1 only supports ego.actor_id=`ego`."
    ego["target_speed_mps"] = _to_float(
        ego.get("target_speed_mps"), DEFAULT_EGO_TARGET_SPEED_MPS
    )
    if ego["target_speed_mps"] < 0.0:
        return None, "ego.target_speed_mps must be non-negative."
    ego.setdefault("behavior", "drive_forward")
    if ego.get("behavior") not in EGO_BEHAVIORS:
        return None, f"ego.behavior must be one of {sorted(EGO_BEHAVIORS)}."

    actors = normalized.get("actors")
    if actors is None:
        actors = []
    if not isinstance(actors, list):
        return None, "`actors` must be a list when present."
    normalized_actors = []
    for index, actor in enumerate(actors):
        if not isinstance(actor, dict):
            return None, f"actors[{index}] must be an object."
        actor_id = str(actor.get("actor_id") or "")
        if not actor_id:
            return None, f"actors[{index}] is missing actor_id."
        if actor_id not in actor_ids:
            return None, f"actors[{index}] references unknown actor id: {actor_id}"
        actor["actor_id"] = actor_id
        actor.setdefault("role", "risk_actor")
        normalized_actors.append(actor)
    normalized["actors"] = normalized_actors

    events = normalized.get("events")
    if not isinstance(events, list) or not events:
        return None, "`events` must be a non-empty list."
    normalized_events = []
    for index, event in enumerate(events):
        normalized_event, error = _normalize_event(event, index, actor_ids)
        if error:
            return None, error
        normalized_events.append(normalized_event)
    normalized["events"] = normalized_events

    normalized["duration_s"] = _to_float(normalized.get("duration_s"), DEFAULT_DURATION_S)
    if normalized["duration_s"] <= 0.0:
        return None, "duration_s must be positive."

    metrics = normalized.get("metrics")
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    if not isinstance(metrics, list):
        return None, "`metrics` must be a list when present."
    normalized["metrics"] = [str(metric) for metric in metrics]

    return normalized, None


def _normalize_event(
    event: Any,
    index: int,
    actor_ids: set,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(event, dict):
        return None, f"events[{index}] must be an object."
    actor_id = str(event.get("actor_id") or "")
    if not actor_id:
        return None, f"events[{index}] is missing actor_id."
    if actor_id == "ego":
        return None, f"events[{index}] must target a non-ego actor."
    if actor_id not in actor_ids:
        return None, f"events[{index}] references unknown actor id: {actor_id}"
    event["actor_id"] = actor_id

    trigger, error = _normalize_trigger(event.get("trigger"), index)
    if error:
        return None, error
    event["trigger"] = trigger

    action, error = _normalize_action(event.get("action"), index, actor_ids)
    if error:
        return None, error
    event["action"] = action
    return event, None


def _normalize_trigger(
    trigger: Any,
    index: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(trigger, dict):
        return None, f"events[{index}].trigger must be an object."
    trigger_type = str(trigger.get("type") or "")
    if trigger_type not in TRIGGER_TYPES:
        return None, (
            f"events[{index}].trigger.type must be one of {sorted(TRIGGER_TYPES)}."
        )
    normalized = {"type": trigger_type}
    if trigger_type == "time_elapsed_above":
        normalized["value_s"] = _to_float(trigger.get("value_s"), 0.0)
        if normalized["value_s"] < 0.0:
            return None, f"events[{index}].trigger.value_s must be non-negative."
    elif trigger_type == "distance_to_ego_below":
        normalized["value_m"] = _to_float(trigger.get("value_m"), 12.0)
        if normalized["value_m"] <= 0.0:
            return None, f"events[{index}].trigger.value_m must be positive."
    return normalized, None


def _normalize_action(
    action: Any,
    index: int,
    actor_ids: set,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(action, dict):
        return None, f"events[{index}].action must be an object."
    action_type = str(action.get("type") or "")
    if action_type not in ACTION_TYPES:
        return None, (
            f"events[{index}].action.type must be one of {sorted(ACTION_TYPES)}."
        )
    if action_type in SUPPORTED_ATOMIC_BEHAVIORS and action_type not in {
        "brake",
        "stop",
    }:
        source = dict(action)
        source["action"] = action_type
        atomic, error = normalize_atomic_behavior(
            source,
            where=f"events[{index}].action",
            actor_ids=actor_ids,
        )
        if error is not None or atomic is None:
            return None, error
        atomic["type"] = atomic.pop("action")
        return atomic, None

    normalized: Dict[str, Any] = {"type": action_type}
    if action_type == "brake":
        normalized["intensity"] = _clamp(_to_float(action.get("intensity"), 0.9), 0.0, 1.0)
    elif action_type in {"set_speed", "accelerate"}:
        normalized["speed_mps"] = _to_float(action.get("speed_mps"), 6.0)
        if normalized["speed_mps"] < 0.0:
            return None, f"events[{index}].action.speed_mps must be non-negative."
    elif action_type in {"steer", "cross"}:
        normalized["steer"] = _clamp(_to_float(action.get("steer"), 0.25), -1.0, 1.0)
        normalized["throttle"] = _clamp(_to_float(action.get("throttle"), 0.35), 0.0, 1.0)
        normalized["duration_s"] = _to_float(action.get("duration_s"), 1.2)
        if normalized["duration_s"] <= 0.0:
            return None, f"events[{index}].action.duration_s must be positive."
    elif action_type == "stop":
        normalized["hand_brake"] = bool(action.get("hand_brake", False))
    lights = action.get("lights")
    if lights is not None:
        source = {"action": "coast", "lights": lights}
        atomic, error = normalize_atomic_behavior(
            source,
            where=f"events[{index}].action",
            actor_ids=actor_ids,
        )
        if error:
            return None, error
        normalized["lights"] = atomic.get("lights")
    return normalized, None


def _collect_actor_ids(spawn_payload: Dict[str, Any]) -> set:
    return {
        str(entity.get("id"))
        for entity in (
            spawn_payload.get("entities", []) if isinstance(spawn_payload, dict) else []
        )
        if isinstance(entity, dict) and entity.get("id") is not None
    }


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
