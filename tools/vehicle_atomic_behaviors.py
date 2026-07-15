"""Canonical atomic vehicle behaviors for deterministic CARLA reconstruction.

The atoms in this module are the complete *road-motion* surface exposed to an
LLM.  They deliberately exclude simulator cheats such as teleporting an actor,
changing physics, or applying an impulse.  A complex maneuver is represented as
a time-ordered composition of these validated atoms; the LLM never writes CARLA
Python or invents a controller name.

An atom owns the vehicle motion channel for its segment.  Turn indicators and
other lights are optional side effects on any atom, so signalling can run in
parallel without introducing overlapping motion segments.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple


ATOMIC_BEHAVIOR_SPECS: Dict[str, Dict[str, str]] = {
    # Longitudinal + lane keeping.
    "lane_follow_speed": {
        "family": "longitudinal",
        "description": "Follow the current lane at a target speed.",
    },
    "accelerate_to_speed": {
        "family": "longitudinal",
        "description": "Follow the lane while accelerating toward a target speed.",
    },
    "decelerate_to_speed": {
        "family": "longitudinal",
        "description": "Follow the lane while decelerating toward a lower target speed.",
    },
    "brake": {
        "family": "longitudinal",
        "description": "Apply a fixed service-brake intensity while preserving heading.",
    },
    "emergency_brake": {
        "family": "longitudinal",
        "description": "Apply full brake immediately.",
    },
    "coast": {
        "family": "longitudinal",
        "description": "Release throttle and brake and let vehicle dynamics coast.",
    },
    "stop": {
        "family": "longitudinal",
        "description": "Brake to zero speed.",
    },
    "hold_position": {
        "family": "longitudinal",
        "description": "Keep the vehicle stationary using brake/hand brake.",
    },
    # Lateral and route motion.
    "steer_offset": {
        "family": "lateral",
        "description": "Apply bounded raw steering briefly for a swerve/encroachment.",
    },
    "lane_change": {
        "family": "lateral",
        "description": "Track a CARLA waypoint route into an adjacent driving lane.",
    },
    "junction_maneuver": {
        "family": "route",
        "description": "Track a left/right/straight/U-turn branch at a junction.",
    },
    "drive_to_location": {
        "family": "route",
        "description": "Steer toward an explicit CARLA world-space target location.",
    },
    "reverse": {
        "family": "route",
        "description": "Drive backward with bounded speed and steering.",
    },
    # Closed-loop interaction atoms. These are atomic from the program's point
    # of view even though their controller evaluates another actor every tick.
    "follow_actor": {
        "family": "interaction",
        "description": "Follow another actor while regulating a desired gap.",
    },
    "approach_actor": {
        "family": "interaction",
        "description": "Close toward another actor to a target gap (zero permits impact).",
    },
    "yield_to_actor": {
        "family": "interaction",
        "description": "Brake/hold while another actor is within a yield distance.",
    },
    # Escape hatch that still has a strict CARLA VehicleControl-shaped schema.
    "raw_vehicle_control": {
        "family": "low_level",
        "description": "Apply bounded throttle/steer/brake/gear VehicleControl values.",
    },
}

SUPPORTED_ATOMIC_BEHAVIORS = frozenset(ATOMIC_BEHAVIOR_SPECS)
LIGHT_STATES = frozenset(
    {
        "none",
        "position",
        "low_beam",
        "high_beam",
        "brake",
        "right_blinker",
        "left_blinker",
        "reverse",
        "fog",
        "interior",
        "special1",
        "special2",
        "all",
    }
)


def behavior_catalog_for_prompt() -> str:
    """Return a compact, stable catalog suitable for an LLM prompt."""
    lines = []
    for name, spec in ATOMIC_BEHAVIOR_SPECS.items():
        lines.append(f"- {name} [{spec['family']}]: {spec['description']}")
    return "\n".join(lines)


def normalize_atomic_behavior(
    segment: Dict[str, Any],
    *,
    where: str,
    actor_ids: Iterable[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize the behavior-specific portion of one segment."""
    actor_ids = {str(actor_id) for actor_id in actor_ids}
    action = str(segment.get("action") or "")
    if action not in SUPPORTED_ATOMIC_BEHAVIORS:
        return None, (
            f"{where} has unknown action/atomic behavior `{action}`; allowed: "
            f"{sorted(SUPPORTED_ATOMIC_BEHAVIORS)}."
        )
    normalized: Dict[str, Any] = {"action": action}

    if action == "lane_follow_speed":
        error = _positive_speed(segment, normalized, where, "target_speed_mps")
        if error:
            return None, error
    elif action == "accelerate_to_speed":
        error = _positive_speed(segment, normalized, where, "target_speed_mps")
        if error:
            return None, error
        normalized["max_accel_mps2"] = _bounded_float(
            segment.get("max_accel_mps2"), 3.0, 0.1, 12.0
        )
    elif action == "decelerate_to_speed":
        error = _positive_speed(segment, normalized, where, "target_speed_mps")
        if error:
            return None, error
        normalized["max_decel_mps2"] = _bounded_float(
            segment.get("max_decel_mps2"), 5.0, 0.1, 15.0
        )
    elif action == "brake":
        normalized["intensity"] = _bounded_float(
            segment.get("intensity"), 0.8, 0.0, 1.0
        )
    elif action in {"emergency_brake", "coast"}:
        pass
    elif action in {"stop", "hold_position"}:
        normalized["hand_brake"] = bool(
            segment.get("hand_brake", action == "hold_position")
        )
    elif action == "steer_offset":
        normalized["steer"] = _bounded_float(segment.get("steer"), 0.25, -1.0, 1.0)
        normalized["throttle"] = _bounded_float(
            segment.get("throttle"), 0.35, 0.0, 1.0
        )
        normalized["brake"] = _bounded_float(segment.get("brake"), 0.0, 0.0, 1.0)
        normalized["duration_s"] = _bounded_float(
            segment.get("duration_s"), 1.2, 0.05, 10.0
        )
    elif action == "lane_change":
        direction = str(segment.get("direction") or "")
        if direction not in {"left", "right"}:
            return None, f"{where} lane_change.direction must be `left` or `right`."
        normalized["direction"] = direction
        lane_count = _to_int(segment.get("lane_count"), 1)
        if lane_count < 1 or lane_count > 3:
            return None, f"{where} lane_change.lane_count must be between 1 and 3."
        normalized["lane_count"] = lane_count
        normalized["target_speed_mps"] = _bounded_float(
            segment.get("target_speed_mps"), 6.0, 0.0, 60.0
        )
        normalized["transition_distance_m"] = _bounded_float(
            segment.get("transition_distance_m"), 18.0, 6.0, 80.0
        )
    elif action == "junction_maneuver":
        direction = str(segment.get("direction") or "")
        if direction not in {"left", "right", "straight", "u_turn"}:
            return None, (
                f"{where} junction_maneuver.direction must be left/right/straight/u_turn."
            )
        normalized["direction"] = direction
        normalized["target_speed_mps"] = _bounded_float(
            segment.get("target_speed_mps"), 5.0, 0.0, 30.0
        )
        normalized["route_distance_m"] = _bounded_float(
            segment.get("route_distance_m"), 45.0, 10.0, 150.0
        )
    elif action == "drive_to_location":
        target = segment.get("target")
        if not isinstance(target, dict):
            return None, f"{where} drive_to_location.target must be an object."
        try:
            normalized["target"] = {
                "x": float(target["x"]),
                "y": float(target["y"]),
                "z": float(target.get("z", 0.0)),
            }
        except (KeyError, TypeError, ValueError):
            return None, f"{where} target requires numeric x/y and optional z."
        normalized["target_speed_mps"] = _bounded_float(
            segment.get("target_speed_mps"), 5.0, 0.0, 30.0
        )
        normalized["acceptance_radius_m"] = _bounded_float(
            segment.get("acceptance_radius_m"), 1.5, 0.2, 10.0
        )
    elif action == "reverse":
        normalized["target_speed_mps"] = _bounded_float(
            segment.get("target_speed_mps"), 2.0, 0.0, 10.0
        )
        normalized["steer"] = _bounded_float(segment.get("steer"), 0.0, -1.0, 1.0)
    elif action in {"follow_actor", "approach_actor", "yield_to_actor"}:
        target_actor_id = str(segment.get("target_actor_id") or "")
        if not target_actor_id or target_actor_id not in actor_ids:
            return None, (
                f"{where} {action}.target_actor_id must reference a real actor; "
                f"got `{target_actor_id}`."
            )
        normalized["target_actor_id"] = target_actor_id
        if action == "follow_actor":
            normalized["desired_gap_m"] = _bounded_float(
                segment.get("desired_gap_m"), 8.0, 0.5, 100.0
            )
            normalized["max_speed_mps"] = _bounded_float(
                segment.get("max_speed_mps"), 15.0, 0.0, 60.0
            )
        elif action == "approach_actor":
            normalized["target_gap_m"] = _bounded_float(
                segment.get("target_gap_m"), 0.0, 0.0, 100.0
            )
            normalized["approach_speed_mps"] = _bounded_float(
                segment.get("approach_speed_mps"), 10.0, 0.0, 60.0
            )
            travel_direction = str(segment.get("travel_direction") or "forward")
            if travel_direction not in {"forward", "reverse"}:
                return None, (
                    f"{where} approach_actor.travel_direction must be forward or reverse."
                )
            normalized["travel_direction"] = travel_direction
            normalized["steer"] = _bounded_float(
                segment.get("steer"), 0.0, -1.0, 1.0
            )
        else:
            normalized["yield_distance_m"] = _bounded_float(
                segment.get("yield_distance_m"), 12.0, 0.5, 100.0
            )
            normalized["resume_speed_mps"] = _bounded_float(
                segment.get("resume_speed_mps"), 5.0, 0.0, 30.0
            )
    elif action == "raw_vehicle_control":
        normalized["throttle"] = _bounded_float(
            segment.get("throttle"), 0.0, 0.0, 1.0
        )
        normalized["steer"] = _bounded_float(segment.get("steer"), 0.0, -1.0, 1.0)
        normalized["brake"] = _bounded_float(segment.get("brake"), 0.0, 0.0, 1.0)
        normalized["hand_brake"] = bool(segment.get("hand_brake", False))
        normalized["reverse"] = bool(segment.get("reverse", False))
        normalized["manual_gear_shift"] = bool(
            segment.get("manual_gear_shift", False)
        )
        normalized["gear"] = _to_int(segment.get("gear"), 0)

    lights = segment.get("lights")
    if lights is not None:
        if not isinstance(lights, list) or any(
            str(light).lower() not in LIGHT_STATES for light in lights
        ):
            return None, f"{where}.lights must contain only {sorted(LIGHT_STATES)}."
        normalized["lights"] = [str(light).lower() for light in lights]
    return deepcopy(normalized), None


def _positive_speed(
    source: Dict[str, Any],
    target: Dict[str, Any],
    where: str,
    field: str,
) -> Optional[str]:
    value = _to_float(source.get(field), None)
    if value is None or value < 0.0:
        return f"{where} `{source.get('action')}` requires {field} to be non-negative."
    target[field] = round(value, 3)
    return None


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    number = _to_float(value, default)
    return round(max(low, min(high, number)), 3)


def _to_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
