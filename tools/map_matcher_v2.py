"""Small topology-first map matcher for CARLA candidate caches.

The v1 matcher blended topology, lane, signal, median, and environment clues
into one score.  This module keeps the decision order explicit:

1. Normalize the VLM road-topology signature.
2. Classify every CARLA candidate into a structural pool.
3. Hard-gate incompatible pools.
4. Rank only compatible candidates with a few bounded terms.

It is intentionally CARLA-free so both cache and live-waypoint candidates can
use the same logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


OPEN_MIN_CONFIDENCE = 0.52
JUNCTION_MIN_CONFIDENCE = 0.50
JUNCTION_RATIO_CUTOFF = 0.18
JUNCTION_APPROACH_REACH_M = 45.0
SIGNAL_HARD_MATCH_REACH_M = 60.0


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "present", "visible"}:
            return True
        if lowered in {"false", "no", "absent", "none", "not_visible"}:
            return False
    return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _norm_topology(value: Any) -> str:
    return str(value or "unknown").strip().lower()


def _junction_type(topology_type: str, junction_type: Any) -> str:
    raw = _norm_topology(junction_type)
    topo = _norm_topology(topology_type)
    text = raw or topo
    if "roundabout" in text or "rotary" in text:
        return "roundabout"
    if "multi" in text or "complex" in text:
        return "multi"
    if "cross" in text or "four" in text:
        return "cross"
    if "t_junction" in text or "t-junction" in text or "three" in text:
        return "t"
    if topo == "t_junction":
        return "t"
    if topo == "cross_intersection":
        return "cross"
    if topo == "multi_branch":
        return "multi"
    return "none"


def build_target_signature(signature: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the existing road_topology_signature into the v2 target."""
    topology_type = _norm_topology(signature.get("topology_type"))
    junction_visible = _as_bool(signature.get("junction_visible"))
    left_parking = _as_bool(signature.get("left_parking_presence"))
    right_parking = _as_bool(signature.get("right_parking_presence"))
    junction_topologies = {
        "t_junction",
        "cross_intersection",
        "multi_branch",
        "roundabout",
    }
    open_topologies = {"straight_road", "straight_two_way", "curve"}

    if junction_visible is True or topology_type in junction_topologies:
        scene_kind = "junction"
    elif junction_visible is False and topology_type in open_topologies:
        scene_kind = "open_road"
    else:
        branches = signature.get("junction_branches") or {}
        if isinstance(branches, dict) and branches.get("known") and (
            branches.get("left") or branches.get("right")
        ):
            scene_kind = "junction"
        else:
            scene_kind = "unknown"

    curve_direction = _norm_topology(signature.get("curve_direction"))
    if topology_type == "curve":
        if curve_direction in {"left", "right"}:
            road_shape = f"curve_{curve_direction}"
        else:
            road_shape = "curve_unknown"
    elif topology_type in {"straight_road", "straight_two_way"}:
        road_shape = "straight"
    else:
        road_shape = "unknown"

    env = signature.get("environment_context") or {}
    side = signature.get("side_context") or {}
    expects_urban = bool(env.get("expects_urban"))
    expects_natural = bool(env.get("expects_natural"))
    if expects_urban:
        environment_class = "urban"
    elif expects_natural:
        environment_class = "natural"
    else:
        environment_class = "unknown"

    return {
        "scene_kind": scene_kind,
        "road_shape": road_shape,
        "junction_type": _junction_type(topology_type, signature.get("junction_type")),
        "branches": signature.get("junction_branches") or {
            "ahead": False,
            "left": False,
            "right": False,
            "known": False,
        },
        "forward_lane_count": _as_int(signature.get("forward_lane_count")),
        "opposing_lane_count": _as_int(signature.get("opposing_lane_count")),
        "driving_lane_count": _as_int(signature.get("driving_lane_count")),
        "ego_lane_from_right": _as_int(signature.get("ego_lane_from_right")),
        "has_center_median": _as_bool(signature.get("has_center_median")),
        "has_crosswalk": _as_bool(signature.get("has_crosswalk")),
        "has_traffic_light": _as_bool(signature.get("has_traffic_light")),
        "ego_to_junction_distance_m": _as_float(
            signature.get("ego_to_junction_distance_m")
        ),
        "environment_class": environment_class,
        "expects_buildings": bool(
            env.get("expects_buildings")
            or side.get("left_continuous_buildings")
            or side.get("right_continuous_buildings")
        ),
        "expects_sidewalks": bool(env.get("expects_sidewalks")),
        "left_parking_presence": left_parking,
        "right_parking_presence": right_parking,
        "parking_presence_known": left_parking is not None or right_parking is not None,
        "raw_topology_type": topology_type,
    }


def classify_candidate(candidate: Dict[str, Any]) -> str:
    junction_ratio = float(candidate.get("junction_waypoint_ratio") or 0.0)
    heading_clusters = int(candidate.get("heading_cluster_count") or 1)
    nearby_roads = int(candidate.get("nearby_road_count") or 1)
    distance_to_junction = candidate.get("distance_to_junction_ahead")

    if bool(candidate.get("is_junction")) or junction_ratio >= JUNCTION_RATIO_CUTOFF:
        return "junction"
    if (
        isinstance(distance_to_junction, (int, float))
        and 0.0 <= float(distance_to_junction) <= JUNCTION_APPROACH_REACH_M
        and (heading_clusters >= 3 or nearby_roads >= 3)
    ):
        return "junction_approach"
    if heading_clusters <= 2 and nearby_roads <= 2 and junction_ratio < 0.12:
        return "open_road"
    if heading_clusters <= 3 and nearby_roads <= 2 and junction_ratio < 0.12:
        return "open_road"
    return "invalid"


def _lane_fit(target_value: Optional[int], candidate_value: Any, tolerance: int = 1) -> float:
    if target_value is None:
        return 0.7
    candidate_int = _as_int(candidate_value)
    if candidate_int is None:
        return 0.5
    delta = abs(candidate_int - int(target_value))
    if delta == 0:
        return 1.0
    if delta <= tolerance:
        return 0.72
    return max(0.0, 0.45 - 0.18 * (delta - tolerance))


def _median_fit(target: Optional[bool], candidate: Any) -> float:
    if target is None:
        return 0.7
    if candidate is None:
        return 0.55
    return 1.0 if bool(candidate) == bool(target) else 0.25


def _parking_fit(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    if not target.get("parking_presence_known"):
        return 0.70
    left_actual = bool(candidate.get("left_parking_lane_present"))
    right_actual = bool(candidate.get("right_parking_lane_present"))
    side_scores: List[float] = []
    for expected, actual in (
        (target.get("left_parking_presence"), left_actual),
        (target.get("right_parking_presence"), right_actual),
    ):
        if expected is None:
            side_scores.append(0.70)
        elif bool(expected):
            side_scores.append(1.0 if actual else 0.45)
        else:
            side_scores.append(1.0 if not actual else 0.15)
    return sum(side_scores) / max(1, len(side_scores))


def _junction_parking_context(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    if not target.get("parking_presence_known"):
        return 0.70
    if target.get("left_parking_presence") or target.get("right_parking_presence"):
        return _parking_fit(target, candidate)
    return 0.70


def _traffic_light_distance(candidate: Dict[str, Any]) -> Optional[float]:
    ahead = candidate.get("distance_to_traffic_light_ahead")
    if isinstance(ahead, (int, float)):
        return float(ahead)
    env = candidate.get("environment_context") or {}
    nearest = env.get("nearest_m") or {}
    value = nearest.get("TrafficLight")
    return float(value) if isinstance(value, (int, float)) else None


def _context_fit(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    env = candidate.get("environment_context") or {}
    if not env:
        return 0.55
    score = 0.55
    urban_score = float(env.get("urban_score") or 0.0)
    natural_score = float(env.get("natural_score") or 0.0)
    if target.get("environment_class") == "urban":
        score += 0.25 * urban_score
        if target.get("expects_buildings"):
            score += 0.10 if env.get("buildings_nearby") else -0.08
        if target.get("expects_sidewalks"):
            score += 0.08 if env.get("sidewalks_nearby") else -0.05
    elif target.get("environment_class") == "natural":
        score += 0.25 * natural_score
        if env.get("buildings_nearby"):
            score -= 0.08
    return _clamp(score)


def _curve_fit(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    shape = target.get("road_shape")
    candidate_is_curve = bool(candidate.get("is_curve"))
    curve_score = float(candidate.get("curve_score") or 0.0)
    candidate_direction = _norm_topology(candidate.get("curve_direction"))
    if shape == "straight":
        return 0.25 if candidate_is_curve and curve_score >= 0.45 else 1.0
    if shape == "curve_unknown":
        return 0.75 + 0.25 * curve_score if candidate_is_curve else 0.25
    if shape in {"curve_left", "curve_right"}:
        wanted = shape.split("_", 1)[1]
        if not candidate_is_curve:
            return 0.20
        if candidate_direction == wanted:
            return 1.0
        if candidate_direction in {"left", "right"}:
            return 0.35
        return 0.72
    return 0.65


def _branch_direction_fit(
    target_branches: Dict[str, Any], candidate: Dict[str, Any]
) -> Tuple[float, List[str]]:
    if not isinstance(target_branches, dict) or not target_branches.get("known"):
        return 0.75, []
    candidate_dirs = candidate.get("physical_junction_arms")
    if isinstance(candidate_dirs, dict) and candidate_dirs.get("known") is False:
        candidate_dirs = None
    if not isinstance(candidate_dirs, dict):
        # Older caches only have ``junction_branch_dirs``, which describes the
        # current lane's reachable maneuvers.  It cannot prove that a physical
        # left/right/ahead arm exists, so a junction target must not accept it.
        return 0.0, ["physical_junction_arms_missing"]
    errors = []
    for side in ("ahead", "left", "right"):
        target_has = bool(target_branches.get(side))
        candidate_has = bool(candidate_dirs.get(side))
        if target_has and not candidate_has:
            errors.append(f"missing_required_{side}_branch")
        elif candidate_has and not target_has:
            errors.append(f"extra_forbidden_{side}_branch")
    if not errors:
        return 1.0, []
    missing = any(item.startswith("missing_required") for item in errors)
    extra = any(item.startswith("extra_forbidden") for item in errors)
    if missing and extra:
        return 0.0, errors
    if missing:
        return 0.25, errors
    return 0.45, errors


def _junction_type_fit(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    target_type = target.get("junction_type")
    if target_type in {None, "none"}:
        return 0.7
    degree = int(candidate.get("estimated_junction_degree") or 1)
    candidate_topology = _norm_topology(candidate.get("candidate_topology_type"))
    if target_type == "t":
        if candidate_topology == "t_junction" or degree == 3:
            return 1.0
        if candidate_topology == "cross_intersection" or degree == 4:
            return 0.55
        return 0.25
    if target_type == "cross":
        if candidate_topology == "cross_intersection" or degree == 4:
            return 1.0
        if candidate_topology == "t_junction" or degree == 3:
            return 0.45
        return 0.25
    if target_type == "multi":
        return 1.0 if degree >= 5 or candidate_topology == "multi_branch" else 0.45
    return 0.7


def _ego_junction_distance_fit(target: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    target_distance = target.get("ego_to_junction_distance_m")
    if not isinstance(target_distance, (int, float)):
        return 0.75
    candidate_distance = candidate.get("distance_to_junction_ahead")
    if not isinstance(candidate_distance, (int, float)):
        return 0.55
    error = abs(float(candidate_distance) - float(target_distance))
    if error <= 8.0:
        return 1.0
    if error <= 20.0:
        return 0.72
    return 0.35


def _junction_lane_hard_mismatches(
    target: Dict[str, Any], candidate: Dict[str, Any]
) -> List[str]:
    reasons: List[str] = []
    target_forward = target.get("forward_lane_count")
    target_total = target.get("driving_lane_count")
    candidate_forward = _as_int(candidate.get("same_direction_lane_count"))
    candidate_total = _as_int(candidate.get("same_road_lane_count"))

    if (
        isinstance(target_forward, int)
        and target_forward >= 2
        and candidate_forward is not None
        and candidate_forward < target_forward
    ):
        reasons.append(
            "junction target rejects candidate with too few approach lanes "
            f"(target_forward_lane_count={target_forward}, "
            f"candidate_same_direction_lane_count={candidate_forward})"
        )
    if (
        isinstance(target_total, int)
        and target_total >= 3
        and candidate_total is not None
        and candidate_total < target_total - 1
    ):
        reasons.append(
            "junction target rejects candidate with severe driving-lane mismatch "
            f"(target_driving_lane_count={target_total}, "
            f"candidate_driving_lane_count={candidate_total})"
        )
    return reasons


def _open_road_lane_hard_mismatches(
    target: Dict[str, Any], candidate: Dict[str, Any]
) -> List[str]:
    reasons: List[str] = []
    target_forward = target.get("forward_lane_count")
    target_total = target.get("driving_lane_count")
    candidate_forward = _as_int(candidate.get("same_direction_lane_count"))
    candidate_total = _as_int(candidate.get("same_road_lane_count"))

    if isinstance(target_forward, int):
        if candidate_forward is None:
            reasons.append(
                "open-road target rejects candidate without same-direction lane count "
                f"(target_forward_lane_count={target_forward})"
            )
        elif candidate_forward != target_forward:
            reasons.append(
                "open-road target rejects same-direction lane mismatch "
                f"(target_forward_lane_count={target_forward}, "
                f"candidate_same_direction_lane_count={candidate_forward})"
            )
    if isinstance(target_total, int):
        if candidate_total is None:
            reasons.append(
                "open-road target rejects candidate without driving-lane count "
                f"(target_driving_lane_count={target_total})"
            )
        elif candidate_total != target_total:
            reasons.append(
                "open-road target rejects driving-lane mismatch "
                f"(target_driving_lane_count={target_total}, "
                f"candidate_driving_lane_count={candidate_total})"
            )
    return reasons


def _curve_open_road_lane_hard_mismatches(
    target: Dict[str, Any], candidate: Dict[str, Any]
) -> List[str]:
    """Reject clearly over-wide curve candidates for narrow open-road targets.

    Curved roads are harder to count from a single image, so keep this narrower
    than the straight-road exact gate.  The important case is preventing a 1+1
    undivided curve from matching a 2+2 boulevard just because the curve and
    side context score well.
    """
    reasons: List[str] = []
    target_forward = target.get("forward_lane_count")
    target_opposing = target.get("opposing_lane_count")
    target_total = target.get("driving_lane_count")
    candidate_forward = _as_int(candidate.get("same_direction_lane_count"))
    candidate_total = _as_int(candidate.get("same_road_lane_count"))

    narrow_two_way_target = (
        isinstance(target_forward, int)
        and isinstance(target_opposing, int)
        and target_forward <= 1
        and target_opposing >= 1
    )
    if (
        narrow_two_way_target
        and candidate_forward is not None
        and candidate_forward > target_forward
    ):
        reasons.append(
            "curve open-road target rejects same-direction lane mismatch "
            f"(target_forward_lane_count={target_forward}, "
            f"candidate_same_direction_lane_count={candidate_forward})"
        )
    if (
        isinstance(target_total, int)
        and target_total <= 2
        and candidate_total is not None
        and candidate_total > target_total + 1
    ):
        reasons.append(
            "curve open-road target rejects overly wide driving-lane mismatch "
            f"(target_driving_lane_count={target_total}, "
            f"candidate_driving_lane_count={candidate_total})"
        )
    return reasons


def _open_road_median_hard_mismatches(
    target: Dict[str, Any], candidate: Dict[str, Any]
) -> List[str]:
    expected = target.get("has_center_median")
    if not isinstance(expected, bool):
        return []
    actual = candidate.get("has_center_median_candidate")
    if actual is None:
        return [
            "open-road target rejects candidate without center-median evidence "
            f"(target_has_center_median={expected})"
        ]
    if bool(actual) != expected:
        return [
            "open-road target rejects center-median mismatch "
            f"(target_has_center_median={expected}, "
            f"candidate_has_center_median={bool(actual)})"
        ]
    return []


def gate_candidate(target: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    kind = classify_candidate(candidate)
    scene_kind = target.get("scene_kind")
    reasons: List[str] = []

    if kind == "invalid":
        reasons.append("candidate structure is too ambiguous")
    if scene_kind in {"open_road", "unknown"}:
        if kind in {"junction", "junction_approach"}:
            reasons.append(f"open-road target rejects {kind} candidate")
        if bool(candidate.get("is_junction")):
            reasons.append("open-road target rejects candidate inside junction")
        if float(candidate.get("junction_waypoint_ratio") or 0.0) >= JUNCTION_RATIO_CUTOFF:
            reasons.append("open-road target rejects high junction waypoint ratio")
        if target.get("road_shape") == "straight":
            reasons.extend(_open_road_lane_hard_mismatches(target, candidate))
            reasons.extend(_open_road_median_hard_mismatches(target, candidate))
        elif str(target.get("road_shape") or "").startswith("curve_"):
            reasons.extend(_curve_open_road_lane_hard_mismatches(target, candidate))
            reasons.extend(_open_road_median_hard_mismatches(target, candidate))
        if (
            target.get("parking_presence_known")
            and target.get("left_parking_presence") is False
            and target.get("right_parking_presence") is False
        ):
            if bool(candidate.get("left_parking_lane_present")):
                reasons.append("open-road target rejects candidate with left parking lane")
            if bool(candidate.get("right_parking_lane_present")):
                reasons.append("open-road target rejects candidate with right parking lane")
    elif scene_kind == "junction":
        if kind not in {"junction", "junction_approach"}:
            reasons.append(f"junction target rejects {kind} candidate")
        branch_score, branch_errors = _branch_direction_fit(target.get("branches") or {}, candidate)
        if branch_errors:
            reasons.extend(
                f"junction target rejects branch mismatch: {item}"
                for item in branch_errors
            )
        reasons.extend(_junction_lane_hard_mismatches(target, candidate))
        if target.get("has_traffic_light") is True:
            tl_dist = _traffic_light_distance(candidate)
            if tl_dist is None or tl_dist > SIGNAL_HARD_MATCH_REACH_M:
                reasons.append("signalized junction target rejects no-signal candidate")

    return not reasons, kind, reasons


def score_candidate_v2(
    signature: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    blacklisted: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    target = build_target_signature(signature)
    gate_ok, candidate_kind, reject_reasons = gate_candidate(target, candidate)
    if blacklisted:
        gate_ok = False
        reject_reasons = list(reject_reasons) + ["candidate is inside a blacklisted rematch region"]

    components: Dict[str, float] = {}
    if target["scene_kind"] in {"open_road", "unknown"}:
        components = {
            "topology": 1.0 if candidate_kind == "open_road" else 0.0,
            "curve": _curve_fit(target, candidate),
            "forward_lanes": _lane_fit(
                target.get("forward_lane_count"),
                candidate.get("same_direction_lane_count"),
            ),
            "driving_lanes": _lane_fit(
                target.get("driving_lane_count"),
                candidate.get("same_road_lane_count"),
            ),
            "median": _median_fit(
                target.get("has_center_median"),
                candidate.get("has_center_median_candidate"),
            ),
            "parking": _parking_fit(target, candidate),
            "context": _context_fit(target, candidate),
        }
        score = (
            0.24 * components["topology"]
            + 0.20 * components["curve"]
            + 0.16 * components["forward_lanes"]
            + 0.12 * components["driving_lanes"]
            + 0.09 * components["median"]
            + 0.10 * components["parking"]
            + 0.09 * components["context"]
        )
        if candidate.get("has_highway_shoulder") and (target.get("driving_lane_count") or 0) <= 2:
            score *= 0.65
        if target.get("has_traffic_light") is False:
            tl_dist = _traffic_light_distance(candidate)
            if tl_dist is not None and tl_dist <= 45.0:
                score *= 0.80
        min_confidence = OPEN_MIN_CONFIDENCE
    else:
        branch_score, branch_errors = _branch_direction_fit(target.get("branches") or {}, candidate)
        components = {
            "topology": 1.0 if candidate_kind in {"junction", "junction_approach"} else 0.0,
            "junction_type": _junction_type_fit(target, candidate),
            "branch_direction": branch_score,
            "ego_distance": _ego_junction_distance_fit(target, candidate),
            "approach_lanes": _lane_fit(
                target.get("forward_lane_count"),
                candidate.get("same_direction_lane_count"),
            ),
            "median": _median_fit(
                target.get("has_center_median"),
                candidate.get("has_center_median_candidate"),
            ),
            "context": _context_fit(target, candidate),
            "parking": _junction_parking_context(target, candidate),
        }
        if branch_errors and not reject_reasons:
            reject_reasons = list(branch_errors)
        score = (
            0.21 * components["topology"]
            + 0.20 * components["junction_type"]
            + 0.22 * components["branch_direction"]
            + 0.13 * components["ego_distance"]
            + 0.10 * components["approach_lanes"]
            + 0.05 * components["median"]
            + 0.06 * components["context"]
            + 0.03 * components["parking"]
        )
        if target.get("has_traffic_light") is True:
            tl_dist = _traffic_light_distance(candidate)
            if tl_dist is not None and tl_dist <= 35.0:
                score = min(1.0, score * 1.05)
            elif tl_dist is None or tl_dist > 75.0:
                score *= 0.70
        min_confidence = JUNCTION_MIN_CONFIDENCE

    confidence_ok = score >= min_confidence
    if not confidence_ok:
        reject_reasons = list(reject_reasons) + [
            f"confidence {score:.3f} below threshold {min_confidence:.3f}"
        ]
    hard_reject = (not gate_ok) or (not confidence_ok)

    details = {
        "matcher_version": "v2",
        "target_scene_kind": target.get("scene_kind"),
        "target_road_shape": target.get("road_shape"),
        "target_junction_type": target.get("junction_type"),
        "candidate_kind": candidate_kind,
        "gate_result": "rejected" if hard_reject else "passed",
        "ranking_components": {k: round(v, 3) for k, v in components.items()},
        "reject_reasons": reject_reasons,
        "hard_reject": hard_reject,
        "blacklisted": blacklisted,
        "uncapped_total_score": score,
        "fallback_total_score": score,
    }
    return round(_clamp(score), 6), details


def summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    details = entry.get("score_details") or {}
    return {
        "world": entry.get("_world_name"),
        "score": entry.get("score"),
        "hard_reject": entry.get("hard_reject"),
        "reject_reason": entry.get("reject_reason"),
        "candidate_kind": details.get("candidate_kind"),
        "candidate_topology_type": entry.get("candidate_topology_type"),
        "is_junction": entry.get("is_junction"),
        "junction_waypoint_ratio": entry.get("junction_waypoint_ratio"),
        "same_road_lane_count": entry.get("same_road_lane_count"),
        "same_direction_lane_count": entry.get("same_direction_lane_count"),
        "physical_junction_arms": entry.get("physical_junction_arms"),
        "left_parking_lane_present": entry.get("left_parking_lane_present"),
        "right_parking_lane_present": entry.get("right_parking_lane_present"),
    }
