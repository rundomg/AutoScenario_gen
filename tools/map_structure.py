"""Build a ``matched_structure`` from a chosen CARLA waypoint.

This is the CARLA-facing half of the structural-matching redesign (Phase 1).
The map matcher selects a candidate waypoint; this module turns that waypoint
into the structural description that :mod:`tools.reference_frame` consumes:

* a junction structure -- the junction centre plus its enumerated legs (each
  leg's outward heading, and whether it carries inbound / outbound driving
  lanes), or
* a road-segment structure -- a lane's id, forward heading at the ego point and
  a sampled centreline (``curve_samples``) so curved roads bend correctly.

Keeping this separate from ``reference_frame`` means the geometry/placement core
stays unit-testable without CARLA, while the (CARLA-dependent) extraction lives
here and is validated against a live server.

Coordinate convention matches CARLA: yaw clockwise in degrees, x east, y south.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _normalize_angle(deg: float) -> float:
    angle = float(deg) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


def _angle_difference(a: float, b: float) -> float:
    return abs(_normalize_angle(a - b))


def _circular_mean(headings: List[float]) -> float:
    if not headings:
        return 0.0
    sx = sum(math.sin(math.radians(h)) for h in headings)
    cx = sum(math.cos(math.radians(h)) for h in headings)
    return _normalize_angle(math.degrees(math.atan2(sx, cx)))


def _cluster_headings(
    items: List[Tuple[float, str, Optional[Dict[str, Any]]]], tolerance_deg: float = 25.0
) -> List[Dict[str, Any]]:
    """Cluster (heading, role, lane) triples by heading into legs.

    Each returned cluster is ``{"heading": mean, "roles": set(...), "lanes": [...]}``.
    """
    clusters: List[Dict[str, Any]] = []
    for item in items:
        heading, role = item[0], item[1]
        lane = item[2] if len(item) > 2 else None
        placed = False
        for cluster in clusters:
            if _angle_difference(heading, cluster["_mean"]) <= tolerance_deg:
                cluster["_headings"].append(heading)
                cluster["roles"].add(role)
                if lane is not None:
                    cluster["lanes"].append(lane)
                cluster["_mean"] = _circular_mean(cluster["_headings"])
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "_headings": [heading],
                    "roles": {role},
                    "lanes": [lane] if lane is not None else [],
                    "_mean": heading,
                }
            )
    result = []
    for cluster in clusters:
        result.append(
            {
                "heading": _circular_mean(cluster["_headings"]),
                "roles": cluster["roles"],
                "lanes": cluster["lanes"],
            }
        )
    return result


def _waypoint_anchor_record(waypoint: Any, center: Tuple[float, float, float]) -> Dict[str, Any]:
    loc = waypoint.transform.location
    yaw = float(getattr(waypoint.transform.rotation, "yaw", 0.0) or 0.0)
    dx = float(loc.x) - float(center[0])
    dy = float(loc.y) - float(center[1])
    dz = float(loc.z) - float(center[2])
    return {
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "yaw": yaw,
        "distance_from_center_m": math.sqrt(dx * dx + dy * dy + dz * dz),
    }


def _sample_lane_anchor(
    waypoint: Any,
    role: str,
    center: Tuple[float, float, float],
    sample_distance_m: float = 12.0,
) -> Dict[str, Any]:
    sample = waypoint
    try:
        if role == "inbound":
            candidates = waypoint.previous(sample_distance_m)
        else:
            candidates = waypoint.next(sample_distance_m)
        if candidates:
            sample = candidates[0]
    except Exception:
        sample = waypoint
    return _waypoint_anchor_record(sample, center)


def _lane_record(
    waypoint: Any,
    role: str,
    center: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    record = {
        "road_id": int(getattr(waypoint, "road_id", 0) or 0),
        "lane_id": int(getattr(waypoint, "lane_id", 0) or 0),
        "role": role,
        "yaw": float(getattr(waypoint.transform.rotation, "yaw", 0.0) or 0.0),
    }
    if center is not None:
        record["anchor"] = _sample_lane_anchor(waypoint, role, center)
    return record


def enumerate_junction_legs(
    junction: Any, lane_type: Any = None
) -> Tuple[Tuple[float, float, float], List[Dict[str, Any]]]:
    """Return ``(center, legs)`` for a ``carla.Junction``.

    ``legs`` is a list of ``{heading_out_deg, has_inbound, has_outbound}``.
    The outward heading points from the junction centre along the leg; a vehicle
    approaching the junction on that leg travels at ``heading_out_deg + 180``.
    """
    if lane_type is None:
        import carla

        lane_type = carla.LaneType.Driving

    bb = junction.bounding_box
    center = (bb.location.x, bb.location.y, bb.location.z)

    observations: List[Tuple[float, str, Dict[str, Any]]] = []
    for entry, exit_wp in junction.get_waypoints(lane_type):
        # entry yaw points INTO the junction -> outward leg heading is +180.
        observations.append((
            _normalize_angle(entry.transform.rotation.yaw + 180.0),
            "in",
            _lane_record(entry, "inbound", center),
        ))
        # exit yaw points OUT of the junction -> already the leg's outward heading.
        observations.append((
            _normalize_angle(exit_wp.transform.rotation.yaw),
            "out",
            _lane_record(exit_wp, "outbound", center),
        ))

    legs = []
    for cluster in _cluster_headings(observations):
        lanes = cluster.get("lanes") or []
        legs.append(
            {
                "heading_out_deg": cluster["heading"],
                "has_inbound": "in" in cluster["roles"],
                "has_outbound": "out" in cluster["roles"],
                "inbound_lanes": [lane for lane in lanes if lane.get("role") == "inbound"],
                "outbound_lanes": [lane for lane in lanes if lane.get("role") == "outbound"],
            }
        )
    legs.sort(key=lambda leg: leg["heading_out_deg"])
    return center, legs


def build_junction_structure(
    waypoint: Any, ego_inbound_yaw: Optional[float] = None, lane_type: Any = None
) -> Optional[Dict[str, Any]]:
    """Build a junction ``matched_structure`` from a junction waypoint."""
    junction = waypoint.get_junction()
    if junction is None:
        return None
    center, legs = enumerate_junction_legs(junction, lane_type)
    if not legs:
        return None
    bb = junction.bounding_box
    junction_radius = float(max(getattr(bb.extent, "x", 0.0), getattr(bb.extent, "y", 0.0)))
    ego_yaw = (
        float(ego_inbound_yaw)
        if ego_inbound_yaw is not None
        else float(waypoint.transform.rotation.yaw)
    )
    # Name legs by their relation to ego for readable debugging. assign_leg in
    # reference_frame recomputes the mapping geometrically; names are advisory.
    ego_leg_out = _normalize_angle(ego_yaw + 180.0)
    named_legs = []
    for i, leg in enumerate(legs):
        rel = _leg_relation_to_ego(leg["heading_out_deg"], ego_yaw, ego_leg_out)
        named_legs.append(
            {
                "name": rel or f"leg_{i}",
                "heading_out_deg": leg["heading_out_deg"],
                "has_inbound": leg["has_inbound"],
                "has_outbound": leg["has_outbound"],
                "inbound_lanes": leg.get("inbound_lanes", []),
                "outbound_lanes": leg.get("outbound_lanes", []),
            }
        )
    return {
        "kind": "junction",
        "junction_id": int(junction.id),
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "ego_approach_heading_deg": ego_yaw,
        "legs": named_legs,
        "leg_count": len(named_legs),
        "junction_radius_m": junction_radius,
    }


def _leg_relation_to_ego(
    leg_out: float, ego_inbound_yaw: float, ego_leg_out: float
) -> Optional[str]:
    if _angle_difference(leg_out, ego_leg_out) <= 35.0:
        return "ego"
    if _angle_difference(leg_out, ego_inbound_yaw) <= 35.0:
        return "opposite"
    if _angle_difference(leg_out, _normalize_angle(ego_leg_out + 90.0)) <= 45.0:
        return "left"
    if _angle_difference(leg_out, _normalize_angle(ego_leg_out - 90.0)) <= 45.0:
        return "right"
    return None


def build_road_structure(
    waypoint: Any, lookahead_m: float = 45.0, step_m: float = 1.0
) -> Dict[str, Any]:
    """Build a road-segment ``matched_structure`` from a non-junction waypoint.

    Samples the lane centreline forward up to ``lookahead_m`` so the placement
    frame can follow a bend (``curve_samples`` = ``[[s, heading], ...]``).  The
    sample step is fine (1 m) because reference-frame placement integrates these
    tangents: a coarse step accumulates lateral drift on sharp curves and can
    land an actor in the opposing lane (validated: 5 m drifts ~0.3 m and snaps
    to the wrong lane, 1 m stays on the correct lane).  Sampling runs once per
    matched candidate, so the extra points are cheap.
    """
    loc = waypoint.transform.location
    forward = float(waypoint.transform.rotation.yaw)
    samples: List[List[float]] = [[0.0, forward]]
    current = waypoint
    s = 0.0
    while s < lookahead_m:
        nxt = current.next(step_m)
        if not nxt:
            break
        current = nxt[0]
        s += step_m
        samples.append([s, float(current.transform.rotation.yaw)])
        if current.is_junction:
            break
    return {
        "kind": "road_segment",
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "origin": {"x": loc.x, "y": loc.y, "z": loc.z},
        "forward_heading_deg": forward,
        "curve_samples": samples,
        "s_range": [0.0, s],
    }


def build_matched_structure_from_waypoint(
    waypoint: Any,
    ego_inbound_yaw: Optional[float] = None,
    junction_lookahead_m: float = 40.0,
    lane_type: Any = None,
) -> Dict[str, Any]:
    """Dispatch: junction structure if the waypoint is at/near a junction,
    otherwise a road-segment structure.

    If the waypoint is an approach lane with a junction a short distance ahead,
    we still anchor on that junction (the common "approaching an intersection"
    case) so ego is placed by distance-to-centre rather than on an arbitrary
    approach point.
    """
    if waypoint.is_junction:
        structure = build_junction_structure(waypoint, ego_inbound_yaw, lane_type)
        if structure is not None:
            return structure

    # Look a short way ahead for a junction (approaching case).
    current = waypoint
    travelled = 0.0
    step = 5.0
    while travelled < junction_lookahead_m:
        nxt = current.next(step)
        if not nxt:
            break
        current = nxt[0]
        travelled += step
        if current.is_junction:
            structure = build_junction_structure(
                current,
                ego_inbound_yaw=float(waypoint.transform.rotation.yaw),
                lane_type=lane_type,
            )
            if structure is not None:
                structure["ego_distance_to_center_m"] = travelled
                return structure
            break

    return build_road_structure(waypoint)
