"""Re-place actors of a junction scene inside the matched junction frame.

Phase-4 wiring (junction scenes only).  The legacy pipeline smears every actor
along a single anchor-lane axis, which puts cross-traffic sideways in front of
ego and cannot represent the separate arms of an intersection.  When the map
matcher returns a ``matched_structure`` of kind ``junction`` we instead rebuild
each actor's position and heading inside the junction reference frame:

* ego is placed back along its approach leg by its distance to the centre;
* oncoming / crossing / turning actors are assigned to the real leg implied by
  their ego-relative heading and placed on it, facing that leg's legal flow
  (flipped only for genuine wrong-way actors);
* an actor whose implied leg does not exist on this junction is left untouched
  and flagged, rather than fabricated sideways in front of ego.

This module is pure Python (no CARLA): it consumes the structural description
the matcher already produced, so it is unit-testable offline and the result is
spawn-validated separately.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from tools.reference_frame import (
    JunctionFrame,
    RoadFrame,
    build_reference_frame,
    check_heading_consistency,
)

DEFAULT_LANE_WIDTH_M = 3.5
DEFAULT_EGO_DISTANCE_M = 12.0
MIN_LEG_DISTANCE_M = 3.0
MIN_VEHICLE_LEG_SPACING_M = 5.5


def _slug(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def reproject_actors_for_structure(
    coordinates: Dict[str, Any],
    matched_structure: Optional[Dict[str, Any]],
    *,
    lane_width: float = DEFAULT_LANE_WIDTH_M,
) -> Dict[str, Any]:
    """Dispatch reprojection by ``matched_structure`` kind (junction / road)."""
    if not isinstance(matched_structure, dict):
        return coordinates
    kind = str(matched_structure.get("kind") or "").lower()
    if kind == "junction":
        return reproject_actors_for_junction(
            coordinates, matched_structure, lane_width=lane_width
        )
    if kind in {"road_segment", "road", "straight", "curve"}:
        return reproject_actors_for_road(
            coordinates, matched_structure, lane_width=lane_width
        )
    return coordinates


def _is_ego(entity: Dict[str, Any]) -> bool:
    return (
        str(entity.get("id") or "").lower() == "ego"
        or str(entity.get("priority") or "").lower() == "ego"
        or str(entity.get("source") or "").lower() == "ego"
    )


def direction_and_motion_for_entity(
    entity: Dict[str, Any]
) -> Tuple[str, str]:
    """Map an actor's ego-relative heading + turn intent to a junction leg.

    Returns ``(direction, motion)`` where ``direction`` is one of
    ``ego`` / ``opposite`` / ``left`` / ``right`` (consumed by
    :meth:`JunctionFrame.assign_leg`) and ``motion`` is ``approaching`` or
    ``leaving`` (which side of the centre the actor's legal flow points).
    """
    if _is_ego(entity):
        return "ego", "approaching"

    layout_anchor = _slug(entity.get("layout_anchor_id"))
    anchor_relation = entity.get("anchor_relation") or {}
    if isinstance(anchor_relation, dict) and layout_anchor:
        direction = {
            "ego_approach": "ego",
            "ego": "ego",
            "oncoming_arm": "opposite",
            "opposite_arm": "opposite",
            "ahead_arm": "opposite",
            "left_arm": "left",
            "right_arm": "right",
        }.get(layout_anchor)
        travel = _slug(
            anchor_relation.get("travel_direction")
            or anchor_relation.get("direction")
            or anchor_relation.get("motion")
        )
        motion = None
        if travel in {"toward_junction", "towards_junction", "approaching", "inbound", "entering"}:
            motion = "approaching"
        elif travel in {"away_from_junction", "leaving", "outbound", "exiting"}:
            motion = "leaving"
        if motion is None:
            # Without an explicit travel direction, default an actor on the arm
            # ahead/across to leaving and any other arm to approaching.
            motion = "leaving" if layout_anchor == "ahead_arm" else "approaching"
        if direction is not None:
            return direction, motion

    heading = str(entity.get("heading_relation") or "unknown").lower()
    turn = str(entity.get("turn_intent") or "none").lower()
    lane_index = entity.get("lane_index_relation")
    try:
        lane_index = int(lane_index)
    except (TypeError, ValueError):
        lane_index = 0

    if heading in {"opposite_direction", "opposite", "oncoming"}:
        # Oncoming traffic sits on the leg across the junction, heading toward ego.
        return "opposite", "approaching"

    if heading == "crossing":
        # Cross traffic enters from a side leg. Prefer turn intent, then the
        # signed lane index, to pick which side.
        if turn == "left":
            return "left", "approaching"
        if turn == "right":
            return "right", "approaching"
        return ("right", "approaching") if lane_index >= 0 else ("left", "approaching")

    # same_direction (or unknown): traveling ego's way. A turn intent means the
    # actor is peeling onto a side leg (leaving the junction); otherwise it stays
    # on ego's approach leg ahead of ego.
    if turn == "left":
        return "left", "leaving"
    if turn == "right":
        return "right", "leaving"
    return "ego", "approaching"


def _leg_distance_for_entity(
    entity: Dict[str, Any],
    is_ego: bool,
    ego_distance_m: float,
    clearance_m: float = MIN_LEG_DISTANCE_M,
) -> float:
    """Distance from the junction centre to place this actor along its leg.

    ``clearance_m`` is a floor (the junction's own radius plus a margin) so an
    approaching actor sits on the approach lane outside the junction box rather
    than inside the intersection where lanes are ill-defined.
    """
    floor = max(MIN_LEG_DISTANCE_M, float(clearance_m))
    if is_ego:
        return max(floor, float(ego_distance_m))
    anchor_relation = entity.get("anchor_relation") or {}
    if isinstance(anchor_relation, dict):
        for key in (
            "distance_to_junction_m",
            "distance_from_junction_m",
            "junction_distance_m",
        ):
            try:
                distance = abs(float(anchor_relation.get(key)))
            except (TypeError, ValueError):
                continue
            if distance > 0:
                return max(floor, distance)
        position = _slug(
            anchor_relation.get("position_along_anchor")
            or anchor_relation.get("position")
        )
        if position in {"near_mouth", "entering", "at_mouth", "mouth"}:
            return max(floor, 6.0)
        if position in {"mid_arm", "middle", "mid"}:
            return max(floor, 18.0)
        if position in {"far_arm", "far"}:
            return max(floor, 38.0)
    longitudinal = entity.get("longitudinal_m")
    try:
        longitudinal = abs(float(longitudinal))
    except (TypeError, ValueError):
        longitudinal = None
    if longitudinal is not None and longitudinal > 0:
        return max(floor, longitudinal)
    return max(floor, float(ego_distance_m))


def _lanes_for_motion(leg: Any, motion: str) -> List[Dict[str, Any]]:
    lanes = leg.lanes_for_motion(motion) if hasattr(leg, "lanes_for_motion") else []
    # OpenDRIVE lane ids grow outward from the reference line.  The rightmost
    # lane on a carriageway is therefore the largest |lane_id|, not lane 1.
    return sorted(lanes, key=lambda lane: -abs(int(lane.get("lane_id", 0) or 0)))


def _entity_lane_index(entity: Dict[str, Any]) -> int:
    try:
        return int(entity.get("lane_index_relation") or 0)
    except (TypeError, ValueError):
        return 0


def _anchor_lane_from_right(entity: Dict[str, Any]) -> Optional[int]:
    anchor_relation = entity.get("anchor_relation")
    if not isinstance(anchor_relation, dict):
        return None
    for key in ("lane_from_right", "lane_index_from_right"):
        try:
            value = int(anchor_relation.get(key))
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _lateral_for_entity(
    entity: Dict[str, Any],
    leg: Any,
    motion: str,
    lane_width: float,
) -> Tuple[float, Optional[Dict[str, Any]], int]:
    """Lateral offset placing the actor in its ego-relative lane band.

    The half-lane offset to the correct side of the *outbound* heading is
    load-bearing for heading fidelity, not a fine placement knob: for right-hand
    traffic an inbound (approaching) actor -- whose travel is opposite the
    outbound heading -- sits at ``-0.5`` lane widths and an outbound (leaving)
    actor at ``+0.5``.  ``lane_index_relation`` is then applied in the actor's
    travel frame, so right-lane and left-lane image evidence does not collapse
    onto the same junction leg centreline.
    """
    lanes = _lanes_for_motion(leg, motion)
    explicit_from_right = _anchor_lane_from_right(entity)
    heading = _slug(entity.get("heading_relation"))
    lane_index = 0 if heading == "crossing" else _entity_lane_index(entity)
    lane_slot = explicit_from_right if explicit_from_right is not None else abs(lane_index)
    spacing_slot = lane_slot if explicit_from_right is not None else lane_index
    if lanes:
        selected_lane = lanes[min(max(0, lane_slot), len(lanes) - 1)]
    else:
        selected_lane = None

    lane_center_offset = 0.5 * float(lane_width)
    if str(motion or "approaching").lower() == "leaving":
        # Right of travel is right of the outbound leg heading.
        lane_center_offset += lane_index * float(lane_width)
    else:
        # Approaching actors travel opposite the outbound leg heading, so their
        # right side is negative in the leg's outbound-lateral frame.
        lane_center_offset = -lane_center_offset
        lane_center_offset -= lane_index * float(lane_width)
    return lane_center_offset, selected_lane, spacing_slot


def _spaced_leg_distance(
    base_distance: float,
    *,
    slots: Dict[Tuple[str, str, int], List[float]],
    leg_name: str,
    motion: str,
    lane_from_right: int,
    is_ego: bool,
) -> float:
    if is_ego:
        return base_distance

    key = (str(leg_name), str(motion), int(lane_from_right))
    distance = float(base_distance)
    occupied = slots.setdefault(key, [])
    while any(abs(distance - other) < MIN_VEHICLE_LEG_SPACING_M for other in occupied):
        distance += MIN_VEHICLE_LEG_SPACING_M
    occupied.append(distance)
    return distance


def reproject_actors_for_junction(
    coordinates: Dict[str, Any],
    matched_structure: Dict[str, Any],
    *,
    lane_width: float = DEFAULT_LANE_WIDTH_M,
    ego_distance_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Rewrite ``coordinates['entities']`` placements using the junction frame.

    Non-junction structures are returned unchanged.  Entities whose implied leg
    is absent keep their original placement and gain ``junction_placement`` =
    ``"unassigned"`` for inspection.
    """
    if str(matched_structure.get("kind") or "").lower() != "junction":
        return coordinates

    frame = build_reference_frame(matched_structure)
    if not isinstance(frame, JunctionFrame) or not frame.legs:
        return coordinates

    ego_dist = (
        float(ego_distance_m)
        if ego_distance_m is not None
        else float(matched_structure.get("ego_distance_to_center_m") or DEFAULT_EGO_DISTANCE_M)
    )
    # Keep approaching actors on the approach lane outside the junction box.
    clearance = float(matched_structure.get("junction_radius_m") or 0.0) + 4.0
    ego_heading = frame.assign_leg("ego")
    ego_heading_deg = (
        frame.legs and frame.ego_approach_heading_deg
    )  # ego inbound heading for the consistency guard

    assignments: List[Dict[str, Any]] = []
    leg_slots: Dict[Tuple[str, str, int], List[float]] = {}
    for entity in coordinates.get("entities") or []:
        is_ego = _is_ego(entity)
        direction, motion = direction_and_motion_for_entity(entity)
        leg = frame.assign_leg(direction)
        record = {"id": entity.get("id"), "direction": direction, "motion": motion}
        if leg is None:
            entity["junction_placement"] = "unassigned"
            record["leg"] = None
            assignments.append(record)
            continue

        z = float((entity.get("location") or {}).get("z", 0.3) or 0.3)
        lateral_m, selected_lane, lane_slot = _lateral_for_entity(
            entity, leg, motion, lane_width
        )
        distance_m = _spaced_leg_distance(
            _leg_distance_for_entity(entity, is_ego, ego_dist, clearance),
            slots=leg_slots,
            leg_name=leg.name,
            motion=motion,
            lane_from_right=lane_slot,
            is_ego=is_ego,
        )
        placement = frame.place(
            leg=leg,
            distance_m=distance_m,
            lateral_m=lateral_m,
            motion=motion,
            flow_compliance=entity.get("flow_compliance", "legal"),
            z=z,
        )
        if placement is None:
            entity["junction_placement"] = "unassigned"
            record["leg"] = None
            assignments.append(record)
            continue

        entity["location"] = dict(placement.location)
        entity["rotation"] = {"pitch": 0.0, "yaw": placement.yaw, "roll": 0.0}
        entity["junction_placement"] = "frame"
        entity["junction_direction"] = direction
        entity["junction_leg"] = leg.name
        entity["junction_motion"] = motion
        entity["junction_distance_m"] = distance_m
        entity["layout_version"] = entity.get("layout_version") or "v2"
        entity["layout_scene_kind"] = "junction"
        entity["placement_reason"] = "junction arm placement"
        if selected_lane is not None:
            entity["projected_lane"] = {
                "road_id": selected_lane.get("road_id"),
                "lane_id": selected_lane.get("lane_id"),
                "role": selected_lane.get("role"),
                "yaw": selected_lane.get("yaw"),
                "anchor": selected_lane.get("anchor"),
                "source": "junction_leg",
            }

        ok, reason = check_heading_consistency(
            yaw=placement.yaw,
            ego_heading=float(ego_heading_deg or 0.0),
            heading_relation=entity.get("heading_relation"),
            flow_compliance=entity.get("flow_compliance"),
        )
        entity["heading_consistency"] = reason
        record.update({
            "leg": leg.name,
            "yaw": round(placement.yaw, 1),
            "ok": ok,
            "lateral_m": round(lateral_m, 3),
            "lane_slot": lane_slot,
            "selected_lane": selected_lane,
        })
        assignments.append(record)

    coordinates.setdefault("metadata", {})["junction_reprojection"] = {
        "junction_id": matched_structure.get("junction_id"),
        "leg_count": matched_structure.get("leg_count"),
        "ego_distance_to_center_m": ego_dist,
        "assignments": assignments,
    }
    coordinates.setdefault("metadata", {})["layout_version"] = "v2"
    coordinates.setdefault("metadata", {})["layout_scene_kind"] = "junction"
    return coordinates


def _decompose_offsets(
    loc: Dict[str, Any], origin: Dict[str, Any], forward_heading_deg: float
) -> Tuple[float, float]:
    """Split a world point into (longitudinal, lateral) in the straight anchor
    frame: longitudinal along ``forward_heading``, lateral to its right."""
    dx = float(loc.get("x", 0.0)) - float(origin.get("x", 0.0))
    dy = float(loc.get("y", 0.0)) - float(origin.get("y", 0.0))
    frad = math.radians(forward_heading_deg)
    fx, fy = math.cos(frad), math.sin(frad)
    rx, ry = math.cos(frad + math.pi / 2.0), math.sin(frad + math.pi / 2.0)
    return dx * fx + dy * fy, dx * rx + dy * ry


def reproject_actors_for_road(
    coordinates: Dict[str, Any],
    matched_structure: Dict[str, Any],
    *,
    lane_width: float = DEFAULT_LANE_WIDTH_M,
) -> Dict[str, Any]:
    """Re-lay actors of a straight/curved road along the real lane centreline.

    Strategy: keep each actor's existing (longitudinal, lateral) offsets -- the
    legacy pipeline already resolves opposing-carriageway and median geometry --
    but re-lay them along the matched ``curve_samples`` so a curved road bends
    instead of drifting off a fixed axis, and derive heading from the local lane
    tangent (flipped only for genuine wrong-way actors).  Genuine wrong-way
    actors are pulled back into their own carriageway band (the legacy path,
    unaware of wrong-way, would have pushed them across to the opposing side).

    On a straight road the re-lay is geometrically a no-op; the only behavioural
    change is tangent-based heading and wrong-way handling, so the battle-tested
    straight-road layout is preserved.
    """
    if str(matched_structure.get("kind") or "").lower() not in {
        "road_segment", "road", "straight", "curve",
    }:
        return coordinates

    frame = build_reference_frame(matched_structure)
    if not isinstance(frame, RoadFrame):
        return coordinates

    origin = matched_structure.get("origin") or {}
    forward_heading = float(matched_structure.get("forward_heading_deg", 0.0) or 0.0)

    assignments: List[Dict[str, Any]] = []
    for entity in coordinates.get("entities") or []:
        if _is_ego(entity):
            continue
        loc = entity.get("location") or {}
        longitudinal, lateral = _decompose_offsets(loc, origin, forward_heading)

        heading = str(entity.get("heading_relation") or "unknown").lower()
        flow = str(entity.get("flow_compliance") or "legal").lower()
        is_oncoming = heading in {"opposite_direction", "opposite", "oncoming"}

        if flow == "wrong_way":
            # Keep the wrong-way actor in its own carriageway band rather than the
            # opposing-side offset the legacy path applied; heading flips below.
            try:
                lane_index = int(entity.get("lane_index_relation") or 0)
            except (TypeError, ValueError):
                lane_index = 0
            lateral = float(lane_index) * float(lane_width)
            side = "same"
        else:
            side = "opposing" if is_oncoming else "same"

        z = float(loc.get("z", 0.3) or 0.3)
        placement = frame.place(
            longitudinal_m=longitudinal,
            lateral_m=lateral,
            side=side,
            flow_compliance=flow,
            z=z,
        )
        entity["location"] = dict(placement.location)
        entity["rotation"] = {"pitch": 0.0, "yaw": placement.yaw, "roll": 0.0}
        entity["road_placement"] = "frame"
        entity["layout_version"] = entity.get("layout_version") or "v2"
        entity["layout_scene_kind"] = "open_road"
        entity["placement_reason"] = "open-road matched-structure placement"

        ok, reason = check_heading_consistency(
            yaw=placement.yaw,
            ego_heading=forward_heading,
            heading_relation=entity.get("heading_relation"),
            flow_compliance=entity.get("flow_compliance"),
        )
        entity["heading_consistency"] = reason
        assignments.append(
            {
                "id": entity.get("id"),
                "side": side,
                "longitudinal_m": round(longitudinal, 1),
                "lateral_m": round(lateral, 1),
                "yaw": round(placement.yaw, 1),
                "ok": ok,
            }
        )

    coordinates.setdefault("metadata", {})["road_reprojection"] = {
        "road_id": matched_structure.get("road_id"),
        "lane_id": matched_structure.get("lane_id"),
        "curved": len(matched_structure.get("curve_samples") or []) > 1,
        "assignments": assignments,
    }
    coordinates.setdefault("metadata", {})["layout_version"] = "v2"
    coordinates.setdefault("metadata", {})["layout_scene_kind"] = "open_road"
    return coordinates
