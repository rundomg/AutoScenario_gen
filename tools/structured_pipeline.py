import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tools.utils import write_to_file


SCENE_UNDERSTANDING_REQUIRED_KEYS = (
    "traffic_subjects",
    "road_network",
    "general_environment",
    "metadata",
)

PAIRWISE_GAP_THRESHOLDS = (
    ("overlap", 1.0),
    ("tight", 3.0),
    ("near", 8.0),
    ("mid", 18.0),
)

DISTANCE_BAND_METERS = {
    "immediate": 4.0,
    "near": 10.0,
    "mid": 22.0,
    "far": 38.0,
}

LANE_WIDTH_METERS = {
    "narrow": 3.2,
    "standard": 3.5,
    "wide": 3.8,
}

PARKING_LANE_WIDTH_METERS = 2.5
SIDEWALK_BUFFER_METERS = 1.5
# A local scene spans at most ~100m; reject candidate lanes farther than this
# from the anchor so a leaked fallback/origin lane can never capture an actor.
MAX_LOCAL_LANE_DISTANCE_M = 150.0
# Lateral width of a center median expressed in lane-width multiples. Used to
# push opposite-direction (oncoming) actors across the divider onto the opposing
# carriageway instead of leaving them in an adjacent same-direction lane.
MEDIAN_GAP_LANE_EQUIV = 1.0

# Yaw deflection (degrees) applied to a vehicle captured mid-turn at a junction.
# Static reconstruction has no continuous trajectory, so a turning actor is
# rendered as its straight-lane heading rotated by this fixed amount toward the
# branch it is entering. ~45 deg reads as "clearly mid-maneuver" without snapping
# the actor fully perpendicular (which would look like the turn already finished).
TURN_INTENT_YAW_DEG = 45.0

BACKGROUND_DENSITY_DEFAULTS = {
    "curbside_row": 5,
    "sidewalk_group": 2,
    "opposing_flow": 2,
    "sparse_filler": 1,
}

BACKGROUND_MAX_ACTORS = 8

CANONICAL_COLOR_MAP = {
    "white": "white",
    "off_white": "white",
    "ivory": "white",
    "cream": "white",
    "silver": "silver",
    "gray": "gray",
    "grey": "gray",
    "dark_gray": "gray",
    "light_gray": "gray",
    "black": "black",
    "blue": "blue",
    "dark_blue": "blue",
    "light_blue": "blue",
    "navy": "blue",
    "red": "red",
    "dark_red": "red",
    "maroon": "red",
    "green": "green",
    "dark_green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "tan": "brown",
    "beige": "brown",
}

COLOR_RGB_MAP = {
    "white": "255,255,255",
    "silver": "192,192,192",
    "gray": "128,128,128",
    "black": "20,20,20",
    "blue": "54,116,168",
    "red": "184,49,47",
    "green": "74,120,66",
    "yellow": "214,179,54",
    "orange": "214,120,36",
    "brown": "133,94,66",
}

VEHICLE_CATEGORIES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}

STATIC_CATEGORIES = {
    "cone_group",
    "barrier_group",
}

WALKER_CATEGORIES = {
    "pedestrian",
}

CANONICAL_CATEGORY_MAP = {
    "two_wheeler": "motorcycle",
    "scooter": "motorcycle",
    "motor_scooter": "motorcycle",
    "motor_scooter_with_rider_and_passenger": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "bicycle",
    "parked_vehicle": "car",
    "parked_motorcycle": "motorcycle",
    "parked_scooter": "motorcycle",
    "parked_motor_scooter": "motorcycle",
    "curbside_motorcycle": "motorcycle",
    "curbside_scooter": "motorcycle",
    "parked_two_wheeler": "motorcycle",
    "parked_two_wheeler_group": "motorcycle",
    "parked_vehicle_row": "car",
    "parked_car_row": "car",
    "car_row": "car",
    "curbside_car_row": "car",
    "curbside_parked_car_row": "car",
    "parked_cars": "car",
    "parked_vehicle_partial": "car",
    "partial_parked_vehicle": "car",
    "partially_visible_parked_vehicle": "car",
    "opposing_vehicle": "car",
    "vehicle": "car",
    "moving_vehicle": "car",
    "moving_car": "car",
    "sidewalk_pedestrian_group": "pedestrian",
    "pedestrian_group": "pedestrian",
    "traffic_cone": "cone_group",
    "traffic_cones": "cone_group",
    "cone": "cone_group",
    "cone_row": "cone_group",
    "construction_cone": "cone_group",
    "constructioncone": "cone_group",
    "cones": "cone_group",
    "barriers": "barrier_group",
}

CANONICAL_LANE_SIDE_MAP = {
    # left_lane variants
    "center_left": "left_lane",
    "left_of_center": "left_lane",
    "left_of_ego": "left_lane",
    "left_of_ego_center": "left_lane",
    "left_front": "left_lane",
    "left_rear": "left_lane",
    "oncoming_lane": "left_lane",
    "opposing_lane": "left_lane",
    "opposing_center_left": "left_lane",
    "opposing_left": "left_lane",
    "opposing_side": "left_lane",
    # right_lane variants
    "center_right": "right_lane",
    "right_of_center": "right_lane",
    "right_of_ego": "right_lane",
    "right_front": "right_lane",
    "right_rear": "right_lane",
    "opposing_center_right": "right_lane",
    "opposing_right": "right_lane",
    # Parking / curbside vehicle variants are normalized to adjacent lanes.
    "curbside_right": "right_lane",
    "right_curb": "right_lane",
    "right_curbside": "right_lane",
    "right_curbside_parking_lane": "right_lane",
    "right_curbside_row": "right_lane",
    "right_parking_lane": "right_lane",
    "right_parking": "right_lane",
    "parking_right": "right_lane",
    "curbside_parking_right": "right_lane",
    "near_right_edge": "right_lane",
    "far_right_edge": "right_lane",
    "right_edge": "right_lane",
    "right_shoulder": "right_lane",
    "curbside_left": "left_lane",
    "left_curb": "left_lane",
    "left_curbside": "left_lane",
    "left_curbside_parking_lane": "left_lane",
    "left_curbside_row": "left_lane",
    "left_parking_lane": "left_lane",
    "left_parking": "left_lane",
    "parking_left": "left_lane",
    "curbside_parking_left": "left_lane",
    "near_left_edge": "left_lane",
    "far_left_edge": "left_lane",
    "left_edge": "left_lane",
    "left_shoulder": "left_lane",
    # sidewalk variants
    "left_sidewalk": "sidewalk_left",
    "sidewalk_left_side": "sidewalk_left",
    "right_sidewalk": "sidewalk_right",
    "sidewalk_right_side": "sidewalk_right",
}


def _deep_copy(value: Any) -> Any:
    return deepcopy(value)


def _coerce_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _coerce_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _slugify(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")


def _normalize_entity_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def _canonical_category(value: Any, subtype: Any = None) -> str:
    normalized = _slugify(value)
    subtype_normalized = _slugify(subtype)
    if subtype_normalized in CANONICAL_CATEGORY_MAP:
        return CANONICAL_CATEGORY_MAP[subtype_normalized]
    return CANONICAL_CATEGORY_MAP.get(normalized, normalized or "car")


def _canonical_entity_category(entity: Dict[str, Any]) -> str:
    return _canonical_category(entity.get("category"), entity.get("subtype"))


def _canonical_subtype(entity: Dict[str, Any], category: str) -> str:
    raw_subtype = str(entity.get("subtype") or entity.get("category") or category or "unknown")
    normalized = _slugify(raw_subtype)
    if normalized in {
        "parked_vehicle",
        "parked_vehicle_row",
        "parked_car_row",
        "car_row",
        "curbside_car_row",
        "curbside_parked_car_row",
        "parked_cars",
        "parked_vehicle_partial",
        "partial_parked_vehicle",
        "partially_visible_parked_vehicle",
        "moving_vehicle",
        "opposing_vehicle",
    }:
        return category
    if normalized in {"parked_scooter", "parked_motor_scooter", "curbside_scooter"}:
        return "motor_scooter"
    if normalized in {"parked_motorcycle", "curbside_motorcycle", "parked_two_wheeler"}:
        return "motorcycle"
    return raw_subtype


def _canonical_lane_side(value: Any) -> str:
    normalized = _slugify(value)
    return CANONICAL_LANE_SIDE_MAP.get(normalized, normalized or "same_lane")


def _lane_index_from_relation(value: Any) -> Optional[int]:
    normalized = _slugify(value)
    if not normalized:
        return 0
    if normalized in {"same_lane", "ego_lane", "center_lane", "crosswalk"}:
        return 0
    if "right" in normalized:
        return 1
    if (
        "left" in normalized
        or "opposing" in normalized
        or "oncoming" in normalized
    ):
        return -1
    return None


def _canonical_lane_index(value: Any, lane_side_relation: Any = None) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        inferred = _lane_index_from_relation(lane_side_relation)
        return 0 if inferred is None else inferred


def _lane_side_from_index(lane_index: int, fallback: Any = None) -> str:
    fallback_relation = _canonical_lane_side(fallback)
    if fallback_relation in {"sidewalk_left", "sidewalk_right", "crosswalk"}:
        return fallback_relation
    if lane_index > 0:
        return "right_lane"
    if lane_index < 0:
        return "left_lane"
    return "same_lane"


def _correct_lane_index_for_heading(
    lane_index: int,
    heading: str,
    opposing_lane_count: int,
) -> int:
    """Enforce the constraint that opposite_direction actors occupy opposing lanes.

    In right-hand traffic the opposing lane is always to the LEFT of ego
    (negative lane_index).  When the VLM assigns a positive lane_index to an
    opposite_direction actor — a common perspective error on curved roads where
    the oncoming car appears to the right in the image — clamp it into the
    valid opposing range [-opposing_lane_count, -1].
    """
    if heading == "opposite_direction" and lane_index > 0:
        return -min(lane_index, max(1, opposing_lane_count))
    return lane_index


def _normalize_lane_fields(entity: Dict[str, Any], category: str) -> Tuple[str, int]:
    lane_side = _canonical_lane_side(entity.get("lane_side_relation"))
    lane_index = _canonical_lane_index(entity.get("lane_index_relation"), lane_side)
    if category in VEHICLE_CATEGORIES:
        lane_side = _lane_side_from_index(lane_index, lane_side)
    return lane_side, lane_index


def _canonical_distance_band(value: Any) -> str:
    normalized = _slugify(value)
    if normalized in DISTANCE_BAND_METERS:
        return normalized
    if "immediate" in normalized:
        return "immediate"
    if "near" in normalized and "mid" not in normalized:
        return "near"
    if "mid" in normalized:
        return "mid"
    if "far" in normalized:
        return "far"
    if normalized == "ahead":
        return "near"
    return "near"


def _canonical_motion_state(value: Any) -> str:
    normalized = _slugify(value)
    if normalized in {"moving", "stopped", "parked", "unknown"}:
        return normalized
    if normalized in {"stationary", "static"}:
        return "parked"
    if normalized in {"mixed"}:
        return "unknown"
    return "unknown"


def _canonical_heading(value: Any) -> str:
    normalized = _slugify(value)
    if normalized in {"same_direction", "opposite_direction", "crossing", "unknown"}:
        return normalized
    if normalized in {"uncertain"}:
        return "unknown"
    return "unknown"


def _canonical_turn_intent(value: Any) -> str:
    """Normalize a vehicle's junction turn maneuver to one of the canonical values.

    Returns one of ``left``, ``right``, ``through``, ``none``. ``none`` means no
    turning maneuver is asserted (steady lane following or simply unknown), so the
    actor keeps its straight-lane heading.
    """
    normalized = _slugify(value)
    if normalized in {"left", "right", "through", "none"}:
        return normalized
    if normalized in {"turn_left", "left_turn", "turning_left", "left_turning"}:
        return "left"
    if normalized in {"turn_right", "right_turn", "turning_right", "right_turning"}:
        return "right"
    if normalized in {"straight", "ahead", "go_straight", "through_movement"}:
        return "through"
    return "none"


def _canonical_flow_compliance(value: Any) -> str:
    """Normalize an actor's against-vs-with lane-flow label.

    Returns one of ``legal``, ``wrong_way``, ``unknown``. ``wrong_way`` means the
    actor travels against the legal flow of the lane band it physically occupies
    (a true wrong-way actor, common for motorcycles) -- distinct from a normal
    oncoming vehicle in the opposing carriageway, which is ``legal``. This label
    only controls heading (a 180 flip downstream); it never moves the actor to a
    different lane band.
    """
    normalized = _slugify(value)
    if normalized in {"wrong_way", "wrongway", "against_flow", "against_traffic",
                      "counterflow", "counter_flow", "ghost", "salmoning", "reverse_flow"}:
        return "wrong_way"
    if normalized in {"legal", "with_flow", "compliant", "normal", "lawful"}:
        return "legal"
    if normalized in {"unknown", "", "none"}:
        return "unknown"
    return "legal"


def _apply_turn_intent(yaw: float, turn_intent: Any) -> float:
    """Deflect a base lane yaw toward the actor's junction turn maneuver.

    Direction is the actor's own driving perspective and applies after any
    opposite-direction flip, so it is uniform regardless of absolute heading. In
    CARLA's coordinate frame (x=east, y=south, yaw clockwise) the driver's right
    is yaw+90, so a left turn decreases yaw and a right turn increases it.
    """
    intent = _canonical_turn_intent(turn_intent)
    if intent == "left":
        return float(yaw) - TURN_INTENT_YAW_DEG
    if intent == "right":
        return float(yaw) + TURN_INTENT_YAW_DEG
    return float(yaw)


def _canonical_road_type(value: Any) -> str:
    normalized = _slugify(value)
    mapping = {
        "urban_local_street": "urban_straight",
        "urban_straight_street": "urban_straight",
        "straight_urban_road": "urban_straight",
    }
    return mapping.get(normalized, normalized or "urban_straight")


def _canonical_density_role(value: Any) -> str:
    normalized = _slugify(value)
    mapping = {
        "curbside_parking_row": "curbside_row",
        "roadside_activity": "sidewalk_group",
        "sparse_opposing_flow": "opposing_flow",
    }
    return mapping.get(normalized, normalized or "sparse_filler")


def _canonical_color_name(value: Any) -> Optional[str]:
    normalized = _slugify(value)
    if not normalized:
        return None
    return CANONICAL_COLOR_MAP.get(normalized, normalized if normalized in COLOR_RGB_MAP else None)


def _normalize_appearance(entity: Dict[str, Any], category: str) -> Dict[str, Any]:
    if category not in VEHICLE_CATEGORIES:
        return {}
    appearance = _coerce_dict(entity.get("appearance"))
    color_name = (
        _canonical_color_name(appearance.get("color"))
        or _canonical_color_name(entity.get("color"))
        or _canonical_color_name(entity.get("vehicle_color"))
    )
    color_confidence = (
        str(appearance.get("color_confidence") or entity.get("color_confidence") or "medium")
        if color_name
        else None
    )
    body_style = str(appearance.get("body_style") or entity.get("body_style") or "").strip()
    normalized = {}
    if color_name:
        normalized["color"] = color_name
        normalized["color_confidence"] = color_confidence or "medium"
        normalized["color_rgb"] = COLOR_RGB_MAP.get(color_name)
    if body_style:
        normalized["body_style"] = body_style
    return normalized


def _canonical_constraint_strength(value: Any) -> str:
    normalized = _slugify(value)
    if normalized in {"hard", "soft", "validation_only"}:
        return normalized
    if normalized in {"weak"}:
        return "soft"
    return "soft"


def _canonical_pairwise_longitudinal(value: Any) -> Optional[str]:
    normalized = _slugify(value)
    mapping = {
        "ahead_of_other": "ahead_of_other",
        "ahead": "ahead_of_other",
        "in_front_of": "ahead_of_other",
        "behind_other": "behind_other",
        "behind_of_other": "behind_other",
        "behind": "behind_other",
        "aligned_with_other": "aligned_with_other",
        "alongside": "aligned_with_other",
        "same_depth": "aligned_with_other",
    }
    return mapping.get(normalized)


def _canonical_pairwise_lateral(value: Any) -> Optional[str]:
    normalized = _slugify(value)
    mapping = {
        "left_of_other": "left_of_other",
        "left_of": "left_of_other",
        "right_of_other": "right_of_other",
        "right_of": "right_of_other",
        "same_lateral_band": "same_lateral_band",
        "same_lane_center": "same_lateral_band",
        "aligned": "same_lateral_band",
    }
    return mapping.get(normalized)


def _canonical_pairwise_lane_relation(value: Any) -> Optional[str]:
    normalized = _slugify(value)
    mapping = {
        "same_lane": "same_lane",
        "same_parking_lane": "same_parking_lane",
        "same_sidewalk_zone": "same_sidewalk_zone",
        "adjacent_left_lane": "adjacent_left_lane",
        "adjacent_right_lane": "adjacent_right_lane",
        "cross_lane": "cross_lane",
    }
    return mapping.get(normalized)


def _canonical_gap_band(value: Any) -> Optional[str]:
    normalized = _slugify(value)
    if normalized in {"overlap", "tight", "near", "mid", "far"}:
        return normalized
    return None


def _normalize_key_pairwise_relations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized_relations = []
    for index, item in enumerate(_coerce_list(payload.get("key_pairwise_relations"))):
        pair = _coerce_dict(item)
        entity_id = str(pair.get("entity_id") or pair.get("group_id") or "").strip()
        other_entity_id = str(
            pair.get("other_entity_id") or pair.get("other_group_id") or ""
        ).strip()
        longitudinal_relation = _canonical_pairwise_longitudinal(
            pair.get("longitudinal_relation")
        )
        lateral_relation = _canonical_pairwise_lateral(pair.get("lateral_relation"))
        lane_relation = _canonical_pairwise_lane_relation(pair.get("lane_relation"))
        if not entity_id or not other_entity_id:
            continue
        if not longitudinal_relation and not lateral_relation and not lane_relation:
            continue
        normalized_relations.append(
            {
                "id": str(pair.get("id") or f"key_pairwise_{index}"),
                "entity_id": entity_id,
                "other_entity_id": other_entity_id,
                "longitudinal_relation": longitudinal_relation,
                "longitudinal_gap_band": _canonical_gap_band(pair.get("longitudinal_gap_band")),
                "lateral_relation": lateral_relation,
                "lane_relation": lane_relation,
                "constraint_strength": _canonical_constraint_strength(
                    pair.get("constraint_strength") or "hard"
                ),
                "confidence": str(pair.get("confidence") or "high"),
                "evidence": str(pair.get("evidence") or ""),
                "relation_source": "vlm",
            }
        )
    return normalized_relations


def normalize_scene_understanding(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(payload, dict):
        return None, "Scene understanding root must be a JSON object."

    missing = [key for key in SCENE_UNDERSTANDING_REQUIRED_KEYS if key not in payload]
    if missing:
        return None, f"Scene understanding is missing required keys: {', '.join(missing)}"

    normalized = {
        "traffic_subjects": [],
        "background_traffic": [],
        "road_network": {},
        "actor_layout": _coerce_dict(payload.get("actor_layout")),
        "general_environment": {},
        "metadata": _coerce_dict(payload.get("metadata")),
        "key_pairwise_relations": _normalize_key_pairwise_relations(payload),
    }

    # Read opposing lane count from the raw payload so we can enforce the
    # heading ↔ lane_index semantic constraint during entity normalization,
    # before road_network normalization happens below.
    _raw_lane_groups = _coerce_list(_coerce_dict(payload.get("road_network")).get("lane_groups"))
    _raw_primary_group = _raw_lane_groups[0] if _raw_lane_groups else {}
    _opposing_lane_count = max(1, int(_raw_primary_group.get("opposing_lane_count", 1)))

    for index, entity in enumerate(_coerce_list(payload.get("traffic_subjects"))):
        if not isinstance(entity, dict):
            return None, f"traffic_subjects[{index}] must be an object."
        category = _canonical_entity_category(entity)
        lane_side_relation, lane_index_relation = _normalize_lane_fields(entity, category)
        heading = _canonical_heading(entity.get("heading_relation_to_ego"))
        if category in VEHICLE_CATEGORIES:
            lane_index_relation = _correct_lane_index_for_heading(
                lane_index_relation, heading, _opposing_lane_count
            )
            lane_side_relation = _lane_side_from_index(lane_index_relation, lane_side_relation)
        normalized["traffic_subjects"].append(
            {
                "id": str(entity.get("id") or _normalize_entity_id("subject", index)),
                "category": category,
                "subtype": _canonical_subtype(entity, category),
                "visual_confidence": str(entity.get("visual_confidence") or "medium"),
                "motion_state": _canonical_motion_state(entity.get("motion_state")),
                "turn_intent": _canonical_turn_intent(entity.get("turn_intent")),
                "heading_relation_to_ego": heading,
                "flow_compliance": _canonical_flow_compliance(entity.get("flow_compliance")),
                "lane_side_relation": lane_side_relation,
                "lane_index_relation": lane_index_relation,
                "longitudinal_relation": str(entity.get("longitudinal_relation") or "ahead"),
                "longitudinal_proximity": _canonical_distance_band(entity.get("longitudinal_proximity")),
                "count": max(1, int(entity.get("count", 1))),
                "evidence": str(entity.get("evidence") or ""),
                "must_reconstruct": True,
                "layout_anchor_id": str(entity.get("layout_anchor_id") or "").strip(),
                "anchor_relation": _coerce_dict(entity.get("anchor_relation")),
                "appearance": _normalize_appearance(entity, category),
            }
        )

    for index, entity in enumerate(_coerce_list(payload.get("background_traffic"))):
        if not isinstance(entity, dict):
            return None, f"background_traffic[{index}] must be an object."
        category = _canonical_entity_category(entity)
        density_role = _canonical_density_role(entity.get("density_role"))
        default_count = BACKGROUND_DENSITY_DEFAULTS.get(density_role, 1)
        lane_side_relation, lane_index_relation = _normalize_lane_fields(entity, category)
        bg_heading = _canonical_heading(entity.get("heading_relation_to_ego"))
        if category in VEHICLE_CATEGORIES:
            lane_index_relation = _correct_lane_index_for_heading(
                lane_index_relation, bg_heading, _opposing_lane_count
            )
            lane_side_relation = _lane_side_from_index(lane_index_relation, lane_side_relation)
        representative_count = _clamp_int(
            entity.get("representative_count", default_count),
            1,
            BACKGROUND_MAX_ACTORS,
            default_count,
        )
        normalized["traffic_subjects"].append(
            {
                "id": str(entity.get("id") or _normalize_entity_id("background", index)),
                "category": category,
                "subtype": _canonical_subtype(entity, category),
                "visual_confidence": str(entity.get("confidence") or "medium"),
                "motion_state": "unknown",
                "turn_intent": _canonical_turn_intent(entity.get("turn_intent")),
                "heading_relation_to_ego": bg_heading,
                "flow_compliance": _canonical_flow_compliance(entity.get("flow_compliance")),
                "lane_side_relation": lane_side_relation,
                "lane_index_relation": lane_index_relation,
                "longitudinal_relation": str(entity.get("longitudinal_relation") or "ahead"),
                "longitudinal_proximity": _canonical_distance_band(entity.get("longitudinal_band")),
                "count": representative_count,
                "evidence": str(entity.get("source") or entity.get("evidence") or ""),
                "must_reconstruct": True,
                "layout_anchor_id": str(entity.get("layout_anchor_id") or "").strip(),
                "anchor_relation": _coerce_dict(entity.get("anchor_relation")),
                "appearance": _normalize_appearance(entity, category),
                "density_role": density_role,
            }
        )

    road_network = _coerce_dict(payload.get("road_network"))

    normalized_road_segments = []
    for road_index, segment in enumerate(_coerce_list(road_network.get("road_segments"))):
        if not isinstance(segment, dict):
            continue
        normalized_road_segments.append(
            {
                "id": str(segment.get("id") or f"road_segment_{road_index}"),
                "geometry_type": str(
                    segment.get("geometry_type")
                    or segment.get("shape")
                    or "line"
                ),
                "curvature_hint": str(
                    segment.get("curvature_hint")
                    or segment.get("curvature")
                    or "straight"
                ),
                "relative_length": str(
                    segment.get("relative_length")
                    or segment.get("relative_position")
                    or "medium"
                ),
            }
        )

    normalized_lane_groups = []
    for lane_index, lane_group in enumerate(_coerce_list(road_network.get("lane_groups"))):
        if not isinstance(lane_group, dict):
            continue
        side = _slugify(lane_group.get("side"))
        lanes = max(1, int(lane_group.get("lanes", 1)))
        normalized_lane_groups.append(
            {
                "forward_lane_count": int(
                    lane_group.get(
                        "forward_lane_count",
                        lanes if side in {"ego_direction", "forward"} else 1,
                    )
                ),
                "opposing_lane_count": int(
                    lane_group.get(
                        "opposing_lane_count",
                        lanes if side in {"opposite_direction", "opposing"} else 1,
                    )
                ),
                "left_parking_lane_count": max(
                    0, int(lane_group.get("left_parking_lane_count", 0))
                ),
                "right_parking_lane_count": max(
                    0, int(lane_group.get("right_parking_lane_count", 0))
                ),
                "lane_width_class": str(lane_group.get("lane_width_class") or "standard"),
                "lane_count_evidence": str(lane_group.get("lane_count_evidence") or ""),
                "lane_count_confidence": str(
                    lane_group.get("lane_count_confidence") or "unknown"
                ),
                "evidence": str(lane_group.get("evidence") or ""),
                "id": str(lane_group.get("id") or f"lane_group_{lane_index}"),
            }
        )
    normalized_road_network = {}
    map_matching = _coerce_dict(road_network.get("map_matching"))
    if map_matching:
        normalized_road_network["map_matching"] = map_matching
    if "road_type" in road_network:
        normalized_road_network["road_type"] = _canonical_road_type(
            road_network.get("road_type")
        )
    if "directionality" in road_network:
        normalized_road_network["directionality"] = str(
            road_network.get("directionality") or "two_way"
        )
    if normalized_road_segments:
        normalized_road_network["road_segments"] = normalized_road_segments
    elif not map_matching:
        normalized_road_network["road_segments"] = [
            {
                "id": "road_segment_0",
                "geometry_type": "straight",
                "curvature_hint": "straight",
                "relative_length": "medium",
            }
        ]
    if normalized_lane_groups:
        normalized_road_network["lane_groups"] = normalized_lane_groups
    elif not map_matching:
        normalized_road_network["lane_groups"] = [
            {
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
                "left_parking_lane_count": 0,
                "right_parking_lane_count": 0,
                "lane_width_class": "standard",
                "lane_count_evidence": "",
                "lane_count_confidence": "unknown",
                "evidence": "",
                "id": "lane_group_0",
            }
        ]
    lane_markings = _coerce_dict(road_network.get("lane_markings"))
    if lane_markings:
        normalized_road_network["lane_markings"] = lane_markings
    special_road_areas = _coerce_list(road_network.get("special_road_areas"))
    if special_road_areas:
        normalized_road_network["special_road_areas"] = special_road_areas
    junctions = _coerce_list(road_network.get("junctions"))
    if junctions:
        normalized_road_network["junctions"] = junctions
    roadside_boundaries = _coerce_dict(road_network.get("roadside_boundaries"))
    if roadside_boundaries:
        normalized_road_network["roadside_boundaries"] = roadside_boundaries
    control_elements = _coerce_list(road_network.get("control_elements"))
    if control_elements:
        normalized_road_network["control_elements"] = control_elements
    normalized["road_network"] = normalized_road_network

    general_environment = _coerce_dict(payload.get("general_environment"))
    normalized["general_environment"] = {
        "weather_hint": str(general_environment.get("weather_hint") or "unknown"),
        "lighting_hint": str(general_environment.get("lighting_hint") or "daylight"),
        "time_of_day_hint": str(general_environment.get("time_of_day_hint") or "day"),
        "urban_density": str(general_environment.get("urban_density") or "urban"),
        "roadside_context_left": _coerce_list(general_environment.get("roadside_context_left")),
        "roadside_context_right": _coerce_list(general_environment.get("roadside_context_right")),
        "occlusion_notes": _coerce_list(general_environment.get("occlusion_notes")),
        "non_spawnable_landmarks": _coerce_list(general_environment.get("non_spawnable_landmarks")),
    }

    _inject_inferred_obstacle_subjects(normalized)
    normalized["metadata"].setdefault("schema_version", "scene-understanding-v1")
    normalized["metadata"].setdefault("input_type", "image")
    return normalized, None


def _inject_inferred_obstacle_subjects(scene_understanding: Dict[str, Any]) -> None:
    traffic_subjects = scene_understanding.get("traffic_subjects", [])

    for control in _coerce_list(scene_understanding.get("road_network", {}).get("control_elements")):
        control_type = _slugify(_coerce_dict(control).get("type"))
        if control_type not in {"traffic_cones", "cones"}:
            continue
        position_hint = str(
            control.get("position")
            or control.get("location")
            or control.get("position_relation")
            or control.get("location_relation")
            or ""
        )
        inferred_entity = {
            "id": "inferred_cone_group",
            "category": "cone_group",
            "subtype": "constructioncone",
            "visual_confidence": str(control.get("confidence") or "medium"),
            "motion_state": "stopped",
            "heading_relation_to_ego": "same_direction",
            "lane_side_relation": "left_edge" if "left" in position_hint else "right_edge",
            "longitudinal_relation": "ahead",
            "longitudinal_proximity": "near",
            "count": 3,
            "evidence": "Inferred from visible traffic cones recorded in road_network.control_elements.",
            "must_reconstruct": True,
        }
        inferred_side = inferred_entity["lane_side_relation"]

        updated_existing = False
        for entity in traffic_subjects:
            if str(entity.get("category") or "") != "cone_group":
                continue
            evidence = str(entity.get("evidence") or "")
            entity_id = str(entity.get("id") or "")
            if "Inferred from visible traffic cones" in evidence or entity_id.startswith("inferred_"):
                entity["lane_side_relation"] = inferred_side
                entity["heading_relation_to_ego"] = inferred_entity["heading_relation_to_ego"]
                entity["longitudinal_relation"] = inferred_entity["longitudinal_relation"]
                entity["longitudinal_proximity"] = inferred_entity["longitudinal_proximity"]
                updated_existing = True

        existing_categories = {entity.get("category") for entity in traffic_subjects}
        if not updated_existing and "cone_group" not in existing_categories and "barrier_group" not in existing_categories:
            traffic_subjects.append(inferred_entity)
        break


def _fallback_anchor_lane() -> Dict[str, Any]:
    return {
        "road_id": 1,
        "lane_id": -1,
        "start": {"x": 0.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
        "end": {"x": 80.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
    }


def select_anchor_lane(spawn_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    topology_sample = _coerce_list((spawn_context or {}).get("topology_sample"))
    for item in topology_sample:
        if isinstance(item, dict) and item.get("lane_id") not in (0, None):
            return _deep_copy(item)
    if topology_sample:
        return _deep_copy(topology_sample[0])
    return _fallback_anchor_lane()


def _lane_anchor_for_relation(
    lane_side_relation: str,
    category: str,
    lane_index_relation: Any = None,
) -> Tuple[str, int, str]:
    relation = str(lane_side_relation or "same_lane")
    lane_index = _canonical_lane_index(lane_index_relation, relation)
    if category in VEHICLE_CATEGORIES:
        if lane_index == 0:
            return "center_of_lane", 0, "same_lane"
        return "lane_center", lane_index, "relative_lane"
    mapping = {
        "same_lane": ("center_of_lane", 0, "same_lane"),
        "left_lane": ("left_lane_center", -1, "adjacent_left"),
        "left_of_center": ("left_lane_center", -1, "adjacent_left"),
        "opposing_center_left": ("left_lane_center", -1, "adjacent_left"),
        "opposing_left": ("left_lane_center", -1, "adjacent_left"),
        "opposing_side": ("left_lane_center", -1, "adjacent_left"),
        "right_lane": ("right_lane_center", 1, "adjacent_right"),
        "right_of_center": ("right_lane_center", 1, "adjacent_right"),
        "opposing_center_right": ("right_lane_center", 1, "adjacent_right"),
        "opposing_right": ("right_lane_center", 1, "adjacent_right"),
        "left_edge": ("left_curbside", -1, "road_edge_left"),
        "right_edge": ("right_curbside", 1, "road_edge_right"),
        "sidewalk_left": ("sidewalk_band", -1, "offroad_sidewalk"),
        "sidewalk_right": ("sidewalk_band", 1, "offroad_sidewalk"),
        "crosswalk": ("crosswalk_band", 0, "same_lane"),
        "shoulder": ("right_curbside", 1, "road_edge_right"),
    }
    anchor, lane_offset, lateral_mode = mapping.get(
        relation, ("center_of_lane", 0, "same_lane")
    )
    return anchor, lane_offset, lateral_mode


def _distance_band_for_entity(entity: Dict[str, Any], source_key: str) -> str:
    if source_key == "traffic_subjects":
        return str(entity.get("longitudinal_proximity") or "near")
    return str(entity.get("longitudinal_band") or "mid")


def _order_relation_for_entity(entity: Dict[str, Any], source_key: str) -> str:
    if source_key == "traffic_subjects":
        return str(entity.get("longitudinal_relation") or "ahead")
    band = str(entity.get("longitudinal_band") or "mid")
    if band == "far":
        return "ahead"
    return "ahead"


def _expected_longitudinal_scalar(entity: Dict[str, Any]) -> float:
    distance_band = str(entity.get("distance_band") or "near")
    base = float(DISTANCE_BAND_METERS.get(distance_band, 10.0))
    base += int(entity.get("distance_order_rank", 0)) * 4.0
    base += int(entity.get("group_instance_index", 0)) * float(entity.get("group_spacing_m", 0.0))
    order_relation = str(entity.get("order_relation") or "ahead")
    if order_relation == "behind":
        return -base
    if order_relation == "alongside":
        return 0.0
    return base


def _expected_lateral_band(entity: Dict[str, Any]) -> float:
    lane_anchor = str(entity.get("lane_anchor") or "")
    lane_index = float(entity.get("lane_index_relation") or 0.0)
    if lane_anchor == "lane_center":
        return lane_index
    mapping = {
        "sidewalk_band": -3.0 if "left" in str(entity.get("lane_side_relation") or "") else 3.0,
        "left_curbside": -2.0,
        "right_curbside": 2.0,
        "left_lane_center": -1.0,
        "right_lane_center": 1.0,
        "center_of_lane": 0.0,
        "crosswalk_band": 0.0,
    }
    if lane_anchor in mapping:
        return float(mapping[lane_anchor])
    return lane_index


def _gap_band_for_distance(distance_m: float) -> str:
    for band, threshold in PAIRWISE_GAP_THRESHOLDS:
        if distance_m <= threshold:
            return band
    return "far"


def _distance_for_gap_band(gap_band: str) -> float:
    return {
        "overlap": 0.5,
        "tight": 2.0,
        "near": 5.0,
        "mid": 12.0,
        "far": 22.0,
    }.get(str(gap_band or "near"), 5.0)


def _is_vehicle_like(category: str) -> bool:
    return category in VEHICLE_CATEGORIES


def _lane_relation_between(entity: Dict[str, Any], other: Dict[str, Any]) -> str:
    entity_relation = str(entity.get("lane_side_relation") or "")
    other_relation = str(other.get("lane_side_relation") or "")
    entity_category = str(entity.get("category") or "")
    other_category = str(other.get("category") or "")
    lateral_delta = _expected_lateral_band(entity) - _expected_lateral_band(other)

    if entity_relation == other_relation:
        if entity_relation in {"left_edge", "right_edge"}:
            return "same_parking_lane" if _is_vehicle_like(entity_category) and _is_vehicle_like(other_category) else "cross_lane"
        if entity_relation in {"sidewalk_left", "sidewalk_right"}:
            return "same_sidewalk_zone"
        return "same_lane"

    if abs(lateral_delta) <= 0.5:
        return "same_lane"
    if lateral_delta < 0:
        return "adjacent_left_lane"
    return "adjacent_right_lane"


def _constraint_strength_for_pair(entity: Dict[str, Any], other: Dict[str, Any], gap_band: str) -> str:
    category = str(entity.get("category") or "")
    other_category = str(other.get("category") or "")
    priority_set = {str(entity.get("priority") or ""), str(other.get("priority") or "")}
    if gap_band == "far" and priority_set == {"background"}:
        return "validation_only"
    if _is_vehicle_like(category) and _is_vehicle_like(other_category):
        return "hard"
    if (
        (category in STATIC_CATEGORIES and _is_vehicle_like(other_category))
        or (other_category in STATIC_CATEGORIES and _is_vehicle_like(category))
    ):
        return "hard" if gap_band in {"overlap", "tight", "near"} or "subject" in priority_set else "soft"
    if (
        (category in WALKER_CATEGORIES and _is_vehicle_like(other_category))
        or (other_category in WALKER_CATEGORIES and _is_vehicle_like(category))
    ):
        crossing = "cross" in str(entity.get("heading_relation") or "") or "cross" in str(other.get("heading_relation") or "")
        if crossing and gap_band in {"overlap", "tight", "near"}:
            return "hard"
        return "soft"
    if category in WALKER_CATEGORIES or other_category in WALKER_CATEGORIES:
        return "soft"
    if category in STATIC_CATEGORIES or other_category in STATIC_CATEGORIES:
        return "soft"
    return "validation_only"


def _derived_pairwise_relation(entity: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    longitudinal_a = _expected_longitudinal_scalar(entity)
    longitudinal_b = _expected_longitudinal_scalar(other)
    lateral_a = _expected_lateral_band(entity)
    lateral_b = _expected_lateral_band(other)
    longitudinal_delta = longitudinal_a - longitudinal_b
    lateral_delta = lateral_a - lateral_b
    gap_band = _gap_band_for_distance(abs(longitudinal_delta))
    if abs(longitudinal_delta) <= 2.0:
        longitudinal_relation = "aligned_with_other"
    elif longitudinal_delta > 0.0:
        longitudinal_relation = "ahead_of_other"
    else:
        longitudinal_relation = "behind_other"

    if abs(lateral_delta) <= 0.5:
        lateral_relation = "same_lateral_band"
    elif lateral_delta < 0.0:
        lateral_relation = "left_of_other"
    else:
        lateral_relation = "right_of_other"

    return {
        "entity_id": entity["entity_id"],
        "other_entity_id": other["entity_id"],
        "relation_source": "derived",
        "constraint_strength": _constraint_strength_for_pair(entity, other, gap_band),
        "longitudinal_relation": longitudinal_relation,
        "longitudinal_gap_band": gap_band,
        "lateral_relation": lateral_relation,
        "lane_relation": _lane_relation_between(entity, other),
        "confidence": "derived",
        "evidence": "Derived from ego-centric ordering and lane-aware bands.",
    }


def _pairwise_map_key(entity_id: str, other_entity_id: str) -> Tuple[str, str]:
    return (str(entity_id), str(other_entity_id))


def _expand_key_pairwise_relations(
    key_pairwise_relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    by_entity_id = {str(entity["entity_id"]): entity for entity in entities}
    for entity in entities:
        by_group.setdefault(str(entity.get("group_id") or entity["entity_id"]), []).append(entity)

    expanded_relations = []
    for relation in key_pairwise_relations:
        left_candidates = by_group.get(str(relation.get("entity_id")), [])
        right_candidates = by_group.get(str(relation.get("other_entity_id")), [])
        if str(relation.get("entity_id")) in by_entity_id:
            left_candidates = [by_entity_id[str(relation.get("entity_id"))]]
        if str(relation.get("other_entity_id")) in by_entity_id:
            right_candidates = [by_entity_id[str(relation.get("other_entity_id"))]]
        for entity in left_candidates:
            for other in right_candidates:
                if entity["entity_id"] == other["entity_id"]:
                    continue
                expanded_relations.append(
                    {
                        "entity_id": entity["entity_id"],
                        "other_entity_id": other["entity_id"],
                        "relation_source": "vlm",
                        "constraint_strength": _canonical_constraint_strength(
                            relation.get("constraint_strength") or "hard"
                        ),
                        "longitudinal_relation": relation.get("longitudinal_relation"),
                        "longitudinal_gap_band": relation.get("longitudinal_gap_band") or "near",
                        "lateral_relation": relation.get("lateral_relation") or "same_lateral_band",
                        "lane_relation": relation.get("lane_relation")
                        or _lane_relation_between(entity, other),
                        "confidence": str(relation.get("confidence") or "high"),
                        "evidence": str(relation.get("evidence") or ""),
                    }
                )
    return expanded_relations


def build_pairwise_relations(
    ego_relations: List[Dict[str, Any]],
    key_pairwise_relations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    pairwise_relations = []
    by_pair = {}
    for relation in _expand_key_pairwise_relations(
        _coerce_list(key_pairwise_relations),
        ego_relations,
    ):
        key = _pairwise_map_key(relation["entity_id"], relation["other_entity_id"])
        by_pair[key] = relation

    for index, entity in enumerate(ego_relations):
        for other in ego_relations[index + 1 :]:
            derived = _derived_pairwise_relation(entity, other)
            key = _pairwise_map_key(derived["entity_id"], derived["other_entity_id"])
            if key not in by_pair:
                by_pair[key] = derived

    pairwise_relations.extend(by_pair.values())
    return pairwise_relations


def _scene_has_center_median(scene_understanding: Dict[str, Any]) -> bool:
    """Detect a physical center divider from the scene understanding.

    A median pushes the opposing carriageway laterally away from ego; opposite
    -direction actors must clear it (see _opposing_carriageway_offset).
    """
    road_network = _coerce_dict(scene_understanding.get("road_network"))
    directionality = str(road_network.get("directionality") or "").lower()
    if any(
        token in directionality
        for token in (
            "not_median",
            "not median",
            "no_median",
            "no median",
            "without_median",
            "without median",
            "lane_markings_not_median",
            "markings_not_median",
            "painted_only",
        )
    ):
        return False
    if "divided" in directionality:
        return True
    for area in _coerce_list(road_network.get("special_road_areas")):
        area_dict = _coerce_dict(area)
        if area_dict.get("presence") is False or area_dict.get("visible") is False:
            continue
        if "median" in str(area_dict.get("type") or "").lower():
            return True
    return False


def _lane_context(scene_understanding: Dict[str, Any], anchor_lane: Dict[str, Any]) -> Dict[str, Any]:
    lane_groups = _coerce_list(scene_understanding.get("road_network", {}).get("lane_groups"))
    lane_group = lane_groups[0] if lane_groups else {}
    forward_lane_count = int(lane_group.get("forward_lane_count", 1))
    opposing_lane_count = int(lane_group.get("opposing_lane_count", 1))
    left_parking_lane_count = int(lane_group.get("left_parking_lane_count", 0) or 0)
    right_parking_lane_count = int(lane_group.get("right_parking_lane_count", 0) or 0)
    return {
        "ego_lane_id": int(anchor_lane.get("lane_id", -1)),
        "lane_catalog": {
            "same_lane": {"role": "driving", "relative_index": 0},
            "left_lane": {"role": "driving", "relative_index": -1},
            "right_lane": {"role": "driving", "relative_index": 1},
            "sidewalk_left": {"role": "sidewalk", "relative_index": -3},
            "sidewalk_right": {"role": "sidewalk", "relative_index": 3},
            "crosswalk": {"role": "crosswalk", "relative_index": 0},
        },
        "lane_roles": {
            "ego": "driving",
            "forward_lane_count": forward_lane_count,
            "opposing_lane_count": opposing_lane_count,
            "left_parking_lane_count": left_parking_lane_count,
            "right_parking_lane_count": right_parking_lane_count,
            "has_center_median": _scene_has_center_median(scene_understanding),
        },
        "lane_adjacency": {
            "same_lane": ["left_lane", "right_lane"],
            "left_lane": ["same_lane", "sidewalk_left"],
            "right_lane": ["same_lane", "sidewalk_right"],
        },
        "allowed_actor_types": {
            "driving": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "parking": ["car", "truck", "bus", "motorcycle", "bicycle", "cone_group", "barrier_group"],
            "parking_or_edge": ["car", "truck", "bus", "motorcycle", "bicycle", "cone_group", "barrier_group"],
            "sidewalk": ["pedestrian"],
            "crosswalk": ["pedestrian"],
        },
    }


def _iter_expanded_entities(scene_understanding: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    total_background = 0
    for entity in _coerce_list(scene_understanding.get("traffic_subjects")):
        count = max(1, int(entity.get("count", 1)))
        for instance_index in range(count):
            expanded = _deep_copy(entity)
            expanded["_source_key"] = "traffic_subjects"
            expanded["_instance_index"] = instance_index
            expanded["_instance_total"] = count
            expanded["_priority"] = "subject"
            expanded["_expanded_id"] = (
                expanded["id"]
                if count == 1
                else f"{expanded['id']}_{instance_index}"
            )
            yield expanded

    for entity in _coerce_list(scene_understanding.get("background_traffic")):
        remaining = BACKGROUND_MAX_ACTORS - total_background
        if remaining <= 0:
            break
        count = min(max(1, int(entity.get("representative_count", 1))), remaining)
        total_background += count
        for instance_index in range(count):
            expanded = _deep_copy(entity)
            expanded["_source_key"] = "traffic_subjects"
            expanded["_instance_index"] = instance_index
            expanded["_instance_total"] = count
            expanded["_priority"] = "subject"
            expanded["_expanded_id"] = (
                expanded["id"]
                if count == 1
                else f"{expanded['id']}_{instance_index}"
            )
            yield expanded


def _spawn_blueprint_for_category(category: str, subtype: str) -> Tuple[str, str]:
    normalized_category = _slugify(category)
    normalized_subtype = _slugify(subtype)
    if normalized_category in {"motorcycle"}:
        return "vehicle", "motorcycle"
    if normalized_category in {"bicycle", "bike"}:
        return "vehicle", "bike"
    if normalized_category in {"truck"}:
        return "vehicle", "truck"
    if normalized_category in {"bus"}:
        return "vehicle", "truck"
    if normalized_category in {"pedestrian"}:
        return "pedestrian", "pedestrian"
    if normalized_category in {"cone_group"}:
        return "static", "constructioncone"
    if normalized_category in {"barrier_group"}:
        return "static", "streetbarrier"
    if normalized_category in {"parked_vehicle"}:
        if normalized_subtype in {
            "truck",
            "bus",
            "van",
            "suv",
            "motorcycle",
            "motor_scooter",
            "scooter",
            "motorbike",
            "two_wheeler",
            "bike",
            "bicycle",
        }:
            if normalized_subtype == "bus":
                return "vehicle", "truck"
            if normalized_subtype in {"motor_scooter", "scooter", "motorbike", "two_wheeler"}:
                return "vehicle", "motorcycle"
            if normalized_subtype == "bicycle":
                return "vehicle", "bike"
            return "vehicle", normalized_subtype
        return "vehicle", "car"
    return "vehicle", "car"


def build_relation_dsl(
    scene_understanding: Dict[str, Any],
    spawn_context: Optional[Dict[str, Any]],
    road_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    anchor_lane = select_anchor_lane(spawn_context)
    lane_groups = _coerce_list(scene_understanding.get("road_network", {}).get("lane_groups"))
    primary_lane_group = lane_groups[0] if lane_groups else {}
    lane_width_class = str(primary_lane_group.get("lane_width_class") or "standard")

    ego_relations = []
    rank_counters: Dict[Tuple[str, str], int] = {}
    for expanded in _iter_expanded_entities(scene_understanding):
        source_key = expanded.pop("_source_key")
        priority = expanded.pop("_priority")
        instance_index = expanded.pop("_instance_index")
        instance_total = expanded.pop("_instance_total")
        expanded_id = expanded.pop("_expanded_id")

        category = _canonical_entity_category(expanded)
        subtype = str(expanded.get("subtype") or category)
        lane_index_relation = _canonical_lane_index(
            expanded.get("lane_index_relation"),
            expanded.get("lane_side_relation"),
        )
        lane_side_relation = (
            _lane_side_from_index(lane_index_relation, expanded.get("lane_side_relation"))
            if category in VEHICLE_CATEGORIES
            else str(expanded.get("lane_side_relation") or "same_lane")
        )
        lane_anchor, lane_index_relation, lateral_mode = _lane_anchor_for_relation(
            lane_side_relation,
            category,
            lane_index_relation,
        )
        distance_band = _distance_band_for_entity(expanded, source_key)
        order_relation = _order_relation_for_entity(expanded, source_key)
        rank_key = (distance_band, lane_anchor)
        distance_order_rank = rank_counters.get(rank_key, 0)
        rank_counters[rank_key] = distance_order_rank + 1
        spawn_kind, blueprint_name = _spawn_blueprint_for_category(category, subtype)

        spacing_m = 0.0
        sub = _slugify(subtype)
        if (
            expanded.get("density_role") == "curbside_row"
            or "row" in sub
            or sub in {"parked_cars", "parked_vehicle", "parked_vehicle_partial", "partial_parked_vehicle"}
        ):
            if sub in {"motorcycle", "motor_scooter", "scooter", "motorbike", "two_wheeler"}:
                spacing_m = 2.5
            elif sub in {"bicycle", "bike"}:
                spacing_m = 1.5
            else:
                spacing_m = 6.0
        elif category == "pedestrian" and source_key == "background_traffic":
            spacing_m = 3.0
        elif category in STATIC_CATEGORIES:
            spacing_m = 1.2

        ego_relations.append(
            {
                "entity_id": expanded_id,
                "group_id": expanded.get("id", expanded_id),
                "source": "traffic_subject",
                "priority": priority,
                "category": category,
                "subtype": subtype,
                "spawn_kind": spawn_kind,
                "blueprint_name": blueprint_name,
                "road_id": int(anchor_lane.get("road_id", 1)),
                "lane_anchor": lane_anchor,
                "lane_index_relation": lane_index_relation,
                "lateral_mode": lateral_mode,
                "reference_entity": "ego",
                "order_relation": order_relation,
                "distance_band": distance_band,
                "distance_order_rank": distance_order_rank,
                "heading_relation": str(expanded.get("heading_relation_to_ego") or "unknown"),
                "flow_compliance": _canonical_flow_compliance(expanded.get("flow_compliance")),
                "lane_side_relation": lane_side_relation,
                "motion_state": str(expanded.get("motion_state") or expanded.get("motion_bias") or "unknown"),
                "turn_intent": _canonical_turn_intent(expanded.get("turn_intent")),
                "layout_anchor_id": str(expanded.get("layout_anchor_id") or "").strip(),
                "anchor_relation": _deep_copy(_coerce_dict(expanded.get("anchor_relation"))),
                "group_instance_index": instance_index,
                "group_instance_total": instance_total,
                "group_spacing_m": spacing_m,
                "evidence": str(expanded.get("evidence") or ""),
                "lane_width_class": lane_width_class,
                "visual_confidence": str(
                    expanded.get("visual_confidence")
                    or expanded.get("confidence")
                    or "medium"
                ),
                "appearance": _deep_copy(_coerce_dict(expanded.get("appearance"))),
            }
        )

    pairwise_relations = build_pairwise_relations(
        ego_relations,
        scene_understanding.get("key_pairwise_relations"),
    )
    return {
        "selected_anchor_lane": anchor_lane,
        "topology_sample": _coerce_list((spawn_context or {}).get("topology_sample")),
        "spawn_points": _coerce_list((spawn_context or {}).get("spawn_points")),
        "lane_width_class": lane_width_class,
        "road_type": scene_understanding.get("road_network", {}).get("road_type", "urban_straight"),
        "ego_relations": ego_relations,
        "entities": ego_relations,
        "pairwise_relations": pairwise_relations,
        "lane_context": _lane_context(scene_understanding, anchor_lane),
        "metadata": {
            "schema_version": "relation-dsl-v1",
            "road_artifact_summary": (road_artifact or {}).get("summary"),
        },
    }


def build_coordinate_program_source(
    relation_dsl_path: str,
    output_path: str,
) -> str:
    return f"""import json
import math
import os

DISTANCE_BAND_METERS = {DISTANCE_BAND_METERS!r}
LANE_WIDTH_METERS = {LANE_WIDTH_METERS!r}
PARKING_LANE_WIDTH_METERS = {PARKING_LANE_WIDTH_METERS!r}
TURN_INTENT_YAW_DEG = {TURN_INTENT_YAW_DEG!r}

RELATION_DSL_PATH = {os.path.basename(relation_dsl_path)!r}
OUTPUT_PATH = {os.path.basename(output_path)!r}


def _lane_width_for_class(name):
    return float(LANE_WIDTH_METERS.get(str(name or "standard"), 3.5))


def _apply_turn_intent(yaw, turn_intent):
    intent = str(turn_intent or "none").strip().lower()
    if intent in ("left", "turn_left", "left_turn"):
        return yaw - TURN_INTENT_YAW_DEG
    if intent in ("right", "turn_right", "right_turn"):
        return yaw + TURN_INTENT_YAW_DEG
    return yaw


def _normalize(vx, vy):
    length = math.hypot(vx, vy)
    if length <= 1e-6:
        return 1.0, 0.0
    return vx / length, vy / length


def _relation_side_sign(lane_side_relation):
    return -1.0 if "left" in str(lane_side_relation or "") else 1.0


def _parking_lane_count_for_relation(lane_side_relation, lane_context):
    lane_roles = (lane_context or {{}}).get("lane_roles") or {{}}
    relation = str(lane_side_relation or "")
    if "left" in relation:
        return int(lane_roles.get("left_parking_lane_count", 0) or 0)
    if "right" in relation:
        return int(lane_roles.get("right_parking_lane_count", 0) or 0)
    return 0


def _lane_index_from_relation(value):
    relation = str(value or "").lower()
    if relation in {{"", "same_lane", "crosswalk"}}:
        return 0
    if "right" in relation:
        return 1
    if "left" in relation or "opposing" in relation or "oncoming" in relation:
        return -1
    return 0


def _canonical_lane_index(value, fallback_relation=None):
    try:
        return int(float(value))
    except Exception:
        return _lane_index_from_relation(fallback_relation)


def _curbside_offset_for_lane_context(lane_side_relation, lane_width, lane_context=None):
    side_sign = _relation_side_sign(lane_side_relation)
    parking_lane_count = _parking_lane_count_for_relation(lane_side_relation, lane_context or {{}})
    if parking_lane_count > 0:
        return side_sign * (lane_width * 0.5 + PARKING_LANE_WIDTH_METERS * 0.5)
    return side_sign * (lane_width * 0.5)


def _lane_anchor_offset(entity, lane_width, lane_context=None):
    anchor = entity.get("lane_anchor")
    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        entity.get("lane_side_relation"),
    )
    if anchor == "center_of_lane":
        return lane_index * lane_width
    if anchor == "lane_center":
        return lane_index * lane_width
    if anchor == "left_lane_center":
        return -lane_width + lane_index * lane_width
    if anchor == "right_lane_center":
        return lane_width + lane_index * lane_width
    if anchor == "left_curbside":
        return _curbside_offset_for_lane_context(
            entity.get("lane_side_relation") or "left_edge",
            lane_width,
            lane_context,
        )
    if anchor == "right_curbside":
        return _curbside_offset_for_lane_context(
            entity.get("lane_side_relation") or "right_edge",
            lane_width,
            lane_context,
        )
    if anchor == "crosswalk_band":
        return 0.0
    if anchor == "sidewalk_band":
        lane_side = str(entity.get("lane_side_relation") or "")
        side = -1.0 if "left" in lane_side else 1.0
        return side * (lane_width * 1.5 + 1.5)
    return lane_index * lane_width


def _longitudinal_distance(entity):
    order_relation = str(entity.get("order_relation") or "ahead")
    distance_band = str(entity.get("distance_band") or "near")
    base = float(DISTANCE_BAND_METERS.get(distance_band, 10.0))
    rank = int(entity.get("distance_order_rank", 0))
    base += rank * 4.0
    spacing = float(entity.get("group_spacing_m", 0.0))
    base += int(entity.get("group_instance_index", 0)) * spacing
    if order_relation == "behind":
        return -base
    if order_relation == "alongside":
        return 0.0
    return base


def main():
    base_dir = os.path.dirname(__file__)
    relation_path = os.path.join(base_dir, RELATION_DSL_PATH)
    output_path = os.path.join(base_dir, OUTPUT_PATH)

    with open(relation_path, "r", encoding="utf-8") as file:
        relation_dsl = json.load(file)

    anchor = relation_dsl["selected_anchor_lane"]
    start = anchor["start"]
    end = anchor["end"]
    lane_width = _lane_width_for_class(relation_dsl.get("lane_width_class"))
    forward_x, forward_y = _normalize(
        float(end["x"]) - float(start["x"]),
        float(end["y"]) - float(start["y"]),
    )
    right_x, right_y = -forward_y, forward_x
    anchor_yaw = math.degrees(math.atan2(forward_y, forward_x))
    anchor_x = float(start["x"])
    anchor_y = float(start["y"])
    anchor_z = float(start.get("z", 0.0))
    lane_context = relation_dsl.get("lane_context") or {{}}

    entities = []
    for entity in relation_dsl.get("ego_relations") or relation_dsl.get("entities", []):
        longitudinal = _longitudinal_distance(entity)
        lateral = _lane_anchor_offset(entity, lane_width, lane_context)
        base_x = anchor_x + forward_x * longitudinal + right_x * lateral
        base_y = anchor_y + forward_y * longitudinal + right_y * lateral
        heading_relation = str(entity.get("heading_relation") or "unknown")
        yaw = anchor_yaw
        if heading_relation == "opposite_direction":
            yaw = anchor_yaw + 180.0
        elif heading_relation == "crossing":
            yaw = anchor_yaw + 90.0
        yaw = _apply_turn_intent(yaw, entity.get("turn_intent"))

        category = str(entity.get("category") or "car")
        if category in ("pedestrian",):
            z = 0.3
        elif category in ("cone_group", "barrier_group"):
            z = 0.5
        else:
            z = 0.3

        entities.append(
            {{
                "id": entity["entity_id"],
                "group_id": entity.get("group_id"),
                "source": entity.get("source"),
                "priority": entity.get("priority"),
                "category": category,
                "subtype": entity.get("subtype"),
                "spawn_kind": entity.get("spawn_kind"),
                "blueprint_name": entity.get("blueprint_name"),
                "motion_state": entity.get("motion_state"),
                "lane_side_relation": entity.get("lane_side_relation"),
                "heading_relation": heading_relation,
                "road_id": entity.get("road_id"),
                "layout_anchor_id": entity.get("layout_anchor_id"),
                "anchor_relation": entity.get("anchor_relation"),
                "appearance": entity.get("appearance"),
                "location": {{"x": base_x, "y": base_y, "z": z}},
                "rotation": {{"pitch": 0.0, "yaw": yaw, "roll": 0.0}},
            }}
        )

    payload = {{
        "selected_anchor_lane": anchor,
        "entities": entities,
        "metadata": {{
            **relation_dsl.get("metadata", {{}}),
            "coordinate_stage": "initial",
        }},
    }}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
"""


def _initial_lane_width_for_class(name: Any) -> float:
    return float(LANE_WIDTH_METERS.get(str(name or "standard"), 3.5))


def _curbside_offset_for_lane_context(
    lane_side_relation: Any,
    lane_width: float,
    lane_context: Optional[Dict[str, Any]] = None,
) -> float:
    side_sign = _relation_side_sign(str(lane_side_relation or ""))
    parking_lane_count = (
        _parking_lane_count_for_relation(str(lane_side_relation or ""), lane_context or {})
        if lane_context
        else 0
    )
    if parking_lane_count > 0:
        return side_sign * (lane_width * 0.5 + PARKING_LANE_WIDTH_METERS * 0.5)
    return side_sign * (lane_width * 0.5)


def _initial_lane_anchor_offset(
    entity: Dict[str, Any],
    lane_width: float,
    lane_context: Optional[Dict[str, Any]] = None,
) -> float:
    anchor = entity.get("lane_anchor")
    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        entity.get("lane_side_relation"),
    )
    if anchor == "center_of_lane":
        return lane_index * lane_width
    if anchor == "lane_center":
        return lane_index * lane_width
    if anchor == "left_lane_center":
        return -lane_width + lane_index * lane_width
    if anchor == "right_lane_center":
        return lane_width + lane_index * lane_width
    if anchor == "left_curbside":
        return _curbside_offset_for_lane_context(
            entity.get("lane_side_relation") or "left_edge",
            lane_width,
            lane_context,
        )
    if anchor == "right_curbside":
        return _curbside_offset_for_lane_context(
            entity.get("lane_side_relation") or "right_edge",
            lane_width,
            lane_context,
        )
    if anchor == "crosswalk_band":
        return 0.0
    if anchor == "sidewalk_band":
        lane_side = str(entity.get("lane_side_relation") or "")
        side = -1.0 if "left" in lane_side else 1.0
        return side * (lane_width * 1.5 + 1.5)
    return lane_index * lane_width


def _initial_longitudinal_distance(entity: Dict[str, Any]) -> float:
    order_relation = str(entity.get("order_relation") or "ahead")
    distance_band = str(entity.get("distance_band") or "near")
    base = float(DISTANCE_BAND_METERS.get(distance_band, 10.0))
    base += int(entity.get("distance_order_rank", 0)) * 4.0
    base += int(entity.get("group_instance_index", 0)) * float(
        entity.get("group_spacing_m", 0.0)
    )
    if order_relation == "behind":
        return -base
    if order_relation == "alongside":
        return 0.0
    return base


def generate_initial_coordinates_from_relation_dsl(
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate initial coordinates directly from the relation DSL.

    This mirrors the math emitted by ``build_coordinate_program_source`` while
    keeping the default VLM path fully in memory.
    """
    anchor = _coerce_dict(relation_dsl.get("selected_anchor_lane")) or _fallback_anchor_lane()
    start = _coerce_dict(anchor.get("start"))
    end = _coerce_dict(anchor.get("end"))
    lane_width = _initial_lane_width_for_class(relation_dsl.get("lane_width_class"))
    lane_context = _coerce_dict(relation_dsl.get("lane_context"))
    forward_x, forward_y = _normalize_vector(
        float(end.get("x", 0.0)) - float(start.get("x", 0.0)),
        float(end.get("y", 0.0)) - float(start.get("y", 0.0)),
    )
    right_x, right_y = -forward_y, forward_x
    anchor_yaw = math.degrees(math.atan2(forward_y, forward_x))
    anchor_x = float(start.get("x", 0.0))
    anchor_y = float(start.get("y", 0.0))

    entities = []
    for entity in _coerce_list(
        relation_dsl.get("ego_relations") or relation_dsl.get("entities")
    ):
        longitudinal = _initial_longitudinal_distance(entity)
        heading_relation = str(entity.get("heading_relation") or "unknown")
        if heading_relation == "opposite_direction" and _is_vehicle_like(
            str(entity.get("category") or "")
        ):
            lateral = _opposing_carriageway_offset(
                entity.get("lane_index_relation"), lane_width, lane_context
            )
        else:
            lateral = _initial_lane_anchor_offset(entity, lane_width, lane_context)
        base_x = anchor_x + forward_x * longitudinal + right_x * lateral
        base_y = anchor_y + forward_y * longitudinal + right_y * lateral
        yaw = anchor_yaw
        if heading_relation == "opposite_direction":
            yaw = anchor_yaw + 180.0
        elif heading_relation == "crossing":
            yaw = anchor_yaw + 90.0
        yaw = _apply_turn_intent(yaw, entity.get("turn_intent"))

        category = str(entity.get("category") or "car")
        if category in ("pedestrian",):
            z = 0.3
        elif category in ("cone_group", "barrier_group"):
            z = 0.5
        else:
            z = 0.3

        entities.append(
            {
                "id": entity["entity_id"],
                "group_id": entity.get("group_id"),
                "source": entity.get("source"),
                "priority": entity.get("priority"),
                "category": category,
                "subtype": entity.get("subtype"),
                "spawn_kind": entity.get("spawn_kind"),
                "blueprint_name": entity.get("blueprint_name"),
                "motion_state": entity.get("motion_state"),
                "turn_intent": _canonical_turn_intent(entity.get("turn_intent")),
                "lane_anchor": entity.get("lane_anchor"),
                "lane_index_relation": entity.get("lane_index_relation", 0),
                "lane_side_relation": entity.get("lane_side_relation"),
                "heading_relation": heading_relation,
                "flow_compliance": _canonical_flow_compliance(entity.get("flow_compliance")),
                "longitudinal_m": longitudinal,
                "road_id": entity.get("road_id"),
                "layout_anchor_id": entity.get("layout_anchor_id"),
                "anchor_relation": _deep_copy(_coerce_dict(entity.get("anchor_relation"))),
                "appearance": _deep_copy(entity.get("appearance")),
                "location": {"x": base_x, "y": base_y, "z": z},
                "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
            }
        )

    return {
        "selected_anchor_lane": anchor,
        "entities": entities,
        "metadata": {
            **_coerce_dict(relation_dsl.get("metadata")),
            "coordinate_stage": "initial",
        },
    }


def write_coordinate_program(
    relation_dsl_path: str,
    program_path: str,
    output_path: str,
) -> str:
    source = build_coordinate_program_source(relation_dsl_path, output_path)
    write_to_file(program_path, source)
    return source


def evaluate_coordinate_program(
    program_path: str,
    cwd: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, program_path],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=cwd or os.path.dirname(program_path) or None,
    )
    if result.returncode != 0:
        error_output = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Coordinate program failed during execution:\n"
            f"{error_output or f'Exited with status code {result.returncode}.'}"
        )

    resolved_output_path = output_path or program_path.replace(
        "_coordinate_program.py", "_coordinates_raw.json"
    )
    with open(resolved_output_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _entities_by_id(entities: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(entity.get("id") or entity.get("entity_id")): entity for entity in entities}


def _solve_priority(entity: Dict[str, Any]) -> int:
    category = str(entity.get("category") or "")
    priority = str(entity.get("priority") or "")
    if priority == "subject" and _is_vehicle_like(category):
        return 0
    if priority == "subject":
        return 1
    if _is_vehicle_like(category):
        return 2
    if category in WALKER_CATEGORIES:
        return 3
    if category in STATIC_CATEGORIES:
        return 4
    return 5


def _ordering_pair_is_active(pair: Dict[str, Any], entities_by_id: Dict[str, Dict[str, Any]]) -> bool:
    entity = entities_by_id.get(str(pair.get("entity_id")))
    other = entities_by_id.get(str(pair.get("other_entity_id")))
    if entity is None or other is None:
        return False
    strength = str(pair.get("constraint_strength") or "soft")
    if strength != "hard":
        return False
    entity_category = str(entity.get("category") or "")
    other_category = str(other.get("category") or "")
    if _is_vehicle_like(entity_category) and _is_vehicle_like(other_category):
        return True
    if (
        entity_category in STATIC_CATEGORIES or other_category in STATIC_CATEGORIES
    ) and (
        _is_vehicle_like(entity_category) or _is_vehicle_like(other_category)
    ):
        return True
    if (
        entity_category in WALKER_CATEGORIES or other_category in WALKER_CATEGORIES
    ) and (
        _is_vehicle_like(entity_category) or _is_vehicle_like(other_category)
    ):
        return "cross" in str(entity.get("heading_relation") or "") or "cross" in str(other.get("heading_relation") or "")
    return False


def _shift_entity_longitudinal(
    entity: Dict[str, Any],
    anchor_lane: Dict[str, Any],
    delta_m: float,
) -> None:
    start = _coerce_dict(anchor_lane.get("start"))
    end = _coerce_dict(anchor_lane.get("end"))
    forward_x, forward_y = _normalize_vector(
        float(end.get("x", 0.0)) - float(start.get("x", 0.0)),
        float(end.get("y", 0.0)) - float(start.get("y", 0.0)),
    )
    entity["location"]["x"] += forward_x * float(delta_m)
    entity["location"]["y"] += forward_y * float(delta_m)


def apply_pairwise_ordering(
    raw_coordinates: Dict[str, Any],
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    ordered = _deep_copy(raw_coordinates)
    entities = _coerce_list(ordered.get("entities"))
    entities_by_id = _entities_by_id(entities)
    anchor_lane = _coerce_dict(
        ordered.get("selected_anchor_lane") or relation_dsl.get("selected_anchor_lane")
    )

    for pair in _coerce_list(relation_dsl.get("pairwise_relations")):
        if not _ordering_pair_is_active(pair, entities_by_id):
            continue
        entity = entities_by_id.get(str(pair.get("entity_id")))
        other = entities_by_id.get(str(pair.get("other_entity_id")))
        if entity is None or other is None:
            continue
        longitudinal_a, _ = _relative_metrics(entity, anchor_lane)
        longitudinal_b, _ = _relative_metrics(other, anchor_lane)
        expected = str(pair.get("longitudinal_relation") or "aligned_with_other")
        target_gap = _distance_for_gap_band(pair.get("longitudinal_gap_band") or "near")

        if expected == "ahead_of_other":
            current_gap = longitudinal_a - longitudinal_b
            if current_gap >= target_gap:
                continue
            mover = entity if _solve_priority(entity) > _solve_priority(other) else other
            delta = target_gap - current_gap + 0.1
            if mover is entity:
                _shift_entity_longitudinal(entity, anchor_lane, delta)
            else:
                _shift_entity_longitudinal(other, anchor_lane, -delta)
        elif expected == "behind_other":
            current_gap = longitudinal_b - longitudinal_a
            if current_gap >= target_gap:
                continue
            mover = entity if _solve_priority(entity) > _solve_priority(other) else other
            delta = target_gap - current_gap + 0.1
            if mover is entity:
                _shift_entity_longitudinal(entity, anchor_lane, -delta)
            else:
                _shift_entity_longitudinal(other, anchor_lane, delta)
        else:
            midpoint = (longitudinal_a + longitudinal_b) / 2.0
            if _solve_priority(entity) >= _solve_priority(other):
                _shift_entity_longitudinal(entity, anchor_lane, midpoint - longitudinal_a)
            else:
                _shift_entity_longitudinal(other, anchor_lane, midpoint - longitudinal_b)

    ordered["metadata"] = {
        **_coerce_dict(ordered.get("metadata")),
        "coordinate_stage": "ordered",
    }
    return ordered


def _project_point_to_segment(
    point_x: float,
    point_y: float,
    start: Dict[str, Any],
    end: Dict[str, Any],
) -> Tuple[float, float, float]:
    sx = float(start["x"])
    sy = float(start["y"])
    ex = float(end["x"])
    ey = float(end["y"])
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return sx, sy, 0.0
    t = ((point_x - sx) * dx + (point_y - sy) * dy) / denom
    t = max(0.0, min(1.0, t))
    return sx + t * dx, sy + t * dy, t


def _build_dense_waypoint_index(
    dense_waypoints: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Group dense waypoints by (road_id, lane_id) for O(1) lane lookup."""
    index: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for wp in dense_waypoints:
        key = (int(wp.get("road_id", 0) or 0), int(wp.get("lane_id", 0) or 0))
        index.setdefault(key, []).append(wp)
    return index


def _snap_entity_to_dense_waypoints(
    entity: Dict[str, Any],
    dense_index: Dict[Tuple[int, int], List[Dict[str, Any]]],
    max_snap_distance_m: float = 8.0,
) -> Dict[str, Any]:
    """Snap z/yaw (and x/y for driving-lane entities) to the nearest dense waypoint.

    Edge/sidewalk entities keep their offset-applied x/y but receive the road
    surface z.  Driving-lane entities are moved to the nearest road point so
    that curve geometry is respected.
    """
    projected_lane = entity.get("projected_lane") or {}
    road_id = projected_lane.get("road_id")
    lane_id = projected_lane.get("lane_id")
    lane_side = str(entity.get("lane_side_relation") or "same_lane")
    ex = float((entity.get("location") or {}).get("x", 0.0))
    ey = float((entity.get("location") or {}).get("y", 0.0))

    # Prefer exact road+lane match, then road-only, then all waypoints.
    key = (int(road_id or 0), int(lane_id or 0))
    candidates: List[Dict[str, Any]] = dense_index.get(key) or []
    exact_lane_candidates = bool(candidates)
    if not candidates and road_id is not None:
        rid = int(road_id)
        candidates = [wp for (r, _), wps in dense_index.items() if r == rid for wp in wps]
    if not candidates:
        candidates = [wp for wps in dense_index.values() for wp in wps]

    best: Optional[Dict[str, Any]] = None
    best_dist = float("inf")
    for wp in candidates:
        dist = math.sqrt((wp["x"] - ex) ** 2 + (wp["y"] - ey) ** 2)
        if dist < best_dist:
            best_dist = dist
            best = wp

    if best is None or best_dist > max_snap_distance_m:
        return entity

    snapped = dict(entity)
    loc = dict(entity.get("location") or {})
    rot = dict(entity.get("rotation") or {})

    is_offset_entity = lane_side in {
        "left_edge", "right_edge", "sidewalk_left", "sidewalk_right",
    }
    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        entity.get("lane_side_relation"),
    )
    if lane_index != 0 and not exact_lane_candidates:
        is_offset_entity = True
    loc["z"] = max(float(best.get("z") or 0.3), 0.3)

    if not is_offset_entity:
        loc["x"] = float(best["x"])
        loc["y"] = float(best["y"])
        heading_relation = str(entity.get("heading_relation") or "unknown")
        projected_lane = _coerce_dict(entity.get("projected_lane"))
        lane_reversed = bool(projected_lane.get("reversed_to_anchor"))
        road_yaw = float(best.get("yaw") or rot.get("yaw") or 0.0)
        rot["yaw"] = _yaw_for_heading_relation(
            road_yaw,
            heading_relation,
            lane_reversed=lane_reversed,
            turn_intent=entity.get("turn_intent"),
        )

    snapped["location"] = loc
    snapped["rotation"] = rot
    return snapped


def _entity_min_distance(entity: Dict[str, Any]) -> float:
    category = str(entity.get("category") or "")
    if category == "pedestrian":
        return 0.8
    if category in STATIC_CATEGORIES:
        return 0.5
    if category in {"truck", "bus"}:
        return 7.0
    if category == "car":
        return 5.0
    if category in {"motorcycle", "bicycle"}:
        return 3.0
    return 2.5


def _priority_rank(entity: Dict[str, Any]) -> int:
    return 0 if entity.get("priority") == "subject" else 1


def _distance_between(entity_a: Dict[str, Any], entity_b: Dict[str, Any]) -> float:
    ax = float(entity_a["location"]["x"])
    ay = float(entity_a["location"]["y"])
    bx = float(entity_b["location"]["x"])
    by = float(entity_b["location"]["y"])
    return math.hypot(ax - bx, ay - by)


def _resolve_collisions(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resolved = [_deep_copy(entity) for entity in entities]
    for index, entity in enumerate(resolved):
        min_distance = _entity_min_distance(entity)
        yaw = math.radians(float(entity["rotation"]["yaw"]))
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        for other_index in range(index):
            other = resolved[other_index]
            other_min = _entity_min_distance(other)
            threshold = max(min_distance, other_min)
            distance = _distance_between(entity, other)
            if distance >= threshold:
                continue
            if _priority_rank(entity) < _priority_rank(other):
                mover = other
            else:
                mover = entity
            shift = threshold - distance + 0.1
            mover["location"]["x"] += forward_x * shift
            mover["location"]["y"] += forward_y * shift
    return resolved


def _topology_has_multiple_lanes(topology_sample: List[Dict[str, Any]]) -> bool:
    unique_lanes = {
        (
            int(_coerce_dict(lane).get("road_id", 0)),
            int(_coerce_dict(lane).get("lane_id", 0)),
        )
        for lane in topology_sample
        if _coerce_dict(lane).get("lane_id") not in (0, None)
    }
    return len(unique_lanes) > 1


def _relation_side_sign(lane_side_relation: str) -> float:
    relation = str(lane_side_relation or "")
    return -1.0 if "left" in relation else 1.0


def _parking_lane_count_for_relation(
    lane_side_relation: str,
    lane_context: Dict[str, Any],
) -> int:
    lane_roles = _coerce_dict(lane_context.get("lane_roles"))
    relation = str(lane_side_relation or "")
    if "left" in relation:
        return int(lane_roles.get("left_parking_lane_count", 0) or 0)
    if "right" in relation:
        return int(lane_roles.get("right_parking_lane_count", 0) or 0)
    return 0


def _vehicle_lateral_offset(
    entity: Dict[str, Any],
    lane_width: float,
    lane_context: Dict[str, Any],
) -> float:
    lane_side_relation = str(entity.get("lane_side_relation") or "")
    if lane_side_relation in {"", "same_lane", "crosswalk"}:
        return 0.0

    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        entity.get("lane_side_relation"),
    )
    if _is_vehicle_like(str(entity.get("category") or "")):
        return lane_index * lane_width

    side_sign = _relation_side_sign(lane_side_relation)
    parking_lane_count = _parking_lane_count_for_relation(lane_side_relation, lane_context)
    category = str(entity.get("category") or "")

    if lane_side_relation in {"left_edge", "right_edge"}:
        if parking_lane_count > 0:
            magnitude = lane_width * 0.5 + PARKING_LANE_WIDTH_METERS * 0.5
        else:
            magnitude = lane_width * 0.5
        return side_sign * magnitude

    if lane_side_relation in {"sidewalk_left", "sidewalk_right"}:
        magnitude = lane_width * 0.5
        if parking_lane_count > 0:
            magnitude += PARKING_LANE_WIDTH_METERS * parking_lane_count
        magnitude += SIDEWALK_BUFFER_METERS
        return side_sign * magnitude

    if lane_side_relation in {"left_lane", "right_lane"} or "opposing" in lane_side_relation:
        return side_sign * lane_width

    # Fallback: any unrecognized curbside/parking relation still gets edge placement.
    # This handles LLM-generated variants not yet in CANONICAL_LANE_SIDE_MAP.
    if "curbside" in lane_side_relation or (
        "parking" in lane_side_relation and lane_side_relation not in {"same_lane"}
    ):
        magnitude = lane_width * 0.5
        return side_sign * magnitude

    return 0.0


def _lane_is_reversed(lane: Dict[str, Any], anchor_lane: Dict[str, Any]) -> bool:
    """Return True when lane travels in roughly the opposite direction to anchor_lane.

    Used to detect when an entity has been placed on an opposing lane whose yaw is
    already reversed relative to the ego direction.  In that case the
    heading_relation_to_ego must be interpreted in the frame of that lane, not the
    anchor frame, to avoid a double 180° flip.
    """
    lane_yaw = float((lane.get("start") or {}).get("yaw", 0.0) or 0.0)
    anchor_yaw = float((anchor_lane.get("start") or {}).get("yaw", 0.0) or 0.0)
    diff = abs(((lane_yaw - anchor_yaw + 180.0) % 360.0) - 180.0)
    return diff > 90.0


def _yaw_for_heading_relation(
    road_yaw: float,
    heading_relation: Any,
    *,
    lane_reversed: bool = False,
    turn_intent: Any = None,
) -> float:
    """Resolve actor yaw in the selected lane's frame.

    If the selected lane already points opposite the ego lane, its waypoint yaw
    already represents oncoming traffic.  Applying another 180 degree turn would
    flip the actor back to ego direction.

    A ``turn_intent`` (left/right) further deflects the resolved heading toward
    the junction branch the actor is entering, so a vehicle captured mid-turn no
    longer renders as if it were following its lane straight through.
    """
    relation = str(heading_relation or "unknown")
    yaw = float(road_yaw)
    if relation == "crossing":
        return _apply_turn_intent(yaw + 90.0, turn_intent)
    if lane_reversed:
        if relation == "same_direction":
            return _apply_turn_intent(yaw + 180.0, turn_intent)
        return _apply_turn_intent(yaw, turn_intent)
    if relation == "opposite_direction":
        return _apply_turn_intent(yaw + 180.0, turn_intent)
    return _apply_turn_intent(yaw, turn_intent)


def _opposing_carriageway_offset(
    lane_index: Any,
    lane_width: float,
    lane_context: Optional[Dict[str, Any]] = None,
) -> float:
    """Lateral offset (road frame, left negative) onto the opposing carriageway.

    Oncoming vehicles must clear *all* of ego's same-direction lanes plus the
    center median before landing in the opposing lanes, otherwise a naive
    ``lane_index * lane_width`` offset leaves them in an adjacent same-direction
    lane. We deliberately cross the full forward-lane count (worst case: ego in
    the rightmost forward lane) so the seed is always on the far side of the
    divider; runtime lane projection then snaps to the nearest real opposing
    lane. ``lane_index`` selects which opposing lane (clamped to the count).
    """
    roles = _coerce_dict((lane_context or {}).get("lane_roles"))
    forward_lane_count = max(1, int(roles.get("forward_lane_count", 1) or 1))
    opposing_lane_count = max(1, int(roles.get("opposing_lane_count", 1) or 1))
    has_center_median = bool(roles.get("has_center_median"))
    opposing_index = min(max(1, abs(_canonical_lane_index(lane_index)) or 1), opposing_lane_count)
    if has_center_median:
        # Divided road: clear every same-direction lane (worst case: ego in the
        # rightmost forward lane) plus the median before entering the opposing lanes.
        steps = forward_lane_count + MEDIAN_GAP_LANE_EQUIV + (opposing_index - 0.5)
    else:
        # Undivided road: the opposing lanes sit directly to ego's left.
        steps = opposing_index
    return -steps * float(lane_width)


def _project_vehicle_entity(
    entity: Dict[str, Any],
    lane: Dict[str, Any],
    lane_width: float,
    lane_context: Dict[str, Any],
    anchor_lane: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    projected = _deep_copy(entity)
    px, py = _project_point_to_segment(
        float(entity["location"]["x"]),
        float(entity["location"]["y"]),
        lane["start"],
        lane["end"],
    )[:2]
    road_yaw = float(lane["start"].get("yaw", 0.0))
    heading_relation = str(entity.get("heading_relation") or "unknown")
    lane_reversed = bool(anchor_lane is not None and _lane_is_reversed(lane, anchor_lane))
    yaw = _yaw_for_heading_relation(
        road_yaw,
        heading_relation,
        lane_reversed=lane_reversed,
        turn_intent=entity.get("turn_intent"),
    )

    # Prefer lane_anchor-based offset (same formula used in initial coordinate
    # generation) so the two stages stay consistent. Fall back to the
    # lane_side_relation heuristic when lane_anchor was not propagated.
    if entity.get("lane_anchor"):
        offset = _initial_lane_anchor_offset(entity, lane_width, lane_context)
    else:
        offset = _vehicle_lateral_offset(entity, lane_width, lane_context)
    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        entity.get("lane_side_relation"),
    )
    if (
        _is_vehicle_like(str(entity.get("category") or ""))
        and lane_index != 0
        and int(lane.get("lane_id", 0) or 0)
        != int((anchor_lane or {}).get("lane_id", 0) or 0)
    ):
        offset = 0.0
    # Oncoming vehicle that fell back to the ego-lane frame (no real opposing lane
    # in the topology sample): push it across the median onto the opposing
    # carriageway. When a real opposing lane was selected instead, lane_id differs
    # from the anchor and the offset is already zeroed just above. lane_reversed is
    # False here (lane == anchor), so _yaw_for_heading_relation keeps the flip.
    if (
        heading_relation == "opposite_direction"
        and _is_vehicle_like(str(entity.get("category") or ""))
        and int(lane.get("lane_id", 0) or 0)
        == int((anchor_lane or {}).get("lane_id", 0) or 0)
    ):
        offset = _opposing_carriageway_offset(lane_index, lane_width, lane_context)
    if abs(offset) > 1e-6:
        # Keep lane-side semantics in the road frame rather than the actor frame.
        # Otherwise an oncoming vehicle would flip left/right when its yaw is reversed.
        # road_yaw + 90 is the rightward perpendicular in CARLA's coordinate system
        # (x=east, y=south, yaw clockwise): right = (-sin yaw, cos yaw) = yaw+90 direction.
        normal_yaw = math.radians(road_yaw + 90.0)
        px += math.cos(normal_yaw) * offset
        py += math.sin(normal_yaw) * offset
    projected["location"] = {"x": px, "y": py, "z": 0.3}
    projected["rotation"] = {"pitch": 0.0, "yaw": yaw, "roll": 0.0}
    projected["projected_lane_reversed_to_anchor"] = lane_reversed
    return projected


def _project_pedestrian_entity(entity: Dict[str, Any], lane: Dict[str, Any], lane_width: float) -> Dict[str, Any]:
    projected = _deep_copy(entity)
    px, py = _project_point_to_segment(
        float(entity["location"]["x"]),
        float(entity["location"]["y"]),
        lane["start"],
        lane["end"],
    )[:2]
    lane_side = str(entity.get("lane_side_relation") or "")
    if lane_side == "crosswalk":
        offset = 0.0
    else:
        side = -1.0 if "left" in lane_side else 1.0
        offset = side * (lane_width * 1.5 + 1.5)
    normal_yaw = math.radians(float(lane["start"].get("yaw", 0.0)) + 90.0)
    px += math.cos(normal_yaw) * offset
    py += math.sin(normal_yaw) * offset
    projected["location"] = {"x": px, "y": py, "z": 0.3}
    return projected


def _project_static_entity(entity: Dict[str, Any], lane: Dict[str, Any], lane_width: float) -> Dict[str, Any]:
    projected = _deep_copy(entity)
    px, py = _project_point_to_segment(
        float(entity["location"]["x"]),
        float(entity["location"]["y"]),
        lane["start"],
        lane["end"],
    )[:2]
    side = -1.0 if "left" in str(entity.get("lane_side_relation") or "") else 1.0
    normal_yaw = math.radians(float(lane["start"].get("yaw", 0.0)) + 90.0)
    # Use 1.5× lane_width so cones land at the road edge/shoulder, not inside the adjacent lane.
    offset = side * (lane_width * 1.5)
    px += math.cos(normal_yaw) * offset
    py += math.sin(normal_yaw) * offset
    projected["location"] = {"x": px, "y": py, "z": 0.5}
    projected["rotation"] = {
        "pitch": 0.0,
        "yaw": float(lane["start"].get("yaw", 0.0)),
        "roll": 0.0,
    }
    return projected


def _closest_lane(point: Dict[str, Any], topology_sample: List[Dict[str, Any]], fallback_lane: Dict[str, Any]) -> Dict[str, Any]:
    if not topology_sample:
        return fallback_lane

    best_lane = fallback_lane
    best_distance = None
    for lane in topology_sample:
        projected_x, projected_y, _ = _project_point_to_segment(
            float(point["x"]),
            float(point["y"]),
            lane["start"],
            lane["end"],
        )
        distance = math.hypot(projected_x - float(point["x"]), projected_y - float(point["y"]))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_lane = lane
    return best_lane


def _normalize_vector(vx: float, vy: float) -> Tuple[float, float]:
    length = math.hypot(vx, vy)
    if length <= 1e-6:
        return 1.0, 0.0
    return vx / length, vy / length


def _relative_metrics(
    entity: Dict[str, Any],
    anchor_lane: Dict[str, Any],
) -> Tuple[float, float]:
    start = _coerce_dict(anchor_lane.get("start"))
    end = _coerce_dict(anchor_lane.get("end"))
    forward_x, forward_y = _normalize_vector(
        float(end.get("x", 0.0)) - float(start.get("x", 0.0)),
        float(end.get("y", 0.0)) - float(start.get("y", 0.0)),
    )
    right_x, right_y = -forward_y, forward_x
    dx = float(entity.get("location", {}).get("x", 0.0)) - float(start.get("x", 0.0))
    dy = float(entity.get("location", {}).get("y", 0.0)) - float(start.get("y", 0.0))
    longitudinal = dx * forward_x + dy * forward_y
    lateral = dx * right_x + dy * right_y
    return longitudinal, lateral


def validate_pairwise_relations(
    relation_dsl: Dict[str, Any],
    projected_coordinates: Dict[str, Any],
    *,
    longitudinal_tolerance_m: float = 2.5,
    lateral_tolerance_m: float = 1.0,
) -> Dict[str, Any]:
    entities_by_id = {
        str(entity.get("id")): entity
        for entity in _coerce_list(projected_coordinates.get("entities"))
        if entity.get("id") is not None
    }
    anchor_lane = _coerce_dict(
        projected_coordinates.get("selected_anchor_lane")
        or relation_dsl.get("selected_anchor_lane")
    )
    results = []
    passed = 0
    failed = 0

    for pair in _coerce_list(relation_dsl.get("pairwise_relations")):
        entity = entities_by_id.get(str(pair.get("entity_id")))
        other = entities_by_id.get(str(pair.get("other_entity_id")))
        if entity is None or other is None:
            results.append(
                {
                    **_coerce_dict(pair),
                    "status": "skipped",
                    "reason": "missing_entity_in_projected_coordinates",
                }
            )
            continue

        longitudinal_a, lateral_a = _relative_metrics(entity, anchor_lane)
        longitudinal_b, lateral_b = _relative_metrics(other, anchor_lane)
        longitudinal_delta = longitudinal_a - longitudinal_b
        lateral_delta = lateral_a - lateral_b

        expected_longitudinal = str(
            pair.get("longitudinal_relation")
            or pair.get("expected_longitudinal_to_other")
            or "aligned_with_other"
        )
        expected_lateral = str(
            pair.get("lateral_relation")
            or pair.get("expected_lateral_to_other")
            or "same_lateral_band"
        )

        longitudinal_ok = True
        if expected_longitudinal == "ahead_of_other":
            longitudinal_ok = longitudinal_delta > longitudinal_tolerance_m
        elif expected_longitudinal == "behind_other":
            longitudinal_ok = longitudinal_delta < -longitudinal_tolerance_m
        elif expected_longitudinal == "aligned_with_other":
            longitudinal_ok = abs(longitudinal_delta) <= longitudinal_tolerance_m

        lateral_ok = True
        if expected_lateral == "left_of_other":
            lateral_ok = lateral_delta < -lateral_tolerance_m
        elif expected_lateral == "right_of_other":
            lateral_ok = lateral_delta > lateral_tolerance_m
        elif expected_lateral == "same_lateral_band":
            lateral_ok = abs(lateral_delta) <= lateral_tolerance_m

        status = "pass" if longitudinal_ok and lateral_ok else "fail"
        if status == "pass":
            passed += 1
        else:
            failed += 1
        results.append(
            {
                **_coerce_dict(pair),
                "status": status,
                "actual_longitudinal_delta_m": round(longitudinal_delta, 3),
                "actual_lateral_delta_m": round(lateral_delta, 3),
                "longitudinal_check": longitudinal_ok,
                "lateral_check": lateral_ok,
            }
        )

    return {
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": len(results) - passed - failed,
        },
        "pairwise_results": results,
        "metadata": {
            "validation_version": "pairwise-relation-validation-v1",
            "longitudinal_tolerance_m": longitudinal_tolerance_m,
            "lateral_tolerance_m": lateral_tolerance_m,
        },
    }


def _lane_legality_for_entity(entity: Dict[str, Any], lane_context: Dict[str, Any]) -> Tuple[bool, str]:
    category = str(entity.get("category") or "")
    relation = str(entity.get("lane_side_relation") or "")
    if category in VEHICLE_CATEGORIES:
        return True, "driving"
    allowed_actor_types = _coerce_dict(lane_context.get("allowed_actor_types"))
    lane_catalog = _coerce_dict(lane_context.get("lane_catalog"))
    catalog_entry = _coerce_dict(lane_catalog.get(relation))
    role = str(catalog_entry.get("role") or "driving")
    allowed = _coerce_list(allowed_actor_types.get(role))
    return category in allowed, role


def _enrich_topology_sample(
    topology_sample: List[Dict[str, Any]],
    anchor_lane: Dict[str, Any],
    lane_width: float,
) -> List[Dict[str, Any]]:
    """Append synthetic opposing and right-adjacent lane entries derived from anchor_lane.

    These are only used when no real topology segment matches an entity's semantic lane.
    Entries are tagged with _synthetic=True for debugging transparency.
    """
    anchor_road_id = int(anchor_lane.get("road_id", 0) or 0)
    anchor_lane_id = int(anchor_lane.get("lane_id", -1) or -1)
    existing_ids = {
        (int(l.get("road_id", 0) or 0), int(l.get("lane_id", 0) or 0))
        for l in topology_sample
    }

    start = _coerce_dict(anchor_lane.get("start"))
    end = _coerce_dict(anchor_lane.get("end"))
    yaw = float(start.get("yaw", 0.0) or 0.0)
    normal_rad = math.radians(yaw + 90.0)
    nx = math.cos(normal_rad)
    ny = math.sin(normal_rad)

    result = list(topology_sample)

    def add_synthetic_lane(lane_id: int, lateral_steps: int, yaw_value: float, reverse: bool = False) -> None:
        if (anchor_road_id, lane_id) in existing_ids:
            return
        start_base = end if reverse else start
        end_base = start if reverse else end
        result.append({
            "road_id": anchor_road_id,
            "lane_id": lane_id,
            "start": {
                "x": float(start_base.get("x", 0.0)) + nx * lane_width * lateral_steps,
                "y": float(start_base.get("y", 0.0)) + ny * lane_width * lateral_steps,
                "z": float(start_base.get("z", 0.0)),
                "yaw": yaw_value,
                "is_junction": False,
            },
            "end": {
                "x": float(end_base.get("x", 0.0)) + nx * lane_width * lateral_steps,
                "y": float(end_base.get("y", 0.0)) + ny * lane_width * lateral_steps,
                "z": float(end_base.get("z", 0.0)),
                "yaw": yaw_value,
                "is_junction": False,
            },
            "_synthetic": True,
        })

    opposing_lane_id = -anchor_lane_id
    if (anchor_road_id, opposing_lane_id) not in existing_ids:
        opp_yaw = (yaw + 180.0) % 360.0
        add_synthetic_lane(opposing_lane_id, -1, opp_yaw, reverse=True)

    right_lane_id = anchor_lane_id + (-1 if anchor_lane_id < 0 else 1)
    if (anchor_road_id, right_lane_id) not in existing_ids:
        add_synthetic_lane(right_lane_id, 1, yaw)

    return result


def _target_lane_id_for_index(anchor_lane_id: int, lane_index: int) -> int:
    if lane_index == 0:
        return anchor_lane_id
    anchor_sign = -1 if anchor_lane_id < 0 else 1
    if lane_index > 0:
        return anchor_lane_id + anchor_sign * lane_index
    opposite_sign = -anchor_sign
    return opposite_sign * (abs(lane_index) or 1)


def _select_semantic_lane(
    entity: Dict[str, Any],
    topology_sample: List[Dict[str, Any]],
    anchor_lane: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the topology segment best matching the entity's lane_side_relation.

    Prefers a real segment from topology_sample (including synthetic entries added by
    _enrich_topology_sample) over pure geometric nearest-neighbour selection.
    Falls back to anchor_lane when no semantic match is found; the lateral offset from
    _vehicle_lateral_offset then handles the correct lateral displacement.
    """
    lane_side = str(entity.get("lane_side_relation") or "same_lane")
    lane_index = _canonical_lane_index(
        entity.get("lane_index_relation"),
        lane_side,
    )
    anchor_road_id = int(anchor_lane.get("road_id", 0) or 0)
    anchor_lane_id = int(anchor_lane.get("lane_id", -1) or -1)

    # Drop lanes that are implausibly far from the anchor (e.g. the synthetic
    # fallback lane at the map origin that can leak into a real topology_sample).
    # Matching an actor to such a lane teleports it ~100m+ off the scene; the
    # anchor itself is always retained so callers still have a safe default.
    anchor_start = _coerce_dict(anchor_lane.get("start"))
    ax = float(anchor_start.get("x", 0.0) or 0.0)
    ay = float(anchor_start.get("y", 0.0) or 0.0)

    def _is_local_lane(lane: Dict[str, Any]) -> bool:
        s = _coerce_dict(lane.get("start"))
        return (
            math.hypot(float(s.get("x", 0.0) or 0.0) - ax, float(s.get("y", 0.0) or 0.0) - ay)
            <= MAX_LOCAL_LANE_DISTANCE_M
        )

    topology_sample = [
        lane for lane in topology_sample if lane is anchor_lane or _is_local_lane(lane)
    ]

    # Entities that stay on the ego lane use the anchor directly.
    if lane_side in {"crosswalk", "", "sidewalk_left", "sidewalk_right"}:
        return anchor_lane
    if lane_index == 0:
        return anchor_lane
    # Oncoming vehicles: prefer a real opposing (reversed) lane from topology when
    # one is known. Otherwise fall back to the ego-lane frame so
    # _project_vehicle_entity can push the seed across the median onto the opposing
    # carriageway and let runtime lane projection snap it to a real lane. The
    # fallback is essential on divided roads where the opposing carriageway is a
    # separate road that never appears in the local topology sample.
    if (
        str(entity.get("heading_relation") or "") == "opposite_direction"
        and _is_vehicle_like(str(entity.get("category") or ""))
    ):
        anchor_yaw = float(_coerce_dict(anchor_lane.get("start")).get("yaw", 0.0) or 0.0)
        for lane in topology_sample:
            if lane.get("_synthetic"):
                continue
            r = int(lane.get("road_id", -9999) or -9999)
            l = int(lane.get("lane_id", 0) or 0)
            if l == 0:
                continue
            ly = float(_coerce_dict(lane.get("start")).get("yaw", 0.0) or 0.0)
            reversed_heading = abs(((ly - anchor_yaw + 180.0) % 360.0) - 180.0) > 135.0
            same_road_opposite_sign = r == anchor_road_id and l * anchor_lane_id < 0
            if same_road_opposite_sign or reversed_heading:
                return lane
        return anchor_lane

    target_id = _target_lane_id_for_index(anchor_lane_id, lane_index)
    for lane in topology_sample:
        r = int(lane.get("road_id", -9999) or -9999)
        l = int(lane.get("lane_id", 0) or 0)
        if r == anchor_road_id and l == target_id:
            return lane

    # Opposing or left_lane: look for opposite-sign lane_id on the same road first,
    # then fall back to a heading-based search (>135° yaw difference).
    if lane_index == -1 and ("opposing" in lane_side or lane_side == "left_lane"):
        for lane in topology_sample:
            r = int(lane.get("road_id", -9999) or -9999)
            l = int(lane.get("lane_id", 0) or 0)
            if r == anchor_road_id and l != 0 and l * anchor_lane_id < 0:
                return lane
        anchor_yaw = float((anchor_lane.get("start") or {}).get("yaw", 0.0) or 0.0)
        for lane in topology_sample:
            if lane is anchor_lane:
                continue
            ly = float((lane.get("start") or {}).get("yaw", 0.0) or 0.0)
            if abs(((ly - anchor_yaw + 180) % 360) - 180) > 135:
                return lane
        return anchor_lane

    # Right adjacent lane: same road, lane_id one step more negative.
    if lane_index == 1 and lane_side == "right_lane":
        target_id = _target_lane_id_for_index(anchor_lane_id, lane_index)
        for lane in topology_sample:
            r = int(lane.get("road_id", -9999) or -9999)
            l = int(lane.get("lane_id", 0) or 0)
            if r == anchor_road_id and l == target_id:
                return lane
        return anchor_lane

    return anchor_lane


def project_entities_to_carla_context(
    raw_coordinates: Dict[str, Any],
    scene_understanding: Dict[str, Any],
    spawn_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # Prefer dense local waypoints pre-sampled after map matching (higher accuracy).
    # Fall back to the coarse topology_sample when dense data is not available.
    dense_local_waypoints = _coerce_list(
        (spawn_context or {}).get("dense_local_waypoints")
    )
    topology_sample = _coerce_list((spawn_context or {}).get("topology_sample"))
    fallback_lane = select_anchor_lane(spawn_context)
    lane_context = _lane_context(scene_understanding, fallback_lane)
    lane_groups = _coerce_list(scene_understanding.get("road_network", {}).get("lane_groups"))
    lane_width_class = str((lane_groups[0] if lane_groups else {}).get("lane_width_class") or "standard")
    lane_width = float(LANE_WIDTH_METERS.get(lane_width_class, 3.5))
    topology_sample = _enrich_topology_sample(topology_sample, fallback_lane, lane_width)

    projected_entities = []
    for entity in _coerce_list(raw_coordinates.get("entities")):
        lane = _select_semantic_lane(entity, topology_sample, fallback_lane)
        category = str(entity.get("category") or "")
        if category == "pedestrian":
            projected = _project_pedestrian_entity(entity, lane, lane_width)
        elif category in STATIC_CATEGORIES:
            projected = _project_static_entity(entity, lane, lane_width)
        else:
            projected = _project_vehicle_entity(entity, lane, lane_width, lane_context, fallback_lane)
        projected["projected_lane"] = {
            "road_id": lane.get("road_id"),
            "lane_id": lane.get("lane_id"),
            "yaw": float(_coerce_dict(lane.get("start")).get("yaw", 0.0) or 0.0),
            "reversed_to_anchor": bool(
                fallback_lane and _lane_is_reversed(lane, fallback_lane)
            ),
            "_synthetic": bool(lane.get("_synthetic")),
        }
        projected_entities.append(projected)

    projected_entities = _resolve_collisions(projected_entities)

    # Post-processing: snap z/yaw (and x/y for driving-lane entities) to the
    # nearest dense waypoint so that road surface height and curve tangent
    # direction are accurate.
    if dense_local_waypoints:
        dense_index = _build_dense_waypoint_index(dense_local_waypoints)
        projected_entities = [
            _snap_entity_to_dense_waypoints(e, dense_index)
            for e in projected_entities
        ]

    return {
        "selected_anchor_lane": raw_coordinates.get("selected_anchor_lane", fallback_lane),
        "entities": projected_entities,
        "metadata": {
            **_coerce_dict(raw_coordinates.get("metadata")),
            "projection_version": "carla-context-projection-v2",
            "coordinate_stage": "projected",
            "dense_waypoints_used": bool(dense_local_waypoints),
        },
    }


def _refine_pairwise_relation(
    entities_by_id: Dict[str, Dict[str, Any]],
    pair: Dict[str, Any],
    anchor_lane: Dict[str, Any],
    lane_context: Dict[str, Any],
) -> None:
    entity = entities_by_id.get(str(pair.get("entity_id")))
    other = entities_by_id.get(str(pair.get("other_entity_id")))
    if entity is None or other is None:
        return

    strength = str(pair.get("constraint_strength") or "soft")
    longitudinal_a, lateral_a = _relative_metrics(entity, anchor_lane)
    longitudinal_b, lateral_b = _relative_metrics(other, anchor_lane)
    expected_longitudinal = str(pair.get("longitudinal_relation") or "aligned_with_other")
    expected_lateral = str(pair.get("lateral_relation") or "same_lateral_band")
    target_gap = _distance_for_gap_band(pair.get("longitudinal_gap_band") or "near")

    if expected_longitudinal == "ahead_of_other":
        current_gap = longitudinal_a - longitudinal_b
        if current_gap < target_gap:
            mover = entity if _solve_priority(entity) > _solve_priority(other) else other
            delta = target_gap - current_gap + 0.1
            _shift_entity_longitudinal(mover, anchor_lane, delta if mover is entity else -delta)
    elif expected_longitudinal == "behind_other":
        current_gap = longitudinal_b - longitudinal_a
        if current_gap < target_gap:
            mover = entity if _solve_priority(entity) > _solve_priority(other) else other
            delta = target_gap - current_gap + 0.1
            _shift_entity_longitudinal(mover, anchor_lane, -delta if mover is entity else delta)

    if strength == "validation_only" or expected_lateral == "same_lateral_band":
        return

    entity_ok, entity_role = _lane_legality_for_entity(entity, lane_context)
    other_ok, other_role = _lane_legality_for_entity(other, lane_context)
    if not entity_ok or not other_ok:
        return
    if _is_vehicle_like(str(entity.get("category") or "")) and _is_vehicle_like(str(other.get("category") or "")):
        return

    mover = entity if _solve_priority(entity) > _solve_priority(other) else other
    mover_role = entity_role if mover is entity else other_role
    if mover_role not in {"sidewalk", "crosswalk", "parking", "parking_or_edge"}:
        return

    _, mover_lateral = _relative_metrics(mover, anchor_lane)
    _, other_lateral = _relative_metrics(other if mover is entity else entity, anchor_lane)
    desired_sign = -1.0 if expected_lateral == "left_of_other" and mover is entity else 1.0
    if expected_lateral == "right_of_other":
        desired_sign = 1.0 if mover is entity else -1.0
    current_delta = mover_lateral - other_lateral
    if desired_sign < 0.0 and current_delta < -1.0:
        return
    if desired_sign > 0.0 and current_delta > 1.0:
        return
    start = _coerce_dict(anchor_lane.get("start"))
    end = _coerce_dict(anchor_lane.get("end"))
    forward_x, forward_y = _normalize_vector(
        float(end.get("x", 0.0)) - float(start.get("x", 0.0)),
        float(end.get("y", 0.0)) - float(start.get("y", 0.0)),
    )
    right_x, right_y = -forward_y, forward_x
    mover["location"]["x"] += right_x * desired_sign * 0.5
    mover["location"]["y"] += right_y * desired_sign * 0.5


def refine_projected_coordinates_with_pairwise_relations(
    projected_coordinates: Dict[str, Any],
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    refined = _deep_copy(projected_coordinates)
    entities = _coerce_list(refined.get("entities"))
    entities_by_id = _entities_by_id(entities)
    anchor_lane = _coerce_dict(
        refined.get("selected_anchor_lane") or relation_dsl.get("selected_anchor_lane")
    )
    lane_context = _coerce_dict(relation_dsl.get("lane_context"))
    pairwise_relations = sorted(
        _coerce_list(relation_dsl.get("pairwise_relations")),
        key=lambda pair: {
            "hard": 0,
            "soft": 1,
            "validation_only": 2,
        }.get(str(pair.get("constraint_strength") or "soft"), 1),
    )
    for pair in pairwise_relations:
        _refine_pairwise_relation(entities_by_id, pair, anchor_lane, lane_context)
    refined["entities"] = _resolve_collisions(entities)
    refined["metadata"] = {
        **_coerce_dict(refined.get("metadata")),
        "coordinate_stage": "refined",
    }
    return refined


def validate_relation_layout(
    relation_dsl: Dict[str, Any],
    projected_coordinates: Dict[str, Any],
) -> Dict[str, Any]:
    entities = _coerce_list(projected_coordinates.get("entities"))
    entities_by_id = _entities_by_id(entities)
    anchor_lane = _coerce_dict(
        projected_coordinates.get("selected_anchor_lane")
        or relation_dsl.get("selected_anchor_lane")
    )
    lane_context = _coerce_dict(relation_dsl.get("lane_context"))
    lane_width_class = str(relation_dsl.get("lane_width_class") or "standard")
    lane_width_m = float(LANE_WIDTH_METERS.get(lane_width_class, LANE_WIDTH_METERS["standard"]))

    ego_results = []
    lane_results = []
    unresolved_conflicts = []

    for ego_relation in _coerce_list(relation_dsl.get("ego_relations") or relation_dsl.get("entities")):
        entity = entities_by_id.get(str(ego_relation.get("entity_id")))
        if entity is None:
            unresolved_conflicts.append({"type": "missing_entity", "entity_id": ego_relation.get("entity_id")})
            continue
        longitudinal, lateral = _relative_metrics(entity, anchor_lane)
        expected_longitudinal = _expected_longitudinal_scalar(ego_relation)
        expected_lateral_band = _expected_lateral_band(ego_relation)
        expected_lateral_m = expected_lateral_band * lane_width_m
        lateral_error_m = lateral - expected_lateral_m
        longitudinal_ok = abs(longitudinal - expected_longitudinal) <= max(
            2.5,
            float(ego_relation.get("group_spacing_m") or 0.0) + 1.0,
        )
        lateral_ok = abs(lateral_error_m) <= max(1.0, lane_width_m * 0.8)
        ego_results.append(
            {
                "entity_id": ego_relation.get("entity_id"),
                "status": "pass" if longitudinal_ok and lateral_ok else "fail",
                "lane_side_relation": ego_relation.get("lane_side_relation"),
                "lane_index_relation": ego_relation.get("lane_index_relation"),
                "longitudinal_check": longitudinal_ok,
                "lateral_check": lateral_ok,
                "actual_longitudinal_m": round(longitudinal, 3),
                "expected_longitudinal_m": round(expected_longitudinal, 3),
                "actual_lateral_m": round(lateral, 3),
                "expected_lateral_band": round(expected_lateral_band, 3),
                "expected_lateral_m": round(expected_lateral_m, 3),
                "lateral_error_m": round(lateral_error_m, 3),
            }
        )
        lane_ok, role = _lane_legality_for_entity(entity, lane_context)
        lane_results.append(
            {
                "entity_id": ego_relation.get("entity_id"),
                "status": "pass" if lane_ok else "fail",
                "expected_role": role,
                "projected_lane": entity.get("projected_lane"),
            }
        )

    pairwise_validation = validate_pairwise_relations(relation_dsl, projected_coordinates)
    unresolved_conflicts.extend(item for item in ego_results if item["status"] == "fail")
    unresolved_conflicts.extend(item for item in lane_results if item["status"] == "fail")
    unresolved_conflicts.extend(
        item for item in pairwise_validation["pairwise_results"] if item.get("status") == "fail"
    )
    return {
        "summary": {
            "ego_consistency": {
                "total": len(ego_results),
                "failed": sum(1 for item in ego_results if item["status"] == "fail"),
            },
            "pairwise_consistency": pairwise_validation["summary"],
            "lane_legality": {
                "total": len(lane_results),
                "failed": sum(1 for item in lane_results if item["status"] == "fail"),
            },
        },
        "entity_results": ego_results,
        "pairwise_results": pairwise_validation["pairwise_results"],
        "lane_results": lane_results,
        "unresolved_conflicts": unresolved_conflicts,
        "lane_context": lane_context,
        "lane_width_m": lane_width_m,
        "metadata": {
            "validation_version": "relation-layout-validation-v1",
            "lane_width_class": lane_width_class,
        },
    }


def color_for_entity(entity_id: str, category: str) -> str:
    seed = sum(ord(ch) for ch in f"{entity_id}:{category}")
    red = 40 + (seed * 37) % 180
    green = 40 + (seed * 59) % 180
    blue = 40 + (seed * 83) % 180
    return f"{red},{green},{blue}"


def preferred_vehicle_color(entity: Dict[str, Any]) -> Optional[str]:
    appearance = _coerce_dict(entity.get("appearance"))
    rgb_value = appearance.get("color_rgb")
    if isinstance(rgb_value, str) and rgb_value.strip():
        return rgb_value.strip()
    color_name = _canonical_color_name(appearance.get("color"))
    if color_name:
        return COLOR_RGB_MAP.get(color_name)
    raw_color = entity.get("color")
    if isinstance(raw_color, str) and raw_color.strip():
        return raw_color.strip()
    return None


def _anchor_frame_from_coordinates(
    projected_coordinates: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float, float]]:
    anchor_lane = _coerce_dict(projected_coordinates.get("selected_anchor_lane"))
    start = _coerce_dict(anchor_lane.get("start"))
    end = _coerce_dict(anchor_lane.get("end"))
    if not start or not end:
        return None
    sx = float(start.get("x", 0.0) or 0.0)
    sy = float(start.get("y", 0.0) or 0.0)
    ex = float(end.get("x", sx) or sx)
    ey = float(end.get("y", sy) or sy)
    fx, fy = _normalize_vector(ex - sx, ey - sy)
    yaw = math.degrees(math.atan2(fy, fx))
    return fx, fy, -fy, fx, yaw


def _apply_curb_row_alignment(projected_coordinates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep representative curbside rows parallel to the road edge."""
    updated = _deep_copy(projected_coordinates)
    frame = _anchor_frame_from_coordinates(updated)
    if frame is None:
        return updated
    forward_x, forward_y, right_x, right_y, anchor_yaw = frame
    entities = _coerce_list(updated.get("entities"))
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for entity in entities:
        if not _is_vehicle_like(str(entity.get("category") or "")):
            continue
        lane_index = int(entity.get("lane_index_relation", 0) or 0)
        if lane_index == 0:
            continue
        group_id = str(entity.get("group_id") or entity.get("source_id") or entity.get("id") or "")
        group_key = group_id.rsplit("_", 1)[0] if "_" in group_id else group_id
        groups.setdefault((group_key, str(lane_index)), []).append(entity)

    for (_group_key, _lane_index), row_entities in groups.items():
        if len(row_entities) < 2:
            continue
        row_entities.sort(
            key=lambda item: (
                float(_coerce_dict(item.get("location")).get("x", 0.0)) * forward_x
                + float(_coerce_dict(item.get("location")).get("y", 0.0)) * forward_y
            )
        )
        first_location = _coerce_dict(row_entities[0].get("location"))
        base_x = float(first_location.get("x", 0.0) or 0.0)
        base_y = float(first_location.get("y", 0.0) or 0.0)
        lateral_values = [
            float(_coerce_dict(item.get("location")).get("x", 0.0)) * right_x
            + float(_coerce_dict(item.get("location")).get("y", 0.0)) * right_y
            for item in row_entities
        ]
        lateral_center = sum(lateral_values) / len(lateral_values)
        base_longitudinal = base_x * forward_x + base_y * forward_y
        base_lateral = (
            base_x * right_x + base_y * right_y
            if abs(lateral_center) <= 1e-6
            else lateral_center
        )
        for index, entity in enumerate(row_entities):
            sub = _slugify(str(entity.get("subtype") or ""))
            row_spacing = (
                2.5 if sub in {"motorcycle", "motor_scooter", "scooter", "motorbike", "two_wheeler"}
                else 1.5 if sub in {"bicycle", "bike"}
                else 5.5
            )
            longitudinal = base_longitudinal + index * row_spacing
            location = entity.setdefault("location", {})
            location["x"] = forward_x * longitudinal + right_x * base_lateral
            location["y"] = forward_y * longitudinal + right_y * base_lateral
            rotation = entity.setdefault("rotation", {})
            rotation["pitch"] = float(rotation.get("pitch", 0.0) or 0.0)
            projected_lane = _coerce_dict(entity.get("projected_lane"))
            road_yaw = float(projected_lane.get("yaw", anchor_yaw) or anchor_yaw)
            rotation["yaw"] = _yaw_for_heading_relation(
                road_yaw,
                entity.get("heading_relation"),
                lane_reversed=bool(projected_lane.get("reversed_to_anchor")),
                turn_intent=entity.get("turn_intent"),
            )
            rotation["roll"] = float(rotation.get("roll", 0.0) or 0.0)
    return updated


# ---------------------------------------------------------------------------
# Ego lane indexing helpers
# ---------------------------------------------------------------------------

_GORE_KEYWORDS = ("gore", "shoulder", "merge", "ramp", "diverge", "buffer")


def _scene_has_center_median(scene_understanding: Dict[str, Any]) -> bool:
    """True when a reliable left boundary (center median / divider) is present.

    A center median makes the left side of ego's carriageway a hard boundary,
    so the left-neighbour count is the trustworthy anchor for ego's lane.
    """
    road_network = _coerce_dict(scene_understanding.get("road_network"))
    directionality = str(road_network.get("directionality") or "").lower()
    if any(
        token in directionality
        for token in (
            "not_median",
            "not median",
            "no_median",
            "no median",
            "without_median",
            "without median",
            "lane_markings_not_median",
            "markings_not_median",
            "painted_only",
        )
    ):
        return False
    if "divided" in directionality:
        return True
    for area in _coerce_list(road_network.get("special_road_areas")):
        if not isinstance(area, dict):
            continue
        if area.get("presence") is False or area.get("visible") is False:
            continue
        area_type = str(area.get("type") or "").lower()
        if any(token in area_type for token in ("median", "divider", "barrier")):
            return True
    cues = _coerce_dict(_coerce_dict(scene_understanding.get("metadata")).get("decisive_map_matching_cues"))
    return cues.get("has_center_median") is True


def _scene_forward_lane_count(scene_understanding: Dict[str, Any]) -> int:
    lane_groups = _coerce_list(
        _coerce_dict(scene_understanding.get("road_network")).get("lane_groups")
    )
    if not lane_groups:
        return 0
    try:
        return int(_coerce_dict(lane_groups[0]).get("forward_lane_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _is_gore_subject(entity: Dict[str, Any]) -> bool:
    """True for subjects in gore/shoulder/merge/ramp areas that report phantom outer lanes."""
    text = " ".join(
        str(entity.get(key) or "") for key in ("lane_side_relation", "evidence")
    ).lower()
    return any(keyword in text for keyword in _GORE_KEYWORDS)


def _infer_ego_lane_offset(
    scene_understanding: Dict[str, Any],
    forward_lane_count: Optional[int] = None,
) -> int:
    """Return how many same-direction driving lanes are to ego's right.

    In CARLA right-hand traffic lane_id=-1 is the innermost forward lane (next
    to the centre line); magnitude grows toward the curb, so the rightmost
    driving lane has the LARGEST |lane_id| (-2, -3, …).  This value is fed to
    ``_find_lane_in_dense`` where offset 0 selects the rightmost (largest |id|)
    lane and offset N-1 the leftmost (|id|=1) forward lane.

    Rather than blindly taking the maximum positive ``lane_index_relation``
    (which a single gore/merge-area actor can inflate, pushing ego an entire
    lane off — see results/auto_result_20260617_220440), the offset is
    reconciled with the forward lane count and the left/right neighbour counts:

    * Only same-direction subjects count — opposing traffic sits across the
      median and must never inflate the lane span.
    * Subjects in gore / shoulder / merge / ramp areas (or longitudinally
      "alongside") are dropped — they routinely report phantom outer lanes.
    * When a center median bounds ego on the left, the left-neighbour count L
      is the reliable anchor: ``offset = (N - 1) - L``. Otherwise the
      right-neighbour count R is used. Both are clamped into ``[0, N-1]``.
    * When the forward lane count is unknown, fall back to the legacy
      max-right behaviour (clamped only at 0).
    """
    if forward_lane_count is None:
        forward_lane_count = _scene_forward_lane_count(scene_understanding)
    n = int(forward_lane_count or 0)

    max_left = 0
    max_right = 0
    for entity in _coerce_list(scene_understanding.get("traffic_subjects")):
        if not isinstance(entity, dict):
            continue
        heading = str(entity.get("heading_relation_to_ego") or "").lower()
        if heading and heading != "same_direction":
            continue
        if entity.get("longitudinal_relation") == "alongside":
            continue
        if _is_gore_subject(entity):
            continue
        try:
            idx = int(entity.get("lane_index_relation"))
        except (TypeError, ValueError):
            continue
        if idx > max_right:
            max_right = idx
        if -idx > max_left:
            max_left = -idx

    if n <= 0:
        # No reliable lane count — preserve legacy behaviour.
        heuristic = max(0, max_right)
    else:
        offset = (
            (n - 1) - max_left
            if _scene_has_center_median(scene_understanding)
            else max_right
        )
        heuristic = max(0, min(n - 1, offset))

    # Prefer an explicit VLM/map ego-lane estimate when present. The
    # same-direction-only heuristic is still useful as a fallback, but in sparse
    # junction images there may be no same-direction actors to support the ego
    # lane, and forcing min(explicit, heuristic) collapses ego back to the
    # rightmost lane even when the scene understanding says otherwise.
    explicit = _coerce_dict(
        _coerce_dict(scene_understanding.get("metadata")).get("ego_localization")
    ).get("ego_lane_from_right")
    if explicit is None:
        explicit = _coerce_dict(
            _coerce_dict(scene_understanding.get("road_network")).get("map_matching")
        ).get("ego_lane_from_right")
    if explicit is not None:
        try:
            explicit_offset = int(explicit)
        except (TypeError, ValueError):
            explicit_offset = None
        if explicit_offset is not None and explicit_offset >= 0:
            if n > 0:
                explicit_offset = max(0, min(n - 1, explicit_offset))
            return explicit_offset

    return heuristic


def _find_lane_in_dense(
    candidate_lane: Dict[str, Any],
    lane_offset: int,
    dense_wps: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the waypoint for ego's target lane within dense_local_waypoints.

    Filters to the same carriageway (same lane_id sign as the anchor) on
    candidate_lane's road, de-duplicates by lane_id keeping the waypoint closest
    to candidate_lane.start, then selects the lane at position lane_offset from
    the rightmost — the lane with the LARGEST |lane_id| (curb side), not -1.

    Returns a waypoint dict {x, y, z, yaw, road_id, lane_id, ...} or None.
    """
    road_id = candidate_lane.get("road_id")
    start = candidate_lane.get("start") or {}
    cx = float(start.get("x", 0.0))
    cy = float(start.get("y", 0.0))

    # Ego must stay on the SAME carriageway the matcher chose as the anchor.
    # CARLA lane_id sign encodes driving direction, but which sign is the
    # driving direction depends on the road's reference-line orientation: it is
    # NOT always negative. Filtering on a hard-coded ``lane_id < 0`` flips ego
    # 180° (and onto the opposing carriageway) whenever the matcher anchored on
    # a positive-id lane — the junction then ends up behind ego. Derive the
    # driving-direction sign from the anchor lane instead.
    try:
        anchor_lane_id = int(candidate_lane.get("lane_id"))
    except (TypeError, ValueError):
        anchor_lane_id = -1  # legacy default: driving lanes are negative ids
    anchor_sign = 1 if anchor_lane_id > 0 else -1

    # Keep only same-direction (same lane_id sign as the anchor) lanes on this
    # road. dense_local_waypoints is BFS-sampled over Driving lanes only, so no
    # shoulder/parking lane can slip in here.
    same_road = [
        wp for wp in dense_wps
        if wp.get("road_id") == road_id and isinstance(wp.get("lane_id"), int)
        and wp["lane_id"] != 0
        and (1 if wp["lane_id"] > 0 else -1) == anchor_sign
    ]
    if not same_road:
        return None

    # De-duplicate: for each lane_id keep the closest waypoint to current start.
    best_by_lane: Dict[int, Dict[str, Any]] = {}
    for wp in same_road:
        lid = wp["lane_id"]
        dx = wp.get("x", 0.0) - cx
        dy = wp.get("y", 0.0) - cy
        dist = math.hypot(dx, dy)
        if lid not in best_by_lane or dist < math.hypot(
            best_by_lane[lid].get("x", 0.0) - cx,
            best_by_lane[lid].get("y", 0.0) - cy,
        ):
            best_by_lane[lid] = wp

    # Sort rightmost-first. In OpenDRIVE/CARLA lane_id magnitude grows OUTWARD
    # from the reference line: |id|=1 is the innermost lane (next to the centre /
    # opposing traffic) and the largest |id| is the curb/rightmost lane. This
    # holds for both carriageway signs (negatives: -3 > -2 > -1 toward the curb;
    # positives: 3 > 2 > 1), so descending |lane_id| puts the rightmost lane at
    # offset 0. (Sorting by raw -lane_id instead would put lane -1 — the LEFT
    # lane — at offset 0 on a negative carriageway.)
    sorted_lanes = sorted(best_by_lane.values(), key=lambda w: -abs(w["lane_id"]))

    if lane_offset >= len(sorted_lanes):
        return None
    return sorted_lanes[lane_offset]


def _select_sibling_lane_cache_only(
    candidate_lane: Dict[str, Any],
    lane_offset: int,
) -> Optional[Dict[str, Any]]:
    """Cache-only analogue of ``_find_lane_in_dense``.

    Uses the sibling-lane geometry baked into
    ``candidate_lane['same_direction_lane_evidence']['inspected_lanes']`` (each
    accepted same-direction Driving lane carries a ``start`` {x, y, z, yaw}) to
    pick the lane at position ``lane_offset`` from the rightmost (offset 0),
    matching ``_find_lane_in_dense``'s descending-|lane_id| ordering.

    Returns ``{road_id, lane_id, start{...}}`` for the target lane, or ``None``
    when the geometry is unavailable (older cache without baked geometry), only
    the anchor lane exists, or the offset is out of range.
    """
    evidence = candidate_lane.get("same_direction_lane_evidence") or {}
    inspected = evidence.get("inspected_lanes") or []

    lanes: Dict[int, Dict[str, Any]] = {}
    for entry in inspected:
        if not entry.get("accepted"):
            continue
        lane_id = entry.get("lane_id")
        start = entry.get("start")
        if not isinstance(lane_id, int) or not isinstance(start, dict):
            continue
        lanes[lane_id] = {
            "road_id": entry.get("road_id", candidate_lane.get("road_id")),
            "lane_id": lane_id,
            "start": start,
        }

    # The anchor lane carries no baked geometry (its start IS the candidate
    # start); seed it so it participates in the rightmost-first ordering.
    try:
        anchor_lane_id = int(candidate_lane.get("lane_id"))
    except (TypeError, ValueError):
        return None
    anchor_start = candidate_lane.get("start")
    if isinstance(anchor_start, dict):
        lanes.setdefault(
            anchor_lane_id,
            {
                "road_id": candidate_lane.get("road_id"),
                "lane_id": anchor_lane_id,
                "start": anchor_start,
            },
        )

    if len(lanes) <= 1:
        return None

    # Rightmost-first: descending |lane_id|. Magnitude grows outward from the
    # reference line, so the largest |id| is the curb/rightmost lane for both
    # signs (negatives: -3 > -2 > -1; positives: 3 > 2 > 1). offset 0 is the
    # rightmost driving lane. (Raw -lane_id would mis-rank lane -1 — the LEFT
    # lane — as rightmost on a negative carriageway.)
    ordered = sorted(lanes.values(), key=lambda lane: -abs(lane["lane_id"]))
    if lane_offset < 0 or lane_offset >= len(ordered):
        return None
    return ordered[lane_offset]


def _next_wp_along_lane(
    wp: Dict[str, Any],
    dense_wps: List[Dict[str, Any]],
    lookahead_m: float = 20.0,
) -> Dict[str, Any]:
    """Return the dense waypoint ~lookahead_m ahead of wp on the same lane.

    Falls back to wp itself when no suitable candidate is found.
    """
    road_id = wp.get("road_id")
    lane_id = wp.get("lane_id")
    wx, wy = float(wp.get("x", 0.0)), float(wp.get("y", 0.0))

    candidates = [
        w for w in dense_wps
        if w.get("road_id") == road_id and w.get("lane_id") == lane_id
    ]
    if not candidates:
        return wp

    # Forward direction from wp's yaw.
    yaw_rad = math.radians(float(wp.get("yaw", 0.0)))
    fw_x, fw_y = math.cos(yaw_rad), math.sin(yaw_rad)

    best = None
    best_diff = float("inf")
    for cand in candidates:
        dx = float(cand.get("x", 0.0)) - wx
        dy = float(cand.get("y", 0.0)) - wy
        longitudinal = dx * fw_x + dy * fw_y
        if longitudinal <= 0:
            continue
        diff = abs(longitudinal - lookahead_m)
        if diff < best_diff:
            best_diff = diff
            best = cand
    return best if best is not None else wp


# A forward reference is only usable for geometric ego placement when the matcher
# can resolve its position on the map. Today that means the junction ahead (and
# the signal / stop line co-located with it). Other landmarks are advisory only.
_MAPPABLE_FORWARD_REFERENCES = (
    "junction", "intersection", "traffic_light", "signal", "stop_line", "crosswalk",
)
# Keep ego at least this far from the junction stop line when sliding forward.
MIN_JUNCTION_CLEARANCE_M = 6.0
# Ignore sub-threshold slides (placement noise) and cap absurd ones.
MIN_EGO_SLIDE_M = 5.0
MAX_EGO_SLIDE_M = 70.0


def _ego_forward_reference_distance(
    metadata: Dict[str, Any],
    map_matching: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Return a VLM-estimated ego→forward-reference distance in metres, or None.

    Reads the optional ``metadata.ego_localization`` block. Accepts either a
    direct ``ego_to_junction_distance_m`` or a ``forward_reference`` of a
    map-anchorable type carrying ``distance_m``. Non-mappable landmarks (e.g. a
    storefront) are deliberately ignored: the matcher has no coordinate for them,
    so their distance cannot position ego.
    """
    ego_loc = _coerce_dict(metadata.get("ego_localization"))

    def _coerce_distance(value: Any) -> Optional[float]:
        try:
            dist = float(value)
        except (TypeError, ValueError):
            return None
        if dist < 0:
            return None
        return dist

    direct = _coerce_distance(ego_loc.get("ego_to_junction_distance_m"))
    if direct is not None:
        return direct

    direct = _coerce_distance(_coerce_dict(map_matching).get("ego_to_junction_distance_m"))
    if direct is not None:
        return direct

    ref = _coerce_dict(ego_loc.get("forward_reference"))
    ref_type = str(ref.get("type") or "").lower()
    if any(tok in ref_type for tok in _MAPPABLE_FORWARD_REFERENCES):
        return _coerce_distance(ref.get("distance_m"))
    return None


def _normalize_vector_2d(dx: float, dy: float) -> Tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return 0.0, 0.0
    return dx / norm, dy / norm


def compute_cache_longitudinal_slide(
    distance_to_junction_ahead: Optional[float],
    target_m: float,
    *,
    min_clearance: float = MIN_JUNCTION_CLEARANCE_M,
    min_abs: float = MIN_EGO_SLIDE_M,
    max_abs: float = MAX_EGO_SLIDE_M,
) -> float:
    """Metres to slide ego forward (toward the junction) in cache-only mode.

    Positive slides move ego forward (closer to the junction). Returns 0.0 when
    no candidate junction distance is known, when the adjustment is below
    ``min_abs`` (noise), or when sliding would overshoot the stop line.
    """
    if not isinstance(distance_to_junction_ahead, (int, float)):
        return 0.0
    actual = float(distance_to_junction_ahead)
    if actual < 0:
        return 0.0
    slide = actual - float(target_m)
    # Never push ego into or past the junction.
    upper = max(0.0, actual - min_clearance)
    slide = max(-max_abs, min(slide, upper, max_abs))
    if abs(slide) < min_abs:
        return 0.0
    return slide


def slide_anchor_along_segment(
    start: Dict[str, Any], end: Dict[str, Any], slide_m: float
) -> Dict[str, Any]:
    """Translate ``start`` by ``slide_m`` along the start→end forward direction.

    CARLA-free linear extrapolation used when dense waypoints are unavailable.
    Returns a new start dict; yaw / is_junction are preserved from ``start``.
    """
    sx, sy = float(start.get("x", 0.0)), float(start.get("y", 0.0))
    ex, ey = float(end.get("x", sx)), float(end.get("y", sy))
    fx, fy = _normalize_vector_2d(ex - sx, ey - sy)
    return {
        "x": sx + fx * slide_m,
        "y": sy + fy * slide_m,
        "z": float(start.get("z", 0.0)),
        "yaw": float(start.get("yaw", 0.0)),
        "is_junction": bool(start.get("is_junction", False)),
    }


def _infer_ego_junction_target(
    scene_understanding: Dict[str, Any],
) -> Tuple[bool, float]:
    """Infer whether to constrain ego's distance to the nearest forward junction.

    Returns (constrain, target_m):
      - (False, 0.0) – no visible junction; skip longitudinal adjustment.
      - (True, target_m) – ego should be ~target_m ahead of the junction.

    Distance mapping (qualitative location field → metres):
      immediate / at / entering  →  8 m
      close / near / approaching → 18 m
      (no qualifier)             → 28 m
    """
    road_network = scene_understanding.get("road_network") or {}
    metadata = scene_understanding.get("metadata") or {}
    cues = metadata.get("decisive_map_matching_cues") or {}
    map_matching = _coerce_dict(road_network.get("map_matching"))

    map_topology = str(map_matching.get("topology_type") or "").lower()
    map_junction_visible = map_matching.get("junction_visible")
    map_matching_says_no_junction = (
        map_junction_visible is False
        or map_topology in {"straight_road", "straight_two_way", "curve"}
    )

    # If the direct map-matching contract says this is not a junction target,
    # ignore any stray ego_to_junction_distance_m emitted by the VLM.
    if map_matching_says_no_junction:
        return False, 0.0

    # Highest priority: an explicit VLM metric estimate of ego's distance to the
    # forward reference. Only references the matcher can geometrically anchor to
    # (the junction / its stop line / signal) can drive the longitudinal slide;
    # arbitrary landmarks have no CARLA coordinate and are ignored here.
    target_m = _ego_forward_reference_distance(metadata, map_matching)
    if target_m is not None:
        return True, target_m

    # Explicit override from decisive cues.
    junction_visible_cue = cues.get("junction_visible")
    if isinstance(junction_visible_cue, bool) and not junction_visible_cue:
        return False, 0.0

    _NEGATIVE = ("far", "distant", "not visible", "not_visible", "background", "offscreen")
    _IMMEDIATE = ("immediate", "at junction", "at_junction", "entering", "in junction")
    _CLOSE     = ("close", "near", "approaching", "just ahead", "right ahead")

    for junction in (road_network.get("junctions") or []):
        if not isinstance(junction, dict):
            continue

        # Skip low-confidence or far junctions.
        confidence = junction.get("confidence")
        if str(confidence).lower() == "low":
            continue
        try:
            if float(confidence) < 0.5:
                continue
        except (TypeError, ValueError):
            pass

        location = str(junction.get("location") or junction.get("position") or "").lower()
        if any(tok in location for tok in _NEGATIVE):
            continue

        # Junction is visible – determine distance from location hint.
        if any(tok in location for tok in _IMMEDIATE):
            return True, 8.0
        if any(tok in location for tok in _CLOSE):
            return True, 18.0
        return True, 28.0

    # Fall back to explicit decisive cue if no junction objects provided distance.
    if junction_visible_cue is True:
        return True, 28.0

    return False, 0.0


def _measure_forward_junction_distance(
    start: Dict[str, Any],
    end: Dict[str, Any],
    dense_wps: List[Dict[str, Any]],
) -> Optional[float]:
    """Measure the longitudinal distance from start to the nearest forward junction.

    Uses the forward direction implied by start → end.
    Returns None when no junction waypoints are found ahead of start.
    """
    sx, sy = float(start.get("x", 0.0)), float(start.get("y", 0.0))
    ex, ey = float(end.get("x", sx)), float(end.get("y", sy))
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        yaw_rad = math.radians(float(start.get("yaw", 0.0)))
        fw_x, fw_y = math.cos(yaw_rad), math.sin(yaw_rad)
    else:
        fw_x, fw_y = dx / length, dy / length

    min_dist: Optional[float] = None
    for wp in dense_wps:
        if not wp.get("is_junction"):
            continue
        long = (float(wp.get("x", 0.0)) - sx) * fw_x + (float(wp.get("y", 0.0)) - sy) * fw_y
        if long <= 0:
            continue
        if min_dist is None or long < min_dist:
            min_dist = long
    return min_dist


def _slide_along_dense_waypoints(
    start: Dict[str, Any],
    end: Dict[str, Any],
    slide_m: float,
    dense_wps: List[Dict[str, Any]],
    candidate_lane: Dict[str, Any],
    min_junction_clearance_m: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Return a new start-point after sliding ego by slide_m along the lane.

    slide_m > 0  →  move ego forward (closer to junction).
    slide_m < 0  →  move ego backward (farther from junction).

    Filters to the same road_id + lane_id as candidate_lane, computes the
    signed longitudinal displacement of each waypoint from start, picks the
    one closest to slide_m, and enforces a minimum clearance from any forward
    junction.

    Returns a waypoint dict or None when no suitable point is found.
    """
    road_id = candidate_lane.get("road_id")
    lane_id = candidate_lane.get("lane_id")

    sx, sy = float(start.get("x", 0.0)), float(start.get("y", 0.0))
    ex, ey = float(end.get("x", sx)), float(end.get("y", sy))
    ddx, ddy = ex - sx, ey - sy
    length = math.hypot(ddx, ddy)
    if length < 1e-6:
        yaw_rad = math.radians(float(start.get("yaw", 0.0)))
        fw_x, fw_y = math.cos(yaw_rad), math.sin(yaw_rad)
    else:
        fw_x, fw_y = ddx / length, ddy / length

    # Pre-compute forward junction distance from the *target* position for
    # clearance enforcement.
    junction_long: Optional[float] = _measure_forward_junction_distance(start, end, dense_wps)

    same_lane = [
        wp for wp in dense_wps
        if wp.get("road_id") == road_id and wp.get("lane_id") == lane_id
    ]
    if not same_lane:
        return None

    best: Optional[Dict[str, Any]] = None
    best_diff = float("inf")
    for wp in same_lane:
        long = (float(wp.get("x", 0.0)) - sx) * fw_x + (float(wp.get("y", 0.0)) - sy) * fw_y
        # Enforce minimum clearance from junction.
        if junction_long is not None:
            new_dist_to_junction = junction_long - long
            if new_dist_to_junction < min_junction_clearance_m:
                continue
        diff = abs(long - slide_m)
        if diff < best_diff:
            best_diff = diff
            best = wp
    return best


# ---------------------------------------------------------------------------

def _min_opposing_abs_lane_index(entities: List[Dict[str, Any]]) -> int:
    """Smallest |lane_index| among oncoming vehicles.

    The nearest oncoming vehicle sits in the opposing lane closest to the median,
    so its |lane_index| marks where the opposing carriageway begins in the VLM's
    ego-relative lane count. Used to map each oncoming vehicle to a 1-based
    opposing-lane ordinal counted from the median.
    """
    indices = [
        abs(int(entity.get("lane_index_relation", 0) or 0))
        for entity in entities
        if str(entity.get("heading_relation") or "") == "opposite_direction"
        and _is_vehicle_like(str(entity.get("category") or ""))
    ]
    return min(indices) if indices else 1


def _should_project_to_junction_lane(entity: Dict[str, Any]) -> bool:
    if str(entity.get("junction_placement") or "").lower() != "frame":
        return False
    projected_lane = _coerce_dict(entity.get("projected_lane"))
    if str(projected_lane.get("source") or "").lower() != "junction_leg":
        return False
    direction = str(entity.get("junction_direction") or "").lower()
    if direction in {"left", "right"}:
        return True
    anchor = str(entity.get("layout_anchor_id") or "").lower()
    if anchor in {"left_arm", "right_arm"}:
        return True
    leg = str(entity.get("junction_leg") or "").lower()
    return leg in {"left", "right"}


def build_projected_spawn_payload(projected_coordinates: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"entities": []}
    aligned_coordinates = _apply_curb_row_alignment(projected_coordinates)
    aligned_entities = _coerce_list(aligned_coordinates.get("entities"))
    min_opposing_abs = _min_opposing_abs_lane_index(aligned_entities)
    for entity in aligned_entities:
        spawn_kind = str(entity.get("spawn_kind") or "vehicle")
        category = str(entity.get("category") or "car")
        lane_side = str(entity.get("lane_side_relation") or "")
        lane_index = int(entity.get("lane_index_relation", 0) or 0)
        heading_relation = str(entity.get("heading_relation") or "unknown")
        opposing_lane_from_median = None
        placement_mode = "project_to_lane"
        if category in {"cone_group", "barrier_group"}:
            placement_mode = "direct"
        elif spawn_kind == "vehicle" and _should_project_to_junction_lane(entity):
            placement_mode = "project_to_junction_lane"
        elif spawn_kind == "vehicle" and heading_relation == "opposite_direction":
            # Cross-median seed lands somewhere on the opposing carriageway; runtime
            # then walks the lane graph to the median-adjacent opposing lane. The
            # nearest oncoming vehicle maps to lane 1 (closest to the median).
            placement_mode = "project_to_opposing_lane"
            opposing_lane_from_median = max(1, abs(lane_index) - min_opposing_abs + 1)
        elif spawn_kind == "vehicle" and lane_index != 0:
            placement_mode = "preserve_xy"
        explicit_color = preferred_vehicle_color(entity)
        payload_entity = {
            "id": entity["id"],
            "spawn_kind": spawn_kind,
            "blueprint_name": entity.get("blueprint_name"),
            "category": category,
            "lane_side_relation": lane_side,
            "lane_index_relation": lane_index,
            "heading_relation": str(entity.get("heading_relation") or "unknown"),
            "motion_state": str(entity.get("motion_state") or "unknown"),
            "projected_lane": _coerce_dict(entity.get("projected_lane")),
            "junction_direction": str(entity.get("junction_direction") or ""),
            "junction_leg": str(entity.get("junction_leg") or ""),
            "junction_motion": str(entity.get("junction_motion") or ""),
            "junction_distance_m": entity.get("junction_distance_m"),
            "location": entity["location"],
            "rotation": entity["rotation"],
            "color": None
            if spawn_kind != "vehicle"
            else (explicit_color or color_for_entity(entity["id"], category)),
            "placement_mode": placement_mode,
            "appearance": _coerce_dict(entity.get("appearance")),
        }
        if opposing_lane_from_median is not None:
            payload_entity["opposing_lane_from_median"] = opposing_lane_from_median
        payload["entities"].append(payload_entity)
    return payload
