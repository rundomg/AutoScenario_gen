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
    "background_traffic",
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

BACKGROUND_DENSITY_DEFAULTS = {
    "curbside_row": 3,
    "sidewalk_group": 2,
    "opposing_flow": 2,
    "sparse_filler": 1,
}

BACKGROUND_MAX_ACTORS = 6

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
    "parked_vehicle",
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
    "parked_two_wheeler_group": "parked_vehicle",
    "parked_vehicle_row": "parked_vehicle",
    "parked_car_row": "parked_vehicle",
    "opposing_vehicle": "car",
    "sidewalk_pedestrian_group": "pedestrian",
    "pedestrian_group": "pedestrian",
    "traffic_cones": "cone_group",
    "cones": "cone_group",
    "barriers": "barrier_group",
}

CANONICAL_LANE_SIDE_MAP = {
    "center_left": "left_lane",
    "left_of_center": "left_lane",
    "center_right": "right_lane",
    "right_of_center": "right_lane",
    "left_sidewalk": "sidewalk_left",
    "right_sidewalk": "sidewalk_right",
    "curbside_right": "right_edge",
    "curbside_left": "left_edge",
    "opposing_center_left": "left_lane",
    "opposing_left": "left_lane",
    "opposing_center_right": "right_lane",
    "opposing_right": "right_lane",
    "opposing_side": "left_lane",
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


def _canonical_lane_side(value: Any) -> str:
    normalized = _slugify(value)
    return CANONICAL_LANE_SIDE_MAP.get(normalized, normalized or "same_lane")


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
        "general_environment": {},
        "metadata": _coerce_dict(payload.get("metadata")),
        "key_pairwise_relations": _normalize_key_pairwise_relations(payload),
    }

    for index, entity in enumerate(_coerce_list(payload.get("traffic_subjects"))):
        if not isinstance(entity, dict):
            return None, f"traffic_subjects[{index}] must be an object."
        category = _canonical_category(entity.get("category"), entity.get("subtype"))
        normalized["traffic_subjects"].append(
            {
                "id": str(entity.get("id") or _normalize_entity_id("subject", index)),
                "category": category,
                "subtype": str(entity.get("subtype") or entity.get("category") or category or "unknown"),
                "visual_confidence": str(entity.get("visual_confidence") or "medium"),
                "motion_state": _canonical_motion_state(entity.get("motion_state")),
                "heading_relation_to_ego": _canonical_heading(entity.get("heading_relation_to_ego")),
                "lane_side_relation": _canonical_lane_side(entity.get("lane_side_relation")),
                "longitudinal_relation": str(entity.get("longitudinal_relation") or "ahead"),
                "longitudinal_proximity": _canonical_distance_band(entity.get("longitudinal_proximity")),
                "count": max(1, int(entity.get("count", 1))),
                "evidence": str(entity.get("evidence") or ""),
                "must_reconstruct": True,
                "appearance": _normalize_appearance(entity, category),
            }
        )

    for index, entity in enumerate(_coerce_list(payload.get("background_traffic"))):
        if not isinstance(entity, dict):
            return None, f"background_traffic[{index}] must be an object."
        category = _canonical_category(entity.get("category"), entity.get("subtype"))
        density_role = _canonical_density_role(entity.get("density_role"))
        default_count = BACKGROUND_DENSITY_DEFAULTS.get(density_role, 1)
        normalized["background_traffic"].append(
            {
                "id": str(entity.get("id") or _normalize_entity_id("background", index)),
                "category": category,
                "source": str(entity.get("source") or "inferred"),
                "representative_count": _clamp_int(
                    entity.get("representative_count", default_count),
                    1,
                    3,
                    default_count,
                ),
                "lane_side_relation": _canonical_lane_side(entity.get("lane_side_relation")),
                "longitudinal_band": _canonical_distance_band(entity.get("longitudinal_band")),
                "motion_bias": _canonical_motion_state(entity.get("motion_bias")),
                "heading_relation_to_ego": _canonical_heading(entity.get("heading_relation_to_ego")),
                "density_role": density_role,
                "confidence": str(entity.get("confidence") or "medium"),
                "spawn_priority": "low",
                "appearance": _normalize_appearance(entity, category),
            }
        )

    road_network = _coerce_dict(payload.get("road_network"))
    required_road_keys = (
        "road_type",
        "directionality",
        "road_segments",
        "lane_groups",
        "lane_markings",
        "special_road_areas",
        "junctions",
        "roadside_boundaries",
        "control_elements",
    )
    missing_road_keys = [key for key in required_road_keys if key not in road_network]
    if missing_road_keys:
        return None, f"road_network is missing required keys: {', '.join(missing_road_keys)}"

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
    if not normalized_lane_groups:
        normalized_lane_groups = [
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

    normalized["road_network"] = {
        "road_type": _canonical_road_type(road_network.get("road_type")),
        "directionality": str(road_network.get("directionality") or "two_way"),
        "road_segments": normalized_road_segments,
        "lane_groups": normalized_lane_groups,
        "lane_markings": _coerce_dict(road_network.get("lane_markings")),
        "special_road_areas": _coerce_list(road_network.get("special_road_areas")),
        "junctions": _coerce_list(road_network.get("junctions")),
        "roadside_boundaries": _coerce_dict(road_network.get("roadside_boundaries")),
        "control_elements": _coerce_list(road_network.get("control_elements")),
    }
    if not normalized["road_network"]["road_segments"]:
        return None, "road_network.road_segments must not be empty."
    if not normalized["road_network"]["lane_groups"]:
        return None, "road_network.lane_groups must not be empty."

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


def build_road_generation_request(scene_understanding: Dict[str, Any]) -> str:
    road_network = scene_understanding.get("road_network", {})
    general_environment = scene_understanding.get("general_environment", {})
    metadata = scene_understanding.get("metadata", {})
    return (
        "Generate an OpenDRIVE Road DSL from the structured scene understanding below.\n"
        "Use only the `road_network` section as geometry truth.\n"
        "Do not place traffic actors in the road DSL.\n"
        "Determine lane count from lane-first evidence: visible ground markings, lane separators, parking-lane separators, edge lines, and roadside boundaries.\n"
        "Treat `lane_markings`, `lane_groups`, and `roadside_boundaries` as the primary lane-count evidence.\n"
        "Do not reduce lane count just because vehicle placement looks sparse or ambiguous.\n"
        "When `left_parking_lane_count` or `right_parking_lane_count` is greater than zero, generate explicit `parking` lanes in the DSL rather than collapsing them into assumptions or `special_road_areas`.\n"
        "Treat `special_road_areas` parking hints as secondary support only; they must not override explicit lane counts.\n"
        "Within each side of a lane section, list lanes from the road center outward.\n"
        "Prefer the simplest valid OpenDRIVE geometry that preserves visible drivable space.\n"
        "Treat control elements and roadside boundaries as road-layout hints only.\n\n"
        f"Metadata:\n{json.dumps(metadata, indent=2, sort_keys=True)}\n\n"
        f"Road network:\n{json.dumps(road_network, indent=2, sort_keys=True)}\n\n"
        f"General environment context:\n{json.dumps(general_environment, indent=2, sort_keys=True)}"
    )


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


def _lane_anchor_for_relation(lane_side_relation: str, category: str) -> Tuple[str, int, str]:
    relation = str(lane_side_relation or "same_lane")
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
    if category in STATIC_CATEGORIES and relation == "left_edge":
        return "left_curbside", -1, "road_edge_left"
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


def _lane_context(scene_understanding: Dict[str, Any], anchor_lane: Dict[str, Any]) -> Dict[str, Any]:
    lane_groups = _coerce_list(scene_understanding.get("road_network", {}).get("lane_groups"))
    lane_group = lane_groups[0] if lane_groups else {}
    forward_lane_count = int(lane_group.get("forward_lane_count", 1))
    opposing_lane_count = int(lane_group.get("opposing_lane_count", 1))
    left_parking_lane_count = int(lane_group.get("left_parking_lane_count", 0) or 0)
    right_parking_lane_count = int(lane_group.get("right_parking_lane_count", 0) or 0)
    left_edge_role = "parking" if left_parking_lane_count > 0 else "parking_or_edge"
    right_edge_role = "parking" if right_parking_lane_count > 0 else "parking_or_edge"
    return {
        "ego_lane_id": int(anchor_lane.get("lane_id", -1)),
        "lane_catalog": {
            "same_lane": {"role": "driving", "relative_index": 0},
            "left_lane": {"role": "driving", "relative_index": -1},
            "right_lane": {"role": "driving", "relative_index": 1},
            "left_edge": {"role": left_edge_role, "relative_index": -2},
            "right_edge": {"role": right_edge_role, "relative_index": 2},
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
        },
        "lane_adjacency": {
            "same_lane": ["left_lane", "right_lane"],
            "left_lane": ["same_lane", "left_edge", "sidewalk_left"],
            "right_lane": ["same_lane", "right_edge", "sidewalk_right"],
            "left_edge": ["left_lane", "sidewalk_left"],
            "right_edge": ["right_lane", "sidewalk_right"],
        },
        "allowed_actor_types": {
            "driving": ["car", "truck", "bus", "motorcycle", "bicycle"],
            "parking": ["parked_vehicle", "cone_group", "barrier_group"],
            "parking_or_edge": ["parked_vehicle", "cone_group", "barrier_group"],
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
            expanded["_source_key"] = "background_traffic"
            expanded["_instance_index"] = instance_index
            expanded["_instance_total"] = count
            expanded["_priority"] = "background"
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
        if normalized_subtype in {"truck", "bus", "van", "suv", "motorcycle", "bike", "bicycle"}:
            if normalized_subtype == "bus":
                return "vehicle", "truck"
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

        category = str(expanded.get("category") or "car")
        subtype = str(expanded.get("subtype") or category)
        lane_anchor, lane_index_relation, lateral_mode = _lane_anchor_for_relation(
            expanded.get("lane_side_relation"),
            category,
        )
        distance_band = _distance_band_for_entity(expanded, source_key)
        order_relation = _order_relation_for_entity(expanded, source_key)
        rank_key = (distance_band, lane_anchor)
        distance_order_rank = rank_counters.get(rank_key, 0)
        rank_counters[rank_key] = distance_order_rank + 1
        spawn_kind, blueprint_name = _spawn_blueprint_for_category(category, subtype)

        spacing_m = 0.0
        if category == "parked_vehicle" or expanded.get("density_role") == "curbside_row":
            spacing_m = 6.0
        elif category == "pedestrian" and source_key == "background_traffic":
            spacing_m = 3.0
        elif category in STATIC_CATEGORIES:
            spacing_m = 1.2

        ego_relations.append(
            {
                "entity_id": expanded_id,
                "group_id": expanded.get("id", expanded_id),
                "source": "traffic_subject" if priority == "subject" else "background_traffic",
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
                "lane_side_relation": str(expanded.get("lane_side_relation") or "same_lane"),
                "motion_state": str(expanded.get("motion_state") or expanded.get("motion_bias") or "unknown"),
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

RELATION_DSL_PATH = {os.path.basename(relation_dsl_path)!r}
OUTPUT_PATH = {os.path.basename(output_path)!r}


def _lane_width_for_class(name):
    return float(LANE_WIDTH_METERS.get(str(name or "standard"), 3.5))


def _normalize(vx, vy):
    length = math.hypot(vx, vy)
    if length <= 1e-6:
        return 1.0, 0.0
    return vx / length, vy / length


def _lane_anchor_offset(entity, lane_width):
    anchor = entity.get("lane_anchor")
    lane_index = int(entity.get("lane_index_relation", 0))
    if anchor == "center_of_lane":
        return lane_index * lane_width
    if anchor == "left_lane_center":
        return -lane_width + lane_index * lane_width
    if anchor == "right_lane_center":
        return lane_width + lane_index * lane_width
    if anchor == "left_curbside":
        return -(lane_width * 1.5)
    if anchor == "right_curbside":
        return lane_width * 1.5
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

    entities = []
    for entity in relation_dsl.get("ego_relations") or relation_dsl.get("entities", []):
        longitudinal = _longitudinal_distance(entity)
        lateral = _lane_anchor_offset(entity, lane_width)
        base_x = anchor_x + forward_x * longitudinal + right_x * lateral
        base_y = anchor_y + forward_y * longitudinal + right_y * lateral
        heading_relation = str(entity.get("heading_relation") or "unknown")
        yaw = anchor_yaw
        if heading_relation == "opposite_direction":
            yaw = anchor_yaw + 180.0
        elif heading_relation == "crossing":
            yaw = anchor_yaw + 90.0

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


def _entity_min_distance(entity: Dict[str, Any]) -> float:
    category = str(entity.get("category") or "")
    if category == "pedestrian":
        return 0.8
    if category in STATIC_CATEGORIES:
        return 0.5
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
            if distance >= threshold or distance <= 1e-6:
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
    topology_has_multiple_lanes: bool,
) -> float:
    lane_side_relation = str(entity.get("lane_side_relation") or "")
    if lane_side_relation in {"", "same_lane", "crosswalk"}:
        return 0.0

    side_sign = _relation_side_sign(lane_side_relation)
    parking_lane_count = _parking_lane_count_for_relation(lane_side_relation, lane_context)
    category = str(entity.get("category") or "")

    if lane_side_relation in {"left_edge", "right_edge"}:
        if category == "parked_vehicle":
            # Keep parked vehicles near the drivable edge instead of the geometric
            # parking-lane center. In CARLA OpenDRIVE worlds, conservative edge
            # placement is more stable than pushing them fully into the outer lane.
            magnitude = lane_width * 0.5
        elif parking_lane_count > 0:
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

    if topology_has_multiple_lanes:
        return 0.0

    if lane_side_relation in {"left_lane", "right_lane"} or "opposing" in lane_side_relation:
        return side_sign * lane_width

    return 0.0


def _project_vehicle_entity(
    entity: Dict[str, Any],
    lane: Dict[str, Any],
    lane_width: float,
    lane_context: Dict[str, Any],
    topology_has_multiple_lanes: bool,
) -> Dict[str, Any]:
    projected = _deep_copy(entity)
    px, py = _project_point_to_segment(
        float(entity["location"]["x"]),
        float(entity["location"]["y"]),
        lane["start"],
        lane["end"],
    )[:2]
    road_yaw = float(lane["start"].get("yaw", 0.0))
    yaw = road_yaw
    heading_relation = str(entity.get("heading_relation") or "unknown")
    if heading_relation == "opposite_direction":
        yaw += 180.0

    offset = _vehicle_lateral_offset(
        entity,
        lane_width,
        lane_context,
        topology_has_multiple_lanes,
    )
    if abs(offset) > 1e-6:
        # Keep lane-side semantics in the road frame rather than the actor frame.
        # Otherwise an oncoming vehicle would flip left/right when its yaw is reversed.
        normal_yaw = math.radians(road_yaw + 90.0)
        px += math.cos(normal_yaw) * offset
        py += math.sin(normal_yaw) * offset
    projected["location"] = {"x": px, "y": py, "z": 0.3}
    projected["rotation"] = {"pitch": 0.0, "yaw": yaw, "roll": 0.0}
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
    offset = side * (lane_width * 0.9)
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
    allowed_actor_types = _coerce_dict(lane_context.get("allowed_actor_types"))
    lane_catalog = _coerce_dict(lane_context.get("lane_catalog"))
    catalog_entry = _coerce_dict(lane_catalog.get(relation))
    role = str(catalog_entry.get("role") or "driving")
    allowed = _coerce_list(allowed_actor_types.get(role))
    return category in allowed, role


def project_entities_to_xodr(
    raw_coordinates: Dict[str, Any],
    scene_understanding: Dict[str, Any],
    spawn_context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    topology_sample = _coerce_list((spawn_context or {}).get("topology_sample"))
    fallback_lane = select_anchor_lane(spawn_context)
    lane_context = _lane_context(scene_understanding, fallback_lane)
    lane_groups = _coerce_list(scene_understanding.get("road_network", {}).get("lane_groups"))
    lane_width_class = str((lane_groups[0] if lane_groups else {}).get("lane_width_class") or "standard")
    lane_width = float(LANE_WIDTH_METERS.get(lane_width_class, 3.5))
    topology_has_multiple_lanes = _topology_has_multiple_lanes(topology_sample)

    projected_entities = []
    for entity in _coerce_list(raw_coordinates.get("entities")):
        lane = _closest_lane(entity.get("location", {}), topology_sample, fallback_lane)
        category = str(entity.get("category") or "")
        if category == "pedestrian":
            projected = _project_pedestrian_entity(entity, lane, lane_width)
        elif category in STATIC_CATEGORIES:
            projected = _project_static_entity(entity, lane, lane_width)
        else:
            projected = _project_vehicle_entity(
                entity,
                lane,
                lane_width,
                lane_context,
                topology_has_multiple_lanes,
            )
        projected["projected_lane"] = {
            "road_id": lane.get("road_id"),
            "lane_id": lane.get("lane_id"),
        }
        projected_entities.append(projected)

    projected_entities = _resolve_collisions(projected_entities)
    return {
        "selected_anchor_lane": raw_coordinates.get("selected_anchor_lane", fallback_lane),
        "entities": projected_entities,
        "metadata": {
            **_coerce_dict(raw_coordinates.get("metadata")),
            "projection_version": "xodr-projection-v1",
            "coordinate_stage": "projected",
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
        longitudinal_ok = abs(longitudinal - expected_longitudinal) <= max(
            2.5,
            float(ego_relation.get("group_spacing_m") or 0.0) + 1.0,
        )
        lateral_ok = abs((lateral / 3.5) - expected_lateral_band) <= 0.8
        ego_results.append(
            {
                "entity_id": ego_relation.get("entity_id"),
                "status": "pass" if longitudinal_ok and lateral_ok else "fail",
                "longitudinal_check": longitudinal_ok,
                "lateral_check": lateral_ok,
                "actual_longitudinal_m": round(longitudinal, 3),
                "expected_longitudinal_m": round(expected_longitudinal, 3),
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
        "metadata": {
            "validation_version": "relation-layout-validation-v1",
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


def build_projected_spawn_payload(projected_coordinates: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"entities": []}
    for entity in _coerce_list(projected_coordinates.get("entities")):
        spawn_kind = str(entity.get("spawn_kind") or "vehicle")
        category = str(entity.get("category") or "car")
        placement_mode = "direct" if category == "parked_vehicle" else "project_to_lane"
        explicit_color = preferred_vehicle_color(entity)
        payload["entities"].append(
            {
                "id": entity["id"],
                "spawn_kind": spawn_kind,
                "blueprint_name": entity.get("blueprint_name"),
                "category": category,
                "location": entity["location"],
                "rotation": entity["rotation"],
                "color": None
                if spawn_kind != "vehicle"
                else (explicit_color or color_for_entity(entity["id"], category)),
                "placement_mode": placement_mode,
                "appearance": _coerce_dict(entity.get("appearance")),
            }
        )
    return payload
