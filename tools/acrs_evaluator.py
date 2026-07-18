"""Deterministic ACRS evaluation for reconstructed static CARLA scenes.

The evaluator deliberately separates reference facts from candidate evidence.  In
particular, ``*_topo.json`` and ``target_*`` fields in map-match reports are never
used as candidate truth.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a declared project dependency.
    linear_sum_assignment = None


SCHEMA_VERSION = "acrs-report-v1"
REFERENCE_SCHEMA_VERSION = "acrs-reference-v1"
UNKNOWN = {None, "", "unknown", "uncertain", "not_visible", "n/a", "na"}
JUNCTION_TYPES = {"t_junction", "cross_intersection", "roundabout", "multi_branch"}
TOPOLOGY_ALIASES = {
    "straight_road": "straight",
    "straight_two_way": "straight",
    "line": "straight",
    "curved_road": "curve",
    "intersection": "cross_intersection",
    "cross_junction": "cross_intersection",
    "signalized_intersection": "cross_intersection",
}
PAIRWISE_RELATION_VALUES = {
    "longitudinal_relation": {"ahead_of_other", "behind_other", "aligned_with_other"},
    "lateral_relation": {"left_of_other", "right_of_other", "same_lateral_band"},
    "lane_relation": {
        "same_lane", "same_parking_lane", "adjacent_left_lane",
        "adjacent_right_lane", "cross_lane", "different_lane",
    },
}

# Spawnable traffic participants belong to the actor dimension, even when they
# are parked.  They must never leak into background-environment set scoring.
ENVIRONMENT_ACTOR_TAGS = {
    "car", "cars", "vehicle", "vehicles", "traffic_participant",
    "traffic_participants", "parkingcar", "parking_car", "parking_cars",
    "parked_car", "parked_cars", "parked_vehicle", "parked_vehicles",
}


class ACRSReferenceError(ValueError):
    """Raised when a human reference cannot support an ACRS evaluation."""


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _slug(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")


def _known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _slug(value) not in UNKNOWN
    return True


def _canonical_topology(value: Any) -> str:
    slug = _slug(value)
    return TOPOLOGY_ALIASES.get(slug, slug)


def _canonical_directionality(value: Any) -> str:
    slug = _slug(value)
    if slug in {"bidirectional", "two_way", "undivided_two_way", "divided_two_way"}:
        return "two_way"
    if slug in {"oneway", "one_way", "single_direction"}:
        return "one_way"
    return slug


def _canonical_category(value: Any) -> str:
    slug = _slug(value)
    aliases = {
        "vehicle": "car", "sedan": "car", "suv": "car", "van": "car",
        "lorry": "truck", "pickup": "truck", "cyclist": "bicycle",
        "bike": "bicycle", "motorbike": "motorcycle", "person": "pedestrian",
    }
    return aliases.get(slug, slug)


def _canonical_distance(value: Any) -> str:
    slug = _slug(value)
    return {"immediate": "near", "close": "near", "medium": "mid", "distant": "far"}.get(slug, slug)


def _canonical_time(value: Any) -> str:
    slug = _slug(value)
    return {"daytime": "day", "nighttime": "night", "sunrise": "dawn", "sunset": "dusk"}.get(slug, slug)


def _canonical_tag(value: Any) -> str:
    slug = _slug(value)
    aliases = {
        "building": "building", "buildings": "building",
        # These are naming variants at ACRS's semantic granularity.  A more
        # specific label such as apartment_building remains distinct.
        "residential_house": "residential_building",
        "residential_houses": "residential_building",
        "residential_buildings": "residential_building",
        "traffic_lights": "traffic_light", "traffic_light": "traffic_light",
        "overhead_traffic_lights": "traffic_light", "traffic_control": "traffic_light",
        "street_lights": "street_light", "street_lamps": "street_light",
        "sidewalks": "sidewalk", "vegetation": "vegetation", "trees": "vegetation",
        "utility_poles": "utility_pole", "poles": "utility_pole",
    }
    return aliases.get(slug, slug)


def _canonical_environment_tags(value: Any) -> Any:
    """Normalize environment sets and remove actor-only labels.

    ``unknown`` remains a scalar so reference coverage semantics are retained.
    """
    if not _known(value):
        return value
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    tags = {
        _canonical_tag(item) for item in raw
        if _known(item) and _slug(item) not in ENVIRONMENT_ACTOR_TAGS
    }
    return sorted(tags)


def _canonical_lane(value: Any) -> str:
    slug = _slug(value)
    aliases = {
        "ego_lane": "same_lane", "left": "left_lane", "right": "right_lane",
        "opposing": "opposing_lane", "oncoming_lane": "opposing_lane",
    }
    return aliases.get(slug, slug)


def _canonical_lane_from_right(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        return _slug(value)


def _canonical_pairwise_relation(field: str, value: Any) -> str:
    slug = _slug(value)
    if field == "longitudinal_relation":
        return {
            "ahead": "ahead_of_other", "ahead_of": "ahead_of_other",
            "behind": "behind_other", "behind_of": "behind_other",
            "aligned": "aligned_with_other", "alongside": "aligned_with_other",
        }.get(slug, slug)
    if field == "lateral_relation":
        return {
            "left": "left_of_other", "right": "right_of_other",
            "aligned": "same_lateral_band", "same_lateral": "same_lateral_band",
        }.get(slug, slug)
    if field == "lane_relation":
        return {
            "same_parking_strip": "same_parking_lane",
            "adjacent_left": "adjacent_left_lane",
            "adjacent_right": "adjacent_right_lane",
            "opposing_lane": "cross_lane",
        }.get(slug, slug)
    return slug


def _as_set(value: Any) -> set:
    if not _known(value):
        return set()
    if isinstance(value, dict):
        return {
            _canonical_tag(key) for key, enabled in value.items()
            if enabled is True and key != "known" and _slug(key) not in ENVIRONMENT_ACTOR_TAGS
        }
    if isinstance(value, (list, tuple, set)):
        return {
            _canonical_tag(item) for item in value
            if _known(item) and _slug(item) not in ENVIRONMENT_ACTOR_TAGS
        }
    return set() if _slug(value) in ENVIRONMENT_ACTOR_TAGS else {_canonical_tag(value)}


def count_similarity(reference: Any, actual: Any) -> float:
    try:
        ref = float(reference)
        act = float(actual)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - abs(act - ref) / max(abs(act), abs(ref), 1.0))


def exact_similarity(reference: Any, actual: Any, canonicalizer=None) -> float:
    if not _known(actual):
        return 0.0
    canonicalizer = canonicalizer or (lambda value: value)
    return 1.0 if canonicalizer(reference) == canonicalizer(actual) else 0.0


def jaccard_similarity(reference: Any, actual: Any) -> float:
    ref, act = _as_set(reference), _as_set(actual)
    if not ref and not act:
        return 1.0
    if not ref:
        return 0.0
    return len(ref & act) / len(ref | act)


@dataclass
class ScoreItem:
    name: str
    reference: Any
    actual: Any
    score: Optional[float]
    weight: float
    source: str = "structured"

    @property
    def applicable(self) -> bool:
        return self.score is not None

    def as_dict(self) -> dict:
        return {
            "reference": self.reference,
            "actual": self.actual,
            "score": None if self.score is None else round(self.score, 6),
            "weight": self.weight,
            "applicable": self.applicable,
            "source": self.source,
        }


def _item(name: str, reference: Any, actual: Any, weight: float, comparator, source="structured") -> ScoreItem:
    score = comparator(reference, actual) if _known(reference) else None
    return ScoreItem(name, reference, actual, score, weight, source)


def _weighted(items: Sequence[ScoreItem]) -> Tuple[Optional[float], float]:
    applicable = [item for item in items if item.applicable]
    total_possible = sum(item.weight for item in items)
    total = sum(item.weight for item in applicable)
    if total <= 0:
        return None, 0.0
    score = sum(float(item.score) * item.weight for item in applicable) / total
    coverage = total / total_possible if total_possible else 0.0
    return score, coverage


def validate_reference(reference: dict) -> dict:
    if not isinstance(reference, dict):
        raise ACRSReferenceError("ACRS reference must be a JSON object.")
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ACRSReferenceError(
            f"reference.schema_version must be {REFERENCE_SCHEMA_VERSION!r}."
        )
    critical = {str(item) for item in _list(reference.get("critical_actor_ids")) if str(item)}
    actors = _list(reference.get("actors"))
    ids = [str(actor.get("id") or "") for actor in actors if isinstance(actor, dict)]
    if len(ids) != len(actors) or any(not actor_id for actor_id in ids):
        raise ACRSReferenceError("Every reference actor must be an object with a non-empty id.")
    missing_categories = [
        actor_id for actor_id, actor in zip(ids, actors)
        if not _known(actor.get("category"))
    ]
    if missing_categories:
        raise ACRSReferenceError(
            f"Reference actors must have known categories: {', '.join(missing_categories)}"
        )
    if not critical:
        raise ACRSReferenceError("critical_actor_ids must contain at least one non-ego actor.")
    missing = sorted(critical - set(ids))
    if missing:
        raise ACRSReferenceError(f"critical_actor_ids not found in actors: {', '.join(missing)}")
    if "ego" in critical or "ego_vehicle" in critical:
        raise ACRSReferenceError("ego is a coordinate anchor and cannot be a critical actor.")
    duplicates = sorted({actor_id for actor_id in ids if actor_id and ids.count(actor_id) > 1})
    if duplicates:
        raise ACRSReferenceError(f"duplicate reference actor ids: {', '.join(duplicates)}")
    valid_relation_ids = set(ids) | {"ego", "ego_vehicle"}
    for index, relation in enumerate(_list(reference.get("pairwise_relations"))):
        if not isinstance(relation, dict):
            raise ACRSReferenceError(f"pairwise_relations[{index}] must be an object.")
        endpoints = {str(relation.get("entity_id") or ""), str(relation.get("other_entity_id") or "")}
        unknown = endpoints - valid_relation_ids
        if unknown:
            raise ACRSReferenceError(
                f"pairwise_relations[{index}] references unknown actors: {', '.join(sorted(unknown))}"
            )
        for field, allowed in PAIRWISE_RELATION_VALUES.items():
            value = relation.get(field)
            if _known(value) and _canonical_pairwise_relation(field, value) not in allowed:
                raise ACRSReferenceError(
                    f"pairwise_relations[{index}].{field} has unsupported value {value!r}; "
                    f"allowed values: {', '.join(sorted(allowed))}."
                )
    normalized = deepcopy(reference)
    environment = _dict(normalized.get("environment"))
    for field in ("roadside_context_left", "roadside_context_right", "landmarks_and_controls"):
        if field in environment:
            environment[field] = _canonical_environment_tags(environment[field])
    for actor in normalized["actors"]:
        actor_id = str(actor.get("id") or "")
        actor["role"] = "critical" if actor_id in critical else _slug(actor.get("role") or "background")
        if actor["role"] not in {"critical", "background"}:
            raise ACRSReferenceError(f"actor {actor_id!r} has invalid role {actor['role']!r}.")
    for relation in _list(normalized.get("pairwise_relations")):
        for field in PAIRWISE_RELATION_VALUES:
            if _known(relation.get(field)):
                relation[field] = _canonical_pairwise_relation(field, relation[field])
    return normalized


def reference_from_scene_understanding(scene_understanding: dict, scene_id: str = "") -> dict:
    """Create an editable reference draft.  It is intentionally not auto-approved."""
    road = _dict(scene_understanding.get("road_network"))
    matching = _dict(road.get("map_matching"))
    lanes = _dict((_list(road.get("lane_groups")) or [{}])[0])
    env = _dict(scene_understanding.get("general_environment"))
    actors = []
    for actor in _list(scene_understanding.get("traffic_subjects")) + _list(scene_understanding.get("background_traffic")):
        if not isinstance(actor, dict):
            continue
        anchor = _dict(actor.get("anchor_relation"))
        actors.append({
            "id": str(actor.get("id") or ""),
            "role": "background",
            "category": actor.get("category", "unknown"),
            "subtype": actor.get("subtype", "unknown"),
            "lane_assignment": actor.get("lane_side_relation", "unknown"),
            "lane_from_right": anchor.get("lane_from_right", "unknown"),
            "branch_assignment": actor.get("layout_anchor_id", "unknown"),
            "heading_relation": actor.get("heading_relation_to_ego", "unknown"),
            "longitudinal_relation": actor.get("longitudinal_relation", "unknown"),
            "distance_band": actor.get("longitudinal_proximity", "unknown"),
        })
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "scene_id": scene_id,
        "road_topology": {
            "topology_type": matching.get("topology_type", "unknown"),
            "directionality": road.get("directionality", "unknown"),
            "forward_lane_count": matching.get("forward_lane_count", lanes.get("forward_lane_count", "unknown")),
            "opposing_lane_count": matching.get("opposing_lane_count", lanes.get("opposing_lane_count", "unknown")),
            "ego_lane_from_right": matching.get("ego_lane_from_right", "unknown"),
            "has_center_median": matching.get("has_center_median", "unknown"),
            "left_parking_presence": matching.get("left_parking_presence", "unknown"),
            "right_parking_presence": matching.get("right_parking_presence", "unknown"),
            "junction_visible": matching.get("junction_visible", "unknown"),
            "junction_type": matching.get("junction_type", "unknown"),
            "junction_branches": matching.get("junction_branches", "unknown"),
            "branch_count": matching.get("target_branch_count", "unknown"),
        },
        "environment": {
            "weather": env.get("weather_hint", "unknown"),
            "lighting": env.get("lighting_hint", "unknown"),
            "time_of_day": env.get("time_of_day_hint", "unknown"),
            "road_surface": env.get("road_surface_hint", "unknown"),
            "urban_density": env.get("urban_density", "unknown"),
            "roadside_context_left": env.get("roadside_context_left", []),
            "roadside_context_right": env.get("roadside_context_right", []),
            "landmarks_and_controls": env.get("non_spawnable_landmarks", []),
        },
        "actors": actors,
        "pairwise_relations": [
            deepcopy(relation)
            for relation in _list(scene_understanding.get("key_pairwise_relations"))
            if isinstance(relation, dict)
            and str(relation.get("entity_id") or "") not in {"ego", "ego_vehicle"}
            and str(relation.get("other_entity_id") or "") not in {"ego", "ego_vehicle"}
        ],
        "critical_actor_ids": [],
        "annotation": {"status": "draft_requires_human_review", "source": "scene_understanding_prefill"},
    }


def _weather_from_preset(preset: Any) -> dict:
    slug = _slug(preset)
    return {
        "weather": "rain" if "rain" in slug else ("fog" if "cloudy" in slug else "clear"),
        "lighting": "night" if "night" in slug else "daylight",
        "time_of_day": "night" if "night" in slug else "day",
        "road_surface": "wet" if ("wet" in slug or "rain" in slug) else "dry",
    }


def _actual_lane_assignment(actor: dict, ego_frame: dict) -> str:
    """Prefer CARLA-realized parking evidence over the coarse ego lateral band."""
    visual_result = _dict(actor.get("visual_position_result"))
    visual_target = _canonical_lane(visual_result.get("target_lane"))
    visual_kind = _slug(visual_result.get("result"))
    parking_result = _slug(actor.get("parking_projection_result"))
    intended = _canonical_lane(actor.get("lane_side_relation"))
    if (
        visual_kind == "parking_lane"
        or parking_result in {"parking_lane", "success", "projected"}
    ):
        if visual_target in {"left_parking_lane", "right_parking_lane"}:
            return visual_target
        if intended in {"left_parking_lane", "right_parking_lane"}:
            return intended
    waypoint = _dict(actor.get("actual_waypoint"))
    basis_lane = _dict(visual_result.get("basis_lane"))
    # A collision adjustment can move the actor centre far enough from the ego
    # centreline for the coarse lateral band to say left/right even though CARLA
    # still resolves the actor onto the requested driving waypoint.  Trust the
    # requested relation only when the realized waypoint confirms that basis
    # road/lane; otherwise retain the runtime ego-frame fallback below.
    if (
        visual_kind == "driving_lane"
        and visual_target in {"same_lane", "left_lane", "right_lane", "opposing_lane"}
        and waypoint.get("road_id") is not None
        and waypoint.get("lane_id") is not None
        and str(waypoint.get("road_id")) == str(basis_lane.get("road_id"))
        and str(waypoint.get("lane_id")) == str(basis_lane.get("lane_id"))
    ):
        return visual_target
    if _slug(waypoint.get("lane_type")) == "parking":
        lateral = ego_frame.get("lateral_m")
        if isinstance(lateral, (int, float)):
            return "right_parking_lane" if lateral >= 0 else "left_parking_lane"
    return _canonical_lane(ego_frame.get("lane_side_relation", "unknown"))


def build_candidate_evidence(
    match_report: dict,
    actors_payload: dict,
    render_graph: dict,
    visual_observation: Optional[dict] = None,
) -> dict:
    """Build candidate facts without consulting target-derived topology fields."""
    best = _dict(match_report.get("best_match"))
    features = _dict(best.get("candidate_features"))
    score_details = _dict(best.get("score_details"))
    branch_detail = _dict(score_details.get("branch_direction_detail"))
    branches = _dict(branch_detail.get("candidate"))
    if not branches:
        # ACRS branch_count is ego-reachable maneuver directions (ahead/left/
        # right/uturn), not the physical leg count including the ego approach.
        branches = (
            _dict(features.get("junction_branch_dirs"))
            or _dict(features.get("lane_maneuver_dirs"))
            or _dict(features.get("physical_junction_arms"))
        )
    same_direction = features.get("same_direction_lane_count")
    same_road = features.get("same_road_lane_count")
    opposing = "unknown"
    if isinstance(same_direction, (int, float)) and isinstance(same_road, (int, float)):
        opposing = max(0, int(same_road) - int(same_direction))
    directionality = "unknown"
    if isinstance(opposing, int):
        directionality = "two_way" if opposing > 0 else "one_way"
    topology = _canonical_topology(features.get("candidate_topology_type"))
    junction_visible = topology in JUNCTION_TYPES
    road = {
        "topology_type": topology,
        "directionality": directionality,
        "forward_lane_count": same_direction,
        "opposing_lane_count": opposing,
        "ego_lane_from_right": features.get("ego_lane_from_right", "unknown"),
        "has_center_median": features.get("has_center_median_candidate", "unknown"),
        "left_parking_presence": features.get("left_parking_lane_present", "unknown"),
        "right_parking_presence": features.get("right_parking_lane_present", "unknown"),
        "junction_visible": junction_visible,
        "junction_type": topology if junction_visible else "none",
        "junction_branches": {key: bool(branches.get(key)) for key in ("ahead", "left", "right", "uturn")} if branches else "unknown",
        "branch_count": branches.get(
            "branch_count",
            branches.get("arm_count", branches.get("leg_count", features.get("estimated_junction_degree", "unknown"))),
        ),
    }

    graph_actors = []
    for actor in _list(render_graph.get("actors")):
        if not isinstance(actor, dict) or actor.get("spawned") is not True:
            continue
        ego_frame = _dict(actor.get("ego_frame"))
        waypoint = _dict(actor.get("actual_waypoint"))
        lane_assignment = _actual_lane_assignment(actor, ego_frame)
        graph_actors.append({
            "id": str(actor.get("id") or ""),
            "category": actor.get("category", "unknown"),
            "subtype": actor.get("subtype", "unknown"),
            "lane_assignment": lane_assignment,
            "lane_from_right": waypoint.get("lane_from_right", actor.get("junction_lane_from_right", "unknown")),
            "branch_assignment": waypoint.get("branch_assignment", waypoint.get("junction_leg", "unknown")),
            "heading_relation": actor.get("heading_relation_to_ego", "unknown"),
            "longitudinal_relation": (
                "ahead" if isinstance(ego_frame.get("longitudinal_m"), (int, float)) and ego_frame["longitudinal_m"] > 1
                else "behind" if isinstance(ego_frame.get("longitudinal_m"), (int, float)) and ego_frame["longitudinal_m"] < -1
                else "aligned" if isinstance(ego_frame.get("longitudinal_m"), (int, float)) else "unknown"
            ),
            "distance_band": ego_frame.get("distance_band", "unknown"),
            "longitudinal_m": ego_frame.get("longitudinal_m"),
            "lateral_m": ego_frame.get("lateral_m"),
            "actual_waypoint": waypoint,
            "is_parking_lane": lane_assignment in {"left_parking_lane", "right_parking_lane"},
        })

    preset_env = _weather_from_preset(_dict(actors_payload.get("metadata")).get("carla_weather_preset"))
    map_env = _dict(features.get("environment_context"))
    environment = {
        **preset_env,
        "urban_density": map_env.get("environment_class", "unknown"),
        "roadside_context_left": map_env.get("roadside_context_left", []),
        "roadside_context_right": map_env.get("roadside_context_right", []),
        "landmarks_and_controls": [
            key for key, enabled in {
                "buildings": map_env.get("buildings_nearby"),
                "sidewalks": map_env.get("sidewalks_nearby"),
                "traffic_control": map_env.get("traffic_control_nearby"),
                "street_light": features.get("has_street_lights"),
                "water": map_env.get("water_nearby"),
            }.items() if enabled is True
        ],
    }
    for field in ("roadside_context_left", "roadside_context_right", "landmarks_and_controls"):
        environment[field] = _canonical_environment_tags(environment[field])
    sources = {key: "carla_weather_preset" for key in preset_env}
    sources.update({key: "map_candidate_features" for key in ("urban_density", "roadside_context_left", "roadside_context_right", "landmarks_and_controls")})
    conflicts = []
    visual = _dict(visual_observation)
    confidence = _dict(visual.get("confidence"))
    for key in ("weather", "lighting", "time_of_day", "road_surface", "urban_density", "roadside_context_left", "roadside_context_right", "landmarks_and_controls"):
        observed = visual.get(key)
        conf = _slug(confidence.get(key, visual.get("overall_confidence", "unknown")))
        if _known(observed) and conf in {"medium", "high"}:
            if key in {"roadside_context_left", "roadside_context_right", "landmarks_and_controls"}:
                observed = _canonical_environment_tags(observed)
            if _known(environment.get(key)) and environment.get(key) != observed:
                conflicts.append({"field": key, "structured": environment.get(key), "visual": observed})
            environment[key] = observed
            sources[key] = "vlm_render_observation"
    return {
        "road_topology": road,
        "environment": environment,
        "environment_sources": sources,
        "actors": graph_actors,
        "evidence_conflicts": conflicts,
        "visual_observation_available": bool(visual_observation),
        "runtime_truth_available": (
            render_graph.get("truth_source") == "carla_actor_transform"
            and not _dict(render_graph.get("metadata")).get("truth_unavailable", False)
        ),
    }


def score_road(reference: dict, actual: dict) -> dict:
    ref, act = _dict(reference), _dict(actual)
    geometry = [_item("topology_type", ref.get("topology_type"), act.get("topology_type"), 1.0, lambda r, a: exact_similarity(r, a, _canonical_topology))]
    lane_items = [
        _item("directionality", ref.get("directionality"), act.get("directionality"), .20, lambda r, a: exact_similarity(r, a, _canonical_directionality)),
        _item("forward_lane_count", ref.get("forward_lane_count"), act.get("forward_lane_count"), .25, count_similarity),
        _item("opposing_lane_count", ref.get("opposing_lane_count"), act.get("opposing_lane_count"), .20, count_similarity),
        _item("ego_lane_from_right", ref.get("ego_lane_from_right"), act.get("ego_lane_from_right"), .15, count_similarity),
        _item("has_center_median", ref.get("has_center_median"), act.get("has_center_median"), .10, exact_similarity),
        _item("left_parking_presence", ref.get("left_parking_presence"), act.get("left_parking_presence"), .05, exact_similarity),
        _item("right_parking_presence", ref.get("right_parking_presence"), act.get("right_parking_presence"), .05, exact_similarity),
    ]
    ref_visible = ref.get("junction_visible")
    if not _known(ref_visible):
        ref_visible = _canonical_topology(ref.get("topology_type")) in JUNCTION_TYPES if _known(ref.get("topology_type")) else "unknown"
    act_visible = act.get("junction_visible")
    connectivity = [_item("junction_visible", ref_visible, act_visible, .25, exact_similarity)]
    if ref_visible is False:
        # Correct absence is the whole connectivity question for an open road.
        connectivity = [_item("junction_visible", False, act_visible, 1.0, exact_similarity)]
    else:
        connectivity.extend([
            _item("junction_type", ref.get("junction_type"), act.get("junction_type"), .25, lambda r, a: exact_similarity(r, a, _canonical_topology)),
            _item("junction_branches", ref.get("junction_branches"), act.get("junction_branches"), .35, jaccard_similarity),
            _item("branch_count", ref.get("branch_count"), act.get("branch_count"), .15, count_similarity),
        ])
    geo_score, geo_cov = _weighted(geometry)
    lane_score, lane_cov = _weighted(lane_items)
    conn_score, conn_cov = _weighted(connectivity)
    groups = [(geo_score, .30), (lane_score, .40), (conn_score, .30)]
    applicable = [(score, weight) for score, weight in groups if score is not None]
    total = sum(weight for _, weight in applicable)
    score = sum(score * weight for score, weight in applicable) / total if total else None
    return {
        "score": score,
        "coverage": .30 * geo_cov + .40 * lane_cov + .30 * conn_cov,
        "components": {"geometry_type": geo_score, "lane_organization": lane_score, "connectivity": conn_score},
        "details": {item.name: item.as_dict() for item in geometry + lane_items + connectivity},
    }


def score_environment(reference: dict, actual: dict, sources: Optional[dict] = None) -> dict:
    ref, act, sources = _dict(reference), _dict(actual), _dict(sources)
    context_items = [
        _item("urban_density", ref.get("urban_density"), act.get("urban_density"), .35, exact_similarity, sources.get("urban_density", "structured")),
        _item("roadside_context_left", ref.get("roadside_context_left"), act.get("roadside_context_left"), .225, jaccard_similarity, sources.get("roadside_context_left", "structured")),
        _item("roadside_context_right", ref.get("roadside_context_right"), act.get("roadside_context_right"), .225, jaccard_similarity, sources.get("roadside_context_right", "structured")),
        _item("landmarks_and_controls", ref.get("landmarks_and_controls"), act.get("landmarks_and_controls"), .20, jaccard_similarity, sources.get("landmarks_and_controls", "structured")),
    ]
    context_score, context_cov = _weighted(context_items)
    primary = [
        _item("weather", ref.get("weather"), act.get("weather"), .25, exact_similarity, sources.get("weather", "structured")),
        _item("lighting", ref.get("lighting"), act.get("lighting"), .25, exact_similarity, sources.get("lighting", "structured")),
        _item("time_of_day", ref.get("time_of_day"), act.get("time_of_day"), .10, lambda r, a: exact_similarity(r, a, _canonical_time), sources.get("time_of_day", "structured")),
        _item("road_surface", ref.get("road_surface"), act.get("road_surface"), .15, exact_similarity, sources.get("road_surface", "structured")),
    ]
    if context_score is not None:
        primary.append(ScoreItem("surrounding_context", ref, act, context_score, .25, "mixed"))
    else:
        primary.append(ScoreItem("surrounding_context", ref, act, None, .25, "mixed"))
    score, primary_cov = _weighted(primary)
    # Replace the all-or-nothing context coverage with its field-level coverage.
    coverage = primary_cov - (.25 if context_score is not None else 0.0) + .25 * context_cov
    details = {item.name: item.as_dict() for item in primary[:-1] + context_items}
    return {"score": score, "coverage": max(0.0, coverage), "components": {"surrounding_context": context_score}, "details": details}


def _category_similarity(reference: Any, actual: Any) -> float:
    ref, act = _canonical_category(reference), _canonical_category(actual)
    if ref == act:
        return 1.0
    vehicles = {"car", "truck", "bus"}
    two_wheel = {"motorcycle", "bicycle"}
    if ref in vehicles and act in vehicles:
        return .4
    if ref in two_wheel and act in two_wheel:
        return .4
    return 0.0


def _identity_similarity(reference: dict, actual: dict) -> float:
    category = _category_similarity(reference.get("category"), actual.get("category"))
    ref_subtype = _slug(reference.get("subtype"))
    if not _known(reference.get("subtype")) or ref_subtype == _canonical_category(reference.get("category")):
        return category
    subtype = exact_similarity(reference.get("subtype"), actual.get("subtype"), _slug)
    return .8 * category + .2 * subtype


def _actor_field_score(reference: dict, actual: dict, fields: Sequence[Tuple[str, Any]]) -> Tuple[Optional[float], float, dict]:
    items = []
    for name, canonicalizer in fields:
        comparator = (lambda r, a, c=canonicalizer: exact_similarity(r, a, c))
        items.append(_item(name, reference.get(name), actual.get(name), 1.0, comparator))
    score, coverage = _weighted(items)
    return score, coverage, {item.name: item.as_dict() for item in items}


def _match_similarity(reference: dict, actual: dict) -> Tuple[float, dict]:
    identity = _identity_similarity(reference, actual)
    if identity == 0:
        return 0.0, {"category": 0.0, "layout": 0.0, "spatial": 0.0}
    layout, _, _ = _actor_field_score(reference, actual, [
        ("lane_assignment", _canonical_lane), ("lane_from_right", lambda value: value),
        ("branch_assignment", _slug), ("heading_relation", _slug),
    ])
    spatial, _, _ = _actor_field_score(reference, actual, [
        ("longitudinal_relation", _slug), ("distance_band", _canonical_distance),
    ])
    # No applicable human field is neutral for matching, but not for reported coverage.
    layout_for_match = .5 if layout is None else layout
    spatial_for_match = .5 if spatial is None else spatial
    total = .35 * identity + .40 * layout_for_match + .25 * spatial_for_match
    return total, {"category": identity, "layout": layout, "spatial": spatial}


def _hungarian(reference: list, actual: list) -> List[Tuple[int, int, float, dict]]:
    if not reference or not actual:
        return []
    similarities = [[_match_similarity(ref, act) for act in actual] for ref in reference]
    costs = [[1.0 - cell[0] for cell in row] for row in similarities]
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(costs)
        pairs = zip(rows.tolist(), cols.tolist())
    else:  # Small deterministic fallback for minimal test environments.
        if len(reference) <= len(actual):
            choices = min(permutations(range(len(actual)), len(reference)), key=lambda cols: sum(costs[row][col] for row, col in enumerate(cols)))
            pairs = enumerate(choices)
        else:
            choices = min(permutations(range(len(reference)), len(actual)), key=lambda rows: sum(costs[row][col] for col, row in enumerate(rows)))
            pairs = ((row, col) for col, row in enumerate(choices))
    output = []
    for row, col in pairs:
        similarity, parts = similarities[row][col]
        if similarity >= .5 and parts["category"] > 0:
            output.append((row, col, similarity, parts))
    return output


def _derive_pairwise(actual_a: dict, actual_b: dict, field: str) -> str:
    if field in {"longitudinal_relation", "longitudinal"}:
        a, b = actual_a.get("longitudinal_m"), actual_b.get("longitudinal_m")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return "unknown"
        return "ahead_of_other" if a > b + 1 else "behind_other" if a < b - 1 else "aligned_with_other"
    if field in {"lateral_relation", "lateral"}:
        a, b = actual_a.get("lateral_m"), actual_b.get("lateral_m")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return "unknown"
        return "right_of_other" if a > b + 1 else "left_of_other" if a < b - 1 else "same_lateral_band"
    if field == "lane_relation":
        lane_a, lane_b = _canonical_lane(actual_a.get("lane_assignment")), _canonical_lane(actual_b.get("lane_assignment"))
        if not _known(lane_a) or not _known(lane_b):
            return "unknown"
        parking_lanes = {"left_parking_lane", "right_parking_lane"}
        if lane_a == lane_b:
            return "same_parking_lane" if lane_a in parking_lanes else "same_lane"
        heading_a = _slug(actual_a.get("heading_relation"))
        heading_b = _slug(actual_b.get("heading_relation"))
        if (
            {heading_a, heading_b} & {"opposite_direction"}
            and heading_a != heading_b
        ) or (
            "opposing_lane" in {lane_a, lane_b} and lane_a != lane_b
        ):
            return "cross_lane"
        lateral_a, lateral_b = actual_a.get("lateral_m"), actual_b.get("lateral_m")
        if isinstance(lateral_a, (int, float)) and isinstance(lateral_b, (int, float)):
            other_offset = lateral_b - lateral_a
            # One standard/wide lane plus positioning tolerance. Larger
            # separations remain merely different rather than adjacent.
            if 1.0 < other_offset <= 5.5:
                return "adjacent_right_lane"
            if -5.5 <= other_offset < -1.0:
                return "adjacent_left_lane"
        return "different_lane"
    return "unknown"


def score_traffic(reference: dict, actual_actors: list) -> dict:
    ref_actors = _list(reference.get("actors"))
    pairs = _hungarian(ref_actors, actual_actors)
    by_ref = {row: (col, similarity, parts) for row, col, similarity, parts in pairs}
    matched_actual = {col for _, col, _, _ in pairs}
    match_rows = []
    for index, ref in enumerate(ref_actors):
        matched = by_ref.get(index)
        match_rows.append({
            "reference_id": ref.get("id"), "role": ref.get("role"),
            "actual_id": actual_actors[matched[0]].get("id") if matched else None,
            "similarity": round(matched[1], 6) if matched else 0.0,
            "parts": matched[2] if matched else None,
        })

    relation_by_role = {"critical": [], "background": []}
    for relation in _list(reference.get("pairwise_relations")):
        if not isinstance(relation, dict):
            continue
        id_to_index = {str(actor.get("id")): idx for idx, actor in enumerate(ref_actors)}
        a_idx = id_to_index.get(str(relation.get("entity_id") or ""))
        b_idx = id_to_index.get(str(relation.get("other_entity_id") or ""))
        # Ego-relative facts are scored per actor above; pairwise scoring is for
        # relationships between two reconstructed participants.
        if a_idx is None or b_idx is None:
            continue
        role = "critical" if any(idx is not None and ref_actors[idx].get("role") == "critical" for idx in (a_idx, b_idx)) else "background"
        for field in ("longitudinal_relation", "lateral_relation", "lane_relation"):
            expected = relation.get(field)
            if not _known(expected):
                continue
            score = 0.0
            observed = "missing_actor"
            if a_idx in by_ref and b_idx in by_ref:
                observed = _derive_pairwise(actual_actors[by_ref[a_idx][0]], actual_actors[by_ref[b_idx][0]], field)
                score = exact_similarity(
                    expected,
                    observed,
                    lambda value, relation_field=field: _canonical_pairwise_relation(relation_field, value),
                )
            relation_by_role[role].append({
                "relation_id": relation.get("id"),
                "entity_id": relation.get("entity_id"),
                "other_entity_id": relation.get("other_entity_id"),
                "field": field,
                "reference": _canonical_pairwise_relation(field, expected),
                "actual": observed,
                "score": score,
            })

    role_reports = {}
    for role in ("critical", "background"):
        indices = [idx for idx, actor in enumerate(ref_actors) if actor.get("role") == role]
        if not indices:
            role_reports[role] = {"applicable": False, "score": None, "coverage": 0.0}
            continue
        category_tp = sum(by_ref[idx][2]["category"] for idx in indices if idx in by_ref)
        recall = category_tp / len(indices)
        if role == "critical":
            recovery = recall
        else:
            matched_critical_actual = {by_ref[idx][0] for idx, actor in enumerate(ref_actors) if actor.get("role") == "critical" and idx in by_ref}
            background_candidate_count = max(0, len(actual_actors) - len(matched_critical_actual))
            precision = category_tp / background_candidate_count if background_candidate_count else (1.0 if not indices else 0.0)
            recovery = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        layout_values, spatial_values = [], []
        layout_applicable = spatial_applicable = 0
        layout_possible = spatial_possible = 0
        for idx in indices:
            ref = ref_actors[idx]
            layout_fields = [("lane_assignment", _canonical_lane), ("lane_from_right", _canonical_lane_from_right), ("branch_assignment", _slug), ("heading_relation", _slug)]
            spatial_fields = [("longitudinal_relation", _slug), ("distance_band", _canonical_distance)]
            layout_possible += len(layout_fields)
            spatial_possible += len(spatial_fields)
            if idx not in by_ref:
                layout_values.extend(0.0 for field, _ in layout_fields if _known(ref.get(field)))
                spatial_values.extend(0.0 for field, _ in spatial_fields if _known(ref.get(field)))
                layout_applicable += sum(_known(ref.get(field)) for field, _ in layout_fields)
                spatial_applicable += sum(_known(ref.get(field)) for field, _ in spatial_fields)
                continue
            actual = actual_actors[by_ref[idx][0]]
            for field, canonicalizer in layout_fields:
                if _known(ref.get(field)):
                    layout_applicable += 1
                    layout_values.append(exact_similarity(ref.get(field), actual.get(field), canonicalizer))
            for field, canonicalizer in spatial_fields:
                if _known(ref.get(field)):
                    spatial_applicable += 1
                    spatial_values.append(exact_similarity(ref.get(field), actual.get(field), canonicalizer))
        for relation in relation_by_role[role]:
            spatial_possible += 1
            spatial_applicable += 1
            spatial_values.append(relation["score"])
        layout_score = sum(layout_values) / len(layout_values) if layout_values else None
        spatial_score = sum(spatial_values) / len(spatial_values) if spatial_values else None
        components = [(recovery, .35), (layout_score, .40), (spatial_score, .25)]
        applicable_components = [(value, weight) for value, weight in components if value is not None]
        role_score = sum(value * weight for value, weight in applicable_components) / sum(weight for _, weight in applicable_components)
        role_reports[role] = {
            "applicable": True, "score": role_score,
            "components": {"recovery": recovery, "road_relative_layout": layout_score, "spatial_relations": spatial_score},
            "coverage": .35 + .40 * (layout_applicable / layout_possible if layout_possible else 0) + .25 * (spatial_applicable / spatial_possible if spatial_possible else 0),
            "reference_count": len(indices), "matched_count": sum(idx in by_ref for idx in indices),
            "pairwise_checks": relation_by_role[role],
        }
    roles = [(role_reports["critical"]["score"], .70)]
    if role_reports["background"]["applicable"]:
        roles.append((role_reports["background"]["score"], .30))
    total_weight = sum(weight for _, weight in roles)
    score = sum(value * weight for value, weight in roles) / total_weight
    coverage = sum(role_reports[name]["coverage"] * weight for name, weight in (("critical", .70), ("background", .30)) if role_reports[name]["applicable"]) / sum(weight for name, weight in (("critical", .70), ("background", .30)) if role_reports[name]["applicable"])
    return {
        "score": score, "coverage": coverage, "roles": role_reports,
        "matches": match_rows,
        "missing_reference_actors": [row["reference_id"] for row in match_rows if row["actual_id"] is None],
        "extra_actual_actors": [actor.get("id") for idx, actor in enumerate(actual_actors) if idx not in matched_actual],
    }


def evaluate_acrs(
    reference: dict,
    candidate: dict,
    *,
    render_image_available: bool,
    mode: str = "hybrid",
    inputs: Optional[dict] = None,
) -> dict:
    reference = validate_reference(reference)
    road = score_road(reference.get("road_topology"), candidate.get("road_topology"))
    environment = score_environment(reference.get("environment"), candidate.get("environment"), candidate.get("environment_sources"))
    traffic = score_traffic(reference, _list(candidate.get("actors")))
    critical_traffic = traffic["roles"]["critical"]
    background_traffic = traffic["roles"]["background"]
    dimension_scores = [road["score"], environment["score"], traffic["score"]]
    acrs = None if any(score is None for score in dimension_scores) else .40 * dimension_scores[0] + .20 * dimension_scores[1] + .40 * dimension_scores[2]
    formal_ready = (
        mode == "hybrid"
        and bool(candidate.get("runtime_truth_available"))
        and render_image_available
        and bool(candidate.get("visual_observation_available"))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if formal_ready else "incomplete",
        "official": formal_ready,
        "scores": {
            "road_topology": None if road["score"] is None else round(100 * road["score"], 4),
            "background_environment": None if environment["score"] is None else round(100 * environment["score"], 4),
            "critical_traffic_participants": None if critical_traffic["score"] is None else round(100 * critical_traffic["score"], 4),
            "background_traffic_participants": None if background_traffic["score"] is None else round(100 * background_traffic["score"], 4),
            "traffic_participants": None if traffic["score"] is None else round(100 * traffic["score"], 4),
            "acrs": None if acrs is None else round(100 * acrs, 4),
        },
        "coverage": {
            "road_topology": round(road["coverage"], 6),
            "background_environment": round(environment["coverage"], 6),
            "critical_traffic_participants": round(critical_traffic["coverage"], 6),
            "background_traffic_participants": round(background_traffic["coverage"], 6),
            "traffic_participants": round(traffic["coverage"], 6),
        },
        "dimensions": {"road_topology": road, "background_environment": environment, "traffic_participants": traffic},
        "evidence_conflicts": _list(candidate.get("evidence_conflicts")),
        "incomplete_reasons": [
            reason for condition, reason in (
                (not candidate.get("runtime_truth_available"), "CARLA runtime actor graph is unavailable."),
                (not render_image_available, "CARLA render image is unavailable."),
                (mode != "hybrid", "Structured-only mode is diagnostic and not an official ACRS result."),
                (mode == "hybrid" and not candidate.get("visual_observation_available"), "VLM render observation is unavailable."),
            ) if condition
        ],
        "inputs": inputs or {},
    }
