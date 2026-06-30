"""Actor-graph verification helpers for static scene layout repair."""

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


VEHICLE_CATEGORIES = {"car", "truck", "bus", "motorcycle", "bicycle"}
FALLBACK_TRUTH_SOURCE = "spawn_payload_fallback"
CARLA_TRUTH_SOURCE = "carla_actor_transform"


def _coerce_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _coerce_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _category(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_vehicle_actor(actor: Dict[str, Any]) -> bool:
    return _category(actor.get("category")) in VEHICLE_CATEGORIES


def _distance_band_from_longitudinal(longitudinal_m: float) -> str:
    distance = abs(float(longitudinal_m))
    if distance < 12.0:
        return "near"
    if distance < 30.0:
        return "mid"
    return "far"


def _lane_side_from_lateral(lateral_m: float) -> str:
    lane_index = _lane_index_from_lateral(lateral_m)
    if lane_index < 0:
        return "left_lane"
    if lane_index > 0:
        return "right_lane"
    return "same_lane"


def _lane_index_from_lateral(lateral_m: float, lane_width: float = 3.5) -> int:
    lateral = float(lateral_m)
    if abs(lateral) < max(1.25, lane_width * 0.35):
        return 0
    return int(round(lateral / lane_width))


def _lane_index_from_relation(value: Any) -> int:
    relation = str(value or "").strip().lower()
    if relation in {"", "same_lane", "crosswalk"}:
        return 0
    if "right" in relation:
        return 1
    if "left" in relation or "opposing" in relation or "oncoming" in relation:
        return -1
    return 0


def _coerce_lane_index(value: Any, fallback_relation: Any = None) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return _lane_index_from_relation(fallback_relation)


def _lane_side_from_index(lane_index: int) -> str:
    if lane_index > 0:
        return "right_lane"
    if lane_index < 0:
        return "left_lane"
    return "same_lane"


def _normalize_yaw(yaw: float) -> float:
    value = float(yaw)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _angle_distance(yaw_a: Any, yaw_b: Any) -> float:
    try:
        return abs(_normalize_yaw(float(yaw_a) - float(yaw_b)))
    except (TypeError, ValueError):
        return 0.0


def _heading_relation(yaw: Any, ego_yaw: Any) -> str:
    if yaw is None or ego_yaw is None:
        return "unknown"
    return "opposite_direction" if _angle_distance(yaw, ego_yaw) > 90.0 else "same_direction"


def _relative_metrics(entity: Dict[str, Any], ego: Dict[str, Any]) -> Tuple[float, float]:
    loc = _coerce_dict(entity.get("location"))
    ego_loc = _coerce_dict(ego.get("location"))
    try:
        yaw = math.radians(float(ego.get("yaw", ego.get("rotation", {}).get("yaw", 0.0))))
    except (TypeError, ValueError, AttributeError):
        yaw = 0.0
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    right_x, right_y = -math.sin(yaw), math.cos(yaw)
    dx = float(loc.get("x", 0.0)) - float(ego_loc.get("x", 0.0))
    dy = float(loc.get("y", 0.0)) - float(ego_loc.get("y", 0.0))
    return dx * forward_x + dy * forward_y, dx * right_x + dy * right_y


def build_source_actor_graph(
    scene_understanding: Dict[str, Any],
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact expected vehicle graph from relation DSL entities."""
    del scene_understanding  # relation_dsl already contains expanded actor semantics.
    actors = []
    ignored = []
    for entity in _coerce_list(
        relation_dsl.get("ego_relations") or relation_dsl.get("entities")
    ):
        lane_index = _coerce_lane_index(
            entity.get("lane_index_relation"),
            entity.get("lane_side_relation"),
        )
        actor = {
            "id": str(entity.get("entity_id") or entity.get("id") or ""),
            "group_id": entity.get("group_id"),
            "category": _category(entity.get("category") or "car"),
            "subtype": str(entity.get("subtype") or entity.get("category") or ""),
            "spawn_kind": str(entity.get("spawn_kind") or "vehicle"),
            "lane_index_relation": lane_index,
            "lane_side_relation": _lane_side_from_index(lane_index),
            "longitudinal_relation": str(entity.get("order_relation") or "ahead"),
            "distance_band": str(entity.get("distance_band") or "near"),
            "heading_relation": str(entity.get("heading_relation") or "unknown"),
            "visual_confidence": str(entity.get("visual_confidence") or "medium"),
        }
        if actor["id"] and _is_vehicle_actor(actor):
            actors.append(actor)
        elif actor["id"]:
            ignored.append(actor)

    ids = [actor["id"] for actor in actors]
    duplicate_ids = sorted([actor_id for actor_id, count in Counter(ids).items() if count > 1])
    return {
        "schema_version": "actor-graph-v1",
        "graph_type": "source_actor_graph",
        "truth_source": "relation_dsl",
        "vehicle_categories": sorted(VEHICLE_CATEGORIES),
        "actors": actors,
        "ignored_actors": ignored,
        "metadata": {
            "actor_count": len(actors),
            "duplicate_ids": duplicate_ids,
            "id_chain_valid": not duplicate_ids,
        },
    }


def build_render_actor_graph_from_spawn_payload(
    spawn_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a limited render graph when CARLA truth is unavailable."""
    actors = []
    ignored = []
    for entity in _coerce_list(spawn_payload.get("entities")):
        actor_id = str(entity.get("id") or "")
        if actor_id in {"ego", "ego_vehicle"}:
            continue
        actor = {
            "id": actor_id,
            "category": _category(entity.get("category") or "car"),
            "subtype": str(entity.get("subtype") or entity.get("category") or ""),
            "spawn_kind": str(entity.get("spawn_kind") or "vehicle"),
            "blueprint_name": entity.get("blueprint_name"),
            "placement_mode": str(entity.get("placement_mode") or "project_to_lane"),
            "projected_lane": _coerce_dict(entity.get("projected_lane")),
            "lane_index_relation": _coerce_lane_index(
                entity.get("lane_index_relation"),
                entity.get("lane_side_relation"),
            ),
            "lane_side_relation": _lane_side_from_index(
                _coerce_lane_index(entity.get("lane_index_relation"), entity.get("lane_side_relation"))
            ),
            "heading_relation_to_ego": str(entity.get("heading_relation") or "unknown"),
            "location": _coerce_dict(entity.get("location")),
            "yaw": _coerce_dict(entity.get("rotation")).get("yaw"),
            "spawned": True,
            "spawn_failure_reason": None,
            "truth_source": FALLBACK_TRUTH_SOURCE,
        }
        if actor_id and _is_vehicle_actor(actor):
            actors.append(actor)
        elif actor_id:
            ignored.append(actor)

    return {
        "schema_version": "actor-graph-v1",
        "graph_type": "render_actor_graph",
        "truth_source": FALLBACK_TRUTH_SOURCE,
        "position_checks_enabled": False,
        "spawn_checks_enabled": False,
        "actors": actors,
        "ignored_actors": ignored,
        "metadata": {
            "actor_count": len(actors),
            "truth_unavailable": True,
        },
    }


def build_render_actor_graph_from_carla_records(
    spawn_payload: Dict[str, Any],
    actor_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a render graph from generated-script CARLA actor transform records."""
    ego = actor_records.get("ego") or actor_records.get("ego_vehicle") or {}
    actors = []
    ignored = []
    for entity in _coerce_list(spawn_payload.get("entities")):
        actor_id = str(entity.get("id") or "")
        if actor_id in {"ego", "ego_vehicle"}:
            continue
        category = _category(entity.get("category") or "car")
        record = actor_records.get(actor_id)
        actor = {
            "id": actor_id,
            "category": category,
            "subtype": str(entity.get("subtype") or entity.get("category") or ""),
            "spawn_kind": str(entity.get("spawn_kind") or "vehicle"),
            "blueprint_name": entity.get("blueprint_name"),
            "placement_mode": str(entity.get("placement_mode") or "project_to_lane"),
            "projected_lane": _coerce_dict(entity.get("projected_lane")),
            "lane_index_relation": _coerce_lane_index(
                entity.get("lane_index_relation"),
                entity.get("lane_side_relation"),
            ),
            "lane_side_relation": _lane_side_from_index(
                _coerce_lane_index(entity.get("lane_index_relation"), entity.get("lane_side_relation"))
            ),
            "truth_source": CARLA_TRUTH_SOURCE,
        }
        if record:
            loc = _coerce_dict(record.get("location"))
            yaw = record.get("yaw")
            actor.update({
                "spawned": True,
                "spawn_failure_reason": None,
                "location": loc,
                "yaw": yaw,
            })
            if isinstance(record.get("actual_waypoint"), dict):
                actor["actual_waypoint"] = record.get("actual_waypoint")
            longitudinal, lateral = _relative_metrics({"location": loc}, ego)
            actor["ego_frame"] = {
                "longitudinal_m": round(longitudinal, 3),
                "lateral_m": round(lateral, 3),
                "distance_band": _distance_band_from_longitudinal(longitudinal),
                "lane_index_relation": _lane_index_from_lateral(lateral),
                "lane_side_relation": _lane_side_from_lateral(lateral),
            }
            actor["heading_relation_to_ego"] = _heading_relation(yaw, ego.get("yaw"))
        else:
            actor.update({
                "spawned": False,
                "spawn_failure_reason": "try_spawn_actor_returned_none_or_collision",
            })
        if actor_id and category in VEHICLE_CATEGORIES:
            actors.append(actor)
        elif actor_id:
            ignored.append(actor)

    return {
        "schema_version": "actor-graph-v1",
        "graph_type": "render_actor_graph",
        "truth_source": CARLA_TRUTH_SOURCE,
        "position_checks_enabled": True,
        "spawn_checks_enabled": True,
        "actors": actors,
        "ignored_actors": ignored,
        "metadata": {
            "actor_count": len(actors),
            "truth_unavailable": False,
        },
    }


def _actors_by_id(graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(actor.get("id")): actor for actor in _coerce_list(graph.get("actors")) if actor.get("id")}


def _issue(
    issue_type: str,
    target_artifact: str,
    operation: str,
    source_actor: Optional[Dict[str, Any]],
    render_actor: Optional[Dict[str, Any]],
    severity: str,
    evidence: str,
    direction: Optional[str] = None,
    reference_entity_id: Optional[str] = None,
    min_spacing_m: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "issue_type": issue_type,
        "target_artifact": target_artifact,
        "operation": operation,
        "source_entity_id": (source_actor or {}).get("id"),
        "render_entity_id": (render_actor or {}).get("id"),
        "severity": severity,
        "evidence": evidence,
        **({"direction": direction} if direction else {}),
        **({"reference_entity_id": reference_entity_id} if reference_entity_id else {}),
        **({"min_spacing_m": min_spacing_m} if min_spacing_m is not None else {}),
    }


def _repair_action_from_issue(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    issue_type = item.get("issue_type")
    source_id = item.get("source_entity_id") or item.get("render_entity_id")
    severity = item.get("severity") or "medium"
    if issue_type == "lane_side_mismatch" and source_id:
        return {
            "type": "lane_side_mismatch",
            "entity_id": source_id,
            "severity": severity,
            "evidence": item.get("evidence", ""),
        }
    if issue_type == "junction_lane_mismatch" and source_id:
        return {
            "type": "junction_lane_mismatch",
            "entity_id": source_id,
            "severity": severity,
            "evidence": item.get("evidence", ""),
            "expected_lane": item.get("expected_lane") or {},
            "actual_waypoint": item.get("actual_waypoint") or {},
        }
    if issue_type == "heading_mismatch" and source_id:
        return {
            "type": "heading_mismatch",
            "entity_id": source_id,
            "severity": severity,
            "evidence": item.get("evidence", ""),
        }
    if issue_type == "longitudinal_mismatch" and source_id:
        direction = str(item.get("direction") or "")
        return {
            "type": "longitudinal_mismatch",
            "entity_id": source_id,
            "severity": severity,
            "direction": direction,
            "delta_m": 6.0 if "far" in direction or "mid" in direction else 3.0,
            "evidence": item.get("evidence", ""),
        }
    if issue_type == "overlap" and source_id:
        return {
            "type": "overlap",
            "entity_id": source_id,
            **(
                {"reference_entity_id": item.get("reference_entity_id")}
                if item.get("reference_entity_id")
                else {}
            ),
            **(
                {"min_spacing_m": item.get("min_spacing_m")}
                if item.get("min_spacing_m") is not None
                else {}
            ),
            "severity": severity,
            "evidence": item.get("evidence", ""),
        }
    if issue_type in {"count_mismatch", "category_mismatch", "missing_actor"}:
        return {
            "type": "count_mismatch" if issue_type != "category_mismatch" else "category_mismatch",
            "entity_id": source_id,
            "severity": severity,
            "evidence": item.get("evidence", ""),
        }
    return None


def _same_previous_issue(current: Dict[str, Any], previous: Dict[str, Any]) -> bool:
    return (
        current.get("issue_type") == previous.get("issue_type")
        and current.get("source_entity_id") == previous.get("source_entity_id")
        and current.get("direction") == previous.get("direction")
    )


def _detect_systematic_issues(
    issues: List[Dict[str, Any]],
    previous_plan: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    grouped: Dict[Tuple[str, str], set] = defaultdict(set)
    for item in issues:
        direction = str(item.get("direction") or "")
        if not direction:
            continue
        grouped[(str(item.get("issue_type")), direction)].add(str(item.get("source_entity_id")))
    for (issue_type, direction), entity_ids in grouped.items():
        if len(entity_ids) >= 3:
            return True, f"{issue_type}:{direction}:multiple_entities"

    previous_issues = _coerce_list((previous_plan or {}).get("issues"))
    for item in issues:
        if any(_same_previous_issue(item, prev) for prev in previous_issues):
            return True, f"{item.get('issue_type')}:{item.get('source_entity_id')}:repeated"
    return False, None


def _actor_min_spacing(actor: Dict[str, Any]) -> float:
    category = _category(actor.get("category"))
    if category in {"truck", "bus"}:
        return 7.0
    if category == "car":
        return 4.5
    if category in {"motorcycle", "bicycle"}:
        return 2.5
    return 2.0


def _overlap_issues(render_actors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    for index, actor in enumerate(render_actors):
        if not actor.get("spawned", True):
            continue
        loc = _coerce_dict(actor.get("location"))
        if "x" not in loc or "y" not in loc:
            continue
        for other in render_actors[:index]:
            if not other.get("spawned", True):
                continue
            other_loc = _coerce_dict(other.get("location"))
            if "x" not in other_loc or "y" not in other_loc:
                continue
            distance = math.hypot(
                float(loc.get("x", 0.0)) - float(other_loc.get("x", 0.0)),
                float(loc.get("y", 0.0)) - float(other_loc.get("y", 0.0)),
            )
            min_spacing = max(_actor_min_spacing(actor), _actor_min_spacing(other))
            if distance < min_spacing:
                issues.append(_issue(
                    "overlap",
                    "spawn_payload",
                    "resolve_overlap",
                    actor,
                    actor,
                    "high",
                    (
                        f"{actor.get('id')} is {distance:.2f}m from {other.get('id')}, "
                        f"below minimum vehicle spacing {min_spacing:.2f}m"
                    ),
                    reference_entity_id=str(other.get("id") or ""),
                    min_spacing_m=min_spacing,
                ))
    return issues


def compare_actor_graphs(
    source_graph: Dict[str, Any],
    render_graph: Dict[str, Any],
    validation: Optional[Dict[str, Any]] = None,
    previous_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare expected and rendered actor graphs and return a repair plan."""
    del validation  # Reserved for later routing refinements.
    truth_source = str(render_graph.get("truth_source") or FALLBACK_TRUTH_SOURCE)
    position_checks = bool(render_graph.get("position_checks_enabled", True))
    spawn_checks = bool(render_graph.get("spawn_checks_enabled", True))
    source_by_id = _actors_by_id(source_graph)
    render_by_id = _actors_by_id(render_graph)
    issues: List[Dict[str, Any]] = []

    if truth_source == FALLBACK_TRUTH_SOURCE:
        issues.append({
            "issue_type": "truth_unavailable",
            "target_artifact": "none",
            "operation": "static_check_only",
            "source_entity_id": None,
            "render_entity_id": None,
            "severity": "info",
            "evidence": "CARLA actor transform truth is unavailable; position and spawn checks are disabled.",
        })

    duplicate_source_ids = _coerce_list(_coerce_dict(source_graph.get("metadata")).get("duplicate_ids"))
    duplicate_render_ids = [
        actor_id for actor_id, count in Counter(
            str(actor.get("id")) for actor in _coerce_list(render_graph.get("actors"))
        ).items()
        if actor_id and count > 1
    ]
    if duplicate_source_ids or duplicate_render_ids:
        issues.append({
            "issue_type": "id_chain_mismatch",
            "target_artifact": "projection_logic",
            "operation": "inspect_id_chain",
            "source_entity_id": None,
            "render_entity_id": None,
            "severity": "high",
            "evidence": f"Duplicate ids source={duplicate_source_ids} render={duplicate_render_ids}",
        })

    source_counts = Counter(actor.get("category") for actor in source_by_id.values())
    render_counts = Counter(actor.get("category") for actor in render_by_id.values())
    if source_counts != render_counts:
        issues.append(_issue(
            "count_mismatch",
            "scene_understanding",
            "revise_scene_understanding",
            None,
            None,
            "high",
            f"category counts differ: source={dict(source_counts)} render={dict(render_counts)}",
        ))

    for actor_id, source_actor in source_by_id.items():
        render_actor = render_by_id.get(actor_id)
        if render_actor is None:
            issues.append(_issue(
                "missing_actor",
                "scene_understanding",
                "revise_scene_understanding",
                source_actor,
                None,
                "high",
                f"{actor_id} exists in source graph but not render graph",
            ))
            continue
        if spawn_checks and not render_actor.get("spawned", True):
            issues.append(_issue(
                "spawn_failure",
                "projection_logic",
                "blocked_for_code_fix",
                source_actor,
                render_actor,
                "high",
                f"{actor_id} failed to spawn: {render_actor.get('spawn_failure_reason')}",
            ))
            continue
        if source_actor.get("category") != render_actor.get("category"):
            issues.append(_issue(
                "category_mismatch",
                "scene_understanding",
                "revise_scene_understanding",
                source_actor,
                render_actor,
                "high",
                f"{actor_id} category source={source_actor.get('category')} render={render_actor.get('category')}",
            ))
        if not position_checks:
            continue
        placement_mode = str(render_actor.get("placement_mode") or "")
        if placement_mode == "project_to_junction_lane":
            expected_lane = _coerce_dict(render_actor.get("projected_lane"))
            actual_waypoint = _coerce_dict(render_actor.get("actual_waypoint"))
            expected_road = expected_lane.get("road_id")
            expected_lane_id = expected_lane.get("lane_id")
            actual_road = actual_waypoint.get("road_id")
            actual_lane_id = actual_waypoint.get("lane_id")
            expected_yaw = expected_lane.get("yaw")
            actual_yaw = actual_waypoint.get("yaw")
            lane_matches = (
                expected_road is not None
                and expected_lane_id is not None
                and str(expected_road) == str(actual_road)
                and str(expected_lane_id) == str(actual_lane_id)
            )
            yaw_matches = True
            try:
                if expected_yaw is not None and actual_yaw is not None:
                    yaw_matches = _angle_distance(float(expected_yaw), float(actual_yaw)) <= 35.0
            except Exception:
                yaw_matches = True
            if expected_lane and (not lane_matches or not yaw_matches):
                issues.append({
                    "issue_type": "junction_lane_mismatch",
                    "target_artifact": "spawn_payload",
                    "operation": "repair_junction_lane_seed",
                    "source_entity_id": source_actor.get("id"),
                    "render_entity_id": render_actor.get("id"),
                    "severity": "high",
                    "evidence": (
                        f"{actor_id} projected junction lane expected "
                        f"road/lane/yaw={expected_road}/{expected_lane_id}/{expected_yaw} "
                        f"actual={actual_road}/{actual_lane_id}/{actual_yaw}"
                    ),
                    "expected_lane": expected_lane,
                    "actual_waypoint": actual_waypoint,
                })
        render_ego = _coerce_dict(render_actor.get("ego_frame"))
        render_lane_index = _coerce_lane_index(
            render_ego.get("lane_index_relation"),
            render_ego.get("lane_side_relation") or render_actor.get("lane_side_relation"),
        )
        source_lane_index = _coerce_lane_index(
            source_actor.get("lane_index_relation"),
            source_actor.get("lane_side_relation"),
        )
        source_heading = str(source_actor.get("heading_relation") or "unknown")
        if source_heading == "opposite_direction":
            # Oncoming actors live on the opposing carriageway across the median.
            # Their exact lane index is not meaningful in the ego frame (the
            # divider adds lateral distance the VLM cannot count), so validate by
            # *side* instead: in right-hand traffic the opposing carriageway is
            # always to ego's left (negative lateral / negative lane index).
            render_lateral = float(render_ego.get("lateral_m", 0.0) or 0.0)
            on_opposing_side = render_lane_index < 0 or render_lateral < -1.25
            if not on_opposing_side:
                placement = str(render_actor.get("placement_mode") or "")
                target = "spawn_payload" if placement in {"direct", "preserve_xy"} else "projection_logic"
                operation = "shift_lateral" if target == "spawn_payload" else "blocked_for_code_fix"
                issues.append(_issue(
                    "lane_side_mismatch",
                    target,
                    operation,
                    source_actor,
                    render_actor,
                    "medium",
                    f"{actor_id} oncoming actor not on opposing side "
                    f"(render lane index={render_lane_index}, lateral={render_lateral:.2f})",
                    direction=f"{source_lane_index}->{render_lane_index}",
                ))
        elif source_lane_index != render_lane_index:
            placement = str(render_actor.get("placement_mode") or "")
            target = "spawn_payload" if placement in {"direct", "preserve_xy"} else "projection_logic"
            operation = "shift_lateral" if target == "spawn_payload" else "blocked_for_code_fix"
            issues.append(_issue(
                "lane_side_mismatch",
                target,
                operation,
                source_actor,
                render_actor,
                "medium",
                f"{actor_id} lane index source={source_lane_index} render={render_lane_index}",
                direction=f"{source_lane_index}->{render_lane_index}",
            ))
        source_band = str(source_actor.get("distance_band") or "")
        render_band = str(render_ego.get("distance_band") or "")
        if source_band and render_band and source_band != render_band:
            issues.append(_issue(
                "longitudinal_mismatch",
                "spawn_payload",
                "shift_longitudinal",
                source_actor,
                render_actor,
                "medium",
                f"{actor_id} distance band source={source_band} render={render_band}",
                direction=f"{source_band}->{render_band}",
            ))
        render_heading = str(render_actor.get("heading_relation_to_ego") or "unknown")
        if (
            source_heading != "unknown"
            and render_heading != "unknown"
            and source_heading != render_heading
        ):
            issues.append(_issue(
                "heading_mismatch",
                "spawn_payload",
                "flip_heading",
                source_actor,
                render_actor,
                "medium",
                f"{actor_id} heading source={source_heading} render={render_heading}",
                direction=f"{source_heading}->{render_heading}",
            ))

    for actor_id, render_actor in render_by_id.items():
        if actor_id not in source_by_id:
            issues.append(_issue(
                "extra_actor",
                "spawn_payload",
                "remove_or_ignore_actor",
                None,
                render_actor,
                "medium",
                f"{actor_id} exists in render graph but not source graph",
            ))

    # Even without live CARLA truth, the spawn payload can expose impossible
    # initial layouts such as two vehicles sharing the same coordinates.
    issues.extend(_overlap_issues(list(render_by_id.values())))

    systematic_error, systematic_reason = _detect_systematic_issues(issues, previous_plan)
    if systematic_error:
        issues.append({
            "issue_type": "systematic_projection_error",
            "target_artifact": "projection_logic",
            "operation": "blocked_for_code_fix",
            "source_entity_id": None,
            "render_entity_id": None,
            "severity": "high",
            "evidence": systematic_reason,
        })

    hard_issues = [
        item for item in issues
        if item.get("issue_type") not in {"truth_unavailable"}
    ]
    requires_code_fix = systematic_error or any(
        item.get("target_artifact") == "projection_logic" for item in hard_issues
    )
    repair_actions = [
        action for action in (_repair_action_from_issue(item) for item in hard_issues)
        if action is not None
    ]
    if truth_source == FALLBACK_TRUTH_SOURCE:
        status = "static_checked" if not hard_issues else "static_failed"
        passed = False
        score = None
    else:
        passed = not hard_issues
        status = "passed" if passed else "failed"
        score = 1.0 if passed else max(0.0, 1.0 - 0.12 * len(hard_issues))
    return {
        "status": status,
        "passed": passed,
        "score": score,
        "truth_source": truth_source,
        "truth_unavailable": truth_source == FALLBACK_TRUTH_SOURCE,
        "systematic_error": systematic_error,
        "systematic_reason": systematic_reason,
        "requires_code_fix": requires_code_fix,
        "stop_repair_loop": requires_code_fix,
        "issues": issues,
        "repair_actions": repair_actions,
    }
