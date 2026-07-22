"""Immutable user-text constraints for the static reconstruction pipeline.

The scene-understanding JSON is an editable intermediate representation: map
matching, normalization and repair are all allowed to refine it.  Explicit user
statements are different.  This module keeps them in a small side-channel and
re-applies them deterministically so an heuristic cannot silently replace a
user-provided value.

Nothing in this module is enabled unless ``metadata.user_constraints`` exists,
which keeps image-only and non-static pipelines behaviorally unchanged.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


USER_CONSTRAINT_SCHEMA_VERSION = "user-constraints-v1"
_PATH_PART = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")
_MISSING = object()


def _path_tokens(path: str) -> Optional[List[Tuple[str, Optional[int]]]]:
    tokens: List[Tuple[str, Optional[int]]] = []
    for raw in str(path or "").split("."):
        match = _PATH_PART.fullmatch(raw)
        if match is None:
            return None
        index = int(match.group(2)) if match.group(2) is not None else None
        tokens.append((match.group(1), index))
    return tokens or None


def get_path(payload: Any, path: str, default: Any = None) -> Any:
    current = payload
    tokens = _path_tokens(path)
    if tokens is None:
        return default
    for key, index in tokens:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
        if index is not None:
            if not isinstance(current, list) or index >= len(current):
                return default
            current = current[index]
    return current


def set_path(payload: Dict[str, Any], path: str, value: Any) -> bool:
    tokens = _path_tokens(path)
    if tokens is None or not isinstance(payload, dict):
        return False
    current: Any = payload
    for position, (key, index) in enumerate(tokens):
        is_last = position == len(tokens) - 1
        if not isinstance(current, dict):
            return False
        if index is None:
            if is_last:
                current[key] = deepcopy(value)
                return True
            if not isinstance(current.get(key), dict):
                current[key] = {}
            current = current[key]
            continue

        if not isinstance(current.get(key), list):
            current[key] = []
        values = current[key]
        while len(values) <= index:
            values.append({})
        if is_last:
            values[index] = deepcopy(value)
            return True
        if not isinstance(values[index], dict):
            values[index] = {}
        current = values[index]
    return False


def normalize_user_constraint_bundle(raw: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and canonicalize the merger-produced constraint side-channel."""

    if isinstance(raw, list):
        raw = {"constraints": raw}
    if not isinstance(raw, dict):
        return {
            "schema_version": USER_CONSTRAINT_SCHEMA_VERSION,
            "source": "user_text",
            "constraints": [],
        }, ["metadata.user_constraints must be an object"]

    constraints = raw.get("constraints")
    if not isinstance(constraints, list):
        return {
            "schema_version": USER_CONSTRAINT_SCHEMA_VERSION,
            "source": "user_text",
            "constraints": [],
        }, ["metadata.user_constraints.constraints must be a list"]

    normalized: List[Dict[str, Any]] = []
    errors: List[str] = []
    by_key: Dict[Tuple[str, str, str, str], int] = {}
    for index, item in enumerate(constraints):
        if not isinstance(item, dict):
            errors.append(f"constraint[{index}] must be an object")
            continue
        target = str(item.get("target") or "").strip().lower()
        path = str(item.get("path") or "").strip()
        strength = str(item.get("strength") or "hard").strip().lower()
        entity_id = str(item.get("entity_id") or "").strip()
        reference_entity_id = str(item.get("reference_entity_id") or "").strip()
        if target not in {"scene", "entity", "relation"}:
            errors.append(f"constraint[{index}] has unsupported target {target!r}")
            continue
        if _path_tokens(path) is None or path.startswith("metadata.user_constraints"):
            errors.append(f"constraint[{index}] has invalid path {path!r}")
            continue
        if target in {"entity", "relation"} and not entity_id:
            errors.append(f"constraint[{index}] requires entity_id")
            continue
        if target == "relation" and not reference_entity_id:
            errors.append(f"constraint[{index}] requires reference_entity_id")
            continue
        if "value" not in item:
            errors.append(f"constraint[{index}] requires value")
            continue
        if strength not in {"hard", "soft"}:
            strength = "hard"
        constraint = {
            "id": str(item.get("id") or f"user_constraint_{index}"),
            "target": target,
            "path": path,
            "value": deepcopy(item.get("value")),
            "source": "user_text",
            "strength": strength,
            "evidence": str(item.get("evidence") or "explicit user statement").strip(),
        }
        if entity_id:
            constraint["entity_id"] = entity_id
        if reference_entity_id:
            constraint["reference_entity_id"] = reference_entity_id
        if isinstance(item.get("tolerance"), (int, float)):
            constraint["tolerance"] = max(0.0, float(item["tolerance"]))
        key = (target, entity_id, reference_entity_id, path)
        if key in by_key:
            normalized[by_key[key]] = constraint
        else:
            by_key[key] = len(normalized)
            normalized.append(constraint)

    return {
        "schema_version": USER_CONSTRAINT_SCHEMA_VERSION,
        "source": "user_text",
        "constraints": normalized,
    }, errors


def user_constraints(scene_understanding: Dict[str, Any], *, hard_only: bool = False) -> List[Dict[str, Any]]:
    metadata = scene_understanding.get("metadata") or {}
    raw = metadata.get("user_constraints")
    if raw is None:
        return []
    bundle, _errors = normalize_user_constraint_bundle(raw)
    constraints = bundle["constraints"]
    if hard_only:
        constraints = [item for item in constraints if item.get("strength") == "hard"]
    return constraints


def hard_scene_constraint_value(
    scene_understanding: Dict[str, Any],
    *paths: str,
) -> Any:
    wanted = set(paths)
    for constraint in reversed(user_constraints(scene_understanding, hard_only=True)):
        if constraint.get("target") == "scene" and constraint.get("path") in wanted:
            return deepcopy(constraint.get("value"))
    return _MISSING


def has_hard_scene_constraint(scene_understanding: Dict[str, Any], *paths: str) -> bool:
    return hard_scene_constraint_value(scene_understanding, *paths) is not _MISSING


def hard_entity_constraint_values(
    scene_understanding: Dict[str, Any], entity_id: str
) -> Dict[str, Any]:
    """Return final semantic values locked by hard constraints for one actor."""

    values: Dict[str, Any] = {}
    for constraint in user_constraints(scene_understanding, hard_only=True):
        if constraint.get("target") != "entity":
            continue
        if str(constraint.get("entity_id") or "") != str(entity_id):
            continue
        values[str(constraint.get("path") or "")] = deepcopy(constraint.get("value"))
    entity = _entity_by_id(scene_understanding, entity_id)
    if entity is not None and any(
        key in values for key in ("anchor_relation.lane_from_right", "lane_from_right")
    ):
        values["lane_side_relation"] = entity.get("lane_side_relation")
        values["lane_index_relation"] = entity.get("lane_index_relation")
    return values


def _entities(scene_understanding: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("traffic_subjects", "background_traffic"):
        for entity in scene_understanding.get(key) or []:
            if isinstance(entity, dict):
                yield entity


def _entity_by_id(scene_understanding: Dict[str, Any], entity_id: str) -> Optional[Dict[str, Any]]:
    for entity in _entities(scene_understanding):
        if str(entity.get("id") or "") == str(entity_id):
            return entity
    return None


def _lane_side_from_index(index: int) -> str:
    if index < 0:
        return "left_lane"
    if index > 0:
        return "right_lane"
    return "same_lane"


def _synchronize_known_aliases(scene_understanding: Dict[str, Any]) -> None:
    ego_path = "road_network.map_matching.ego_lane_from_right"
    ego_lane = hard_scene_constraint_value(
        scene_understanding,
        ego_path,
        "actor_layout.ego_approach.ego_lane_from_right",
        "metadata.ego_localization.ego_lane_from_right",
    )
    if ego_lane is _MISSING:
        return
    try:
        ego_lane = max(0, int(ego_lane))
    except (TypeError, ValueError):
        return
    set_path(scene_understanding, ego_path, ego_lane)
    set_path(scene_understanding, "actor_layout.ego_approach.ego_lane_from_right", ego_lane)
    set_path(scene_understanding, "metadata.ego_localization.ego_lane_from_right", ego_lane)
    set_path(scene_understanding, "metadata.ego_localization.ego_lane_source", "user_text")

    # Absolute lane slots are counted from the right.  The regular actor DSL
    # uses ego-relative signs (+right, -left), hence ego_slot - actor_slot.
    constrained_actor_slots: Dict[str, int] = {}
    for constraint in user_constraints(scene_understanding, hard_only=True):
        if constraint.get("target") != "entity":
            continue
        if constraint.get("path") not in {
            "anchor_relation.lane_from_right",
            "lane_from_right",
        }:
            continue
        try:
            constrained_actor_slots[str(constraint.get("entity_id"))] = int(
                constraint.get("value")
            )
        except (TypeError, ValueError):
            continue
    for entity_id, actor_slot in constrained_actor_slots.items():
        entity = _entity_by_id(scene_understanding, entity_id)
        if entity is None:
            continue
        relative_index = ego_lane - actor_slot
        entity.setdefault("anchor_relation", {})["lane_from_right"] = actor_slot
        entity["lane_index_relation"] = relative_index
        entity["lane_side_relation"] = _lane_side_from_index(relative_index)


def apply_user_constraints_to_scene(
    scene_understanding: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Re-apply explicit text facts after scene normalization/merging."""

    if not isinstance(scene_understanding, dict):
        return scene_understanding, {"enabled": False, "reason": "invalid_scene"}
    metadata = scene_understanding.setdefault("metadata", {})
    if "user_constraints" not in metadata:
        return scene_understanding, {"enabled": False, "reason": "no_user_constraints"}
    bundle, errors = normalize_user_constraint_bundle(metadata.get("user_constraints"))
    metadata["user_constraints"] = bundle
    applied: List[str] = []
    unresolved: List[str] = []
    for constraint in bundle["constraints"]:
        target = constraint["target"]
        if target == "scene":
            success = set_path(
                scene_understanding,
                constraint["path"],
                constraint["value"],
            )
        elif target == "entity":
            entity = _entity_by_id(scene_understanding, constraint.get("entity_id", ""))
            success = entity is not None and set_path(
                entity,
                constraint["path"],
                constraint["value"],
            )
        else:
            relation = None
            for candidate in scene_understanding.setdefault("key_pairwise_relations", []):
                if not isinstance(candidate, dict):
                    continue
                if (
                    str(candidate.get("entity_id") or "") == constraint.get("entity_id")
                    and str(candidate.get("other_entity_id") or "")
                    == constraint.get("reference_entity_id")
                ):
                    relation = candidate
                    break
            if relation is None:
                relation = {
                    "id": f"{constraint['id']}_relation",
                    "entity_id": constraint.get("entity_id"),
                    "other_entity_id": constraint.get("reference_entity_id"),
                    "constraint_strength": constraint.get("strength", "hard"),
                    "confidence": "user",
                    "evidence": constraint.get("evidence", ""),
                    "relation_source": "user_text",
                }
                scene_understanding["key_pairwise_relations"].append(relation)
            success = set_path(relation, constraint["path"], constraint["value"])
            relation["relation_source"] = "user_text"
        (applied if success else unresolved).append(constraint["id"])

    _synchronize_known_aliases(scene_understanding)
    report = {
        "enabled": True,
        "schema_version": USER_CONSTRAINT_SCHEMA_VERSION,
        "applied": applied,
        "unresolved": unresolved,
        "schema_errors": errors,
    }
    metadata["user_constraint_application"] = report
    return scene_understanding, report


def apply_user_constraints_to_relation_dsl(
    scene_understanding: Dict[str, Any],
    relation_dsl: Dict[str, Any],
) -> Dict[str, Any]:
    """Carry constrained actor semantics into the coordinate-generation IR."""

    constraints = user_constraints(scene_understanding, hard_only=True)
    if not constraints:
        return relation_dsl
    entities = relation_dsl.get("ego_relations") or relation_dsl.get("entities") or []
    for constraint in constraints:
        if constraint.get("target") != "entity":
            continue
        entity_id = str(constraint.get("entity_id") or "")
        path = str(constraint.get("path") or "")
        value = deepcopy(constraint.get("value"))
        for entity in entities:
            if str(entity.get("entity_id") or "") != entity_id and str(
                entity.get("group_id") or ""
            ) != entity_id:
                continue
            if path == "heading_relation_to_ego":
                entity["heading_relation"] = value
                entity["original_heading_relation"] = value
            elif path == "longitudinal_relation":
                entity["order_relation"] = value
            elif path == "longitudinal_proximity":
                entity["distance_band"] = value
            elif path == "longitudinal_m":
                try:
                    entity["user_target_longitudinal_m"] = float(value)
                except (TypeError, ValueError):
                    pass
            elif path == "lane_side_relation":
                entity["lane_side_relation"] = value
            elif path == "lane_index_relation":
                try:
                    entity["lane_index_relation"] = int(value)
                except (TypeError, ValueError):
                    pass
            else:
                set_path(entity, path, value)
            entity.setdefault("user_constraint_ids", []).append(constraint["id"])
    relation_dsl.setdefault("metadata", {})["user_constraints"] = deepcopy(
        (scene_understanding.get("metadata") or {}).get("user_constraints")
    )
    return relation_dsl


def _equal(expected: Any, actual: Any, tolerance: float = 0.0) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), abs_tol=tolerance)
    return str(expected) == str(actual)


def _payload_actors(payload: Dict[str, Any], entity_id: str) -> List[Dict[str, Any]]:
    return [
        actor
        for actor in payload.get("entities") or []
        if str(actor.get("id") or "") == entity_id
        or str(actor.get("actor_group_id") or "") == entity_id
    ]


def _effective_actors(
    payload: Dict[str, Any],
    render_graph: Dict[str, Any],
    entity_id: str,
) -> List[Dict[str, Any]]:
    payload_actors = _payload_actors(payload, entity_id)
    payload_ids = {str(actor.get("id") or "") for actor in payload_actors}
    render_actors = [
        actor
        for actor in render_graph.get("actors") or []
        if (
            str(actor.get("id") or "") == entity_id
            or str(actor.get("id") or "") in payload_ids
        )
        and actor.get("spawned") is not False
    ]
    if entity_id in {"ego", "ego_vehicle"} and isinstance(render_graph.get("ego"), dict):
        render_actors = [{"id": "ego", **render_graph["ego"]}]
    if not render_actors:
        if render_graph.get("truth_source") == "carla_actor_transform":
            return []
        return payload_actors
    payload_by_id = {str(actor.get("id") or ""): actor for actor in payload_actors}
    effective = []
    for rendered in render_actors:
        merged = deepcopy(payload_by_id.get(str(rendered.get("id") or "")) or {})
        merged.update(deepcopy(rendered))
        effective.append(merged)
    return effective


def _actual_scene_value(
    path: str,
    scene_understanding: Dict[str, Any],
    match_report: Dict[str, Any],
    spawn_context: Dict[str, Any],
) -> Tuple[Any, str]:
    features = ((match_report.get("best_match") or {}).get("candidate_features") or {})
    if path.endswith("ego_lane_from_right"):
        lane = ((spawn_context.get("topology_sample") or [{}])[0] or {})
        return lane.get("resolved_ego_lane_from_right", _MISSING), "carla_lane_anchor"
    if path.endswith("forward_lane_count"):
        return features.get("same_direction_lane_count", _MISSING), "map_candidate"
    if path.endswith("topology_type"):
        return features.get("candidate_topology_type", _MISSING), "map_candidate"
    if path.endswith("has_center_median"):
        return features.get("has_center_median_candidate", _MISSING), "map_candidate"
    return get_path(scene_understanding, path, _MISSING), "resolved_scene"


def _actual_actor_value(actor: Dict[str, Any], path: str) -> Any:
    aliases = {
        "heading_relation_to_ego": "heading_relation",
        "anchor_relation.lane_from_right": "lane_from_right",
        "longitudinal_m": "longitudinal_m",
    }
    if path == "appearance.color":
        appearance = actor.get("appearance") or {}
        return appearance.get("color", actor.get("color", _MISSING))
    if path == "longitudinal_m":
        return (actor.get("ego_frame") or {}).get(
            "longitudinal_m", actor.get("longitudinal_m", _MISSING)
        )
    if path == "longitudinal_proximity":
        return (actor.get("ego_frame") or {}).get(
            "distance_band", actor.get("longitudinal_proximity", _MISSING)
        )
    if path == "lane_side_relation":
        return (actor.get("ego_frame") or {}).get(
            "lane_side_relation", actor.get("lane_side_relation", _MISSING)
        )
    if path in {"anchor_relation.lane_from_right", "lane_from_right"}:
        lane_from_right = actor.get("lane_from_right")
        if lane_from_right is None:
            lane_from_right = actor.get("junction_lane_from_right", _MISSING)
        return lane_from_right
    return get_path(actor, aliases.get(path, path), _MISSING)


def evaluate_user_constraints(
    scene_understanding: Dict[str, Any],
    spawn_payload: Dict[str, Any],
    match_report: Optional[Dict[str, Any]] = None,
    spawn_context: Optional[Dict[str, Any]] = None,
    render_graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate hard text facts against the final spawn payload/map anchor."""

    constraints = user_constraints(scene_understanding, hard_only=True)
    results: List[Dict[str, Any]] = []
    match_report = match_report or {}
    spawn_context = spawn_context or {}
    render_graph = render_graph or {}
    for constraint in constraints:
        target = constraint["target"]
        expected = constraint.get("value")
        tolerance = float(
            constraint.get("tolerance")
            if isinstance(constraint.get("tolerance"), (int, float))
            else 1.5 if constraint.get("path") == "longitudinal_m" else 0.0
        )
        actual: Any = _MISSING
        source = "unsupported"
        if target == "scene":
            actual, source = _actual_scene_value(
                constraint["path"], scene_understanding, match_report, spawn_context
            )
        elif target == "entity":
            actors = _effective_actors(
                spawn_payload,
                render_graph,
                str(constraint.get("entity_id") or ""),
            )
            source = (
                "carla_actor_transform"
                if render_graph.get("truth_source") == "carla_actor_transform"
                else "spawn_payload"
            )
            if constraint["path"] in {"count", "representative_count"}:
                actual = len(actors)
            elif actors:
                values = [_actual_actor_value(actor, constraint["path"]) for actor in actors]
                actual = values[0] if len(values) == 1 else values
        elif target == "relation":
            actors = _effective_actors(
                spawn_payload, render_graph, str(constraint.get("entity_id") or "")
            )
            references = _effective_actors(
                spawn_payload,
                render_graph,
                str(constraint.get("reference_entity_id") or ""),
            )
            source = (
                "carla_actor_transform_relation"
                if render_graph.get("truth_source") == "carla_actor_transform"
                else "spawn_payload_relation"
            )
            if actors and references:
                path = constraint["path"]
                if path == "lane_relation" and constraint.get("reference_entity_id") == "ego":
                    side = str(actors[0].get("lane_side_relation") or "")
                    actual = {
                        "same_lane": "same_lane",
                        "left_lane": "adjacent_left_lane",
                        "right_lane": "adjacent_right_lane",
                    }.get(side, side)
                elif path == "longitudinal_distance_m":
                    actor_m = _actual_actor_value(actors[0], "longitudinal_m")
                    reference_m = _actual_actor_value(references[0], "longitudinal_m")
                    if reference_m is _MISSING and constraint.get("reference_entity_id") == "ego":
                        reference_m = 0.0
                    if isinstance(actor_m, (int, float)) and isinstance(reference_m, (int, float)):
                        actual = float(actor_m) - float(reference_m)
                else:
                    actual = _actual_actor_value(actors[0], path)

        satisfied = actual is not _MISSING and (
            all(_equal(expected, value, tolerance) for value in actual)
            if isinstance(actual, list)
            else _equal(expected, actual, tolerance)
        )
        results.append(
            {
                "constraint_id": constraint["id"],
                "target": target,
                "path": constraint["path"],
                "entity_id": constraint.get("entity_id"),
                "reference_entity_id": constraint.get("reference_entity_id"),
                "requested": deepcopy(expected),
                "actual": None if actual is _MISSING else deepcopy(actual),
                "actual_source": source,
                "status": "satisfied" if satisfied else "failed",
                "evidence": constraint.get("evidence", ""),
            }
        )
    failures = [item for item in results if item["status"] == "failed"]
    return {
        "enabled": bool(constraints),
        "schema_version": USER_CONSTRAINT_SCHEMA_VERSION,
        "status": "failed" if failures else "satisfied",
        "hard_constraint_count": len(constraints),
        "failure_count": len(failures),
        "results": results,
    }
