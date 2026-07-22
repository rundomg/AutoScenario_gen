"""Adapters from the current VLM/static-reconstruction JSON to fitting contracts."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    AccidentSpecification,
    ActorDimensions,
    ActorState,
    CollisionPairHypothesis,
    SceneState,
    clamp,
)


_TYPE_ALIASES = {
    "rear": "rear_end",
    "rear-end": "rear_end",
    "rear end": "rear_end",
    "追尾": "rear_end",
    "intersection": "intersection",
    "cross": "intersection",
    "turn": "intersection",
    "路口": "intersection",
    "交叉": "intersection",
    "lane_change": "lane_change",
    "lane change": "lane_change",
    "cut-in": "lane_change",
    "cut in": "lane_change",
    "sideswipe": "lane_change",
    "换道": "lane_change",
    "侧擦": "lane_change",
}


def normalize_accident_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    for token, normalized in _TYPE_ALIASES.items():
        if token in text:
            return normalized
    return "rear_end"


def build_scene_state(
    spawn_payload: Dict[str, Any],
    *,
    duration_s: float = 6.0,
    fixed_delta_seconds: float = 0.05,
    map_name: Optional[str] = None,
    default_speed_limit_mps: float = 13.9,
) -> SceneState:
    actors: Dict[str, ActorState] = {}
    for entity in spawn_payload.get("entities") or []:
        if not isinstance(entity, dict) or entity.get("id") is None:
            continue
        actor_id = str(entity["id"])
        location = entity.get("location") or {}
        rotation = entity.get("rotation") or {}
        projected = entity.get("projected_lane") or {}
        dimensions = _dimensions_from_entity(entity)
        actors[actor_id] = ActorState(
            actor_id=actor_id,
            x=_float(location.get("x"), 0.0),
            y=_float(location.get("y"), 0.0),
            yaw_deg=_float(rotation.get("yaw"), 0.0),
            speed_mps=max(0.0, _initial_speed(entity)),
            category=str(entity.get("category") or entity.get("spawn_kind") or "vehicle"),
            lane_id=_int_or_none(projected.get("lane_id")),
            road_id=_int_or_none(projected.get("road_id")),
            speed_limit_mps=_positive_or_none(entity.get("speed_limit_mps")),
            dimensions=dimensions,
            metadata=dict(entity),
        )
    if not actors:
        raise ValueError("Spawn payload contains no actor entities.")
    return SceneState(
        actors=actors,
        duration_s=max(0.5, float(duration_s)),
        fixed_delta_seconds=max(0.01, float(fixed_delta_seconds)),
        map_name=map_name,
        default_speed_limit_mps=max(1.0, float(default_speed_limit_mps)),
    )


def build_accident_specification(
    understanding: Dict[str, Any],
    actor_ids: Iterable[str],
    *,
    duration_s: float = 6.0,
    top_k: int = 3,
) -> AccidentSpecification:
    """Normalize current and proposed VLM schemas without trusting numeric motion."""

    known_ids = {str(actor_id) for actor_id in actor_ids}
    if "ego" not in known_ids:
        raise ValueError("Collision fitting requires an `ego` actor.")
    accident_type = normalize_accident_type(understanding.get("accident_type"))
    participants = understanding.get("participants") or []
    ref_to_ids: Dict[str, List[Tuple[str, float]]] = {}
    participant_by_ref: Dict[str, Dict[str, Any]] = {}
    for row in participants:
        if not isinstance(row, dict):
            continue
        ref = str(row.get("ref") or "")
        if not ref:
            continue
        participant_by_ref[ref] = row
        choices: List[Tuple[str, float]] = []
        matched = row.get("matched_actor_id")
        if matched is not None and str(matched) in known_ids:
            choices.append(
                (str(matched), clamp(_float(row.get("match_confidence"), 0.6), 0.0, 1.0))
            )
        for candidate in row.get("candidate_actor_ids") or []:
            if isinstance(candidate, dict):
                candidate_id = str(candidate.get("actor_id") or "")
                confidence = _float(candidate.get("confidence"), 0.35)
            else:
                candidate_id, confidence = str(candidate), 0.35
            if candidate_id in known_ids and candidate_id not in {item[0] for item in choices}:
                choices.append((candidate_id, clamp(confidence, 0.0, 1.0)))
        ref_to_ids[ref] = choices

    end_state = understanding.get("end_state") or {}
    pair_refs = list(end_state.get("collision_pair") or [])
    explicit_striking = understanding.get("striking_actor_id") or understanding.get(
        "striking_actor_ref"
    )
    explicit_struck = understanding.get("struck_actor_id") or understanding.get(
        "struck_actor_ref"
    )
    if explicit_striking is not None and explicit_struck is not None:
        pair_refs = [explicit_striking, explicit_struck]
    if len(pair_refs) < 2:
        pair_refs = _infer_pair_refs(participants)

    hypotheses = _build_pair_hypotheses(
        pair_refs, ref_to_ids, known_ids, accident_type, participant_by_ref, top_k
    )
    if not hypotheses:
        fallback_other = next((actor_id for actor_id in sorted(known_ids) if actor_id != "ego"), None)
        if fallback_other is None:
            raise ValueError("Collision fitting needs at least two actors.")
        hypotheses = [
            CollisionPairHypothesis(
                actor_a="ego",
                actor_b=fallback_other,
                accident_type=accident_type,
                confidence=0.2,
                evidence="fallback to the only/first non-ego actor",
            )
        ]

    primary = hypotheses[0]
    striking, struck = _orient_pair(
        primary.actor_a,
        primary.actor_b,
        accident_type,
        explicit_striking,
        explicit_struck,
        ref_to_ids,
        participant_by_ref,
    )
    for hypothesis in hypotheses:
        hypothesis.striking_actor_id, hypothesis.struck_actor_id = _orient_pair(
            hypothesis.actor_a,
            hypothesis.actor_b,
            accident_type,
            explicit_striking,
            explicit_struck,
            ref_to_ids,
            participant_by_ref,
        )

    actor_maneuvers: Dict[str, str] = {}
    behavior_sequences: Dict[str, List[str]] = {}
    for ref, row in participant_by_ref.items():
        ids = ref_to_ids.get(ref) or []
        if not ids:
            continue
        actor_id = ids[0][0]
        actor_maneuvers[actor_id] = _normalize_maneuver(
            row.get("maneuver") or row.get("role") or "keep_lane"
        )
        sequence = row.get("behavior_sequence")
        if isinstance(sequence, str):
            sequence = [part.strip() for part in sequence.replace("→", "->").split("->")]
        if isinstance(sequence, list):
            behavior_sequences[actor_id] = [
                _normalize_behavior(item) for item in sequence if str(item).strip()
            ]
    _fill_behaviors_from_events(
        understanding.get("event_sequence") or [],
        behavior_sequences,
        actor_maneuvers,
        ref_to_ids,
        known_ids,
    )
    actor_maneuvers.setdefault(striking, "keep_lane")
    actor_maneuvers.setdefault(struck, "keep_lane")
    behavior_sequences.setdefault(striking, ["keep"])
    behavior_sequences.setdefault(struck, ["keep"])

    time_range = _collision_time_range(understanding, duration_s)
    confidence = dict(understanding.get("confidence") or {})
    confidence.setdefault("accident_type", _float(understanding.get("accident_type_confidence"), 0.6))
    confidence.setdefault("collision_pair", primary.confidence)
    return AccidentSpecification(
        accident_type=accident_type,
        striking_actor_id=striking,
        struck_actor_id=struck,
        actor_maneuvers=actor_maneuvers,
        behavior_sequences=behavior_sequences,
        relative_motion_constraints=list(
            understanding.get("relative_motion_constraints")
            or understanding.get("relative_motion")
            or []
        ),
        event_order=list(understanding.get("event_sequence") or []),
        collision_time_range=time_range,
        confidence={str(key): clamp(_float(value, 0.5), 0.0, 1.0) for key, value in confidence.items()},
        pair_hypotheses=hypotheses,
        expected_contact_sides=dict(understanding.get("expected_contact_sides") or {}),
        metadata={"source_schema": "video_understanding", "summary": understanding.get("summary")},
    )


def _build_pair_hypotheses(
    pair_refs: List[Any],
    ref_to_ids: Dict[str, List[Tuple[str, float]]],
    known_ids: set,
    accident_type: str,
    participant_by_ref: Dict[str, Dict[str, Any]],
    top_k: int,
) -> List[CollisionPairHypothesis]:
    if len(pair_refs) < 2:
        return []
    choices: List[List[Tuple[str, float]]] = []
    for raw in pair_refs[:2]:
        key = str(raw)
        if key in known_ids:
            choices.append([(key, 1.0)])
        else:
            choices.append(list(ref_to_ids.get(key) or []))
    if not all(choices):
        return []
    output: List[CollisionPairHypothesis] = []
    for actor_a, conf_a in choices[0]:
        for actor_b, conf_b in choices[1]:
            if actor_a == actor_b:
                continue
            output.append(
                CollisionPairHypothesis(
                    actor_a=actor_a,
                    actor_b=actor_b,
                    accident_type=accident_type,
                    confidence=clamp(math.sqrt(conf_a * conf_b), 0.0, 1.0),
                    evidence="VLM collision pair mapped through participant candidates",
                )
            )
    output.sort(key=lambda item: (-item.confidence, item.actor_a, item.actor_b))
    return output[: max(1, int(top_k))]


def _orient_pair(
    actor_a: str,
    actor_b: str,
    accident_type: str,
    explicit_striking: Any,
    explicit_struck: Any,
    ref_to_ids: Dict[str, List[Tuple[str, float]]],
    participant_by_ref: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    def resolve(value: Any) -> Optional[str]:
        key = str(value or "")
        candidates = ref_to_ids.get(key) or []
        return candidates[0][0] if candidates else key or None

    explicit_a, explicit_b = resolve(explicit_striking), resolve(explicit_struck)
    if {explicit_a, explicit_b} == {actor_a, actor_b}:
        return str(explicit_a), str(explicit_b)
    roles = {}
    for ref, row in participant_by_ref.items():
        candidates = ref_to_ids.get(ref) or []
        if candidates:
            roles[candidates[0][0]] = str(row.get("role") or "").lower()
    striking_tokens = ("following", "rear", "cut_in", "crossing", "striking")
    for actor_id in (actor_a, actor_b):
        if any(token in roles.get(actor_id, "") for token in striking_tokens):
            return actor_id, actor_b if actor_id == actor_a else actor_a
    if accident_type == "rear_end" and "ego" in {actor_a, actor_b}:
        # Dashcam ego is normally the following/striking vehicle in this dataset.
        return "ego", actor_b if actor_a == "ego" else actor_a
    return actor_a, actor_b


def _collision_time_range(understanding: Dict[str, Any], duration_s: float) -> Tuple[float, float]:
    value = understanding.get("collision_time_range_s") or understanding.get("collision_time_range")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low, high = _float(value[0], 0.0), _float(value[1], duration_s)
    else:
        collision_second = understanding.get("collision_second")
        if collision_second is not None:
            center = _float(collision_second, duration_s * 0.75)
            low, high = center - 0.35, center + 0.35
        else:
            low, high = duration_s * 0.55, duration_s * 0.9
    low = clamp(low, 0.05, max(0.05, duration_s))
    high = clamp(high, low, max(low, duration_s))
    return (round(low, 3), round(high, 3))


def _infer_pair_refs(participants: List[Any]) -> List[str]:
    refs = [str(row.get("ref")) for row in participants if isinstance(row, dict) and row.get("ref")]
    return ["ego", refs[0]] if refs else []


def _fill_behaviors_from_events(events, sequences, maneuvers, ref_to_ids, known_ids):
    for event in events:
        if not isinstance(event, dict):
            continue
        raw_id = str(event.get("actor_id") or event.get("actor_ref") or "")
        actor_id = raw_id if raw_id in known_ids else ((ref_to_ids.get(raw_id) or [(None, 0.0)])[0][0])
        if not actor_id:
            continue
        action = _normalize_behavior(event.get("action"))
        if action and action not in sequences.get(actor_id, []):
            sequences.setdefault(actor_id, []).append(action)
        if action in {"lane_change", "turn_left", "turn_right", "straight"}:
            maneuvers[actor_id] = action


def _normalize_behavior(value: Any) -> str:
    text = str(value or "keep").strip().lower().replace(" ", "_")
    aliases = {"maintain": "keep", "decelerate": "decelerate", "stop": "stop", "cut_in": "lane_change"}
    return aliases.get(text, text)


def _normalize_maneuver(value: Any) -> str:
    text = str(value or "keep_lane").lower().replace("-", "_").replace(" ", "_")
    if text in {"keep", "keep_lane", "lane_follow", "lead_vehicle", "following_vehicle"}:
        return "keep_lane"
    if "left" in text and "lane" not in text:
        return "turn_left"
    if "right" in text and "lane" not in text:
        return "turn_right"
    if "straight" in text or "cross" in text:
        return "straight"
    if "lane_change" in text or "change_lane" in text or "cut_in" in text or "sideswipe" in text:
        return "lane_change_left" if "left" in text else "lane_change_right"
    return "keep_lane"


def _dimensions_from_entity(entity: Dict[str, Any]) -> ActorDimensions:
    raw = entity.get("dimensions") or entity.get("bounding_box") or {}
    extent = raw.get("extent") if isinstance(raw, dict) else None
    if isinstance(extent, dict):
        length = 2.0 * _float(extent.get("x"), 2.3)
        width = 2.0 * _float(extent.get("y"), 0.95)
        height = 2.0 * _float(extent.get("z"), 0.8)
    else:
        length = _float(raw.get("length") if isinstance(raw, dict) else None, 0.0)
        width = _float(raw.get("width") if isinstance(raw, dict) else None, 0.0)
        height = _float(raw.get("height") if isinstance(raw, dict) else None, 0.0)
    category = str(entity.get("category") or "car").lower()
    defaults = {
        "truck": (7.5, 2.5, 3.0),
        "bus": (10.0, 2.6, 3.2),
        "motorcycle": (2.2, 0.8, 1.4),
        "bicycle": (1.8, 0.7, 1.4),
    }.get(category, (4.6, 1.9, 1.6))
    return ActorDimensions(
        length=length if length > 0.1 else defaults[0],
        width=width if width > 0.1 else defaults[1],
        height=height if height > 0.1 else defaults[2],
    )


def _initial_speed(entity: Dict[str, Any]) -> float:
    for key in ("initial_speed_mps", "speed_mps", "target_speed_mps"):
        if entity.get(key) is not None:
            return _float(entity.get(key), 0.0)
    return 0.0


def _float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positive_or_none(value: Any) -> Optional[float]:
    number = _float(value, 0.0)
    return number if number > 0.0 else None
