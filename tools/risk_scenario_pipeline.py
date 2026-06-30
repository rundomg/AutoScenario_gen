import math
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from tools.accident_template_library import get_template


RISK_SCHEMA_VERSION = "risk-scenario-v1"
FIXED_EGO_TARGET_SPEED_MPS = 10.0
FIXED_EGO_SPEED_LABEL = "fixed"
DEFAULT_SPEED_HYPOTHESES_MPS = {
    FIXED_EGO_SPEED_LABEL: FIXED_EGO_TARGET_SPEED_MPS,
}
SPEED_LABELS = ("low", "medium", "high")


def build_actor_context(spawn_payload: Dict[str, Any]) -> Dict[str, Any]:
    entities = spawn_payload.get("entities", []) if isinstance(spawn_payload, dict) else []
    actor_by_id = {
        str(entity.get("id")): entity
        for entity in entities
        if isinstance(entity, dict) and entity.get("id") is not None
    }
    ego = actor_by_id.get("ego") or actor_by_id.get("ego_vehicle") or {}
    ego_location = ego.get("location") or {}
    ego_rotation = ego.get("rotation") or {}
    ego_x = _to_float(ego_location.get("x"), 0.0)
    ego_y = _to_float(ego_location.get("y"), 0.0)
    ego_yaw = math.radians(_to_float(ego_rotation.get("yaw"), 0.0))
    forward = (math.cos(ego_yaw), math.sin(ego_yaw))
    right = (-math.sin(ego_yaw), math.cos(ego_yaw))

    rows = []
    for entity_id, entity in actor_by_id.items():
        location = entity.get("location") or {}
        dx = _to_float(location.get("x"), 0.0) - ego_x
        dy = _to_float(location.get("y"), 0.0) - ego_y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        rows.append(
            {
                "id": entity_id,
                "category": entity.get("category"),
                "spawn_kind": entity.get("spawn_kind"),
                "lane_side_relation": entity.get("lane_side_relation"),
                "x": round(_to_float(location.get("x"), 0.0), 3),
                "y": round(_to_float(location.get("y"), 0.0), 3),
                "yaw": round(_to_float((entity.get("rotation") or {}).get("yaw"), 0.0), 3),
                "relative_to_ego": {
                    "longitudinal_m": round(longitudinal, 3),
                    "lateral_m": round(lateral, 3),
                    "distance_m": round(math.hypot(dx, dy), 3),
                },
            }
        )
    rows.sort(key=lambda row: (0 if row["id"] == "ego" else 1, row["id"]))
    return {"ego_actor_id": "ego", "actors": rows}


def validate_risk_scenario_spec(
    spec: Dict[str, Any],
    spawn_payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(spec, dict):
        return None, "Risk scenario spec must be a JSON object."

    normalized = deepcopy(spec)
    normalized.setdefault("schema_version", RISK_SCHEMA_VERSION)
    if normalized.get("schema_version") != RISK_SCHEMA_VERSION:
        return None, f"Unsupported risk scenario schema: {normalized.get('schema_version')}"

    actor_ids = _collect_actor_ids(spawn_payload)
    if "ego" not in actor_ids:
        return None, "Spawn payload must contain actor id `ego`."

    ego = normalized.setdefault("ego", {})
    if not isinstance(ego, dict):
        return None, "`ego` must be an object."
    ego.setdefault("actor_id", "ego")
    ego.setdefault("controller_mode", "scripted")
    ego.setdefault("route_source", "map_forward_waypoints")
    ego.setdefault("behavior_profile", "accident_reproduction")
    if ego.get("actor_id") != "ego":
        return None, "v1 only supports ego.actor_id=`ego`."
    if ego.get("controller_mode") not in {"scripted", "vla_adapter"}:
        return None, "ego.controller_mode must be `scripted` or `vla_adapter`."
    speed_hypotheses = ego.get("speed_hypotheses_mps")
    if speed_hypotheses is not None:
        normalized_speeds, speed_error = normalize_speed_hypotheses(speed_hypotheses)
        if not speed_error:
            ego["speed_hypotheses_mps"] = normalized_speeds

    candidates = normalized.get("risk_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None, "`risk_candidates` must be a non-empty list."

    seen_ids = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            return None, f"risk_candidates[{index}] must be an object."
        candidate.setdefault("id", f"risk_{index + 1:03d}")
        if candidate["id"] in seen_ids:
            return None, f"Duplicate risk candidate id: {candidate['id']}"
        seen_ids.add(candidate["id"])

        template_id = str(candidate.get("template_id") or "")
        template = get_template(template_id)
        if template is None:
            return None, f"Unknown risk template id: {template_id}"
        candidate["template_id"] = template_id
        candidate.setdefault("accident_type", template["family"])
        candidate["family"] = template["family"]
        candidate["controller"] = template["controller"]
        candidate["execution_status"] = template.get(
            "execution_status",
            "planned_not_implemented",
        )
        candidate["metrics"] = list(template.get("metrics") or [])

        involved = candidate.get("involved_actor_ids")
        if not isinstance(involved, list) or "ego" not in [str(item) for item in involved]:
            return None, f"{candidate['id']} must include `ego` in involved_actor_ids."
        missing = [str(actor_id) for actor_id in involved if str(actor_id) not in actor_ids]
        if missing:
            return None, f"{candidate['id']} references unknown actor ids: {missing}"
        if len({str(actor_id) for actor_id in involved}) < 2:
            return None, f"{candidate['id']} must involve ego and at least one risk actor."
        candidate["involved_actor_ids"] = [str(actor_id) for actor_id in involved]

        candidate["confidence"] = _clamp(_to_float(candidate.get("confidence"), 0.5), 0.0, 1.0)
        parameters = candidate.get("parameter_ranges", candidate.get("parameters", {}))
        if not isinstance(parameters, dict):
            return None, f"{candidate['id']}.parameter_ranges must be an object."
        normalized_parameters = {}
        for name, default_range in template["parameter_ranges"].items():
            normalized_range, error = _normalize_range(parameters.get(name), default_range)
            if error:
                return None, f"{candidate['id']}.{name}: {error}"
            normalized_parameters[name] = normalized_range
        candidate["parameter_ranges"] = normalized_parameters
        candidate["parameters"] = normalized_parameters

    unsupported = normalized.get("unsupported_risk_hypotheses", [])
    if unsupported is None:
        unsupported = []
    if not isinstance(unsupported, list):
        return None, "`unsupported_risk_hypotheses` must be a list when present."
    normalized["unsupported_risk_hypotheses"] = unsupported
    normalized["risk_evidence_warnings"] = _risk_evidence_warnings(
        normalized.get("risk_evidence"),
        actor_ids,
    )

    return normalized, None


def normalize_speed_hypotheses(value: Any) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    if not isinstance(value, dict):
        return None, "ego.speed_hypotheses_mps must be an object."
    missing = [label for label in SPEED_LABELS if label not in value]
    if missing:
        return None, f"ego.speed_hypotheses_mps missing labels: {missing}"
    try:
        speeds = {label: float(value[label]) for label in SPEED_LABELS}
    except (TypeError, ValueError):
        return None, "ego.speed_hypotheses_mps values must be numeric."
    if any(speed < 0.0 for speed in speeds.values()):
        return None, "ego.speed_hypotheses_mps values must be non-negative."
    if not (speeds["low"] < speeds["medium"] < speeds["high"]):
        return None, "ego.speed_hypotheses_mps must satisfy low < medium < high."
    return speeds, None


def require_speed_hypotheses(spec: Dict[str, Any]) -> Optional[str]:
    return None


def require_risk_evidence(spec: Dict[str, Any]) -> Optional[str]:
    if not isinstance(spec, dict) or "risk_evidence" not in spec:
        return "VLM-generated risk specs must include risk_evidence."
    if not isinstance(spec.get("risk_evidence"), dict):
        return "risk_evidence must be an object."
    return None


def ensure_speed_hypotheses(
    spec: Dict[str, Any],
    source: str = "provided",
) -> Tuple[Dict[str, Any], str]:
    normalized = deepcopy(spec)
    ego = normalized.setdefault("ego", {})
    if not isinstance(ego, dict):
        ego = {}
        normalized["ego"] = ego
    ego.pop("speed_hypotheses_mps", None)
    ego["target_speed_mps"] = FIXED_EGO_TARGET_SPEED_MPS
    normalized["speed_hypotheses_source"] = "fixed_10mps"
    return normalized, "fixed_10mps"


def expand_candidates_by_speed(spec: Dict[str, Any]) -> Dict[str, Any]:
    expanded_spec, speed_source = ensure_speed_hypotheses(
        spec,
        source=str(spec.get("speed_hypotheses_source") or "provided"),
    )
    expanded_candidates = []
    for source_index, candidate in enumerate(expanded_spec.get("risk_candidates") or []):
        base_id = str(candidate.get("id") or f"risk_{source_index + 1:03d}")
        speed = FIXED_EGO_TARGET_SPEED_MPS
        speed_candidate = deepcopy(candidate)
        speed_candidate["id"] = f"{base_id}_{FIXED_EGO_SPEED_LABEL}"
        speed_candidate["source_candidate_index"] = source_index
        speed_candidate["source_candidate_id"] = base_id
        speed_candidate["ego_speed_label"] = FIXED_EGO_SPEED_LABEL
        speed_candidate["ego_target_speed_mps"] = speed
        parameter_ranges = deepcopy(
            speed_candidate.get("parameter_ranges")
            or speed_candidate.get("parameters")
            or {}
        )
        parameter_ranges["ego_target_speed_mps"] = [speed, speed]
        speed_candidate["parameter_ranges"] = parameter_ranges
        speed_candidate["parameters"] = parameter_ranges
        expanded_candidates.append(speed_candidate)
    expanded_spec["risk_candidates"] = expanded_candidates
    expanded_spec["speed_hypotheses_source"] = speed_source
    return expanded_spec


def sample_risk_scenario(
    spec: Dict[str, Any],
    sample_index: int = 0,
    seed: Optional[int] = None,
    candidate_index: int = 0,
) -> Dict[str, Any]:
    candidates = spec.get("risk_candidates") or []
    if not candidates:
        raise ValueError("Cannot sample a risk scenario without candidates.")
    candidate = candidates[candidate_index]
    rng = random.Random(seed if seed is not None else sample_index)
    sampled = {}
    for name, bounds in (candidate.get("parameter_ranges") or candidate.get("parameters") or {}).items():
        if isinstance(bounds, list) and len(bounds) == 2:
            lo = _to_float(bounds[0], 0.0)
            hi = _to_float(bounds[1], lo)
            sampled[name] = round(rng.uniform(min(lo, hi), max(lo, hi)), 4)
        else:
            sampled[name] = bounds

    risk_actor_id = next(
        actor_id for actor_id in candidate["involved_actor_ids"] if actor_id != "ego"
    )
    ego_target_speed_mps = _to_float(
        sampled.get("ego_target_speed_mps"),
        _to_float(candidate.get("ego_target_speed_mps"), 0.0),
    )
    return {
        "schema_version": "risk-sample-v1",
        "source_scene_id": spec.get("source_scene_id"),
        "sample_index": int(sample_index),
        "candidate_index": int(candidate_index),
        "source_candidate_index": candidate.get("source_candidate_index"),
        "source_candidate_id": candidate.get("source_candidate_id"),
        "seed": seed,
        "risk_candidate_id": candidate.get("id"),
        "template_id": candidate.get("template_id"),
        "accident_type": candidate.get("accident_type"),
        "family": candidate.get("family"),
        "controller": candidate.get("controller"),
        "execution_status": candidate.get("execution_status"),
        "metrics": list(candidate.get("metrics") or []),
        "ego": deepcopy(spec.get("ego") or {}),
        "ego_speed_label": candidate.get("ego_speed_label"),
        "ego_target_speed_mps": ego_target_speed_mps,
        "risk_actor_id": risk_actor_id,
        "involved_actor_ids": list(candidate.get("involved_actor_ids") or []),
        "sampled_parameters": sampled,
    }


def sample_all_risk_scenarios(
    spec: Dict[str, Any],
    samples_per_candidate: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    samples = []
    candidates = spec.get("risk_candidates") or []
    for candidate_index, _candidate in enumerate(candidates):
        for candidate_sample_index in range(max(1, int(samples_per_candidate))):
            sample_seed = None
            if seed is not None:
                sample_seed = int(seed) + candidate_index * 1009 + candidate_sample_index
            samples.append(
                sample_risk_scenario(
                    spec,
                    sample_index=candidate_sample_index,
                    seed=sample_seed,
                    candidate_index=candidate_index,
                )
            )
    return samples


def _risk_evidence_warnings(evidence: Any, actor_ids: set) -> List[str]:
    if evidence is None:
        return []
    if not isinstance(evidence, dict):
        return ["risk_evidence should be an object when present."]

    warnings = []
    for path, value in _walk_values(evidence):
        if _looks_like_actor_ref(path):
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None:
                    continue
                actor_id = str(item)
                if actor_id not in actor_ids:
                    warnings.append(
                        f"risk_evidence.{'.'.join(path)} references unknown actor id: {actor_id}"
                    )
    return warnings


def _walk_values(value: Any, path: Optional[List[str]] = None):
    path = path or []
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_values(child, path + [str(key)])
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_values(child, path + [str(index)])
    else:
        yield path, value


def _looks_like_actor_ref(path: List[str]) -> bool:
    if not path:
        return False
    key = path[-1]
    return (
        key in {"actor_id", "subject_actor_id", "object_actor_id"}
        or key.endswith("_actor_id")
        or "actors" in path
    )


def _collect_actor_ids(spawn_payload: Dict[str, Any]) -> set:
    return {
        str(entity.get("id"))
        for entity in (spawn_payload.get("entities", []) if isinstance(spawn_payload, dict) else [])
        if isinstance(entity, dict) and entity.get("id") is not None
    }


def _normalize_range(value: Any, default_range: List[float]) -> Tuple[Optional[List[float]], Optional[str]]:
    if value is None:
        return [float(default_range[0]), float(default_range[1])], None
    if isinstance(value, (int, float)):
        number = float(value)
        return [number, number], None
    if not isinstance(value, list) or len(value) != 2:
        return None, "expected a two-number [min, max] range."
    try:
        lo = float(value[0])
        hi = float(value[1])
    except (TypeError, ValueError):
        return None, "range bounds must be numeric."
    if lo < 0.0 or hi < 0.0:
        return None, "range bounds must be non-negative."
    if lo > hi:
        lo, hi = hi, lo
    return [lo, hi], None


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
