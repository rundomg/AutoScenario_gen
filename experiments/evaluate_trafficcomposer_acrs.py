#!/usr/bin/env python3
"""Evaluate TrafficComposer YAML IR files with the deterministic ACRS scorer.

The public TrafficComposer repository stops at its merged YAML IR.  This
adapter therefore reports a structured, diagnostic ACRS result.  It never
claims that a YAML-only result is an official ACRS measurement because no
CARLA runtime actor graph or render observation is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.acrs_evaluator import ACRSReferenceError, evaluate_acrs

DIMENSIONS = (
    "road_topology",
    "background_environment",
    "critical_traffic_participants",
    "background_traffic_participants",
    "traffic_participants",
    "acrs",
)
UNKNOWN = "unknown"


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Some TrafficComposer outputs contain a YAML document serialized as a
    # scalar.  Decode that form once more, matching the upstream loader.
    if isinstance(payload, str):
        payload = yaml.safe_load(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def slug(value: Any) -> str:
    if value is None:
        return UNKNOWN
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if not text or text in {"none", "null", "n/a", "na", "unknown"}:
        return UNKNOWN
    return " ".join(text.split())


def integer(value: Any) -> Any:
    if isinstance(value, bool):
        return UNKNOWN
    if isinstance(value, int):
        return value
    text = slug(value)
    try:
        return int(text)
    except (TypeError, ValueError):
        return UNKNOWN


def canonical_topology(value: Any) -> str:
    text = slug(value)
    if text in {"straight", "straight road", "road", "street", "highway"}:
        return "straight"
    if text in {"t intersection", "t junction"}:
        return "t_junction"
    if text in {"intersection", "cross intersection", "junction"}:
        return "intersection"
    if text == "roundabout":
        return "roundabout"
    return UNKNOWN


def canonical_weather(value: Any) -> str:
    text = slug(value)
    return {
        "rainy": "rain", "raining": "rain", "wet": "rain",
        "foggy": "fog", "snowy": "snow", "overcast": "cloudy",
    }.get(text, text)


def canonical_time(value: Any) -> tuple[str, str]:
    text = slug(value)
    if text in {"night", "nighttime", "evening"}:
        return "night", "night"
    if text in {"day", "daytime", "morning", "afternoon"}:
        return "day", "daylight"
    return UNKNOWN, UNKNOWN


def as_relation_text(entry: dict) -> str:
    values = [entry.get("position_relation"), entry.get("position_target")]
    parts = []
    for value in values:
        if isinstance(value, list):
            parts.extend(slug(item) for item in value)
        else:
            parts.append(slug(value))
    return " ".join(part for part in parts if part != UNKNOWN)


def actor_layout(entry: dict, ego_lane: Any) -> dict:
    relation = as_relation_text(entry)
    lane_idx = integer(entry.get("lane_idx"))
    lane_assignment = UNKNOWN
    if lane_idx != UNKNOWN and ego_lane != UNKNOWN:
        if lane_idx == ego_lane:
            lane_assignment = "same_lane"
        elif lane_idx < ego_lane:
            lane_assignment = "left_lane"
        else:
            lane_assignment = "right_lane"
    elif "opposite" in relation:
        lane_assignment = "opposing_lane"
    elif "left" in relation:
        lane_assignment = "left_lane"
    elif "right" in relation:
        lane_assignment = "right_lane"
    elif any(token in relation for token in ("front", "ahead", "behind")):
        lane_assignment = "same_lane"

    if any(token in relation for token in ("front", "ahead")):
        longitudinal = "ahead"
    elif "behind" in relation:
        longitudinal = "behind"
    else:
        longitudinal = UNKNOWN

    behavior = slug(entry.get("current_behavior"))
    if "cross" in behavior or "perpendicular" in relation:
        heading = "crossing"
    elif "opposite" in relation:
        heading = "opposite_direction"
    elif behavior in {"go forward", "stop", "yield", "change lane to left", "change lane to right"}:
        heading = "same_direction"
    else:
        heading = UNKNOWN
    return {
        "lane_assignment": lane_assignment,
        "lane_from_right": UNKNOWN,
        "branch_assignment": UNKNOWN,
        "heading_relation": heading,
        "longitudinal_relation": longitudinal,
        "distance_band": UNKNOWN,
    }


def trafficcomposer_ir_to_candidate(payload: dict) -> dict:
    road_ir = payload.get("road_network") if isinstance(payload.get("road_network"), dict) else {}
    env_ir = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    participants = payload.get("participant") if isinstance(payload.get("participant"), dict) else {}
    topology = canonical_topology(road_ir.get("road_type"))
    is_junction = topology in {"intersection", "t_junction", "roundabout"}

    # TrafficComposer's lane_number is a total visual lane count.  ACRS uses
    # separate ego-direction and opposing-direction counts, so copying the
    # total into either field would fabricate evidence.  Keep both unknown.
    road = {
        "topology_type": topology,
        "directionality": UNKNOWN,
        "forward_lane_count": UNKNOWN,
        "opposing_lane_count": UNKNOWN,
        "ego_lane_from_right": UNKNOWN,
        "has_center_median": UNKNOWN,
        "left_parking_presence": UNKNOWN,
        "right_parking_presence": UNKNOWN,
        "junction_visible": is_junction if topology != UNKNOWN else UNKNOWN,
        "junction_type": topology if is_junction else ("none" if topology == "straight" else UNKNOWN),
        "junction_branches": UNKNOWN,
        "branch_count": UNKNOWN,
    }

    time_of_day, lighting = canonical_time(env_ir.get("time"))
    controls = []
    traffic_sign = slug(road_ir.get("traffic_sign"))
    if traffic_sign != UNKNOWN:
        controls.append(traffic_sign.replace(" ", "_"))
    # ACRS evaluates the presence of a traffic control in the static scene,
    # not its signal phase.  TrafficComposer stores the phase (red/green).
    if slug(road_ir.get("traffic_light")) != UNKNOWN:
        controls.append("traffic_light")
    environment = {
        "weather": canonical_weather(env_ir.get("weather")),
        "lighting": lighting,
        "time_of_day": time_of_day,
        "road_surface": UNKNOWN,
        "urban_density": UNKNOWN,
        "roadside_context_left": UNKNOWN,
        "roadside_context_right": UNKNOWN,
        "landmarks_and_controls": controls if controls else UNKNOWN,
    }
    environment_sources = {
        key: "trafficcomposer_textual_ir"
        for key, value in environment.items()
        if value != UNKNOWN
    }

    ego = participants.get("ego_vehicle") if isinstance(participants.get("ego_vehicle"), dict) else {}
    ego_lane = integer(ego.get("lane_idx"))
    actors = []
    for actor_id, entry in participants.items():
        if actor_id in {"ego", "ego_vehicle"} or not isinstance(entry, dict):
            continue
        category = slug(entry.get("type"))
        category = {"vehicle": "car", "sedan": "car", "cyclist": "bicycle"}.get(category, category)
        actors.append({
            "id": str(actor_id),
            "category": category,
            "subtype": UNKNOWN,
            **actor_layout(entry, ego_lane),
        })

    return {
        "road_topology": road,
        "environment": environment,
        "environment_sources": environment_sources,
        "actors": actors,
        "runtime_truth_available": False,
        "visual_observation_available": False,
        "evidence_conflicts": [],
        "adapter_diagnostics": {
            "reported_total_lane_number": integer(payload.get("lane_number", road_ir.get("lane_number"))),
            "participant_count_excluding_ego": len(actors),
        },
    }


def aggregate_scores(reports: list[dict]) -> dict:
    output = {}
    for dimension in DIMENSIONS:
        values = [report.get("scores", {}).get(dimension) for report in reports]
        values = [float(value) for value in values if isinstance(value, (int, float))]
        output[dimension] = {
            "count": len(values),
            "mean": round(statistics.mean(values), 4) if values else None,
            "std": round(statistics.pstdev(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
        }
    return output


def evaluate_ir(ir_path: Path, reference_path: Path) -> dict:
    candidate = trafficcomposer_ir_to_candidate(load_yaml(ir_path))
    report = evaluate_acrs(
        load_json(reference_path), candidate,
        render_image_available=False,
        mode="structured",
        inputs={
            "source": "trafficcomposer_yaml_ir",
            "traffic_ir": str(ir_path.resolve()),
            "reference": str(reference_path.resolve()),
            "reported_total_lane_number": candidate["adapter_diagnostics"]["reported_total_lane_number"],
        },
    )
    report["adapter_diagnostics"] = candidate["adapter_diagnostics"]
    report["diagnostic_reason"] = (
        "The public TrafficComposer artifact is YAML IR, not a CARLA-realized scene; "
        "runtime actor and render evidence are unavailable."
    )
    return report


def load_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        rows = payload.get("scenes", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Manifest must be a JSON list, {\"scenes\": [...]}, or JSONL objects.")
    return rows


def write_summary(reports: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["scene_id", *DIMENSIONS, "status", "official", "output_path", "error"]
    with (output_dir / "trafficcomposer_acrs_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            row = {key: report.get(key) for key in fields if key not in DIMENSIONS}
            row.update({key: report.get("scores", {}).get(key) for key in DIMENSIONS})
            writer.writerow(row)
    scored = [report for report in reports if isinstance(report.get("scores", {}).get("acrs"), (int, float))]
    write_json(output_dir / "trafficcomposer_acrs_summary.json", {
        "schema_version": "trafficcomposer-acrs-batch-summary-v1",
        "scene_count": len(reports),
        "scored_scene_count": len(scored),
        "official_scene_count": 0,
        "aggregates": aggregate_scores(scored),
        "diagnostic_only": True,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic-ir", help="One TrafficComposer merged/textual YAML IR.")
    parser.add_argument("--reference", help="Human-reviewed acrs-reference-v1 JSON.")
    parser.add_argument("--manifest", help="Batch JSON/JSONL with traffic_ir and reference per item.")
    parser.add_argument("--output-dir", default="results/trafficcomposer_acrs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest and (not args.traffic_ir or not args.reference):
        raise SystemExit("Single-scene mode requires --traffic-ir and --reference.")
    specs = load_manifest(Path(args.manifest).resolve()) if args.manifest else [{
        "traffic_ir": args.traffic_ir, "reference": args.reference,
    }]
    output_dir = Path(args.output_dir).resolve()
    reports = []
    for index, spec in enumerate(specs):
        ir_path = Path(str(spec.get("traffic_ir") or "")).resolve()
        reference_path = Path(str(spec.get("reference") or "")).resolve()
        scene_id = str(spec.get("scene_id") or ir_path.stem or f"scene_{index:04d}")
        try:
            report = evaluate_ir(ir_path, reference_path)
            report["scene_id"] = scene_id
            output_path = output_dir / f"{scene_id}_trafficcomposer_acrs.json"
            report["output_path"] = str(output_path)
            write_json(output_path, report)
            print(f"{scene_id}: ACRS={report['scores']['acrs']} (diagnostic)")
        except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError, ACRSReferenceError) as exc:
            report = {"scene_id": scene_id, "status": "error", "official": False, "error": str(exc), "scores": {}}
            print(f"{scene_id}: ERROR: {exc}", file=sys.stderr)
        reports.append(report)
    write_summary(reports, output_dir)
    return 1 if any(report.get("status") == "error" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
