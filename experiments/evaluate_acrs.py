#!/usr/bin/env python3
"""Single-scene and batch CLI for ACRS static-scene evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.acrs_visual_extractor import ACRSVisualExtractor
from agents.task_agent import OPENAI_MODEL
from tools.acrs_evaluator import (
    ACRSReferenceError,
    build_candidate_evidence,
    evaluate_acrs,
    reference_from_scene_understanding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate static CARLA scenes with ACRS.")
    parser.add_argument("--result-dir", help="Folder containing one scene's generated artifacts.")
    parser.add_argument("--scene-id", help="Candidate scene id, for example s0000_c0.")
    parser.add_argument("--reference", help="Human-corrected acrs-reference-v1 JSON.")
    parser.add_argument("--manifest", help="JSON/JSONL batch manifest; overrides single-scene arguments.")
    parser.add_argument("--output-dir", help="Output folder; defaults to each result folder.")
    parser.add_argument("--render-image", default="auto", help="Render image path, 'auto', or 'none'.")
    parser.add_argument("--render-graph", default="auto", help="Runtime actor graph path or 'auto'.")
    parser.add_argument("--visual-observation", help="Cached acrs-visual-observation-v1 JSON.")
    parser.add_argument("--mode", choices=("hybrid", "structured"), default="hybrid")
    parser.add_argument("--skip-vlm", action="store_true", help="Use structured fallbacks even when a render exists.")
    parser.add_argument("--create-reference-from-su", help="Create an editable reference draft from this _su.json and exit.")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--request-retries", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _round_number(path: Path) -> int:
    match = re.search(r"_r(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_path(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.is_file()]
    return max(existing, key=lambda path: (_round_number(path), path.stat().st_mtime)) if existing else None


def resolve_graph(result_dir: Path, scene_id: str, requested: str) -> Optional[Path]:
    if requested not in {"auto", "none", ""}:
        return Path(requested).resolve()
    if requested == "none":
        return None
    return latest_path(result_dir.glob(f"{scene_id}_render_actor_graph_r*.json"))


def resolve_render_images(result_dir: Path, scene_id: str, requested: str) -> list[Path]:
    if requested not in {"auto", "none", ""}:
        path = Path(requested).resolve()
        return [path] if path.is_file() else []
    if requested == "none":
        return []
    ego = latest_path(result_dir.glob(f"{scene_id}_ego_r*.png"))
    bev = latest_path(result_dir.glob(f"{scene_id}_bev_r*.png"))
    return [path for path in (ego, bev) if path is not None]


def resolve_artifact(result_dir: Path, scene_id: str, suffix: str) -> Path:
    path = result_dir / f"{scene_id}_{suffix}.json"
    if path.is_file():
        return path
    raise FileNotFoundError(f"Required artifact not found: {path}")


def run_scene(spec: dict, args: argparse.Namespace) -> dict:
    started = time.time()
    result_dir = Path(spec.get("result_dir") or args.result_dir or "").resolve()
    scene_id = str(spec.get("scene_id") or args.scene_id or "")
    reference_path = Path(spec.get("reference") or args.reference or "").resolve()
    if not scene_id or not reference_path.is_file():
        raise ValueError("Each scene requires scene_id and an existing reference path.")
    match_path = resolve_artifact(result_dir, scene_id, "match")
    actors_path = resolve_artifact(result_dir, scene_id, "actors")
    graph_path = resolve_graph(result_dir, scene_id, str(spec.get("render_graph") or args.render_graph))
    images = resolve_render_images(result_dir, scene_id, str(spec.get("render_image") or args.render_image))
    graph = load_json(graph_path) if graph_path else {}

    visual_path_value = spec.get("visual_observation") or args.visual_observation
    visual_path = Path(visual_path_value).resolve() if visual_path_value else None
    visual = load_json(visual_path) if visual_path and visual_path.is_file() else None
    visual_error = None
    output_dir = Path(spec.get("output_dir") or args.output_dir or result_dir).resolve()
    if args.mode == "hybrid" and images and visual is None and not args.skip_vlm:
        request_options = {}
        if args.request_timeout is not None:
            request_options["request_timeout"] = args.request_timeout
        if args.request_retries is not None:
            request_options["request_retries"] = args.request_retries
        visual, visual_error = ACRSVisualExtractor().extract([str(path) for path in images], **request_options)
        if visual:
            visual["model"] = OPENAI_MODEL
            visual_path = output_dir / f"{scene_id}_acrs_visual_observation.json"
            write_json(visual_path, visual)

    candidate = build_candidate_evidence(
        load_json(match_path), load_json(actors_path), graph, visual,
    )
    inputs = {
        "reference": str(reference_path), "match_report": str(match_path),
        "actors_payload": str(actors_path), "render_actor_graph": str(graph_path) if graph_path else None,
        "render_images": [str(path) for path in images],
        "visual_observation": str(visual_path) if visual_path else None,
        "visual_prompt_version": visual.get("prompt_version") if visual else None,
        "visual_model": visual.get("model") if visual else None,
        "visual_error": visual_error,
        "mode": args.mode,
    }
    report = evaluate_acrs(
        load_json(reference_path), candidate,
        render_image_available=bool(images), mode=args.mode, inputs=inputs,
    )
    report["scene_id"] = scene_id
    report["duration_s"] = round(time.time() - started, 3)
    output_path = output_dir / f"{scene_id}_acrs.json"
    write_json(output_path, report)
    report["output_path"] = str(output_path)
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
    csv_path = output_dir / "acrs_summary.csv"
    fields = ["scene_id", "status", "official", "road_topology", "background_environment", "traffic_participants", "acrs", "duration_s", "output_path", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            scores = report.get("scores", {})
            writer.writerow({
                "scene_id": report.get("scene_id"), "status": report.get("status"),
                "official": report.get("official"), "duration_s": report.get("duration_s"),
                "output_path": report.get("output_path"), "error": report.get("error"),
                **{name: scores.get(name) for name in fields[3:7]},
            })
    official = [report for report in reports if report.get("official")]
    dimensions = ("road_topology", "background_environment", "traffic_participants", "acrs")
    def aggregate(rows: list[dict]) -> dict:
        result: Dict[str, Any] = {}
        for dimension in dimensions:
            values = [report.get("scores", {}).get(dimension) for report in rows]
            values = [float(value) for value in values if isinstance(value, (int, float))]
            result[dimension] = {
                "count": len(values),
                "mean": round(statistics.mean(values), 4) if values else None,
                "std": round(statistics.pstdev(values), 4) if values else None,
                "min": round(min(values), 4) if values else None,
                "max": round(max(values), 4) if values else None,
            }
        return result

    version_groups: Dict[str, list] = {}
    for report in official:
        inputs = report.get("inputs", {})
        key = f"{inputs.get('visual_model') or 'unknown-model'}::{inputs.get('visual_prompt_version') or 'unknown-prompt'}"
        version_groups.setdefault(key, []).append(report)
    mixed_versions = len(version_groups) > 1
    write_json(output_dir / "acrs_summary.json", {
        "scene_count": len(reports),
        "official_scene_count": len(official),
        "mixed_visual_versions": mixed_versions,
        "aggregates": None if mixed_versions else aggregate(official),
        "aggregates_by_visual_version": {
            version: aggregate(rows) for version, rows in sorted(version_groups.items())
        },
    })


def main() -> int:
    args = parse_args()
    if args.create_reference_from_su:
        su_path = Path(args.create_reference_from_su).resolve()
        scene_id = args.scene_id or (
            su_path.name[:-8] if su_path.name.endswith("_su.json") else su_path.stem
        )
        output_dir = Path(args.output_dir or su_path.parent).resolve()
        output_path = output_dir / f"{scene_id}_acrs_reference.json"
        write_json(output_path, reference_from_scene_understanding(load_json(su_path), scene_id))
        print(f"Created draft reference requiring human review: {output_path}")
        return 0

    specs = load_manifest(Path(args.manifest).resolve()) if args.manifest else [{}]
    reports = []
    for spec in specs:
        try:
            report = run_scene(spec, args)
            reports.append(report)
            print(f"{report['scene_id']}: ACRS={report['scores']['acrs']} status={report['status']}")
        except (OSError, ValueError, ACRSReferenceError, json.JSONDecodeError) as exc:
            scene_id = str(spec.get("scene_id") or args.scene_id or "unknown")
            reports.append({"scene_id": scene_id, "status": "error", "official": False, "error": str(exc), "scores": {}})
            print(f"{scene_id}: ERROR: {exc}", file=sys.stderr)
    if args.manifest:
        output_dir = Path(args.output_dir or Path(args.manifest).resolve().parent).resolve()
        write_summary(reports, output_dir)
    return 1 if any(report.get("status") == "error" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
