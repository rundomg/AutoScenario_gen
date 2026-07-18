#!/usr/bin/env python3
"""Evaluate ScenicNL outputs with AutoScenario's ACRS metric.

The evaluator compile-checks and samples each probabilistic Scenic program in
the Scenic conda environment, then evaluates every successful sample with the
same deterministic ACRS scorer used by AutoScenario.  Because this path has no
CARLA runtime actor graph or render evidence, its ACRS results are deliberately
diagnostic (``official=false``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.acrs_evaluator import ACRSReferenceError, evaluate_acrs

EXTRACTOR = PROJECT_ROOT / "tools" / "scenicnl_sample_extractor.py"
DIMENSIONS = (
    "road_topology",
    "background_environment",
    "critical_traffic_participants",
    "background_traffic_participants",
    "traffic_participants",
    "acrs",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def build_extract_command(
    scenic_path: Path,
    output_path: Path,
    *,
    conda_executable: str,
    conda_env: str,
    samples: int,
    max_iterations: int,
    seed: int,
) -> list[str]:
    return [
        conda_executable, "run", "--no-capture-output", "-n", conda_env,
        "python", str(EXTRACTOR), str(scenic_path), str(output_path),
        "--samples", str(samples), "--max-iterations", str(max_iterations),
        "--seed", str(seed),
    ]


def extract_samples(
    scenic_path: Path,
    *,
    conda_executable: str,
    conda_env: str,
    samples: int,
    max_iterations: int,
    seed: int,
    timeout: int,
    scenic_pythonpath: str | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="scenicnl_acrs_") as folder:
        output_path = Path(folder) / "sampled_evidence.json"
        command = build_extract_command(
            scenic_path.resolve(), output_path,
            conda_executable=conda_executable, conda_env=conda_env,
            samples=samples, max_iterations=max_iterations, seed=seed,
        )
        try:
            environment = os.environ.copy()
            if scenic_pythonpath:
                previous = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    f"{scenic_pythonpath}{os.pathsep}{previous}" if previous else scenic_pythonpath
                )
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, env=environment,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Conda executable not found: {conda_executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Scenic extraction timed out after {timeout}s: {scenic_path}") from exc
        if process.returncode != 0 or not output_path.is_file():
            detail = (process.stderr or process.stdout or "no subprocess output").strip()
            raise RuntimeError(f"Scenic extractor failed ({process.returncode}): {detail[-4000:]}")
        payload = load_json(output_path)
        payload["extractor_stdout"] = (process.stdout or "")[-4000:]
        payload["extractor_stderr"] = (process.stderr or "")[-4000:]
        return payload


def aggregate_scores(reports: Iterable[dict]) -> dict:
    rows = list(reports)
    output: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        values = [row.get("scores", {}).get(dimension) for row in rows]
        values = [float(value) for value in values if isinstance(value, (int, float))]
        output[dimension] = {
            "count": len(values),
            "mean": round(statistics.mean(values), 4) if values else None,
            "std": round(statistics.pstdev(values), 4) if values else None,
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
        }
    return output


def evaluate_program(
    scenic_path: Path,
    reference_path: Path,
    *,
    conda_executable: str,
    conda_env: str,
    samples: int,
    max_iterations: int,
    seed: int,
    timeout: int,
    scenic_pythonpath: str | None = None,
    source_id: str = "scenicnl",
    source_display: str = "ScenicNL",
    extractor=extract_samples,
) -> dict:
    reference = load_json(reference_path)
    extracted = extractor(
        scenic_path,
        conda_executable=conda_executable, conda_env=conda_env,
        samples=samples, max_iterations=max_iterations, seed=seed, timeout=timeout,
        scenic_pythonpath=scenic_pythonpath,
    )
    sample_reports = []
    for sampled in extracted.get("samples", []):
        row = {
            key: sampled.get(key)
            for key in ("sample_index", "seed", "success", "iterations", "error", "traceback")
        }
        if sampled.get("success") and isinstance(sampled.get("candidate"), dict):
            report = evaluate_acrs(
                reference, sampled["candidate"], render_image_available=False,
                mode="structured",
                inputs={
                    "source": f"{source_id}_sample",
                    "scenic_path": str(scenic_path.resolve()),
                    "reference": str(reference_path.resolve()),
                    "sample_index": sampled.get("sample_index"),
                    "seed": sampled.get("seed"),
                },
            )
            row["acrs_report"] = report
            row["scores"] = report["scores"]
        sample_reports.append(row)

    successful = [row["acrs_report"] for row in sample_reports if "acrs_report" in row]
    requested = int(extracted.get("requested_samples") or samples)
    return {
        "schema_version": f"{source_id}-acrs-evaluation-v1",
        "scenic_path": str(scenic_path.resolve()),
        "reference": str(reference_path.resolve()),
        "compiled": bool(extracted.get("compiled")),
        "compile_error": extracted.get("compile_error"),
        "requested_samples": requested,
        "successful_samples": len(successful),
        "sample_success_rate": round(len(successful) / requested, 6) if requested else 0.0,
        "official": False,
        "status": "complete_diagnostic" if successful else "no_successful_samples",
        "diagnostic_reason": (
            f"Sampled {source_display} Scenic evidence has no CARLA runtime actor graph or render/VLM evidence; "
            "scores must not be reported as official ACRS."
        ),
        "aggregates": aggregate_scores(successful),
        "samples": sample_reports,
        "extractor_diagnostics": {
            "stdout": extracted.get("extractor_stdout", ""),
            "stderr": extracted.get("extractor_stderr", ""),
        },
    }


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


def write_batch_summary(reports: list[dict], output_dir: Path, source_id: str = "scenicnl") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_id", "compiled", "requested_samples", "successful_samples",
        "sample_success_rate", *DIMENSIONS, "status", "output_path", "error",
    ]
    with (output_dir / f"{source_id}_acrs_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            means = {name: report.get("aggregates", {}).get(name, {}).get("mean") for name in DIMENSIONS}
            writer.writerow({key: report.get(key) for key in fields if key not in DIMENSIONS} | means)
    valid = [report for report in reports if report.get("successful_samples", 0) > 0]
    pooled_reports = [
        sample["acrs_report"]
        for report in valid for sample in report.get("samples", [])
        if isinstance(sample.get("acrs_report"), dict)
    ]
    write_json(output_dir / f"{source_id}_acrs_summary.json", {
        "schema_version": f"{source_id}-acrs-batch-summary-v1",
        "scene_count": len(reports),
        "compiled_scene_count": sum(bool(report.get("compiled")) for report in reports),
        "scenes_with_successful_samples": len(valid),
        "official_scene_count": 0,
        "macro_scene_means": {
            dimension: round(statistics.mean(values), 4) if values else None
            for dimension in DIMENSIONS
            for values in [[
                report["aggregates"][dimension]["mean"] for report in valid
                if report.get("aggregates", {}).get(dimension, {}).get("mean") is not None
            ]]
        },
        "pooled_sample_aggregates": aggregate_scores(pooled_reports),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenic-path", help="One generated .scenic program.")
    parser.add_argument("--reference", help="Human-reviewed acrs-reference-v1 JSON.")
    parser.add_argument("--manifest", help="Batch JSON/JSONL with scenic_path and reference per item.")
    parser.add_argument("--output-dir", default="results/scenicnl_acrs")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--conda-env", default="scenicNL")
    parser.add_argument("--conda-executable", default="conda")
    parser.add_argument(
        "--scenic-pythonpath",
        help="Optional Scenic source tree, e.g. ChatScene/Scenic/src for Scenic 2.1 programs.",
    )
    parser.add_argument("--source-id", default="scenicnl")
    parser.add_argument("--source-display", default="ScenicNL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest and (not args.scenic_path or not args.reference):
        raise SystemExit("Single-scene mode requires --scenic-path and --reference.")
    specs = load_manifest(Path(args.manifest).resolve()) if args.manifest else [{
        "scenic_path": args.scenic_path, "reference": args.reference,
    }]
    output_dir = Path(args.output_dir).resolve()
    reports = []
    for index, spec in enumerate(specs):
        scenic_path = Path(str(spec.get("scenic_path") or "")).resolve()
        reference_path = Path(str(spec.get("reference") or "")).resolve()
        scene_id = str(spec.get("scene_id") or scenic_path.stem or f"scene_{index:04d}")
        try:
            report = evaluate_program(
                scenic_path, reference_path,
                conda_executable=args.conda_executable, conda_env=args.conda_env,
                samples=max(1, int(spec.get("samples") or args.samples)),
                max_iterations=max(1, args.max_iterations), seed=args.seed + index * 10000,
                timeout=max(1, args.timeout),
                scenic_pythonpath=args.scenic_pythonpath,
                source_id=args.source_id,
                source_display=args.source_display,
            )
            report["scene_id"] = scene_id
            output_path = output_dir / f"{scene_id}_{args.source_id}_acrs.json"
            report["output_path"] = str(output_path)
            write_json(output_path, report)
            print(
                f"{scene_id}: compiled={report['compiled']} samples="
                f"{report['successful_samples']}/{report['requested_samples']} "
                f"ACRS={report['aggregates']['acrs']['mean']} (diagnostic)"
            )
        except (OSError, ValueError, RuntimeError, ACRSReferenceError, json.JSONDecodeError) as exc:
            report = {"scene_id": scene_id, "status": "error", "official": False, "error": str(exc)}
            print(f"{scene_id}: ERROR: {exc}", file=sys.stderr)
        reports.append(report)
    write_batch_summary(reports, output_dir, source_id=args.source_id)
    return 1 if any(report.get("status") == "error" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
