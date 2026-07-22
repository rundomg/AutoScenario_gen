#!/usr/bin/env python3
"""Generate ChatScene Scenic programs for the shared Nexar ACRS inputs.

This preserves ChatScene's public v1 pipeline: an LLM decomposes each textual
scenario, sentence-t5-large retrieves the nearest behavior/geometry/spawn
snippet, and pure retrieval selects the top-1 snippet in each category.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
CHATSCENE_ROOT = WORKSPACE_ROOT / "ChatScene"
RETRIEVE_ROOT = CHATSCENE_ROOT / "retrieve"
DEFAULT_DESCRIPTIONS = WORKSPACE_ROOT / "scenicNL" / "eval_txts" / "image_txts"
DEFAULT_OUTPUT = (
    CHATSCENE_ROOT / "safebench" / "scenario" / "scenario_data"
    / "scenic_data" / "acrs_nexar"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "annotations" / "chatscene_image_txts" / "chatscene_acrs_manifest.json"
REFERENCE_ROOT = PROJECT_ROOT / "annotations" / "scenicnl_image_txts"
SCENIC21_ROOT = CHATSCENE_ROOT / "Scenic" / "src"

HEAD = """param map = localPath(f'../maps/{Town}.xodr')
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def api_base_url(chat_completions_url: str) -> str:
    suffix = "/chat/completions"
    normalized = str(chat_completions_url or "").rstrip("/")
    return normalized[:-len(suffix)] if normalized.endswith(suffix) else normalized


def parse_extraction(text: str) -> dict[str, str]:
    match = re.search(
        r"Adversarial Object:\s*(.*?)\s*Behavior:\s*(.*?)\s*"
        r"Geometry:\s*(.*?)\s*Spawn Position:\s*(.*)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("ChatScene extraction response did not match the required four fields.")
    adversarial_object, behavior, geometry, spawn = [item.strip() for item in match.groups()]
    allowed = {"car": "Car", "pedestrian": "Pedestrian", "bicycle": "Bicycle", "motorcycle": "Motorcycle"}
    key = adversarial_object.lower().rstrip(".")
    if key not in allowed:
        raise ValueError(f"Unsupported adversarial object from extraction: {adversarial_object!r}")
    return {
        "adversarial_object": allowed[key],
        "behavior": behavior,
        "geometry": geometry,
        "spawn": spawn,
    }


def local_hf_snapshot(model_name: str) -> str:
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots = sorted((Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots").glob("*"))
    return str(snapshots[-1]) if snapshots else model_name


def embedding_worker(input_path: Path, output_path: Path, model_name: str, device: str) -> int:
    from sentence_transformers import SentenceTransformer

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(local_hf_snapshot(model_name), device=device)
    rankings: dict[str, list[dict[str, Any]]] = {}
    for category in ("behavior", "geometry", "spawn"):
        descriptions = payload["database"][category]
        query_rows = payload["queries"][category]
        description_embeddings = model.encode(descriptions, convert_to_tensor=True, device=device)
        query_embeddings = model.encode(query_rows, convert_to_tensor=True, device=device)
        scores = query_embeddings @ description_embeddings.T
        top_scores, top_indices = scores.topk(k=min(int(payload["topk"]), len(descriptions)), dim=1)
        rankings[category] = [
            [
                {"index": int(index), "score": float(score)}
                for index, score in zip(indices, values)
            ]
            for indices, values in zip(top_indices.tolist(), top_scores.tolist())
        ]
    write_json(output_path, {"model": model_name, "rankings": rankings})
    return 0


def run_ranker(
    database: dict,
    components: list[dict[str, str]],
    *,
    python: str,
    model_name: str,
    device: str,
    topk: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="chatscene_retrieval_") as folder:
        root = Path(folder)
        input_path = root / "input.json"
        output_path = root / "output.json"
        write_json(input_path, {
            "topk": topk,
            "database": {
                category: list(database[category]["description"])
                for category in ("behavior", "geometry", "spawn")
            },
            "queries": {
                category: [row[category] for row in components]
                for category in ("behavior", "geometry", "spawn")
            },
        })
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            python, str(Path(__file__).resolve()), "--embedding-worker",
            str(input_path), str(output_path), "--embedding-model", model_name,
            "--embedding-device", device,
        ]
        process = subprocess.run(command, capture_output=True, text=True, env=environment)
        if process.returncode != 0 or not output_path.is_file():
            detail = (process.stderr or process.stdout or "no worker output").strip()
            raise RuntimeError(f"ChatScene embedding worker failed ({process.returncode}): {detail[-4000:]}")
        return json.loads(output_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description-path", default=str(DEFAULT_DESCRIPTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--embedding-python", default="/home/zx/miniconda3/envs/autovfx/bin/python")
    parser.add_argument("--embedding-model", default="sentence-transformers/sentence-t5-large")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding-worker", nargs=2, metavar=("INPUT", "OUTPUT"), help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.embedding_worker:
        return embedding_worker(
            Path(args.embedding_worker[0]), Path(args.embedding_worker[1]),
            args.embedding_model, args.embedding_device,
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from openai import OpenAI
    from agents.task_agent import (
        OPENAI_CONNECT_TIMEOUT,
        OPENAI_KEY,
        OPENAI_MAX_TOKENS,
        OPENAI_MODEL,
        OPENAI_REQUEST_RETRIES,
        OPENAI_TIMEOUT,
        OPENAI_URL,
    )

    missing = [name for name, value in (
        ("OPENAI_KEY", OPENAI_KEY), ("OPENAI_URL", OPENAI_URL), ("OPENAI_MODEL", OPENAI_MODEL),
    ) if not value]
    if missing:
        raise RuntimeError(f"Missing AutoScenario API configuration: {', '.join(missing)}")

    description_root = Path(args.description_path).resolve()
    files = [description_root] if description_root.is_file() else sorted(description_root.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt descriptions found in {description_root}")
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "_raw_extractions"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    extraction_prompt = load_text(RETRIEVE_ROOT / "prompts" / "extraction.txt")
    with (RETRIEVE_ROOT / "database_v1.pkl").open("rb") as stream:
        database = pickle.load(stream)
    client = OpenAI(
        api_key=OPENAI_KEY,
        base_url=api_base_url(OPENAI_URL),
        timeout=(float(OPENAI_CONNECT_TIMEOUT), float(OPENAI_TIMEOUT)),
        max_retries=max(0, int(OPENAI_REQUEST_RETRIES)),
    )

    components: list[dict[str, str]] = []
    generation_rows = []
    for path in files:
        raw_path = raw_dir / f"{path.stem}.txt"
        started = time.time()
        if raw_path.is_file() and not args.overwrite:
            raw = load_text(raw_path)
            source = "cached"
        else:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": extraction_prompt.format(scenario=load_text(path))},
                ],
                temperature=0,
                max_tokens=OPENAI_MAX_TOKENS,
            )
            raw = response.choices[0].message.content
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError(f"Empty extraction response for {path.stem}")
            raw_path.write_text(raw, encoding="utf-8")
            source = "generated"
        component = parse_extraction(raw)
        components.append(component)
        generation_rows.append({
            "scene_id": path.stem, "description": str(path), "raw_extraction": str(raw_path),
            "extraction_source": source, "components": component,
            "duration_s": round(time.time() - started, 3),
        })

    ranked = run_ranker(
        database, components, python=args.embedding_python,
        model_name=args.embedding_model, device=args.embedding_device,
        topk=max(1, args.topk),
    )
    manifest_rows = []
    for row_index, (path, component, report) in enumerate(zip(files, components, generation_rows)):
        selected: dict[str, dict[str, Any]] = {}
        for category in ("behavior", "geometry", "spawn"):
            ranking = ranked["rankings"][category][row_index]
            best = ranking[0]
            index = int(best["index"])
            selected[category] = {
                "index": index,
                "score": best["score"],
                "description": database[category]["description"][index],
                "snippet": database[category]["snippet"][index],
                "topk": ranking,
            }
        geometry = selected["geometry"]["snippet"]
        if "\n" not in geometry or not geometry.lstrip().startswith("Town"):
            raise ValueError(f"Retrieved geometry snippet lacks a Town header for {path.stem}")
        town, geometry_body = geometry.split("\n", 1)
        scenic_code = "\n".join([
            f"'''{load_text(path).strip()}'''", town, HEAD.rstrip(),
            selected["behavior"]["snippet"], geometry_body,
            selected["spawn"]["snippet"].format(AdvObject=component["adversarial_object"]),
        ]) + "\n"
        scenic_path = output_dir / f"{path.stem}.scenic"
        scenic_path.write_text(scenic_code, encoding="utf-8")
        reference = (
            PROJECT_ROOT / "annotations" / "00302" / "s0000_c0_acrs_reference.json"
            if path.stem.startswith("nexar_00302_")
            else REFERENCE_ROOT / f"{path.stem}_acrs_reference.json"
        )
        manifest_rows.append({
            "scene_id": path.stem, "scenic_path": str(scenic_path),
            "reference": str(reference), "samples": 30,
        })
        report.update({"scenic_path": str(scenic_path), "selected": selected})

    write_json(Path(args.manifest).resolve(), {
        "schema_version": "chatscene-acrs-manifest-v1",
        "scenes": manifest_rows,
    })
    write_json(output_dir / "generation_summary.json", {
        "schema_version": "chatscene-generation-summary-v1",
        "pipeline": "ChatScene v1 pure retrieval (top-1 of top-k)",
        "extraction_model": OPENAI_MODEL,
        "embedding_model": args.embedding_model,
        "topk": max(1, args.topk),
        "scenic_version": "2.1.0b4 (ChatScene vendored fork)",
        "scenes": generation_rows,
    })
    print(f"Generated {len(manifest_rows)} ChatScene programs in {output_dir}")
    print(f"Manifest: {Path(args.manifest).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
