import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.scene_understanding_interpreter import SceneUnderstandingInterpreter
from agents.task_agent import OPENAI_MODEL
from agents.vehicle_orientation_interpreter import VehicleOrientationInterpreter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_GLOBS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the first scene-understanding module on image batches. "
            "Each sample gets its own prompt, raw response, parsed JSON, and review summary."
        )
    )
    parser.add_argument("--image-dir", default=None, help="Folder containing input images.")
    parser.add_argument("--images", nargs="*", default=[], help="Image files or folders to scan.")
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Image glob for folders. Repeatable. Defaults to common image suffixes.",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan image folders recursively.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images after sorting.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "scene_understanding_eval"),
        help="Root output folder for evaluation runs.",
    )
    parser.add_argument("--run-name", default=None, help="Run folder name. Default uses timestamp.")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Editable road-scene prompt file for split/road-only mode, or legacy prompt in single-agent mode.",
    )
    parser.add_argument(
        "--orientation-prompt-file",
        default=None,
        help="Editable vehicle-orientation prompt file used by split mode.",
    )
    parser.add_argument(
        "--init-prompt-file",
        default=None,
        help="Export a built-in prompt to this file and exit.",
    )
    parser.add_argument(
        "--init-prompt-kind",
        choices=["road_scene", "legacy", "orientation"],
        default="road_scene",
        help="Which built-in prompt to export with --init-prompt-file.",
    )
    parser.add_argument(
        "--single-agent",
        action="store_true",
        help="Run the legacy single scene-understanding prompt instead of split mode.",
    )
    parser.add_argument(
        "--road-scene-only",
        action="store_true",
        help="Run only the road-scene submodule prompt and output.",
    )
    parser.add_argument(
        "--no-vehicle-detector",
        action="store_true",
        help="Disable detector/crop/orientation evidence in modes that normally use it.",
    )
    parser.add_argument(
        "--user-input",
        default="",
        help="Optional authoritative scene description appended to the prompt.",
    )
    parser.add_argument(
        "--user-input-file",
        default=None,
        help="Read the authoritative scene description from a text file.",
    )
    parser.add_argument(
        "--user-request",
        default="",
        help="Optional extra request appended as 'User request'.",
    )
    parser.add_argument(
        "--expectations-file",
        default=None,
        help=(
            "Optional JSON mapping image stem/name/path to expected dotted fields, "
            "for example {\"0107\": {\"road_network.map_matching.topology_type\": \"straight_road\"}}."
        ),
    )
    parser.add_argument("--request-timeout", type=int, default=180, help="VLM read timeout seconds.")
    parser.add_argument("--request-max-tokens", type=int, default=8000, help="Scene response token cap.")
    parser.add_argument(
        "--request-retries",
        type=int,
        default=None,
        help="Retry count passed to TaskAgent. Default uses OPENAI_REQUEST_RETRIES.",
    )
    parser.add_argument("--no-copy-images", action="store_true", help="Do not copy images into sample folders.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed sample.")
    args = parser.parse_args()
    if args.single_agent and args.road_scene_only:
        parser.error("--single-agent and --road-scene-only are mutually exclusive.")
    return args


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def read_text_file(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def default_prompt(kind: str) -> str:
    if kind == "orientation":
        return VehicleOrientationInterpreter().pre_prompt.strip()
    agent = SceneUnderstandingInterpreter()
    if kind == "legacy":
        return agent.pre_prompt.strip()
    return agent.road_scene_prompt.strip()


def collect_from_dir(folder: Path, patterns: list[str], recursive: bool) -> list[Path]:
    images = []
    for pattern in patterns:
        iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
        images.extend(path for path in iterator if path.is_file())
    return images


def discover_images(args: argparse.Namespace) -> list[Path]:
    patterns = args.glob or DEFAULT_GLOBS
    candidates = []
    if args.image_dir:
        candidates.extend(collect_from_dir(Path(args.image_dir), patterns, args.recursive))
    for item in args.images:
        path = Path(item)
        if path.is_dir():
            candidates.extend(collect_from_dir(path, patterns, args.recursive))
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(path)

    seen = set()
    images = []
    for path in sorted(candidates, key=lambda p: str(p).lower()):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        images.append(resolved)
    return images[: args.max_images] if args.max_images is not None else images


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return cleaned.strip("_") or "image"


def active_mode(args: argparse.Namespace) -> str:
    if args.road_scene_only:
        return "road_scene_only"
    if args.single_agent:
        return "single_agent"
    return "split_scene_agents"


def prompt_kind_for_run(args: argparse.Namespace) -> str:
    return "legacy" if args.single_agent else "road_scene"


def load_expectations(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Expectations file must be a JSON object.")
    return payload


def expectation_for_image(expectations: dict, image_path: Path) -> dict:
    candidates = [
        str(image_path),
        str(image_path.resolve()),
        image_path.name,
        image_path.stem,
    ]
    for key in candidates:
        value = expectations.get(key)
        if isinstance(value, dict):
            return value
    return {}


def get_dotted(payload: dict, dotted_path: str):
    value = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def evaluate_expectations(payload: dict, expected: dict) -> dict:
    checks = []
    passed = True
    for dotted_path, expected_value in sorted(expected.items()):
        actual = get_dotted(payload, dotted_path)
        ok = actual == expected_value
        passed = passed and ok
        checks.append(
            {
                "path": dotted_path,
                "expected": expected_value,
                "actual": actual,
                "passed": ok,
            }
        )
    return {"passed": passed, "checks": checks, "count": len(checks)}


def summarize_payload(payload: dict, expectation_report: dict | None = None) -> dict:
    warnings = []
    required = [
        "traffic_subjects",
        "key_pairwise_relations",
        "road_network",
        "actor_layout",
        "general_environment",
        "metadata",
    ]
    for key in required:
        if key not in payload:
            warnings.append(f"missing top-level key: {key}")

    road_network = payload.get("road_network") if isinstance(payload, dict) else {}
    if not isinstance(road_network, dict):
        road_network = {}
        warnings.append("road_network is not an object")
    map_matching = road_network.get("map_matching") or {}
    if not isinstance(map_matching, dict):
        map_matching = {}
        warnings.append("road_network.map_matching is not an object")

    subjects = payload.get("traffic_subjects") or []
    if not isinstance(subjects, list):
        subjects = []
        warnings.append("traffic_subjects is not a list")
    actor_ids = [str(item.get("id") or "") for item in subjects if isinstance(item, dict)]
    duplicate_ids = sorted({actor_id for actor_id in actor_ids if actor_ids.count(actor_id) > 1 and actor_id})
    if duplicate_ids:
        warnings.append(f"duplicate actor ids: {', '.join(duplicate_ids)}")

    category_counts = {}
    unknown_heading_count = 0
    for item in subjects:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        if str(item.get("heading_relation_to_ego") or "unknown") == "unknown":
            unknown_heading_count += 1

    summary = {
        "topology_type": map_matching.get("topology_type"),
        "junction_visible": map_matching.get("junction_visible"),
        "junction_type": map_matching.get("junction_type"),
        "forward_lane_count": map_matching.get("forward_lane_count"),
        "opposing_lane_count": map_matching.get("opposing_lane_count"),
        "ego_lane_from_right": map_matching.get("ego_lane_from_right"),
        "actor_count": len(subjects),
        "category_counts": category_counts,
        "unknown_heading_count": unknown_heading_count,
        "warnings": warnings,
    }
    if expectation_report is not None:
        summary["expectations"] = expectation_report
    return summary


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "scene_id",
        "status",
        "image_path",
        "sample_dir",
        "prompt_path",
        "output_json",
        "topology_type",
        "junction_visible",
        "junction_type",
        "actor_count",
        "expectations_passed",
        "duration_s",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_add_info(
    *,
    args: argparse.Namespace,
    scene_id: str,
    image_path: Path,
    sample_dir: Path,
    output_fn: Path,
    prompt_file: Path | None,
    orientation_prompt_file: Path | None,
    user_scene_description: str,
) -> dict:
    add_info = {
        "scene_id": scene_id,
        "image_path": str(image_path),
        "output_fn": str(output_fn),
        "prompt_output_fn": str(sample_dir / "prompt.txt"),
        "split_scene_agents": active_mode(args) == "split_scene_agents",
        "use_vehicle_detector": not args.no_vehicle_detector,
        "request_timeout": args.request_timeout,
        "request_max_tokens": args.request_max_tokens,
    }
    if prompt_file:
        add_info["prompt_file"] = str(prompt_file)
    if orientation_prompt_file:
        add_info["orientation_prompt_file"] = str(orientation_prompt_file)
    if user_scene_description:
        add_info["user_scene_description"] = user_scene_description
    if args.request_retries is not None:
        add_info["request_retries"] = args.request_retries

    mode = active_mode(args)
    if mode == "split_scene_agents":
        add_info["road_scene_raw_output_fn"] = str(sample_dir / "road_scene_raw_response.txt")
        add_info["orientation_raw_output_fn"] = str(sample_dir / "orientation_raw_response.txt")
        add_info["orientation_prompt_output_fn"] = str(sample_dir / "orientation_prompt.txt")
    else:
        add_info["raw_output_fn"] = str(sample_dir / "raw_response.txt")
    if args.road_scene_only:
        add_info["road_scene_only"] = True
    return add_info


def run_sample(
    *,
    interpreter: SceneUnderstandingInterpreter,
    args: argparse.Namespace,
    image_path: Path,
    sample_dir: Path,
    scene_id: str,
    prompt_file: Path | None,
    orientation_prompt_file: Path | None,
    user_request: str,
    user_scene_description: str,
    expectations: dict,
) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)
    output_name = "road_scene.json" if args.road_scene_only else "scene_understanding.json"
    output_fn = sample_dir / output_name

    copied_image = None
    if not args.no_copy_images:
        copied_image = sample_dir / f"input{image_path.suffix.lower()}"
        shutil.copy2(image_path, copied_image)

    add_info = build_add_info(
        args=args,
        scene_id=scene_id,
        image_path=image_path,
        sample_dir=sample_dir,
        output_fn=output_fn,
        prompt_file=prompt_file,
        orientation_prompt_file=orientation_prompt_file,
        user_scene_description=user_scene_description,
    )
    write_json(
        sample_dir / "sample_manifest.json",
        {
            "scene_id": scene_id,
            "mode": active_mode(args),
            "source_image": str(image_path),
            "copied_image": str(copied_image) if copied_image else None,
            "prompt_file": str(prompt_file) if prompt_file else None,
            "orientation_prompt_file": str(orientation_prompt_file) if orientation_prompt_file else None,
            "prompt_path": str(sample_dir / "prompt.txt"),
            "orientation_prompt_path": str(sample_dir / "orientation_prompt.txt")
            if active_mode(args) == "split_scene_agents"
            else None,
            "output_json": str(output_fn),
            "model": OPENAI_MODEL,
            "use_vehicle_detector": not args.no_vehicle_detector,
            "user_request": user_request,
            "has_user_scene_description": bool(user_scene_description),
        },
    )

    start = time.perf_counter()
    if args.road_scene_only:
        payload = interpreter._call_road_scene_agent(user_request, add_info)
    else:
        payload = interpreter.call_agent(user_request, add_info)
    duration_s = round(time.perf_counter() - start, 3)

    expected = expectation_for_image(expectations, image_path)
    expectation_report = evaluate_expectations(payload, expected) if expected else None
    review_summary = summarize_payload(payload, expectation_report)
    write_json(sample_dir / "review_summary.json", review_summary)

    return {
        "scene_id": scene_id,
        "status": "ok",
        "image_path": str(image_path),
        "sample_dir": str(sample_dir),
        "prompt_path": str(sample_dir / "prompt.txt"),
        "output_json": str(output_fn),
        "topology_type": review_summary.get("topology_type"),
        "junction_visible": review_summary.get("junction_visible"),
        "junction_type": review_summary.get("junction_type"),
        "actor_count": review_summary.get("actor_count"),
        "expectations_passed": ""
        if expectation_report is None
        else str(bool(expectation_report.get("passed"))).lower(),
        "duration_s": duration_s,
        "error": "",
    }


def main() -> int:
    args = parse_args()
    if args.init_prompt_file:
        target = Path(args.init_prompt_file)
        write_text(target, default_prompt(args.init_prompt_kind) + "\n")
        print(f"Wrote editable {args.init_prompt_kind} prompt to {target.resolve()}")
        return 0

    images = discover_images(args)
    if not images:
        raise SystemExit("No input images found. Provide --image-dir or --images.")

    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None
    orientation_prompt_file = (
        Path(args.orientation_prompt_file).resolve() if args.orientation_prompt_file else None
    )
    for path, label in ((prompt_file, "prompt"), (orientation_prompt_file, "orientation prompt")):
        if path and not path.exists():
            raise SystemExit(f"{label.capitalize()} file not found: {path}")

    expectations = load_expectations(args.expectations_file)
    user_scene_description = "\n\n".join(
        part
        for part in [args.user_input.strip(), read_text_file(args.user_input_file)]
        if part
    )

    run_name = args.run_name or f"scene_understanding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_prompt = (
        prompt_file.read_text(encoding="utf-8").strip()
        if prompt_file
        else default_prompt(prompt_kind_for_run(args))
    )
    write_text(run_dir / "base_prompt.txt", base_prompt + "\n")
    if orientation_prompt_file:
        write_text(
            run_dir / "orientation_base_prompt.txt",
            orientation_prompt_file.read_text(encoding="utf-8").strip() + "\n",
        )
    elif active_mode(args) == "split_scene_agents":
        write_text(run_dir / "orientation_base_prompt.txt", default_prompt("orientation") + "\n")
    if user_scene_description:
        write_text(run_dir / "user_scene_description.txt", user_scene_description + "\n")
    if args.user_request.strip():
        write_text(run_dir / "user_request.txt", args.user_request.strip() + "\n")

    write_json(
        run_dir / "run_config.json",
        {
            "mode": active_mode(args),
            "image_count": len(images),
            "images": [str(path) for path in images],
            "prompt_file": str(prompt_file) if prompt_file else None,
            "orientation_prompt_file": str(orientation_prompt_file) if orientation_prompt_file else None,
            "expectations_file": args.expectations_file,
            "model": OPENAI_MODEL,
            "use_vehicle_detector": not args.no_vehicle_detector,
            "request_timeout": args.request_timeout,
            "request_max_tokens": args.request_max_tokens,
            "request_retries": args.request_retries,
        },
    )

    interpreter = SceneUnderstandingInterpreter()
    summary_jsonl = run_dir / "summary.jsonl"
    rows = []
    print(f"Scene-understanding eval output: {run_dir.resolve()}")
    for index, image_path in enumerate(images, start=1):
        scene_id = f"s{index - 1:04d}_{safe_name(image_path.stem)}"
        sample_dir = run_dir / "samples" / scene_id
        print(f"[{index}/{len(images)}] {image_path}")
        try:
            row = run_sample(
                interpreter=interpreter,
                args=args,
                image_path=image_path,
                sample_dir=sample_dir,
                scene_id=scene_id,
                prompt_file=prompt_file,
                orientation_prompt_file=orientation_prompt_file,
                user_request=args.user_request.strip(),
                user_scene_description=user_scene_description,
                expectations=expectations,
            )
        except Exception as exc:  # noqa: BLE001 - keep diagnostics for failed samples
            sample_dir.mkdir(parents=True, exist_ok=True)
            write_text(sample_dir / "error.txt", str(exc) + "\n")
            row = {
                "scene_id": scene_id,
                "status": "failed",
                "image_path": str(image_path),
                "sample_dir": str(sample_dir),
                "prompt_path": str(sample_dir / "prompt.txt"),
                "output_json": "",
                "topology_type": "",
                "junction_visible": "",
                "junction_type": "",
                "actor_count": "",
                "expectations_passed": "",
                "duration_s": "",
                "error": str(exc),
            }
            print(f"  failed: {exc}")
            if args.fail_fast:
                append_jsonl(summary_jsonl, row)
                rows.append(row)
                break
        append_jsonl(summary_jsonl, row)
        rows.append(row)

    write_csv_summary(run_dir / "summary.csv", rows)
    print(f"Summary: {run_dir / 'summary.csv'}")
    return 0 if rows and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
