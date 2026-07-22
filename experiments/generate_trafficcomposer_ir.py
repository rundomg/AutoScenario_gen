#!/usr/bin/env python3
"""Generate TrafficComposer textual IR using AutoScenario's API configuration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TRAFFICCOMPOSER_ROOT = WORKSPACE_ROOT / "TrafficComposer"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAFFICCOMPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAFFICCOMPOSER_ROOT))

from agents.task_agent import (
    OPENAI_CONNECT_TIMEOUT,
    OPENAI_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_REQUEST_RETRIES,
    OPENAI_TIMEOUT,
    OPENAI_URL,
)
from trafficcomposer.gen_textual_ir.gen_textual_ir import post_process
from trafficcomposer.gen_textual_ir.text_parser_gen_prompt import gen_prompt

REFERENCE_IMAGES = {
    "nexar_00003_t016900": WORKSPACE_ROOT / "AutoScenario_gen/data/00000_00099_frames/00003/frames/00003_00_t016.900s_f000507.jpg",
    "nexar_00006_t017133": WORKSPACE_ROOT / "AutoScenario_gen/data/00000_00099_frames/00006/frames/00006_00_t017.133s_f000514.jpg",
    "nexar_00160_t017567": WORKSPACE_ROOT / "AutoScenario_gen/data/batch_video_reconstruction/效果较好/00160/sampled/frames/00160_00_t017.567s_f000527.jpg",
    "nexar_00213_t016020": WORKSPACE_ROOT / "AutoScenario_gen/data/batch_video_reconstruction/效果较好/00213/sampled/frames/00213_00_t016.020s_f000479.jpg",
    "nexar_00234_t006633": WORKSPACE_ROOT / "AutoScenario_gen/data/batch_video_reconstruction/效果较好/00234/sampled/frames/00234_00_t006.633s_f000199.jpg",
}


def api_base_url(chat_completions_url: str) -> str:
    suffix = "/chat/completions"
    normalized = str(chat_completions_url or "").rstrip("/")
    return normalized[:-len(suffix)] if normalized.endswith(suffix) else normalized


def build_client() -> OpenAI:
    missing = [
        name for name, value in (
            ("OPENAI_KEY", OPENAI_KEY),
            ("OPENAI_URL", OPENAI_URL),
            ("OPENAI_MODEL", OPENAI_MODEL),
        ) if not value
    ]
    if missing:
        raise RuntimeError(f"Missing AutoScenario API configuration: {', '.join(missing)}")
    return OpenAI(
        api_key=OPENAI_KEY,
        base_url=api_base_url(OPENAI_URL),
        timeout=(float(OPENAI_CONNECT_TIMEOUT), float(OPENAI_TIMEOUT)),
        max_retries=max(0, int(OPENAI_REQUEST_RETRIES)),
    )


def generate_one(description: str, client: Any, messages: Any = None) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages if messages is not None else gen_prompt(description),
        max_tokens=OPENAI_MAX_TOKENS,
    )
    raw = response.choices[0].message.content
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("TrafficComposer API returned empty content.")
    processed = post_process(raw)
    if not processed:
        raise RuntimeError("TrafficComposer output did not contain a valid <YAML> block.")
    return raw, processed


def compact_multimodal_messages(messages: list[dict]) -> list[dict]:
    """Drop only few-shot images while retaining their text and target image.

    TrafficComposer's two bundled PNG demonstrations total roughly 12 MB and
    exceeded the configured proxy's practical request latency.  Their textual
    descriptions and gold YAML remain in the prompt; the final target image is
    always preserved.
    """
    compact = []
    last_index = len(messages) - 1
    for index, message in enumerate(messages):
        row = dict(message)
        content = row.get("content")
        if index != last_index and isinstance(content, list):
            row["content"] = [item for item in content if item.get("type") != "image_url"]
        compact.append(row)
    return compact


def description_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description-path",
        default=str(WORKSPACE_ROOT / "scenicNL" / "eval_txts" / "image_txts"),
        help="A .txt description or a folder of descriptions.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--mode", choices=("textual", "multimodal"), default="textual")
    parser.add_argument(
        "--reprocess-raw", action="store_true",
        help="Re-run TrafficComposer post_process on cached raw responses without an API call.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.description_path).resolve()
    default_name = "scenicnl_image_txts_multimodal" if args.mode == "multimodal" else "scenicnl_image_txts"
    output_dir = Path(args.output_dir or (TRAFFICCOMPOSER_ROOT / "results" / default_name)).resolve()
    raw_dir = output_dir / "_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = description_files(source)
    if not files:
        raise SystemExit(f"No .txt descriptions found in {source}")
    client = None if args.reprocess_raw else build_client()
    multimodal_prompt = None
    if args.mode == "multimodal" and not args.reprocess_raw:
        from trafficcomposer.baseline.multi_modal_gpt.multi_modal_gen_prompt import gen_multi_modal_prompt
        multimodal_prompt = gen_multi_modal_prompt
    reports = []
    for path in files:
        output_path = output_dir / f"{path.stem}.yaml"
        raw_path = raw_dir / f"{path.stem}.txt"
        started = time.time()
        if args.reprocess_raw:
            if not raw_path.is_file():
                reports.append({"scene_id": path.stem, "status": "error", "error": f"Missing raw response: {raw_path}"})
                print(f"{path.stem}: ERROR: missing {raw_path}", file=sys.stderr)
                continue
            processed = post_process(raw_path.read_text(encoding="utf-8"))
            if not processed:
                reports.append({"scene_id": path.stem, "status": "error", "error": "Invalid cached raw response"})
                continue
            output_path.write_text(processed, encoding="utf-8")
            reports.append({"scene_id": path.stem, "status": "reprocessed", "output": str(output_path)})
            print(f"{path.stem}: reprocessed {output_path}")
            continue
        if output_path.is_file() and not args.overwrite:
            reports.append({"scene_id": path.stem, "status": "skipped", "output": str(output_path)})
            print(f"{path.stem}: skipped (already exists)")
            continue
        try:
            description = path.read_text(encoding="utf-8")
            messages = None
            if multimodal_prompt is not None:
                image_path = REFERENCE_IMAGES.get(path.stem)
                if image_path is None or not image_path.is_file():
                    raise FileNotFoundError(f"Reference image unavailable for {path.stem}: {image_path}")
                messages = compact_multimodal_messages(
                    multimodal_prompt(description, str(image_path), detail="high")
                )
            raw, processed = generate_one(description, client, messages=messages)
            raw_path.write_text(raw, encoding="utf-8")
            output_path.write_text(processed, encoding="utf-8")
            reports.append({
                "scene_id": path.stem, "status": "generated", "output": str(output_path),
                "raw_output": str(raw_path), "duration_s": round(time.time() - started, 3),
                "model": OPENAI_MODEL,
            })
            print(f"{path.stem}: generated {output_path}")
        except Exception as exc:
            reports.append({
                "scene_id": path.stem, "status": "error", "error": str(exc),
                "duration_s": round(time.time() - started, 3), "model": OPENAI_MODEL,
            })
            print(f"{path.stem}: ERROR: {exc}", file=sys.stderr)
    (output_dir / "generation_summary.json").write_text(
        json.dumps({"model": OPENAI_MODEL, "mode": args.mode, "scenes": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if any(row["status"] == "error" for row in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
