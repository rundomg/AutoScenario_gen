#!/usr/bin/env python3
"""Fuse reproducible TrafficComposer text and multimodal-GPT IR artifacts.

This is a fallback for the unpublished/unavailable CLRNet checkpoint path.  It
follows TrafficComposer's intended information split: text supplies behavior
and relations, while the image-backed IR supplies lane indices and additional
visible actors.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def slug(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("_", " ").replace("-", " ")


def known(value: Any) -> bool:
    return slug(value) not in {"", "none", "null", "unknown", "n/a"}


def direction_tokens(actor: dict) -> set[str]:
    text = f"{slug(actor.get('position_relation'))} {slug(actor.get('position_target'))}"
    return {token for token in ("left", "right", "front", "ahead", "behind", "opposite") if token in text}


def actor_similarity(text_actor: dict, visual_actor: dict) -> float:
    score = 0.0
    if slug(text_actor.get("type")) == slug(visual_actor.get("type")):
        score += 3.0
    left, right = direction_tokens(text_actor), direction_tokens(visual_actor)
    score += 2.0 * len(left & right) / max(1, len(left | right))
    if "ego vehicle" in slug(text_actor.get("position_target")) and "ego vehicle" in slug(visual_actor.get("position_target")):
        score += 1.0
    if slug(text_actor.get("current_behavior")) == slug(visual_actor.get("current_behavior")):
        score += 0.5
    return score


def fuse_ir(textual: dict, multimodal: dict) -> dict:
    fused = deepcopy(textual)
    text_participants = fused.setdefault("participant", {})
    visual_participants = multimodal.get("participant") if isinstance(multimodal.get("participant"), dict) else {}

    text_ego = text_participants.get("ego_vehicle") if isinstance(text_participants.get("ego_vehicle"), dict) else {}
    visual_ego = visual_participants.get("ego_vehicle") if isinstance(visual_participants.get("ego_vehicle"), dict) else {}
    if known(visual_ego.get("lane_idx")):
        text_ego["lane_idx"] = visual_ego["lane_idx"]
    text_participants["ego_vehicle"] = text_ego

    text_keys = [key for key in text_participants if key != "ego_vehicle" and isinstance(text_participants[key], dict)]
    visual_keys = [key for key in visual_participants if key != "ego_vehicle" and isinstance(visual_participants[key], dict)]
    available = set(visual_keys)
    for text_key in text_keys:
        ranked = sorted(
            ((actor_similarity(text_participants[text_key], visual_participants[key]), key) for key in available),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 3.0:
            continue
        _, visual_key = ranked[0]
        visual_actor = visual_participants[visual_key]
        if known(visual_actor.get("lane_idx")):
            text_participants[text_key]["lane_idx"] = visual_actor["lane_idx"]
        available.remove(visual_key)

    next_index = 1
    while f"other_actor_{next_index}" in text_participants:
        next_index += 1
    for visual_key in sorted(available):
        text_participants[f"other_actor_{next_index}"] = deepcopy(visual_participants[visual_key])
        next_index += 1

    text_road = fused.setdefault("road_network", {})
    visual_road = multimodal.get("road_network") if isinstance(multimodal.get("road_network"), dict) else {}
    if known(visual_road.get("lane_number")):
        text_road["lane_number"] = visual_road["lane_number"]
    for field in ("road_type", "traffic_sign", "traffic_light"):
        if not known(text_road.get(field)) and known(visual_road.get(field)):
            text_road[field] = visual_road[field]
    fused["fusion_metadata"] = {
        "method": "text_relations_plus_multimodal_lane_indices_v1",
        "visual_backend": "TrafficComposer multi_modal_gpt compact compatibility mode",
        "matched_visual_actors": len(visual_keys) - len(available),
        "added_visual_actors": len(available),
    }
    return fused


def load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-dir", default="/home/zx/code/TrafficComposer/results/scenicnl_image_txts")
    parser.add_argument("--multimodal-dir", default="/home/zx/code/TrafficComposer/results/scenicnl_image_txts_multimodal")
    parser.add_argument("--output-dir", default="/home/zx/code/TrafficComposer/results/scenicnl_image_txts_reproducible_fused")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text_dir, multimodal_dir, output_dir = map(lambda value: Path(value).resolve(), (args.text_dir, args.multimodal_dir, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for text_path in sorted(text_dir.glob("*.yaml")):
        visual_path = multimodal_dir / text_path.name
        if not visual_path.is_file():
            print(f"{text_path.stem}: missing multimodal IR")
            failures += 1
            continue
        fused = fuse_ir(load_yaml(text_path), load_yaml(visual_path))
        output_path = output_dir / text_path.name
        output_path.write_text(yaml.safe_dump(fused, sort_keys=False), encoding="utf-8")
        print(f"{text_path.stem}: fused {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
