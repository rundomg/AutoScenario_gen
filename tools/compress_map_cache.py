#!/usr/bin/env python3
"""Structurally compress an existing topology cache without CARLA."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tools.cache_map_topology import compress_structural_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--max-representatives", type=int, default=3)
    parser.add_argument("--region-size", type=float, default=200.0)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as stream:
        payload = json.load(stream)
    candidates, stats = compress_structural_candidates(
        payload.get("candidates") or [],
        max_representatives=args.max_representatives,
        region_size_m=args.region_size,
    )
    payload["candidates"] = candidates
    payload["candidate_count"] = len(candidates)
    payload["structural_compression"] = stats
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    os.replace(temporary, args.output)
    print(f"{args.input}: {stats['before']} -> {stats['after']} candidates -> {args.output}")


if __name__ == "__main__":
    main()
