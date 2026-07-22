#!/usr/bin/env python3
"""Generate a Scenic program from text using AutoScenario's configured API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.task_agent import TaskAgent


class ScenicTextAgent(TaskAgent):
    def refine_request(self, user_request=None, add_info=None):
        auto_map = bool(add_info.get("auto_map"))
        if auto_map:
            maps_dir = add_info["maps_dir"]
            map_instruction = f"""Select the best matching CARLA town yourself and use its
absolute OpenDRIVE path under {maps_dir!r}:
- Town01: small town, T-junctions, mixed buildings and small bridges.
- Town02: small residential/commercial town with many T-junctions and a park.
- Town03: dense downtown with roundabout, underpasses and overpasses.
- Town04: small town with a figure-eight multilane road.
- Town05: urban environment with a raised highway and large multilane roads/junctions.
- Town06: low-density environment with large 4--6 lane roads and special junctions.

The program preamble must be:
param map = '<absolute path to the selected TownXX.xodr>'
param carla_map = '<selected TownXX>'
model scenic.simulators.carla.model"""
        else:
            map_name = add_info["map_name"]
            map_path = add_info["map_path"]
            map_instruction = f"""Required preamble:
param map = {map_path!r}
param carla_map = {map_name!r}
model scenic.simulators.carla.model"""
        return f"""You are an expert in Scenic 3 for CARLA. Return only one complete
Scenic program, with no Markdown fences and no prose before or after the code.
The program will be compiled with mode2D=True and sampled without starting CARLA.

{map_instruction}

Use fixed numeric values in behavior conditions and terminate conditions. Anchor
vehicles to a sampled road lane using patterns such as:
lane = Uniform(*network.lanes)
anchor = new OrientedPoint on lane.centerline
lead = new Car at anchor
ego = new Car following roadDirection from lead for -12
Do not access laneToLeft/laneToRight on a Lane, do not use bare separator lines,
and express RGB components in the Scenic-required range 0 through 1. Keep
off-road context objects optional or give them regionContainedIn None,
requireVisible False, and allowCollisions True. Use Scenic 3 syntax only.
Represent the essential road setting, weather, ego vehicle, relevant actors,
and their spatial relationships. When the description is near a junction,
prefer sampling from network.intersections or an approach lane instead of an
unconstrained random lane. Think through objects, relations, missing values,
map choice, and behavior before emitting the code, but do not print that
reasoning.

Natural-language scene description:
{user_request}
"""


def _strip_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        value = "\n".join(lines).strip()
    return value + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--map-name")
    parser.add_argument("--map-path")
    parser.add_argument("--auto-map", action="store_true")
    parser.add_argument(
        "--maps-dir",
        default="/home/zx/code/autodirve/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive",
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.auto_map and (not args.map_name or not args.map_path):
        parser.error("--map-name and --map-path are required unless --auto-map is used")

    response = ScenicTextAgent().send_request(
        args.input_path.read_text(encoding="utf-8"),
        {
            "map_name": args.map_name,
            "map_path": args.map_path,
            "auto_map": args.auto_map,
            "maps_dir": args.maps_dir,
            "request_timeout": args.timeout,
            "request_retries": 1,
            "request_label": args.input_path.stem,
        },
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(_strip_fences(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
