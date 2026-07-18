#!/usr/bin/env python3
"""Extract deterministic ACRS facts from sampled Scenic scenes.

This module is executed inside the ``scenicNL`` conda environment.  It only
depends on Scenic and the standard library; the parent evaluator runs in the
AutoScenario environment and consumes the JSON written here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import traceback
from pathlib import Path
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position(obj: Any) -> tuple[float, float]:
    value = getattr(obj, "position", (0.0, 0.0))
    return _number(value[0]), _number(value[1])


def _category(obj: Any) -> str:
    name = type(obj).__name__.lower()
    blueprint = str(getattr(obj, "blueprint", "") or "").lower()
    text = f"{name} {blueprint}"
    for needle, category in (
        ("pedestrian", "pedestrian"), ("walker", "pedestrian"),
        ("motorcycle", "motorcycle"), ("bike", "bicycle"),
        ("bicycle", "bicycle"), ("truck", "truck"), ("bus", "bus"),
        ("car", "car"), ("vehicle", "car"),
    ):
        if needle in text:
            return category
    return name or "unknown"


def _distance_band(distance: float) -> str:
    if distance <= 10.0:
        return "near"
    if distance <= 30.0:
        return "mid"
    return "far"


def _heading_relation(ego_heading: float, actor_heading: float) -> str:
    delta = abs((actor_heading - ego_heading + math.pi) % (2 * math.pi) - math.pi)
    if delta <= math.radians(30):
        return "same_direction"
    if delta >= math.radians(150):
        return "opposite_direction"
    return "crossing"


def _lane_assignment(network: Any, ego_lane: Any, obj: Any, lateral: float) -> str:
    try:
        lane = network.laneAt(getattr(obj, "position"), reject=False)
    except Exception:
        lane = None
    if lane is not None and ego_lane is not None:
        if lane is ego_lane:
            return "same_lane"
        if getattr(lane, "road", None) is getattr(ego_lane, "road", None):
            if getattr(lane, "group", None) is not getattr(ego_lane, "group", None):
                return "opposing_lane"
    if abs(lateral) <= 1.75:
        return "same_lane"
    return "right_lane" if lateral > 0 else "left_lane"


def _weather(params: dict) -> dict:
    value = params.get("weather", "unknown")
    preset = str("unknown" if value is None else value).lower()
    if "options(" in preset:
        preset = "unknown"
    known = preset != "unknown"
    return {
        "weather": ("rain" if "rain" in preset else "clear") if known else "unknown",
        "lighting": ("night" if "night" in preset else "daylight") if known else "unknown",
        "time_of_day": ("night" if "night" in preset else "day") if known else "unknown",
        "road_surface": ("wet" if ("rain" in preset or "wet" in preset) else "dry") if known else "unknown",
        "urban_density": "unknown",
        "roadside_context_left": "unknown",
        "roadside_context_right": "unknown",
        "landmarks_and_controls": "unknown",
    }


def _road_facts(network: Any, ego: Any) -> dict:
    unknown = {
        "topology_type": "unknown", "directionality": "unknown",
        "forward_lane_count": "unknown", "opposing_lane_count": "unknown",
        "ego_lane_from_right": "unknown", "has_center_median": "unknown",
        "left_parking_presence": "unknown", "right_parking_presence": "unknown",
        "junction_visible": "unknown", "junction_type": "unknown",
        "junction_branches": "unknown", "branch_count": "unknown",
    }
    if network is None:
        return unknown
    position = getattr(ego, "position", None)
    try:
        lane = network.laneAt(position, reject=False)
        road = network.roadAt(position, reject=False)
        junction = network.intersectionAt(position, reject=False)
    except Exception:
        return unknown

    facts = dict(unknown)
    if junction is not None:
        road_count = len(getattr(junction, "roads", ()) or ())
        topology = "t_junction" if road_count == 3 else "cross_intersection" if road_count == 4 else "multi_branch"
        maneuvers = getattr(junction, "maneuvers", ()) or ()
        maneuver_names = {str(getattr(item, "type", "")).lower() for item in maneuvers}
        branches = {
            "ahead": any("straight" in name for name in maneuver_names),
            "left": any("left" in name for name in maneuver_names),
            "right": any("right" in name for name in maneuver_names),
            "uturn": any("u_turn" in name or "uturn" in name for name in maneuver_names),
        }
        facts.update({
            "topology_type": topology, "junction_visible": True,
            "junction_type": topology, "junction_branches": branches,
            "branch_count": sum(branches.values()),
        })
    elif road is not None:
        facts.update({
            "topology_type": "straight", "junction_visible": False,
            "junction_type": "none", "junction_branches": {}, "branch_count": 0,
        })

    group = getattr(lane, "group", None) if lane is not None else None
    # ``LaneGroup.opposite`` rejects on one-way roads; use the nullable backing
    # field since absence of an opposing group is valid evidence here.
    opposite = getattr(group, "_opposite", None) if group is not None else None
    if callable(opposite):
        opposite = opposite()
    forward = len(getattr(group, "lanes", ()) or ()) if group is not None else None
    opposing = len(getattr(opposite, "lanes", ()) or ()) if opposite is not None else 0 if group is not None else None
    if forward is not None:
        facts["forward_lane_count"] = forward
        facts["opposing_lane_count"] = opposing
        facts["directionality"] = "two_way" if opposing else "one_way"
        try:
            facts["ego_lane_from_right"] = list(group.lanes).index(lane) + 1
        except (ValueError, AttributeError):
            pass
    return facts


def scene_to_candidate(scene: Any, scenario: Any) -> dict:
    ego = scene.egoObject
    ex, ey = _position(ego)
    heading = _number(getattr(ego, "heading", 0.0))
    # Scenic heading zero points north (+y); positive lateral is ego-right.
    forward = (-math.sin(heading), math.cos(heading))
    right = (math.cos(heading), math.sin(heading))
    network = getattr(getattr(scenario, "workspace", None), "network", None)
    try:
        ego_lane = network.laneAt(ego.position, reject=False) if network else None
    except Exception:
        ego_lane = None

    actors = []
    actor_index = 0
    for obj in scene.objects:
        if obj is ego:
            continue
        actor_index += 1
        x, y = _position(obj)
        dx, dy = x - ex, y - ey
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * right[0] + dy * right[1]
        distance = math.hypot(dx, dy)
        blueprint = str(getattr(obj, "blueprint", "") or "unknown")
        actors.append({
            "id": f"actor_{actor_index:03d}",
            "category": _category(obj),
            "subtype": blueprint,
            "lane_assignment": _lane_assignment(network, ego_lane, obj, lateral),
            "lane_from_right": "unknown",
            "branch_assignment": "unknown",
            "heading_relation": _heading_relation(heading, _number(getattr(obj, "heading", heading))),
            "longitudinal_relation": "ahead" if longitudinal > 1 else "behind" if longitudinal < -1 else "aligned",
            "distance_band": _distance_band(distance),
            "longitudinal_m": round(longitudinal, 4),
            "lateral_m": round(lateral, 4),
            "position": {"x": round(x, 4), "y": round(y, 4)},
        })
    scene_params = getattr(scene, "params", None)
    if scene_params is None:
        scene_params = getattr(scenario, "params", {})
    return {
        "road_topology": _road_facts(network, ego),
        "environment": _weather(dict(scene_params)),
        "environment_sources": {},
        "actors": actors,
        "evidence_conflicts": [],
        "visual_observation_available": False,
        "runtime_truth_available": False,
    }


def extract(path: Path, samples: int, max_iterations: int, seed: int) -> dict:
    import scenic

    # Scenic modules can construct distributions while they are being loaded.
    # Seed before compilation as well as before each sample so repeated
    # evaluations are fully deterministic.
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass

    result = {
        "schema_version": "scenicnl-sampled-evidence-v1",
        "scenic_path": str(path.resolve()),
        "compiled": False,
        "compile_error": None,
        "requested_samples": samples,
        "samples": [],
    }
    try:
        try:
            scenario = scenic.scenarioFromFile(str(path.resolve()), mode2D=True)
        except TypeError as exc:
            # ChatScene vendors Scenic 2.1, whose public loader predates the
            # ``mode2D`` keyword.  Falling back only for that API mismatch
            # keeps the extractor usable with both Scenic 2.1 and 3.x.
            if "mode2D" not in str(exc):
                raise
            scenario = scenic.scenarioFromFile(str(path.resolve()))
        result["compiled"] = True
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
        return result

    for index in range(samples):
        sample_seed = seed + index
        random.seed(sample_seed)
        try:
            import numpy as np
            np.random.seed(sample_seed % (2**32 - 1))
        except ImportError:
            pass
        try:
            scene, iterations = scenario.generate(maxIterations=max_iterations)
            result["samples"].append({
                "sample_index": index, "seed": sample_seed, "success": True,
                "iterations": iterations, "candidate": scene_to_candidate(scene, scenario),
            })
        except Exception as exc:
            result["samples"].append({
                "sample_index": index, "seed": sample_seed, "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenic_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = extract(args.scenic_path, max(1, args.samples), max(1, args.max_iterations), args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
