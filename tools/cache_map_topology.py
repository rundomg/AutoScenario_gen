#!/usr/bin/env python3
"""Offline per-map topology cache builder.

Connects to a running CARLA server, loads the specified map, extracts waypoint
features using SceneMapMatcher helpers, and saves them to a JSON file.  Run once
per map; the resulting cache lets _match_from_cache score across all maps without
any CARLA connection at inference time.

Usage:
    python tools/cache_map_topology.py --map Town01
    python tools/cache_map_topology.py --map Town03 --sample-step 5.0
    python tools/cache_map_topology.py --map Town10HD_Opt --large-map
    python tools/cache_map_topology.py --map /Game/Carla/Maps/Town12 --large-map

The output is written to:
    data/map_cache/<world_name>.json
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_CACHE_DIR = os.path.join(ROOT, "data", "map_cache")
DEFAULT_ENVIRONMENT_RADIUS_M = 60.0

ENVIRONMENT_LABEL_NAMES = (
    "Buildings",
    "Sidewalks",
    "TrafficSigns",
    "TrafficLight",
    "Poles",
    "Walls",
    "Fences",
    "Vegetation",
    "Terrain",
    "Ground",
    "Water",
)


def _safe_map_filename(world_name: str) -> str:
    return world_name.replace("/", "_").replace("\\", "_") + ".json"


def _environment_object_location(obj):
    transform = getattr(obj, "transform", None)
    location = getattr(transform, "location", None)
    if location is not None:
        return location

    bounding_box = getattr(obj, "bounding_box", None)
    location = getattr(bounding_box, "location", None)
    if location is not None:
        return location
    return None


def _load_environment_points(world, carla_module):
    points = []
    loaded_counts = {}
    for label_name in ENVIRONMENT_LABEL_NAMES:
        label = getattr(carla_module.CityObjectLabel, label_name, None)
        if label is None:
            continue
        try:
            objects = list(world.get_environment_objects(label))
        except Exception as exc:
            print(f"  WARNING: get_environment_objects({label_name}) failed ({exc})")
            continue

        loaded_counts[label_name] = len(objects)
        for obj in objects:
            location = _environment_object_location(obj)
            if location is None:
                continue
            points.append(
                {
                    "label": label_name,
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(getattr(location, "z", 0.0)),
                }
            )
    return points, loaded_counts


def _flatten_crosswalk_locations(raw_crosswalks):
    locations = []
    for item in raw_crosswalks or []:
        if hasattr(item, "x") and hasattr(item, "y"):
            locations.append(item)
            continue
        try:
            locations.extend(
                point
                for point in item
                if hasattr(point, "x") and hasattr(point, "y")
            )
        except TypeError:
            continue
    return locations


def _count_score(count: int, saturation: int) -> float:
    return min(1.0, float(count) / max(1, saturation))


def _summarize_environment_context(location, environment_points, radius_m):
    radius_m = float(radius_m)
    counts = {}
    nearest = {}
    radius_sq = radius_m * radius_m
    loc_x = float(location.x)
    loc_y = float(location.y)

    for point in environment_points:
        dx = point["x"] - loc_x
        dy = point["y"] - loc_y
        distance_sq = dx * dx + dy * dy
        distance = math.sqrt(distance_sq)
        label = point["label"]
        if label not in nearest or distance < nearest[label]:
            nearest[label] = distance
        if distance_sq <= radius_sq:
            counts[label] = counts.get(label, 0) + 1

    traffic_count = counts.get("TrafficSigns", 0) + counts.get("TrafficLight", 0)
    boundary_count = counts.get("Poles", 0) + counts.get("Walls", 0) + counts.get("Fences", 0)
    urban_score = (
        0.35 * _count_score(counts.get("Buildings", 0), 8)
        + 0.25 * _count_score(counts.get("Sidewalks", 0), 4)
        + 0.20 * _count_score(traffic_count, 2)
        + 0.20 * _count_score(boundary_count, 8)
    )
    natural_score = (
        0.35 * _count_score(counts.get("Vegetation", 0), 8)
        + 0.25 * _count_score(counts.get("Terrain", 0), 4)
        + 0.15 * _count_score(counts.get("Ground", 0), 4)
        + 0.25 * _count_score(counts.get("Water", 0), 1)
    )

    if not counts:
        environment_class = "unknown"
    elif urban_score > 0.0 and natural_score == 0.0:
        environment_class = "urban_like"
    elif natural_score > 0.0 and urban_score == 0.0:
        environment_class = "natural_like"
    elif urban_score >= natural_score + 0.20:
        environment_class = "urban_like"
    elif natural_score >= urban_score + 0.20:
        environment_class = "natural_like"
    else:
        environment_class = "mixed"

    nearest_m = {
        label: round(distance, 2)
        for label, distance in sorted(nearest.items())
        if distance <= radius_m * 2.0
    }
    return {
        "radius_m": radius_m,
        "counts": {label: counts[label] for label in sorted(counts)},
        "nearest_m": nearest_m,
        "urban_score": round(urban_score, 3),
        "natural_score": round(natural_score, 3),
        "water_nearby": counts.get("Water", 0) > 0,
        "buildings_nearby": counts.get("Buildings", 0) > 0,
        "sidewalks_nearby": counts.get("Sidewalks", 0) > 0,
        "traffic_control_nearby": traffic_count > 0,
        "environment_class": environment_class,
    }


def _walk_road_neighbors(wp, steps: int, dist_m: float):
    """Return waypoints obtained by walking forward and backward from *wp*.

    Each step calls wp.next(dist_m) / wp.previous(dist_m).  Only the first
    branch is followed at junctions to keep the walk deterministic.  The
    original *wp* is NOT included in the returned list.
    """
    result = []
    for direction in ("next", "previous"):
        current = [wp]
        for _ in range(steps):
            nxt = []
            for cwp in current:
                try:
                    candidates = getattr(cwp, direction)(dist_m)
                except Exception:
                    candidates = []
                if candidates:
                    nxt.append(candidates[0])
            if not nxt:
                break
            result.extend(nxt)
            current = nxt
    return result


def _waypoint_dedup_key(wp, bucket_m: float):
    """Bucket key that merges waypoints closer than *bucket_m* along the same lane."""
    return (wp.road_id, wp.section_id, wp.lane_id, int(wp.s / max(bucket_m, 0.1)))


def build_cache(
    host: str,
    port: int,
    timeout: float,
    map_name: str,
    sample_step: float,
    cache_dir: str,
    large_map: bool,
    road_walk_steps: int = 0,
    road_walk_dist: float = 15.0,
) -> None:
    try:
        import carla
    except ImportError:
        sys.exit("ERROR: carla module not found. Activate the autoscenario conda env.")

    from tools.scene_map_matcher import SceneMapMatcher

    client = carla.Client(host, port)
    client.set_timeout(timeout)
    print(f"Loading map: {map_name} …")
    world = client.load_world(map_name)
    world_map = world.get_map()
    world_name = world_map.name
    print(f"Loaded: {world_name}")

    matcher = SceneMapMatcher(host=host, port=port, timeout=timeout)

    candidates = []
    environment_points, environment_counts = _load_environment_points(world, carla)
    print(
        "  environment objects pre-computed: "
        + ", ".join(
            f"{label}={count}" for label, count in sorted(environment_counts.items())
        )
    )

    # Pre-compute crosswalk centre locations once (avoids per-candidate CARLA calls).
    try:
        crosswalk_locs = _flatten_crosswalk_locations(world_map.get_crosswalks())
        print(f"  {len(crosswalk_locs)} crosswalk vertices pre-computed")
    except Exception as exc:
        crosswalk_locs = []
        print(f"  WARNING: get_crosswalks() failed ({exc}); has_crosswalk_nearby will be False")

    if large_map:
        spawn_points = world_map.get_spawn_points()
        walk_desc = (
            f"road_walk_steps={road_walk_steps}, road_walk_dist={road_walk_dist}m"
            if road_walk_steps > 0
            else "no road walk"
        )
        print(f"  {len(spawn_points)} spawn points (large-map mode, {walk_desc})")

        seen_keys: set = set()

        def _add_waypoint(cwp, location) -> bool:
            """Extract features for *cwp* and append to candidates. Returns True on success."""
            key = _waypoint_dedup_key(cwp, road_walk_dist if road_walk_steps > 0 else 1.0)
            if key in seen_keys:
                return False
            seen_keys.add(key)
            try:
                features = matcher._extract_local_candidate_features(
                    world_map, cwp, crosswalk_locations=crosswalk_locs
                )
                lane_dict = matcher._waypoint_to_lane_dict(cwp)
                environment_context = _summarize_environment_context(
                    location,
                    environment_points,
                    DEFAULT_ENVIRONMENT_RADIUS_M,
                )
            except Exception as exc:
                print(f"  WARNING: waypoint skipped — {exc}")
                return False
            candidates.append(
                {
                    "location": {
                        "x": float(location.x),
                        "y": float(location.y),
                        "z": float(location.z),
                    },
                    "yaw": float(cwp.transform.rotation.yaw),
                    "candidate_lane": lane_dict,
                    "environment_context": environment_context,
                    **features,
                }
            )
            return True

        for i, sp in enumerate(spawn_points):
            wp = world_map.get_waypoint(sp.location, project_to_road=True)
            if wp is None:
                continue

            # Spawn point itself.
            _add_waypoint(wp, sp.location)

            # Road-walk extension: walk forward and backward from each spawn.
            if road_walk_steps > 0:
                for neighbor in _walk_road_neighbors(wp, road_walk_steps, road_walk_dist):
                    _add_waypoint(neighbor, neighbor.transform.location)

            if (i + 1) % 50 == 0:
                print(
                    f"  processed {i + 1}/{len(spawn_points)} spawn points, "
                    f"{len(candidates)} candidates so far"
                )
    else:
        waypoints = world_map.generate_waypoints(sample_step)
        print(f"  {len(waypoints)} waypoints at step={sample_step}m")
        for i, wp in enumerate(waypoints):
            try:
                features = matcher._extract_candidate_features(
                    waypoints, wp, crosswalk_locations=crosswalk_locs
                )
                lane_dict = matcher._waypoint_to_lane_dict(wp)
                environment_context = _summarize_environment_context(
                    wp.transform.location,
                    environment_points,
                    DEFAULT_ENVIRONMENT_RADIUS_M,
                )
            except Exception as exc:
                print(f"  WARNING: waypoint {i} skipped — {exc}")
                continue
            candidates.append(
                {
                    "location": {
                        "x": wp.transform.location.x,
                        "y": wp.transform.location.y,
                        "z": wp.transform.location.z,
                    },
                    "yaw": wp.transform.rotation.yaw,
                    "candidate_lane": lane_dict,
                    "environment_context": environment_context,
                    **features,
                }
            )
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(waypoints)} waypoints")

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, _safe_map_filename(world_name))
    payload = {
        "world_name": world_name,
        "sample_step": sample_step,
        "large_map": large_map,
        "candidate_count": len(candidates),
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates": candidates,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"Saved {len(candidates)} candidates → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build offline topology cache for one CARLA map."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--map", required=True, help="CARLA map name, e.g. Town01")
    parser.add_argument(
        "--sample-step",
        type=float,
        default=5.0,
        help="Waypoint sampling interval in metres (small-map mode only)",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Output directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--large-map",
        action="store_true",
        help="Use spawn-point sampling instead of full waypoint generation",
    )
    parser.add_argument(
        "--road-walk-steps",
        type=int,
        default=0,
        help=(
            "In large-map mode: number of waypoints to walk forward AND backward "
            "from each spawn point along the road (default: 0 = disabled). "
            "E.g. --road-walk-steps 8 adds 16 extra points per spawn."
        ),
    )
    parser.add_argument(
        "--road-walk-dist",
        type=float,
        default=15.0,
        help=(
            "Distance in metres between each road-walk step (default: 15.0). "
            "Also used as the deduplication bucket size."
        ),
    )
    args = parser.parse_args()
    build_cache(
        args.host,
        args.port,
        args.timeout,
        args.map,
        args.sample_step,
        args.cache_dir,
        args.large_map,
        road_walk_steps=args.road_walk_steps,
        road_walk_dist=args.road_walk_dist,
    )
