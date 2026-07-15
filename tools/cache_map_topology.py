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
from collections import defaultdict
import json
import math
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_CACHE_DIR = os.path.join(ROOT, "data", "map_cache")
DEFAULT_ENVIRONMENT_RADIUS_M = 60.0
DEFAULT_STREET_LIGHT_RADIUS_M = 60.0

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


def _load_street_light_points(world, carla_module):
    """Return positions of CARLA lights explicitly tagged as street lights."""
    light_group = getattr(getattr(carla_module, "LightGroup", None), "Street", None)
    if light_group is None:
        return []
    try:
        lights = world.get_lightmanager().get_all_lights(light_group)
    except Exception as exc:
        print(f"  WARNING: loading street lights failed ({exc})")
        return []

    points = []
    for light in lights:
        location = getattr(light, "location", None)
        if location is not None:
            points.append(location)
    return points


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


def _summarize_environment_context(
    location,
    environment_points,
    radius_m,
    street_light_points=None,
    street_light_radius_m=DEFAULT_STREET_LIGHT_RADIUS_M,
):
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
    street_light_radius_sq = float(street_light_radius_m) ** 2
    has_street_lights = any(
        (float(point.x) - loc_x) ** 2 + (float(point.y) - loc_y) ** 2
        <= street_light_radius_sq
        for point in (street_light_points or [])
    )
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
        "has_street_lights": has_street_lights,
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


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _adaptive_waypoint_sample(
    waypoints,
    *,
    straight_spacing_m: float = 50.0,
    curve_spacing_m: float = 10.0,
    junction_spacing_m: float = 5.0,
    junction_radius_m: float = 80.0,
    curve_yaw_threshold_deg: float = 8.0,
):
    """Reduce a dense CARLA waypoint set without losing structural regions.

    Junctions and their surroundings stay dense, curves use a medium spacing,
    and homogeneous straight lane sections use a coarse spacing.  The first
    and last point of every road/section/lane run are always retained so lane
    starts, ends, and topology transitions cannot disappear.
    """
    if not waypoints:
        return [], {"dense": 0, "curve": 0, "straight": 0, "endpoints": 0}

    junction_cell_m = max(float(junction_radius_m), 1.0)
    junction_grid = {}
    for wp in waypoints:
        if not bool(getattr(wp, "is_junction", False)):
            continue
        loc = wp.transform.location
        cell = (math.floor(float(loc.x) / junction_cell_m), math.floor(float(loc.y) / junction_cell_m))
        junction_grid.setdefault(cell, []).append((float(loc.x), float(loc.y)))

    def _near_junction(wp) -> bool:
        if bool(getattr(wp, "is_junction", False)):
            return True
        loc = wp.transform.location
        x, y = float(loc.x), float(loc.y)
        cx, cy = math.floor(x / junction_cell_m), math.floor(y / junction_cell_m)
        radius_sq = float(junction_radius_m) ** 2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for jx, jy in junction_grid.get((cx + dx, cy + dy), ()):
                    if (jx - x) ** 2 + (jy - y) ** 2 <= radius_sq:
                        return True
        return False

    lane_runs = {}
    for wp in waypoints:
        key = (wp.road_id, wp.section_id, wp.lane_id)
        lane_runs.setdefault(key, []).append(wp)

    selected = []
    selected_ids = set()
    stats = {"dense": 0, "curve": 0, "straight": 0, "endpoints": 0}

    for run in lane_runs.values():
        run.sort(key=lambda item: float(item.s))
        last_kept_s = None
        for index, wp in enumerate(run):
            endpoint = index == 0 or index == len(run) - 1
            near_junction = _near_junction(wp)
            lo = max(0, index - 2)
            hi = min(len(run) - 1, index + 2)
            yaw_lo = run[lo].transform.rotation.yaw
            yaw_hi = run[hi].transform.rotation.yaw
            is_curve = _angle_delta_deg(yaw_lo, yaw_hi) >= float(curve_yaw_threshold_deg)

            if near_junction:
                spacing = float(junction_spacing_m)
                category = "dense"
            elif is_curve:
                spacing = float(curve_spacing_m)
                category = "curve"
            else:
                spacing = float(straight_spacing_m)
                category = "straight"

            current_s = float(wp.s)
            keep = endpoint or last_kept_s is None or current_s - last_kept_s >= spacing - 1e-3
            if not keep:
                continue
            marker = id(wp)
            if marker in selected_ids:
                continue
            selected_ids.add(marker)
            selected.append(wp)
            last_kept_s = current_s
            stats["endpoints" if endpoint else category] += 1

    return selected, stats


def _value_bucket(value, cuts):
    if value is None:
        return -1
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1
    for index, cut in enumerate(cuts):
        if number <= cut:
            return index
    return len(cuts)


def _candidate_structure_signature(candidate):
    """Features that can materially change v2 gating or ranking."""
    arms = candidate.get("physical_junction_arms") or {}
    environment = candidate.get("environment_context") or {}
    nearest = environment.get("nearest_m") or {}
    yaw = float(candidate.get("yaw") or 0.0) % 360.0
    return (
        bool(candidate.get("is_junction")),
        candidate.get("candidate_topology_type"),
        candidate.get("same_direction_lane_count"),
        candidate.get("same_road_lane_count"),
        bool(candidate.get("has_center_median_candidate")),
        bool(candidate.get("has_highway_shoulder")),
        bool(candidate.get("left_parking_lane_present")),
        bool(candidate.get("right_parking_lane_present")),
        candidate.get("curve_direction"),
        _value_bucket(candidate.get("curve_abs_yaw_delta_deg"), (8.0, 30.0)),
        arms.get("left"),
        arms.get("right"),
        arms.get("arm_count"),
        _value_bucket(candidate.get("distance_to_junction_ahead"), (10, 25, 40, 80)),
        environment.get("environment_class"),
        _value_bucket(environment.get("urban_score"), (0.25, 0.5, 0.75)),
        _value_bucket(environment.get("natural_score"), (0.25, 0.5, 0.75)),
        bool(environment.get("buildings_nearby")),
        bool(environment.get("sidewalks_nearby")),
        environment.get("has_street_lights"),
        _value_bucket(nearest.get("TrafficLight"), (30, 75)),
        int((yaw + 15.0) // 30.0) % 12,
    )


def compress_structural_candidates(candidates, max_representatives=3, region_size_m=200.0):
    """Keep geographically diverse representatives of each match signature."""
    if max_representatives <= 0 or not candidates:
        return list(candidates), {"before": len(candidates), "after": len(candidates)}

    region_size_m = max(float(region_size_m), 1.0)
    grouped = defaultdict(dict)
    for candidate in candidates:
        location = candidate.get("location") or {}
        x = float(location.get("x") or 0.0)
        y = float(location.get("y") or 0.0)
        region = (math.floor(x / region_size_m), math.floor(y / region_size_m))
        grouped[_candidate_structure_signature(candidate)].setdefault(region, candidate)

    selected = []
    for region_candidates in grouped.values():
        pool = list(region_candidates.items())
        pool.sort(key=lambda item: item[0])
        chosen = [pool.pop(0)]
        while pool and len(chosen) < int(max_representatives):
            def separation(item):
                rx, ry = item[0]
                return min((rx - cx) ** 2 + (ry - cy) ** 2 for (cx, cy), _ in chosen)

            best = max(pool, key=lambda item: (separation(item), item[0]))
            pool.remove(best)
            chosen.append(best)
        selected.extend(candidate for _, candidate in chosen)

    return selected, {
        "before": len(candidates),
        "after": len(selected),
        "signature_count": len(grouped),
        "max_representatives_per_signature": int(max_representatives),
        "region_size_m": region_size_m,
    }


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
    adaptive_sampling: bool = False,
    straight_spacing: float = 50.0,
    curve_spacing: float = 10.0,
    junction_spacing: float = 5.0,
    junction_radius: float = 80.0,
    max_structure_representatives: int = 0,
    representative_region_size: float = 200.0,
) -> None:
    try:
        import carla
    except ImportError:
        sys.exit("ERROR: carla module not found. Activate the autoscenario conda env.")

    from tools.scene_map_matcher import (
        SceneMapMatcher,
        TOPOLOGY_CACHE_FEATURE_SET,
        TOPOLOGY_CACHE_SCHEMA_VERSION,
    )

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
    street_light_points = _load_street_light_points(world, carla)
    print(f"  {len(street_light_points)} street lights pre-computed")

    # Pre-compute crosswalk centre locations once (avoids per-candidate CARLA calls).
    try:
        crosswalk_locs = _flatten_crosswalk_locations(world_map.get_crosswalks())
        print(f"  {len(crosswalk_locs)} crosswalk vertices pre-computed")
    except Exception as exc:
        crosswalk_locs = []
        print(f"  WARNING: get_crosswalks() failed ({exc}); has_crosswalk_nearby will be False")

    adaptive_stats = None
    dense_waypoint_count = None
    if adaptive_sampling:
        dense_waypoints = world_map.generate_waypoints(sample_step)
        dense_waypoint_count = len(dense_waypoints)
        waypoints, adaptive_stats = _adaptive_waypoint_sample(
            dense_waypoints,
            straight_spacing_m=straight_spacing,
            curve_spacing_m=curve_spacing,
            junction_spacing_m=junction_spacing,
            junction_radius_m=junction_radius,
        )
        print(
            f"  adaptive sampling: {len(dense_waypoints)} dense waypoints -> "
            f"{len(waypoints)} candidates ({adaptive_stats})"
        )
        for i, wp in enumerate(waypoints):
            try:
                features = matcher._extract_local_candidate_features(
                    world_map, wp, crosswalk_locations=crosswalk_locs
                )
                lane_dict = matcher._waypoint_to_lane_dict(wp)
                environment_context = _summarize_environment_context(
                    wp.transform.location,
                    environment_points,
                    DEFAULT_ENVIRONMENT_RADIUS_M,
                    street_light_points,
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
                    "has_street_lights": environment_context["has_street_lights"],
                    **features,
                }
            )
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(waypoints)} adaptive waypoints")
    elif large_map:
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
                    street_light_points,
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
                    "has_street_lights": environment_context["has_street_lights"],
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
                    street_light_points,
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
                    "has_street_lights": environment_context["has_street_lights"],
                    **features,
                }
            )
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(waypoints)} waypoints")

    structural_compression = None
    if max_structure_representatives > 0:
        candidates, structural_compression = compress_structural_candidates(
            candidates,
            max_representatives=max_structure_representatives,
            region_size_m=representative_region_size,
        )
        print(
            "  structural compression: "
            f"{structural_compression['before']} -> {structural_compression['after']} candidates"
        )

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, _safe_map_filename(world_name))
    payload = {
        "schema_version": TOPOLOGY_CACHE_SCHEMA_VERSION,
        "feature_set": TOPOLOGY_CACHE_FEATURE_SET,
        "world_name": world_name,
        "sample_step": sample_step,
        "large_map": large_map,
        "adaptive_sampling": adaptive_sampling,
        "adaptive_sampling_config": {
            "straight_spacing_m": straight_spacing,
            "curve_spacing_m": curve_spacing,
            "junction_spacing_m": junction_spacing,
            "junction_radius_m": junction_radius,
            "dense_waypoint_count": dense_waypoint_count,
            "selection_stats": adaptive_stats,
        } if adaptive_sampling else None,
        "structural_compression": structural_compression,
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
        "--adaptive-sampling",
        action="store_true",
        help="Full-map adaptive sampling: dense near junctions, sparse on long straight roads",
    )
    parser.add_argument("--straight-spacing", type=float, default=50.0)
    parser.add_argument("--curve-spacing", type=float, default=10.0)
    parser.add_argument("--junction-spacing", type=float, default=5.0)
    parser.add_argument("--junction-radius", type=float, default=80.0)
    parser.add_argument(
        "--max-structure-representatives",
        type=int,
        default=0,
        help="Keep at most N geographically diverse candidates per matching signature",
    )
    parser.add_argument("--representative-region-size", type=float, default=200.0)
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
        adaptive_sampling=args.adaptive_sampling,
        straight_spacing=args.straight_spacing,
        curve_spacing=args.curve_spacing,
        junction_spacing=args.junction_spacing,
        junction_radius=args.junction_radius,
        max_structure_representatives=args.max_structure_representatives,
        representative_region_size=args.representative_region_size,
    )
