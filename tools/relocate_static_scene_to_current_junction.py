#!/usr/bin/env python3
"""Relocate a generated static scene onto the junction under CARLA spectator."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from copy import deepcopy
from pathlib import Path

from tools.customize_town13_intersection_scene import _lane_pose, _unique_lanes
from tools.map_structure import build_matched_structure_from_waypoint


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _driving_lanes_across(waypoint, carla) -> list:
    found = []
    queue = [waypoint]
    seen = set()
    while queue:
        current = queue.pop(0)
        key = (int(current.road_id), int(current.lane_id))
        if key in seen:
            continue
        seen.add(key)
        if current.lane_type == carla.LaneType.Driving:
            found.append(current)
        for neighbor in (current.get_left_lane(), current.get_right_lane()):
            if neighbor is not None and neighbor.lane_type == carla.LaneType.Driving:
                queue.append(neighbor)
    return found


def _structure_under_spectator(world, carla) -> tuple[dict, dict]:
    world_map = world.get_map()
    spectator = world.get_spectator().get_transform()
    waypoint = world_map.get_waypoint(
        spectator.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    candidates = []
    for lane in _driving_lanes_across(waypoint, carla):
        structure = build_matched_structure_from_waypoint(
            lane,
            ego_inbound_yaw=float(lane.transform.rotation.yaw),
            junction_lookahead_m=80.0,
        )
        if structure.get("kind") != "junction":
            continue
        candidates.append((len(structure.get("legs") or []), structure, lane))
    if not candidates:
        raise RuntimeError("No forward junction was found under/near the spectator.")
    _, structure, lane = max(candidates, key=lambda item: item[0])
    return structure, {
        "spectator": {
            "x": spectator.location.x,
            "y": spectator.location.y,
            "z": spectator.location.z,
        },
        "anchor_lane": {
            "road_id": lane.road_id,
            "lane_id": lane.lane_id,
            "yaw": lane.transform.rotation.yaw,
        },
        "map": world_map.name,
    }


def _crosswalk_groups(world_map) -> list[dict]:
    groups = []
    current = []
    for point in world_map.get_crosswalks() or []:
        current.append(point)
        if len(current) >= 4 and point.distance(current[0]) <= 0.25:
            groups.append(
                {
                    "x": sum(item.x for item in current) / len(current),
                    "y": sum(item.y for item in current) / len(current),
                    "z": sum(item.z for item in current) / len(current),
                }
            )
            current = []
    return groups


def _right_arm_crosswalk(structure: dict, groups: list[dict]) -> dict:
    center = structure.get("center") or {}
    leg = next(leg for leg in structure.get("legs") or [] if leg.get("name") == "right")
    heading = float(leg.get("heading_out_deg") or 0.0)

    def score(group: dict) -> tuple[float, float]:
        dx = float(group["x"]) - float(center["x"])
        dy = float(group["y"]) - float(center["y"])
        distance = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx))
        angle = abs((bearing - heading + 180.0) % 360.0 - 180.0)
        return angle, distance

    nearby = [group for group in groups if score(group)[1] <= 35.0 and score(group)[0] <= 50.0]
    if not nearby:
        raise RuntimeError("Target junction has no right-arm crosswalk.")
    return min(nearby, key=score)


def relocate(source_script: Path, output_dir: Path, host: str, port: int) -> dict:
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("CARLA Python API is unavailable") from exc

    source_dir = source_script.parent
    scene_id = source_script.stem
    actors_name = f"{scene_id[:-7]}_actors.json" if scene_id.endswith("_static") else ""
    if not actors_name:
        raise ValueError("Expected a script named <scene_id>_static.py")
    actors_source = source_dir / actors_name
    structure_source = source_dir / actors_name.replace("_actors.json", "_matched_structure.json")
    if not actors_source.exists() or not structure_source.exists():
        raise FileNotFoundError("Source actor or matched-structure JSON is missing")

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    target_structure, target_context = _structure_under_spectator(world, carla)
    target_legs = {str(leg.get("name")): leg for leg in target_structure.get("legs") or []}
    required_legs = {"ego", "right", "opposite"}
    missing = sorted(required_legs - set(target_legs))
    if missing:
        raise RuntimeError("Target junction lacks required arms: " + ", ".join(missing))

    source_structure_wrapper = json.loads(structure_source.read_text(encoding="utf-8"))
    source_structure = source_structure_wrapper["matched_structure"]
    source_legs = {
        str(leg.get("name")): leg for leg in source_structure.get("legs") or []
    }
    source_center = source_structure.get("center") or {}
    target_center = target_structure.get("center") or {}
    payload = json.loads(actors_source.read_text(encoding="utf-8"))

    needs_right_crosswalk = any(
        str(entity.get("placement_mode_hint") or "") == "crosswalk_actor"
        for entity in payload.get("entities") or []
    )
    right_crosswalk = (
        _right_arm_crosswalk(target_structure, _crosswalk_groups(world.get_map()))
        if needs_right_crosswalk
        else None
    )
    placements = []
    for entity in payload.get("entities") or []:
        entity_id = str(entity.get("id") or "")
        old_location = deepcopy(entity.get("location") or {})
        if str(entity.get("placement_mode_hint") or "") == "crosswalk_actor":
            assert right_crosswalk is not None
            entity["location"] = {
                "x": float(right_crosswalk["x"]),
                "y": float(right_crosswalk["y"]),
                "z": float(right_crosswalk["z"]) + 0.3,
            }
            right_heading = float(target_legs["right"].get("heading_out_deg") or 0.0)
            entity["rotation"] = {
                "pitch": 0.0,
                "yaw": (right_heading + 90.0 + 180.0) % 360.0 - 180.0,
                "roll": 0.0,
            }
            entity["placement_mode"] = "direct"
            entity["placement_mode_hint"] = "crosswalk_actor"
            entity["projected_lane"] = {}
            placements.append({"id": entity_id, "mode": "right_crosswalk"})
            continue

        leg_name = str(entity.get("junction_leg") or "ego")
        if leg_name not in target_legs:
            leg_name = "ego"
        original_motion = str(entity.get("junction_motion") or "approaching")
        motion = original_motion
        if motion not in {"approaching", "leaving", "parked"}:
            motion = "approaching"
        lane_motion = "leaving" if motion == "parked" else motion
        lanes = _unique_lanes(target_legs[leg_name], lane_motion)
        if not lanes:
            raise RuntimeError(f"No {lane_motion} lanes on target arm {leg_name}")
        try:
            lane_slot = int(entity.get("junction_lane_from_right") or 0)
        except (TypeError, ValueError):
            lane_slot = 0
        lane_slot = min(max(0, lane_slot), len(lanes) - 1)
        distance = entity.get("junction_distance_m")
        if not isinstance(distance, (int, float)):
            distance = math.hypot(
                float(old_location.get("x", 0.0)) - float(source_center.get("x", 0.0)),
                float(old_location.get("y", 0.0)) - float(source_center.get("y", 0.0)),
            )
        location, rotation = _lane_pose(
            target_legs[leg_name], lanes[lane_slot], float(distance), lane_motion
        )

        # Preserve a directly placed parked actor's roadside offset relative to
        # its source arm. This keeps far-side curb parking out of a driving lane.
        if motion == "parked" and leg_name in source_legs:
            source_lane = deepcopy(entity.get("projected_lane") or {})
            if source_lane.get("anchor"):
                source_base, _ = _lane_pose(
                    source_legs[leg_name],
                    source_lane,
                    float(distance),
                    "leaving",
                )
                offset_x = float(old_location.get("x", 0.0)) - float(source_base["x"])
                offset_y = float(old_location.get("y", 0.0)) - float(source_base["y"])
                source_heading = math.radians(
                    float(source_legs[leg_name].get("heading_out_deg") or 0.0)
                )
                forward = offset_x * math.cos(source_heading) + offset_y * math.sin(source_heading)
                rightward = offset_x * -math.sin(source_heading) + offset_y * math.cos(source_heading)
                target_heading = math.radians(
                    float(target_legs[leg_name].get("heading_out_deg") or 0.0)
                )
                location["x"] += forward * math.cos(target_heading) - rightward * math.sin(target_heading)
                location["y"] += forward * math.sin(target_heading) + rightward * math.cos(target_heading)
                source_actor_yaw = float((entity.get("rotation") or {}).get("yaw") or 0.0)
                source_heading_deg = float(
                    source_legs[leg_name].get("heading_out_deg") or 0.0
                )
                target_heading_deg = float(
                    target_legs[leg_name].get("heading_out_deg") or 0.0
                )
                rotation["yaw"] = (
                    source_actor_yaw + target_heading_deg - source_heading_deg + 180.0
                ) % 360.0 - 180.0
        entity["location"] = location
        entity["rotation"] = rotation
        entity["junction_leg"] = leg_name
        entity["junction_direction"] = leg_name
        entity["junction_motion"] = motion
        entity["junction_distance_m"] = float(distance)
        entity["junction_lane_from_right"] = lane_slot
        entity["lane_from_right"] = lane_slot
        original_placement_mode = str(entity.get("placement_mode") or "")
        if motion == "parked":
            entity["placement_mode"] = "direct"
        elif original_placement_mode == "project_to_parking_lane":
            entity["placement_mode"] = "project_to_parking_lane"
        else:
            entity["placement_mode"] = "project_to_junction_lane"
        entity["projected_lane"] = {
            **deepcopy(lanes[lane_slot]),
            "source": "junction_leg",
            "preserve_input_yaw": True,
        }
        entity.pop("visual_position_override", None)
        placements.append(
            {
                "id": entity_id,
                "mode": "target_junction_lane",
                "leg": leg_name,
                "motion": motion,
                "lane_from_right": lane_slot,
                "distance_m": float(distance),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    script_target = output_dir / source_script.name
    script_text = source_script.read_text(encoding="utf-8")
    map_leaf = str(target_context["map"]).rstrip("/").split("/")[-1]
    script_text = re.sub(
        r"^_AUTOSCENARIO_CARLA_MAP\s*=.*$",
        f"_AUTOSCENARIO_CARLA_MAP = {map_leaf!r}",
        script_text,
        flags=re.MULTILINE,
    )
    script_text = re.sub(
        r"^_autoscenario_target_map\s*=.*$",
        f"_autoscenario_target_map = {map_leaf!r}",
        script_text,
        flags=re.MULTILINE,
    )
    script_text = script_text.replace(
        "_autoscenario_init(world, blueprint_library, client)",
        "_autoscenario_init(world, blueprint_library)",
    )
    compatibility_anchor = "_autoscenario_helpers_module._autoscenario_init(world, blueprint_library)"
    compatibility_shims = """

# Compatibility shims for static scripts generated by the older verification runtime.
def _autoscenario_destroy_actors_safely(actors):
    for actor in actors:
        try:
            actor.destroy()
        except Exception:
            pass

def _autoscenario_prepare_static_verification(*_args, **_kwargs):
    pass

def _autoscenario_activate_static_verification_streaming(*_args, **_kwargs):
    pass

def _autoscenario_static_verify_enabled():
    return False

def _autoscenario_wait_for_world_ticks(count):
    for _ in range(max(0, int(count))):
        try:
            world.wait_for_tick()
        except Exception:
            time.sleep(0.05)

def _autoscenario_is_large_map():
    return False

def _autoscenario_cleanup_spawned_actors():
    # Keep the relocated actors alive after this one-shot static script exits.
    pass
"""
    if compatibility_anchor in script_text:
        script_text = script_text.replace(
            compatibility_anchor,
            compatibility_anchor + compatibility_shims,
            1,
        )
    freeze_anchor = (
        "    _autoscenario_configure_vehicle_lights_for_environment("
        "_autoscenario_actor_by_id, spawn_payload, _autoscenario_weather)"
    )
    freeze_block = """    # Keep the relocated static layout fixed on sloped junction approaches.
    for _autoscenario_static_actor in _autoscenario_actor_by_id.values():
        try:
            if str(_autoscenario_static_actor.type_id).startswith('vehicle.'):
                _autoscenario_static_actor.set_autopilot(False)
                _autoscenario_static_actor.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0,
                        steer=0.0,
                        hand_brake=True,
                    )
                )
        except Exception:
            pass
"""
    if freeze_anchor in script_text:
        script_text = script_text.replace(
            freeze_anchor,
            freeze_block + freeze_anchor,
            1,
        )
    script_target.write_text(script_text, encoding="utf-8")
    _write(output_dir / actors_name, payload)
    _write(
        output_dir / structure_source.name,
        {
            "source": "live_carla_current_spectator",
            "summary": {
                "map": target_context["map"],
                "kind": target_structure.get("kind"),
                "junction_id": target_structure.get("junction_id"),
                "leg_count": target_structure.get("leg_count"),
                "leg_names": [leg.get("name") for leg in target_structure.get("legs") or []],
            },
            "matched_structure": target_structure,
        },
    )
    for optional_name in ("s0000_su.json", f"{scene_id[:-7]}_manual_edit.json"):
        optional_source = source_dir / optional_name
        if optional_source.exists():
            shutil.copy2(optional_source, output_dir / optional_name)

    manifest = {
        "schema_version": "static-scene-relocation-v1",
        "source_script": str(source_script.resolve()),
        "target_script": str(script_target.resolve()),
        "source_map": str(
            (payload.get("metadata") or {}).get("carla_map") or "Town12"
        ),
        "target_map": target_context["map"],
        "target_spectator": target_context["spectator"],
        "target_junction": {
            "junction_id": target_structure.get("junction_id"),
            "kind": target_structure.get("kind"),
            "leg_count": target_structure.get("leg_count"),
            "center": target_center,
            "ego_approach_heading_deg": target_structure.get("ego_approach_heading_deg"),
        },
        "right_crosswalk": right_crosswalk,
        "actor_count": len(payload.get("entities") or []),
        "placements": placements,
    }
    _write(output_dir / f"{scene_id[:-7]}_relocation.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=2000, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            relocate(args.source_script.resolve(), args.output_dir.resolve(), args.host, args.port),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
