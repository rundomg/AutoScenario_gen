#!/usr/bin/env python3
"""Apply the requested deterministic actor edit to the Town13 reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from copy import deepcopy
from pathlib import Path

from tools.user_text_constraints import evaluate_user_constraints


WHITE = (255, 255, 255)
BLACK = (20, 20, 20)


def _write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _color(entity: dict, name: str, rgb: tuple[int, int, int]) -> None:
    rgb_text = ",".join(str(value) for value in rgb)
    entity["color"] = rgb_text
    entity["appearance"] = {"color": name, "color_rgb": rgb_text}


def _unique_lanes(leg: dict, motion: str) -> list[dict]:
    key = "inbound_lanes" if motion == "approaching" else "outbound_lanes"
    result = []
    for lane in leg.get(key) or []:
        anchor = lane.get("anchor") or {}
        signature = (round(float(anchor.get("x", 0.0)), 2), round(float(anchor.get("y", 0.0)), 2))
        if any(item[0] == signature for item in result):
            continue
        result.append((signature, lane))
    return [lane for _, lane in result]


def _lane_pose(leg: dict, lane: dict, distance_m: float, motion: str) -> tuple[dict, dict]:
    anchor = lane.get("anchor") or {}
    anchor_distance = float(anchor.get("distance_from_center_m") or distance_m)
    heading_out = float(leg.get("heading_out_deg") or 0.0)
    radians = math.radians(heading_out)
    delta = float(distance_m) - anchor_distance
    location = {
        "x": float(anchor.get("x", 0.0)) + math.cos(radians) * delta,
        "y": float(anchor.get("y", 0.0)) + math.sin(radians) * delta,
        "z": float(anchor.get("z", 0.3)) + 0.3,
    }
    yaw = float(lane.get("yaw", heading_out))
    if motion == "approaching" and abs((yaw - heading_out + 180.0) % 360.0 - 180.0) < 90.0:
        yaw += 180.0
    if motion == "leaving" and abs((yaw - heading_out + 180.0) % 360.0 - 180.0) > 90.0:
        yaw += 180.0
    rotation = {"pitch": 0.0, "yaw": (yaw + 180.0) % 360.0 - 180.0, "roll": 0.0}
    return location, rotation


def _projected_actor(
    template: dict,
    *,
    entity_id: str,
    leg: dict,
    lane: dict,
    lane_slot: int,
    distance_m: float,
    motion: str,
    color_name: str,
    color_rgb: tuple[int, int, int],
    layout_anchor_id: str,
    heading_relation: str,
) -> dict:
    entity = deepcopy(template)
    location, rotation = _lane_pose(leg, lane, distance_m, motion)
    for key in (
        "visual_position_override",
        "parking_projection_lane",
        "parking_projection_result",
        "degraded_reason",
    ):
        entity.pop(key, None)
    entity.update(
        {
            "id": entity_id,
            "actor_group_id": entity_id,
            "category": "car",
            "blueprint_name": "car",
            "spawn_kind": "vehicle",
            "motion_state": "stopped",
            "location": location,
            "rotation": rotation,
            "heading_relation": heading_relation,
            "original_heading_relation": heading_relation,
            "layout_anchor_id": layout_anchor_id,
            "junction_direction": str(leg.get("name") or ""),
            "junction_leg": str(leg.get("name") or ""),
            "junction_motion": motion,
            "junction_distance_m": float(distance_m),
            "junction_lane_from_right": int(lane_slot),
            "lane_from_right": int(lane_slot),
            "placement_mode": "project_to_junction_lane",
            "placement_mode_hint": "normal_lane_actor",
            "projected_lane": {
                **deepcopy(lane),
                "source": "junction_leg",
                "preserve_input_yaw": True,
            },
            "anchor_relation": {
                "lane_from_right": int(lane_slot),
                "position_along_anchor": "approach",
                "travel_direction": "toward_junction" if motion == "approaching" else "away_from_junction",
            },
        }
    )
    _color(entity, color_name, color_rgb)
    return entity


def _subject_from_actor(actor: dict) -> dict:
    return {
        "id": actor["id"],
        "category": actor.get("category", "car"),
        "subtype": actor.get("category", "car"),
        "appearance": deepcopy(actor.get("appearance") or {}),
        "motion_state": actor.get("motion_state", "stopped"),
        "heading_relation_to_ego": actor.get("heading_relation", "unknown"),
        "lane_side_relation": actor.get("lane_side_relation", "unknown"),
        "lane_index_relation": actor.get("lane_index_relation", 0),
        "layout_anchor_id": actor.get("layout_anchor_id", ""),
        "anchor_relation": deepcopy(actor.get("anchor_relation") or {}),
    }


def customize(
    output_dir: Path, scene_id: str, seed: int, *, reset_from_backup: bool = False
) -> dict:
    actors_path = output_dir / f"{scene_id}_actors.json"
    structure_path = output_dir / f"{scene_id}_matched_structure.json"
    scene_path = output_dir / "s0000_su.json"
    for path in (actors_path, structure_path, scene_path):
        if not path.exists():
            raise FileNotFoundError(path)
        backup = path.with_suffix(path.suffix + ".pre_custom_edit")
        if not backup.exists():
            shutil.copy2(path, backup)

    actors_source = (
        actors_path.with_suffix(actors_path.suffix + ".pre_custom_edit")
        if reset_from_backup
        else actors_path
    )
    scene_source = (
        scene_path.with_suffix(scene_path.suffix + ".pre_custom_edit")
        if reset_from_backup
        else scene_path
    )
    payload = json.loads(actors_source.read_text(encoding="utf-8"))
    structure = json.loads(structure_path.read_text(encoding="utf-8"))["matched_structure"]
    scene = json.loads(scene_source.read_text(encoding="utf-8"))
    entities = payload.get("entities") or []
    by_id = {str(entity.get("id")): entity for entity in entities}
    legs = {str(leg.get("name")): leg for leg in structure.get("legs") or []}
    center = structure.get("center") or {}

    # Shift ego and the two stopped ego-side vehicles 10 m away from the junction.
    ego_out = math.radians(float(legs["ego"].get("heading_out_deg") or 0.0))
    dx, dy = 10.0 * math.cos(ego_out), 10.0 * math.sin(ego_out)
    for entity_id in ("ego", "det_3", "det_8"):
        entity = by_id[entity_id]
        entity["location"]["x"] = float(entity["location"]["x"]) + dx
        entity["location"]["y"] = float(entity["location"]["y"]) + dy
        if isinstance(entity.get("junction_distance_m"), (int, float)):
            entity["junction_distance_m"] = float(entity["junction_distance_m"]) + 10.0
        entity.pop("visual_position_override", None)
        if entity_id != "ego" and entity.get("projected_lane"):
            entity["placement_mode"] = "project_to_junction_lane"
            entity["placement_mode_hint"] = "normal_lane_actor"
    _color(by_id["det_8"], "white", WHITE)

    # Place the bicycle on the right-arm crosswalk and keep its XY unsnapped.
    coverage = structure.get("crosswalk_arm_coverage") or {}
    right_crosswalk = next(
        item for item in coverage.get("assignments") or [] if item.get("arm") == "right"
    )
    bicycle = by_id["det_4"]
    bicycle["location"].update(right_crosswalk["crosswalk_center"])
    bicycle["rotation"] = {
        "pitch": 0.0,
        "yaw": (float(legs["right"].get("heading_out_deg") or 0.0) + 90.0) % 360.0,
        "roll": 0.0,
    }
    bicycle["placement_mode"] = "direct"
    bicycle["placement_mode_hint"] = "crosswalk_actor"
    bicycle["layout_anchor_id"] = "right_arm"
    bicycle["anchor_relation"] = {"position_along_anchor": "crosswalk"}
    bicycle["junction_lane_from_right"] = None
    bicycle["lane_from_right"] = None
    bicycle.pop("visual_position_override", None)

    car_template = by_id["det_3"]
    added = []

    # Two white vehicles, front/back, on the right-arm inbound lane.
    right_inbound = _unique_lanes(legs["right"], "approaching")[0]
    for index, distance in enumerate((18.0, 30.0), start=1):
        actor = _projected_actor(
            car_template,
            entity_id=f"right_arm_white_{index}",
            leg=legs["right"],
            lane=right_inbound,
            lane_slot=0,
            distance_m=distance,
            motion="approaching",
            color_name="white",
            color_rgb=WHITE,
            layout_anchor_id="right_arm",
            heading_relation="crossing",
        )
        actor["lane_side_relation"] = "right_lane"
        actor["lane_index_relation"] = 1
        added.append(actor)

    # Two vehicles on the opposing carriageway immediately left of ego.
    ego_outbound = _unique_lanes(legs["ego"], "leaving")
    for index, (lane, distance, color_name, rgb) in enumerate(
        zip(ego_outbound[:2], (18.0, 30.0), ("white", "black"), (WHITE, BLACK)),
        start=1,
    ):
        actor = _projected_actor(
            car_template,
            entity_id=f"ego_left_oncoming_{index}",
            leg=legs["ego"],
            lane=lane,
            lane_slot=index - 1,
            distance_m=distance,
            motion="leaving",
            color_name=color_name,
            color_rgb=rgb,
            layout_anchor_id="ego_approach",
            heading_relation="opposite_direction",
        )
        actor["lane_side_relation"] = "opposing_lane"
        actor["lane_index_relation"] = -index
        added.append(actor)

    # Four deterministic-random vehicles on the far side, split across both flows.
    rng = random.Random(seed)
    opposite_in = _unique_lanes(legs["opposite"], "approaching")
    opposite_out = _unique_lanes(legs["opposite"], "leaving")
    specifications = [
        ("approaching", opposite_in, "black", BLACK, 24.0),
        ("leaving", opposite_out, "white", WHITE, 25.0),
        ("approaching", opposite_in, "white", WHITE, 39.0),
        ("leaving", opposite_out, "black", BLACK, 40.0),
    ]
    for index, (motion, lanes, color_name, rgb, base_distance) in enumerate(
        specifications, start=1
    ):
        lane_slot = rng.randrange(len(lanes))
        distance = base_distance + rng.uniform(-2.0, 2.0)
        actor = _projected_actor(
            car_template,
            entity_id=f"opposite_random_{index}",
            leg=legs["opposite"],
            lane=lanes[lane_slot],
            lane_slot=lane_slot,
            distance_m=distance,
            motion=motion,
            color_name=color_name,
            color_rgb=rgb,
            layout_anchor_id="oncoming_arm",
            heading_relation=("opposite_direction" if motion == "approaching" else "same_direction"),
        )
        actor["lane_side_relation"] = "opposing_lane"
        actor["lane_index_relation"] = -(lane_slot + 1)
        added.append(actor)

    # Replace the original two far-side vehicles with the randomized four.
    entities[:] = [entity for entity in entities if entity.get("id") not in {"det_1", "det_2"}]
    entities.extend(added)
    payload.setdefault("metadata", {})["manual_scene_edit"] = {
        "schema_version": "town13-intersection-edit-v1",
        "random_seed": seed,
        "ego_side_shift_m": -10.0,
        "bicycle_crosswalk_arm": "right",
        "added_actor_ids": [actor["id"] for actor in added],
        "removed_actor_ids": ["det_1", "det_2"],
    }

    # Mirror the edit into scene understanding and its immutable user constraint channel.
    subjects = [
        subject
        for subject in scene.get("traffic_subjects") or []
        if subject.get("id") not in {"det_1", "det_2"}
    ]
    existing_subjects = {str(subject.get("id")): subject for subject in subjects}
    existing_subjects["det_8"].setdefault("appearance", {}).update(
        {"color": "white", "color_rgb": "255,255,255"}
    )
    existing_subjects["det_4"]["placement_mode_hint"] = "crosswalk_actor"
    existing_subjects["det_4"]["anchor_relation"] = {"position_along_anchor": "crosswalk"}
    subjects.extend(_subject_from_actor(actor) for actor in added)
    scene["traffic_subjects"] = subjects
    bundle = scene.setdefault("metadata", {}).setdefault("user_constraints", {})
    constraints = [
        item
        for item in bundle.get("constraints") or []
        if item.get("entity_id") not in {"det_1", "det_2"}
        and item.get("id") not in {"user_rel_0", "user_rel_1"}
    ]
    edits = [
        ("det_8", "appearance.color", "white", "自车左侧车辆改成白色"),
        ("det_4", "placement_mode_hint", "crosswalk_actor", "自行车放到右臂人行横道上"),
        ("det_4", "anchor_relation.position_along_anchor", "crosswalk", "自行车位于人行横道"),
    ]
    for actor in added:
        edits.extend(
            [
                (actor["id"], "appearance.color", actor["appearance"]["color"], "用户指定车辆颜色"),
                (actor["id"], "layout_anchor_id", actor["layout_anchor_id"], "用户指定路口区域"),
                (actor["id"], "heading_relation_to_ego", actor["heading_relation"], "用户指定正向或反向"),
                (actor["id"], "anchor_relation.lane_from_right", actor["lane_from_right"], "用户指定车道"),
            ]
        )
    for index, (entity_id, path, value, evidence) in enumerate(edits):
        constraints.append(
            {
                "id": f"user_edit_{index}",
                "target": "entity",
                "entity_id": entity_id,
                "path": path,
                "value": value,
                "source": "user_text",
                "strength": "hard",
                "evidence": evidence,
                "tolerance": 1.5,
            }
        )
    bundle["constraints"] = constraints
    scene["metadata"]["manual_scene_edit"] = deepcopy(payload["metadata"]["manual_scene_edit"])

    manifest = {
        "schema_version": "town13-intersection-edit-v1",
        "scene_id": scene_id,
        "random_seed": seed,
        "ego_location": by_id["ego"]["location"],
        "bicycle_location": bicycle["location"],
        "actor_count": len(entities),
        "actors": [
            {
                "id": entity.get("id"),
                "color": (entity.get("appearance") or {}).get("color"),
                "location": entity.get("location"),
                "yaw": (entity.get("rotation") or {}).get("yaw"),
                "junction_leg": entity.get("junction_leg"),
                "junction_motion": entity.get("junction_motion"),
                "junction_lane_from_right": entity.get("junction_lane_from_right"),
            }
            for entity in entities
        ],
    }
    match_path = output_dir / f"{scene_id}_match.json"
    match_report = json.loads(match_path.read_text(encoding="utf-8"))
    constraint_report = evaluate_user_constraints(
        scene,
        payload,
        match_report,
        {"topology_sample": [{"resolved_ego_lane_from_right": 0}]},
        {},
    )
    manifest["constraint_validation"] = {
        "status": constraint_report.get("status"),
        "hard_constraint_count": constraint_report.get("hard_constraint_count"),
        "failure_count": constraint_report.get("failure_count"),
    }
    _write(actors_path, payload)
    _write(scene_path, scene)
    _write(output_dir / f"{scene_id}_manual_edit.json", manifest)
    constraint_path = output_dir / f"{scene_id}_user_constraints.json"
    constraint_backup = constraint_path.with_suffix(
        constraint_path.suffix + ".pre_custom_edit"
    )
    if constraint_path.exists() and not constraint_backup.exists():
        shutil.copy2(constraint_path, constraint_backup)
    _write(constraint_path, constraint_report)
    summary_path = output_dir / "run_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    summary.update(
        {
            "status": "customized_pending_carla_validation",
            "scene_id": scene_id,
            "static_script": str((output_dir / f"{scene_id}_static.py").resolve()),
            "traffic_subject_count": len(entities) - 1,
            "user_constraints": constraint_report,
            "manual_scene_edit": manifest,
        }
    )
    _write(summary_path, summary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scene-id", default="s0000_c0")
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument("--reset-from-backup", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            customize(
                args.output_dir,
                args.scene_id,
                args.seed,
                reset_from_backup=args.reset_from_backup,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
