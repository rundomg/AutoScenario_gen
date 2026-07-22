#!/usr/bin/env python3
"""Convert an AutoScenario dynamic reconstruction to ScenarioRunner XML.

The generated file keeps the stock ScenarioRunner actor/weather schema and adds
an ``autoscenario`` element containing the dynamic DSL timeline.  Stock
ScenarioRunner ignores that extension; ``lmdrive_dynamic_scenario.py`` consumes
it when the scenario is executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


EGO_IDS = {"ego", "ego_vehicle"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _single_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one {!r} file in {}, found {}".format(
                pattern, folder, len(matches)
            )
        )
    return matches[0]


def _format_number(value) -> str:
    return "{:.6f}".format(float(value)).rstrip("0").rstrip(".")


def _indent(element: ET.Element, level: int = 0) -> None:
    """Apply stable two-space indentation on Python 3.8 and newer."""
    indentation = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def _vehicle_model(entity: dict, is_ego: bool) -> str:
    model = str(entity.get("blueprint_name") or "").strip()
    if model.startswith("vehicle."):
        return model
    if is_ego:
        return "vehicle.lincoln.mkz_2020"
    return "vehicle.tesla.model3"


def _weather_attributes(weather_name: str) -> dict[str, str]:
    # CARLA 0.9.15 presets after the same night visibility adjustment used by
    # autoscenario_scene_helpers._autoscenario_apply_weather().
    if weather_name == "ClearNoon":
        return {
            "cloudiness": "5",
            "precipitation": "0",
            "precipitation_deposits": "0",
            "wind_intensity": "10",
            "sun_azimuth_angle": "-1",
            "sun_altitude_angle": "45",
            "fog_density": "2",
            "fog_distance": "0.75",
            "wetness": "0",
        }
    if weather_name == "ClearNight":
        return {
            "cloudiness": "5",
            "precipitation": "0",
            "precipitation_deposits": "0",
            "wind_intensity": "10",
            "sun_azimuth_angle": "-1",
            "sun_altitude_angle": "-8",
            "fog_density": "15",
            "fog_distance": "75",
            "wetness": "0",
        }
    if weather_name == "WetNight":
        # Matches autoscenario_scene_helpers._autoscenario_apply_weather(),
        # including its visibility-preserving night adjustments.
        return {
            "cloudiness": "5",
            "precipitation": "0",
            "precipitation_deposits": "50",
            "wind_intensity": "10",
            "sun_azimuth_angle": "-1",
            "sun_altitude_angle": "-8",
            "fog_density": "15",
            "fog_distance": "75",
            "wetness": "60",
        }
    if weather_name == "MidRainyNight":
        return {
            "cloudiness": "80",
            "precipitation": "60",
            "precipitation_deposits": "60",
            "wind_intensity": "60",
            "sun_azimuth_angle": "-1",
            "sun_altitude_angle": "-8",
            "fog_density": "15",
            "fog_distance": "20",
            "wetness": "80",
        }
    return {
        "cloudiness": "0",
        "precipitation": "0",
        "precipitation_deposits": "0",
        "wind_intensity": "0",
        "sun_azimuth_angle": "0",
        "sun_altitude_angle": "75",
        "fog_density": "0",
        "fog_distance": "0",
        "wetness": "0",
    }


def _scenario_runner_map_name(map_name: str) -> str:
    """Return the short map name accepted by CARLA ``load_world``."""
    map_name = str(map_name).strip()
    map_name = map_name.rstrip("/").rsplit("/", 1)[-1]
    # Both local 0.9.15 and 0.9.16 packages crash while loading the monolithic
    # Town03 asset. Town03_Opt has identical road coordinates/topology and is
    # the supported streamed variant used for these evaluations.
    if map_name == "Town03":
        return "Town03_Opt"
    return map_name


def _generated_map_name(dynamic_dir: Path):
    scripts = sorted(dynamic_dir.glob("*_dynamic_reconstruction.py"))
    if not scripts:
        return None
    match = re.search(
        r"_autoscenario_target_map\s*=\s*['\"]([^'\"]+)",
        scripts[0].read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _actor_payload_path(dynamic_dir: Path, scene_id: str) -> Path:
    candidates = (
        dynamic_dir.parent / "{}_actors.json".format(scene_id),
        dynamic_dir.parent / "static" / "{}_actors.json".format(scene_id),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Actor spawn payload not found; checked: {}".format(
            ", ".join(str(item) for item in candidates)
        )
    )


def _lmdrive_notice_name(risk: dict):
    """Map generated risk semantics to LMDrive's six native notice classes."""
    description = " ".join(
        str(value or "").lower()
        for value in (
            risk.get("accident_type"),
            (risk.get("metadata") or {}).get("description"),
        )
    )
    actors = risk.get("actors") or []
    actor_text = " ".join(
        str(actor.get("role") or actor.get("type") or "").lower() for actor in actors
    )
    action_types = {
        str((event.get("action") or {}).get("type") or "").lower()
        for event in risk.get("events") or []
    }
    combined = "{} {}".format(description, actor_text)
    if any(word in combined for word in ("pedestrian", "walker")):
        return "Scenario3"
    if any(word in combined for word in ("bicycle", "cyclist", "bike")):
        return "Scenario4"
    if "red light" in combined:
        return "Scenario8" if "left" in combined else "Scenario7"
    if "rough road" in combined or "bumpy" in combined:
        return "Scenario1"
    if action_types.intersection(
        {"stop", "brake", "emergency_brake", "decelerate_to_speed"}
    ) or any(word in combined for word in ("slow", "stopped", "lead vehicle")):
        return "Scenario2"
    return None


def convert(dynamic_dir: Path, output_path: Path) -> Path:
    dynamic_dir = dynamic_dir.resolve()
    risk_paths = sorted(dynamic_dir.glob("*_video_risk_dsl.json"))
    summary_paths = sorted(dynamic_dir.glob("*_video_reconstruction_summary.json"))
    reconstruction_paths = sorted(dynamic_dir.glob("*_video_reconstruction.json"))
    if not risk_paths and not reconstruction_paths:
        raise FileNotFoundError("No risk DSL or numeric reconstruction in {}".format(dynamic_dir))
    risk_path = risk_paths[0] if risk_paths else None
    summary_path = summary_paths[0] if summary_paths else None
    reconstruction_path = reconstruction_paths[0] if reconstruction_paths else None
    risk = _load_json(risk_path) if risk_path else {}
    summary = _load_json(summary_path) if summary_path else {}
    reconstruction = _load_json(reconstruction_path) if reconstruction_path else {}

    source_path = risk_path or reconstruction_path
    scene_id = str(
        risk.get("scene_id")
        or reconstruction.get("scene_id")
        or source_path.name.split("_video_")[0]
    )
    spawn_path = _actor_payload_path(dynamic_dir, scene_id)
    spawn_payload = _load_json(spawn_path)
    entities = list(spawn_payload.get("entities") or [])

    ego = next(
        (entity for entity in entities if str(entity.get("id")) in EGO_IDS),
        None,
    )
    if ego is None:
        raise ValueError("Spawn payload has no ego entity")
    other_entities = [
        entity
        for entity in entities
        if str(entity.get("id")) not in EGO_IDS
        and str(entity.get("spawn_kind") or "vehicle") == "vehicle"
    ]
    selected_actor_ids = {str(entity.get("id")) for entity in other_entities}

    map_name = _scenario_runner_map_name(
        str(_generated_map_name(dynamic_dir) or summary.get("map_name") or "Town01_Opt")
    )
    weather_name = str(
        (spawn_payload.get("metadata") or {}).get("carla_weather_preset")
        or "ClearNoon"
    )
    duration_s = float(risk.get("duration_s") or reconstruction.get("duration_s") or 12.0)
    ego_motion = next(
        (
            actor
            for actor in reconstruction.get("actors") or []
            if str(actor.get("actor_id")) in EGO_IDS
        ),
        {},
    )
    ego_keyframes = ego_motion.get("keyframes") or []
    ego_target_speed = float(
        (risk.get("ego") or {}).get("target_speed_mps")
        or (ego_keyframes[0].get("speed_mps") if ego_keyframes else 0.0)
        or summary.get("ego_speed_hint_mps")
        or 6.0
    )

    root = ET.Element("scenarios")
    scenario = ET.SubElement(
        root,
        "scenario",
        {
            "name": "AutoScenario_{}".format(scene_id),
            "type": "AutoScenarioDynamic",
            "town": map_name,
        },
    )

    def actor_attributes(entity: dict, role_name: str, is_ego: bool) -> dict[str, str]:
        location = entity.get("location") or {}
        rotation = entity.get("rotation") or {}
        attrs = {
            "id": str(entity.get("id") or role_name),
            "x": _format_number(location.get("x", 0.0)),
            "y": _format_number(location.get("y", 0.0)),
            "z": _format_number(location.get("z", 0.3)),
            "yaw": _format_number(rotation.get("yaw", 0.0)),
            "model": _vehicle_model(entity, is_ego),
            "rolename": role_name,
        }
        color = entity.get("color")
        if color:
            attrs["color"] = str(color)
        return attrs

    ego_attrs = actor_attributes(ego, "hero", True)
    ET.SubElement(scenario, "ego_vehicle", ego_attrs)
    for entity in other_entities:
        ET.SubElement(
            scenario,
            "other_actor",
            actor_attributes(entity, "other_vehicles", False),
        )

    ET.SubElement(scenario, "weather", _weather_attributes(weather_name))
    extension = ET.SubElement(
        scenario,
        "autoscenario",
        {
            "scene_id": scene_id,
            "duration_s": _format_number(duration_s),
            "source_spawn_payload": str(spawn_path),
            "source_risk_dsl": str(risk_path or reconstruction_path),
            "metrics_output": str(
                dynamic_dir / "{}_lmdrive_metrics.json".format(scene_id)
            ),
        },
    )
    ET.SubElement(
        extension,
        "ego_controller",
        {
            "mode": "lmdrive_agent",
            "target_speed_mps": _format_number(ego_target_speed),
            "warmup_s": "0.0",
            "takeover_speed_tolerance_mps": "0.2",
        },
    )
    ET.SubElement(
        extension,
        "spectator",
        {
            "mode": "chase",
            "distance_m": "9.0",
            "height_m": "4.5",
            "pitch": "-15.0",
        },
    )
    ET.SubElement(
        extension,
        "leaderboard",
        {
            "route_length_m": "40.0",
            "route_step_m": "1.0",
            "timeout_s": "15.0",
        },
    )
    notice_name = _lmdrive_notice_name(risk)
    if notice_name:
        ego_location = ego.get("location") or {}
        ET.SubElement(
            extension,
            "notice",
            {
                "name": notice_name,
                "trigger_x": _format_number(ego_location.get("x", 0.0)),
                "trigger_y": _format_number(ego_location.get("y", 0.0)),
                "trigger_z": _format_number(ego_location.get("z", 0.3)),
                "source_accident_type": str(risk.get("accident_type") or ""),
            },
        )
    timeline = ET.SubElement(extension, "timeline")
    for event in risk.get("events") or []:
        if str(event.get("actor_id") or "") not in selected_actor_ids | EGO_IDS:
            continue
        trigger = event.get("trigger") or {}
        action = event.get("action") or {}
        start_s = 0.0
        if trigger.get("type") == "time_elapsed_above":
            start_s = float(trigger.get("value_s") or 0.0)
        attrs = {
            "actor_id": str(event.get("actor_id") or ""),
            "trigger": str(trigger.get("type") or "immediate"),
            "start_s": _format_number(start_s),
            "active_duration_s": _format_number(
                event.get("active_duration_s") or duration_s
            ),
            "action": str(action.get("type") or ""),
        }
        for key, value in action.items():
            if key == "type":
                continue
            if isinstance(value, list):
                attrs[key] = ",".join(str(item) for item in value)
            elif isinstance(value, bool):
                attrs[key] = str(value).lower()
            else:
                attrs[key] = str(value)
        if attrs["action"] == "decelerate_to_speed":
            # Same flying-start convention used by the generated Python scene.
            attrs["initial_speed_mps"] = "4.0"
        ET.SubElement(timeline, "event", attrs)

    if not risk_path:
        trajectories = ET.SubElement(extension, "trajectories")
        for actor in reconstruction.get("actors") or []:
            actor_id = str(actor.get("actor_id") or "")
            if actor_id in EGO_IDS or actor_id not in selected_actor_ids:
                continue
            trajectory = ET.SubElement(trajectories, "trajectory", {"actor_id": actor_id})
            for keyframe in actor.get("keyframes") or []:
                attrs = {
                    key: _format_number(keyframe.get(key, 0.0))
                    for key in (
                        "t_s",
                        "forward_m",
                        "right_m",
                        "heading_delta_deg",
                        "speed_mps",
                    )
                }
                lights = keyframe.get("lights") or []
                if lights:
                    attrs["lights"] = ",".join(str(item) for item in lights)
                ET.SubElement(trajectory, "keyframe", attrs)

    _indent(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
        stream.write(b"\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dynamic_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dynamic_dir = args.dynamic_dir.resolve()
    source_paths = sorted(dynamic_dir.glob("*_video_risk_dsl.json")) or sorted(
        dynamic_dir.glob("*_video_reconstruction.json")
    )
    if not source_paths:
        raise FileNotFoundError("No convertible dynamic source in {}".format(dynamic_dir))
    scene_id = source_paths[0].name.split("_video_")[0]
    output = args.output or dynamic_dir / "{}_lmdrive_scenario.xml".format(scene_id)
    print(convert(dynamic_dir, output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
