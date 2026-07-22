#!/usr/bin/env python3
"""Run generated numeric actor trajectories in a live CARLA world.

This module is intentionally self-contained so generated reconstruction scripts
can import it without ScenarioRunner or Leaderboard.
"""

from __future__ import print_function

import json
import math
import os
import time

import carla

import autoscenario_scene_helpers as scene_helpers


EGO_IDS = {"ego", "ego_vehicle"}


def _load_json(path):
    with open(os.path.abspath(path), "r", encoding="utf-8") as stream:
        return json.load(stream)


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _interpolate(keyframes, elapsed):
    if not keyframes:
        return {
            "forward_m": 0.0,
            "right_m": 0.0,
            "heading_delta_deg": 0.0,
            "speed_mps": 0.0,
            "lights": [],
        }
    before = keyframes[0]
    after = keyframes[-1]
    for keyframe in keyframes:
        if float(keyframe.get("t_s", 0.0)) <= elapsed:
            before = keyframe
        if float(keyframe.get("t_s", 0.0)) >= elapsed:
            after = keyframe
            break
    start = float(before.get("t_s", 0.0))
    end = float(after.get("t_s", start))
    alpha = 0.0 if end <= start else max(0.0, min(1.0, (elapsed - start) / (end - start)))
    result = {}
    for key in ("forward_m", "right_m", "heading_delta_deg", "speed_mps"):
        first = float(before.get(key, 0.0))
        second = float(after.get(key, first))
        result[key] = first + (second - first) * alpha
    result["lights"] = before.get("lights") or []
    return result


def _resolve_transform(entity):
    location_data = entity.get("location") or {}
    rotation_data = entity.get("rotation") or {}
    location = carla.Location(
        x=float(location_data.get("x", 0.0)),
        y=float(location_data.get("y", 0.0)),
        z=float(location_data.get("z", 0.3)),
    )
    rotation = carla.Rotation(
        pitch=float(rotation_data.get("pitch", 0.0)),
        yaw=float(rotation_data.get("yaw", 0.0)),
        roll=float(rotation_data.get("roll", 0.0)),
    )
    mode = str(entity.get("placement_mode") or "project_to_lane")
    if mode == "project_to_visual_pose":
        location, rotation, _ = scene_helpers._autoscenario_project_vehicle_to_visual_pose(
            location,
            rotation,
            entity.get("visual_position_override"),
            entity.get("projected_lane"),
            entity.get("lane_side_relation"),
        )
    elif mode == "project_to_opposing_lane":
        location, rotation = scene_helpers._autoscenario_project_to_opposing_lane(
            location,
            rotation,
            int(entity.get("opposing_lane_from_median") or 1),
        )
    elif mode == "project_to_parking_lane":
        location, rotation = scene_helpers._autoscenario_project_vehicle_to_parking_lane(
            location,
            rotation,
            entity.get("projected_lane"),
            entity.get("lane_side_relation"),
        )
    elif mode == "project_to_junction_lane":
        location, rotation = scene_helpers._autoscenario_project_vehicle_to_specific_lane(
            location,
            rotation,
            entity.get("projected_lane"),
            entity.get("junction_distance_m"),
        )
    elif mode not in ("exact_world_pose", "direct", "preserve_xy"):
        location, rotation = scene_helpers._autoscenario_project_vehicle_to_lane(
            location, rotation
        )
    return carla.Transform(location, rotation)


def _safe_model(entity, is_ego):
    model = str(entity.get("blueprint_name") or "")
    if model.startswith("vehicle."):
        return model
    return "vehicle.lincoln.mkz_2020" if is_ego else "vehicle.tesla.model3"


def _spawn_vehicle(world, blueprint_library, entity, is_ego):
    model = _safe_model(entity, is_ego)
    blueprint = blueprint_library.find(model)
    color = entity.get("color")
    if color and blueprint.has_attribute("color"):
        blueprint.set_attribute("color", str(color))
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero" if is_ego else "other_vehicles")
    transform = _resolve_transform(entity)
    # Add only vertical clearance. Lateral/longitudinal retries would corrupt
    # the reconstructed geometry.
    actor = None
    for extra_z in (0.2, 0.5, 0.8):
        candidate = carla.Transform(
            carla.Location(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z + extra_z,
            ),
            transform.rotation,
        )
        actor = world.try_spawn_actor(blueprint, candidate)
        if actor is not None:
            break
    if actor is None:
        raise RuntimeError(
            "Unable to spawn {} ({}) at {}".format(entity.get("id"), model, transform)
        )
    actor.set_autopilot(False)
    return actor


def _clear_dynamic_actors(world):
    actors = []
    for pattern in ("controller.ai.walker", "walker.*", "vehicle.*"):
        actors.extend(list(world.get_actors().filter(pattern)))
    for actor in actors:
        try:
            if actor.type_id == "controller.ai.walker":
                actor.stop()
        except RuntimeError:
            pass
        try:
            actor.destroy()
        except RuntimeError:
            pass


def _apply_risk_ego_speed_override(reconstruction, reconstruction_path):
    risk_path = reconstruction_path.replace(
        "_video_reconstruction.json", "_video_risk_dsl.json"
    )
    if not os.path.isfile(risk_path):
        return None
    risk = _load_json(risk_path)
    speed = float((risk.get("ego") or {}).get("target_speed_mps") or 0.0)
    if speed <= 0.0:
        return None
    for actor in reconstruction.get("actors") or []:
        if str(actor.get("actor_id")) not in EGO_IDS:
            continue
        keyframes = actor.get("keyframes") or []
        old_speed = float(keyframes[0].get("speed_mps", 0.0)) if keyframes else 0.0
        scale = speed / old_speed if old_speed > 1e-6 else 1.0
        for keyframe in keyframes:
            keyframe["forward_m"] = float(keyframe.get("forward_m", 0.0)) * scale
            keyframe["right_m"] = float(keyframe.get("right_m", 0.0)) * scale
            keyframe["speed_mps"] = speed
        return speed
    return None


def _risk_ego_profile(reconstruction_path):
    risk_path = reconstruction_path.replace(
        "_video_reconstruction.json", "_video_risk_dsl.json"
    )
    if not os.path.isfile(risk_path):
        return {}
    risk = _load_json(risk_path)
    ego = dict(risk.get("ego") or {})
    for event in risk.get("events") or []:
        if str(event.get("actor_id")) not in EGO_IDS:
            continue
        action = event.get("action") or {}
        if action.get("type") == "decelerate_to_speed":
            ego.update(action)
            ego["braking"] = True
            break
    return ego


def _set_actor_target(actor, origin, frame, brake=False):
    yaw = math.radians(origin.rotation.yaw)
    forward = float(frame["forward_m"])
    right = float(frame["right_m"])
    location = carla.Location(
        x=origin.location.x + math.cos(yaw) * forward - math.sin(yaw) * right,
        y=origin.location.y + math.sin(yaw) * forward + math.cos(yaw) * right,
        z=origin.location.z,
    )
    rotation = carla.Rotation(
        pitch=origin.rotation.pitch,
        yaw=origin.rotation.yaw + float(frame["heading_delta_deg"]),
        roll=origin.rotation.roll,
    )
    actor.set_transform(carla.Transform(location, rotation))
    speed = max(0.0, float(frame["speed_mps"]))
    radians = math.radians(rotation.yaw)
    actor.set_target_velocity(
        carla.Vector3D(math.cos(radians) * speed, math.sin(radians) * speed, 0.0)
    )
    actor.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0 if brake else 0.0,
            hand_brake=False,
        )
    )
    state = scene_helpers._autoscenario_vehicle_light_base_state()
    for light in frame.get("lights") or []:
        attribute = {
            "brake": "Brake",
            "left_blinker": "LeftBlinker",
            "right_blinker": "RightBlinker",
        }.get(str(light))
        if attribute and hasattr(carla.VehicleLightState, attribute):
            state |= getattr(carla.VehicleLightState, attribute)
    actor.set_light_state(carla.VehicleLightState(state))
    return location, rotation, speed


def _update_spectator(world, ego):
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    location = transform.location - forward * 9.0 + carla.Location(z=4.5)
    rotation = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def run_numeric_trajectory_scenario(
    client,
    world,
    blueprint_library,
    actor_payload_path,
    reconstruction_path,
    metrics_output_path,
    recording_start_callback=None,
):
    """Spawn the source actors and replay all numeric keyframes."""
    del client
    actor_payload_path = os.path.abspath(actor_payload_path)
    reconstruction_path = os.path.abspath(reconstruction_path)
    metrics_output_path = os.path.abspath(metrics_output_path)
    payload = _load_json(actor_payload_path)
    reconstruction = _load_json(reconstruction_path)
    ego_profile = _risk_ego_profile(reconstruction_path)
    speed_override = _apply_risk_ego_speed_override(
        reconstruction, reconstruction_path
    )

    original_settings = world.get_settings()
    settings_changed = not original_settings.synchronous_mode
    if settings_changed:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

    actors = {}
    collision_sensor = None
    collisions = []
    min_distance = float("inf")
    tracking = {}
    try:
        if os.environ.get("AUTOSCENARIO_CLEAR_EXISTING", "1") != "0":
            _clear_dynamic_actors(world)
            world.tick()
        scene_helpers._autoscenario_init(world, blueprint_library)
        weather_name = (payload.get("metadata") or {}).get(
            "carla_weather_preset", "ClearNoon"
        )
        scene_helpers._autoscenario_apply_weather(weather_name)
        ego_yaw = scene_helpers._autoscenario_payload_ego_yaw(payload)
        for entity in payload.get("entities") or []:
            if str(entity.get("spawn_kind") or "vehicle") != "vehicle":
                continue
            entity = dict(entity)
            rotation = dict(entity.get("rotation") or {})
            adjusted = scene_helpers._autoscenario_apply_heading_relation(
                entity,
                carla.Rotation(
                    pitch=float(rotation.get("pitch", 0.0)),
                    yaw=float(rotation.get("yaw", 0.0)),
                    roll=float(rotation.get("roll", 0.0)),
                ),
                ego_yaw,
            )
            entity["rotation"] = {
                "pitch": adjusted.pitch,
                "yaw": adjusted.yaw,
                "roll": adjusted.roll,
            }
            actor_id = str(entity.get("id"))
            print("Spawning reconstructed actor {}".format(actor_id), flush=True)
            actors[actor_id] = _spawn_vehicle(
                world, blueprint_library, entity, actor_id in EGO_IDS
            )
            world.tick()
        ego = next((actors[key] for key in EGO_IDS if key in actors), None)
        if ego is None:
            raise RuntimeError("Actor payload contains no ego vehicle")
        scene_helpers._autoscenario_configure_vehicle_lights_for_environment(
            actors, payload, weather_name
        )

        collision_bp = blueprint_library.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        collision_sensor.listen(
            lambda event: collisions.append(
                {
                    "frame": int(event.frame),
                    "other_actor_id": int(event.other_actor.id),
                    "other_actor_type": str(event.other_actor.type_id),
                }
            )
        )
        origins = {actor_id: actor.get_transform() for actor_id, actor in actors.items()}
        trajectories = {
            str(item.get("actor_id")): item
            for item in reconstruction.get("actors") or []
        }
        for actor_id in actors:
            tracking[actor_id] = []

        if speed_override is not None:
            print(
                "Applied ego speed override from risk DSL: {:.3f}m/s".format(
                    speed_override
                ),
                flush=True,
            )
        if ego_profile.get("braking"):
            print(
                "Applied ego braking profile: initial={:.3f}m/s target={:.3f}m/s decel={:.3f}m/s^2".format(
                    float(ego_profile.get("initial_speed_mps", 0.0)),
                    float(ego_profile.get("target_speed_mps", 0.0)),
                    float(ego_profile.get("max_decel_mps2", 0.0)),
                ),
                flush=True,
            )
        if recording_start_callback is not None:
            recording_start_callback()

        duration = float(reconstruction.get("duration_s", 4.0))
        fixed_delta = 0.05
        steps = int(math.ceil(duration / fixed_delta))
        realtime = os.environ.get("AUTOSCENARIO_REALTIME", "1") != "0"
        for step in range(steps + 1):
            elapsed = min(duration, step * fixed_delta)
            targets = {}
            for actor_id, actor in actors.items():
                trajectory = trajectories.get(actor_id) or {}
                frame = _interpolate(trajectory.get("keyframes") or [], elapsed)
                targets[actor_id] = _set_actor_target(
                    actor,
                    origins[actor_id],
                    frame,
                    brake=(actor_id in EGO_IDS and ego_profile.get("braking", False)),
                )
            _update_spectator(world, ego)
            world.tick()
            ego_location = ego.get_location()
            for actor_id, actor in actors.items():
                target_location, target_rotation, target_speed = targets[actor_id]
                actual = actor.get_transform()
                error = actual.location.distance(target_location)
                tracking[actor_id].append(
                    {
                        "t_s": round(elapsed, 3),
                        "position_error_m": round(error, 6),
                        "speed_error_mps": round(abs(_speed(actor) - target_speed), 6),
                    }
                )
                if actor_id not in EGO_IDS:
                    min_distance = min(
                        min_distance, ego_location.distance(actual.location)
                    )
            if realtime:
                time.sleep(fixed_delta)

        metric_payload = {
            "schema_version": "numeric-trajectory-runtime-v1",
            "scene_id": reconstruction.get("scene_id"),
            "duration_s": duration,
            "ego_speed_override_mps": speed_override,
            "ego_profile": ego_profile,
            "collision": bool(collisions),
            "collision_events": collisions,
            "min_center_distance_m": (
                None if math.isinf(min_distance) else round(min_distance, 6)
            ),
            "actor_tracking": {},
        }
        for actor_id, samples in tracking.items():
            errors = [sample["position_error_m"] for sample in samples]
            metric_payload["actor_tracking"][actor_id] = {
                "sample_count": len(samples),
                "position_max_m": max(errors) if errors else None,
                "position_rmse_m": (
                    math.sqrt(sum(value * value for value in errors) / len(errors))
                    if errors
                    else None
                ),
                "samples": samples,
            }
        with open(metrics_output_path, "w", encoding="utf-8") as stream:
            json.dump(metric_payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print("Numeric trajectory metrics: {}".format(metrics_output_path), flush=True)
    finally:
        if collision_sensor is not None:
            try:
                collision_sensor.stop()
                collision_sensor.destroy()
            except RuntimeError:
                pass
        if os.environ.get("AUTOSCENARIO_KEEP_ACTORS", "0") != "1":
            for actor in actors.values():
                try:
                    actor.destroy()
                except RuntimeError:
                    pass
        if settings_changed:
            world.apply_settings(original_settings)
