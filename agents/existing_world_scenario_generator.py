import textwrap
from typing import Optional

from agents.scenario_generator import ScenarioGenerator


class ExistingWorldScenarioGenerator(ScenarioGenerator):
    """Deterministic CARLA scene generator for existing-world spawn payloads."""

    @staticmethod
    def _build_scene_helper_block_LEGACY() -> str:
        base_block = ScenarioGenerator._build_scene_helper_block_LEGACY()
        existing_world_block = textwrap.dedent(
            """


            def _autoscenario_blueprint_matches_category(blueprint, category, semantic_name):
                if blueprint is None or category != "vehicle":
                    return blueprint is not None

                normalized_name = _autoscenario_normalize_key(semantic_name)
                try:
                    if blueprint.has_attribute("number_of_wheels"):
                        wheels_attr = blueprint.get_attribute("number_of_wheels")
                        wheels = int(getattr(wheels_attr, "as_int", lambda: wheels_attr)())
                    else:
                        wheels = None
                except Exception:
                    wheels = None

                if wheels is None:
                    return True
                if normalized_name in {"bike", "motorcycle"}:
                    return wheels <= 2
                return wheels >= 4


            def _autoscenario_pick_blueprint(category, semantic_name):
                normalized_name = _autoscenario_normalize_key(semantic_name)
                if category == "vehicle":
                    candidate_ids = list(_AUTOSCENARIO_VEHICLE_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = ["vehicle.*"]
                elif category == "static":
                    candidate_ids = list(_AUTOSCENARIO_STATIC_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = [
                        f"static.prop.*{normalized_name}*",
                        "static.prop.*",
                    ]
                else:
                    candidate_ids = []
                    fallback_patterns = []

                if semantic_name and "." in str(semantic_name):
                    candidate_ids.insert(0, str(semantic_name))
                elif semantic_name:
                    prefix = "vehicle." if category == "vehicle" else "static.prop."
                    candidate_ids.extend(
                        [
                            f"{prefix}{semantic_name}",
                            f"{prefix}{normalized_name}",
                        ]
                    )

                seen = set()
                for blueprint_id in candidate_ids:
                    if not blueprint_id or blueprint_id in seen:
                        continue
                    seen.add(blueprint_id)
                    try:
                        blueprint = blueprint_library.find(blueprint_id)
                    except Exception:
                        continue
                    if _autoscenario_blueprint_matches_category(
                        blueprint, category, semantic_name
                    ):
                        return blueprint

                for pattern in fallback_patterns:
                    try:
                        matches = list(blueprint_library.filter(pattern))
                    except Exception:
                        matches = []
                    filtered = [
                        blueprint
                        for blueprint in matches
                        if _autoscenario_blueprint_matches_category(
                            blueprint, category, semantic_name
                        )
                    ]
                    if filtered:
                        return random.choice(filtered)

                return None


            def _autoscenario_normalize_yaw(yaw_value):
                while yaw_value <= -180.0:
                    yaw_value += 360.0
                while yaw_value > 180.0:
                    yaw_value -= 360.0
                return yaw_value


            def _autoscenario_angle_distance(yaw_a, yaw_b):
                return abs(_autoscenario_normalize_yaw(yaw_a - yaw_b))


            _AUTOSCENARIO_EXISTING_VEHICLES = []


            def _autoscenario_actor_xy_distance(actor, location):
                try:
                    actor_location = actor.get_transform().location
                    return math.sqrt(
                        (float(actor_location.x) - float(location.x)) ** 2
                        + (float(actor_location.y) - float(location.y)) ** 2
                    )
                except Exception:
                    return float("inf")


            def _autoscenario_collect_existing_vehicles():
                try:
                    return [
                        actor
                        for actor in list(world.get_actors().filter("vehicle.*"))
                        if actor not in _AUTOSCENARIO_SPAWNED_ACTORS
                    ]
                except Exception:
                    return []


            def _autoscenario_spawn_location_is_clear(location, min_distance_m=2.5):
                try_location = _autoscenario_to_location(location)
                for actor in list(_AUTOSCENARIO_EXISTING_VEHICLES) + list(_AUTOSCENARIO_SPAWNED_ACTORS):
                    if actor is None:
                        continue
                    if _autoscenario_actor_xy_distance(actor, try_location) < float(min_distance_m):
                        return False
                return True


            def _autoscenario_min_spawn_spacing_for_blueprint(blueprint):
                try:
                    if blueprint is not None and blueprint.has_attribute("number_of_wheels"):
                        wheels_attr = blueprint.get_attribute("number_of_wheels")
                        wheels = int(getattr(wheels_attr, "as_int", lambda: wheels_attr)())
                        if wheels <= 2:
                            return 2.4
                except Exception:
                    pass
                return 4.5


            def _autoscenario_try_spawn(blueprint, location, rotation, actor_kind):
                if blueprint is None:
                    return None

                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                yaw_radians = math.radians(base_rotation.yaw)
                forward_x = math.cos(yaw_radians)
                forward_y = math.sin(yaw_radians)
                right_x = -math.sin(yaw_radians)
                right_y = math.cos(yaw_radians)

                if actor_kind == "vehicle":
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (4.0, 0.0, 0.35),
                        (-4.0, 0.0, 0.35),
                        (8.0, 0.0, 0.35),
                        (-8.0, 0.0, 0.35),
                        (0.0, 2.5, 0.35),
                        (0.0, -2.5, 0.35),
                    ]
                    min_spacing = _autoscenario_min_spawn_spacing_for_blueprint(blueprint)
                elif actor_kind == "walker":
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (1.0, 0.0, 0.35),
                        (-1.0, 0.0, 0.35),
                        (0.0, 1.0, 0.35),
                        (0.0, -1.0, 0.35),
                    ]
                    min_spacing = 0.8
                else:
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (0.8, 0.0, 0.35),
                        (-0.8, 0.0, 0.35),
                        (0.0, 0.8, 0.35),
                        (0.0, -0.8, 0.35),
                    ]
                    min_spacing = 0.8

                for forward_offset, right_offset, dz in offsets:
                    try_location = carla.Location(
                        base_location.x
                        + forward_x * forward_offset
                        + right_x * right_offset,
                        base_location.y
                        + forward_y * forward_offset
                        + right_y * right_offset,
                        base_location.z + dz,
                    )
                    if actor_kind == "vehicle" and not _autoscenario_spawn_location_is_clear(
                        try_location,
                        min_spacing,
                    ):
                        continue
                    actor = world.try_spawn_actor(
                        blueprint, carla.Transform(try_location, base_rotation)
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    if actor_kind == "vehicle":
                        try:
                            actor.set_autopilot(False)
                        except Exception:
                            pass
                        try:
                            actor.apply_control(carla.VehicleControl(brake=1.0))
                        except Exception:
                            pass
                    elif actor_kind == "static":
                        try:
                            actor.set_simulate_physics(True)
                        except Exception:
                            pass
                    return actor
                return None


            def _autoscenario_project_vehicle_to_lane(location, rotation):
                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                try:
                    world_map = world.get_map()
                except Exception:
                    return base_location, base_rotation

                waypoint = None
                lane_type = getattr(carla.LaneType, "Driving", None)
                try:
                    if lane_type is not None:
                        waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                            lane_type=lane_type,
                        )
                    else:
                        waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                        )
                except TypeError:
                    try:
                        waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        waypoint = None
                except Exception:
                    waypoint = None

                if waypoint is None:
                    return base_location, base_rotation

                snapped_location = waypoint.transform.location
                waypoint_yaw = float(waypoint.transform.rotation.yaw)
                candidate_yaws = [waypoint_yaw, waypoint_yaw + 180.0]
                snapped_yaw = min(
                    candidate_yaws,
                    key=lambda yaw_value: _autoscenario_angle_distance(
                        yaw_value,
                        float(base_rotation.yaw),
                    ),
                )

                projected_location = carla.Location(
                    snapped_location.x,
                    snapped_location.y,
                    snapped_location.z + 0.35,
                )
                projected_rotation = carla.Rotation(
                    float(base_rotation.pitch),
                    _autoscenario_normalize_yaw(snapped_yaw),
                    float(base_rotation.roll),
                )
                return projected_location, projected_rotation


            def _autoscenario_project_vehicle_to_parking_lane(location, rotation, projected_lane=None):
                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                projected_lane = projected_lane if isinstance(projected_lane, dict) else {}
                try:
                    world_map = world.get_map()
                except Exception:
                    return base_location, base_rotation

                parking_type = getattr(carla.LaneType, "Parking", None)
                try:
                    if parking_type is not None:
                        waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                            lane_type=parking_type,
                        )
                    else:
                        waypoint = world_map.get_waypoint(base_location, project_to_road=True)
                except TypeError:
                    try:
                        waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        waypoint = None
                except Exception:
                    waypoint = None

                if waypoint is None:
                    return base_location, base_rotation
                try:
                    is_parking = waypoint.lane_type == carla.LaneType.Parking
                except Exception:
                    is_parking = "parking" in str(getattr(waypoint, "lane_type", "")).lower()
                if not is_parking:
                    return base_location, base_rotation
                # The projected lane may be the fallback driving lane; the nearest
                # Parking waypoint is the stronger semantic signal here.

                snapped_location = waypoint.transform.location
                waypoint_yaw = float(waypoint.transform.rotation.yaw)
                try:
                    yaw_reference = float(projected_lane.get("yaw"))
                except Exception:
                    yaw_reference = float(base_rotation.yaw)
                snapped_yaw = min(
                    [waypoint_yaw, waypoint_yaw + 180.0],
                    key=lambda yaw_value: _autoscenario_angle_distance(
                        yaw_value,
                        yaw_reference,
                    ),
                )
                return (
                    carla.Location(snapped_location.x, snapped_location.y, snapped_location.z + 0.35),
                    carla.Rotation(
                        float(base_rotation.pitch),
                        _autoscenario_normalize_yaw(snapped_yaw),
                        float(base_rotation.roll),
                    ),
                )


            def _autoscenario_collect_vehicle_spawn_candidates(location, rotation):
                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                try:
                    world_map = world.get_map()
                except Exception:
                    return [(base_location, base_rotation)]

                lane_type = getattr(carla.LaneType, "Driving", None)
                try:
                    if lane_type is not None:
                        base_waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                            lane_type=lane_type,
                        )
                    else:
                        base_waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                        )
                except TypeError:
                    try:
                        base_waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        base_waypoint = None
                except Exception:
                    base_waypoint = None

                if base_waypoint is None:
                    return [(base_location, base_rotation)]

                def _candidate_from_waypoint(waypoint):
                    location = waypoint.transform.location
                    waypoint_yaw = float(waypoint.transform.rotation.yaw)
                    candidate_yaws = [waypoint_yaw, waypoint_yaw + 180.0]
                    snapped_yaw = min(
                        candidate_yaws,
                        key=lambda yaw_value: _autoscenario_angle_distance(
                            yaw_value,
                            float(base_rotation.yaw),
                        ),
                    )
                    return (
                        carla.Location(location.x, location.y, location.z + 0.6),
                        carla.Rotation(
                            float(base_rotation.pitch),
                            _autoscenario_normalize_yaw(snapped_yaw),
                            float(base_rotation.roll),
                        ),
                    )

                candidates = []
                seen = set()

                def _append_waypoint(waypoint):
                    if waypoint is None:
                        return
                    key = (
                        round(float(waypoint.transform.location.x), 2),
                        round(float(waypoint.transform.location.y), 2),
                        int(waypoint.road_id),
                        int(waypoint.lane_id),
                    )
                    if key in seen:
                        return
                    seen.add(key)
                    candidates.append(_candidate_from_waypoint(waypoint))

                _append_waypoint(base_waypoint)

                for distance in (8.0, 16.0, 24.0, 32.0):
                    try:
                        next_waypoints = base_waypoint.next(distance)
                    except Exception:
                        next_waypoints = []
                    for waypoint in next_waypoints or []:
                        _append_waypoint(waypoint)

                    previous_method = getattr(base_waypoint, "previous", None)
                    if previous_method is None:
                        continue
                    try:
                        previous_waypoints = previous_method(distance)
                    except Exception:
                        previous_waypoints = []
                    for waypoint in previous_waypoints or []:
                        _append_waypoint(waypoint)

                if not candidates:
                    candidates.append((base_location, base_rotation))
                return candidates


            def _autoscenario_try_spawn_vehicle_actor(blueprint, location, rotation):
                if blueprint is None:
                    return None

                for candidate_location, candidate_rotation in _autoscenario_collect_vehicle_spawn_candidates(
                    location,
                    rotation,
                ):
                    if not _autoscenario_spawn_location_is_clear(
                        candidate_location,
                        _autoscenario_min_spawn_spacing_for_blueprint(blueprint),
                    ):
                        continue
                    actor = world.try_spawn_actor(
                        blueprint,
                        carla.Transform(candidate_location, candidate_rotation),
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    try:
                        actor.set_autopilot(False)
                    except Exception:
                        pass
                    try:
                        actor.apply_control(carla.VehicleControl(brake=1.0))
                    except Exception:
                        pass
                    return actor
                return None


            def _autoscenario_project_static_to_ground(location):
                base_location = _autoscenario_to_location(location)
                ground_z = max(0.02, float(base_location.z))
                try:
                    world_map = world.get_map()
                    waypoint = world_map.get_waypoint(
                        base_location,
                        project_to_road=True,
                    )
                except TypeError:
                    try:
                        waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        waypoint = None
                except Exception:
                    waypoint = None

                if waypoint is not None:
                    ground_z = max(0.02, float(waypoint.transform.location.z) + 0.02)
                return carla.Location(float(base_location.x), float(base_location.y), ground_z)


            def _autoscenario_try_spawn_static(blueprint, location, rotation):
                if blueprint is None:
                    return None

                base_location = _autoscenario_project_static_to_ground(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                offsets = [
                    (0.0, 0.0, 0.0),
                    (0.3, 0.0, 0.0),
                    (-0.3, 0.0, 0.0),
                    (0.0, 0.3, 0.0),
                    (0.0, -0.3, 0.0),
                ]

                for dx, dy, dz in offsets:
                    try_location = carla.Location(
                        base_location.x + dx,
                        base_location.y + dy,
                        base_location.z + dz,
                    )
                    actor = world.try_spawn_actor(
                        blueprint, carla.Transform(try_location, base_rotation)
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    try:
                        actor.set_simulate_physics(True)
                    except Exception:
                        pass
                    return actor
                return None


            def _autoscenario_spawn_static_prop(blueprint_name, location, rotation):
                snapped_location = _autoscenario_project_static_to_ground(location)
                _autoscenario_record_focus_point(snapped_location)
                blueprint = _autoscenario_pick_blueprint("static", blueprint_name)
                return _autoscenario_try_spawn_static(
                    blueprint,
                    snapped_location,
                    rotation,
                )


            def _autoscenario_apply_role_name(blueprint, role_name):
                if blueprint is None or not role_name:
                    return
                try:
                    if blueprint.has_attribute("role_name"):
                        blueprint.set_attribute("role_name", str(role_name))
                except Exception:
                    pass


            def _autoscenario_spawn_vehicle(blueprint_name, location, rotation, color=None, role_name=None):
                snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_lane(location, rotation)
                _autoscenario_record_focus_point(snapped_location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                _autoscenario_apply_role_name(blueprint, role_name)
                return _autoscenario_try_spawn_vehicle_actor(
                    blueprint,
                    snapped_location,
                    snapped_rotation,
                )


            def _autoscenario_spawn_vehicle_parking_lane(blueprint_name, location, rotation, projected_lane=None, color=None, role_name=None):
                snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_parking_lane(
                    location,
                    rotation,
                    projected_lane,
                )
                _autoscenario_record_focus_point(snapped_location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                _autoscenario_apply_role_name(blueprint, role_name)
                return _autoscenario_try_spawn_vehicle_actor_strict_lane(
                    blueprint,
                    snapped_location,
                    snapped_rotation,
                )


            def _autoscenario_spawn_vehicle_direct(blueprint_name, location, rotation, color=None, role_name=None):
                _autoscenario_record_focus_point(location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                _autoscenario_apply_role_name(blueprint, role_name)
                return _autoscenario_try_spawn(
                    blueprint,
                    location,
                    rotation,
                    "vehicle",
                )


            def _autoscenario_capture_bev_if_requested():
                output_path = os.environ.get("AUTOSCENARIO_BEV_OUTPUT")
                if not output_path:
                    return

                focus_locations = []
                for actor in list(_AUTOSCENARIO_SPAWNED_ACTORS):
                    try:
                        focus_locations.append(actor.get_transform().location)
                    except Exception:
                        pass
                if not focus_locations:
                    focus_locations = list(_AUTOSCENARIO_FOCUS_POINTS)
                if not focus_locations:
                    return

                avg_x = sum(location.x for location in focus_locations) / len(focus_locations)
                avg_y = sum(location.y for location in focus_locations) / len(focus_locations)
                max_z = max(location.z for location in focus_locations)
                camera_height = float(os.environ.get("AUTOSCENARIO_BEV_HEIGHT", "80"))
                image_size = str(os.environ.get("AUTOSCENARIO_BEV_SIZE", "1024"))
                fov = str(os.environ.get("AUTOSCENARIO_BEV_FOV", "55"))

                try:
                    blueprint = blueprint_library.find("sensor.camera.rgb")
                    blueprint.set_attribute("image_size_x", image_size)
                    blueprint.set_attribute("image_size_y", image_size)
                    blueprint.set_attribute("fov", fov)
                    blueprint.set_attribute("sensor_tick", "0.05")
                except Exception:
                    return

                transform = carla.Transform(
                    carla.Location(avg_x, avg_y, max_z + camera_height),
                    carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
                )
                sensor = None
                image_queue = queue.Queue()
                try:
                    sensor = world.spawn_actor(blueprint, transform)
                    sensor.listen(lambda image: image_queue.put(image))
                    try:
                        world.tick()
                    except Exception:
                        world.wait_for_tick()
                    image = image_queue.get(timeout=5.0)
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    image.save_to_disk(output_path)
                except Exception as exc:
                    print(f"BEV capture failed: {exc}")
                finally:
                    if sensor is not None:
                        try:
                            sensor.stop()
                        except Exception:
                            pass
                        try:
                            sensor.destroy()
                        except Exception:
                            pass


            def _autoscenario_capture_ego_view_if_requested(actor_by_id=None):
                output_path = os.environ.get("AUTOSCENARIO_EGO_VIEW_OUTPUT")
                if not output_path:
                    return
                actor_by_id = actor_by_id or {}
                ego_actor = actor_by_id.get("ego") or actor_by_id.get("ego_vehicle")
                if ego_actor is None:
                    return

                image_size = str(os.environ.get("AUTOSCENARIO_EGO_VIEW_SIZE", "1024"))
                fov = str(os.environ.get("AUTOSCENARIO_EGO_VIEW_FOV", "90"))
                try:
                    blueprint = blueprint_library.find("sensor.camera.rgb")
                    blueprint.set_attribute("image_size_x", image_size)
                    blueprint.set_attribute("image_size_y", image_size)
                    blueprint.set_attribute("fov", fov)
                    blueprint.set_attribute("sensor_tick", "0.05")
                except Exception:
                    return

                transform = carla.Transform(
                    carla.Location(x=1.2, y=0.0, z=1.6),
                    carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
                )
                sensor = None
                image_queue = queue.Queue()
                try:
                    sensor = world.spawn_actor(blueprint, transform, attach_to=ego_actor)
                    sensor.listen(lambda image: image_queue.put(image))
                    try:
                        world.tick()
                    except Exception:
                        world.wait_for_tick()
                    image = image_queue.get(timeout=5.0)
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    image.save_to_disk(output_path)
                except Exception as exc:
                    print(f"Ego-view capture failed: {exc}")
                finally:
                    if sensor is not None:
                        try:
                            sensor.stop()
                        except Exception:
                            pass
                        try:
                            sensor.destroy()
                        except Exception:
                            pass


            def _autoscenario_distance_band_from_longitudinal(longitudinal_m):
                distance = abs(float(longitudinal_m))
                if distance < 12.0:
                    return "near"
                if distance < 30.0:
                    return "mid"
                return "far"


            def _autoscenario_lane_side_from_lateral(lateral_m):
                lateral = float(lateral_m)
                if lateral <= -1.25:
                    return "left_lane"
                if lateral >= 1.25:
                    return "right_lane"
                return "same_lane"


            def _autoscenario_heading_relation_to_ego(actor_yaw, ego_yaw):
                if actor_yaw is None or ego_yaw is None:
                    return "unknown"
                return (
                    "opposite_direction"
                    if _autoscenario_angle_distance(float(actor_yaw), float(ego_yaw)) > 90.0
                    else "same_direction"
                )


            def _autoscenario_actor_transform_record(actor):
                transform = actor.get_transform()
                location = transform.location
                rotation = transform.rotation
                try:
                    blueprint_id = actor.type_id
                except Exception:
                    blueprint_id = None
                return {
                    "location": {
                        "x": float(location.x),
                        "y": float(location.y),
                        "z": float(location.z),
                    },
                    "yaw": float(rotation.yaw),
                    "blueprint_id": blueprint_id,
                }


            def _autoscenario_ego_frame(location, ego_record):
                ego_location = ego_record.get("location") or {}
                ego_yaw = float(ego_record.get("yaw") or 0.0)
                yaw_rad = math.radians(ego_yaw)
                forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
                right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)
                dx = float(location.get("x", 0.0)) - float(ego_location.get("x", 0.0))
                dy = float(location.get("y", 0.0)) - float(ego_location.get("y", 0.0))
                longitudinal = dx * forward_x + dy * forward_y
                lateral = dx * right_x + dy * right_y
                return {
                    "longitudinal_m": round(longitudinal, 3),
                    "lateral_m": round(lateral, 3),
                    "distance_band": _autoscenario_distance_band_from_longitudinal(longitudinal),
                    "lane_side_relation": _autoscenario_lane_side_from_lateral(lateral),
                }


            def _autoscenario_capture_render_actor_graph_if_requested(spawn_payload, actor_by_id=None):
                output_path = os.environ.get("AUTOSCENARIO_RENDER_ACTOR_GRAPH_OUTPUT")
                if not output_path:
                    return
                actor_by_id = actor_by_id or {}
                ego_actor = actor_by_id.get("ego") or actor_by_id.get("ego_vehicle")
                ego_record = _autoscenario_actor_transform_record(ego_actor) if ego_actor is not None else {}
                actors = []
                ignored_actors = []
                for entity in spawn_payload.get("entities", []):
                    entity_id = str(entity.get("id") or "")
                    if entity_id in {"ego", "ego_vehicle"}:
                        continue
                    category = str(entity.get("category") or "car")
                    actor = actor_by_id.get(entity_id)
                    record = {
                        "id": entity_id,
                        "category": category,
                        "subtype": str(entity.get("subtype") or category),
                        "spawn_kind": str(entity.get("spawn_kind") or "vehicle"),
                        "blueprint_name": entity.get("blueprint_name"),
                        "lane_side_relation": str(entity.get("lane_side_relation") or "same_lane"),
                        "placement_mode": str(entity.get("placement_mode") or "project_to_lane"),
                        "truth_source": "carla_actor_transform",
                    }
                    if actor is None:
                        record["spawned"] = False
                        record["spawn_failure_reason"] = "try_spawn_actor_returned_none_or_collision"
                    else:
                        actor_record = _autoscenario_actor_transform_record(actor)
                        record["spawned"] = True
                        record["spawn_failure_reason"] = None
                        record["location"] = actor_record["location"]
                        record["yaw"] = actor_record["yaw"]
                        record["blueprint_id"] = actor_record.get("blueprint_id")
                        if ego_record:
                            record["ego_frame"] = _autoscenario_ego_frame(
                                actor_record["location"],
                                ego_record,
                            )
                            record["heading_relation_to_ego"] = _autoscenario_heading_relation_to_ego(
                                actor_record.get("yaw"),
                                ego_record.get("yaw"),
                            )
                        else:
                            record["heading_relation_to_ego"] = "unknown"
                    if category in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                        actors.append(record)
                    else:
                        ignored_actors.append(record)
                graph = {
                    "schema_version": "actor-graph-v1",
                    "graph_type": "render_actor_graph",
                    "truth_source": "carla_actor_transform",
                    "position_checks_enabled": True,
                    "spawn_checks_enabled": True,
                    "actors": actors,
                    "ignored_actors": ignored_actors,
                    "metadata": {
                        "actor_count": len(actors),
                        "truth_unavailable": False,
                    },
                }
                try:
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as file:
                        json.dump(graph, file, indent=2, sort_keys=True)
                except Exception as exc:
                    print(f"Render actor graph capture failed: {exc}")
            """
        )
        return base_block + existing_world_block

    @staticmethod
    def _build_spawn_payload_loop() -> str:
        return (
            "def _autoscenario_load_spawn_payload():\n"
            "    payload_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_SPAWN_PAYLOAD)\n"
            "    with open(payload_path, 'r', encoding='utf-8') as file:\n"
            "        return json.load(file)\n\n"
            "def _autoscenario_payload_ego_yaw(spawn_payload):\n"
            "    for payload_entity in spawn_payload.get('entities', []):\n"
            "        if str(payload_entity.get('id')) in {'ego', 'ego_vehicle'}:\n"
            "            try:\n"
            "                return float((payload_entity.get('rotation') or {}).get('yaw'))\n"
            "            except Exception:\n"
            "                return None\n"
            "    return None\n\n"
            "def _autoscenario_apply_heading_relation(entity, rotation, ego_yaw):\n"
            "    if ego_yaw is None:\n"
            "        return rotation\n"
            "    if str(entity.get('flow_compliance') or '').lower() == 'wrong_way':\n"
            "        return rotation\n"
            "    relation = str(entity.get('heading_relation') or 'unknown')\n"
            "    yaw = float(rotation.yaw)\n"
            "    if relation == 'opposite_direction' and _autoscenario_angle_distance(yaw, ego_yaw) < 90.0:\n"
            "        rotation.yaw = _autoscenario_normalize_yaw(yaw + 180.0)\n"
            "    elif relation == 'same_direction' and _autoscenario_angle_distance(yaw, ego_yaw) > 90.0:\n"
            "        rotation.yaw = _autoscenario_normalize_yaw(yaw + 180.0)\n"
            "    return rotation\n\n"
            "def _autoscenario_clear_existing_dynamic_actors():\n"
            "    for actor_filter in ('controller.ai.walker', 'walker.*', 'vehicle.*'):\n"
            "        try:\n"
            "            existing_actors = list(world.get_actors().filter(actor_filter))\n"
            "        except Exception:\n"
            "            existing_actors = []\n"
            "        for actor in existing_actors:\n"
            "            try:\n"
            "                if actor_filter == 'controller.ai.walker':\n"
            "                    actor.stop()\n"
            "            except Exception:\n"
            "                pass\n"
            "            try:\n"
            "                actor.destroy()\n"
            "            except Exception:\n"
            "                pass\n"
            "    try:\n"
            "        world.wait_for_tick()\n"
            "    except Exception:\n"
            "        time.sleep(0.5)\n\n"
            "spawn_payload = _autoscenario_load_spawn_payload()\n"
            "_autoscenario_weather = os.environ.get('AUTOSCENARIO_WEATHER_OVERRIDE') or (spawn_payload.get('metadata') or {}).get('carla_weather_preset') or 'ClearNoon'\n"
            "_autoscenario_apply_weather(_autoscenario_weather)\n"
            "if os.environ.get('AUTOSCENARIO_CLEAR_EXISTING') == '1':\n"
            "    _autoscenario_clear_existing_dynamic_actors()\n"
            "_AUTOSCENARIO_EXISTING_VEHICLES = _autoscenario_collect_existing_vehicles()\n"
            "_autoscenario_ego_yaw = _autoscenario_payload_ego_yaw(spawn_payload)\n"
            "_autoscenario_actor_by_id = {}\n"
            "for entity in spawn_payload.get('entities', []):\n"
            "    location = carla.Location(\n"
            "        x=float(entity['location']['x']),\n"
            "        y=float(entity['location']['y']),\n"
            "        z=float(entity['location']['z']),\n"
            "    )\n"
            "    rotation = carla.Rotation(\n"
            "        pitch=float(entity['rotation']['pitch']),\n"
            "        yaw=float(entity['rotation']['yaw']),\n"
            "        roll=float(entity['rotation']['roll']),\n"
            "    )\n"
            "    rotation = _autoscenario_apply_heading_relation(entity, rotation, _autoscenario_ego_yaw)\n"
            "    spawn_kind = str(entity.get('spawn_kind') or 'vehicle')\n"
            "    entity_id = str(entity.get('id') or '')\n"
            "    actor = None\n"
            "    if spawn_kind == 'pedestrian':\n"
            "        actor = _autoscenario_spawn_pedestrian(location, rotation)\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    blueprint_name = entity.get('blueprint_name')\n"
            "    if spawn_kind == 'static':\n"
            "        actor = _autoscenario_spawn_static_prop(blueprint_name, location, rotation)\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    placement_mode = str(entity.get('placement_mode') or 'project_to_lane')\n"
            "    if placement_mode in {'direct', 'preserve_xy'}:\n"
            "        actor = _autoscenario_spawn_vehicle_direct(\n"
            "            blueprint_name,\n"
            "            location,\n"
            "            rotation,\n"
            "            entity.get('color'),\n"
            "        )\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    if placement_mode == 'project_to_opposing_lane':\n"
            "        actor = _autoscenario_spawn_vehicle_opposing(\n"
            "            blueprint_name,\n"
            "            location,\n"
            "            rotation,\n"
            "            int(entity.get('opposing_lane_from_median') or 1),\n"
            "            entity.get('color'),\n"
            "        )\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    if placement_mode == 'project_to_parking_lane':\n"
            "        actor = _autoscenario_spawn_vehicle_parking_lane(\n"
            "            blueprint_name,\n"
            "            location,\n"
            "            rotation,\n"
            "            entity.get('projected_lane'),\n"
            "            entity.get('color'),\n"
            "        )\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    if placement_mode == 'project_to_junction_lane':\n"
            "        actor = _autoscenario_spawn_vehicle_junction_lane(\n"
            "            blueprint_name,\n"
            "            location,\n"
            "            rotation,\n"
            "            entity.get('projected_lane'),\n"
            "            entity.get('color'),\n"
            "            junction_distance_m=entity.get('junction_distance_m'),\n"
            "        )\n"
            "        if actor is not None and entity_id:\n"
            "            _autoscenario_actor_by_id[entity_id] = actor\n"
            "        continue\n"
            "    actor = _autoscenario_spawn_vehicle(\n"
            "        blueprint_name,\n"
            "        location,\n"
            "        rotation,\n"
            "        entity.get('color'),\n"
            "    )\n\n"
            "    if actor is not None and entity_id:\n"
            "        _autoscenario_actor_by_id[entity_id] = actor\n\n"
            "_autoscenario_focus_spectator()\n"
            "_autoscenario_capture_render_actor_graph_if_requested(spawn_payload, _autoscenario_actor_by_id)\n"
            "_autoscenario_capture_ego_view_if_requested(_autoscenario_actor_by_id)\n"
            "_autoscenario_capture_bev_if_requested()\n"
            "time.sleep(2)\n"
        )

    @staticmethod
    def _build_dynamic_risk_loop() -> str:
        return textwrap.dedent(
            """
            def _autoscenario_load_spawn_payload():
                payload_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_SPAWN_PAYLOAD)
                with open(payload_path, 'r', encoding='utf-8') as file:
                    return json.load(file)


            def _autoscenario_load_risk_sample():
                sample_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_RISK_SAMPLE)
                with open(sample_path, 'r', encoding='utf-8') as file:
                    return json.load(file)


            def _autoscenario_clear_existing_dynamic_actors():
                for actor_filter in ('controller.ai.walker', 'walker.*', 'vehicle.*'):
                    try:
                        existing_actors = list(world.get_actors().filter(actor_filter))
                    except Exception:
                        existing_actors = []
                    for actor in existing_actors:
                        try:
                            if actor_filter == 'controller.ai.walker':
                                actor.stop()
                        except Exception:
                            pass
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                try:
                    world.wait_for_tick()
                except Exception:
                    time.sleep(0.5)


            def _autoscenario_spawn_payload_actors(spawn_payload):
                actor_by_id = {}
                for entity in spawn_payload.get('entities', []):
                    location = carla.Location(
                        x=float(entity['location']['x']),
                        y=float(entity['location']['y']),
                        z=float(entity['location']['z']),
                    )
                    rotation = carla.Rotation(
                        pitch=float(entity['rotation']['pitch']),
                        yaw=float(entity['rotation']['yaw']),
                        roll=float(entity['rotation']['roll']),
                    )
                    spawn_kind = str(entity.get('spawn_kind') or 'vehicle')
                    entity_id = str(entity.get('id') or '')
                    actor = None
                    if spawn_kind == 'pedestrian':
                        actor = _autoscenario_spawn_pedestrian(location, rotation)
                    elif spawn_kind == 'static':
                        actor = _autoscenario_spawn_static_prop(
                            entity.get('blueprint_name'),
                            location,
                            rotation,
                        )
                    else:
                        placement_mode = str(entity.get('placement_mode') or 'project_to_lane')
                        role_name = 'hero' if entity_id == 'ego' else entity.get('role_name')
                        if placement_mode in {'direct', 'preserve_xy'}:
                            actor = _autoscenario_spawn_vehicle_direct(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('color'),
                                role_name=role_name,
                            )
                        elif placement_mode == 'project_to_parking_lane':
                            actor = _autoscenario_spawn_vehicle_parking_lane(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('projected_lane'),
                                entity.get('color'),
                                role_name=role_name,
                            )
                        elif placement_mode == 'project_to_junction_lane':
                            actor = _autoscenario_spawn_vehicle_junction_lane(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('projected_lane'),
                                entity.get('color'),
                                role_name=role_name,
                                junction_distance_m=entity.get('junction_distance_m'),
                            )
                        else:
                            actor = _autoscenario_spawn_vehicle(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('color'),
                                role_name=role_name,
                            )
                    if actor is not None and entity_id:
                        actor_by_id[entity_id] = actor
                return actor_by_id


            def _autoscenario_actor_distance(a, b):
                try:
                    loc_a = a.get_transform().location
                    loc_b = b.get_transform().location
                    return math.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)
                except Exception:
                    return float('inf')


            def _autoscenario_vehicle_speed(actor):
                try:
                    velocity = actor.get_velocity()
                    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                except Exception:
                    return 0.0


            def _autoscenario_hold_vehicle_stationary(actor):
                if actor is None:
                    return
                try:
                    actor.set_autopilot(False)
                except Exception:
                    pass
                try:
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                try:
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                try:
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                    )
                except Exception:
                    pass


            def _autoscenario_clear_vehicle_hold(actor):
                if actor is None:
                    return
                try:
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
                    )
                except Exception:
                    pass


            def _autoscenario_apply_target_velocity(actor, target_speed_mps):
                try:
                    transform = actor.get_transform()
                    yaw_radians = math.radians(float(transform.rotation.yaw))
                    speed = max(0.0, float(target_speed_mps))
                    actor.set_target_velocity(
                        carla.Vector3D(
                            math.cos(yaw_radians) * speed,
                            math.sin(yaw_radians) * speed,
                            0.0,
                        )
                    )
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
                    )
                    return True
                except Exception:
                    return False


            def _autoscenario_apply_speed_control(actor, target_speed_mps, max_throttle=0.55):
                if os.environ.get('AUTOSCENARIO_USE_VELOCITY_CONTROL', '0') != '0':
                    if _autoscenario_apply_target_velocity(actor, target_speed_mps):
                        return
                current_speed = _autoscenario_vehicle_speed(actor)
                error = float(target_speed_mps) - current_speed
                if error > 0.4:
                    throttle = min(float(max_throttle), error * 0.035)
                    actor.apply_control(carla.VehicleControl(throttle=throttle, brake=0.0))
                elif error < -0.4:
                    brake = min(0.7, abs(error) * 0.12)
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=brake))
                else:
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


            def _autoscenario_tick_world():
                try:
                    world.tick()
                except Exception:
                    world.wait_for_tick()


            def _autoscenario_sample_forward_route(actor, max_distance_m=80.0, step_m=3.0):
                route = []
                try:
                    world_map = world.get_map()
                    waypoint = world_map.get_waypoint(
                        actor.get_transform().location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                except Exception:
                    waypoint = None
                travelled = 0.0
                while waypoint is not None and travelled <= max_distance_m:
                    route.append(waypoint.transform)
                    try:
                        next_waypoints = waypoint.next(step_m)
                    except Exception:
                        next_waypoints = []
                    waypoint = next_waypoints[0] if next_waypoints else None
                    travelled += step_m
                if not route:
                    try:
                        route.append(actor.get_transform())
                    except Exception:
                        pass
                return route


            class EgoController:
                def __init__(self, actor, risk_sample):
                    self.actor = actor
                    self.risk_sample = risk_sample or {}
                    self.params = self.risk_sample.get('sampled_parameters', {})
                    self.route = _autoscenario_sample_forward_route(actor)
                    self.route_index = 0
                    self.target_speed_mps = float(self.params.get('ego_target_speed_mps', 8.0))
                    self.reaction_delay_s = float(self.params.get('ego_reaction_delay_s', 1.5))
                    self.startup_ramp_s = 0.8

                def _startup_target_speed(self, elapsed_s, target_speed_mps):
                    ramp_ratio = min(1.0, max(0.0, float(elapsed_s) / max(self.startup_ramp_s, 0.01)))
                    return float(target_speed_mps) * ramp_ratio

                def tick(self, elapsed_s):
                    if self.actor is None:
                        return
                    # Accident-reproduction mode keeps ego on its nominal route and delays strong braking.
                    if elapsed_s < self.reaction_delay_s:
                        _autoscenario_apply_speed_control(
                            self.actor,
                            self._startup_target_speed(elapsed_s, self.target_speed_mps),
                        )
                    else:
                        _autoscenario_apply_speed_control(
                            self.actor,
                            self._startup_target_speed(elapsed_s, self.target_speed_mps * 0.92),
                        )


            class VLAControllerAdapter(EgoController):
                def tick(self, elapsed_s):
                    # Integration point: replace this method with a VLA action provider.
                    return super().tick(elapsed_s)


            class NpcRiskController:
                def __init__(self, ego_actor, risk_actor, risk_sample):
                    self.ego_actor = ego_actor
                    self.risk_actor = risk_actor
                    self.risk_sample = risk_sample or {}
                    self.template_id = self.risk_sample.get('template_id')
                    self.params = self.risk_sample.get('sampled_parameters', {})
                    self.triggered = False
                    self.trigger_time_s = None
                    self.pre_brake_hold_s = 0.75
                    self.brake_ramp_duration_s = 1.25
                    self.startup_ramp_s = 0.8

                def _startup_target_speed(self, elapsed_s, target_speed_mps):
                    ramp_ratio = min(1.0, max(0.0, float(elapsed_s) / max(self.startup_ramp_s, 0.01)))
                    return float(target_speed_mps) * ramp_ratio

                def _maybe_trigger(self, elapsed_s):
                    if self.triggered or self.ego_actor is None or self.risk_actor is None:
                        return
                    trigger_distance = float(self.params.get('npc_trigger_distance_m', 12.0))
                    if _autoscenario_actor_distance(self.ego_actor, self.risk_actor) <= trigger_distance:
                        self.triggered = True
                        self.trigger_time_s = elapsed_s

                def tick(self, elapsed_s):
                    if self.risk_actor is None:
                        return
                    self._maybe_trigger(elapsed_s)
                    if not self.triggered:
                        target_speed = float(self.params.get('npc_target_speed_mps', 2.0))
                        _autoscenario_apply_speed_control(
                            self.risk_actor,
                            self._startup_target_speed(elapsed_s, target_speed),
                            max_throttle=0.35,
                        )
                        return
                    if self.template_id in {'lead_vehicle_hard_brake'}:
                        brake = min(1.0, max(0.0, float(self.params.get('npc_brake_intensity', 0.9))))
                        target_speed = float(self.params.get('npc_target_speed_mps', 2.0))
                        elapsed_since_trigger = max(
                            0.0,
                            elapsed_s - (self.trigger_time_s or elapsed_s),
                        )
                        if elapsed_since_trigger < self.pre_brake_hold_s:
                            _autoscenario_apply_speed_control(
                                self.risk_actor,
                                target_speed * 0.9,
                                max_throttle=0.2,
                            )
                            return
                        ramp_elapsed = elapsed_since_trigger - self.pre_brake_hold_s
                        if ramp_elapsed < self.brake_ramp_duration_s:
                            ramp_ratio = ramp_elapsed / max(self.brake_ramp_duration_s, 0.01)
                            staged_brake = brake * (0.25 + 0.55 * ramp_ratio)
                            self.risk_actor.apply_control(
                                carla.VehicleControl(throttle=0.0, brake=staged_brake)
                            )
                            return
                        self.risk_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=brake))
                        return
                    steer = float(self.params.get('npc_steer_intensity', 0.25))
                    target_speed = float(self.params.get('npc_target_speed_mps', 6.0))
                    if self.template_id in {
                        'adjacent_vehicle_cut_in',
                        'roadside_vehicle_pull_out',
                        'lateral_encroachment',
                        'oncoming_lane_invasion',
                        'opposing_turn_conflict',
                        'left_turn_across_opposite',
                    }:
                        self.risk_actor.apply_control(
                            carla.VehicleControl(throttle=0.35, steer=steer, brake=0.0)
                        )
                    elif self.template_id in {
                        'straight_crossing_path',
                        'cross_traffic_red_light',
                        'pedestrian_nearside_crossing',
                        'pedestrian_farside_crossing',
                        'bicyclist_crossing',
                        'motorcyclist_or_scooter_crossing',
                    }:
                        self.risk_actor.apply_control(
                            carla.VehicleControl(throttle=0.3, steer=steer * 0.6, brake=0.0)
                        )
                    else:
                        _autoscenario_apply_speed_control(self.risk_actor, target_speed)


            class RiskMetricsRecorder:
                def __init__(self, ego_actor, actor_by_id, output_path):
                    self.ego_actor = ego_actor
                    self.actor_by_id = actor_by_id
                    self.output_path = output_path
                    self.min_distance_m = float('inf')
                    self.min_ttc_s = float('inf')
                    self.collisions = []
                    self.collision_sensor = None
                    self._attach_collision_sensor()

                def _attach_collision_sensor(self):
                    if self.ego_actor is None:
                        return
                    try:
                        blueprint = blueprint_library.find('sensor.other.collision')
                        self.collision_sensor = world.spawn_actor(
                            blueprint,
                            carla.Transform(),
                            attach_to=self.ego_actor,
                        )
                        self.collision_sensor.listen(
                            lambda event: self.collisions.append({'frame': int(event.frame)})
                        )
                    except Exception:
                        self.collision_sensor = None

                def tick(self):
                    if self.ego_actor is None:
                        return
                    ego_speed = _autoscenario_vehicle_speed(self.ego_actor)
                    for actor_id, actor in list(self.actor_by_id.items()):
                        if actor_id == 'ego' or actor is None:
                            continue
                        distance = _autoscenario_actor_distance(self.ego_actor, actor)
                        self.min_distance_m = min(self.min_distance_m, distance)
                        other_speed = _autoscenario_vehicle_speed(actor)
                        closing_speed = max(0.0, ego_speed - other_speed)
                        if closing_speed > 0.1 and distance < float('inf'):
                            self.min_ttc_s = min(self.min_ttc_s, distance / closing_speed)

                def close(self):
                    if self.collision_sensor is not None:
                        try:
                            self.collision_sensor.stop()
                        except Exception:
                            pass
                        try:
                            self.collision_sensor.destroy()
                        except Exception:
                            pass
                    data = {
                        'collision': bool(self.collisions),
                        'collision_events': self.collisions,
                        'min_distance_m': None if self.min_distance_m == float('inf') else self.min_distance_m,
                        'min_ttc_s': None if self.min_ttc_s == float('inf') else self.min_ttc_s,
                        'actor_ids': sorted(self.actor_by_id.keys()),
                    }
                    os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
                    with open(self.output_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, indent=2, sort_keys=True)


            _autoscenario_apply_weather(carla.WeatherParameters.ClearNoon)
            if os.environ.get('AUTOSCENARIO_CLEAR_EXISTING') == '1':
                _autoscenario_clear_existing_dynamic_actors()
            _AUTOSCENARIO_EXISTING_VEHICLES = _autoscenario_collect_existing_vehicles()
            spawn_payload = _autoscenario_load_spawn_payload()
            risk_sample = _autoscenario_load_risk_sample()
            actor_by_id = _autoscenario_spawn_payload_actors(spawn_payload)
            _autoscenario_focus_spectator()

            ego_actor = actor_by_id.get('ego')
            risk_actor = actor_by_id.get(str(risk_sample.get('risk_actor_id')))
            ego_mode = str((risk_sample.get('ego') or {}).get('controller_mode') or 'scripted')
            ego_controller = (
                VLAControllerAdapter(ego_actor, risk_sample)
                if ego_mode == 'vla_adapter'
                else EgoController(ego_actor, risk_sample)
            )
            npc_controller = NpcRiskController(ego_actor, risk_actor, risk_sample)
            metrics = RiskMetricsRecorder(ego_actor, actor_by_id, _AUTOSCENARIO_RISK_METRICS)

            duration_s = float(os.environ.get('AUTOSCENARIO_RISK_DURATION', '12.0'))
            tick_dt = float(os.environ.get('AUTOSCENARIO_RISK_TICK_DT', '0.05'))
            max_ticks = max(1, int(duration_s / max(tick_dt, 0.01)))
            # Drive the scenario in synchronous fixed-step mode so each iteration
            # advances physics by a deterministic tick_dt. In async mode world.tick()
            # is a no-op/raises and the loop blows through max_ticks before the ego
            # accelerates, making low/medium/high speed runs visually identical.
            _autoscenario_original_settings = world.get_settings()
            _autoscenario_sync_settings = world.get_settings()
            _autoscenario_sync_settings.synchronous_mode = True
            _autoscenario_sync_settings.fixed_delta_seconds = tick_dt
            world.apply_settings(_autoscenario_sync_settings)

            # Actors the VLM reasoned about but that are neither the ego nor the
            # scripted risk NPC, plus any pure background vehicle, are handed to the
            # Traffic Manager so the scene is not frozen around the ego. The Traffic
            # Manager must run in sync mode too, otherwise autopilot vehicles stall
            # or behave erratically under fixed-step ticking.
            _autoscenario_risk_actor_id = str(risk_sample.get('risk_actor_id') or '')
            _autoscenario_scripted_ids = {'ego', _autoscenario_risk_actor_id}
            _autoscenario_traffic_manager = None
            try:
                _autoscenario_traffic_manager = client.get_trafficmanager()
                _autoscenario_traffic_manager.set_synchronous_mode(True)
            except Exception:
                _autoscenario_traffic_manager = None
            _autoscenario_background_actors = []

            # Allow freshly spawned actors to settle onto the road surface before
            # starting the scenario. Without this the physics engine is still
            # resolving the initial drop (actors spawn at z+0.3) while the speed
            # controller is already applying throttle, causing stuttering movement
            # in the first ~1 s.
            _autoscenario_settle_ticks = int(os.environ.get('AUTOSCENARIO_SETTLE_TICKS', '20'))
            for _ in range(_autoscenario_settle_ticks):
                for _settle_actor in actor_by_id.values():
                    try:
                        if _settle_actor is not None and 'vehicle' in _settle_actor.type_id:
                            _autoscenario_hold_vehicle_stationary(_settle_actor)
                    except Exception:
                        pass
                world.tick()

            for _settle_actor in actor_by_id.values():
                try:
                    if _settle_actor is not None and 'vehicle' in _settle_actor.type_id:
                        _autoscenario_clear_vehicle_hold(_settle_actor)
                except Exception:
                    pass

            _autoscenario_release_ticks = int(os.environ.get('AUTOSCENARIO_RELEASE_TICKS', '3'))
            for _ in range(_autoscenario_release_ticks):
                for _release_actor in actor_by_id.values():
                    try:
                        if _release_actor is not None and 'vehicle' in _release_actor.type_id:
                            _autoscenario_apply_target_velocity(_release_actor, 0.0)
                    except Exception:
                        pass
                world.tick()

            for _bg_id, _bg_actor in actor_by_id.items():
                if _bg_id in _autoscenario_scripted_ids or _bg_actor is None:
                    continue
                try:
                    if 'vehicle' not in _bg_actor.type_id:
                        continue
                    if _autoscenario_traffic_manager is not None:
                        _bg_actor.set_autopilot(True, _autoscenario_traffic_manager.get_port())
                    else:
                        _bg_actor.set_autopilot(True)
                    _autoscenario_background_actors.append(_bg_actor)
                except Exception:
                    pass

            _autoscenario_spectator = None
            try:
                _autoscenario_spectator = world.get_spectator()
            except Exception:
                pass

            def _autoscenario_update_spectator(ego, spectator):
                if spectator is None or ego is None:
                    return
                try:
                    ego_tf = ego.get_transform()
                    import math as _math
                    yaw_rad = _math.radians(ego_tf.rotation.yaw)
                    # Position spectator 12 m behind and 6 m above ego, looking forward
                    offset_x = -12.0 * _math.cos(yaw_rad)
                    offset_y = -12.0 * _math.sin(yaw_rad)
                    spec_loc = carla.Location(
                        ego_tf.location.x + offset_x,
                        ego_tf.location.y + offset_y,
                        ego_tf.location.z + 6.0,
                    )
                    spec_rot = carla.Rotation(pitch=-20.0, yaw=ego_tf.rotation.yaw, roll=0.0)
                    spectator.set_transform(carla.Transform(spec_loc, spec_rot))
                except Exception:
                    pass

            try:
                for tick_index in range(max_ticks):
                    elapsed_s = tick_index * tick_dt
                    ego_controller.tick(elapsed_s)
                    npc_controller.tick(elapsed_s)
                    world.tick()
                    time.sleep(0.05)
                    metrics.tick()
                    _autoscenario_update_spectator(ego_actor, _autoscenario_spectator)
            finally:
                metrics.close()
                for _bg_actor in _autoscenario_background_actors:
                    try:
                        _bg_actor.set_autopilot(False)
                    except Exception:
                        pass
                if _autoscenario_traffic_manager is not None:
                    try:
                        _autoscenario_traffic_manager.set_synchronous_mode(False)
                    except Exception:
                        pass
                try:
                    world.apply_settings(_autoscenario_original_settings)
                except Exception:
                    pass
            """
        )

    @staticmethod
    def _build_dsl_risk_loop() -> str:
        return textwrap.dedent(
            """
            def _autoscenario_load_spawn_payload():
                payload_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_SPAWN_PAYLOAD)
                with open(payload_path, 'r', encoding='utf-8') as file:
                    return json.load(file)


            def _autoscenario_load_risk_dsl():
                dsl_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_RISK_DSL)
                with open(dsl_path, 'r', encoding='utf-8') as file:
                    return json.load(file)


            def _autoscenario_clear_existing_dynamic_actors():
                for actor_filter in ('controller.ai.walker', 'walker.*', 'vehicle.*'):
                    try:
                        existing_actors = list(world.get_actors().filter(actor_filter))
                    except Exception:
                        existing_actors = []
                    for actor in existing_actors:
                        try:
                            if actor_filter == 'controller.ai.walker':
                                actor.stop()
                        except Exception:
                            pass
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                try:
                    world.wait_for_tick()
                except Exception:
                    time.sleep(0.5)


            def _autoscenario_spawn_payload_actors(spawn_payload):
                actor_by_id = {}
                for entity in spawn_payload.get('entities', []):
                    location = carla.Location(
                        x=float(entity['location']['x']),
                        y=float(entity['location']['y']),
                        z=float(entity['location']['z']),
                    )
                    rotation = carla.Rotation(
                        pitch=float(entity['rotation']['pitch']),
                        yaw=float(entity['rotation']['yaw']),
                        roll=float(entity['rotation']['roll']),
                    )
                    spawn_kind = str(entity.get('spawn_kind') or 'vehicle')
                    entity_id = str(entity.get('id') or '')
                    actor = None
                    if spawn_kind == 'pedestrian':
                        actor = _autoscenario_spawn_pedestrian(location, rotation)
                    elif spawn_kind == 'static':
                        actor = _autoscenario_spawn_static_prop(
                            entity.get('blueprint_name'),
                            location,
                            rotation,
                        )
                    else:
                        placement_mode = str(entity.get('placement_mode') or 'project_to_lane')
                        role_name = 'hero' if entity_id == 'ego' else entity.get('role_name')
                        if placement_mode in {'direct', 'preserve_xy'}:
                            actor = _autoscenario_spawn_vehicle_direct(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('color'),
                                role_name=role_name,
                            )
                        elif placement_mode == 'project_to_parking_lane':
                            actor = _autoscenario_spawn_vehicle_parking_lane(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('projected_lane'),
                                entity.get('color'),
                                role_name=role_name,
                            )
                        elif placement_mode == 'project_to_junction_lane':
                            actor = _autoscenario_spawn_vehicle_junction_lane(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('projected_lane'),
                                entity.get('color'),
                                role_name=role_name,
                                junction_distance_m=entity.get('junction_distance_m'),
                            )
                        else:
                            actor = _autoscenario_spawn_vehicle(
                                entity.get('blueprint_name'),
                                location,
                                rotation,
                                entity.get('color'),
                                role_name=role_name,
                            )
                    if actor is not None and entity_id:
                        actor_by_id[entity_id] = actor
                return actor_by_id


            def _autoscenario_actor_distance(a, b):
                try:
                    loc_a = a.get_transform().location
                    loc_b = b.get_transform().location
                    return math.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)
                except Exception:
                    return float('inf')


            def _autoscenario_vehicle_speed(actor):
                try:
                    velocity = actor.get_velocity()
                    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                except Exception:
                    return 0.0


            def _autoscenario_hold_vehicle_stationary(actor):
                if actor is None:
                    return
                try:
                    actor.set_autopilot(False)
                except Exception:
                    pass
                try:
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                try:
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                try:
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                    )
                except Exception:
                    pass


            def _autoscenario_clear_vehicle_hold(actor):
                if actor is None:
                    return
                try:
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
                    )
                except Exception:
                    pass


            def _autoscenario_apply_target_velocity(actor, target_speed_mps):
                try:
                    transform = actor.get_transform()
                    yaw_radians = math.radians(float(transform.rotation.yaw))
                    speed = max(0.0, float(target_speed_mps))
                    actor.set_target_velocity(
                        carla.Vector3D(
                            math.cos(yaw_radians) * speed,
                            math.sin(yaw_radians) * speed,
                            0.0,
                        )
                    )
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    actor.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
                    )
                    return True
                except Exception:
                    return False


            def _autoscenario_apply_speed_control(actor, target_speed_mps, max_throttle=0.55):
                if os.environ.get('AUTOSCENARIO_USE_VELOCITY_CONTROL', '0') != '0':
                    if _autoscenario_apply_target_velocity(actor, target_speed_mps):
                        return
                current_speed = _autoscenario_vehicle_speed(actor)
                error = float(target_speed_mps) - current_speed
                if error > 0.4:
                    throttle = min(float(max_throttle), error * 0.035)
                    actor.apply_control(carla.VehicleControl(throttle=throttle, brake=0.0))
                elif error < -0.4:
                    brake = min(0.7, abs(error) * 0.12)
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=brake))
                else:
                    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


            def _autoscenario_sample_forward_route(actor, max_distance_m=80.0, step_m=3.0):
                route = []
                try:
                    world_map = world.get_map()
                    waypoint = world_map.get_waypoint(
                        actor.get_transform().location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                except Exception:
                    waypoint = None
                travelled = 0.0
                while waypoint is not None and travelled <= max_distance_m:
                    route.append(waypoint.transform)
                    try:
                        next_waypoints = waypoint.next(step_m)
                    except Exception:
                        next_waypoints = []
                    waypoint = next_waypoints[0] if next_waypoints else None
                    travelled += step_m
                if not route:
                    try:
                        route.append(actor.get_transform())
                    except Exception:
                        pass
                return route


            def _autoscenario_normalize_angle(angle_degrees):
                value = float(angle_degrees)
                while value <= -180.0:
                    value += 360.0
                while value > 180.0:
                    value -= 360.0
                return value


            def _autoscenario_lane_follow_control(
                actor,
                target_speed_mps,
                route,
                route_index=0,
                lookahead_m=8.0,
                max_throttle=0.55,
            ):
                if actor is None:
                    return route_index
                route = route or []
                if not route:
                    _autoscenario_apply_speed_control(actor, target_speed_mps, max_throttle=max_throttle)
                    return route_index
                try:
                    transform = actor.get_transform()
                    location = transform.location
                    yaw = float(transform.rotation.yaw)
                except Exception:
                    _autoscenario_apply_speed_control(actor, target_speed_mps, max_throttle=max_throttle)
                    return route_index

                route_index = max(0, min(int(route_index or 0), len(route) - 1))
                best_index = route_index
                for index in range(route_index, len(route)):
                    target_location = route[index].location
                    distance = math.sqrt(
                        (target_location.x - location.x) ** 2
                        + (target_location.y - location.y) ** 2
                    )
                    best_index = index
                    if distance >= lookahead_m:
                        break

                target_location = route[best_index].location
                target_yaw = math.degrees(
                    math.atan2(target_location.y - location.y, target_location.x - location.x)
                )
                yaw_error = _autoscenario_normalize_angle(target_yaw - yaw)
                steer = max(-0.55, min(0.55, yaw_error / 45.0))

                current_speed = _autoscenario_vehicle_speed(actor)
                error = float(target_speed_mps) - current_speed
                if error > 0.4:
                    throttle = min(float(max_throttle), error * 0.035)
                    brake = 0.0
                elif error < -0.4:
                    throttle = 0.0
                    brake = min(0.7, abs(error) * 0.12)
                else:
                    throttle = 0.0
                    brake = 0.0
                actor.apply_control(
                    carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                )
                return best_index


            def _autoscenario_dsl_initial_speed(actor_id, risk_dsl, cruise_speed_mps):
                # Speed an actor should already have at t=0 (its flying-start speed).
                ego_spec = risk_dsl.get('ego') or {}
                if actor_id == 'ego':
                    return float(ego_spec.get('target_speed_mps', 10.0))
                for event in risk_dsl.get('events') or []:
                    if str(event.get('actor_id')) != actor_id:
                        continue
                    trigger = event.get('trigger') or {}
                    if trigger.get('type') != 'immediate':
                        continue
                    action = event.get('action') or {}
                    action_type = action.get('type')
                    if action_type in ('set_speed', 'accelerate'):
                        return float(action.get('speed_mps', cruise_speed_mps))
                    if action_type == 'stop':
                        return 0.0
                    # immediate brake/steer/cross: actor is still rolling at cruise.
                    return float(cruise_speed_mps)
                # No immediate event -> actor cruises until its (deferred) trigger.
                return float(cruise_speed_mps)


            class EgoController:
                def __init__(self, actor, risk_dsl):
                    self.actor = actor
                    self.risk_dsl = risk_dsl or {}
                    ego_spec = self.risk_dsl.get('ego') or {}
                    self.route = _autoscenario_sample_forward_route(actor)
                    self.route_index = 0
                    self.target_speed_mps = float(ego_spec.get('target_speed_mps', 10.0))

                def tick(self, elapsed_s):
                    if self.actor is None:
                        return
                    # The ego is given a flying start (initial velocity) before the
                    # loop. It still follows the CARLA lane route instead of driving
                    # as a pure straight-line velocity controller.
                    self.route_index = _autoscenario_lane_follow_control(
                        self.actor,
                        self.target_speed_mps,
                        self.route,
                        self.route_index,
                    )


            class VLAControllerAdapter(EgoController):
                def tick(self, elapsed_s):
                    # Integration point: replace this method with a VLA action provider.
                    return super().tick(elapsed_s)


            class DslEventController:
                # Generic, template-free interpreter for risk-dsl-v1 events. Each event
                # is {actor_id, trigger, action}; once a trigger fires it stays latched
                # and the action is applied every tick. Before any of an actor's events
                # fire the actor cruises forward just under the ego speed so the scene
                # is not frozen and distance triggers can still close.
                def __init__(self, ego_actor, actor_by_id, risk_dsl):
                    self.ego_actor = ego_actor
                    self.actor_by_id = actor_by_id or {}
                    self.risk_dsl = risk_dsl or {}
                    self.events = list(self.risk_dsl.get('events') or [])
                    self.triggered = [False] * len(self.events)
                    self.trigger_times = [None] * len(self.events)
                    ego_spec = self.risk_dsl.get('ego') or {}
                    self.cruise_speed_mps = float(ego_spec.get('target_speed_mps', 10.0)) * 0.9
                    self.routes = {}
                    self.route_indices = {}
                    for event in self.events:
                        actor_id = str(event.get('actor_id'))
                        actor = self.actor_by_id.get(actor_id)
                        if actor is None or actor_id in self.routes:
                            continue
                        self.routes[actor_id] = _autoscenario_sample_forward_route(actor)
                        self.route_indices[actor_id] = 0

                def _trigger_fires(self, event, elapsed_s):
                    trigger = event.get('trigger') or {}
                    trigger_type = trigger.get('type')
                    if trigger_type == 'immediate':
                        return True
                    if trigger_type == 'time_elapsed_above':
                        return elapsed_s >= float(trigger.get('value_s', 0.0))
                    if trigger_type == 'distance_to_ego_below':
                        actor = self.actor_by_id.get(str(event.get('actor_id')))
                        if self.ego_actor is None or actor is None:
                            return False
                        distance = _autoscenario_actor_distance(self.ego_actor, actor)
                        return distance <= float(trigger.get('value_m', 12.0))
                    return False

                def _follow_actor_lane(self, actor_id, actor, target_speed_mps, max_throttle=0.45):
                    route = self.routes.get(actor_id)
                    route_index = self.route_indices.get(actor_id, 0)
                    self.route_indices[actor_id] = _autoscenario_lane_follow_control(
                        actor,
                        target_speed_mps,
                        route,
                        route_index,
                        max_throttle=max_throttle,
                    )

                def _apply_action(self, actor_id, actor, action, elapsed_since_trigger):
                    action_type = action.get('type')
                    if action_type == 'brake':
                        intensity = min(1.0, max(0.0, float(action.get('intensity', 0.9))))
                        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=intensity))
                    elif action_type in ('set_speed', 'accelerate'):
                        self._follow_actor_lane(
                            actor_id,
                            actor,
                            float(action.get('speed_mps', 6.0)),
                        )
                    elif action_type in ('steer', 'cross'):
                        duration_s = float(action.get('duration_s', 1.2))
                        if elapsed_since_trigger > duration_s:
                            self._follow_actor_lane(
                                actor_id,
                                actor,
                                self.cruise_speed_mps,
                                max_throttle=0.35,
                            )
                            return
                        steer = max(-1.0, min(1.0, float(action.get('steer', 0.25))))
                        throttle = min(1.0, max(0.0, float(action.get('throttle', 0.35))))
                        actor.apply_control(
                            carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)
                        )
                    elif action_type == 'stop':
                        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

                def tick(self, elapsed_s):
                    actors_with_active_event = set()
                    for index, event in enumerate(self.events):
                        actor_id = str(event.get('actor_id'))
                        actor = self.actor_by_id.get(actor_id)
                        if actor is None:
                            continue
                        if not self.triggered[index]:
                            if self._trigger_fires(event, elapsed_s):
                                self.triggered[index] = True
                                self.trigger_times[index] = elapsed_s
                            else:
                                continue
                        trigger_time = self.trigger_times[index]
                        elapsed_since_trigger = (
                            0.0 if trigger_time is None else max(0.0, elapsed_s - trigger_time)
                        )
                        self._apply_action(
                            actor_id,
                            actor,
                            event.get('action') or {},
                            elapsed_since_trigger,
                        )
                        actors_with_active_event.add(actor_id)
                    # Pre-trigger cruise for event actors that have not fired yet.
                    for event in self.events:
                        actor_id = str(event.get('actor_id'))
                        if actor_id in actors_with_active_event:
                            continue
                        actor = self.actor_by_id.get(actor_id)
                        if actor is None:
                            continue
                        # Pre-trigger actors keep their flying-start cruise speed
                        # while following the lane rather than drifting straight.
                        self._follow_actor_lane(
                            actor_id,
                            actor,
                            self.cruise_speed_mps,
                            max_throttle=0.35,
                        )


            class RiskMetricsRecorder:
                def __init__(self, ego_actor, actor_by_id, output_path):
                    self.ego_actor = ego_actor
                    self.actor_by_id = actor_by_id
                    self.output_path = output_path
                    self.min_distance_m = float('inf')
                    self.min_ttc_s = float('inf')
                    self.collisions = []
                    self.collision_sensor = None
                    self._attach_collision_sensor()

                def _attach_collision_sensor(self):
                    if self.ego_actor is None:
                        return
                    try:
                        blueprint = blueprint_library.find('sensor.other.collision')
                        self.collision_sensor = world.spawn_actor(
                            blueprint,
                            carla.Transform(),
                            attach_to=self.ego_actor,
                        )
                        self.collision_sensor.listen(
                            lambda event: self.collisions.append({'frame': int(event.frame)})
                        )
                    except Exception:
                        self.collision_sensor = None

                def tick(self):
                    if self.ego_actor is None:
                        return
                    ego_speed = _autoscenario_vehicle_speed(self.ego_actor)
                    for actor_id, actor in list(self.actor_by_id.items()):
                        if actor_id == 'ego' or actor is None:
                            continue
                        distance = _autoscenario_actor_distance(self.ego_actor, actor)
                        self.min_distance_m = min(self.min_distance_m, distance)
                        other_speed = _autoscenario_vehicle_speed(actor)
                        closing_speed = max(0.0, ego_speed - other_speed)
                        if closing_speed > 0.1 and distance < float('inf'):
                            self.min_ttc_s = min(self.min_ttc_s, distance / closing_speed)

                def close(self):
                    if self.collision_sensor is not None:
                        try:
                            self.collision_sensor.stop()
                        except Exception:
                            pass
                        try:
                            self.collision_sensor.destroy()
                        except Exception:
                            pass
                    data = {
                        'collision': bool(self.collisions),
                        'collision_events': self.collisions,
                        'min_distance_m': None if self.min_distance_m == float('inf') else self.min_distance_m,
                        'min_ttc_s': None if self.min_ttc_s == float('inf') else self.min_ttc_s,
                        'actor_ids': sorted(self.actor_by_id.keys()),
                    }
                    os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
                    with open(self.output_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, indent=2, sort_keys=True)


            _autoscenario_apply_weather(carla.WeatherParameters.ClearNoon)
            if os.environ.get('AUTOSCENARIO_CLEAR_EXISTING') == '1':
                _autoscenario_clear_existing_dynamic_actors()
            _AUTOSCENARIO_EXISTING_VEHICLES = _autoscenario_collect_existing_vehicles()
            spawn_payload = _autoscenario_load_spawn_payload()
            risk_dsl = _autoscenario_load_risk_dsl()
            actor_by_id = _autoscenario_spawn_payload_actors(spawn_payload)
            _autoscenario_focus_spectator()

            def _autoscenario_sampled_flying_speed(actor_id, cruise_speed_mps, low=0.85, high=1.0):
                # Deterministic per-actor speed in [low, high] x cruise so background
                # traffic is not a lock-step formation, while runs of the same scene stay
                # reproducible. A same-lane lead car therefore rolls slightly below the
                # ego instead of sitting still or matching it exactly.
                import hashlib
                digest = hashlib.md5(str(actor_id).encode('utf-8')).hexdigest()
                frac = (int(digest[:8], 16) % 10000) / 10000.0
                return float(cruise_speed_mps) * (low + (high - low) * frac)

            ego_actor = actor_by_id.get('ego')
            ego_mode = str((risk_dsl.get('ego') or {}).get('controller_mode') or 'scripted')
            ego_controller = (
                VLAControllerAdapter(ego_actor, risk_dsl)
                if ego_mode == 'vla_adapter'
                else EgoController(ego_actor, risk_dsl)
            )
            dsl_controller = DslEventController(ego_actor, actor_by_id, risk_dsl)
            metrics = RiskMetricsRecorder(ego_actor, actor_by_id, _AUTOSCENARIO_RISK_METRICS)

            duration_s = float(
                os.environ.get('AUTOSCENARIO_RISK_DURATION', str(risk_dsl.get('duration_s', 12.0)))
            )
            tick_dt = float(os.environ.get('AUTOSCENARIO_RISK_TICK_DT', '0.05'))
            max_ticks = max(1, int(duration_s / max(tick_dt, 0.01)))
            # Drive the scenario in synchronous fixed-step mode so each iteration
            # advances physics by a deterministic tick_dt.
            _autoscenario_original_settings = world.get_settings()
            _autoscenario_sync_settings = world.get_settings()
            _autoscenario_sync_settings.synchronous_mode = True
            _autoscenario_sync_settings.fixed_delta_seconds = tick_dt
            world.apply_settings(_autoscenario_sync_settings)

            # Hand any actor that is neither the ego nor a scripted DSL-event actor to
            # the Traffic Manager so the scene is not frozen around the ego.
            _autoscenario_scripted_ids = {'ego'}
            for _event in (risk_dsl.get('events') or []):
                _autoscenario_scripted_ids.add(str(_event.get('actor_id')))
            _autoscenario_traffic_manager = None
            try:
                _autoscenario_traffic_manager = client.get_trafficmanager()
                _autoscenario_traffic_manager.set_synchronous_mode(True)
            except Exception:
                _autoscenario_traffic_manager = None
            _autoscenario_background_actors = []

            # Allow freshly spawned actors to settle onto the road surface before
            # starting the scenario.
            _autoscenario_settle_ticks = int(os.environ.get('AUTOSCENARIO_SETTLE_TICKS', '20'))
            for _ in range(_autoscenario_settle_ticks):
                for _settle_actor in actor_by_id.values():
                    try:
                        if _settle_actor is not None and 'vehicle' in _settle_actor.type_id:
                            _autoscenario_hold_vehicle_stationary(_settle_actor)
                    except Exception:
                        pass
                world.tick()

            for _settle_actor in actor_by_id.values():
                try:
                    if _settle_actor is not None and 'vehicle' in _settle_actor.type_id:
                        _autoscenario_clear_vehicle_hold(_settle_actor)
                except Exception:
                    pass

            _autoscenario_release_ticks = int(os.environ.get('AUTOSCENARIO_RELEASE_TICKS', '3'))
            for _ in range(_autoscenario_release_ticks):
                for _release_actor in actor_by_id.values():
                    try:
                        if _release_actor is not None and 'vehicle' in _release_actor.type_id:
                            _autoscenario_apply_target_velocity(_release_actor, 0.0)
                    except Exception:
                        pass
                world.tick()

            # Flying start: give the ego and each scripted DSL-event actor their
            # predicted speed *at their captured positions*, so the accident develops
            # from the geometry in the original image instead of during a slow
            # 0-to-speed ramp. Velocity direction follows each actor's facing -- the
            # same basis the throttle controllers use afterwards to hold speed.
            _autoscenario_cruise_speed_mps = float(
                (risk_dsl.get('ego') or {}).get('target_speed_mps', 10.0)
            ) * 0.9
            if os.environ.get('AUTOSCENARIO_DISABLE_FLYING_START', '0') == '0':
                for _init_id in _autoscenario_scripted_ids:
                    _init_actor = actor_by_id.get(_init_id)
                    if _init_actor is None:
                        continue
                    try:
                        if 'vehicle' not in _init_actor.type_id:
                            continue
                    except Exception:
                        continue
                    _autoscenario_apply_target_velocity(
                        _init_actor,
                        _autoscenario_dsl_initial_speed(
                            _init_id,
                            risk_dsl,
                            _autoscenario_cruise_speed_mps,
                        ),
                    )

            for _bg_id, _bg_actor in actor_by_id.items():
                if _bg_id in _autoscenario_scripted_ids or _bg_actor is None:
                    continue
                try:
                    if 'vehicle' not in _bg_actor.type_id:
                        continue
                    # Background vehicles get a sampled flying start so e.g. a
                    # same-lane lead car is already rolling instead of being a
                    # 0-speed obstacle for the flying-start ego, then autopilot
                    # keeps them driving.
                    if os.environ.get('AUTOSCENARIO_DISABLE_FLYING_START', '0') == '0':
                        _autoscenario_apply_target_velocity(
                            _bg_actor,
                            _autoscenario_sampled_flying_speed(
                                _bg_id,
                                _autoscenario_cruise_speed_mps,
                            ),
                        )
                    if _autoscenario_traffic_manager is not None:
                        _bg_actor.set_autopilot(True, _autoscenario_traffic_manager.get_port())
                    else:
                        _bg_actor.set_autopilot(True)
                    _autoscenario_background_actors.append(_bg_actor)
                except Exception:
                    pass

            _autoscenario_spectator = None
            try:
                _autoscenario_spectator = world.get_spectator()
            except Exception:
                pass

            def _autoscenario_update_spectator(ego, spectator):
                if spectator is None or ego is None:
                    return
                try:
                    ego_tf = ego.get_transform()
                    import math as _math
                    yaw_rad = _math.radians(ego_tf.rotation.yaw)
                    offset_x = -12.0 * _math.cos(yaw_rad)
                    offset_y = -12.0 * _math.sin(yaw_rad)
                    spec_loc = carla.Location(
                        ego_tf.location.x + offset_x,
                        ego_tf.location.y + offset_y,
                        ego_tf.location.z + 6.0,
                    )
                    spec_rot = carla.Rotation(pitch=-20.0, yaw=ego_tf.rotation.yaw, roll=0.0)
                    spectator.set_transform(carla.Transform(spec_loc, spec_rot))
                except Exception:
                    pass

            try:
                for tick_index in range(max_ticks):
                    elapsed_s = tick_index * tick_dt
                    ego_controller.tick(elapsed_s)
                    dsl_controller.tick(elapsed_s)
                    world.tick()
                    time.sleep(0.05)
                    metrics.tick()
                    _autoscenario_update_spectator(ego_actor, _autoscenario_spectator)
            finally:
                metrics.close()
                for _bg_actor in _autoscenario_background_actors:
                    try:
                        _bg_actor.set_autopilot(False)
                    except Exception:
                        pass
                if _autoscenario_traffic_manager is not None:
                    try:
                        _autoscenario_traffic_manager.set_synchronous_mode(False)
                    except Exception:
                        pass
                try:
                    world.apply_settings(_autoscenario_original_settings)
                except Exception:
                    pass
            """
        )

    def build_existing_world_scene_script(
        self,
        spawn_payload_filename: str,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
        scene_match_status: Optional[str] = None,
        scene_match_reason: Optional[str] = None,
    ) -> str:
        helper_block = self._build_scene_helper_block()
        status_comment = f"# scene_match_status: {scene_match_status or 'unknown'}"
        if scene_match_reason:
            status_comment += f"; reason: {scene_match_reason}"
        world_loader = (
            f"world = client.load_world({carla_map!r})\n"
            if carla_map
            else "world = client.get_world()\n"
        )
        return (
            f"{status_comment}\n"
            f"{helper_block}\n\n"
            "import json\n"
            "import os\n"
            "import queue\n"
            "import carla\n"
            "import time\n\n"
            f"_AUTOSCENARIO_SPAWN_PAYLOAD = {spawn_payload_filename!r}\n\n"
            f"client = carla.Client({carla_host!r}, {int(carla_port)})\n"
            "client.set_timeout(10.0)\n"
            f"{world_loader}"
            "try:\n"
            "    world.wait_for_tick()\n"
            "except Exception:\n"
            "    time.sleep(1.0)\n"
            "blueprint_library = world.get_blueprint_library()\n"
            "_autoscenario_helpers_module._autoscenario_init(world, blueprint_library)\n\n"
            f"{self._build_spawn_payload_loop()}"
        )

    def build_dynamic_risk_scene_script(
        self,
        spawn_payload_filename: str,
        risk_sample_filename: str,
        risk_metrics_filename: str,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
        scene_match_status: Optional[str] = None,
        scene_match_reason: Optional[str] = None,
    ) -> str:
        helper_block = self._build_scene_helper_block()
        status_comment = f"# scene_match_status: {scene_match_status or 'unknown'}"
        if scene_match_reason:
            status_comment += f"; reason: {scene_match_reason}"
        world_loader = (
            f"world = client.load_world({carla_map!r})\n"
            if carla_map
            else "world = client.get_world()\n"
        )
        return (
            f"{status_comment}\n"
            "# dynamic_risk_scene: ego-scripted risk scenario\n"
            f"{helper_block}\n\n"
            "import json\n"
            "import math\n"
            "import os\n"
            "import queue\n"
            "import carla\n"
            "import time\n\n"
            f"_AUTOSCENARIO_SPAWN_PAYLOAD = {spawn_payload_filename!r}\n"
            f"_AUTOSCENARIO_RISK_SAMPLE = {risk_sample_filename!r}\n"
            f"_AUTOSCENARIO_RISK_METRICS = {risk_metrics_filename!r}\n\n"
            f"client = carla.Client({carla_host!r}, {int(carla_port)})\n"
            "client.set_timeout(10.0)\n"
            f"{world_loader}"
            "try:\n"
            "    world.wait_for_tick()\n"
            "except Exception:\n"
            "    time.sleep(1.0)\n"
            "blueprint_library = world.get_blueprint_library()\n"
            "_autoscenario_helpers_module._autoscenario_init(world, blueprint_library)\n\n"
            f"{self._build_dynamic_risk_loop()}"
        )

    def build_dsl_risk_scene_script(
        self,
        spawn_payload_filename: str,
        dsl_filename: str,
        risk_metrics_filename: str,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
        scene_match_status: Optional[str] = None,
        scene_match_reason: Optional[str] = None,
    ) -> str:
        helper_block = self._build_scene_helper_block()
        status_comment = f"# scene_match_status: {scene_match_status or 'unknown'}"
        if scene_match_reason:
            status_comment += f"; reason: {scene_match_reason}"
        world_loader = (
            f"world = client.load_world({carla_map!r})\n"
            if carla_map
            else "world = client.get_world()\n"
        )
        return (
            f"{status_comment}\n"
            "# dsl_risk_scene: pure-LLM DSL-driven risk scenario\n"
            f"{helper_block}\n\n"
            "import json\n"
            "import math\n"
            "import os\n"
            "import queue\n"
            "import carla\n"
            "import time\n\n"
            f"_AUTOSCENARIO_SPAWN_PAYLOAD = {spawn_payload_filename!r}\n"
            f"_AUTOSCENARIO_RISK_DSL = {dsl_filename!r}\n"
            f"_AUTOSCENARIO_RISK_METRICS = {risk_metrics_filename!r}\n\n"
            f"client = carla.Client({carla_host!r}, {int(carla_port)})\n"
            "client.set_timeout(10.0)\n"
            f"{world_loader}"
            "try:\n"
            "    world.wait_for_tick()\n"
            "except Exception:\n"
            "    time.sleep(1.0)\n"
            "blueprint_library = world.get_blueprint_library()\n"
            "_autoscenario_helpers_module._autoscenario_init(world, blueprint_library)\n\n"
            f"{self._build_dsl_risk_loop()}"
        )
