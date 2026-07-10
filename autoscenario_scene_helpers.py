import json
import math
import os
import queue
import random

try:
    import carla
except ImportError:
    carla = None

_AUTOSCENARIO_SPAWNED_ACTORS = []
_AUTOSCENARIO_FOCUS_POINTS = []
_AUTOSCENARIO_EXISTING_VEHICLES = []
_AUTOSCENARIO_PARKING_PROJECTION_RESULTS = {}
_AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT = {
    "result": "unavailable",
    "lane": None,
}

_AUTOSCENARIO_VEHICLE_BLUEPRINTS = {
    "bike": [
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    ],
    "car": [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.citroen.c3",
        "vehicle.mini.cooper_s",
    ],
    "jeep": [
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.nissan.patrol",
    ],
    "motorcycle": [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
    ],
    "suv": [
        "vehicle.audi.etron",
        "vehicle.nissan.patrol",
        "vehicle.tesla.cybertruck",
    ],
    "truck": [
        "vehicle.carlamotors.carlacola",
        "vehicle.tesla.cybertruck",
        "vehicle.ford.ambulance",
    ],
    "van": [
        "vehicle.mercedes.sprinter",
        "vehicle.ford.ambulance",
    ],
}

_AUTOSCENARIO_STATIC_BLUEPRINTS = {
    "warningconstruction": ["static.prop.warningconstruction"],
    "streetbarrier": ["static.prop.streetbarrier"],
    "constructioncone": ["static.prop.constructioncone"],
    "warningaccident": ["static.prop.warningaccident"],
}

world = None
blueprint_library = None


def _autoscenario_init(w, bl):
    global world, blueprint_library
    world = w
    blueprint_library = bl
    _AUTOSCENARIO_PARKING_PROJECTION_RESULTS.clear()


def _autoscenario_normalize_key(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _autoscenario_to_location(location):
    if hasattr(location, "x") and hasattr(location, "y") and hasattr(location, "z"):
        return carla.Location(float(location.x), float(location.y), float(location.z))
    if isinstance(location, (list, tuple)) and len(location) >= 3:
        return carla.Location(float(location[0]), float(location[1]), float(location[2]))
    raise ValueError(f"Unsupported location value: {location!r}")


def _autoscenario_to_rotation(rotation):
    if hasattr(rotation, "pitch") and hasattr(rotation, "yaw") and hasattr(rotation, "roll"):
        return carla.Rotation(
            float(rotation.pitch), float(rotation.yaw), float(rotation.roll)
        )
    if hasattr(rotation, "x") and hasattr(rotation, "y") and hasattr(rotation, "z"):
        return carla.Rotation(float(rotation.x), float(rotation.y), float(rotation.z))
    if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
        return carla.Rotation(float(rotation[0]), float(rotation[1]), float(rotation[2]))
    return carla.Rotation()


def _autoscenario_record_focus_point(location):
    try:
        _AUTOSCENARIO_FOCUS_POINTS.append(_autoscenario_to_location(location))
    except Exception:
        return


def _autoscenario_normalize_yaw(yaw_value):
    while yaw_value <= -180.0:
        yaw_value += 360.0
    while yaw_value > 180.0:
        yaw_value -= 360.0
    return yaw_value


def _autoscenario_angle_distance(yaw_a, yaw_b):
    return abs(_autoscenario_normalize_yaw(yaw_a - yaw_b))


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
            bp = blueprint_library.find(blueprint_id)
        except Exception:
            continue
        if _autoscenario_blueprint_matches_category(bp, category, semantic_name):
            return bp

    for pattern in fallback_patterns:
        try:
            matches = list(blueprint_library.filter(pattern))
        except Exception:
            matches = []
        filtered = [
            bp
            for bp in matches
            if _autoscenario_blueprint_matches_category(bp, category, semantic_name)
        ]
        if filtered:
            return random.choice(filtered)

    return None


def _autoscenario_apply_vehicle_color(blueprint, color):
    if blueprint is None or not color:
        return
    try:
        has_color = blueprint.has_attribute("color")
    except Exception:
        has_color = False
    if not has_color:
        return

    if isinstance(color, (list, tuple)) and len(color) >= 3:
        color_value = ",".join(str(int(channel)) for channel in color[:3])
    else:
        color_value = str(color).strip()
    if not color_value:
        return

    try:
        blueprint.set_attribute("color", color_value)
    except Exception:
        return


def _autoscenario_apply_role_name(blueprint, role_name):
    if blueprint is None or not role_name:
        return
    try:
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", str(role_name))
    except Exception:
        pass


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
    global _AUTOSCENARIO_EXISTING_VEHICLES
    try:
        result = [
            actor
            for actor in list(world.get_actors().filter("vehicle.*"))
            if actor not in _AUTOSCENARIO_SPAWNED_ACTORS
        ]
    except Exception:
        result = []
    _AUTOSCENARIO_EXISTING_VEHICLES = result
    return result


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
            base_location.x + forward_x * forward_offset + right_x * right_offset,
            base_location.y + forward_y * forward_offset + right_y * right_offset,
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


def _autoscenario_get_driving_waypoint(location):
    try:
        world_map = world.get_map()
    except Exception:
        return None
    lane_type = getattr(carla.LaneType, "Driving", None)
    try:
        if lane_type is not None:
            return world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=lane_type,
            )
        return world_map.get_waypoint(location, project_to_road=True)
    except TypeError:
        try:
            return world_map.get_waypoint(location, True)
        except Exception:
            return None
    except Exception:
        return None


def _autoscenario_step_waypoint(waypoint, distance_m, forward=True):
    if waypoint is None or abs(float(distance_m)) <= 0.05:
        return waypoint
    try:
        candidates = waypoint.next(float(distance_m)) if forward else waypoint.previous(float(distance_m))
    except Exception:
        candidates = []
    return candidates[0] if candidates else waypoint


def _autoscenario_step_waypoint_same_lane(
    waypoint, distance_m, forward=True, target_road_id=None, target_lane_id=None
):
    stepped = _autoscenario_step_waypoint(waypoint, distance_m, forward=forward)
    if stepped is None:
        return waypoint
    try:
        if (
            int(stepped.road_id) == int(target_road_id)
            and int(stepped.lane_id) == int(target_lane_id)
        ):
            return stepped
    except Exception:
        pass
    return waypoint


def _autoscenario_project_vehicle_to_specific_lane(
    location, rotation, projected_lane, junction_distance_m=None
):
    projected_lane = projected_lane or {}
    anchor = projected_lane.get("anchor") if isinstance(projected_lane, dict) else None
    if not isinstance(anchor, dict):
        return _autoscenario_project_vehicle_to_lane(location, rotation)

    base_rotation = _autoscenario_to_rotation(rotation)
    anchor_location = carla.Location(
        float(anchor.get("x", 0.0)),
        float(anchor.get("y", 0.0)),
        float(anchor.get("z", 0.0)),
    )
    waypoint = _autoscenario_get_driving_waypoint(anchor_location)
    if waypoint is None:
        return _autoscenario_project_vehicle_to_lane(location, rotation)

    try:
        target_road_id = int(projected_lane.get("road_id"))
        target_lane_id = int(projected_lane.get("lane_id"))
        if int(waypoint.road_id) != target_road_id or int(waypoint.lane_id) != target_lane_id:
            return _autoscenario_project_vehicle_to_lane(location, rotation)
    except Exception:
        return _autoscenario_project_vehicle_to_lane(location, rotation)

    try:
        desired_distance = float(junction_distance_m)
        anchor_distance = float(anchor.get("distance_from_center_m"))
        delta = desired_distance - anchor_distance
    except Exception:
        delta = 0.0

    role = str(projected_lane.get("role") or "")
    if delta > 0.05:
        waypoint = _autoscenario_step_waypoint_same_lane(
            waypoint,
            delta,
            forward=(role == "outbound"),
            target_road_id=target_road_id,
            target_lane_id=target_lane_id,
        )
    elif delta < -0.05:
        waypoint = _autoscenario_step_waypoint_same_lane(
            waypoint,
            -delta,
            forward=(role != "outbound"),
            target_road_id=target_road_id,
            target_lane_id=target_lane_id,
        )

    snapped_location = waypoint.transform.location
    waypoint_yaw = float(waypoint.transform.rotation.yaw)
    try:
        yaw_reference = float(projected_lane.get("yaw"))
    except Exception:
        yaw_reference = float(base_rotation.yaw)
    snapped_yaw = min(
        [waypoint_yaw, waypoint_yaw + 180.0],
        key=lambda yaw_value: _autoscenario_angle_distance(yaw_value, yaw_reference),
    )
    return (
        carla.Location(snapped_location.x, snapped_location.y, snapped_location.z + 0.35),
        carla.Rotation(
            float(base_rotation.pitch),
            _autoscenario_normalize_yaw(snapped_yaw),
            float(base_rotation.roll),
        ),
    )


def _autoscenario_waypoint_matches_lane(waypoint, road_id, lane_id):
    if waypoint is None:
        return False
    try:
        return int(waypoint.road_id) == int(road_id) and int(waypoint.lane_id) == int(lane_id)
    except Exception:
        return False


def _autoscenario_lane_is_parking(waypoint):
    if waypoint is None:
        return False
    try:
        return waypoint.lane_type == carla.LaneType.Parking
    except Exception:
        return "parking" in str(getattr(waypoint, "lane_type", "")).lower()


def _autoscenario_find_adjacent_waypoint_by_id(seed_waypoint, road_id, lane_id, max_depth=8):
    if seed_waypoint is None:
        return None
    queue = [(seed_waypoint, 0)]
    visited = set()
    while queue:
        waypoint, depth = queue.pop(0)
        try:
            key = (int(waypoint.road_id), int(waypoint.lane_id))
        except Exception:
            key = (id(waypoint), depth)
        if key in visited:
            continue
        visited.add(key)
        if _autoscenario_waypoint_matches_lane(waypoint, road_id, lane_id):
            return waypoint
        if depth >= max_depth:
            continue
        for getter in ("get_left_lane", "get_right_lane"):
            try:
                nxt = getattr(waypoint, getter)()
            except Exception:
                nxt = None
            if nxt is not None:
                queue.append((nxt, depth + 1))
    return None


def _autoscenario_find_adjacent_parking_waypoint(seed_waypoint, road_id=None, max_depth=8):
    if seed_waypoint is None:
        return None
    queue = [(seed_waypoint, 0)]
    visited = set()
    while queue:
        waypoint, depth = queue.pop(0)
        try:
            key = (int(waypoint.road_id), int(waypoint.lane_id))
        except Exception:
            key = (id(waypoint), depth)
        if key in visited:
            continue
        visited.add(key)
        same_road = road_id is None
        try:
            same_road = same_road or int(waypoint.road_id) == int(road_id)
        except Exception:
            pass
        if same_road and _autoscenario_lane_is_parking(waypoint):
            return waypoint
        if depth >= max_depth:
            continue
        for getter in ("get_left_lane", "get_right_lane"):
            try:
                nxt = getattr(waypoint, getter)()
            except Exception:
                nxt = None
            if nxt is not None:
                queue.append((nxt, depth + 1))
    return None


def _autoscenario_rightmost_same_direction_driving_waypoint(seed_waypoint, max_depth=8):
    if seed_waypoint is None or not _autoscenario_lane_is_driving(seed_waypoint):
        return None
    current = seed_waypoint
    try:
        base_yaw = float(current.transform.rotation.yaw)
    except Exception:
        base_yaw = None
    for _ in range(max_depth):
        try:
            candidate = current.get_right_lane()
        except Exception:
            candidate = None
        if candidate is None or not _autoscenario_lane_is_driving(candidate):
            break
        if base_yaw is not None:
            try:
                candidate_yaw = float(candidate.transform.rotation.yaw)
                if _autoscenario_angle_distance(candidate_yaw, base_yaw) > 45.0:
                    break
            except Exception:
                pass
        current = candidate
    return current


def _autoscenario_get_waypoint_for_lane_type(location, lane_type):
    try:
        world_map = world.get_map()
    except Exception:
        return None
    try:
        if lane_type is not None:
            return world_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=lane_type,
            )
        return world_map.get_waypoint(location, project_to_road=True)
    except TypeError:
        try:
            return world_map.get_waypoint(location, True)
        except Exception:
            return None
    except Exception:
        return None


def _autoscenario_project_vehicle_to_parking_lane(
    location,
    rotation,
    projected_lane=None,
    lane_side_relation=None,
):
    global _AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT
    base_location = _autoscenario_to_location(location)
    base_rotation = _autoscenario_to_rotation(rotation)
    projected_lane = projected_lane if isinstance(projected_lane, dict) else {}

    target_road_id = projected_lane.get("road_id")
    target_lane_id = projected_lane.get("lane_id")
    parking_type = getattr(carla.LaneType, "Parking", None)
    driving_type = getattr(carla.LaneType, "Driving", None)

    waypoint = _autoscenario_get_waypoint_for_lane_type(base_location, parking_type)
    if _autoscenario_lane_is_parking(waypoint) and target_road_id is not None:
        try:
            if int(waypoint.road_id) != int(target_road_id):
                waypoint = None
        except Exception:
            waypoint = None

    driving_seed = _autoscenario_get_waypoint_for_lane_type(base_location, driving_type)
    if target_road_id is not None and target_lane_id is not None:
        exact_seed = _autoscenario_find_adjacent_waypoint_by_id(
            driving_seed,
            target_road_id,
            target_lane_id,
        )
        if exact_seed is not None:
            driving_seed = exact_seed

    if not _autoscenario_lane_is_parking(waypoint):
        waypoint = _autoscenario_find_adjacent_parking_waypoint(
            driving_seed,
            target_road_id,
        )

    projection_result = "parking_lane"
    if waypoint is None and str(lane_side_relation or "") == "right_parking_lane":
        waypoint = _autoscenario_rightmost_same_direction_driving_waypoint(
            driving_seed
        )
        projection_result = "rightmost_driving_fallback"

    if waypoint is None:
        _AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT = {
            "result": "unavailable",
            "lane": None,
        }
        return base_location, base_rotation

    snapped_location = waypoint.transform.location
    waypoint_yaw = float(waypoint.transform.rotation.yaw)
    try:
        yaw_reference = float(projected_lane.get("yaw"))
    except Exception:
        yaw_reference = float(base_rotation.yaw)
    snapped_yaw = min(
        [waypoint_yaw, waypoint_yaw + 180.0],
        key=lambda yaw_value: _autoscenario_angle_distance(yaw_value, yaw_reference),
    )
    try:
        projection_lane = {
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
        }
    except Exception:
        projection_lane = None
    _AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT = {
        "result": projection_result,
        "lane": projection_lane,
    }
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
        loc = waypoint.transform.location
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
            carla.Location(loc.x, loc.y, loc.z + 0.6),
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


def _autoscenario_collect_strict_lane_spawn_candidates(location, rotation):
    base_location = _autoscenario_to_location(location)
    base_rotation = _autoscenario_to_rotation(rotation)
    yaw_radians = math.radians(base_rotation.yaw)
    forward_x = math.cos(yaw_radians)
    forward_y = math.sin(yaw_radians)
    candidates = []
    for forward_offset in (0.0, 5.5, -5.5, 11.0, -11.0, 16.5, -16.5):
        candidates.append(
            (
                carla.Location(
                    base_location.x + forward_x * forward_offset,
                    base_location.y + forward_y * forward_offset,
                    base_location.z,
                ),
                base_rotation,
            )
        )
    return candidates


def _autoscenario_try_spawn_vehicle_actor_strict_lane(
    blueprint, location, rotation, min_spacing_m=None
):
    if blueprint is None:
        return None
    min_spacing = (
        _autoscenario_min_spawn_spacing_for_blueprint(blueprint)
        if min_spacing_m is None
        else float(min_spacing_m)
    )

    for candidate_location, candidate_rotation in _autoscenario_collect_strict_lane_spawn_candidates(
        location,
        rotation,
    ):
        if not _autoscenario_spawn_location_is_clear(
            candidate_location,
            min_spacing,
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
    bp = _autoscenario_pick_blueprint("static", blueprint_name)
    return _autoscenario_try_spawn_static(bp, snapped_location, rotation)


def _autoscenario_spawn_vehicle(blueprint_name, location, rotation, color=None, role_name=None):
    snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_lane(location, rotation)
    _autoscenario_record_focus_point(snapped_location)
    bp = _autoscenario_pick_blueprint("vehicle", blueprint_name)
    _autoscenario_apply_vehicle_color(bp, color)
    _autoscenario_apply_role_name(bp, role_name)
    return _autoscenario_try_spawn_vehicle_actor(bp, snapped_location, snapped_rotation)


def _autoscenario_spawn_vehicle_junction_lane(
    blueprint_name,
    location,
    rotation,
    projected_lane=None,
    color=None,
    role_name=None,
    junction_distance_m=None,
):
    snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_specific_lane(
        location, rotation, projected_lane, junction_distance_m
    )
    if snapped_location is None or snapped_rotation is None:
        return None
    if isinstance(projected_lane, dict) and projected_lane.get("preserve_input_yaw"):
        snapped_rotation.yaw = float(_autoscenario_to_rotation(rotation).yaw)
    _autoscenario_record_focus_point(snapped_location)
    bp = _autoscenario_pick_blueprint("vehicle", blueprint_name)
    _autoscenario_apply_vehicle_color(bp, color)
    _autoscenario_apply_role_name(bp, role_name)
    return _autoscenario_try_spawn_vehicle_actor_strict_lane(
        bp, snapped_location, snapped_rotation, 0.5
    )


def _autoscenario_spawn_vehicle_parking_lane(
    blueprint_name,
    location,
    rotation,
    projected_lane=None,
    color=None,
    role_name=None,
    entity_id=None,
    lane_side_relation=None,
):
    snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_parking_lane(
        location,
        rotation,
        projected_lane,
        lane_side_relation=lane_side_relation,
    )
    projection_key = str(entity_id or role_name or "")
    if projection_key:
        _AUTOSCENARIO_PARKING_PROJECTION_RESULTS[projection_key] = dict(
            _AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT
        )
    _autoscenario_record_focus_point(snapped_location)
    bp = _autoscenario_pick_blueprint("vehicle", blueprint_name)
    _autoscenario_apply_vehicle_color(bp, color)
    _autoscenario_apply_role_name(bp, role_name)
    return _autoscenario_try_spawn_vehicle_actor_strict_lane(
        bp,
        snapped_location,
        snapped_rotation,
        3.5,
    )


def _autoscenario_lane_is_driving(waypoint):
    if waypoint is None:
        return False
    try:
        return waypoint.lane_type == carla.LaneType.Driving
    except Exception:
        return True


def _autoscenario_innermost_opposing_lane(base_waypoint, opposing_index=1):
    """Resolve which opposing-carriageway lane to use.

    ``base_waypoint`` is any waypoint already on the opposing carriageway (the
    cross-median seed snaps somewhere onto it, often the far lane). In right-hand
    traffic the median sits to the *left* of the opposing direction of travel, so
    walking ``get_left_lane`` repeatedly reaches the lane nearest the median --
    i.e. the opposing carriageway's leftmost lane. ``opposing_index`` (1-based
    from the median) then steps back outward toward the far edge if needed.
    """
    innermost = base_waypoint
    for _ in range(12):
        try:
            nxt = innermost.get_left_lane()
        except Exception:
            nxt = None
        if not _autoscenario_lane_is_driving(nxt):
            break
        # A reversed heading means we crossed the median back into the
        # same-direction carriageway; stop before leaving the opposing side.
        if _autoscenario_angle_distance(
            float(nxt.transform.rotation.yaw),
            float(innermost.transform.rotation.yaw),
        ) > 90.0:
            break
        innermost = nxt

    target = innermost
    for _ in range(max(1, int(opposing_index or 1)) - 1):
        try:
            nxt = target.get_right_lane()
        except Exception:
            nxt = None
        if not _autoscenario_lane_is_driving(nxt):
            break
        if _autoscenario_angle_distance(
            float(nxt.transform.rotation.yaw),
            float(target.transform.rotation.yaw),
        ) > 90.0:
            break
        target = nxt
    return target


def _autoscenario_project_to_opposing_lane(location, rotation, opposing_index=1):
    base_location = _autoscenario_to_location(location)
    base_rotation = _autoscenario_to_rotation(rotation)
    try:
        world_map = world.get_map()
    except Exception:
        return base_location, base_rotation

    lane_type = getattr(carla.LaneType, "Driving", None)
    try:
        if lane_type is not None:
            waypoint = world_map.get_waypoint(
                base_location, project_to_road=True, lane_type=lane_type
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

    target = _autoscenario_innermost_opposing_lane(waypoint, opposing_index)
    snapped_location = target.transform.location
    waypoint_yaw = float(target.transform.rotation.yaw)
    snapped_yaw = min(
        [waypoint_yaw, waypoint_yaw + 180.0],
        key=lambda yaw_value: _autoscenario_angle_distance(
            yaw_value, float(base_rotation.yaw)
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


def _autoscenario_spawn_vehicle_opposing(
    blueprint_name, location, rotation, opposing_index=1, color=None, role_name=None
):
    snapped_location, snapped_rotation = _autoscenario_project_to_opposing_lane(
        location, rotation, opposing_index
    )
    _autoscenario_record_focus_point(snapped_location)
    bp = _autoscenario_pick_blueprint("vehicle", blueprint_name)
    _autoscenario_apply_vehicle_color(bp, color)
    _autoscenario_apply_role_name(bp, role_name)
    return _autoscenario_try_spawn_vehicle_actor(bp, snapped_location, snapped_rotation)


def _autoscenario_spawn_vehicle_direct(blueprint_name, location, rotation, color=None, role_name=None):
    _autoscenario_record_focus_point(location)
    bp = _autoscenario_pick_blueprint("vehicle", blueprint_name)
    _autoscenario_apply_vehicle_color(bp, color)
    _autoscenario_apply_role_name(bp, role_name)
    return _autoscenario_try_spawn(bp, location, rotation, "vehicle")


def _autoscenario_spawn_pedestrian(location, rotation):
    _autoscenario_record_focus_point(location)
    try:
        pedestrian_blueprints = list(blueprint_library.filter("walker.pedestrian.*"))
    except Exception:
        pedestrian_blueprints = []
    if not pedestrian_blueprints:
        return None
    return _autoscenario_try_spawn(
        random.choice(pedestrian_blueprints),
        location,
        rotation,
        "walker",
    )


def _autoscenario_default_weather():
    try:
        return carla.WeatherParameters.ClearNoon
    except Exception:
        return carla.WeatherParameters()


def _autoscenario_apply_weather(weather_value=None):
    if weather_value is None:
        world.set_weather(_autoscenario_default_weather())
        return

    if isinstance(weather_value, str):
        preset = getattr(carla.WeatherParameters, weather_value, None)
        if preset is not None:
            world.set_weather(preset)
            return
        world.set_weather(_autoscenario_default_weather())
        return

    if isinstance(weather_value, dict):
        weather = carla.WeatherParameters()
        for key, value in weather_value.items():
            if hasattr(weather, key):
                setattr(weather, key, value)
        world.set_weather(weather)
        return

    world.set_weather(weather_value)


def _autoscenario_focus_spectator():
    try:
        spectator = world.get_spectator()
    except Exception:
        return

    focus_locations = []
    for actor in _AUTOSCENARIO_SPAWNED_ACTORS:
        if actor is None:
            continue
        try:
            focus_locations.append(actor.get_transform().location)
        except Exception:
            continue

    if not focus_locations:
        focus_locations = list(_AUTOSCENARIO_FOCUS_POINTS)
    if not focus_locations:
        return

    avg_x = sum(loc.x for loc in focus_locations) / len(focus_locations)
    avg_y = sum(loc.y for loc in focus_locations) / len(focus_locations)
    avg_z = sum(loc.z for loc in focus_locations) / len(focus_locations)
    spectator_location = carla.Location(avg_x, avg_y, avg_z + 18.0)
    spectator_rotation = carla.Rotation(pitch=-55.0, yaw=0.0, roll=0.0)

    try:
        spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
    except Exception:
        return


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

    avg_x = sum(loc.x for loc in focus_locations) / len(focus_locations)
    avg_y = sum(loc.y for loc in focus_locations) / len(focus_locations)
    max_z = max(loc.z for loc in focus_locations)
    camera_height = float(os.environ.get("AUTOSCENARIO_BEV_HEIGHT", "80"))
    image_size = str(os.environ.get("AUTOSCENARIO_BEV_SIZE", "1024"))
    fov = str(os.environ.get("AUTOSCENARIO_BEV_FOV", "55"))

    try:
        bp = blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", image_size)
        bp.set_attribute("image_size_y", image_size)
        bp.set_attribute("fov", fov)
        bp.set_attribute("sensor_tick", "0.05")
    except Exception:
        return

    transform = carla.Transform(
        carla.Location(avg_x, avg_y, max_z + camera_height),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    sensor = None
    image_queue = queue.Queue()
    try:
        sensor = world.spawn_actor(bp, transform)
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
        bp = blueprint_library.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", image_size)
        bp.set_attribute("image_size_y", image_size)
        bp.set_attribute("fov", fov)
        bp.set_attribute("sensor_tick", "0.05")
    except Exception:
        return

    transform = carla.Transform(
        carla.Location(x=1.2, y=0.0, z=1.6),
        carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
    )
    sensor = None
    image_queue = queue.Queue()
    try:
        sensor = world.spawn_actor(bp, transform, attach_to=ego_actor)
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
    delta = _autoscenario_angle_distance(float(actor_yaw), float(ego_yaw))
    if delta <= 45.0:
        return "same_direction"
    if delta < 135.0:
        return "crossing"
    return "opposite_direction"


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


def _autoscenario_waypoint_record(location):
    try:
        world_map = world.get_map()
        carla_location = carla.Location(
            float(location.get("x", 0.0)),
            float(location.get("y", 0.0)),
            float(location.get("z", 0.0)),
        )
        lane_type = getattr(carla.LaneType, "Driving", None)
        if lane_type is not None:
            waypoint = world_map.get_waypoint(
                carla_location,
                project_to_road=True,
                lane_type=lane_type,
            )
        else:
            waypoint = world_map.get_waypoint(carla_location, project_to_road=True)
    except TypeError:
        try:
            waypoint = world_map.get_waypoint(carla_location, True)
        except Exception:
            waypoint = None
    except Exception:
        waypoint = None
    if waypoint is None:
        return None
    try:
        loc = waypoint.transform.location
        rot = waypoint.transform.rotation
        return {
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "x": float(loc.x),
            "y": float(loc.y),
            "z": float(loc.z),
            "yaw": float(rot.yaw),
        }
    except Exception:
        return None


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
            "projected_lane": entity.get("projected_lane") or {},
            "truth_source": "carla_actor_transform",
        }
        parking_projection = _AUTOSCENARIO_PARKING_PROJECTION_RESULTS.get(entity_id)
        if isinstance(parking_projection, dict):
            record["parking_projection_result"] = parking_projection.get("result")
            record["parking_projection_lane"] = parking_projection.get("lane")
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
            record["actual_waypoint"] = _autoscenario_waypoint_record(
                actor_record["location"]
            )
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
