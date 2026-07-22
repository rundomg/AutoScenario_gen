#!/usr/bin/env python3
"""Place the vehicles from ``data/huandao.png`` at the current CARLA view.

The script does not load a map.  It projects the current spectator onto the
roundabout approach, uses the lane on screen-left as the inner reference, and
leaves the five-vehicle static scene in the running world by default.
"""

from __future__ import annotations

import argparse
import math

import carla


ROLE_PREFIX = "autoscenario_huandao_"
TOWN03_ROUNDABOUT_CENTER = (0.0, 0.0)
TOWN03_ORIGINAL_VIEW_ANCHOR = (36.319473, -4.440626, -0.019535)
INNER_VEHICLE_FORWARD_SHIFT = 5.0


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _same_direction(a: carla.Waypoint, b: carla.Waypoint | None) -> bool:
    return (
        b is not None
        and b.lane_type == carla.LaneType.Driving
        and _angle_delta(a.transform.rotation.yaw, b.transform.rotation.yaw) < 45.0
    )


def _screen_left_lane(anchor: carla.Waypoint) -> carla.Waypoint:
    """Return the adjacent driving lane visible to the left of the camera."""
    left = anchor.get_left_lane()
    if _same_direction(anchor, left):
        return left
    right = anchor.get_right_lane()
    if _same_direction(anchor, right):
        return right
    raise RuntimeError("当前视角附近没有找到与自车并行的第二条环岛车道")


def _advance(wp: carla.Waypoint, distance: float) -> carla.Waypoint:
    """Follow the smoothest continuation so junction exits are not selected."""
    current = wp
    remaining = abs(float(distance))
    method = "next" if distance >= 0.0 else "previous"
    while remaining > 0.05:
        step = min(1.0, remaining)
        options = list(getattr(current, method)(step))
        if not options:
            raise RuntimeError(f"车道无法沿环岛继续延伸 {distance:.1f} m")
        yaw = current.transform.rotation.yaw
        current = min(
            options,
            key=lambda item: (
                _angle_delta(yaw, item.transform.rotation.yaw),
                0 if item.is_junction else 1,
            ),
        )
        remaining -= step
    return current


def _blend_transform(
    outer: carla.Waypoint, inner: carla.Waypoint, inner_weight: float
) -> carla.Transform:
    a = outer.transform
    b = inner.transform
    weight = max(0.0, min(1.0, float(inner_weight)))
    location = carla.Location(
        x=a.location.x + (b.location.x - a.location.x) * weight,
        y=a.location.y + (b.location.y - a.location.y) * weight,
        z=a.location.z + (b.location.z - a.location.z) * weight,
    )
    return carla.Transform(location, a.rotation)


def _offset_transform_local(
    transform: carla.Transform, forward_m: float, right_m: float
) -> carla.Transform:
    yaw = math.radians(transform.rotation.yaw)
    forward_x, forward_y = math.cos(yaw), math.sin(yaw)
    right_x, right_y = forward_y, -forward_x
    return carla.Transform(
        carla.Location(
            x=transform.location.x + forward_x * forward_m + right_x * right_m,
            y=transform.location.y + forward_y * forward_m + right_y * right_m,
            z=transform.location.z,
        ),
        transform.rotation,
    )


def _innermost_lane(wp: carla.Waypoint) -> carla.Waypoint:
    """Choose the same-direction lane closest to Town03's center island."""
    candidates = [wp]
    for neighbor in (wp.get_left_lane(), wp.get_right_lane()):
        if _same_direction(wp, neighbor):
            candidates.append(neighbor)
    center_x, center_y = TOWN03_ROUNDABOUT_CENTER
    return min(
        candidates,
        key=lambda item: math.hypot(
            item.transform.location.x - center_x,
            item.transform.location.y - center_y,
        ),
    )


def _inner_companion_transform(
    black_transform: carla.Transform,
    longitudinal_offset: float,
    centerward_offset: float,
) -> carla.Transform:
    """Place a companion on the same inner arc without ambiguous map links."""
    center_x, center_y = TOWN03_ROUNDABOUT_CENTER
    forward = black_transform.get_forward_vector()
    x = black_transform.location.x + forward.x * longitudinal_offset
    y = black_transform.location.y + forward.y * longitudinal_offset
    radius = math.hypot(x - center_x, y - center_y)
    if radius > 1e-6:
        x += (center_x - x) * centerward_offset / radius
        y += (center_y - y) * centerward_offset / radius
    radial_yaw = math.degrees(math.atan2(y - center_y, x - center_x))
    tangent_candidates = (radial_yaw - 90.0, radial_yaw + 90.0)
    yaw = min(
        tangent_candidates,
        key=lambda candidate: _angle_delta(candidate, black_transform.rotation.yaw),
    )
    return carla.Transform(
        carla.Location(x=x, y=y, z=black_transform.location.z),
        carla.Rotation(yaw=yaw),
    )


def _move_along_inner_arc(
    transform: carla.Transform, distance: float
) -> carla.Transform:
    """Move a vehicle by arc length while preserving its roundabout radius."""
    center_x, center_y = TOWN03_ROUNDABOUT_CENTER
    dx = transform.location.x - center_x
    dy = transform.location.y - center_y
    radius = math.hypot(dx, dy)
    if radius <= 1e-6:
        raise RuntimeError("车辆位于环岛中心，无法沿内圈移动")
    radial_yaw = math.degrees(math.atan2(dy, dx))
    clockwise_yaw = radial_yaw - 90.0
    counterclockwise_yaw = radial_yaw + 90.0
    direction = (
        -1.0
        if _angle_delta(transform.rotation.yaw, clockwise_yaw)
        <= _angle_delta(transform.rotation.yaw, counterclockwise_yaw)
        else 1.0
    )
    angle = direction * float(distance) / radius
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return carla.Transform(
        carla.Location(
            x=center_x + dx * cos_angle - dy * sin_angle,
            y=center_y + dx * sin_angle + dy * cos_angle,
            z=transform.location.z,
        ),
        carla.Rotation(yaw=transform.rotation.yaw + math.degrees(angle)),
    )


def _blueprint(
    world: carla.World,
    role: str,
    color: str,
    choices: tuple[str, ...],
) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    selected = None
    for blueprint_id in choices:
        matches = library.filter(blueprint_id)
        if matches:
            selected = library.find(matches[0].id)
            break
    if selected is None:
        selected = library.find(library.filter("vehicle.*")[0].id)
    if selected.has_attribute("role_name"):
        selected.set_attribute("role_name", ROLE_PREFIX + role)
    if selected.has_attribute("color"):
        selected.set_attribute("color", color)
    return selected


def _cleanup(world: carla.World) -> int:
    old = [
        actor
        for actor in world.get_actors().filter("vehicle.*")
        if actor.attributes.get("role_name", "").startswith(ROLE_PREFIX)
    ]
    for actor in old:
        actor.destroy()
    return len(old)


def _set_warm_afternoon_weather(world: carla.World) -> None:
    """Use clear, low afternoon sunlight with a warm yellow atmosphere."""
    weather = world.get_weather()
    weather.cloudiness = 8.0
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 0.0
    weather.wind_intensity = 5.0
    weather.sun_azimuth_angle = 250.0
    weather.sun_altitude_angle = 22.0
    weather.fog_density = 0.0
    weather.fog_distance = 1000.0
    weather.fog_falloff = 0.2
    weather.scattering_intensity = 1.0
    weather.mie_scattering_scale = 0.08
    weather.rayleigh_scattering_scale = 0.02
    if hasattr(weather, "dust_storm"):
        weather.dust_storm = 0.0
    world.set_weather(weather)


def _spawn(
    world: carla.World,
    role: str,
    transform: carla.Transform,
    color: str,
    choices: tuple[str, ...],
) -> carla.Vehicle:
    transform.location.z += 0.35
    actor = world.try_spawn_actor(_blueprint(world, role, color, choices), transform)
    if actor is None:
        raise RuntimeError(f"车辆 {role} 在目标位置生成失败（可能与已有物体碰撞）")
    actor.set_simulate_physics(False)
    actor.set_target_velocity(carla.Vector3D())
    actor.set_target_angular_velocity(carla.Vector3D())
    return actor


def _camera_at_ego(world: carla.World, ego: carla.Vehicle) -> None:
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    location = transform.location + carla.Location(
        x=forward.x * 0.85,
        y=forward.y * 0.85,
        z=1.55,
    )
    world.get_spectator().set_transform(
        carla.Transform(
            location,
            carla.Rotation(pitch=-3.0, yaw=transform.rotation.yaw, roll=0.0),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="只删除该脚本之前生成的 huandao 场景车辆",
    )
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    spectator = world.get_spectator().get_transform()
    anchor_location = spectator.location
    if "Town03" in carla_map.name:
        anchor_location = carla.Location(*TOWN03_ORIGINAL_VIEW_ANCHOR)
    outer_anchor = carla_map.get_waypoint(
        anchor_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if outer_anchor is None:
        raise RuntimeError("无法把当前 spectator 投影到环岛车道")
    inner_anchor = _screen_left_lane(outer_anchor)

    # The ego changes to its right-hand driving lane and advances 8 m.  The
    # white vehicle formerly ahead of it changes to its own left-hand driving
    # lane, then advances another 5 m along that lane.
    ego_transform = _advance(inner_anchor, 8.0).transform
    middle_current_lane = _advance(inner_anchor, 5.0)
    middle_left_lane = middle_current_lane.get_left_lane()
    if not _same_direction(middle_current_lane, middle_left_lane):
        raise RuntimeError("前方白车左侧没有同向可行驶车道")
    middle_white_transform = _advance(middle_left_lane, 5.0).transform
    inner_black_base_transform = _innermost_lane(
        _advance(inner_anchor, 22.0)
    ).transform
    inner_white_near_base_transform = _inner_companion_transform(
        inner_black_base_transform,
        longitudinal_offset=-9.0,
        centerward_offset=1.5,
    )
    inner_white_far_base_transform = _inner_companion_transform(
        inner_black_base_transform,
        longitudinal_offset=9.0,
        centerward_offset=1.2,
    )
    inner_white_near_transform = _move_along_inner_arc(
        inner_white_near_base_transform, INNER_VEHICLE_FORWARD_SHIFT
    )
    inner_black_transform = _move_along_inner_arc(
        inner_black_base_transform, INNER_VEHICLE_FORWARD_SHIFT
    )
    inner_white_far_transform = _move_along_inner_arc(
        inner_white_far_base_transform, INNER_VEHICLE_FORWARD_SHIFT
    )
    planned = [
        (
            "ego",
            ego_transform,
            "28,38,58",
            ("vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"),
        ),
        (
            "middle_white",
            middle_white_transform,
            "255,255,255",
            ("vehicle.audi.etron", "vehicle.tesla.model3"),
        ),
        (
            "inner_white_near",
            inner_white_near_transform,
            "255,255,255",
            ("vehicle.nissan.patrol_2021", "vehicle.audi.etron"),
        ),
        (
            "inner_black",
            inner_black_transform,
            "8,8,10",
            ("vehicle.mercedes.coupe_2020", "vehicle.audi.a2"),
        ),
        (
            "inner_white_far",
            inner_white_far_transform,
            "255,255,255",
            ("vehicle.nissan.patrol_2021", "vehicle.audi.etron"),
        ),
    ]

    print(f"[huandao] map={carla_map.name}")
    print(
        f"[huandao] ego anchor road={outer_anchor.road_id} lane={outer_anchor.lane_id} "
        f"inner lane={inner_anchor.lane_id} spectator_yaw={spectator.rotation.yaw:.1f}"
    )
    for role, transform, color, _ in planned:
        loc = transform.location
        print(
            f"[huandao] {role:>18} xyz=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}) "
            f"yaw={transform.rotation.yaw:.1f} color={color}"
        )
    if args.dry_run:
        return

    removed = _cleanup(world)
    if args.cleanup_only:
        print(f"[huandao] removed={removed}")
        return

    spawned: list[carla.Vehicle] = []
    try:
        for role, transform, color, choices in planned:
            spawned.append(_spawn(world, role, transform, color, choices))
        _set_warm_afternoon_weather(world)
        _camera_at_ego(world, spawned[0])
    except Exception:
        for actor in spawned:
            actor.destroy()
        raise
    print(
        f"[huandao] spawned={len(spawned)} removed_old={removed}; "
        "warm afternoon weather applied; scene kept in CARLA"
    )


if __name__ == "__main__":
    main()
