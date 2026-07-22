#!/usr/bin/env python3
"""Rebuild frame 00205 as a static night highway scene in the current CARLA world.

The script deliberately does not call ``load_world``.  It anchors the scene to
the driving lanes below/nearest the current spectator, chooses a middle lane
for the ego vehicle, and leaves the spawned scene in CARLA with ``--keep``.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import carla


SCENE_ROLE_PREFIX = "autoscenario_00205_"


@dataclass(frozen=True)
class VehicleSpec:
    role: str
    lane: str
    distance: float
    color: str
    blueprints: tuple[str, ...]
    braking: bool = False


VEHICLES = (
    VehicleSpec("ego", "ego", 0.0, "75,75,80", ("vehicle.lincoln.mkz_2020", "vehicle.tesla.model3")),
    VehicleSpec("black_ahead", "ego", 12.0, "5,5,8", ("vehicle.mercedes.coupe_2020", "vehicle.audi.a2"), True),
    VehicleSpec("blue_right", "right", 9.0, "12,42,125", ("vehicle.mercedes.coupe_2020", "vehicle.tesla.model3"), True),
    VehicleSpec("left_added_white", "left", 12.0, "255,255,255", ("vehicle.tesla.model3", "vehicle.mercedes.coupe_2020")),
    VehicleSpec("left_added_white_2", "left", 52.0, "255,255,255", ("vehicle.tesla.model3", "vehicle.mercedes.coupe_2020")),
    VehicleSpec("left_far_1", "left", 43.0, "205,205,210", ("vehicle.tesla.model3", "vehicle.audi.tt")),
    VehicleSpec("left_far_2", "left", 67.0, "45,50,58", ("vehicle.audi.tt", "vehicle.lincoln.mkz_2020")),
    VehicleSpec("left_far_3", "left", 92.0, "220,220,215", ("vehicle.mercedes.coupe_2020", "vehicle.tesla.model3")),
    VehicleSpec("oncoming_1", "opposite", 31.0, "235,235,230", ("vehicle.tesla.model3", "vehicle.audi.tt")),
    VehicleSpec("oncoming_2", "opposite", 60.0, "35,38,45", ("vehicle.lincoln.mkz_2020", "vehicle.audi.a2")),
)


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _is_driving(wp: carla.Waypoint | None) -> bool:
    return wp is not None and wp.lane_type == carla.LaneType.Driving


def _same_direction(a: carla.Waypoint, b: carla.Waypoint) -> bool:
    return _is_driving(b) and _angle_delta(a.transform.rotation.yaw, b.transform.rotation.yaw) < 60.0


def _lane_chain(wp: carla.Waypoint) -> list[carla.Waypoint]:
    left: list[carla.Waypoint] = []
    current = wp
    for _ in range(8):
        candidate = current.get_left_lane()
        if not _same_direction(wp, candidate):
            break
        left.append(candidate)
        current = candidate

    right: list[carla.Waypoint] = []
    current = wp
    for _ in range(8):
        candidate = current.get_right_lane()
        if not _same_direction(wp, candidate):
            break
        right.append(candidate)
        current = candidate
    return list(reversed(left)) + [wp] + right


def _choose_middle_lane(carla_map: carla.Map, spectator: carla.Transform) -> carla.Waypoint:
    nearest = carla_map.get_waypoint(
        spectator.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    candidates = [nearest]
    # If the camera is over a shoulder or ramp, prefer a three-lane main road
    # still close to the current view.
    for wp in carla_map.generate_waypoints(8.0):
        if wp.transform.location.distance(spectator.location) <= 90.0:
            candidates.append(wp)

    viable: list[tuple[float, carla.Waypoint]] = []
    for wp in candidates:
        chain = _lane_chain(wp)
        if len(chain) < 3:
            continue
        middle = chain[len(chain) // 2]
        distance = middle.transform.location.distance(spectator.location)
        # Strongly retain the road already below the spectator.
        same_road_penalty = 0.0 if middle.road_id == nearest.road_id else 25.0
        viable.append((distance + same_road_penalty, middle))
    if not viable:
        raise RuntimeError("当前 spectator 附近没有检测到至少三条同向车道的高速主路")
    return min(viable, key=lambda item: item[0])[1]


def _align_with_nearest_street_light(
    world: carla.World, ego_wp: carla.Waypoint
) -> carla.Waypoint:
    """Move the anchor longitudinally under the nearest lamp on this highway."""
    try:
        manager = world.get_lightmanager()
        lights = list(manager.get_all_lights(carla.LightGroup.Street))
    except Exception:
        return ego_wp

    carla_map = world.get_map()
    nearby = sorted(
        lights,
        key=lambda light: light.location.distance(ego_wp.transform.location),
    )
    for light in nearby:
        if light.location.distance(ego_wp.transform.location) > 80.0:
            break
        projected = carla_map.get_waypoint(
            light.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if projected is None or projected.road_id != ego_wp.road_id:
            continue
        chain = _lane_chain(projected)
        if len(chain) < 3:
            continue
        middle = chain[len(chain) // 2]
        if _angle_delta(
            middle.transform.rotation.yaw, ego_wp.transform.rotation.yaw
        ) < 60.0:
            return middle
    return ego_wp


def _advance(wp: carla.Waypoint, distance: float) -> carla.Waypoint:
    current = wp
    remaining = abs(float(distance))
    getter = "next" if distance >= 0.0 else "previous"
    while remaining > 0.05:
        step = min(2.0, remaining)
        options = getattr(current, getter)(step)
        if not options:
            raise RuntimeError(f"车道 {current.road_id}/{current.lane_id} 无法继续延伸 {distance:.1f}m")
        matching = [
            item for item in options
            if item.road_id == current.road_id and item.lane_id == current.lane_id
        ]
        current = (matching or options)[0]
        remaining -= step
    return current


def _target_location(ego_wp: carla.Waypoint, distance: float) -> carla.Location:
    return _advance(ego_wp, distance).transform.location


def _opposite_waypoint(
    samples: list[carla.Waypoint], ego_wp: carla.Waypoint, distance: float
) -> carla.Waypoint:
    target = _target_location(ego_wp, distance)
    ego_yaw = ego_wp.transform.rotation.yaw
    candidates = [
        wp for wp in samples
        if _angle_delta(ego_yaw, wp.transform.rotation.yaw) >= 130.0
        and wp.transform.location.distance(target) <= 45.0
    ]
    if not candidates:
        raise RuntimeError("当前高速路附近没有检测到对向车道")
    same_road = [wp for wp in candidates if wp.road_id == ego_wp.road_id]
    return min(same_road or candidates, key=lambda wp: wp.transform.location.distance(target))


def _blueprint(world: carla.World, spec: VehicleSpec) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    bp = None
    for blueprint_id in spec.blueprints:
        matches = library.filter(blueprint_id)
        if matches:
            bp = matches[0]
            break
    if bp is None:
        bp = library.filter("vehicle.*")[0]
    bp = library.find(bp.id)  # independent mutable blueprint instance
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", SCENE_ROLE_PREFIX + spec.role)
    if bp.has_attribute("color"):
        bp.set_attribute("color", spec.color)
    return bp


def _vehicle_lights(actor: carla.Vehicle, braking: bool = False) -> None:
    state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
    if hasattr(carla.VehicleLightState, "HighBeam"):
        state |= carla.VehicleLightState.HighBeam
    if braking:
        state |= carla.VehicleLightState.Brake
    actor.set_light_state(carla.VehicleLightState(state))


def _set_night_and_street_lights(world: carla.World) -> int:
    weather = world.get_weather()
    weather.cloudiness = 12.0
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 0.0
    weather.sun_altitude_angle = -8.0
    weather.sun_azimuth_angle = 25.0
    weather.fog_density = min(float(weather.fog_density), 2.0)
    weather.fog_distance = max(float(weather.fog_distance), 120.0)
    weather.mie_scattering_scale = 0.01
    world.set_weather(weather)

    manager = world.get_lightmanager()
    lights = list(manager.get_all_lights(carla.LightGroup.Street))
    if lights:
        manager.turn_on(lights)
        # Town12's emissive street-lamp meshes are quite dim at the default
        # intensity when the sun is well below the horizon.  Keep the night
        # ambience while making the road surface and nearby cars readable.
        manager.set_intensity(lights, 100000.0)
    return len(lights)


def _cleanup_previous(world: carla.World) -> int:
    stale = [
        actor for actor in world.get_actors().filter("vehicle.*")
        if actor.attributes.get("role_name", "").startswith(SCENE_ROLE_PREFIX)
    ]
    for actor in stale:
        actor.destroy()
    return len(stale)


def _camera_at_ego(world: carla.World, ego: carla.Vehicle) -> None:
    transform = ego.get_transform()
    forward = transform.get_forward_vector()
    camera_location = transform.location + carla.Location(
        x=forward.x * 0.75,
        y=forward.y * 0.75,
        z=1.55,
    )
    camera_rotation = carla.Rotation(
        pitch=-2.5, yaw=transform.rotation.yaw, roll=0.0
    )
    world.get_spectator().set_transform(carla.Transform(camera_location, camera_rotation))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--hold", type=float, default=0.0, help="保持进程的秒数，0 表示生成后退出")
    parser.add_argument("--keep", action="store_true", help="退出时保留车辆，便于继续在 CARLA 中观察")
    parser.add_argument("--dry-run", action="store_true", help="只检查车道与规划落点，不修改 CARLA")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    spectator_transform = world.get_spectator().get_transform()
    ego_wp = _choose_middle_lane(carla_map, spectator_transform)
    ego_wp = _align_with_nearest_street_light(world, ego_wp)
    chain = _lane_chain(ego_wp)
    ego_index = min(range(len(chain)), key=lambda index: chain[index].transform.location.distance(ego_wp.transform.location))
    if ego_index == 0 or ego_index == len(chain) - 1:
        raise RuntimeError("无法为自车同时找到左、右同向车道")
    left_wp = chain[ego_index - 1]
    right_wp = chain[ego_index + 1]
    samples = carla_map.generate_waypoints(3.0)

    planned: list[tuple[VehicleSpec, carla.Waypoint]] = []
    for spec in VEHICLES:
        if spec.lane == "ego":
            wp = _advance(ego_wp, spec.distance)
        elif spec.lane == "left":
            wp = _advance(left_wp, spec.distance)
        elif spec.lane == "right":
            wp = _advance(right_wp, spec.distance)
        else:
            wp = _opposite_waypoint(samples, ego_wp, spec.distance)
        planned.append((spec, wp))

    print(f"[00205] 当前地图: {carla_map.name}")
    print(
        f"[00205] 自车锚点: road={ego_wp.road_id} section={ego_wp.section_id} "
        f"lane={ego_wp.lane_id} xyz=({ego_wp.transform.location.x:.1f}, "
        f"{ego_wp.transform.location.y:.1f}, {ego_wp.transform.location.z:.1f})"
    )
    for spec, wp in planned:
        loc = wp.transform.location
        print(f"[00205] 规划 {spec.role:>13}: lane={wp.lane_id:>3} xyz=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
    if args.dry_run:
        return

    removed = _cleanup_previous(world)
    street_light_count = _set_night_and_street_lights(world)
    spawned: list[carla.Vehicle] = []
    actor_by_role: dict[str, carla.Vehicle] = {}
    try:
        for spec, wp in planned:
            transform = wp.transform
            transform.location.z += 0.35
            actor = world.try_spawn_actor(_blueprint(world, spec), transform)
            if actor is None:
                raise RuntimeError(f"车辆 {spec.role} 在规划位置生成失败")
            actor.set_simulate_physics(False)
            _vehicle_lights(actor, spec.braking)
            spawned.append(actor)
            actor_by_role[spec.role] = actor
        _camera_at_ego(world, actor_by_role["ego"])
        print(
            f"[00205] 完成: 生成 {len(spawned)} 辆车，开启 {street_light_count} 盏路灯，"
            f"清理旧版场景车 {removed} 辆。"
        )
        if args.hold > 0.0:
            print(f"[00205] 保持 {args.hold:.0f}s，Ctrl+C 可提前结束。")
            end = time.monotonic() + args.hold
            while time.monotonic() < end:
                time.sleep(min(0.5, end - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        if args.keep:
            print("[00205] 已按 --keep 将场景留在当前 CARLA 世界。")
        else:
            for actor in spawned:
                actor.destroy()


if __name__ == "__main__":
    main()
