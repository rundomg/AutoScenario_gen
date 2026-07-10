"""Minimal CARLA collision smoke test.

Spawns two vehicles on the same lane, holds the front vehicle still, drives the
rear vehicle into it, and writes a small collision report JSON.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import carla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple CARLA collision smoke test.")
    parser.add_argument("--host", default="localhost", help="CARLA host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="CARLA RPC timeout seconds.")
    parser.add_argument("--map", default="", help="Optional map to load, e.g. Town03.")
    parser.add_argument("--duration", type=float, default=8.0, help="Scenario duration seconds.")
    parser.add_argument("--tick-dt", type=float, default=0.05, help="Synchronous fixed delta seconds.")
    parser.add_argument("--gap", type=float, default=14.0, help="Initial gap between vehicles in metres.")
    parser.add_argument("--speed", type=float, default=12.0, help="Rear vehicle target speed in m/s.")
    parser.add_argument(
        "--drive-mode",
        choices=("control", "velocity"),
        default="control",
        help="Use realistic VehicleControl speed tracking or direct target velocity.",
    )
    parser.add_argument(
        "--rear-throttle",
        type=float,
        default=1.0,
        help="Maximum rear vehicle throttle when --drive-mode control is used.",
    )
    parser.add_argument(
        "--rear-min-throttle",
        type=float,
        default=0.4,
        help="Minimum rear throttle while it is below target speed.",
    )
    parser.add_argument(
        "--rear-speed-gain",
        type=float,
        default=0.16,
        help="Rear speed-controller gain from speed error to throttle.",
    )
    parser.add_argument(
        "--front-speed",
        type=float,
        default=4.0,
        help="Front vehicle cruising speed before braking; use 0 for a static target.",
    )
    parser.add_argument(
        "--front-blueprint",
        default="vehicle.diamondback.century",
        help=(
            "Front actor blueprint. Default is a bicycle; common alternatives "
            "include vehicle.bh.crossbike and vehicle.gazelle.omafiets."
        ),
    )
    parser.add_argument(
        "--front-direction",
        choices=("opposite", "same"),
        default="opposite",
        help="Whether the front actor faces/drives opposite to the rear actor or in the same lane direction.",
    )
    parser.add_argument(
        "--front-brake-after",
        type=float,
        default=2.0,
        help="Seconds before the front vehicle brakes to create a rear-end case.",
    )
    parser.add_argument(
        "--spawn-z-offset",
        type=float,
        default=0.05,
        help="Small spawn height offset above the waypoint transform.",
    )
    parser.add_argument(
        "--post-collision-hold",
        type=float,
        default=4.0,
        help="Seconds to keep the collision visible before cleanup.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run simulation ticks as fast as possible instead of sleeping per tick.",
    )
    parser.add_argument(
        "--spectator-distance",
        type=float,
        default=18.0,
        help="Spectator chase distance in metres.",
    )
    parser.add_argument(
        "--spectator-height",
        type=float,
        default=7.0,
        help="Spectator height above the vehicles in metres.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "carla_collision_smoke.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--keep-actors",
        action="store_true",
        help="Do not destroy spawned actors after the test.",
    )
    return parser.parse_args()


def vehicle_speed(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def distance_xy(a: carla.Actor, b: carla.Actor) -> float:
    loc_a = a.get_transform().location
    loc_b = b.get_transform().location
    return math.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.y - loc_b.y) ** 2)


def lift_transform(transform: carla.Transform, dz: float = 0.05) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            transform.location.x,
            transform.location.y,
            transform.location.z + dz,
        ),
        transform.rotation,
    )


def reverse_transform(transform: carla.Transform) -> carla.Transform:
    return carla.Transform(
        transform.location,
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw + 180.0,
            roll=transform.rotation.roll,
        ),
    )


def yaw_diff_deg(a: float, b: float) -> float:
    diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(diff)


def choose_straight_pair(
    world: carla.World,
    gap_m: float,
    spawn_z_offset: float,
) -> Optional[Tuple[carla.Transform, carla.Transform]]:
    carla_map = world.get_map()
    for spawn_transform in carla_map.get_spawn_points():
        try:
            rear_wp = carla_map.get_waypoint(
                spawn_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            front_candidates = rear_wp.next(float(gap_m))
        except Exception:
            continue
        if not front_candidates:
            continue
        front_wp = front_candidates[0]
        if yaw_diff_deg(rear_wp.transform.rotation.yaw, front_wp.transform.rotation.yaw) > 8.0:
            continue
        return (
            lift_transform(rear_wp.transform, spawn_z_offset),
            lift_transform(front_wp.transform, spawn_z_offset),
        )
    return None


def find_vehicle_blueprint(
    blueprint_library: carla.BlueprintLibrary,
    preferred: str,
    role_name: str,
    fallbacks: Optional[List[str]] = None,
) -> carla.ActorBlueprint:
    blueprints = list(blueprint_library.filter(preferred))
    for fallback in fallbacks or []:
        if blueprints:
            break
        blueprints = list(blueprint_library.filter(fallback))
    if not blueprints:
        blueprints = list(blueprint_library.filter("vehicle.*"))
    if not blueprints:
        raise RuntimeError("No vehicle blueprints found in this CARLA world.")
    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def spawn_test_vehicles(
    world: carla.World,
    gap_m: float,
    spawn_z_offset: float,
    front_blueprint: str,
    front_direction: str,
) -> Tuple[carla.Vehicle, carla.Vehicle]:
    pair = choose_straight_pair(world, gap_m, spawn_z_offset)
    if pair is None:
        raise RuntimeError("Could not find a straight drivable lane segment.")
    rear_tf, front_tf = pair
    if front_direction == "opposite":
        front_tf = reverse_transform(front_tf)
    blueprints = world.get_blueprint_library()
    ego_bp = find_vehicle_blueprint(blueprints, "vehicle.tesla.model3", "hero")
    front_bp = find_vehicle_blueprint(
        blueprints,
        front_blueprint,
        "collision_target",
        fallbacks=[
            "vehicle.bh.crossbike",
            "vehicle.gazelle.omafiets",
            "vehicle.diamondback.century",
        ],
    )

    front = world.try_spawn_actor(front_bp, front_tf)
    if front is None:
        raise RuntimeError("Failed to spawn front vehicle.")
    ego = world.try_spawn_actor(ego_bp, rear_tf)
    if ego is None:
        front.destroy()
        raise RuntimeError("Failed to spawn rear/ego vehicle.")
    return ego, front


def hold_stationary(actor: carla.Actor) -> None:
    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))


def brake_vehicle(actor: carla.Actor, hand_brake: bool = False) -> None:
    actor.apply_control(
        carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0, hand_brake=hand_brake)
    )


def drive_forward_velocity(actor: carla.Actor, target_speed_mps: float) -> None:
    transform = actor.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    actor.set_target_velocity(
        carla.Vector3D(
            math.cos(yaw) * float(target_speed_mps),
            math.sin(yaw) * float(target_speed_mps),
            0.0,
        )
    )
    actor.apply_control(carla.VehicleControl(throttle=0.75, brake=0.0, steer=0.0))


def drive_forward_control(
    actor: carla.Actor,
    target_speed_mps: float,
    max_throttle: float,
    min_throttle: float = 0.18,
    speed_gain: float = 0.08,
) -> None:
    current_speed = vehicle_speed(actor)
    error = float(target_speed_mps) - current_speed
    if error > 0.35:
        throttle = min(
            max(0.0, float(max_throttle)),
            max(max(0.0, float(min_throttle)), error * float(speed_gain)),
        )
        brake = 0.0
    elif error < -0.35:
        throttle = 0.0
        brake = min(0.8, abs(error) * 0.14)
    else:
        throttle = 0.0
        brake = 0.0
    actor.apply_control(
        carla.VehicleControl(throttle=throttle, brake=brake, steer=0.0, hand_brake=False)
    )


def drive_forward(
    actor: carla.Actor,
    target_speed_mps: float,
    drive_mode: str,
    max_throttle: float,
    min_throttle: float = 0.18,
    speed_gain: float = 0.08,
) -> None:
    if drive_mode == "velocity":
        drive_forward_velocity(actor, target_speed_mps)
        return
    drive_forward_control(
        actor,
        target_speed_mps,
        max_throttle=max_throttle,
        min_throttle=min_throttle,
        speed_gain=speed_gain,
    )


def control_front_vehicle(front: carla.Actor, elapsed_s: float, args: argparse.Namespace) -> None:
    if float(args.front_speed) <= 0.0:
        hold_stationary(front)
        return
    if elapsed_s >= float(args.front_brake_after):
        brake_vehicle(front)
        return
    drive_forward(
        front,
        float(args.front_speed),
        str(args.drive_mode),
        max_throttle=min(0.45, float(args.rear_throttle)),
        min_throttle=0.18,
        speed_gain=0.08,
    )


def sleep_for_tick(args: argparse.Namespace) -> None:
    if args.no_realtime:
        return
    time.sleep(max(0.0, float(args.tick_dt)))


def update_spectator(
    world: carla.World,
    ego: Optional[carla.Actor],
    front: Optional[carla.Actor],
    distance_m: float,
    height_m: float,
) -> None:
    if ego is None or front is None:
        return
    try:
        ego_tf = ego.get_transform()
        front_loc = front.get_transform().location
        yaw_rad = math.radians(float(ego_tf.rotation.yaw))
        mid_x = (ego_tf.location.x + front_loc.x) * 0.5
        mid_y = (ego_tf.location.y + front_loc.y) * 0.5
        mid_z = max(ego_tf.location.z, front_loc.z)
        spectator = world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(
                    x=mid_x - math.cos(yaw_rad) * float(distance_m),
                    y=mid_y - math.sin(yaw_rad) * float(distance_m),
                    z=mid_z + float(height_m),
                ),
                carla.Rotation(pitch=-20.0, yaw=ego_tf.rotation.yaw, roll=0.0),
            )
        )
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    client = carla.Client(args.host, int(args.port))
    client.set_timeout(float(args.timeout))
    world = client.load_world(args.map) if args.map else client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = float(args.tick_dt)
    world.apply_settings(settings)

    actors: List[carla.Actor] = []
    collisions: List[Dict[str, object]] = []
    min_distance_m = float("inf")
    sensor = None
    ego = None
    front = None
    try:
        ego, front = spawn_test_vehicles(
            world,
            float(args.gap),
            float(args.spawn_z_offset),
            str(args.front_blueprint),
            str(args.front_direction),
        )
        actors.extend([ego, front])
        update_spectator(
            world,
            ego,
            front,
            float(args.spectator_distance),
            float(args.spectator_height),
        )

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        actors.append(sensor)

        def on_collision(event: carla.CollisionEvent) -> None:
            other = event.other_actor
            impulse = event.normal_impulse
            impulse_norm = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            collisions.append(
                {
                    "frame": int(event.frame),
                    "other_actor_id": int(other.id) if other is not None else None,
                    "other_actor_type": other.type_id if other is not None else None,
                    "normal_impulse": impulse_norm,
                }
            )

        sensor.listen(on_collision)

        for _ in range(20):
            hold_stationary(ego)
            hold_stationary(front)
            update_spectator(
                world,
                ego,
                front,
                float(args.spectator_distance),
                float(args.spectator_height),
            )
            world.tick()
            sleep_for_tick(args)

        max_ticks = max(1, int(float(args.duration) / max(float(args.tick_dt), 0.01)))
        start_time = time.time()
        collision_tick = None
        for tick_index in range(max_ticks):
            elapsed_s = tick_index * float(args.tick_dt)
            control_front_vehicle(front, elapsed_s, args)
            drive_forward(
                ego,
                float(args.speed),
                str(args.drive_mode),
                max_throttle=float(args.rear_throttle),
                min_throttle=float(args.rear_min_throttle),
                speed_gain=float(args.rear_speed_gain),
            )
            update_spectator(
                world,
                ego,
                front,
                float(args.spectator_distance),
                float(args.spectator_height),
            )
            world.tick()
            sleep_for_tick(args)
            min_distance_m = min(min_distance_m, distance_xy(ego, front))
            if collisions and tick_index > 5:
                collision_tick = tick_index
                break

        hold_ticks = max(
            0,
            int(float(args.post_collision_hold) / max(float(args.tick_dt), 0.01)),
        )
        for _ in range(hold_ticks):
            brake_vehicle(ego)
            brake_vehicle(front)
            update_spectator(
                world,
                ego,
                front,
                float(args.spectator_distance),
                float(args.spectator_height),
            )
            world.tick()
            sleep_for_tick(args)

        elapsed_wall_s = time.time() - start_time
        report = {
            "collision": bool(collisions),
            "collision_events": collisions,
            "ego_actor_id": int(ego.id),
            "target_actor_id": int(front.id),
            "map": world.get_map().name,
            "gap_m": float(args.gap),
            "target_speed_mps": float(args.speed),
            "drive_mode": str(args.drive_mode),
            "rear_throttle": float(args.rear_throttle),
            "rear_min_throttle": float(args.rear_min_throttle),
            "rear_speed_gain": float(args.rear_speed_gain),
            "front_speed_mps": float(args.front_speed),
            "front_brake_after_s": float(args.front_brake_after),
            "front_blueprint": str(args.front_blueprint),
            "front_direction": str(args.front_direction),
            "spawn_z_offset_m": float(args.spawn_z_offset),
            "duration_s": float(args.duration),
            "tick_dt": float(args.tick_dt),
            "collision_tick": collision_tick,
            "post_collision_hold_s": float(args.post_collision_hold),
            "realtime": not bool(args.no_realtime),
            "min_distance_m": None if min_distance_m == float("inf") else min_distance_m,
            "elapsed_wall_s": elapsed_wall_s,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if collisions else 2
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        if not args.keep_actors:
            for actor in reversed(actors):
                try:
                    actor.destroy()
                except Exception:
                    pass
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
