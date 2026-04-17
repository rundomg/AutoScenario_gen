"""Load an OpenDRIVE map into CARLA and focus the spectator on it.

Typical usage:

    conda run -n autoscenario python tools/view_xodr_in_carla.py \
        --xodr data/map_from_osm.xodr

This script expects the CARLA simulator server to already be running.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional, Sequence


DEFAULT_GENERATION_PARAMS = {
    "vertex_distance": 2.0,
    "max_road_length": 50.0,
    "wall_height": 0.0,
    "additional_width": 1.5,
    "smooth_junctions": True,
    "enable_mesh_visibility": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an OpenDRIVE map into CARLA and move the spectator to a top view."
    )
    parser.add_argument(
        "--xodr",
        default="data/map_from_osm.xodr",
        help="Path to the .xodr file to load. Default: data/map_from_osm.xodr",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host. Default: localhost")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port. Default: 2000")
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Client timeout in seconds. Default: 20.0",
    )
    parser.add_argument(
        "--spectator-height",
        type=float,
        default=220.0,
        help="Height in meters for the top-down spectator view. Default: 220.0",
    )
    parser.add_argument(
        "--spectator-pitch",
        type=float,
        default=-90.0,
        help="Spectator pitch angle in degrees. Default: -90.0",
    )
    parser.add_argument(
        "--spawn-demo-vehicle",
        action="store_true",
        help="Spawn a demo vehicle at the first available spawn point.",
    )
    parser.add_argument(
        "--vehicle-filter",
        default="vehicle.*",
        help="Blueprint filter for the demo vehicle. Default: vehicle.*",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the process alive after loading so you can inspect the map without rerunning.",
    )
    parser.add_argument(
        "--vertex-distance",
        type=float,
        default=float(DEFAULT_GENERATION_PARAMS["vertex_distance"]),
        help="CARLA OpenDRIVE mesh parameter vertex_distance.",
    )
    parser.add_argument(
        "--max-road-length",
        type=float,
        default=float(DEFAULT_GENERATION_PARAMS["max_road_length"]),
        help="CARLA OpenDRIVE mesh parameter max_road_length.",
    )
    parser.add_argument(
        "--wall-height",
        type=float,
        default=float(DEFAULT_GENERATION_PARAMS["wall_height"]),
        help="CARLA OpenDRIVE mesh parameter wall_height.",
    )
    parser.add_argument(
        "--additional-width",
        type=float,
        default=float(DEFAULT_GENERATION_PARAMS["additional_width"]),
        help="CARLA OpenDRIVE mesh parameter additional_width.",
    )
    parser.add_argument(
        "--disable-smooth-junctions",
        action="store_true",
        help="Disable smooth_junctions when generating the OpenDRIVE world.",
    )
    parser.add_argument(
        "--hide-mesh",
        action="store_true",
        help="Disable mesh visibility for the generated OpenDRIVE world.",
    )
    return parser.parse_args()


def _import_carla():
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not available. Please activate the CARLA-enabled environment."
        ) from exc
    return carla


def _build_generation_parameters(carla_module, args: argparse.Namespace):
    return carla_module.OpendriveGenerationParameters(
        vertex_distance=float(args.vertex_distance),
        max_road_length=float(args.max_road_length),
        wall_height=float(args.wall_height),
        additional_width=float(args.additional_width),
        smooth_junctions=not bool(args.disable_smooth_junctions),
        enable_mesh_visibility=not bool(args.hide_mesh),
    )


def _load_xodr_text(xodr_path: str) -> str:
    if not os.path.exists(xodr_path):
        raise FileNotFoundError(f"OpenDRIVE file not found: {xodr_path}")

    with open(xodr_path, "r", encoding="utf-8") as file:
        xodr_text = file.read()
    if not xodr_text.strip():
        raise RuntimeError(f"OpenDRIVE file is empty: {xodr_path}")
    return xodr_text


def _focus_spectator(
    carla_module,
    world,
    spawn_points: Sequence,
    topology: Sequence,
    height: float,
    pitch: float,
) -> Optional[object]:
    focus_locations: List[object] = []

    for transform in spawn_points[:20]:
        focus_locations.append(transform.location)

    if not focus_locations:
        for start_waypoint, _ in topology[:20]:
            focus_locations.append(start_waypoint.transform.location)

    if not focus_locations:
        return None

    avg_x = sum(location.x for location in focus_locations) / len(focus_locations)
    avg_y = sum(location.y for location in focus_locations) / len(focus_locations)
    max_z = max(location.z for location in focus_locations)

    spectator = world.get_spectator()
    transform = carla_module.Transform(
        carla_module.Location(x=avg_x, y=avg_y, z=max_z + height),
        carla_module.Rotation(pitch=pitch, yaw=0.0, roll=0.0),
    )
    spectator.set_transform(transform)
    return transform


def _spawn_demo_vehicle(
    world,
    spawn_points: Sequence,
    vehicle_filter: str,
):
    if not spawn_points:
        print("No spawn points available; skipped demo vehicle.")
        return None

    blueprint_library = world.get_blueprint_library()
    blueprints = blueprint_library.filter(vehicle_filter)
    if not blueprints:
        print(f"No blueprint matched '{vehicle_filter}'; skipped demo vehicle.")
        return None

    blueprint = blueprints[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")

    vehicle = world.try_spawn_actor(blueprint, spawn_points[0])
    if vehicle is None:
        print("Failed to spawn demo vehicle at the first spawn point.")
        return None

    try:
        vehicle.set_autopilot(True)
    except Exception:
        pass

    print(f"Spawned demo vehicle: {vehicle.type_id}")
    return vehicle


def main() -> None:
    args = _parse_args()
    carla = _import_carla()

    xodr_path = os.path.abspath(args.xodr)
    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.timeout))

    xodr_text = _load_xodr_text(xodr_path)
    generation_params = _build_generation_parameters(carla, args)

    print(f"Loading OpenDRIVE into CARLA from: {xodr_path}")
    try:
        world = client.generate_opendrive_world(xodr_text, generation_params)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load the OpenDRIVE world through CARLA at {args.host}:{args.port}. "
            "Please make sure the CARLA simulator is already running and fully initialized."
        ) from exc

    try:
        world.wait_for_tick()
    except Exception:
        time.sleep(1.0)

    world_map = world.get_map()
    spawn_points = list(world_map.get_spawn_points())
    topology = list(world_map.get_topology())

    try:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    except Exception:
        pass

    spectator_transform = _focus_spectator(
        carla_module=carla,
        world=world,
        spawn_points=spawn_points,
        topology=topology,
        height=float(args.spectator_height),
        pitch=float(args.spectator_pitch),
    )

    demo_vehicle = None
    if args.spawn_demo_vehicle:
        demo_vehicle = _spawn_demo_vehicle(world, spawn_points, args.vehicle_filter)

    print(f"Map name: {getattr(world_map, 'name', 'OpenDriveWorld')}")
    print(f"Spawn points: {len(spawn_points)}")
    print(f"Topology segments: {len(topology)}")
    if spectator_transform is not None:
        location = spectator_transform.location
        rotation = spectator_transform.rotation
        print(
            "Spectator transform: "
            f"x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}, "
            f"pitch={rotation.pitch:.2f}, yaw={rotation.yaw:.2f}"
        )

    if spawn_points:
        first = spawn_points[0]
        print(
            "First spawn point: "
            f"x={first.location.x:.2f}, y={first.location.y:.2f}, z={first.location.z:.2f}, "
            f"yaw={first.rotation.yaw:.2f}"
        )

    if not args.keep_alive:
        return

    print("Map is loaded. Press Ctrl+C to exit and keep CARLA open.")
    try:
        while True:
            time.sleep(1.0)
            if demo_vehicle is not None:
                try:
                    demo_vehicle.get_location()
                except Exception:
                    demo_vehicle = None
    except KeyboardInterrupt:
        print("Stopping viewer.")


if __name__ == "__main__":
    main()
