"""Load CARLA Town12 and optionally focus the spectator.

Typical usage:

    python tools/load_carla_town12.py

The CARLA simulator server must already be running.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, Optional


DEFAULT_MAP_CANDIDATES = (
    "Town12",
    "Town12HD",
    "Town12_Opt",
    "Town12HD_Opt",
    "Carla/Maps/Town12",
    "Carla/Maps/Town12HD",
    "Carla/Maps/Town12_Opt",
    "Carla/Maps/Town12HD_Opt",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a CARLA Town12 map in a running CARLA server."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host. Default: localhost")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port. Default: 2000")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Client timeout in seconds. Default: 30.0",
    )
    parser.add_argument(
        "--map",
        default="Town12",
        help="Map name to load. Default: Town12",
    )
    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="Print available map names before loading.",
    )
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="Only try the exact --map value instead of common Town12 variants.",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=0,
        help="Spawn point index used as the spectator focus. Default: 0.",
    )
    parser.add_argument(
        "--spectator-height",
        type=float,
        default=35.0,
        help="Spectator height above the selected spawn point in meters. Default: 35.0",
    )
    parser.add_argument(
        "--spectator-pitch",
        type=float,
        default=-65.0,
        help="Spectator pitch angle in degrees. Default: -65.0",
    )
    parser.add_argument(
        "--spectator-yaw",
        type=float,
        default=None,
        help="Spectator yaw angle in degrees. Default: selected spawn point yaw.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the script alive after loading so the map stays easy to inspect.",
    )
    return parser.parse_args()


def _import_carla():
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not available. Activate the environment that has carla installed."
        ) from exc
    return carla


def _candidate_map_names(requested_map: str, use_fallbacks: bool) -> Iterable[str]:
    seen = set()
    candidates = [requested_map]
    if use_fallbacks and requested_map.lower().replace("carla/maps/", "").startswith("town12"):
        candidates.extend(DEFAULT_MAP_CANDIDATES)

    for candidate in candidates:
        normalized = str(candidate or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def _load_world(client, map_names: Iterable[str]):
    last_error: Optional[Exception] = None
    for map_name in map_names:
        try:
            print(f"Loading CARLA map: {map_name}")
            return map_name, client.load_world(map_name)
        except Exception as exc:  # CARLA raises RuntimeError for missing maps.
            last_error = exc
            print(f"Failed to load {map_name}: {exc}")
    raise RuntimeError(f"Failed to load any requested Town12 map variant: {last_error}")


def _focus_spectator(
    carla_module,
    world,
    spawn_index: int,
    height: float,
    pitch: float,
    yaw: Optional[float],
) -> None:
    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    if spawn_points:
        clamped_index = max(0, min(int(spawn_index), len(spawn_points) - 1))
        spawn_transform = spawn_points[clamped_index]
        center = spawn_transform.location
        if yaw is None:
            yaw = float(spawn_transform.rotation.yaw)
        print(
            "Focusing spectator on spawn point "
            f"{clamped_index}: x={center.x:.2f}, y={center.y:.2f}, z={center.z:.2f}, yaw={yaw:.2f}"
        )
    else:
        clamped_index = None
        center = carla_module.Location(0.0, 0.0, 0.0)
        if yaw is None:
            yaw = 0.0
        print("No spawn points found; focusing spectator at world origin.")

    spectator = world.get_spectator()
    spectator.set_transform(
        carla_module.Transform(
            carla_module.Location(center.x, center.y, center.z + height),
            carla_module.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
        )
    )


def main() -> int:
    args = _parse_args()
    carla = _import_carla()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    if args.list_maps:
        print("Available CARLA maps:")
        for map_name in client.get_available_maps():
            print(f"  {map_name}")

    loaded_name, world = _load_world(
        client,
        _candidate_map_names(args.map, use_fallbacks=not args.no_fallbacks),
    )
    _focus_spectator(
        carla,
        world,
        spawn_index=args.spawn_index,
        height=args.spectator_height,
        pitch=args.spectator_pitch,
        yaw=args.spectator_yaw,
    )

    world_map = world.get_map()
    print(f"Loaded world: {world_map.name} via request {loaded_name}")
    print(f"Spawn points: {len(world_map.get_spawn_points())}")

    if args.keep_alive:
        print("Keeping process alive. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
