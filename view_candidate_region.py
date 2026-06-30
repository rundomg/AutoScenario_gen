"""Focus the CARLA spectator on the selected cached map-match candidate.

Default candidate:
    Carla/Maps/Town06, location=(36.25, 48.86, 0.0), yaw=-0.1

Typical usage:
    python view_candidate_region.py
    python view_candidate_region.py --view oblique
    python view_candidate_region.py --no-load-map

The CARLA simulator server must already be running.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable, Optional


DEFAULT_WORLD = "Carla/Maps/Town06"
DEFAULT_X = 36.25
DEFAULT_Y = 48.86
DEFAULT_Z = 0.0
DEFAULT_YAW = -0.1
DEFAULT_ROAD_ID = 15
DEFAULT_LANE_ID = -6

DEFAULT_LANE_START = (36.25, 48.86, 0.0)
DEFAULT_LANE_END = (56.25, 48.82, 0.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load/focus CARLA on the most similar cached candidate region."
    )
    parser.add_argument("--host", default="localhost", help="CARLA host. Default: localhost")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port. Default: 2000")
    parser.add_argument("--timeout", type=float, default=30.0, help="CARLA timeout seconds.")
    parser.add_argument("--map", default=DEFAULT_WORLD, help=f"CARLA map. Default: {DEFAULT_WORLD}")
    parser.add_argument(
        "--no-load-map",
        action="store_true",
        help="Use the currently loaded world instead of calling client.load_world().",
    )
    parser.add_argument("--list-maps", action="store_true", help="Print available maps first.")
    parser.add_argument("--x", type=float, default=DEFAULT_X, help="Target x coordinate.")
    parser.add_argument("--y", type=float, default=DEFAULT_Y, help="Target y coordinate.")
    parser.add_argument("--z", type=float, default=DEFAULT_Z, help="Target z coordinate.")
    parser.add_argument("--yaw", type=float, default=DEFAULT_YAW, help="Road heading/yaw.")
    parser.add_argument(
        "--view",
        choices=("top", "oblique", "ego"),
        default="top",
        help="Spectator view mode. Default: top.",
    )
    parser.add_argument("--height", type=float, default=None, help="Override spectator height.")
    parser.add_argument("--pitch", type=float, default=None, help="Override spectator pitch.")
    parser.add_argument(
        "--marker-life",
        type=float,
        default=120.0,
        help="Debug marker lifetime in seconds. Default: 120.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Refresh markers until Ctrl+C.",
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


def _candidate_map_names(requested_map: str) -> Iterable[str]:
    requested = str(requested_map or "").strip()
    base = requested.replace("Carla/Maps/", "")
    candidates = [
        requested,
        base,
        f"Carla/Maps/{base}",
        f"{base}HD",
        f"Carla/Maps/{base}HD",
        f"{base}_Opt",
        f"Carla/Maps/{base}_Opt",
    ]
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _load_world(client, requested_map: str):
    last_error: Optional[Exception] = None
    for map_name in _candidate_map_names(requested_map):
        try:
            print(f"Loading CARLA map: {map_name}")
            return client.load_world(map_name)
        except Exception as exc:
            last_error = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"Could not load requested map variants for {requested_map}: {last_error}")


def _resolve_target(carla, world, x: float, y: float, z: float):
    raw_location = carla.Location(x=x, y=y, z=z)
    try:
        waypoint = world.get_map().get_waypoint(raw_location, project_to_road=True)
    except Exception:
        waypoint = None
    if waypoint is None:
        return raw_location, None
    location = waypoint.transform.location
    return carla.Location(location.x, location.y, location.z + 0.15), waypoint


def _spectator_transform(carla, center, yaw: float, view: str, height: Optional[float], pitch: Optional[float]):
    if view == "ego":
        distance = 9.0
        actual_height = 3.0 if height is None else height
        actual_pitch = -8.0 if pitch is None else pitch
        radians = math.radians(yaw)
        location = carla.Location(
            x=center.x - math.cos(radians) * distance,
            y=center.y - math.sin(radians) * distance,
            z=center.z + actual_height,
        )
        rotation = carla.Rotation(pitch=actual_pitch, yaw=yaw, roll=0.0)
        return carla.Transform(location, rotation)

    if view == "oblique":
        distance = 45.0
        actual_height = 28.0 if height is None else height
        actual_pitch = -35.0 if pitch is None else pitch
        camera_yaw = yaw + 180.0
        radians = math.radians(yaw)
        location = carla.Location(
            x=center.x - math.cos(radians) * distance,
            y=center.y - math.sin(radians) * distance,
            z=center.z + actual_height,
        )
        rotation = carla.Rotation(pitch=actual_pitch, yaw=camera_yaw, roll=0.0)
        return carla.Transform(location, rotation)

    actual_height = 85.0 if height is None else height
    actual_pitch = -88.0 if pitch is None else pitch
    location = carla.Location(center.x, center.y, center.z + actual_height)
    rotation = carla.Rotation(pitch=actual_pitch, yaw=yaw, roll=0.0)
    return carla.Transform(location, rotation)


def _draw_circle(carla, world, center, radius: float, color, life_time: float) -> None:
    points = []
    for index in range(48):
        angle = 2.0 * math.pi * index / 48.0
        points.append(
            carla.Location(
                x=center.x + radius * math.cos(angle),
                y=center.y + radius * math.sin(angle),
                z=center.z + 0.35,
            )
        )
    for start, end in zip(points, points[1:] + points[:1]):
        world.debug.draw_line(start, end, thickness=0.08, color=color, life_time=life_time)


def _draw_markers(carla, world, center, yaw: float, waypoint, life_time: float) -> None:
    red = carla.Color(255, 40, 40)
    green = carla.Color(40, 220, 80)
    blue = carla.Color(50, 140, 255)
    yellow = carla.Color(255, 220, 40)
    white = carla.Color(255, 255, 255)

    world.debug.draw_point(center, size=0.35, color=red, life_time=life_time)
    world.debug.draw_string(
        carla.Location(center.x, center.y, center.z + 2.5),
        f"Best cached candidate: Town06 road {DEFAULT_ROAD_ID}, lane {DEFAULT_LANE_ID}",
        draw_shadow=True,
        color=yellow,
        life_time=life_time,
    )

    start = carla.Location(*DEFAULT_LANE_START)
    end = carla.Location(*DEFAULT_LANE_END)
    start.z = center.z + 0.45
    end.z = center.z + 0.45
    world.debug.draw_line(start, end, thickness=0.25, color=green, life_time=life_time)
    try:
        world.debug.draw_arrow(start, end, thickness=0.2, arrow_size=1.2, color=green, life_time=life_time)
    except TypeError:
        pass

    radians = math.radians(yaw)
    forward = carla.Location(
        x=center.x + math.cos(radians) * 18.0,
        y=center.y + math.sin(radians) * 18.0,
        z=center.z + 0.8,
    )
    world.debug.draw_line(center, forward, thickness=0.18, color=blue, life_time=life_time)

    _draw_circle(carla, world, center, radius=15.0, color=white, life_time=life_time)
    _draw_circle(carla, world, center, radius=35.0, color=yellow, life_time=life_time)

    if waypoint is not None:
        lane_text = (
            f"snapped waypoint: road={waypoint.road_id}, lane={waypoint.lane_id}, "
            f"junction={waypoint.is_junction}"
        )
        world.debug.draw_string(
            carla.Location(center.x, center.y, center.z + 4.2),
            lane_text,
            draw_shadow=True,
            color=white,
            life_time=life_time,
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

    world = client.get_world() if args.no_load_map else _load_world(client, args.map)
    try:
        world.tick()
    except Exception:
        world.wait_for_tick()

    center, waypoint = _resolve_target(carla, world, args.x, args.y, args.z)
    spectator = world.get_spectator()
    spectator_transform = _spectator_transform(
        carla,
        center,
        yaw=float(args.yaw),
        view=args.view,
        height=args.height,
        pitch=args.pitch,
    )
    spectator.set_transform(spectator_transform)
    _draw_markers(carla, world, center, float(args.yaw), waypoint, float(args.marker_life))

    world_name = world.get_map().name
    print(f"World: {world_name}")
    print(f"Focused candidate center: x={center.x:.2f}, y={center.y:.2f}, z={center.z:.2f}")
    print(f"Candidate heading yaw: {float(args.yaw):.2f}")
    if waypoint is not None:
        print(
            "Snapped waypoint: "
            f"road_id={waypoint.road_id}, lane_id={waypoint.lane_id}, "
            f"is_junction={waypoint.is_junction}"
        )
    print(
        "Open the CARLA/Unreal viewport to inspect the highlighted region. "
        "Use --view oblique or --view ego for other angles."
    )

    if args.keep_alive:
        print("Refreshing debug markers. Press Ctrl+C to stop.")
        try:
            while True:
                _draw_markers(carla, world, center, float(args.yaw), waypoint, 2.0)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
