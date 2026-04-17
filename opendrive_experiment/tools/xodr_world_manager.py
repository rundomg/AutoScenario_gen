import os
from typing import Any, Dict, Iterable, Optional


DEFAULT_GENERATION_PARAMS: Dict[str, Any] = {
    "vertex_distance": 2.0,
    "max_road_length": 50.0,
    "wall_height": 0.0,
    "additional_width": 1.5,
    "smooth_junctions": True,
    "enable_mesh_visibility": True,
}


def _import_carla_module():
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is required to load an OpenDRIVE world."
        ) from exc
    return carla


def _coerce_generation_params(
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized = dict(DEFAULT_GENERATION_PARAMS)
    if params:
        normalized.update(params)
    return normalized


def _build_generation_parameters(carla_module, params: Dict[str, Any]):
    return carla_module.OpendriveGenerationParameters(
        vertex_distance=float(params["vertex_distance"]),
        max_road_length=float(params["max_road_length"]),
        wall_height=float(params["wall_height"]),
        additional_width=float(params["additional_width"]),
        smooth_junctions=bool(params["smooth_junctions"]),
        enable_mesh_visibility=bool(params["enable_mesh_visibility"]),
    )


def _sample_spawn_points(
    spawn_points: Iterable[Any], limit: int
) -> list[Dict[str, Any]]:
    sampled = []
    for index, transform in enumerate(list(spawn_points)[:limit]):
        sampled.append(
            {
                "index": index,
                "location": {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                },
                "rotation": {
                    "pitch": float(transform.rotation.pitch),
                    "yaw": float(transform.rotation.yaw),
                    "roll": float(transform.rotation.roll),
                },
            }
        )
    return sampled


def _sample_topology(world_map, limit: int) -> list[Dict[str, Any]]:
    samples = []
    for start_waypoint, end_waypoint in list(world_map.get_topology())[:limit]:
        samples.append(
            {
                "road_id": int(start_waypoint.road_id),
                "lane_id": int(start_waypoint.lane_id),
                "start": {
                    "x": float(start_waypoint.transform.location.x),
                    "y": float(start_waypoint.transform.location.y),
                    "z": float(start_waypoint.transform.location.z),
                    "yaw": float(start_waypoint.transform.rotation.yaw),
                    "is_junction": bool(start_waypoint.is_junction),
                },
                "end": {
                    "x": float(end_waypoint.transform.location.x),
                    "y": float(end_waypoint.transform.location.y),
                    "z": float(end_waypoint.transform.location.z),
                    "yaw": float(end_waypoint.transform.rotation.yaw),
                    "is_junction": bool(end_waypoint.is_junction),
                },
            }
        )
    return samples


def load_xodr_world(
    xodr_path: str,
    carla_host: str = "localhost",
    carla_port: int = 2000,
    generation_params: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    spawn_point_limit: int = 12,
    topology_limit: int = 12,
    client_factory=None,
    carla_module=None,
) -> Dict[str, Any]:
    """Load an OpenDRIVE world in CARLA and return normalized spawn metadata."""
    if not os.path.exists(xodr_path):
        raise FileNotFoundError(f"OpenDRIVE file not found: {xodr_path}")

    if carla_module is None:
        carla_module = _import_carla_module()

    params_dict = _coerce_generation_params(generation_params)
    generation_parameters = _build_generation_parameters(carla_module, params_dict)

    client = (
        client_factory(carla_host, carla_port)
        if client_factory is not None
        else carla_module.Client(carla_host, carla_port)
    )
    client.set_timeout(timeout)

    with open(xodr_path, "r", encoding="utf-8") as file:
        xodr_content = file.read()
    if not xodr_content.strip():
        raise RuntimeError(f"OpenDRIVE file is empty: {xodr_path}")

    world = client.generate_opendrive_world(xodr_content, generation_parameters)
    try:
        world.wait_for_tick()
    except Exception:
        pass

    world_map = world.get_map()
    spawn_points = _sample_spawn_points(world_map.get_spawn_points(), spawn_point_limit)
    topology_sample = _sample_topology(world_map, topology_limit)

    return {
        "map_name": getattr(world_map, "name", "OpenDriveWorld"),
        "spawn_points": spawn_points,
        "topology_sample": topology_sample,
        "generation_params": params_dict,
        "xodr_path": xodr_path,
    }
