"""CARLA-map constrained reference paths and dependency-free polyline geometry."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import ActorState, Point2D, ReferencePath, clamp


class MapConstrainedPathBuilder:
    """Build a route from CARLA waypoints, with a deterministic offline fallback."""

    def __init__(
        self,
        carla_map: Any = None,
        *,
        step_m: float = 2.0,
        horizon_m: float = 80.0,
    ) -> None:
        self.carla_map = carla_map
        self.step_m = max(0.5, float(step_m))
        self.horizon_m = max(10.0, float(horizon_m))

    def build(self, actor: ActorState, maneuver: str = "keep_lane") -> ReferencePath:
        if self.carla_map is not None:
            path = self._build_from_carla(actor, maneuver)
            if path is not None and path.length > self.step_m:
                return path
        return self._build_fallback(actor, maneuver)

    def _build_from_carla(
        self, actor: ActorState, maneuver: str
    ) -> Optional[ReferencePath]:
        location = _carla_location(actor.x, actor.y)
        if location is None:
            return None
        try:
            waypoint = self.carla_map.get_waypoint(location, project_to_road=True)
        except Exception:
            return None
        if waypoint is None:
            return None

        points: List[Point2D] = []
        lane_ids: List[Optional[int]] = []
        road_ids: List[Optional[int]] = []
        current = waypoint
        travelled = 0.0
        branch_chosen = False
        target_lane = None
        transition_distance = 18.0
        while current is not None and travelled <= self.horizon_m:
            point = _waypoint_point(current)
            if not points or distance(points[-1], point) > 0.05:
                points.append(point)
                lane_ids.append(_safe_int(getattr(current, "lane_id", None)))
                road_ids.append(_safe_int(getattr(current, "road_id", None)))
            try:
                candidates = list(current.next(self.step_m) or [])
            except Exception:
                candidates = []
            if not candidates:
                break

            if maneuver.startswith("lane_change") and travelled >= transition_distance * 0.4:
                if target_lane is None:
                    getter_name = "get_left_lane" if "left" in maneuver else "get_right_lane"
                    getter = getattr(current, getter_name, None)
                    try:
                        target_lane = getter() if callable(getter) else None
                    except Exception:
                        target_lane = None
                if target_lane is not None:
                    try:
                        target_candidates = list(target_lane.next(self.step_m) or [])
                    except Exception:
                        target_candidates = []
                    if target_candidates:
                        target_lane = target_candidates[0]
                        alpha = clamp(
                            (travelled - transition_distance * 0.4)
                            / max(transition_distance * 0.6, 0.1),
                            0.0,
                            1.0,
                        )
                        source_point = _waypoint_point(candidates[0])
                        target_point = _waypoint_point(target_lane)
                        blended = (
                            source_point[0] * (1.0 - alpha) + target_point[0] * alpha,
                            source_point[1] * (1.0 - alpha) + target_point[1] * alpha,
                        )
                        points.append(blended)
                        lane_ids.append(_safe_int(getattr(target_lane, "lane_id", None)))
                        road_ids.append(_safe_int(getattr(target_lane, "road_id", None)))
                        current = target_lane if alpha >= 0.99 else candidates[0]
                        travelled += self.step_m
                        continue

            if maneuver in {"turn_left", "turn_right", "straight", "u_turn"} and len(candidates) > 1 and not branch_chosen:
                current = _choose_branch(current, candidates, maneuver)
                branch_chosen = True
            else:
                current = candidates[0]
            travelled += self.step_m

        if len(points) < 2:
            return None
        return make_reference_path(
            actor.actor_id,
            points,
            lane_ids=lane_ids,
            road_ids=road_ids,
            maneuver=maneuver,
            source="carla_waypoint_topology",
        )

    def _build_fallback(self, actor: ActorState, maneuver: str) -> ReferencePath:
        yaw = math.radians(actor.yaw_deg)
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        points: List[Point2D] = []
        count = int(self.horizon_m / self.step_m) + 1
        for index in range(count):
            s = index * self.step_m
            lateral = 0.0
            if maneuver.startswith("lane_change"):
                direction = -1.0 if "left" in maneuver else 1.0
                progress = clamp((s - 6.0) / 18.0, 0.0, 1.0)
                # Cubic smoothstep avoids a steering discontinuity at both ends.
                progress = progress * progress * (3.0 - 2.0 * progress)
                lateral = direction * 3.5 * progress
            elif maneuver in {"turn_left", "turn_right"} and s > 8.0:
                direction = -1.0 if maneuver == "turn_left" else 1.0
                theta = direction * min(math.pi / 2.0, (s - 8.0) / 15.0)
                radius = 15.0
                local_forward = 8.0 + radius * math.sin(abs(theta))
                lateral = direction * radius * (1.0 - math.cos(theta))
                points.append(
                    (
                        actor.x + forward[0] * local_forward + right[0] * lateral,
                        actor.y + forward[1] * local_forward + right[1] * lateral,
                    )
                )
                continue
            points.append(
                (
                    actor.x + forward[0] * s + right[0] * lateral,
                    actor.y + forward[1] * s + right[1] * lateral,
                )
            )
        return make_reference_path(
            actor.actor_id,
            points,
            lane_ids=[actor.lane_id] * len(points),
            road_ids=[actor.road_id] * len(points),
            maneuver=maneuver,
            source="fallback_heading",
        )


def make_reference_path(
    actor_id: str,
    points: Iterable[Sequence[float]],
    *,
    lane_ids: Optional[List[Optional[int]]] = None,
    road_ids: Optional[List[Optional[int]]] = None,
    maneuver: str = "keep_lane",
    source: str = "polyline",
) -> ReferencePath:
    clean = [(float(point[0]), float(point[1])) for point in points]
    if not clean:
        raise ValueError("Reference path must contain at least one waypoint.")
    cumulative = [0.0]
    for previous, current in zip(clean, clean[1:]):
        cumulative.append(cumulative[-1] + distance(previous, current))
    return ReferencePath(
        actor_id=str(actor_id),
        waypoints=clean,
        cumulative_distances=cumulative,
        lane_ids=list(lane_ids or [None] * len(clean)),
        road_ids=list(road_ids or [None] * len(clean)),
        maneuver=maneuver,
        source=source,
    )


def point_at(path: ReferencePath, path_distance: float) -> Point2D:
    if not path.waypoints:
        raise ValueError("Cannot query an empty reference path.")
    if len(path.waypoints) == 1 or path_distance <= 0.0:
        return path.waypoints[0]
    s = min(float(path_distance), path.length)
    for index in range(1, len(path.cumulative_distances)):
        if path.cumulative_distances[index] < s:
            continue
        lower_s, upper_s = path.cumulative_distances[index - 1 : index + 1]
        ratio = 0.0 if upper_s <= lower_s else (s - lower_s) / (upper_s - lower_s)
        first, second = path.waypoints[index - 1 : index + 1]
        return (
            first[0] + ratio * (second[0] - first[0]),
            first[1] + ratio * (second[1] - first[1]),
        )
    return path.waypoints[-1]


def heading_at(path: ReferencePath, path_distance: float) -> float:
    if len(path.waypoints) < 2:
        return 0.0
    s = clamp(path_distance, 0.0, path.length)
    index = 1
    while index < len(path.cumulative_distances) and path.cumulative_distances[index] < s:
        index += 1
    index = min(index, len(path.waypoints) - 1)
    first, second = path.waypoints[index - 1], path.waypoints[index]
    return math.atan2(second[1] - first[1], second[0] - first[0])


def project_onto_path(path: ReferencePath, point: Sequence[float]) -> Tuple[float, float, Point2D]:
    """Return along-path s, Euclidean residual, and closest point."""

    if len(path.waypoints) == 1:
        only = path.waypoints[0]
        return 0.0, distance(only, point), only
    best = (0.0, float("inf"), path.waypoints[0])
    px, py = float(point[0]), float(point[1])
    for index, (first, second) in enumerate(zip(path.waypoints, path.waypoints[1:])):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_sq = dx * dx + dy * dy
        ratio = 0.0 if length_sq <= 1e-12 else clamp(
            ((px - first[0]) * dx + (py - first[1]) * dy) / length_sq, 0.0, 1.0
        )
        closest = (first[0] + ratio * dx, first[1] + ratio * dy)
        residual = distance(closest, (px, py))
        along = path.cumulative_distances[index] + ratio * math.sqrt(length_sq)
        if residual < best[1]:
            best = (along, residual, closest)
    return best


def resample_path(path: ReferencePath, step_m: float = 0.5) -> List[Tuple[float, Point2D]]:
    step = max(0.1, float(step_m))
    samples = []
    s = 0.0
    while s < path.length:
        samples.append((s, point_at(path, s)))
        s += step
    samples.append((path.length, point_at(path, path.length)))
    return samples


def distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))


def obb_separation(
    center_a: Point2D,
    yaw_a: float,
    length_a: float,
    width_a: float,
    center_b: Point2D,
    yaw_b: float,
    length_b: float,
    width_b: float,
) -> float:
    """SAT separation: <=0 means the two oriented rectangles overlap."""

    axes = [
        (math.cos(yaw_a), math.sin(yaw_a)),
        (-math.sin(yaw_a), math.cos(yaw_a)),
        (math.cos(yaw_b), math.sin(yaw_b)),
        (-math.sin(yaw_b), math.cos(yaw_b)),
    ]
    delta = (center_b[0] - center_a[0], center_b[1] - center_a[1])
    maximum_gap = -float("inf")
    for axis in axes:
        center_projection = abs(delta[0] * axis[0] + delta[1] * axis[1])
        radius_a = _obb_projection_radius(yaw_a, length_a, width_a, axis)
        radius_b = _obb_projection_radius(yaw_b, length_b, width_b, axis)
        maximum_gap = max(maximum_gap, center_projection - radius_a - radius_b)
    return maximum_gap


def _obb_projection_radius(yaw, length, width, axis):
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    return (
        0.5 * length * abs(forward[0] * axis[0] + forward[1] * axis[1])
        + 0.5 * width * abs(right[0] * axis[0] + right[1] * axis[1])
    )


def _choose_branch(current: Any, candidates: List[Any], maneuver: str) -> Any:
    current_yaw = _waypoint_yaw(current)
    desired = {"turn_left": -90.0, "turn_right": 90.0, "straight": 0.0, "u_turn": 180.0}[maneuver]
    return min(
        candidates,
        key=lambda candidate: abs(_normalize_angle(_waypoint_yaw(candidate) - current_yaw) - desired),
    )


def _waypoint_point(waypoint: Any) -> Point2D:
    location = waypoint.transform.location
    return (float(location.x), float(location.y))


def _waypoint_yaw(waypoint: Any) -> float:
    try:
        return float(waypoint.transform.rotation.yaw)
    except Exception:
        return 0.0


def _normalize_angle(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def _carla_location(x: float, y: float) -> Any:
    try:
        import carla  # type: ignore

        return carla.Location(x=float(x), y=float(y), z=0.3)
    except (ImportError, AttributeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
