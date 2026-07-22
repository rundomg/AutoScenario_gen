"""Analytical initialisation and a deterministic kinematic rollout backend."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .models import (
    AccidentSpecification,
    ActorBehaviorParameters,
    CollisionAnchor,
    ReferencePath,
    RolloutResult,
    SceneState,
    clamp,
)
from .paths import heading_at, obb_separation, point_at


SPEED_LEVEL_FRACTIONS = {
    "stopped": (0.0, 0.0),
    "low": (0.20, 0.45),
    "medium": (0.40, 0.75),
    "high": (0.70, 1.00),
}


def speed_interval(
    level: str,
    speed_limit_mps: float,
    *,
    category: str = "vehicle",
) -> Tuple[float, float]:
    """Map a qualitative VLM speed to a road- and actor-bounded interval."""

    low_fraction, high_fraction = SPEED_LEVEL_FRACTIONS.get(
        str(level or "medium").lower(), SPEED_LEVEL_FRACTIONS["medium"]
    )
    limit = max(2.0, float(speed_limit_mps))
    if str(category).lower() in {"truck", "bus"}:
        limit = min(limit, 16.7)
    elif str(category).lower() in {"bicycle"}:
        limit = min(limit, 8.0)
    return (round(limit * low_fraction, 3), round(limit * high_fraction, 3))


def analytical_initial_estimate(
    scene: SceneState,
    spec: AccidentSpecification,
    anchor: CollisionAnchor,
) -> Dict[str, ActorBehaviorParameters]:
    """Obtain a physically bounded initial guess before any expensive rollout."""

    target_time = max(0.2, sum(anchor.target_time_range) / 2.0)
    parameters: Dict[str, ActorBehaviorParameters] = {}
    for actor_id in {spec.striking_actor_id, spec.struck_actor_id}:
        actor = scene.actors[actor_id]
        speed_limit = actor.speed_limit_mps or scene.default_speed_limit_mps
        sequence = spec.behavior_sequences.get(actor_id) or ["keep"]
        level = str(actor.metadata.get("speed_level") or "medium")
        interval = speed_interval(level, speed_limit, category=actor.category)
        initial_speed = actor.speed_mps if actor.speed_mps > 0.1 else sum(interval) / 2.0
        initial_speed = clamp(initial_speed, 0.0, speed_limit * 1.15)
        brakes = any(action in {"brake", "stop", "decelerate"} for action in sequence)
        deceleration = 5.0 if "brake" in sequence or "stop" in sequence else 2.5 if brakes else 0.0
        brake_start = max(0.1, target_time * 0.55) if brakes else math.inf
        target_speed = 0.0 if "stop" in sequence else initial_speed
        parameters[actor_id] = ActorBehaviorParameters(
            initial_speed=initial_speed,
            target_speed=target_speed,
            acceleration=2.5 if "accelerate" in sequence else 0.0,
            deceleration=deceleration,
            brake_start_time=brake_start,
            start_delay=0.0,
            phase_durations=[],
        )

    if spec.accident_type == "rear_end":
        following = parameters[spec.striking_actor_id]
        lead = parameters[spec.struck_actor_id]
        clearance = max(0.0, anchor.initial_clearance_m or 0.0)
        lead_distance = piecewise_displacement(lead, target_time)
        required_speed = (clearance + lead_distance) / target_time
        following.initial_speed = clamp(
            required_speed,
            0.5,
            (scene.actors[spec.striking_actor_id].speed_limit_mps or scene.default_speed_limit_mps) * 1.25,
        )
        following.target_speed = max(following.target_speed, following.initial_speed)
        if math.isfinite(following.brake_start_time):
            following.brake_start_time = max(target_time * 0.8, following.brake_start_time)
    else:
        for actor_id, path_distance in anchor.actor_path_distances.items():
            params = parameters[actor_id]
            available = max(0.2, target_time - params.start_delay)
            required_speed = max(0.5, path_distance / available)
            limit = scene.actors[actor_id].speed_limit_mps or scene.default_speed_limit_mps
            params.initial_speed = clamp(required_speed, 0.5, limit * 1.25)
            params.target_speed = params.initial_speed
            params.acceleration = 0.0
            params.deceleration = 0.0
            params.brake_start_time = math.inf
        if spec.accident_type == "lane_change":
            changing_id = next(
                (
                    actor_id
                    for actor_id, maneuver in spec.actor_maneuvers.items()
                    if str(maneuver).startswith("lane_change")
                ),
                spec.striking_actor_id,
            )
            changing = parameters[changing_id]
            changing.lane_change_start_time = max(0.1, target_time * 0.35)
            changing.lane_change_duration = max(1.2, target_time * 0.45)
    return parameters


def piecewise_displacement(
    parameters: ActorBehaviorParameters,
    time_s: float,
    *,
    integration_step_s: float = 0.01,
) -> float:
    """Integrate the low-dimensional phases (delay, target tracking, braking)."""

    time_s = max(0.0, float(time_s))
    if time_s <= parameters.start_delay:
        return 0.0
    elapsed = time_s - parameters.start_delay
    step = max(0.002, min(float(integration_step_s), elapsed))
    speed, distance_m, current = max(0.0, parameters.initial_speed), 0.0, 0.0
    while current < elapsed - 1e-9:
        dt = min(step, elapsed - current)
        absolute_time = current + parameters.start_delay
        next_speed = _next_speed(parameters, speed, absolute_time, dt)
        distance_m += 0.5 * (speed + next_speed) * dt
        speed = next_speed
        current += dt
    return distance_m


def _next_speed(params, speed, absolute_time, dt):
    if absolute_time >= params.brake_start_time:
        return max(0.0, speed - max(0.0, params.deceleration) * dt)
    target = max(0.0, params.target_speed)
    if speed < target:
        return min(target, speed + max(0.0, params.acceleration) * dt)
    if speed > target and params.deceleration > 0.0:
        return max(target, speed - params.deceleration * dt)
    return speed


class KinematicRolloutEvaluator:
    """Fast search backend using path motion plus exact 2-D OBB overlap.

    This is an initializer/surrogate, not a replacement for final CARLA
    validation.  The same solver accepts a CARLA evaluator callable.
    """

    def __init__(
        self,
        scene: SceneState,
        spec: AccidentSpecification,
        paths: Dict[str, ReferencePath],
        anchor: CollisionAnchor,
        *,
        log_stride: int = 2,
    ) -> None:
        self.scene = scene
        self.spec = spec
        self.paths = paths
        self.anchor = anchor
        self.log_stride = max(1, int(log_stride))

    def __call__(
        self, parameters: Dict[str, ActorBehaviorParameters]
    ) -> RolloutResult:
        actor_ids = [self.spec.striking_actor_id, self.spec.struck_actor_id]
        state = {
            actor_id: {
                "s": 0.0,
                "speed": (
                    max(0.0, parameters[actor_id].initial_speed)
                    if parameters[actor_id].start_delay <= 0.0
                    else 0.0
                ),
            }
            for actor_id in actor_ids
        }
        logs = {actor_id: [] for actor_id in actor_ids}
        arrival_times: Dict[str, Optional[float]] = {actor_id: None for actor_id in actor_ids}
        minimum_separation = float("inf")
        collision_time = None
        collision_location = None
        impact_speed = None
        contact_sides = {}
        dt = self.scene.fixed_delta_seconds
        steps = int(math.ceil(self.scene.duration_s / dt)) + 1
        previous_time = 0.0
        for tick in range(steps):
            time_s = min(self.scene.duration_s, tick * dt)
            if tick > 0:
                for actor_id in actor_ids:
                    params = parameters[actor_id]
                    actor_motion = state[actor_id]
                    speed = actor_motion["speed"]
                    if time_s <= params.start_delay:
                        next_speed = 0.0
                        actor_motion["s"] += 0.0
                    elif previous_time < params.start_delay:
                        # CARLA uses a flying start after the configured launch
                        # delay; only the active fraction of this tick travels.
                        next_speed = max(0.0, params.initial_speed)
                        actor_motion["s"] += next_speed * (time_s - params.start_delay)
                    else:
                        next_speed = _next_speed(params, speed, previous_time, dt)
                        actor_motion["s"] += 0.5 * (speed + next_speed) * dt
                    actor_motion["speed"] = next_speed
            poses = {}
            for actor_id in actor_ids:
                path = self.paths[actor_id]
                s = min(state[actor_id]["s"], path.length)
                position, yaw = self._pose(actor_id, path, s, time_s, parameters[actor_id])
                poses[actor_id] = (position, yaw)
                anchor_s = self.anchor.actor_path_distances.get(actor_id)
                if anchor_s is not None and arrival_times[actor_id] is None and s >= anchor_s:
                    arrival_times[actor_id] = time_s
                if tick % self.log_stride == 0:
                    logs[actor_id].append(
                        {
                            "time_s": round(time_s, 3),
                            "x": round(position[0], 4),
                            "y": round(position[1], 4),
                            "yaw_deg": round(math.degrees(yaw), 3),
                            "speed_mps": round(state[actor_id]["speed"], 4),
                            "path_distance_m": round(s, 4),
                        }
                    )
            actor_a, actor_b = (self.scene.actors[actor_id] for actor_id in actor_ids)
            pose_a, pose_b = poses[actor_ids[0]], poses[actor_ids[1]]
            separation = obb_separation(
                pose_a[0], pose_a[1], actor_a.dimensions.length, actor_a.dimensions.width,
                pose_b[0], pose_b[1], actor_b.dimensions.length, actor_b.dimensions.width,
            )
            minimum_separation = min(minimum_separation, max(0.0, separation))
            if separation <= 0.0 and collision_time is None:
                collision_time = time_s
                collision_location = (
                    (pose_a[0][0] + pose_b[0][0]) / 2.0,
                    (pose_a[0][1] + pose_b[0][1]) / 2.0,
                )
                velocity_a = (
                    state[actor_ids[0]]["speed"] * math.cos(pose_a[1]),
                    state[actor_ids[0]]["speed"] * math.sin(pose_a[1]),
                )
                velocity_b = (
                    state[actor_ids[1]]["speed"] * math.cos(pose_b[1]),
                    state[actor_ids[1]]["speed"] * math.sin(pose_b[1]),
                )
                impact_speed = math.hypot(
                    velocity_a[0] - velocity_b[0], velocity_a[1] - velocity_b[1]
                )
                contact_sides = _contact_sides(
                    actor_ids[0], pose_a[0], pose_a[1],
                    actor_ids[1], pose_b[0], pose_b[1],
                )
                break
            previous_time = time_s

        arrival_error = 0.0
        arrivals = [value for value in arrival_times.values() if value is not None]
        if self.spec.accident_type != "rear_end" and len(arrivals) == 2:
            arrival_error = abs(arrivals[0] - arrivals[1])
        return RolloutResult(
            collided=collision_time is not None,
            collision_pair=tuple(actor_ids) if collision_time is not None else None,
            collision_time=collision_time,
            collision_location=collision_location,
            relative_impact_speed=impact_speed,
            trajectory_logs=logs,
            minimum_pair_distance=minimum_separation,
            relation_errors={"arrival_time_delta_s": arrival_error},
            arrival_times=arrival_times,
            contact_sides=contact_sides,
            control_smoothness=_control_smoothness(parameters),
            metadata={"backend": "kinematic_obb", "fixed_delta_seconds": dt},
        )

    def _pose(self, actor_id, path, path_distance, time_s, params):
        maneuver = str(self.spec.actor_maneuvers.get(actor_id, "keep_lane"))
        if not maneuver.startswith("lane_change"):
            return point_at(path, path_distance), heading_at(path, path_distance)
        actor = self.scene.actors[actor_id]
        base_yaw = math.radians(actor.yaw_deg)
        forward = (math.cos(base_yaw), math.sin(base_yaw))
        right = (-math.sin(base_yaw), math.cos(base_yaw))
        start = params.lane_change_start_time or 0.0
        duration = max(0.5, params.lane_change_duration or 2.0)
        progress = clamp((time_s - start) / duration, 0.0, 1.0)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        direction = -1.0 if "left" in maneuver else 1.0
        lateral = direction * 3.5 * smooth
        position = (
            actor.x + forward[0] * path_distance + right[0] * lateral,
            actor.y + forward[1] * path_distance + right[1] * lateral,
        )
        lateral_rate = direction * 3.5 * (6.0 * progress * (1.0 - progress)) / duration
        yaw = base_yaw + math.atan2(lateral_rate, max(0.2, params.initial_speed))
        return position, yaw


def _control_smoothness(parameters):
    total = 0.0
    for params in parameters.values():
        total += max(0.0, params.acceleration - 4.0) ** 2
        total += max(0.0, params.deceleration - 8.0) ** 2
    return total


def _contact_sides(actor_a_id, position_a, yaw_a, actor_b_id, position_b, yaw_b):
    return {
        actor_a_id: _side_toward(position_a, yaw_a, position_b),
        actor_b_id: _side_toward(position_b, yaw_b, position_a),
    }


def _side_toward(origin, yaw, target):
    bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
    relative = (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi
    degrees = math.degrees(relative)
    if abs(degrees) <= 45.0:
        return "front"
    if abs(degrees) >= 135.0:
        return "rear"
    return "right" if degrees > 0.0 else "left"
