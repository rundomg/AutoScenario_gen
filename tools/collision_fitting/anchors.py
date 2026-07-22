"""Vehicle-size-aware collision anchors for the three planned accident families."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .models import AccidentSpecification, CollisionAnchor, ReferencePath, SceneState
from .paths import (
    distance,
    heading_at,
    obb_separation,
    point_at,
    project_onto_path,
    resample_path,
)


class CollisionAnchorBuilder:
    def __init__(self, *, sample_step_m: float = 0.5, safety_margin_m: float = 0.15) -> None:
        self.sample_step_m = max(0.2, float(sample_step_m))
        self.safety_margin_m = max(0.0, float(safety_margin_m))

    def build(
        self,
        scene_state: SceneState,
        accident_spec: AccidentSpecification,
        paths: Dict[str, ReferencePath],
    ) -> CollisionAnchor:
        if accident_spec.accident_type == "intersection":
            return self._intersection(scene_state, accident_spec, paths)
        if accident_spec.accident_type == "lane_change":
            return self._lane_change(scene_state, accident_spec, paths)
        return self._rear_end(scene_state, accident_spec, paths)

    def _rear_end(self, scene, spec, paths) -> CollisionAnchor:
        striking = scene.actors[spec.striking_actor_id]
        struck = scene.actors[spec.struck_actor_id]
        striking_path = paths[striking.actor_id]
        delta_x, delta_y = struck.x - striking.x, struck.y - striking.y
        yaw = math.radians(striking.yaw_deg)
        center_longitudinal = delta_x * math.cos(yaw) + delta_y * math.sin(yaw)
        clearance = center_longitudinal - 0.5 * (
            striking.dimensions.length + struck.dimensions.length
        )
        start_offsets = {}
        # Static image reconstruction can place actor centres closer than their
        # physical CARLA boxes allow. Move only the following actor backward by
        # the minimum required amount and persist the offset in the final config.
        minimum_spawn_clearance = 0.5
        if clearance < minimum_spawn_clearance:
            offset = minimum_spawn_clearance - clearance
            _shift_path_backward(striking_path, striking.yaw_deg, offset)
            start_offsets[striking.actor_id] = offset
            center_longitudinal += offset
            clearance = minimum_spawn_clearance
        lateral = abs(-delta_x * math.sin(yaw) + delta_y * math.cos(yaw))
        allowed_lateral = 0.5 * (striking.dimensions.width + struck.dimensions.width) + 0.5
        ahead = center_longitudinal > 0.0
        feasible = ahead and lateral <= allowed_lateral
        reason = "same-path bumper clearance" if feasible else (
            "struck actor is not ahead" if not ahead else "actors are not laterally aligned"
        )
        # The rear-end anchor represents required relative closure rather than an
        # impossible center-point coincidence.
        conflict_s = max(0.0, center_longitudinal - 0.5 * struck.dimensions.length)
        return CollisionAnchor(
            accident_type="rear_end",
            conflict_position=point_at(striking_path, min(conflict_s, striking_path.length)),
            actor_path_distances={
                striking.actor_id: max(0.0, clearance),
                struck.actor_id: 0.0,
            },
            target_time_range=spec.collision_time_range,
            expected_contact_sides=spec.expected_contact_sides
            or {striking.actor_id: "front", struck.actor_id: "rear"},
            actor_start_offsets=start_offsets,
            initial_clearance_m=max(0.0, clearance),
            feasible=feasible,
            reason=reason,
        )

    def _intersection(self, scene, spec, paths) -> CollisionAnchor:
        actor_a = scene.actors[spec.striking_actor_id]
        actor_b = scene.actors[spec.struck_actor_id]
        path_a, path_b = paths[actor_a.actor_id], paths[actor_b.actor_id]
        # A centerline-only intersection misses real contacts near corners.  The
        # threshold is the sum of conservative half-widths, so this finds a
        # conflict *region* swept out by both vehicle boxes.
        threshold = (
            0.5 * actor_a.dimensions.width
            + 0.5 * actor_b.dimensions.width
            + self.safety_margin_m
        )
        conflicts = _path_conflicts(
            path_a,
            path_b,
            threshold,
            self.sample_step_m,
            dimensions_a=actor_a.dimensions,
            dimensions_b=actor_b.dimensions,
            safety_margin=self.safety_margin_m,
        )
        if not conflicts:
            closest = _closest_path_samples(path_a, path_b, self.sample_step_m)
            s_a, point_a, s_b, point_b, residual = closest
            return CollisionAnchor(
                accident_type="intersection",
                conflict_position=((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0),
                actor_path_distances={actor_a.actor_id: s_a, actor_b.actor_id: s_b},
                target_time_range=spec.collision_time_range,
                expected_contact_sides=spec.expected_contact_sides,
                feasible=False,
                reason=f"path corridors do not overlap (closest={residual:.2f}m)",
            )
        s_values_a = [item[0] for item in conflicts]
        s_values_b = [item[2] for item in conflicts]
        central = min(conflicts, key=lambda item: item[4])
        conflict_position = (
            (central[1][0] + central[3][0]) / 2.0,
            (central[1][1] + central[3][1]) / 2.0,
        )
        return CollisionAnchor(
            accident_type="intersection",
            conflict_position=conflict_position,
            actor_path_distances={actor_a.actor_id: central[0], actor_b.actor_id: central[2]},
            target_time_range=spec.collision_time_range,
            expected_contact_sides=spec.expected_contact_sides,
            conflict_intervals={
                actor_a.actor_id: (min(s_values_a), max(s_values_a)),
                actor_b.actor_id: (min(s_values_b), max(s_values_b)),
            },
            feasible=True,
            reason="vehicle-width swept path conflict region",
        )

    def _lane_change(self, scene, spec, paths) -> CollisionAnchor:
        changing_id = next(
            (
                actor_id
                for actor_id, maneuver in spec.actor_maneuvers.items()
                if str(maneuver).startswith("lane_change")
            ),
            spec.striking_actor_id,
        )
        other_id = spec.struck_actor_id if changing_id == spec.striking_actor_id else spec.striking_actor_id
        changing, other = scene.actors[changing_id], scene.actors[other_id]
        changing_path, other_path = paths[changing_id], paths[other_id]
        threshold = 0.5 * (changing.dimensions.width + other.dimensions.width) + self.safety_margin_m
        conflicts = _path_conflicts(
            changing_path,
            other_path,
            threshold,
            self.sample_step_m,
            dimensions_a=changing.dimensions,
            dimensions_b=other.dimensions,
            safety_margin=self.safety_margin_m,
        )
        if not conflicts:
            closest = _closest_path_samples(changing_path, other_path, self.sample_step_m)
            s_a, point_a, s_b, point_b, residual = closest
            return CollisionAnchor(
                accident_type="lane_change",
                conflict_position=((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0),
                actor_path_distances={changing_id: s_a, other_id: s_b},
                target_time_range=spec.collision_time_range,
                expected_contact_sides=spec.expected_contact_sides,
                feasible=False,
                reason=f"lane-change path never enters target actor corridor ({residual:.2f}m)",
            )
        # First corridor entry is the useful anchor for a sideswipe; longitudinal
        # overlap is resolved by arrival-time fitting and final OBB validation.
        first = min(conflicts, key=lambda item: item[0])
        s_values_a = [item[0] for item in conflicts]
        s_values_b = [item[2] for item in conflicts]
        return CollisionAnchor(
            accident_type="lane_change",
            conflict_position=((first[1][0] + first[3][0]) / 2.0, (first[1][1] + first[3][1]) / 2.0),
            actor_path_distances={changing_id: first[0], other_id: first[2]},
            target_time_range=spec.collision_time_range,
            expected_contact_sides=spec.expected_contact_sides
            or {changing_id: "side", other_id: "side"},
            conflict_intervals={
                changing_id: (min(s_values_a), max(s_values_a)),
                other_id: (min(s_values_b), max(s_values_b)),
            },
            feasible=True,
            reason="first lane-entry corridor overlap; OBB contact required in rollout",
        )


def _path_conflicts(
    path_a,
    path_b,
    threshold,
    step,
    *,
    dimensions_a=None,
    dimensions_b=None,
    safety_margin=0.0,
) -> List[Tuple]:
    samples_a, samples_b = resample_path(path_a, step), resample_path(path_b, step)
    output = []
    for s_a, point_a in samples_a:
        for s_b, point_b in samples_b:
            residual = distance(point_a, point_b)
            overlaps = residual <= threshold
            if dimensions_a is not None and dimensions_b is not None:
                separation = obb_separation(
                    point_a,
                    heading_at(path_a, s_a),
                    dimensions_a.length,
                    dimensions_a.width,
                    point_b,
                    heading_at(path_b, s_b),
                    dimensions_b.length,
                    dimensions_b.width,
                )
                overlaps = separation <= float(safety_margin)
            if overlaps:
                output.append((s_a, point_a, s_b, point_b, residual))
    return output


def _closest_path_samples(path_a, path_b, step):
    best = None
    for s_a, point_a in resample_path(path_a, step):
        s_b, residual, point_b = project_onto_path(path_b, point_a)
        candidate = (s_a, point_a, s_b, point_b, residual)
        if best is None or residual < best[4]:
            best = candidate
    return best


def _shift_path_backward(path, yaw_deg, offset_m):
    yaw = math.radians(float(yaw_deg))
    dx = -math.cos(yaw) * float(offset_m)
    dy = -math.sin(yaw) * float(offset_m)
    path.waypoints = [(point[0] + dx, point[1] + dy) for point in path.waypoints]
    path.source = f"{path.source}+rear_end_spawn_offset"
