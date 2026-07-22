import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collision_fitting import (
    CollisionAnchoredFittingPipeline,
    RolloutResult,
    build_accident_specification,
    build_scene_state,
)
from tools.collision_fitting.anchors import CollisionAnchorBuilder
from tools.collision_fitting.carla_rollout import _rollout_from_metrics
from tools.collision_fitting.integration import (
    apply_start_offsets_to_spawn_payload,
    build_fitted_trajectory_dsl,
)
from tools.collision_fitting.kinematics import speed_interval
from tools.collision_fitting.models import ActorBehaviorParameters
from tools.collision_fitting.objective import CollisionObjective
from tools.collision_fitting.paths import MapConstrainedPathBuilder, make_reference_path
from tools.collision_fitting.repair import DirectedRepairPlanner, RepairAction
from tools.collision_fitting.solver import BehaviorParameterSolver
from tools.video_trajectory_dsl import validate_video_trajectory_dsl


SPAWN = {
    "entities": [
        {
            "id": "ego",
            "category": "car",
            "location": {"x": 0.0, "y": 0.0, "z": 0.3},
            "rotation": {"yaw": 0.0},
            "projected_lane": {"road_id": 1, "lane_id": -1},
        },
        {
            "id": "lead",
            "category": "car",
            "location": {"x": 15.0, "y": 0.0, "z": 0.3},
            "rotation": {"yaw": 0.0},
            "projected_lane": {"road_id": 1, "lane_id": -1},
        },
        {
            "id": "candidate",
            "category": "car",
            "location": {"x": 16.0, "y": 3.5, "z": 0.3},
            "rotation": {"yaw": 0.0},
        },
    ]
}


UNDERSTANDING = {
    "accident_type": "rear-end",
    "striking_actor_ref": "ego",
    "struck_actor_ref": "v1",
    "collision_time_range_s": [2.5, 3.5],
    "participants": [
        {
            "ref": "v1",
            "matched_actor_id": "lead",
            "match_confidence": 0.9,
            "candidate_actor_ids": [
                {"actor_id": "candidate", "confidence": 0.35}
            ],
            "role": "lead_vehicle",
            "maneuver": "keep_lane",
            "behavior_sequence": ["keep", "brake", "stop"],
        }
    ],
    "event_sequence": [
        {"t_rel": "mid", "actor_ref": "v1", "action": "brake"}
    ],
    "end_state": {"collision": True, "collision_pair": ["ego", "v1"]},
}


class SpecificationTest(unittest.TestCase):
    def test_current_vlm_schema_maps_to_top_k_discrete_spec(self):
        scene = build_scene_state(SPAWN)
        spec = build_accident_specification(
            UNDERSTANDING, scene.actors.keys(), duration_s=6.0, top_k=2
        )
        self.assertEqual(spec.accident_type, "rear_end")
        self.assertEqual(spec.striking_actor_id, "ego")
        self.assertEqual(spec.struck_actor_id, "lead")
        self.assertEqual(spec.collision_time_range, (2.5, 3.5))
        self.assertEqual(len(spec.pair_hypotheses), 2)
        self.assertEqual(spec.behavior_sequences["lead"], ["keep", "brake", "stop"])

    def test_speed_levels_respect_speed_limit(self):
        low = speed_interval("low", 20.0)
        high = speed_interval("high", 20.0)
        self.assertLess(low[1], high[0] + 1.0)
        self.assertLessEqual(high[1], 20.0)


class GeometryAndAnchorTest(unittest.TestCase):
    def test_rear_end_uses_bumper_clearance_not_center_distance(self):
        scene = build_scene_state(SPAWN)
        spec = build_accident_specification(UNDERSTANDING, scene.actors.keys())
        builder = MapConstrainedPathBuilder(horizon_m=30.0)
        paths = {
            actor_id: builder.build(scene.actors[actor_id])
            for actor_id in ("ego", "lead")
        }
        anchor = CollisionAnchorBuilder().build(scene, spec, paths)
        self.assertAlmostEqual(anchor.initial_clearance_m, 10.4, places=2)
        self.assertEqual(anchor.expected_contact_sides, {"ego": "front", "lead": "rear"})

    def test_intersection_conflict_is_a_region(self):
        payload = {
            "entities": [
                {"id": "ego", "location": {"x": -10, "y": 0}, "rotation": {"yaw": 0}},
                {"id": "cross", "location": {"x": 0, "y": -10}, "rotation": {"yaw": 90}},
            ]
        }
        understanding = {
            "accident_type": "intersection crossing",
            "participants": [{"ref": "v1", "matched_actor_id": "cross"}],
            "end_state": {"collision": True, "collision_pair": ["ego", "v1"]},
        }
        scene = build_scene_state(payload)
        spec = build_accident_specification(understanding, scene.actors.keys())
        paths = {
            "ego": make_reference_path("ego", [(-10, 0), (10, 0)]),
            "cross": make_reference_path("cross", [(0, -10), (0, 10)]),
        }
        anchor = CollisionAnchorBuilder(sample_step_m=0.5).build(scene, spec, paths)
        self.assertTrue(anchor.feasible)
        self.assertLess(anchor.conflict_intervals["ego"][0], anchor.conflict_intervals["ego"][1])
        self.assertAlmostEqual(anchor.conflict_position[0], 0.0, delta=0.5)

    def test_overlapping_static_boxes_receive_reproducible_start_offset(self):
        payload = {
            "entities": [
                {"id": "ego", "location": {"x": 0, "y": 0}, "rotation": {"yaw": 0}},
                {"id": "lead", "location": {"x": 1.5, "y": 0}, "rotation": {"yaw": 0}},
            ]
        }
        scene = build_scene_state(payload)
        spec = build_accident_specification(
            {
                "accident_type": "rear-end",
                "participants": [{"ref": "v1", "matched_actor_id": "lead"}],
                "end_state": {"collision": True, "collision_pair": ["ego", "v1"]},
            },
            scene.actors.keys(),
        )
        builder = MapConstrainedPathBuilder(horizon_m=20.0)
        paths = {actor_id: builder.build(scene.actors[actor_id]) for actor_id in ("ego", "lead")}
        anchor = CollisionAnchorBuilder().build(scene, spec, paths)
        self.assertGreater(anchor.actor_start_offsets["ego"], 0.0)
        result = CollisionAnchoredFittingPipeline(
            solver=BehaviorParameterSolver(max_rollouts=1), top_k=1
        ).fit(scene, spec)
        fitted = apply_start_offsets_to_spawn_payload(payload, result)
        ego = next(row for row in fitted["entities"] if row["id"] == "ego")
        self.assertLess(ego["location"]["x"], 0.0)
        self.assertIn("path_start_offset_m", ego["dynamic_fitting"])


class SolverAndObjectiveTest(unittest.TestCase):
    def test_rear_end_pipeline_hits_requested_time_window(self):
        scene = build_scene_state(SPAWN, duration_s=6.0)
        spec = build_accident_specification(UNDERSTANDING, scene.actors.keys(), top_k=1)
        result = CollisionAnchoredFittingPipeline(
            solver=BehaviorParameterSolver(max_rollouts=36), top_k=1
        ).fit(scene, spec)
        self.assertTrue(result.solver_result.rollout.collided)
        self.assertTrue(result.solver_result.converged)
        self.assertGreaterEqual(result.solver_result.rollout.collision_time, 2.5)
        self.assertLessEqual(result.solver_result.rollout.collision_time, 3.5)

        dsl = build_fitted_trajectory_dsl(
            result, scene_id="s0", duration_s=6.0
        )
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN)
        self.assertIsNone(error)
        self.assertEqual(normalized["metadata"]["parameter_source"], "collision_anchored_fitting")

    def test_wrong_pair_is_worse_than_missing_target_collision(self):
        scene = build_scene_state(SPAWN)
        spec = build_accident_specification(UNDERSTANDING, scene.actors.keys(), top_k=1)
        paths = {
            actor_id: MapConstrainedPathBuilder().build(scene.actors[actor_id])
            for actor_id in ("ego", "lead")
        }
        anchor = CollisionAnchorBuilder().build(scene, spec, paths)
        params = {
            actor_id: ActorBehaviorParameters(5.0, 5.0)
            for actor_id in ("ego", "lead")
        }
        objective = CollisionObjective()
        wrong, _ = objective.evaluate(
            spec,
            anchor,
            RolloutResult(collided=True, collision_pair=("ego", "candidate")),
            params,
        )
        missing, _ = objective.evaluate(
            spec,
            anchor,
            RolloutResult(collided=False, minimum_pair_distance=2.0),
            params,
        )
        self.assertGreater(wrong, missing)

    def test_repair_rules_are_directional(self):
        scene = build_scene_state(SPAWN)
        spec = build_accident_specification(UNDERSTANDING, scene.actors.keys(), top_k=1)
        paths = {
            actor_id: MapConstrainedPathBuilder().build(scene.actors[actor_id])
            for actor_id in ("ego", "lead")
        }
        anchor = CollisionAnchorBuilder().build(scene, spec, paths)
        actions = DirectedRepairPlanner().propose(
            spec, anchor, RolloutResult(collided=False, minimum_pair_distance=3.0)
        )
        action_names = {item["action"] for item in actions}
        self.assertIn(RepairAction.INCREASE_STRIKER_SPEED.value, action_names)
        self.assertIn(RepairAction.DELAY_STRIKER_BRAKE.value, action_names)


class CarlaMetricsAdapterTest(unittest.TestCase):
    def test_metrics_preserve_pair_time_location_and_non_target_collision(self):
        rollout = _rollout_from_metrics(
            {
                "collision_events": [
                    {
                        "time_s": 2.75,
                        "collision_pair": ["ego", "lead"],
                        "location": {"x": 4.0, "y": 1.0},
                    },
                    {"time_s": 3.0, "collision_pair": ["ego", "wall"]},
                ],
                "min_distance_m": 0.0,
                "trajectory_logs": {"ego": [{"time_s": 0.0}]},
            },
            target_pair=("ego", "lead"),
            candidate_index=2,
        )
        self.assertEqual(rollout.collision_pair, ("ego", "lead"))
        self.assertEqual(rollout.collision_time, 2.75)
        self.assertEqual(rollout.collision_location, (4.0, 1.0))
        self.assertEqual(rollout.non_target_collisions, [("ego", "wall")])


if __name__ == "__main__":
    unittest.main()
