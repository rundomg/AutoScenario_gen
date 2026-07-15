import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vehicle_atomic_behaviors import (  # noqa: E402
    SUPPORTED_ATOMIC_BEHAVIORS,
    normalize_atomic_behavior,
)
from tools.video_trajectory_dsl import (  # noqa: E402
    lower_to_risk_dsl,
    validate_video_trajectory_dsl,
)


ACTOR_IDS = {"ego", "lead", "cut_in"}
SPAWN_PAYLOAD = {"entities": [{"id": actor_id} for actor_id in ACTOR_IDS]}


class VehicleAtomicBehaviorValidationTest(unittest.TestCase):
    def normalize(self, payload):
        return normalize_atomic_behavior(
            payload,
            where="segment",
            actor_ids=ACTOR_IDS,
        )

    def test_catalog_covers_motion_route_interaction_and_raw_control(self):
        expected = {
            "lane_follow_speed",
            "accelerate_to_speed",
            "decelerate_to_speed",
            "brake",
            "emergency_brake",
            "coast",
            "stop",
            "hold_position",
            "steer_offset",
            "lane_change",
            "junction_maneuver",
            "drive_to_location",
            "reverse",
            "follow_actor",
            "approach_actor",
            "yield_to_actor",
            "raw_vehicle_control",
        }
        self.assertEqual(SUPPORTED_ATOMIC_BEHAVIORS, expected)

    def test_lane_change_normalizes_parameters_and_lights(self):
        normalized, error = self.normalize(
            {
                "action": "lane_change",
                "direction": "left",
                "lane_count": 1,
                "target_speed_mps": 8.0,
                "transition_distance_m": 20.0,
                "lights": ["left_blinker"],
            }
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["direction"], "left")
        self.assertEqual(normalized["target_speed_mps"], 8.0)
        self.assertEqual(normalized["lights"], ["left_blinker"])

    def test_interaction_atom_rejects_invented_target_actor(self):
        _, error = self.normalize(
            {
                "action": "approach_actor",
                "target_actor_id": "ghost",
                "target_gap_m": 0.0,
                "approach_speed_mps": 12.0,
            }
        )
        self.assertIn("real actor", error)

    def test_approach_actor_supports_reverse_collision_motion(self):
        normalized, error = self.normalize(
            {
                "action": "approach_actor",
                "target_actor_id": "lead",
                "target_gap_m": 0.0,
                "approach_speed_mps": 3.0,
                "travel_direction": "reverse",
                "steer": 0.15,
            }
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["travel_direction"], "reverse")
        self.assertEqual(normalized["steer"], 0.15)

    def test_raw_control_is_bounded_to_vehicle_control_ranges(self):
        normalized, error = self.normalize(
            {
                "action": "raw_vehicle_control",
                "throttle": 4.0,
                "steer": -3.0,
                "brake": -1.0,
            }
        )
        self.assertIsNone(error)
        self.assertEqual(normalized["throttle"], 1.0)
        self.assertEqual(normalized["steer"], -1.0)
        self.assertEqual(normalized["brake"], 0.0)


class ComplexBehaviorCompositionTest(unittest.TestCase):
    def test_cut_in_then_emergency_brake_composes_and_lowers(self):
        dsl = {
            "schema_version": "video-trajectory-dsl-v2",
            "scene_id": "s0",
            "ego": {"actor_id": "ego", "target_speed_mps": 10.0},
            "duration_s": 8.0,
            "trajectories": [
                {
                    "actor_id": "cut_in",
                    "role": "cut_in_vehicle",
                    "segments": [
                        {
                            "start_s": 0.0,
                            "end_s": 2.0,
                            "action": "lane_follow_speed",
                            "target_speed_mps": 8.0,
                        },
                        {
                            "start_s": 2.0,
                            "end_s": 4.0,
                            "action": "lane_change",
                            "direction": "left",
                            "target_speed_mps": 7.0,
                            "lights": ["left_blinker"],
                        },
                        {
                            "start_s": 4.0,
                            "end_s": 5.0,
                            "action": "emergency_brake",
                            "lights": ["brake"],
                        },
                        {
                            "start_s": 5.0,
                            "end_s": 8.0,
                            "action": "hold_position",
                        },
                    ],
                }
            ],
        }
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        risk = lower_to_risk_dsl(normalized)
        action_types = [event["action"]["type"] for event in risk["events"]]
        self.assertEqual(
            action_types,
            ["set_speed", "lane_change", "emergency_brake", "stop"],
        )
        self.assertEqual(risk["events"][1]["action"]["direction"], "left")
        self.assertEqual(
            risk["events"][1]["action"]["lights"], ["left_blinker"]
        )
        self.assertEqual(
            [event["active_duration_s"] for event in risk["events"]],
            [2.0, 2.0, 1.0, 3.0],
        )

    def test_approach_actor_allows_zero_gap_for_collision_reproduction(self):
        dsl = {
            "schema_version": "video-trajectory-dsl-v2",
            "scene_id": "s0",
            "ego": {"actor_id": "ego", "target_speed_mps": 10.0},
            "duration_s": 5.0,
            "trajectories": [
                {
                    "actor_id": "ego",
                    "segments": [
                        {
                            "start_s": 0.0,
                            "end_s": 5.0,
                            "action": "approach_actor",
                            "target_actor_id": "lead",
                            "target_gap_m": 0.0,
                            "approach_speed_mps": 12.0,
                        }
                    ],
                }
            ],
        }
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        action = lower_to_risk_dsl(normalized)["events"][0]["action"]
        self.assertEqual(action["type"], "approach_actor")
        self.assertEqual(action["target_actor_id"], "lead")
        self.assertEqual(action["target_gap_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
