import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from tools.video_trajectory_dsl import (
    lower_to_risk_dsl,
    validate_video_trajectory_dsl,
)


SPAWN_PAYLOAD = {
    "entities": [
        {"id": "ego", "category": "car"},
        {"id": "veh_1", "category": "car"},
        {"id": "veh_2", "category": "car"},
    ]
}


def _valid_dsl():
    return {
        "schema_version": "video-trajectory-dsl-v1",
        "scene_id": "s0000_c0",
        "accident_type": "rear-end",
        "ego": {"actor_id": "ego", "target_speed_mps": 10.0},
        "duration_s": 8.0,
        "trajectories": [
            {
                "actor_id": "veh_1",
                "role": "lead_vehicle",
                "segments": [
                    {"start_s": 0.0, "end_s": 2.5, "action": "lane_follow_speed", "target_speed_mps": 5.0},
                    {"start_s": 2.5, "end_s": 4.0, "action": "brake", "intensity": 0.9},
                    {"start_s": 4.0, "end_s": 8.0, "action": "stop"},
                ],
            }
        ],
    }


class ValidateVideoTrajectoryDslTest(unittest.TestCase):
    def test_valid_document_passes_and_sorts(self):
        dsl = _valid_dsl()
        # Shuffle segment order to confirm normalization sorts by start_s.
        dsl["trajectories"][0]["segments"].reverse()
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        starts = [seg["start_s"] for seg in normalized["trajectories"][0]["segments"]]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(normalized["schema_version"], "video-trajectory-dsl-v1")

    def test_unknown_actor_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["actor_id"] = "ghost"
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("unknown actor id", error)

    def test_full_vehicle_coverage_reports_missing_background_actor(self):
        _, error = validate_video_trajectory_dsl(
            _valid_dsl(),
            SPAWN_PAYLOAD,
            require_all_vehicle_actors=True,
        )
        self.assertIn("Missing trajectories", error)
        self.assertIn("veh_2", error)

    def test_duplicate_actor_trajectory_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"].append(dict(dsl["trajectories"][0]))
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("Duplicate trajectory", error)

    def test_lane_change_requires_a_direction(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"][1]["action"] = "lane_change"
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("lane_change", error)
        self.assertIn("direction", error)

    def test_unknown_action_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"][0]["action"] = "teleport"
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("unknown action", error)

    def test_negative_time_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"][0]["start_s"] = -1.0
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("start_s must be >= 0", error)

    def test_end_before_start_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"][0]["end_s"] = 0.0
        dsl["trajectories"][0]["segments"][0]["start_s"] = 1.0
        # make non-overlapping otherwise; this segment alone is invalid
        dsl["trajectories"][0]["segments"] = [dsl["trajectories"][0]["segments"][0]]
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("end_s must be >= start_s", error)

    def test_overlapping_segments_rejected(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"] = [
            {"start_s": 0.0, "end_s": 3.0, "action": "lane_follow_speed", "target_speed_mps": 5.0},
            {"start_s": 1.0, "end_s": 4.0, "action": "stop"},
        ]
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("overlapping", error)

    def test_lane_follow_requires_speed(self):
        dsl = _valid_dsl()
        del dsl["trajectories"][0]["segments"][0]["target_speed_mps"]
        _, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIn("requires target_speed_mps", error)

    def test_missing_ego_in_payload_rejected(self):
        _, error = validate_video_trajectory_dsl(_valid_dsl(), {"entities": [{"id": "veh_1"}]})
        self.assertIn("must contain actor id `ego`", error)


class LowerToRiskDslTest(unittest.TestCase):
    def test_lowering_produces_risk_dsl_v1_events(self):
        normalized, error = validate_video_trajectory_dsl(_valid_dsl(), SPAWN_PAYLOAD)
        self.assertIsNone(error)
        risk = lower_to_risk_dsl(normalized, ego_speed_mps=10.0)

        self.assertEqual(risk["schema_version"], "risk-dsl-v1")
        self.assertEqual(risk["ego"]["target_speed_mps"], 10.0)
        self.assertEqual(risk["duration_s"], 8.0)

        events = risk["events"]
        self.assertEqual(len(events), 3)
        # First segment at t=0 -> immediate; later -> time_elapsed_above ascending.
        self.assertEqual(events[0]["trigger"], {"type": "immediate"})
        self.assertEqual(events[1]["trigger"], {"type": "time_elapsed_above", "value_s": 2.5})
        self.assertEqual(events[2]["trigger"], {"type": "time_elapsed_above", "value_s": 4.0})

        # Action mapping.
        self.assertEqual(events[0]["action"], {"type": "set_speed", "speed_mps": 5.0})
        self.assertEqual(events[1]["action"], {"type": "brake", "intensity": 0.9})
        self.assertEqual(events[2]["action"], {"type": "stop"})

        self.assertEqual(
            [a["actor_id"] for a in risk["actors"]], ["veh_1"]
        )

    def test_steer_offset_lowers_to_steer_with_duration(self):
        dsl = _valid_dsl()
        dsl["trajectories"][0]["segments"] = [
            {"start_s": 1.0, "end_s": 2.0, "action": "steer_offset", "steer": 0.4, "throttle": 0.5},
        ]
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        risk = lower_to_risk_dsl(normalized)
        action = risk["events"][0]["action"]
        self.assertEqual(action["type"], "steer")
        self.assertAlmostEqual(action["steer"], 0.4)
        self.assertAlmostEqual(action["duration_s"], 1.0)

    def test_events_sorted_across_actors_by_start(self):
        dsl = _valid_dsl()
        dsl["trajectories"].append(
            {
                "actor_id": "veh_2",
                "role": "second",
                "segments": [
                    {"start_s": 1.0, "end_s": 5.0, "action": "lane_follow_speed", "target_speed_mps": 4.0}
                ],
            }
        )
        normalized, error = validate_video_trajectory_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        risk = lower_to_risk_dsl(normalized)
        triggers = [
            e["trigger"].get("value_s", 0.0) for e in risk["events"]
        ]
        self.assertEqual(triggers, sorted(triggers))


if __name__ == "__main__":
    unittest.main()
