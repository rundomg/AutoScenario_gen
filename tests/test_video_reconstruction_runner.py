import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from tools.video_reconstruction_runner import (
    VideoReconstructionRunner,
    evaluate_end_anchor,
)


SPAWN_PAYLOAD = {
    "entities": [
        {"id": "ego", "category": "car", "location": {"x": 0.0, "y": 0.0}, "rotation": {"yaw": 0.0}},
        {"id": "veh_1", "category": "car", "location": {"x": 12.0, "y": 0.0}, "rotation": {"yaw": 0.0}},
    ]
}
MATCH_REPORT = {"world_name": "Town10HD", "status": "ok", "reason": "anchor lane"}

UNDERSTANDING = {
    "scene_id": "s0000_c0",
    "accident_type": "rear-end",
    "summary": "lead vehicle stops, ego rear-ends it",
    "participants": [{"ref": "v1", "type": "car", "role": "lead_vehicle", "mapping_hint": "veh_1"}],
    "event_sequence": [{"t_rel": "end", "actor_ref": "v1", "action": "stop", "description": "stops"}],
    "end_state": {"collision": True, "collision_pair": ["ego", "v1"], "final_motion": "stopped"},
}

TRAJ_DSL_TEXT = json.dumps(
    {
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
                    {"start_s": 0.0, "end_s": 3.0, "action": "lane_follow_speed", "target_speed_mps": 4.0},
                    {"start_s": 3.0, "end_s": 8.0, "action": "stop"},
                ],
            }
        ],
    }
)


class FakeInterpreter:
    def call_agent(self, user_request, added_info):
        Path(added_info["output_fn"]).write_text(json.dumps(UNDERSTANDING))
        return UNDERSTANDING


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return TRAJ_DSL_TEXT


class FakeCodegen:
    def build_dsl_risk_scene_script(self, **kwargs):
        self.kwargs = kwargs
        return "# generated script\nprint('ok')\n"


def _fake_frame_extractor(video_path, start_s, end_s, output_dir, scene_id, context_sample_rate_s=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    start = str(Path(output_dir) / f"{scene_id}_start_frame.jpg")
    end = str(Path(output_dir) / f"{scene_id}_end_anchor_frame.jpg")
    Path(start).write_text("start")
    Path(end).write_text("end")
    return {
        "start_frame_path": start,
        "end_anchor_frame_path": end,
        "frames": [
            {"path": start, "is_start": True, "is_end_anchor": False},
            {"path": end, "is_start": False, "is_end_anchor": True},
        ],
    }


class VideoReconstructionRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        (self.folder / "s0000_c0_actors.json").write_text(json.dumps(SPAWN_PAYLOAD))
        (self.folder / "s0000_c0_match.json").write_text(json.dumps(MATCH_REPORT))
        self.video = self.folder / "v.mp4"
        self.video.write_bytes(b"\x00")

    def _runner(self, **overrides):
        kwargs = dict(
            output_folder=str(self.folder),
            scene_id="s0000_c0",
            video_path=str(self.video),
            start_s=4.0,
            end_s=7.0,
            enable_compile=False,
            interpreter=FakeInterpreter(),
            generator=FakeGenerator(),
            codegen=FakeCodegen(),
            frame_extractor=_fake_frame_extractor,
        )
        kwargs.update(overrides)
        return VideoReconstructionRunner(**kwargs)

    def test_end_to_end_produces_lowered_risk_dsl_and_script(self):
        runner = self._runner()
        summary = runner.run()

        artifact = summary["artifact"]
        self.assertTrue(artifact["dsl_valid"])
        self.assertEqual(artifact["attempts"], 1)
        self.assertEqual(summary["map_name"], "Town10HD")

        # Trajectory DSL persisted.
        traj = json.loads(Path(artifact["trajectory_dsl_path"]).read_text())
        self.assertEqual(traj["schema_version"], "video-trajectory-dsl-v1")

        # Lowered risk-dsl-v1 persisted and consumable by the existing generator.
        risk = json.loads(Path(artifact["risk_dsl_path"]).read_text())
        self.assertEqual(risk["schema_version"], "risk-dsl-v1")
        self.assertEqual(risk["events"][0]["trigger"], {"type": "immediate"})
        self.assertEqual(risk["events"][1]["trigger"]["type"], "time_elapsed_above")

        self.assertTrue(Path(artifact["script_path"]).exists())
        self.assertEqual(artifact["end_anchor_check"]["status"], "pending")

    def test_repair_loop_on_invalid_then_valid(self):
        class FlakyGenerator:
            def __init__(self):
                self.calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({"schema_version": "video-trajectory-dsl-v1", "trajectories": "bad"})
                return TRAJ_DSL_TEXT

        gen = FlakyGenerator()
        runner = self._runner(generator=gen)
        summary = runner.run()
        self.assertTrue(summary["artifact"]["dsl_valid"])
        self.assertEqual(summary["artifact"]["attempts"], 2)
        self.assertEqual(gen.calls, 2)


class EvaluateEndAnchorTest(unittest.TestCase):
    def test_pending_when_no_metrics(self):
        result = evaluate_end_anchor({"collision": True}, None)
        self.assertEqual(result["status"], "pending")

    def test_pass_when_collision_matches(self):
        result = evaluate_end_anchor({"collision": True}, {"collision_events": [{"frame": 10}]})
        self.assertEqual(result["status"], "pass")

    def test_mismatch_expected_collision_but_none(self):
        result = evaluate_end_anchor({"collision": True}, {"collision_events": []})
        self.assertEqual(result["status"], "mismatch")
        self.assertIn("brake harder", result["repair_hint"])

    def test_mismatch_unexpected_collision(self):
        result = evaluate_end_anchor({"collision": False}, {"collisions": [{"frame": 3}]})
        self.assertEqual(result["status"], "mismatch")
        self.assertIn("no contact", result["repair_hint"])


if __name__ == "__main__":
    unittest.main()
