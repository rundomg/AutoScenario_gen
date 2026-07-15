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
from agents.video_accident_interpreter import VideoAccidentInterpreter


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
    def __init__(self):
        self.added_info = None

    def call_agent(self, user_request, added_info):
        self.added_info = added_info
        Path(added_info["output_fn"]).write_text(json.dumps(UNDERSTANDING))
        return UNDERSTANDING


class FakeGenerator:
    def __init__(self):
        self.calls = 0
        self.kwargs = None

    def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
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

    def test_user_request_reaches_trajectory_generator(self):
        generator = FakeGenerator()
        runner = self._runner(generator=generator)
        request = "Both cars start stopped; lead brakes exactly 1 s after launch."

        summary = runner.run(user_request=request)

        self.assertEqual(generator.kwargs["user_request"], request)
        self.assertEqual(summary["user_request"], request)

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

    def test_adaptive_manifest_reuses_all_frames_without_extractor(self):
        frame_dir = self.folder / "sampled" / "frames"
        frame_dir.mkdir(parents=True)
        rows = []
        for index, timestamp in enumerate((16.9, 17.333, 19.233)):
            path = frame_dir / f"frame_{index}.jpg"
            path.write_text(str(index))
            rows.append(
                {
                    "sample_index": index,
                    "frame_index": 507 + index,
                    "timestamp_s": timestamp,
                    "window_progress": index / 2,
                    "phase": "early" if index == 0 else "critical",
                    "path": str(path),
                }
            )
        manifest_path = self.folder / "sampled" / "frames_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "video_path": str(self.video),
                    "time_of_alert_s": 16.9,
                    "time_of_event_s": 19.233,
                    "frames": rows,
                }
            )
        )
        interpreter = FakeInterpreter()

        def must_not_extract(*args, **kwargs):
            raise AssertionError("video extraction must not run")

        runner = self._runner(
            video_path=None,
            start_s=None,
            end_s=None,
            frames_manifest_path=str(manifest_path),
            interpreter=interpreter,
            frame_extractor=must_not_extract,
        )
        summary = runner.run()

        self.assertEqual(summary["frame_source"], "adaptive_manifest")
        self.assertEqual(summary["frame_count"], 3)
        self.assertEqual(summary["start_s"], 16.9)
        self.assertEqual(summary["end_s"], 19.233)
        sequence = interpreter.added_info["frame_sequence"]
        self.assertEqual(len(sequence), 3)
        self.assertEqual(
            [row["id"] for row in interpreter.added_info["actor_context"]["actors"]],
            ["ego", "veh_1"],
        )
        self.assertTrue(sequence[0]["is_start"])
        self.assertTrue(sequence[-1]["is_end_anchor"])
        self.assertEqual(sequence[-1]["relative_timestamp_s"], 2.333)


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


class VideoAccidentInterpreterFrameSequenceTest(unittest.TestCase):
    def test_labels_include_timing_and_sampling_metadata(self):
        frames = [
            {
                "path": "/tmp/start.jpg",
                "timestamp_s": 16.9,
                "relative_timestamp_s": 0.0,
                "window_progress": 0.0,
                "phase": "early",
                "frame_index": 507,
            },
            {
                "path": "/tmp/end.jpg",
                "timestamp_s": 19.233,
                "relative_timestamp_s": 2.333,
                "window_progress": 1.0,
                "phase": "critical",
                "frame_index": 577,
            },
        ]
        ordered = VideoAccidentInterpreter._ordered_frames(
            {"frame_sequence": frames}
        )

        self.assertEqual(len(ordered), 2)
        self.assertIn("FRAME 1/2 | START", ordered[0][0])
        self.assertIn("source_t=16.900s", ordered[0][0])
        self.assertIn("t_rel=2.333s", ordered[1][0])
        self.assertIn("phase=critical", ordered[1][0])
        self.assertIn("source_frame=577", ordered[1][0])

    def test_prompt_contains_compact_actor_table(self):
        interpreter = VideoAccidentInterpreter()
        content = interpreter.refine_request(
            "",
            {
                "scene_id": "s0000_c0",
                "actor_context": {
                    "actors": [
                        {
                            "id": "veh_1",
                            "relative_to_ego": {
                                "longitudinal_m": 12.0,
                                "lateral_m": 0.2,
                            },
                        }
                    ]
                },
                "frame_sequence": [],
            },
        )

        prompt = content[0]["text"]
        self.assertIn("Compact CARLA actor table", prompt)
        self.assertIn('"id": "veh_1"', prompt)
        self.assertIn('"longitudinal_m": 12.0', prompt)

    def test_unknown_matched_actor_id_is_rejected(self):
        error = VideoAccidentInterpreter._validate_actor_matches(
            {"participants": [{"matched_actor_id": "invented_car"}]},
            {"actors": [{"id": "ego"}, {"id": "veh_1"}]},
        )

        self.assertIn("unknown actor id", error)
        self.assertIn("veh_1", error)

    def test_motion_classification_must_cover_every_non_ego_vehicle(self):
        error = VideoAccidentInterpreter._validate_actor_matches(
            {
                "participants": [{"matched_actor_id": "veh_1"}],
                "actor_motion_states": [
                    {
                        "actor_id": "veh_1",
                        "motion_state": "stationary",
                        "evidence": "unchanged across frames",
                    }
                ],
            },
            {
                "actors": [
                    {"id": "ego", "category": "car"},
                    {"id": "veh_1", "category": "car"},
                    {"id": "veh_2", "category": "car"},
                ]
            },
        )

        self.assertIn("Missing actor_motion_states", error)
        self.assertIn("veh_2", error)


if __name__ == "__main__":
    unittest.main()
