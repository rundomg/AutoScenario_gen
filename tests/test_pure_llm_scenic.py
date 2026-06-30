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

from tools.pure_llm_scenic_runner import PureLlmScenicRunner


SPAWN_PAYLOAD = {
    "entities": [
        {
            "id": "ego",
            "category": "car",
            "blueprint_name": "vehicle.lincoln.mkz_2017",
            "location": {"x": 61.8, "y": -206.5, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": -178.6, "roll": 0.0},
        },
        {
            "id": "veh_1",
            "category": "car",
            "blueprint_name": "vehicle.tesla.model3",
            "location": {"x": 63.9, "y": -204.9, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": -178.6, "roll": 0.0},
        },
    ]
}

CANDIDATES = {
    "scene_id": "s0000_c0",
    "ego": {"speed_mps": 10.0},
    "participants": [{"id": "p1", "type": "car"}],
    "accident_candidates": [
        {
            "id": "acc_001",
            "accident_type": "rear-end",
            "involved": ["ego", "p1"],
            "mechanism": "lead brakes hard",
        }
    ],
}


def write_static_outputs(tmp: str) -> str:
    scene_id = "s0000_c0"
    root = Path(tmp)
    (root / f"{scene_id}_actors.json").write_text(json.dumps(SPAWN_PAYLOAD), encoding="utf-8")
    (root / f"{scene_id}_match.json").write_text(
        json.dumps({"status": "matched", "world_name": "Carla/Maps/Town03"}),
        encoding="utf-8",
    )
    return scene_id


class FakeGenerator:
    def __init__(self, body="ego = new Car at (0 @ 0)"):
        self.body = body
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.body


class FakeCompile:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, scenic_path, conda_env="scenicNL"):
        self.calls.append((scenic_path, conda_env))
        return self.results.pop(0)


class FakePredictor:
    def __init__(self, payload):
        self.payload = payload

    def call_agent(self, user_request, add_info):
        Path(add_info["output_fn"]).write_text(json.dumps(self.payload), encoding="utf-8")
        return self.payload


class TestPureLlmScenicRunner(unittest.TestCase):
    def test_candidates_mode_writes_scenic_and_compile_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator()
            comp = FakeCompile([{"compiled": True, "returncode": 0, "error": None}])
            runner = PureLlmScenicRunner(
                tmp, scene_id, generator=gen, compile_fn=comp, max_retries=2
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(summary["num_candidates"], 1)
            self.assertEqual(summary["num_compiled"], 1)
            self.assertEqual(summary["map_name"], "Town03")

            scenic_file = Path(tmp) / f"{scene_id}_cand000.scenic"
            self.assertTrue(scenic_file.exists())
            program = scenic_file.read_text(encoding="utf-8")
            self.assertIn("Town03.xodr", program)
            self.assertIn("model scenic.simulators.carla.model", program)
            self.assertIn(gen.body, program)

            compile_file = Path(tmp) / f"{scene_id}_cand000_compile.json"
            self.assertTrue(compile_file.exists())
            self.assertTrue(json.loads(compile_file.read_text())["compiled"])

    def test_compile_failure_triggers_repair_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator()
            comp = FakeCompile(
                [
                    {"compiled": False, "returncode": 1, "error": "syntax error"},
                    {"compiled": True, "returncode": 0, "error": None},
                ]
            )
            runner = PureLlmScenicRunner(
                tmp, scene_id, generator=gen, compile_fn=comp, max_retries=2
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(summary["artifacts"][0]["attempts"], 2)
            self.assertTrue(summary["artifacts"][0]["compiled"])
            # Second generate call must carry the compiler error back for repair.
            self.assertIsNone(gen.calls[0]["compile_error"])
            self.assertEqual(gen.calls[1]["compile_error"], "syntax error")
            self.assertEqual(gen.calls[1]["prior_attempt"], gen.body)

    def test_no_compile_mode_skips_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator()
            comp = FakeCompile([])
            runner = PureLlmScenicRunner(
                tmp, scene_id, generator=gen, compile_fn=comp, enable_compile=False
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(comp.calls, [])
            self.assertEqual(summary["artifacts"][0]["attempts"], 1)
            self.assertIsNone(summary["artifacts"][0]["compiled"])

    def test_predictor_mode_runs_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            gen = FakeGenerator()
            comp = FakeCompile([{"compiled": True, "returncode": 0, "error": None}])
            runner = PureLlmScenicRunner(
                tmp,
                scene_id,
                predictor=FakePredictor(CANDIDATES),
                generator=gen,
                compile_fn=comp,
            )
            summary = runner.run(image_path="dummy.png")

            self.assertTrue((Path(tmp) / f"{scene_id}_candidates.json").exists())
            self.assertEqual(summary["num_candidates"], 1)

    def test_actor_context_carries_absolute_coords_and_blueprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            runner = PureLlmScenicRunner(tmp, scene_id, enable_compile=False)
            context = runner._build_actor_context(SPAWN_PAYLOAD)
            ego = next(a for a in context["actors"] if a["id"] == "ego")
            self.assertEqual(ego["blueprint_name"], "vehicle.lincoln.mkz_2017")
            self.assertEqual(ego["x"], 61.8)
            self.assertEqual(ego["yaw"], -178.6)


if __name__ == "__main__":
    unittest.main()
