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

from tools.pure_llm_risk_runner import PureLlmRiskRunner
from tools.python_compile import check_python_compile
from tools.risk_dsl import validate_risk_dsl


SPAWN_PAYLOAD = {
    "entities": [
        {
            "id": "ego",
            "category": "car",
            "blueprint_name": "car",
            "spawn_kind": "vehicle",
            "location": {"x": 51.2, "y": 133.8, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": -0.8, "roll": 0.0},
        },
        {
            "id": "ts1",
            "category": "motorcycle",
            "blueprint_name": "motorcycle",
            "spawn_kind": "vehicle",
            "location": {"x": 61.2, "y": 133.7, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": -0.8, "roll": 0.0},
        },
    ]
}

CANDIDATES = {
    "scene_id": "s0000_c0",
    "ego": {"speed_mps": 10.0},
    "participants": [{"id": "p1", "type": "motorcycle"}],
    "accident_candidates": [
        {
            "id": "acc_001",
            "accident_type": "rear-end",
            "involved": ["ego", "p1"],
            "mechanism": "lead brakes hard",
        },
        {
            "id": "acc_002",
            "accident_type": "cut-in",
            "involved": ["ego", "p1"],
            "mechanism": "lead cuts in",
        },
    ],
}

VALID_DSL = {
    "schema_version": "risk-dsl-v1",
    "scene_id": "s0000_c0",
    "accident_type": "rear-end",
    "ego": {"actor_id": "ego", "target_speed_mps": 10.0, "behavior": "drive_forward"},
    "actors": [{"actor_id": "ts1", "role": "lead_vehicle"}],
    "events": [
        {
            "actor_id": "ts1",
            "trigger": {"type": "distance_to_ego_below", "value_m": 12.0},
            "action": {"type": "brake", "intensity": 0.9},
        }
    ],
    "duration_s": 12.0,
    "metrics": ["collision", "min_ttc_s", "min_distance_m"],
}


def write_static_outputs(tmp: str) -> str:
    scene_id = "s0000_c0"
    root = Path(tmp)
    (root / f"{scene_id}_actors.json").write_text(json.dumps(SPAWN_PAYLOAD), encoding="utf-8")
    (root / f"{scene_id}_match.json").write_text(
        json.dumps({"status": "matched", "world_name": "Carla/Maps/Town06"}),
        encoding="utf-8",
    )
    return scene_id


class FakeGenerator:
    def __init__(self, bodies):
        # `bodies` is a queue of raw DSL strings the generator returns in order.
        self.bodies = list(bodies)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.bodies.pop(0)


class FakePredictor:
    def __init__(self, payload):
        self.payload = payload

    def call_agent(self, user_request, add_info):
        Path(add_info["output_fn"]).write_text(json.dumps(self.payload), encoding="utf-8")
        return self.payload


class FakeCodegen:
    def __init__(self):
        self.calls = []

    def build_dsl_risk_scene_script(self, **kwargs):
        self.calls.append(kwargs)
        return "# generated script\nx = 1\n"


class FakeCompile:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, py_path):
        self.calls.append(py_path)
        return self.results.pop(0)


class TestPureLlmRiskRunner(unittest.TestCase):
    def test_candidates_mode_writes_dsl_script_and_compile_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator([json.dumps(VALID_DSL)])
            codegen = FakeCodegen()
            comp = FakeCompile([{"compiled": True, "error": None}])
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                generator=gen,
                codegen=codegen,
                compile_fn=comp,
                num_accidents=1,
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(summary["num_candidates"], 1)
            self.assertEqual(summary["num_dsl_valid"], 1)
            self.assertEqual(summary["num_compiled"], 1)
            self.assertEqual(summary["map_name"], "Town06")

            dsl_file = Path(tmp) / f"{scene_id}_r000_dsl.json"
            self.assertTrue(dsl_file.exists())
            self.assertEqual(json.loads(dsl_file.read_text())["ego"]["target_speed_mps"], 10.0)

            self.assertTrue((Path(tmp) / f"{scene_id}_r000.py").exists())
            # Codegen must receive absolute spawn/dsl/metrics paths.
            self.assertTrue(codegen.calls[0]["spawn_payload_filename"].endswith(".json"))
            self.assertTrue(Path(codegen.calls[0]["dsl_filename"]).is_absolute())

    def test_schema_failure_triggers_repair_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            bad_dsl = dict(VALID_DSL)
            bad_dsl = json.loads(json.dumps(VALID_DSL))
            bad_dsl["events"][0]["actor_id"] = "ghost"  # unknown actor id

            gen = FakeGenerator([json.dumps(bad_dsl), json.dumps(VALID_DSL)])
            codegen = FakeCodegen()
            comp = FakeCompile([{"compiled": True, "error": None}])
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                generator=gen,
                codegen=codegen,
                compile_fn=comp,
                num_accidents=1,
                max_retries=2,
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(summary["artifacts"][0]["attempts"], 2)
            self.assertTrue(summary["artifacts"][0]["dsl_valid"])
            # First attempt has no error context, second carries it back for repair.
            self.assertIsNone(gen.calls[0]["schema_error"])
            self.assertIn("ghost", gen.calls[1]["schema_error"])
            self.assertEqual(gen.calls[1]["prior_attempt"], json.dumps(bad_dsl))

    def test_repair_exhausted_marks_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator(["not json", "still not json"])
            codegen = FakeCodegen()
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                generator=gen,
                codegen=codegen,
                num_accidents=1,
                max_retries=1,
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertFalse(summary["artifacts"][0]["dsl_valid"])
            self.assertIsNone(summary["artifacts"][0]["script_path"])
            self.assertEqual(codegen.calls, [])

    def test_no_compile_mode_skips_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator([json.dumps(VALID_DSL)])
            comp = FakeCompile([])
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                generator=gen,
                codegen=FakeCodegen(),
                compile_fn=comp,
                enable_compile=False,
                num_accidents=1,
            )
            summary = runner.run(candidates_path=str(candidates_path))

            self.assertEqual(comp.calls, [])
            self.assertIsNone(summary["artifacts"][0]["compiled"])
            self.assertTrue(summary["artifacts"][0]["dsl_valid"])

    def test_num_accidents_limits_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            candidates_path = Path(tmp) / "cands.json"
            candidates_path.write_text(json.dumps(CANDIDATES), encoding="utf-8")

            gen = FakeGenerator([json.dumps(VALID_DSL)])
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                generator=gen,
                codegen=FakeCodegen(),
                compile_fn=FakeCompile([{"compiled": True, "error": None}]),
                num_accidents=1,
            )
            summary = runner.run(candidates_path=str(candidates_path))
            self.assertEqual(summary["num_candidates"], 1)
            self.assertEqual(len(summary["artifacts"]), 1)

    def test_predictor_mode_runs_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            gen = FakeGenerator([json.dumps(VALID_DSL)])
            runner = PureLlmRiskRunner(
                tmp,
                scene_id,
                predictor=FakePredictor(CANDIDATES),
                generator=gen,
                codegen=FakeCodegen(),
                compile_fn=FakeCompile([{"compiled": True, "error": None}]),
                num_accidents=1,
            )
            summary = runner.run(image_path="dummy.png")
            self.assertTrue((Path(tmp) / f"{scene_id}_candidates.json").exists())
            self.assertEqual(summary["num_dsl_valid"], 1)


class TestRiskDslValidator(unittest.TestCase):
    def test_unknown_actor_rejected(self):
        dsl = json.loads(json.dumps(VALID_DSL))
        dsl["events"][0]["actor_id"] = "ghost"
        normalized, error = validate_risk_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(normalized)
        self.assertIn("ghost", error)

    def test_bad_trigger_type_rejected(self):
        dsl = json.loads(json.dumps(VALID_DSL))
        dsl["events"][0]["trigger"] = {"type": "telepathy"}
        normalized, error = validate_risk_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(normalized)
        self.assertIn("trigger.type", error)

    def test_default_ego_speed_filled(self):
        dsl = json.loads(json.dumps(VALID_DSL))
        dsl["ego"].pop("target_speed_mps")
        normalized, error = validate_risk_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(normalized["ego"]["target_speed_mps"], 10.0)

    def test_event_targeting_ego_rejected(self):
        dsl = json.loads(json.dumps(VALID_DSL))
        dsl["events"][0]["actor_id"] = "ego"
        normalized, error = validate_risk_dsl(dsl, SPAWN_PAYLOAD)
        self.assertIsNone(normalized)
        self.assertIn("non-ego", error)


class TestRealCodegenCompiles(unittest.TestCase):
    def test_generated_script_passes_py_compile(self):
        from agents.existing_world_scenario_generator import ExistingWorldScenarioGenerator

        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            (Path(tmp) / f"{scene_id}_r000_dsl.json").write_text(
                json.dumps(VALID_DSL), encoding="utf-8"
            )
            generator = ExistingWorldScenarioGenerator()
            script = generator.build_dsl_risk_scene_script(
                spawn_payload_filename=str(Path(tmp) / f"{scene_id}_actors.json"),
                dsl_filename=str(Path(tmp) / f"{scene_id}_r000_dsl.json"),
                risk_metrics_filename=str(Path(tmp) / f"{scene_id}_r000_metrics.json"),
                carla_map="Town06",
            )
            self.assertIn("client.set_timeout(30.0)", script)
            script_path = Path(tmp) / f"{scene_id}_r000.py"
            script_path.write_text(script, encoding="utf-8")
            result = check_python_compile(str(script_path))
            self.assertTrue(result["compiled"], result.get("error"))


if __name__ == "__main__":
    unittest.main()
