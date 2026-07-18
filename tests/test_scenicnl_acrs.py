import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "evaluate_scenicnl_acrs.py"
SPEC = importlib.util.spec_from_file_location("evaluate_scenicnl_acrs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EXTRACTOR_PATH = Path(__file__).resolve().parents[1] / "tools" / "scenicnl_sample_extractor.py"
EXTRACTOR_SPEC = importlib.util.spec_from_file_location("scenicnl_sample_extractor", EXTRACTOR_PATH)
EXTRACTOR = importlib.util.module_from_spec(EXTRACTOR_SPEC)
assert EXTRACTOR_SPEC.loader is not None
EXTRACTOR_SPEC.loader.exec_module(EXTRACTOR)


def reference():
    return {
        "schema_version": "acrs-reference-v1",
        "scene_id": "sample",
        "road_topology": {
            "topology_type": "straight", "directionality": "two_way",
            "forward_lane_count": 1, "opposing_lane_count": 1,
            "ego_lane_from_right": 1, "has_center_median": "unknown",
            "left_parking_presence": "unknown", "right_parking_presence": "unknown",
            "junction_visible": False, "junction_type": "none",
            "junction_branches": {}, "branch_count": 0,
        },
        "environment": {
            "weather": "clear", "lighting": "daylight", "time_of_day": "day",
            "road_surface": "dry", "urban_density": "unknown",
            "roadside_context_left": "unknown", "roadside_context_right": "unknown",
            "landmarks_and_controls": "unknown",
        },
        "actors": [{
            "id": "target", "role": "critical", "category": "car",
            "subtype": "unknown", "lane_assignment": "same_lane",
            "lane_from_right": "unknown", "branch_assignment": "unknown",
            "heading_relation": "same_direction", "longitudinal_relation": "ahead",
            "distance_band": "near",
        }],
        "pairwise_relations": [],
        "critical_actor_ids": ["target"],
    }


def candidate():
    return {
        "road_topology": reference()["road_topology"],
        "environment": reference()["environment"],
        "environment_sources": {},
        "actors": [{
            "id": "actor_001", "category": "car", "subtype": "vehicle.tesla.model3",
            "lane_assignment": "same_lane", "lane_from_right": "unknown",
            "branch_assignment": "unknown", "heading_relation": "same_direction",
            "longitudinal_relation": "ahead", "distance_band": "near",
            "longitudinal_m": 8.0, "lateral_m": 0.0,
        }],
        "runtime_truth_available": False,
        "visual_observation_available": False,
    }


class ScenicNLACRSTests(unittest.TestCase):
    def test_sampled_positions_are_converted_to_ego_frame(self):
        class Actor:
            def __init__(self, position, heading=0.0, blueprint="vehicle.tesla.model3"):
                self.position = position
                self.heading = heading
                self.blueprint = blueprint

        class Network:
            def laneAt(self, position, reject=False):
                del position, reject
                return None

            def roadAt(self, position, reject=False):
                del position, reject
                return None

            def intersectionAt(self, position, reject=False):
                del position, reject
                return None

        ego = Actor((0.0, 0.0))
        other = Actor((2.0, 8.0))
        scene = type("Scene", (), {"egoObject": ego, "objects": [ego, other]})()
        workspace = type("Workspace", (), {"network": Network()})()
        scenario = type("Scenario", (), {"workspace": workspace, "params": {"weather": "ClearNoon"}})()
        evidence = EXTRACTOR.scene_to_candidate(scene, scenario)
        actor = evidence["actors"][0]
        self.assertEqual(actor["longitudinal_m"], 8.0)
        self.assertEqual(actor["lateral_m"], 2.0)
        self.assertEqual(actor["lane_assignment"], "right_lane")
        self.assertEqual(actor["distance_band"], "near")
        self.assertEqual(evidence["environment"]["weather"], "clear")

    def test_build_extract_command_is_argument_safe(self):
        command = MODULE.build_extract_command(
            Path("/tmp/a scene.scenic"), Path("/tmp/out.json"),
            conda_executable="/opt/conda/bin/conda", conda_env="scenicNL",
            samples=3, max_iterations=42, seed=7,
        )
        self.assertEqual(command[0], "/opt/conda/bin/conda")
        self.assertIn("/tmp/a scene.scenic", command)
        self.assertEqual(command[-6:], ["--samples", "3", "--max-iterations", "42", "--seed", "7"])

    def test_evaluate_program_aggregates_successes_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenic_path = root / "generated.scenic"
            scenic_path.write_text("# test", encoding="utf-8")
            reference_path = root / "reference.json"
            MODULE.write_json(reference_path, reference())

            def extractor(*args, **kwargs):
                del args, kwargs
                return {
                    "compiled": True, "requested_samples": 2,
                    "samples": [
                        {"sample_index": 0, "seed": 1, "success": True, "candidate": candidate()},
                        {"sample_index": 1, "seed": 2, "success": False, "error": "rejection"},
                    ],
                }

            report = MODULE.evaluate_program(
                scenic_path, reference_path, conda_executable="conda",
                conda_env="scenicNL", samples=2, max_iterations=10,
                seed=1, timeout=10, extractor=extractor,
            )
        self.assertTrue(report["compiled"])
        self.assertEqual(report["successful_samples"], 1)
        self.assertEqual(report["sample_success_rate"], 0.5)
        self.assertFalse(report["official"])
        self.assertEqual(report["status"], "complete_diagnostic")
        self.assertEqual(report["aggregates"]["acrs"]["count"], 1)
        self.assertEqual(report["aggregates"]["critical_traffic_participants"]["count"], 1)
        self.assertEqual(report["aggregates"]["background_traffic_participants"]["count"], 0)
        self.assertEqual(report["samples"][1]["error"], "rejection")

    def test_no_successful_samples_has_no_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenic_path = root / "bad.scenic"
            scenic_path.write_text("bad", encoding="utf-8")
            reference_path = root / "reference.json"
            MODULE.write_json(reference_path, reference())

            def extractor(*args, **kwargs):
                del args, kwargs
                return {
                    "compiled": False, "compile_error": "syntax error",
                    "requested_samples": 1, "samples": [],
                }

            report = MODULE.evaluate_program(
                scenic_path, reference_path, conda_executable="conda",
                conda_env="scenicNL", samples=1, max_iterations=10,
                seed=1, timeout=10, extractor=extractor,
            )
        self.assertEqual(report["status"], "no_successful_samples")
        self.assertIsNone(report["aggregates"]["acrs"]["mean"])


if __name__ == "__main__":
    unittest.main()
