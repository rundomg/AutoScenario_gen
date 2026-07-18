import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "evaluate_trafficcomposer_acrs.py"
SPEC = importlib.util.spec_from_file_location("evaluate_trafficcomposer_acrs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def reference():
    return {
        "schema_version": "acrs-reference-v1",
        "scene_id": "sample",
        "road_topology": {
            "topology_type": "intersection", "directionality": "unknown",
            "forward_lane_count": "unknown", "opposing_lane_count": "unknown",
            "ego_lane_from_right": "unknown", "has_center_median": "unknown",
            "left_parking_presence": "unknown", "right_parking_presence": "unknown",
            "junction_visible": True, "junction_type": "intersection",
            "junction_branches": "unknown", "branch_count": "unknown",
        },
        "environment": {
            "weather": "rain", "lighting": "night", "time_of_day": "night",
            "road_surface": "unknown", "urban_density": "unknown",
            "roadside_context_left": "unknown", "roadside_context_right": "unknown",
            "landmarks_and_controls": ["traffic_light"],
        },
        "actors": [{
            "id": "lead", "role": "critical", "category": "car", "subtype": "unknown",
            "lane_assignment": "same_lane", "lane_from_right": "unknown",
            "branch_assignment": "unknown", "heading_relation": "same_direction",
            "longitudinal_relation": "ahead", "distance_band": "unknown",
        }],
        "pairwise_relations": [],
        "critical_actor_ids": ["lead"],
    }


def traffic_ir():
    return {
        "environment": {"weather": "rainy", "time": "nighttime"},
        "road_network": {
            "road_type": "intersection", "traffic_sign": None,
            "traffic_light": "green", "lane_number": 4,
        },
        "participant": {
            "ego_vehicle": {"lane_idx": 2, "current_behavior": "go forward"},
            "other_actor_1": {
                "type": "car", "lane_idx": 2, "current_behavior": "go forward",
                "position_target": "ego vehicle", "position_relation": "front",
            },
            "other_actor_2": {
                "type": "cyclist", "lane_idx": 1, "current_behavior": "crossing",
                "position_target": ["left", "ego vehicle"], "position_relation": "left front",
            },
        },
    }


class TrafficComposerACRSTests(unittest.TestCase):
    def test_ir_mapping_preserves_only_supported_facts(self):
        candidate = MODULE.trafficcomposer_ir_to_candidate(traffic_ir())
        self.assertEqual(candidate["road_topology"]["topology_type"], "intersection")
        self.assertEqual(candidate["road_topology"]["forward_lane_count"], "unknown")
        self.assertEqual(candidate["adapter_diagnostics"]["reported_total_lane_number"], 4)
        self.assertEqual(candidate["environment"]["weather"], "rain")
        self.assertEqual(candidate["environment"]["lighting"], "night")
        self.assertEqual(candidate["environment"]["landmarks_and_controls"], ["traffic_light"])
        self.assertEqual(candidate["actors"][0]["lane_assignment"], "same_lane")
        self.assertEqual(candidate["actors"][0]["longitudinal_relation"], "ahead")
        self.assertEqual(candidate["actors"][1]["category"], "bicycle")
        self.assertEqual(candidate["actors"][1]["lane_assignment"], "left_lane")
        self.assertFalse(candidate["runtime_truth_available"])

    def test_yaml_ir_evaluation_is_never_official(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ir_path = root / "scene.yaml"
            ir_path.write_text(MODULE.yaml.safe_dump(traffic_ir()), encoding="utf-8")
            reference_path = root / "reference.json"
            MODULE.write_json(reference_path, reference())
            report = MODULE.evaluate_ir(ir_path, reference_path)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["official"])
        self.assertIsNotNone(report["scores"]["acrs"])
        self.assertIn("CARLA runtime actor graph is unavailable.", report["incomplete_reasons"])

    def test_batch_summary_includes_diagnostic_scores(self):
        report = MODULE.evaluate_acrs(
            reference(), MODULE.trafficcomposer_ir_to_candidate(traffic_ir()),
            render_image_available=False, mode="structured",
        )
        report["scene_id"] = "sample"
        with tempfile.TemporaryDirectory() as folder:
            MODULE.write_summary([report], Path(folder))
            summary = MODULE.load_json(Path(folder) / "trafficcomposer_acrs_summary.json")
        self.assertEqual(summary["scored_scene_count"], 1)
        self.assertEqual(summary["official_scene_count"], 0)
        self.assertEqual(summary["aggregates"]["acrs"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
