import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from agents.acrs_visual_extractor import ACRSVisualExtractor
from tools.acrs_evaluator import (
    ACRSReferenceError,
    _derive_pairwise,
    build_candidate_evidence,
    evaluate_acrs,
    reference_from_scene_understanding,
)
from experiments.evaluate_acrs import run_scene, write_summary


def reference():
    return {
        "schema_version": "acrs-reference-v1",
        "scene_id": "sample",
        "road_topology": {
            "topology_type": "t_junction", "directionality": "two_way",
            "forward_lane_count": 2, "opposing_lane_count": 1,
            "ego_lane_from_right": "unknown", "has_center_median": False,
            "left_parking_presence": "unknown", "right_parking_presence": "unknown",
            "junction_visible": True, "junction_type": "t_junction",
            "junction_branches": {"ahead": True, "left": True, "right": True, "uturn": False},
            "branch_count": 3,
        },
        "environment": {
            "weather": "clear", "lighting": "daylight", "time_of_day": "day",
            "road_surface": "dry", "urban_density": "urban_like",
            "roadside_context_left": [], "roadside_context_right": [],
            "landmarks_and_controls": ["buildings", "traffic_control"],
        },
        "actors": [
            {
                "id": "danger", "role": "critical", "category": "car",
                "lane_assignment": "left_lane", "lane_from_right": "unknown",
                "branch_assignment": "unknown", "heading_relation": "opposite_direction",
                "longitudinal_relation": "ahead", "distance_band": "far",
            },
            {
                "id": "background", "role": "background", "category": "truck",
                "lane_assignment": "right_lane", "lane_from_right": "unknown",
                "branch_assignment": "unknown", "heading_relation": "same_direction",
                "longitudinal_relation": "ahead", "distance_band": "mid",
            },
        ],
        "critical_actor_ids": ["danger"],
        "pairwise_relations": [
            {"entity_id": "danger", "other_entity_id": "background", "longitudinal_relation": "ahead_of_other"}
        ],
    }


def candidate():
    return {
        "road_topology": {
            "topology_type": "t_junction", "directionality": "two_way",
            "forward_lane_count": 2, "opposing_lane_count": 1,
            "ego_lane_from_right": "unknown", "has_center_median": False,
            "left_parking_presence": "unknown", "right_parking_presence": "unknown",
            "junction_visible": True, "junction_type": "t_junction",
            "junction_branches": {"ahead": True, "left": True, "right": True, "uturn": False},
            "branch_count": 3,
        },
        "environment": {
            "weather": "clear", "lighting": "daylight", "time_of_day": "day",
            "road_surface": "dry", "urban_density": "urban_like",
            "roadside_context_left": [], "roadside_context_right": [],
            "landmarks_and_controls": ["buildings", "traffic_control"],
        },
        "environment_sources": {},
        "actors": [
            {
                "id": "candidate_a", "category": "car", "lane_assignment": "left_lane",
                "heading_relation": "opposite_direction", "longitudinal_relation": "ahead",
                "distance_band": "far", "longitudinal_m": 30.0, "lateral_m": -3.5,
            },
            {
                "id": "candidate_b", "category": "truck", "lane_assignment": "right_lane",
                "heading_relation": "same_direction", "longitudinal_relation": "ahead",
                "distance_band": "mid", "longitudinal_m": 15.0, "lateral_m": 3.5,
            },
        ],
        "runtime_truth_available": True,
        "visual_observation_available": True,
        "evidence_conflicts": [],
    }


class ACRSEvaluatorTests(unittest.TestCase):
    def test_rich_pairwise_lane_relations_are_derived(self):
        same_parking_a = {
            "lane_assignment": "right_parking_lane", "heading_relation": "same_direction",
            "lateral_m": 4.2,
        }
        same_parking_b = {
            "lane_assignment": "right_parking_lane", "heading_relation": "same_direction",
            "lateral_m": 4.4,
        }
        self.assertEqual(
            _derive_pairwise(same_parking_a, same_parking_b, "lane_relation"),
            "same_parking_lane",
        )
        adjacent = {
            "lane_assignment": "right_lane", "heading_relation": "same_direction",
            "lateral_m": 4.3,
        }
        driving = {
            "lane_assignment": "same_lane", "heading_relation": "same_direction",
            "lateral_m": 0.0,
        }
        self.assertEqual(
            _derive_pairwise(driving, adjacent, "lane_relation"),
            "adjacent_right_lane",
        )
        opposing = {
            "lane_assignment": "left_parking_lane", "heading_relation": "opposite_direction",
            "lateral_m": -7.9,
        }
        self.assertEqual(
            _derive_pairwise(driving, opposing, "lane_relation"),
            "cross_lane",
        )

    def test_candidate_builder_preserves_realized_parking_lane(self):
        graph = {
            "truth_source": "carla_actor_transform", "metadata": {},
            "actors": [{
                "id": "parked", "spawned": True, "category": "car",
                "lane_side_relation": "right_parking_lane",
                "visual_position_result": {"result": "parking_lane", "target_lane": "right_parking_lane"},
                "ego_frame": {"lane_side_relation": "right_lane", "lateral_m": 4.5},
            }],
        }
        evidence = build_candidate_evidence({}, {"metadata": {}}, graph)
        self.assertEqual(evidence["actors"][0]["lane_assignment"], "right_parking_lane")
        self.assertTrue(evidence["actors"][0]["is_parking_lane"])

    def test_candidate_builder_preserves_confirmed_driving_lane_after_collision_shift(self):
        graph = {
            "truth_source": "carla_actor_transform", "metadata": {},
            "actors": [{
                "id": "shifted", "spawned": True, "category": "car",
                "ego_frame": {"lane_side_relation": "left_lane", "lateral_m": -1.5},
                "actual_waypoint": {"road_id": 30, "lane_id": -1, "lane_type": "Driving"},
                "visual_position_result": {
                    "result": "driving_lane", "target_lane": "same_lane",
                    "basis_lane": {"road_id": 30, "lane_id": -1},
                    "collision_adjustment_m": 2.121,
                },
            }],
        }
        evidence = build_candidate_evidence({}, {"metadata": {}}, graph)
        self.assertEqual(evidence["actors"][0]["lane_assignment"], "same_lane")

    def test_candidate_builder_rejects_unconfirmed_driving_lane_target(self):
        graph = {
            "truth_source": "carla_actor_transform", "metadata": {},
            "actors": [{
                "id": "shifted", "spawned": True, "category": "car",
                "ego_frame": {"lane_side_relation": "left_lane", "lateral_m": -3.5},
                "actual_waypoint": {"road_id": 30, "lane_id": -2, "lane_type": "Driving"},
                "visual_position_result": {
                    "result": "driving_lane", "target_lane": "same_lane",
                    "basis_lane": {"road_id": 30, "lane_id": -1},
                },
            }],
        }
        evidence = build_candidate_evidence({}, {"metadata": {}}, graph)
        self.assertEqual(evidence["actors"][0]["lane_assignment"], "left_lane")

    def test_pairwise_report_names_both_actors(self):
        report = evaluate_acrs(reference(), candidate(), render_image_available=True)
        checks = report["dimensions"]["traffic_participants"]["roles"]["critical"]["pairwise_checks"]
        self.assertEqual(checks[0]["entity_id"], "danger")
        self.assertEqual(checks[0]["other_entity_id"], "background")

    def test_perfect_scene_scores_100(self):
        report = evaluate_acrs(reference(), candidate(), render_image_available=True)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["scores"]["road_topology"], 100.0)
        self.assertEqual(report["scores"]["background_environment"], 100.0)
        self.assertEqual(report["scores"]["critical_traffic_participants"], 100.0)
        self.assertEqual(report["scores"]["background_traffic_participants"], 100.0)
        self.assertEqual(report["scores"]["traffic_participants"], 100.0)
        self.assertEqual(report["scores"]["acrs"], 100.0)

    def test_lane_from_right_normalizes_numeric_string_and_integer(self):
        ref = reference()
        changed = candidate()
        ref["actors"][0]["lane_from_right"] = "1"
        changed["actors"][0]["lane_from_right"] = 1
        report = evaluate_acrs(ref, changed, render_image_available=True)
        self.assertEqual(
            report["scores"]["critical_traffic_participants"],
            100.0,
        )

    def test_lane_count_difference_is_continuous(self):
        changed = candidate()
        changed["road_topology"]["forward_lane_count"] = 3
        report = evaluate_acrs(reference(), changed, render_image_available=True)
        self.assertLess(report["scores"]["road_topology"], 100)
        self.assertGreater(report["scores"]["road_topology"], 90)
        self.assertEqual(report["scores"]["background_environment"], 100)

    def test_wrong_weather_only_changes_background(self):
        changed = candidate()
        changed["environment"]["weather"] = "rain"
        report = evaluate_acrs(reference(), changed, render_image_available=True)
        self.assertEqual(report["scores"]["road_topology"], 100)
        self.assertLess(report["scores"]["background_environment"], 100)
        self.assertEqual(report["scores"]["traffic_participants"], 100)

    def test_missing_critical_actor_reduces_traffic(self):
        changed = candidate()
        changed["actors"] = changed["actors"][1:]
        report = evaluate_acrs(reference(), changed, render_image_available=True)
        self.assertLess(report["scores"]["critical_traffic_participants"], 50)
        self.assertLess(report["scores"]["traffic_participants"], 50)
        self.assertIn("danger", report["dimensions"]["traffic_participants"]["missing_reference_actors"])

    def test_extra_background_actor_hurts_background_f1(self):
        baseline = evaluate_acrs(reference(), candidate(), render_image_available=True)
        changed = candidate()
        changed["actors"].append({"id": "extra", "category": "car"})
        report = evaluate_acrs(reference(), changed, render_image_available=True)
        self.assertEqual(
            report["scores"]["critical_traffic_participants"],
            baseline["scores"]["critical_traffic_participants"],
        )
        self.assertLess(
            report["scores"]["background_traffic_participants"],
            baseline["scores"]["background_traffic_participants"],
        )
        self.assertLess(report["scores"]["traffic_participants"], baseline["scores"]["traffic_participants"])

    def test_unknown_reference_is_skipped(self):
        ref = reference()
        ref["environment"]["weather"] = "unknown"
        changed = candidate()
        changed["environment"]["weather"] = "rain"
        report = evaluate_acrs(ref, changed, render_image_available=True)
        self.assertEqual(report["scores"]["background_environment"], 100)
        self.assertLess(report["coverage"]["background_environment"], 1)

    def test_missing_runtime_evidence_is_incomplete(self):
        changed = candidate()
        changed["runtime_truth_available"] = False
        report = evaluate_acrs(reference(), changed, render_image_available=False)
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["official"])
        self.assertIsNotNone(report["scores"]["acrs"])
        self.assertEqual(len(report["incomplete_reasons"]), 2)

    def test_candidate_builder_ignores_target_fields(self):
        match = {
            "best_match": {
                "candidate_features": {
                    "candidate_topology_type": "straight_road",
                    "same_direction_lane_count": 1, "same_road_lane_count": 1,
                },
                "score_details": {
                    "target_topology_type": "cross_intersection",
                    "target_forward_lane_count": 9,
                },
            }
        }
        evidence = build_candidate_evidence(match, {"metadata": {}}, {"truth_source": "spawn_payload_fallback"})
        self.assertEqual(evidence["road_topology"]["topology_type"], "straight")
        self.assertEqual(evidence["road_topology"]["forward_lane_count"], 1)

    def test_reference_requires_critical_actor(self):
        ref = reference()
        ref["critical_actor_ids"] = []
        with self.assertRaises(ACRSReferenceError):
            evaluate_acrs(ref, candidate(), render_image_available=True)

    def test_reference_prefill_is_draft(self):
        draft = reference_from_scene_understanding({
            "road_network": {"map_matching": {"topology_type": "straight_road"}},
            "general_environment": {"weather_hint": "clear"},
            "traffic_subjects": [{"id": "v1", "category": "car"}],
        }, "s1")
        self.assertEqual(draft["schema_version"], "acrs-reference-v1")
        self.assertEqual(draft["annotation"]["status"], "draft_requires_human_review")
        self.assertEqual(draft["critical_actor_ids"], [])


class ACRSVisualExtractorTests(unittest.TestCase):
    def test_normalization_rejects_free_form_enums_and_scores(self):
        payload = ACRSVisualExtractor.normalize({
            "weather": "sunny-ish", "lighting": "daylight", "score": 99,
            "roadside_context_left": ["Tall Buildings", "Tall Buildings"],
            "confidence": {"lighting": "high"},
        })
        self.assertEqual(payload["weather"], "unknown")
        self.assertEqual(payload["roadside_context_left"], ["tall_buildings"])
        self.assertNotIn("score", payload)

    def test_environment_normalization_excludes_parked_actors_and_unifies_housing(self):
        payload = ACRSVisualExtractor.normalize({
            "roadside_context_left": [
                "parked_cars", "parkingcar", "residential_houses", "vegetation",
            ],
            "roadside_context_right": ["residential_building"],
        })
        self.assertEqual(payload["roadside_context_left"], ["residential_building", "vegetation"])
        self.assertEqual(payload["roadside_context_right"], ["residential_building"])

    def test_environment_score_does_not_double_count_parked_cars(self):
        ref = reference()
        ref["environment"]["roadside_context_left"] = ["residential_houses"]
        changed = candidate()
        changed["environment"]["roadside_context_left"] = [
            "residential_building", "parked_cars",
        ]
        report = evaluate_acrs(ref, changed, render_image_available=True)
        item = report["dimensions"]["background_environment"]["details"]["roadside_context_left"]
        self.assertEqual(item["score"], 1.0)
        self.assertEqual(item["reference"], ["residential_building"])

    def test_parse_fenced_json(self):
        payload = ACRSVisualExtractor.parse_response(
            '```json\n{"weather":"clear","confidence":{"weather":"high"}}\n```'
        )
        self.assertEqual(payload["weather"], "clear")
        self.assertEqual(payload["confidence"]["weather"], "high")


class ACRSCLITests(unittest.TestCase):
    def test_run_scene_with_cached_visual_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene_id = "sample"
            ref_path = root / "reference.json"
            ref_path.write_text(json.dumps(reference()), encoding="utf-8")
            (root / f"{scene_id}_match.json").write_text(json.dumps({
                "best_match": {
                    "candidate_features": {
                        "candidate_topology_type": "t_junction",
                        "same_direction_lane_count": 2, "same_road_lane_count": 3,
                        "has_center_median_candidate": False,
                        "environment_context": {"environment_class": "urban_like", "buildings_nearby": True, "traffic_control_nearby": True},
                    },
                    "score_details": {"branch_direction_detail": {"candidate": {"ahead": True, "left": True, "right": True, "uturn": False, "branch_count": 3}}},
                }
            }), encoding="utf-8")
            (root / f"{scene_id}_actors.json").write_text(json.dumps({
                "metadata": {"carla_weather_preset": "ClearNoon"}
            }), encoding="utf-8")
            graph_path = root / f"{scene_id}_render_actor_graph_r1.json"
            graph_path.write_text(json.dumps({
                "truth_source": "carla_actor_transform", "metadata": {"truth_unavailable": False},
                "actors": [
                    {"id": "a", "spawned": True, "category": "car", "subtype": "car", "heading_relation_to_ego": "opposite_direction", "ego_frame": {"lane_side_relation": "left_lane", "longitudinal_m": 30, "lateral_m": -3.5, "distance_band": "far"}},
                    {"id": "b", "spawned": True, "category": "truck", "subtype": "truck", "heading_relation_to_ego": "same_direction", "ego_frame": {"lane_side_relation": "right_lane", "longitudinal_m": 15, "lateral_m": 3.5, "distance_band": "mid"}},
                ],
            }), encoding="utf-8")
            image_path = root / f"{scene_id}_ego_r1.png"
            image_path.write_bytes(b"cached observation means image decoding is not needed")
            visual_path = root / "visual.json"
            visual_path.write_text(json.dumps({
                "weather": "clear", "lighting": "daylight", "time_of_day": "day",
                "road_surface": "dry", "urban_density": "urban_like",
                "roadside_context_left": [], "roadside_context_right": [],
                "landmarks_and_controls": ["buildings", "traffic_control"],
                "confidence": {}, "prompt_version": "test",
            }), encoding="utf-8")
            args = Namespace(
                result_dir=str(root), scene_id=scene_id, reference=str(ref_path),
                render_graph="auto", render_image="auto", visual_observation=str(visual_path),
                output_dir=str(root), mode="hybrid", skip_vlm=True,
                request_timeout=None, request_retries=None,
            )
            report = run_scene({}, args)
            self.assertEqual(report["status"], "complete")
            self.assertTrue(report["official"])
            self.assertIsNotNone(report["scores"]["acrs"])
            self.assertTrue((root / f"{scene_id}_acrs.json").is_file())

    def test_summary_separates_visual_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [
                {"scene_id": "a", "official": True, "scores": {"acrs": 90}, "inputs": {"visual_model": "m1", "visual_prompt_version": "p1"}},
                {"scene_id": "b", "official": True, "scores": {"acrs": 80}, "inputs": {"visual_model": "m2", "visual_prompt_version": "p1"}},
            ]
            write_summary(rows, Path(folder))
            summary = json.loads((Path(folder) / "acrs_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["mixed_visual_versions"])
            self.assertIsNone(summary["aggregates"])
            self.assertEqual(set(summary["aggregates_by_visual_version"]), {"m1::p1", "m2::p1"})


if __name__ == "__main__":
    unittest.main()
