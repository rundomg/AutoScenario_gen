import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from opendrive_experiment.agents.scene_understanding_interpreter import (
    SceneUnderstandingInterpreter,
)
from opendrive_experiment.agents.scenario_generator_xodr import XODRScenarioGenerator
from opendrive_experiment.auto_generate_all_vlm_xodr import AutoGenerator
from opendrive_experiment.tools.structured_pipeline import (
    apply_pairwise_ordering,
    build_road_generation_request,
    build_projected_spawn_payload,
    build_relation_dsl,
    evaluate_coordinate_program,
    normalize_scene_understanding,
    project_entities_to_xodr,
    refine_projected_coordinates_with_pairwise_relations,
    validate_relation_layout,
    validate_pairwise_relations,
    write_coordinate_program,
)


def _sample_scene_understanding():
    return {
        "traffic_subjects": [
            {
                "id": "motorcycle_main",
                "category": "motorcycle",
                "subtype": "scooter",
                "appearance": {"color": "blue", "color_confidence": "high"},
                "visual_confidence": "high",
                "motion_state": "moving",
                "heading_relation_to_ego": "same_direction",
                "lane_side_relation": "left_lane",
                "longitudinal_relation": "ahead",
                "longitudinal_proximity": "immediate",
                "count": 1,
                "evidence": "Two-rider scooter ahead-left",
                "must_reconstruct": True,
            },
            {
                "id": "cone_row",
                "category": "cone_group",
                "subtype": "constructioncone",
                "visual_confidence": "high",
                "motion_state": "stopped",
                "heading_relation_to_ego": "same_direction",
                "lane_side_relation": "left_edge",
                "longitudinal_relation": "ahead",
                "longitudinal_proximity": "near",
                "count": 3,
                "evidence": "Cones narrowing the roadside",
                "must_reconstruct": True,
            },
        ],
        "background_traffic": [
            {
                "id": "parked_row",
                "category": "parked_vehicle",
                "appearance": {"color": "white", "color_confidence": "medium"},
                "source": "observed",
                "representative_count": 3,
                "lane_side_relation": "right_edge",
                "longitudinal_band": "mid",
                "motion_bias": "parked",
                "heading_relation_to_ego": "same_direction",
                "density_role": "curbside_row",
                "confidence": "medium",
                "spawn_priority": "low",
            },
            {
                "id": "sidewalk_people",
                "category": "pedestrian",
                "source": "observed",
                "representative_count": 2,
                "lane_side_relation": "sidewalk_left",
                "longitudinal_band": "near",
                "motion_bias": "mixed",
                "heading_relation_to_ego": "crossing",
                "density_role": "sidewalk_group",
                "confidence": "medium",
                "spawn_priority": "low",
            },
        ],
        "key_pairwise_relations": [
            {
                "entity_id": "parked_row",
                "other_entity_id": "motorcycle_main",
                "longitudinal_relation": "ahead_of_other",
                "longitudinal_gap_band": "near",
                "lateral_relation": "right_of_other",
                "lane_relation": "cross_lane",
                "constraint_strength": "hard",
                "confidence": "high",
                "evidence": "Parked cars appear farther ahead than the near scooter.",
            }
        ],
        "road_network": {
            "road_type": "urban_straight",
            "directionality": "two_way",
            "road_segments": [
                {
                    "id": "road_1",
                    "geometry_type": "line",
                    "curvature_hint": "straight",
                    "relative_length": "long",
                }
            ],
            "lane_groups": [
                {
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "left_parking_lane_count": 1,
                    "right_parking_lane_count": 1,
                    "lane_width_class": "standard",
                    "lane_count_evidence": "Visible outer parking-lane separators on both sides.",
                    "lane_count_confidence": "high",
                }
            ],
            "lane_markings": {
                "centerline": "single",
                "edge_lines": "visible",
                "lane_dividers": "single",
            },
            "special_road_areas": ["crosswalk", "parking_strip"],
            "junctions": [],
            "roadside_boundaries": {
                "left": ["curb", "sidewalk"],
                "right": ["curb", "parking_edge", "sidewalk"],
            },
            "control_elements": [
                {"type": "traffic_light", "state": "red", "relation": "ahead"}
            ],
        },
        "general_environment": {
            "weather_hint": "clear",
            "lighting_hint": "daylight",
            "time_of_day_hint": "day",
            "urban_density": "urban",
            "roadside_context_left": ["storefronts"],
            "roadside_context_right": ["parked cars"],
            "occlusion_notes": [],
            "non_spawnable_landmarks": ["overhead sign"],
        },
        "metadata": {"schema_version": "scene-understanding-v1", "input_type": "image"},
    }


def _sample_spawn_context():
    return {
        "spawn_points": [
            {
                "index": 0,
                "location": {"x": 0.0, "y": -1.75, "z": 0.0},
                "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            }
        ],
        "topology_sample": [
            {
                "road_id": 1,
                "lane_id": -1,
                "start": {"x": 0.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
                "end": {"x": 80.0, "y": -1.75, "z": 0.0, "yaw": 0.0, "is_junction": False},
            },
            {
                "road_id": 1,
                "lane_id": 1,
                "start": {"x": 0.0, "y": 1.75, "z": 0.0, "yaw": 180.0, "is_junction": False},
                "end": {"x": 80.0, "y": 1.75, "z": 0.0, "yaw": 180.0, "is_junction": False},
            },
        ],
    }


class TestSceneUnderstandingNormalization(unittest.TestCase):
    def test_normalizes_valid_payload(self):
        payload, error = normalize_scene_understanding(_sample_scene_understanding())
        self.assertIsNone(error)
        self.assertEqual(payload["traffic_subjects"][0]["must_reconstruct"], True)
        self.assertEqual(payload["background_traffic"][0]["representative_count"], 3)
        self.assertEqual(payload["traffic_subjects"][0]["appearance"]["color"], "blue")
        self.assertEqual(payload["background_traffic"][0]["appearance"]["color_rgb"], "255,255,255")
        self.assertEqual(payload["key_pairwise_relations"][0]["relation_source"], "vlm")
        lane_group = payload["road_network"]["lane_groups"][0]
        self.assertEqual(lane_group["left_parking_lane_count"], 1)
        self.assertEqual(lane_group["right_parking_lane_count"], 1)
        self.assertEqual(lane_group["lane_count_confidence"], "high")

    def test_lane_group_defaults_keep_backward_compatibility(self):
        sample = _sample_scene_understanding()
        sample["road_network"]["lane_groups"] = [
            {
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
                "lane_width_class": "standard",
            }
        ]
        payload, error = normalize_scene_understanding(sample)
        self.assertIsNone(error)
        lane_group = payload["road_network"]["lane_groups"][0]
        self.assertEqual(lane_group["left_parking_lane_count"], 0)
        self.assertEqual(lane_group["right_parking_lane_count"], 0)
        self.assertEqual(lane_group["lane_count_evidence"], "")
        self.assertEqual(lane_group["lane_count_confidence"], "unknown")

    def test_infers_cone_group_from_control_elements(self):
        sample = _sample_scene_understanding()
        sample["traffic_subjects"] = [
            entity for entity in sample["traffic_subjects"] if entity["category"] != "cone_group"
        ]
        sample["road_network"]["control_elements"].append(
            {"type": "traffic_cones", "position": "left_edge_ahead", "confidence": "high"}
        )
        payload, error = normalize_scene_understanding(sample)
        self.assertIsNone(error)
        cone_group = next(
            entity for entity in payload["traffic_subjects"] if entity["category"] == "cone_group"
        )
        self.assertEqual(cone_group["lane_side_relation"], "left_edge")

    def test_control_element_location_corrects_existing_inferred_cone_side(self):
        sample = _sample_scene_understanding()
        sample["traffic_subjects"] = [
            {
                "id": "inferred_cone_group",
                "category": "cone_group",
                "subtype": "constructioncone",
                "visual_confidence": "medium",
                "motion_state": "stopped",
                "heading_relation_to_ego": "same_direction",
                "lane_side_relation": "right_edge",
                "longitudinal_relation": "ahead",
                "longitudinal_proximity": "near",
                "count": 3,
                "evidence": "Inferred from visible traffic cones recorded in road_network.control_elements.",
                "must_reconstruct": True,
            }
        ]
        sample["road_network"]["control_elements"] = [
            {"type": "traffic_cones", "location": "left_edge_mid_ahead", "confidence": "high"}
        ]
        payload, error = normalize_scene_understanding(sample)
        self.assertIsNone(error)
        cone_group = next(
            entity for entity in payload["traffic_subjects"] if entity["id"] == "inferred_cone_group"
        )
        self.assertEqual(cone_group["lane_side_relation"], "left_edge")

    def test_rejects_missing_road_network(self):
        broken = _sample_scene_understanding()
        broken.pop("road_network")
        payload, error = normalize_scene_understanding(broken)
        self.assertIsNone(payload)
        self.assertIn("road_network", error)

    def test_interpreter_extracts_and_normalizes_fenced_json(self):
        interpreter = SceneUnderstandingInterpreter()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            path.write_text(
                "```json\n" + json.dumps(_sample_scene_understanding()) + "\n```",
                encoding="utf-8",
            )
            payload, error = interpreter.extract_decision_data(str(path))
            self.assertIsNone(error)
            self.assertEqual(payload["metadata"]["input_type"], "image")

    def test_road_generation_request_prioritizes_lane_first_evidence(self):
        request = build_road_generation_request(_sample_scene_understanding())
        self.assertIn("Determine lane count from lane-first evidence", request)
        self.assertIn("left_parking_lane_count", request)
        self.assertIn("right_parking_lane_count", request)
        self.assertIn("generate explicit `parking` lanes", request)


class TestStructuredPipeline(unittest.TestCase):
    def test_relation_dsl_expands_groups(self):
        relation = build_relation_dsl(
            _sample_scene_understanding(),
            _sample_spawn_context(),
            road_artifact={"summary": "simple road"},
        )
        entity_ids = {entity["entity_id"] for entity in relation["entities"]}
        self.assertIn("motorcycle_main", entity_ids)
        self.assertIn("cone_row_0", entity_ids)
        self.assertIn("parked_row_2", entity_ids)
        self.assertIn("sidewalk_people_1", entity_ids)
        motorcycle = next(
            entity for entity in relation["entities"] if entity["entity_id"] == "motorcycle_main"
        )
        self.assertEqual(motorcycle["appearance"]["color"], "blue")
        self.assertTrue(relation["pairwise_relations"])
        self.assertTrue(relation["ego_relations"])
        self.assertIn("lane_context", relation)
        self.assertIn("ego_lane_id", relation["lane_context"])
        self.assertEqual(relation["lane_context"]["lane_roles"]["left_parking_lane_count"], 1)
        self.assertEqual(relation["lane_context"]["lane_roles"]["right_parking_lane_count"], 1)

    def test_lane_context_prefers_explicit_parking_lane_roles(self):
        relation = build_relation_dsl(
            _sample_scene_understanding(),
            _sample_spawn_context(),
            road_artifact={"summary": "simple road"},
        )
        lane_catalog = relation["lane_context"]["lane_catalog"]
        self.assertEqual(lane_catalog["left_edge"]["role"], "parking")
        self.assertEqual(lane_catalog["right_edge"]["role"], "parking")

    def test_normalization_maps_opposing_and_parked_two_wheeler_labels(self):
        sample = _sample_scene_understanding()
        sample["background_traffic"].append(
            {
                "id": "opposing_car",
                "category": "car",
                "source": "observed",
                "representative_count": 1,
                "lane_side_relation": "opposing_center_left",
                "longitudinal_band": "far",
                "motion_bias": "moving",
                "heading_relation_to_ego": "opposite_direction",
                "density_role": "opposing_flow",
                "confidence": "medium",
                "spawn_priority": "low",
            }
        )
        sample["background_traffic"].append(
            {
                "id": "parked_two_wheelers",
                "category": "parked_two_wheeler_group",
                "source": "observed",
                "representative_count": 1,
                "lane_side_relation": "sidewalk_right",
                "longitudinal_band": "mid",
                "motion_bias": "parked",
                "heading_relation_to_ego": "unknown",
                "density_role": "sparse_filler",
                "confidence": "medium",
                "spawn_priority": "low",
            }
        )
        payload, error = normalize_scene_understanding(sample)
        self.assertIsNone(error)
        by_id = {entity["id"]: entity for entity in payload["background_traffic"]}
        self.assertEqual(by_id["opposing_car"]["lane_side_relation"], "left_lane")
        self.assertEqual(by_id["parked_two_wheelers"]["category"], "parked_vehicle")

    def test_pairwise_validation_detects_expected_vehicle_order(self):
        relation = build_relation_dsl(
            _sample_scene_understanding(),
            _sample_spawn_context(),
            road_artifact={"summary": "simple road"},
        )
        projected = {
            "selected_anchor_lane": _sample_spawn_context()["topology_sample"][1],
            "entities": [
                {
                    "id": "motorcycle_main",
                    "spawn_kind": "vehicle",
                    "category": "motorcycle",
                    "location": {"x": 10.0, "y": -1.75, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
                {
                    "id": "parked_row_0",
                    "spawn_kind": "vehicle",
                    "category": "parked_vehicle",
                    "location": {"x": 28.0, "y": 4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
                {
                    "id": "parked_row_1",
                    "spawn_kind": "vehicle",
                    "category": "parked_vehicle",
                    "location": {"x": 34.0, "y": 4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
                {
                    "id": "parked_row_2",
                    "spawn_kind": "vehicle",
                    "category": "parked_vehicle",
                    "location": {"x": 40.0, "y": 4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
            ],
        }
        validation = validate_pairwise_relations(relation, projected)
        self.assertGreater(validation["summary"]["total"], 0)
        self.assertEqual(validation["summary"]["failed"], 0)

    def test_pairwise_ordering_moves_parked_vehicle_ahead_of_motorcycle(self):
        relation = build_relation_dsl(
            _sample_scene_understanding(),
            _sample_spawn_context(),
            road_artifact={"summary": "simple road"},
        )
        raw_initial = {
            "selected_anchor_lane": _sample_spawn_context()["topology_sample"][0],
            "entities": [
                {
                    "id": "motorcycle_main",
                    "priority": "subject",
                    "category": "motorcycle",
                    "spawn_kind": "vehicle",
                    "location": {"x": 10.0, "y": -1.75, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "parked_row_0",
                    "priority": "background",
                    "category": "parked_vehicle",
                    "spawn_kind": "vehicle",
                    "location": {"x": 7.0, "y": 4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
            "metadata": {"coordinate_stage": "initial"},
        }
        ordered = apply_pairwise_ordering(raw_initial, relation)
        entities = {entity["id"]: entity for entity in ordered["entities"]}
        self.assertGreater(
            entities["parked_row_0"]["location"]["x"],
            entities["motorcycle_main"]["location"]["x"],
        )
        self.assertEqual(ordered["metadata"]["coordinate_stage"], "ordered")

    def test_coordinate_program_generates_raw_coordinates(self):
        scene = _sample_scene_understanding()
        relation = build_relation_dsl(scene, _sample_spawn_context(), road_artifact={})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            relation_path = tmp_path / "scene_relation_dsl.json"
            relation_path.write_text(json.dumps(relation), encoding="utf-8")
            program_path = tmp_path / "scene_coordinate_program.py"
            raw_path = tmp_path / "scene_coordinates_raw.json"
            write_coordinate_program(
                str(relation_path),
                str(program_path),
                str(raw_path),
            )
            raw = evaluate_coordinate_program(str(program_path), cwd=str(tmp_path))
            ids = {entity["id"] for entity in raw["entities"]}
            self.assertIn("motorcycle_main", ids)
            self.assertIn("parked_row_0", ids)
            self.assertTrue(raw_path.exists())
            motorcycle = next(entity for entity in raw["entities"] if entity["id"] == "motorcycle_main")
            self.assertEqual(motorcycle["appearance"]["color"], "blue")

    def test_projection_preserves_subject_priority(self):
        raw_coordinates = {
            "selected_anchor_lane": _sample_spawn_context()["topology_sample"][0],
            "entities": [
                {
                    "id": "subject_car",
                    "priority": "subject",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "heading_relation": "same_direction",
                    "lane_side_relation": "same_lane",
                    "location": {"x": 10.0, "y": -1.75, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "bg_car",
                    "priority": "background",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "heading_relation": "same_direction",
                    "lane_side_relation": "same_lane",
                    "location": {"x": 10.1, "y": -1.75, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
            "metadata": {},
        }
        projected = project_entities_to_xodr(
            raw_coordinates,
            _sample_scene_understanding(),
            _sample_spawn_context(),
        )
        entities = {entity["id"]: entity for entity in projected["entities"]}
        self.assertAlmostEqual(entities["subject_car"]["location"]["x"], 10.0, places=1)
        self.assertGreater(
            abs(entities["bg_car"]["location"]["x"] - entities["subject_car"]["location"]["x"]),
            2.4,
        )

    def test_projection_preserves_sparse_topology_lane_semantics(self):
        sparse_spawn_context = {
            "topology_sample": [_sample_spawn_context()["topology_sample"][0]],
            "spawn_points": _sample_spawn_context()["spawn_points"],
        }
        scene_understanding = _sample_scene_understanding()
        raw_coordinates = {
            "selected_anchor_lane": sparse_spawn_context["topology_sample"][0],
            "entities": [
                {
                    "id": "left_lane_motorcycle",
                    "priority": "subject",
                    "category": "motorcycle",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "motorcycle",
                    "heading_relation": "same_direction",
                    "lane_side_relation": "left_lane",
                    "location": {"x": 10.0, "y": -5.25, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "opposing_car",
                    "priority": "background",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "heading_relation": "opposite_direction",
                    "lane_side_relation": "opposing_center_left",
                    "location": {"x": 24.0, "y": -5.25, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
                {
                    "id": "sidewalk_two_wheeler",
                    "priority": "background",
                    "category": "parked_vehicle",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "heading_relation": "unknown",
                    "lane_side_relation": "sidewalk_right",
                    "location": {"x": 28.0, "y": 8.5, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
            "metadata": {},
        }
        projected = project_entities_to_xodr(
            raw_coordinates,
            scene_understanding,
            sparse_spawn_context,
        )
        entities = {entity["id"]: entity for entity in projected["entities"]}
        lane_center_y = sparse_spawn_context["topology_sample"][0]["start"]["y"]
        self.assertLess(entities["left_lane_motorcycle"]["location"]["y"], lane_center_y - 1.0)
        self.assertLess(entities["opposing_car"]["location"]["y"], lane_center_y - 1.0)
        self.assertGreater(entities["sidewalk_two_wheeler"]["location"]["y"], lane_center_y + 3.0)

    def test_projection_keeps_parked_vehicle_near_driveable_edge(self):
        sparse_spawn_context = {
            "topology_sample": [_sample_spawn_context()["topology_sample"][0]],
            "spawn_points": _sample_spawn_context()["spawn_points"],
        }
        raw_coordinates = {
            "selected_anchor_lane": sparse_spawn_context["topology_sample"][0],
            "entities": [
                {
                    "id": "parked_row_0",
                    "priority": "background",
                    "category": "parked_vehicle",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "heading_relation": "same_direction",
                    "lane_side_relation": "right_edge",
                    "location": {"x": 24.0, "y": 7.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                }
            ],
            "metadata": {},
        }
        projected = project_entities_to_xodr(
            raw_coordinates,
            _sample_scene_understanding(),
            sparse_spawn_context,
        )
        parked = projected["entities"][0]
        lane_center_y = sparse_spawn_context["topology_sample"][0]["start"]["y"]
        self.assertGreater(parked["location"]["y"], lane_center_y + 1.5)
        self.assertLess(parked["location"]["y"], lane_center_y + 2.5)

    def test_refine_and_validation_produce_refined_layout_outputs(self):
        relation = build_relation_dsl(
            _sample_scene_understanding(),
            _sample_spawn_context(),
            road_artifact={"summary": "simple road"},
        )
        projected = {
            "selected_anchor_lane": _sample_spawn_context()["topology_sample"][0],
            "entities": [
                {
                    "id": "motorcycle_main",
                    "priority": "subject",
                    "category": "motorcycle",
                    "spawn_kind": "vehicle",
                    "lane_side_relation": "left_lane",
                    "heading_relation": "same_direction",
                    "location": {"x": 10.0, "y": 1.75, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "parked_row_0",
                    "priority": "background",
                    "category": "parked_vehicle",
                    "spawn_kind": "vehicle",
                    "lane_side_relation": "right_edge",
                    "heading_relation": "same_direction",
                    "location": {"x": 9.0, "y": -4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "sidewalk_people_0",
                    "priority": "background",
                    "category": "pedestrian",
                    "spawn_kind": "pedestrian",
                    "lane_side_relation": "sidewalk_left",
                    "heading_relation": "crossing",
                    "location": {"x": 12.0, "y": 8.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                },
            ],
            "metadata": {"coordinate_stage": "projected"},
        }
        refined = refine_projected_coordinates_with_pairwise_relations(projected, relation)
        self.assertEqual(refined["metadata"]["coordinate_stage"], "refined")
        validation = validate_relation_layout(relation, refined)
        self.assertIn("ego_consistency", validation["summary"])
        self.assertIn("lane_legality", validation["summary"])
        sidewalk_result = next(
            item for item in validation["lane_results"] if item["entity_id"] == "sidewalk_people_0"
        )
        self.assertEqual(sidewalk_result["status"], "pass")


class TestStructuredAutoGenerator(unittest.TestCase):
    def test_auto_generator_defaults_to_stage2(self):
        generator = AutoGenerator("/tmp/autoscenario_test", {"input_type": "image"})
        self.assertEqual(generator.opendoive_stage, "stage2")
        self.assertEqual(generator.generation_params["wall_height"], 0.0)
        self.assertEqual(generator.generation_params["additional_width"], 1.5)
        self.assertTrue(generator._relation_validation_path("scene_1").endswith("scene_1_relation_validation.json"))
        self.assertTrue(generator._raw_initial_coordinates_path("scene_1").endswith("scene_1_coordinates_raw_initial.json"))
        self.assertTrue(generator._raw_ordered_coordinates_path("scene_1").endswith("scene_1_coordinates_raw_ordered.json"))
        self.assertTrue(generator._projected_refined_coordinates_path("scene_1").endswith("scene_1_coordinates_projected_refined.json"))

    def test_auto_generator_widens_generation_params_for_roadside_space(self):
        generator = AutoGenerator("/tmp/autoscenario_test", {"input_type": "image"})
        scene_understanding = _sample_scene_understanding()
        generator.adapt_generation_params_for_scene(scene_understanding)
        self.assertGreaterEqual(generator.generation_params["additional_width"], 6.0)

    def test_deterministic_scene_script_reads_projected_coordinates(self):
        generator = XODRScenarioGenerator()
        projected = {
            "entities": [
                {
                    "id": "motorcycle_main",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "motorcycle",
                    "category": "motorcycle",
                    "appearance": {"color": "blue", "color_rgb": "54,116,168"},
                    "location": {"x": 1.0, "y": 2.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                    "priority": "subject",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            xodr_path = tmp_path / "scene.xodr"
            xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")
            projected_path = tmp_path / "scene_coordinates_projected.json"
            output_path = generator.build_scene_script_from_projected_coordinates(
                scene_id="scene_0001",
                output_folder=str(tmp_path),
                projected_coordinates=projected,
                projected_coordinates_path=str(projected_path),
                xodr_path=str(xodr_path),
                generation_params={"wall_height": 0.0, "additional_width": 1.5},
            )
            content = Path(output_path).read_text(encoding="utf-8")
            self.assertIn("spawn_entities.json", content)
            self.assertIn("_AUTOSCENARIO_SPAWN_PAYLOAD", content)
            self.assertIn("scene.xodr", content)

    def test_spawn_payload_prefers_vlm_vehicle_color(self):
        projected = {
            "entities": [
                {
                    "id": "parked_row_0",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "category": "parked_vehicle",
                    "appearance": {"color": "white", "color_rgb": "255,255,255"},
                    "location": {"x": 10.0, "y": -4.9, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                }
            ]
        }
        payload = build_projected_spawn_payload(projected)
        self.assertEqual(payload["entities"][0]["color"], "255,255,255")
        self.assertEqual(payload["entities"][0]["appearance"]["color"], "white")


if __name__ == "__main__":
    unittest.main()
