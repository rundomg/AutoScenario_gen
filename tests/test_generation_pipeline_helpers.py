import sys
import types
import unittest
import tempfile
import json
import math
import py_compile
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
matplotlib_module = types.ModuleType("matplotlib")
matplotlib_pyplot = types.ModuleType("matplotlib.pyplot")
matplotlib_module.pyplot = matplotlib_pyplot
sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_pyplot)

from agents.obstacle_generator import ObstacleGenerator
from agents import task_agent as task_agent_module
from agents.task_agent import TaskAgent
from agents.scene_understanding_interpreter import SceneUnderstandingInterpreter
from agents.vehicle_orientation_interpreter import VehicleOrientationInterpreter
from agents.vehicle_position_interpreter import VehiclePositionInterpreter
from agents.scene_verification_agent import SceneVerificationAgent
from experiments.auto_generate_all_vlm import AutoGenerator
from tools import cache_map_topology
from tools.scene_map_matcher import SceneMapMatcher
from tools.map_matcher_v2 import score_candidate_v2
from tools import vehicle_detector
from tools.structured_pipeline import (
    build_relation_dsl,
    build_projected_spawn_payload,
    generate_initial_coordinates_from_relation_dsl,
    validate_relation_layout,
    project_entities_to_carla_context,
    refine_projected_coordinates_with_pairwise_relations,
    normalize_scene_understanding,
    TURN_INTENT_YAW_DEG,
    _canonical_turn_intent,
    _apply_turn_intent,
    _yaw_for_heading_relation,
)
from tools.junction_placement import reproject_actors_for_junction, reproject_actors_for_structure


SPLIT_TEXT = """## Road Net Description:
Straight city street with a crosswalk ahead.

## Road Users Description:
A scooter is ahead-left and pedestrians are near the sidewalk.

## Static Objects Description:
Cones are on the roadside and a traffic light is visible.

## Vehicles' Locations and Behaviors:
The scooter is ahead-left and appears slow or stopped.

## Scenario Description:
The ego vehicle approaches a crosswalk with a nearby scooter and roadside cones.
"""


class _FakeHTTPError(Exception):
    def __init__(self, response=None):
        super().__init__("fake http error")
        self.response = response


class _FakeTimeout(Exception):
    pass


class _FakeConnectionError(Exception):
    pass


class _FakeResponse:
    def __init__(self, payload=None, text="", status_code=200, headers=None, json_error=None):
        self.payload = payload
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeHTTPError(self)
        return None

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class _FakeRequests:
    exceptions = types.SimpleNamespace(
        Timeout=_FakeTimeout,
        ConnectionError=_FakeConnectionError,
        HTTPError=_FakeHTTPError,
    )

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


class _EchoTaskAgent(TaskAgent):
    def refine_request(self, user_request=None, add_info=None):
        return user_request


class TestGenerationPipelineHelpers(unittest.TestCase):
    def test_vehicle_detector_dedupes_overlapping_boxes_and_keeps_row_hint(self):
        detections = [
            {
                "id": "raw_1",
                "label": "car",
                "conf": 0.9,
                "bbox_xyxy": [10, 10, 110, 110],
                "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                "center_norm": [0.2, 0.2],
            },
            {
                "id": "raw_2",
                "label": "car",
                "conf": 0.8,
                "bbox_xyxy": [12, 12, 108, 108],
                "bbox_norm": [0.105, 0.105, 0.295, 0.295],
                "center_norm": [0.2, 0.2],
            },
            {
                "id": "raw_3",
                "label": "truck",
                "conf": 0.7,
                "bbox_xyxy": [180, 12, 260, 108],
                "bbox_norm": [0.55, 0.11, 0.75, 0.29],
                "center_norm": [0.65, 0.2],
            },
        ]

        selected = vehicle_detector.select_representative_detections(detections)
        row_hints = vehicle_detector.infer_row_group_hints(selected)

        self.assertEqual(len(selected), 2)
        self.assertEqual([det["id"] for det in selected], ["det_1", "det_2"])
        self.assertTrue(row_hints)
        self.assertEqual(row_hints[0]["det_ids"], ["det_1", "det_2"])

    def test_orientation_response_parser_normalizes_hints(self):
        response = json.dumps(
            {
                "vehicle_orientation_hints": [
                    {
                        "det_id": "det_1",
                        "visible_end": "rear",
                        "front_points_image_direction": "toward_camera",
                        "vehicle_region_hint": "ahead_arm",
                        "junction_center_relative_to_vehicle": "behind",
                        "heading_relation_to_ego": "opposite_direction",
                        "junction_travel_direction": "away_from_junction",
                        "motion_state_hint": "stopped",
                        "confidence": "high",
                        "evidence": "rear lights visible",
                    }
                ]
            }
        )

        hints = VehicleOrientationInterpreter.extract_orientation_hints(response)

        self.assertEqual(hints[0]["det_id"], "det_1")
        self.assertEqual(hints[0]["junction_travel_direction"], "away_from_junction")
        self.assertEqual(hints[0]["heading_relation_to_ego"], "opposite_direction")
        self.assertEqual(hints[0]["front_points_image_direction"], "toward_camera")
        self.assertEqual(hints[0]["vehicle_region_hint"], "ahead_arm")

    def test_orientation_response_parser_normalizes_ignored_detections(self):
        payload = VehicleOrientationInterpreter.extract_orientation_payload(
            json.dumps(
                {
                    "ignored_detections": [
                        {"id": "det_2", "evidence": "duplicate box"},
                        "det_3",
                        {"det_id": "det_2", "reason": "duplicate repeat"},
                    ],
                    "vehicle_orientation_hints": [],
                }
            )
        )

        self.assertEqual(
            payload["ignored_detections"],
            [
                {"det_id": "det_2", "reason": "duplicate box"},
                {"det_id": "det_3", "reason": ""},
            ],
        )

    def test_orientation_response_parser_empty_payload_includes_ignored_detections(self):
        payload = VehicleOrientationInterpreter.extract_orientation_payload("")

        self.assertEqual(payload["ignored_detections"], [])
        self.assertEqual(payload["vehicle_orientation_hints"], [])

    def test_orientation_sanitizer_removes_junction_fields_for_open_road(self):
        payload = VehicleOrientationInterpreter.extract_orientation_payload(
            json.dumps(
                {
                    "ignored_detections": [
                        {"det_id": "det_9", "reason": "outside drivable scene"}
                    ],
                    "vehicle_orientation_hints": [
                        {
                            "det_id": "det_1",
                            "visible_end": "rear",
                            "front_points_image_direction": "away_from_camera",
                            "vehicle_region_hint": "right_arm",
                            "junction_center_relative_to_vehicle": "left",
                            "junction_travel_direction": "away_from_junction",
                            "heading_relation_to_ego": "same_direction",
                            "confidence": "high",
                        }
                    ]
                }
            )
        )
        sanitized = VehicleOrientationInterpreter.sanitize_for_road_scene(
            payload,
            {
                "road_network": {
                    "map_matching": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                    }
                }
            },
        )

        hint = sanitized["vehicle_orientation_hints"][0]
        self.assertNotIn("vehicle_region_hint", hint)
        self.assertNotIn("junction_center_relative_to_vehicle", hint)
        self.assertNotIn("junction_travel_direction", hint)
        self.assertEqual(hint["heading_relation_to_ego"], "same_direction")
        self.assertEqual(
            sanitized["ignored_detections"],
            [{"det_id": "det_9", "reason": "outside drivable scene"}],
        )

    def test_orientation_prompt_explains_full_image_and_crop_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            crop_path = str(Path(tmp) / "det_1_crop.jpg")
            Path(image_path).write_bytes(b"fake")
            Path(crop_path).write_bytes(b"fake")
            with mock.patch(
                "agents.vehicle_orientation_interpreter.VehicleOrientationInterpreter._image_to_base64",
                return_value="abc",
            ):
                content = VehicleOrientationInterpreter().refine_request(
                    "",
                    {
                        "image_path": image_path,
                        "annotated_path": image_path,
                        "detections": [
                            {
                                "id": "det_1",
                                "label": "truck",
                                "conf": 0.9,
                                "bbox_norm": [0.6, 0.3, 0.9, 0.6],
                                "crop_path": crop_path,
                            }
                        ],
                    },
                )

        prompt = content[0]["text"]
        self.assertIn("The first image is the annotated full image", prompt)
        self.assertIn("following images are vehicle crops", prompt)
        self.assertIn("Road-scene branch: OPEN_ROAD", prompt)
        self.assertIn('"road_region_hint"', prompt)
        self.assertNotIn('"junction_travel_direction"', prompt)
        self.assertNotIn('"vehicle_region_hint"', prompt)

    def test_orientation_prompt_uses_junction_branch_when_road_scene_is_junction(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            Path(image_path).write_bytes(b"fake")
            with mock.patch(
                "agents.vehicle_orientation_interpreter.VehicleOrientationInterpreter._image_to_base64",
                return_value="abc",
            ):
                content = VehicleOrientationInterpreter().refine_request(
                    "",
                    {
                        "image_path": image_path,
                        "annotated_path": image_path,
                        "road_scene": {
                            "road_network": {
                                "map_matching": {
                                    "topology_type": "t_junction",
                                    "junction_visible": True,
                                }
                            }
                        },
                        "detections": [
                            {
                                "id": "det_1",
                                "label": "truck",
                                "conf": 0.9,
                                "bbox_norm": [0.6, 0.3, 0.9, 0.6],
                            }
                        ],
                    },
                )

        prompt = content[0]["text"]
        self.assertIn("Road-scene branch: JUNCTION", prompt)
        self.assertIn('"vehicle_region_hint"', prompt)
        self.assertIn('"junction_travel_direction"', prompt)
        self.assertIn("bbox center-x alone", prompt)
        self.assertIn("vehicle_region_hint=ego_approach", prompt)
        self.assertIn("front_points_image_direction=right", prompt)

    def test_scene_understanding_detector_prompt_uses_orientation_not_depth(self):
        detections = [
            {
                "id": "det_1",
                "label": "truck",
                "conf": 0.91,
                "bbox_xyxy": [10, 10, 110, 110],
                "bbox_norm": [0.1, 0.1, 0.3, 0.3],
                "center_norm": [0.2, 0.2],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            Path(image_path).write_bytes(b"fake")
            output_path = str(Path(tmp) / "s0000_su.json")
            with mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.detect_vehicles",
                return_value=detections,
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.select_representative_detections",
                return_value=detections,
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.infer_row_group_hints",
                return_value=[
                    {
                        "id": "row_hint_1",
                        "det_ids": ["det_1"],
                        "alignment": "horizontal_image_row",
                        "order_rule_hint": "image_left_to_right",
                    }
                ],
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.annotate_image",
                return_value=str(Path(tmp) / "s0000_detections.jpg"),
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.write_detection_crops",
                return_value=detections,
            ), mock.patch(
                "agents.scene_understanding_interpreter.SceneUnderstandingInterpreter._image_to_base64",
                return_value="abc",
            ), mock.patch(
                "agents.scene_understanding_interpreter.SceneUnderstandingInterpreter._build_vehicle_orientation_hints",
                return_value=[
                    {
                        "det_id": "det_1",
                        "visible_end": "rear",
                        "heading_relation_to_ego": "opposite_direction",
                        "junction_travel_direction": "away_from_junction",
                        "motion_state_hint": "stopped",
                        "confidence": "high",
                        "evidence": "rear visible",
                    }
                ],
            ):
                content = SceneUnderstandingInterpreter().refine_request(
                    "",
                    {"image_path": image_path, "output_fn": output_path},
                )

        prompt = content[0]["text"]
        self.assertIn("vehicle_orientation_hints", prompt)
        self.assertIn("actor_group_type=individual_vehicle", prompt)
        self.assertNotIn("actor_group_type=traffic_queue", prompt)
        self.assertNotIn("depth≈", prompt)
        self.assertNotIn("depth_rank", prompt)

    def test_split_scene_agents_merge_road_and_orientation_into_final_su(self):
        detections = [
            {
                "id": "det_1",
                "label": "truck",
                "conf": 0.91,
                "bbox_xyxy": [500, 220, 700, 420],
                "bbox_norm": [0.65, 0.3, 0.9, 0.58],
                "center_norm": [0.78, 0.44],
            },
            {
                "id": "det_2",
                "label": "car",
                "conf": 0.82,
                "bbox_xyxy": [510, 430, 690, 590],
                "bbox_norm": [0.66, 0.59, 0.88, 0.82],
                "center_norm": [0.77, 0.71],
            },
        ]
        road_payload = {
            "road_scene_brief": {
                "scene_kind_observation": "signalized junction with right arm",
                "junction_evidence": ["traffic light", "right branch"],
            },
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "t_junction",
                    "junction_visible": True,
                    "junction_type": "t_junction",
                    "junction_branches": {
                        "ahead": True,
                        "left": False,
                        "right": True,
                        "known": True,
                    },
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "has_traffic_light": True,
                    "curve_direction": "straight",
                }
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }
        orientation_response = json.dumps(
            {
                "vehicle_orientation_brief": {
                    "overall_observation": "right arm vehicles face away from junction"
                },
                "vehicle_orientation_hints": [
                    {
                        "det_id": "det_1",
                        "visible_end": "rear",
                        "heading_relation_to_ego": "crossing",
                        "junction_travel_direction": "away_from_junction",
                        "motion_state_hint": "stopped",
                        "confidence": "high",
                        "evidence": "rear visible on right arm",
                    },
                    {
                        "det_id": "det_2",
                        "visible_end": "rear",
                        "heading_relation_to_ego": "crossing",
                        "junction_travel_direction": "away_from_junction",
                        "motion_state_hint": "stopped",
                        "confidence": "high",
                        "evidence": "queued behind det_1",
                    },
                ],
            }
        )
        position_response = json.dumps(
            {
                "vehicle_position_brief": {
                    "scene_kind": "junction",
                    "overall_observation": "two vehicles on right arm",
                    "uncertainties": [],
                },
                "traffic_subjects": [
                    {
                        "id": "det_1",
                        "category": "truck",
                        "subtype": "truck",
                        "motion_state": "stopped",
                        "heading_relation_to_ego": "crossing",
                        "lane_side_relation": "right_lane",
                        "lane_index_relation": 1,
                        "longitudinal_relation": "ahead",
                        "longitudinal_proximity": "mid",
                        "layout_anchor_id": "right_arm",
                        "anchor_relation": {
                            "travel_direction": "away_from_junction",
                            "position_along_anchor": "mid_arm",
                        },
                        "actor_group_type": "individual_vehicle",
                        "actor_group_id": "det_1",
                        "placement_mode_hint": "normal_lane_actor",
                        "visual_confidence": "high",
                        "evidence": "position agent right arm",
                    },
                    {
                        "id": "det_2",
                        "category": "car",
                        "subtype": "car",
                        "motion_state": "stopped",
                        "heading_relation_to_ego": "crossing",
                        "lane_side_relation": "right_lane",
                        "lane_index_relation": 1,
                        "longitudinal_relation": "ahead",
                        "longitudinal_proximity": "near",
                        "layout_anchor_id": "right_arm",
                        "anchor_relation": {
                            "travel_direction": "away_from_junction",
                            "position_along_anchor": "near_mouth",
                        },
                        "actor_group_type": "individual_vehicle",
                        "actor_group_id": "det_2",
                        "placement_mode_hint": "normal_lane_actor",
                        "visual_confidence": "high",
                        "evidence": "position agent right arm",
                    },
                ],
                "key_pairwise_relations": [
                    {
                        "entity_id": "det_2",
                        "other_entity_id": "det_1",
                        "longitudinal_relation": "behind_other",
                        "longitudinal_gap_band": "near",
                        "lane_relation": "same_lane",
                        "constraint_strength": "soft",
                        "confidence": "high",
                        "evidence": "position agent graph",
                    }
                ],
                "vehicle_position_graph": {
                    "nodes": [
                        {
                            "det_id": "det_1",
                            "lane_band": "right_arm",
                            "motion_state": "stopped",
                            "longitudinal_band": "mid",
                            "confidence": "high",
                            "evidence": "position agent right arm",
                        }
                    ],
                    "edges": [],
                },
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            output_path = str(Path(tmp) / "s0000_su.json")
            Path(image_path).write_bytes(b"fake")

            def fake_send_request(_self, _user_request, add_info=None):
                Path(add_info["output_fn"]).write_text(json.dumps(road_payload))
                return json.dumps(road_payload)

            with mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.detect_vehicles",
                return_value=detections,
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.select_representative_detections",
                return_value=detections,
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.infer_row_group_hints",
                return_value=[
                    {
                        "id": "row_hint_1",
                        "det_ids": ["det_1", "det_2"],
                        "alignment": "same_image_column",
                        "order_rule_hint": "image_bottom_to_top",
                    }
                ],
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.annotate_image",
                return_value=image_path,
            ), mock.patch(
                "agents.scene_understanding_interpreter.vehicle_detector.write_detection_crops",
                return_value=detections,
            ), mock.patch.object(
                SceneUnderstandingInterpreter,
                "send_request",
                fake_send_request,
            ), mock.patch(
                "agents.vehicle_orientation_interpreter.VehicleOrientationInterpreter.send_request",
                return_value=orientation_response,
            ), mock.patch(
                "agents.scene_understanding_interpreter.VehiclePositionInterpreter.send_request",
                return_value=position_response,
            ):
                result = SceneUnderstandingInterpreter().call_agent(
                    "",
                    {"image_path": image_path, "output_fn": output_path},
                )
                position_file_written = (Path(tmp) / "s0000_position.json").exists()

        self.assertEqual(len(result["traffic_subjects"]), 2)
        first = result["traffic_subjects"][0]
        self.assertEqual(first["heading_relation_to_ego"], "crossing")
        self.assertEqual(
            first["anchor_relation"]["travel_direction"],
            "away_from_junction",
        )
        self.assertEqual(first["actor_group_type"], "individual_vehicle")
        self.assertEqual(first["layout_anchor_id"], "right_arm")
        self.assertIn("vehicle_orientation_hints", result["metadata"])
        self.assertIn("vehicle_position_graph", result["metadata"])
        self.assertIn("road_scene_brief", result["metadata"])
        self.assertTrue(position_file_written)

    def test_split_merge_keeps_ignored_candidate_when_position_keeps_subject(self):
        result = SceneUnderstandingInterpreter._merge_split_scene_payloads(
            road_payload={
                "traffic_subjects": [],
                "key_pairwise_relations": [],
                "road_network": {
                    "map_matching": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                    }
                },
                "actor_layout": {},
                "general_environment": {},
                "metadata": {},
            },
            detections=[
                {"id": "det_1", "label": "car", "center_norm": [0.4, 0.5]},
                {"id": "det_2", "label": "car", "center_norm": [0.6, 0.5]},
            ],
            row_hints=[],
            orientation_payload={
                "ignored_detections": [
                    {"det_id": "det_2", "reason": "roadside candidate"}
                ],
                "vehicle_orientation_hints": [],
            },
            position_payload={
                "traffic_subjects": [
                    {"id": "det_1", "category": "car"},
                    {"id": "det_2", "category": "car"},
                ],
                "key_pairwise_relations": [
                    {
                        "entity_id": "det_2",
                        "other_entity_id": "det_1",
                        "longitudinal_relation": "behind_other",
                    }
                ],
            },
        )

        self.assertEqual([item["id"] for item in result["traffic_subjects"]], ["det_1", "det_2"])
        self.assertEqual(
            result["metadata"]["ignored_detection_candidates"],
            [{"det_id": "det_2", "reason": "roadside candidate"}],
        )
        self.assertEqual(len(result["key_pairwise_relations"]), 1)

    def test_split_merge_allows_position_to_omit_ignored_candidate_and_drops_relations(self):
        result = SceneUnderstandingInterpreter._merge_split_scene_payloads(
            road_payload={
                "traffic_subjects": [],
                "key_pairwise_relations": [],
                "road_network": {
                    "map_matching": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                    }
                },
                "actor_layout": {},
                "general_environment": {},
                "metadata": {},
            },
            detections=[
                {"id": "det_1", "label": "car", "center_norm": [0.4, 0.5]},
                {"id": "det_2", "label": "car", "center_norm": [0.6, 0.5]},
            ],
            row_hints=[
                {
                    "id": "row_hint_1",
                    "det_ids": ["det_1", "det_2"],
                    "order_rule_hint": "image_bottom_to_top",
                }
            ],
            orientation_payload={
                "ignored_detections": [
                    {"det_id": "det_2", "reason": "outside drivable scene"}
                ],
                "vehicle_orientation_hints": [],
            },
            position_payload={
                "traffic_subjects": [{"id": "det_1", "category": "car"}],
                "key_pairwise_relations": [
                    {
                        "entity_id": "det_1",
                        "other_entity_id": "det_2",
                        "longitudinal_relation": "ahead_of_other",
                    }
                ],
            },
        )

        self.assertEqual([item["id"] for item in result["traffic_subjects"]], ["det_1"])
        self.assertEqual(result["key_pairwise_relations"], [])
        self.assertEqual(
            result["metadata"]["ignored_detection_candidates"],
            [{"det_id": "det_2", "reason": "outside drivable scene"}],
        )

    def test_fallback_merge_does_not_hard_filter_ignored_candidates(self):
        result = SceneUnderstandingInterpreter._merge_road_and_vehicle_orientation(
            road_payload={
                "traffic_subjects": [],
                "key_pairwise_relations": [],
                "road_network": {
                    "map_matching": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                    }
                },
                "actor_layout": {},
                "general_environment": {},
                "metadata": {},
            },
            detections=[
                {
                    "id": "det_1",
                    "label": "car",
                    "bbox_norm": [0.2, 0.4, 0.4, 0.7],
                    "center_norm": [0.3, 0.55],
                },
                {
                    "id": "det_2",
                    "label": "car",
                    "bbox_norm": [0.55, 0.4, 0.75, 0.7],
                    "center_norm": [0.65, 0.55],
                },
            ],
            row_hints=[],
            orientation_payload={
                "ignored_detections": [
                    {"det_id": "det_2", "reason": "soft candidate only"}
                ],
                "vehicle_orientation_hints": [],
            },
        )

        self.assertEqual([item["id"] for item in result["traffic_subjects"]], ["det_1", "det_2"])
        self.assertEqual(
            result["metadata"]["ignored_detection_candidates"],
            [{"det_id": "det_2", "reason": "soft candidate only"}],
        )

    def test_road_scene_only_prompt_does_not_request_vehicle_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            Path(image_path).write_bytes(b"fake")
            with mock.patch(
                "agents.scene_understanding_interpreter.SceneUnderstandingInterpreter._image_to_base64",
                return_value="abc",
            ):
                content = SceneUnderstandingInterpreter().refine_request(
                    "",
                    {
                        "image_path": image_path,
                        "output_fn": str(Path(tmp) / "s0000_road_scene.json"),
                        "road_scene_only": True,
                    },
                )

        prompt = content[0]["text"]
        self.assertIn("road_scene_brief", prompt)
        self.assertIn("Do not output vehicle orientation", prompt)
        self.assertIn("Classify junction only from visible road-geometry branches", prompt)
        self.assertIn("not upgrade a straight road into a junction", prompt)
        self.assertIn('"traffic_subjects": []', prompt)

    def test_vehicle_position_payload_extracts_compact_relations(self):
        payload = VehiclePositionInterpreter.extract_position_payload(
            json.dumps(
                {
                    "vehicle_position_brief": {
                        "scene_kind": "open_road",
                        "overall_observation": "right curb parking row",
                        "lane_assignment_basis": "curb alignment and road topology",
                        "pairwise_relation_basis": "parking row image order",
                        "uncertainties": ["small overlapping boxes"],
                    },
                    "traffic_subjects": [
                        {
                            "id": "det_1",
                            "category": "car",
                            "lane_side_relation": "right_parking_lane",
                            "layout_anchor_id": "",
                            "anchor_relation": {},
                            "evidence": "this evidence is intentionally too long for compact output",
                        }
                    ],
                    "key_pairwise_relations": [
                        {
                            "entity_id": "det_1",
                            "other_entity_id": "det_2",
                            "lane_relation": "same_parking_lane",
                            "confidence": "high",
                            "constraint_strength": "soft",
                        }
                    ],
                    "vehicle_position_graph": {
                        "nodes": [{"det_id": "det_1", "lane_band": "right_parking_lane"}],
                        "edges": [
                            {
                                "source_id": "det_1",
                                "target_id": "det_2",
                                "lane_relation": "same_parking_lane",
                            }
                        ],
                    },
                }
            )
        )

        self.assertEqual(payload["vehicle_position_brief"]["scene_kind"], "open_road")
        self.assertEqual(
            payload["vehicle_position_brief"]["overall_observation"],
            "right curb parking row",
        )
        self.assertEqual(payload["traffic_subjects"][0]["id"], "det_1")
        self.assertNotIn("vehicle_position_graph", payload)
        self.assertNotIn("layout_anchor_id", payload["traffic_subjects"][0])
        self.assertNotIn("anchor_relation", payload["traffic_subjects"][0])
        self.assertNotIn("confidence", payload["key_pairwise_relations"][0])
        self.assertNotIn("constraint_strength", payload["key_pairwise_relations"][0])
        self.assertEqual(
            payload["key_pairwise_relations"][0]["lane_relation"],
            "same_parking_lane",
        )

    def test_vehicle_position_prompt_requests_compact_payload(self):
        prompt = VehiclePositionInterpreter().pre_prompt

        self.assertIn('"traffic_subjects"', prompt)
        self.assertIn('"key_pairwise_relations"', prompt)
        self.assertIn('"vehicle_position_brief"', prompt)
        self.assertIn("short visible reasoning summary", prompt)
        self.assertIn("bbox center-x and horizontal overlap", prompt)
        self.assertIn("vehicle left/right ordering", prompt)
        self.assertNotIn('"vehicle_position_graph"', prompt)
        self.assertNotIn('"confidence": "high | medium | low"', prompt)
        self.assertIn("Do not build a complete all-pairs relation set", prompt)
        self.assertIn("ignored_detections", prompt)
        self.assertIn("When uncertain", prompt)
        self.assertIn("Do not omit curbside parked vehicles", prompt)

    def test_vehicle_position_open_road_prompt_uses_boxes_for_left_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            Path(image_path).write_bytes(b"fake")
            with mock.patch(
                "agents.vehicle_position_interpreter.VehiclePositionInterpreter._image_to_base64",
                return_value="abc",
            ):
                content = VehiclePositionInterpreter().refine_request(
                    "",
                    {
                        "image_path": image_path,
                        "annotated_path": image_path,
                        "road_scene": {
                            "road_network": {
                                "map_matching": {
                                    "topology_type": "straight_road",
                                    "junction_visible": False,
                                }
                            }
                        },
                        "detections": [
                            {
                                "id": "det_1",
                                "label": "car",
                                "conf": 0.9,
                                "bbox_norm": [0.2, 0.4, 0.4, 0.7],
                                "center_norm": [0.3, 0.55],
                            },
                            {
                                "id": "det_2",
                                "label": "car",
                                "conf": 0.88,
                                "bbox_norm": [0.55, 0.4, 0.75, 0.7],
                                "center_norm": [0.65, 0.55],
                            },
                        ],
                    },
                )

        prompt = content[0]["text"]
        self.assertIn("Road-scene branch: OPEN_ROAD", prompt)
        self.assertIn("bbox center-x and horizontal overlap", prompt)
        self.assertIn("vehicle left/right relations", prompt)
        self.assertIn("ego-centric", prompt)

    def test_vehicle_position_junction_prompt_keeps_ego_approach_lanes_out_of_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = str(Path(tmp) / "scene.jpg")
            Path(image_path).write_bytes(b"fake")
            with mock.patch(
                "agents.vehicle_position_interpreter.VehiclePositionInterpreter._image_to_base64",
                return_value="abc",
            ):
                content = VehiclePositionInterpreter().refine_request(
                    "三辆车都在ego前面，分别在左车道、同车道、右车道",
                    {
                        "image_path": image_path,
                        "annotated_path": image_path,
                        "road_scene": {
                            "road_network": {
                                "map_matching": {
                                    "topology_type": "cross_intersection",
                                    "junction_visible": True,
                                }
                            }
                        },
                        "detections": [
                            {
                                "id": "det_1",
                                "label": "car",
                                "conf": 0.9,
                                "bbox_norm": [0.05, 0.35, 0.2, 0.75],
                                "center_norm": [0.12, 0.55],
                            }
                        ],
                    },
                )

        prompt = content[0]["text"]
        self.assertIn("Road-scene branch: JUNCTION", prompt)
        self.assertIn("bbox center-x alone", prompt)
        self.assertIn("remain layout_anchor_id=ego_approach", prompt)
        self.assertIn("Additional context", prompt)

    def test_junction_detection_fallback_keeps_side_boxes_on_ego_approach(self):
        self.assertEqual(
            SceneUnderstandingInterpreter._layout_anchor_for_detection(0.12, 0.55, True),
            "ego_approach",
        )
        self.assertEqual(
            SceneUnderstandingInterpreter._layout_anchor_for_detection(0.88, 0.55, True),
            "ego_approach",
        )
        self.assertEqual(
            SceneUnderstandingInterpreter._layout_anchor_for_detection(0.50, 0.30, True),
            "ego_approach",
        )

    def test_junction_detection_fallback_respects_explicit_ahead_arm_hint(self):
        subject = SceneUnderstandingInterpreter._subject_from_detection(
            {
                "id": "det_far",
                "label": "car",
                "conf": 0.9,
                "bbox_norm": [0.45, 0.25, 0.55, 0.40],
                "center_norm": [0.50, 0.32],
            },
            {
                "vehicle_region_hint": "ahead_arm",
                "junction_travel_direction": "toward_junction",
                "heading_relation_to_ego": "opposite_direction",
            },
            {},
            is_junction=True,
        )

        self.assertEqual(subject["layout_anchor_id"], "ahead_arm")
        self.assertEqual(
            subject["anchor_relation"]["travel_direction"],
            "toward_junction",
        )

    def test_user_ego_lane_layout_corrects_position_arm_assignments(self):
        payload = {
            "traffic_subjects": [
                {
                    "id": "det_2",
                    "layout_anchor_id": "left_arm",
                    "lane_side_relation": "left_lane",
                    "heading_relation_to_ego": "crossing",
                    "longitudinal_relation": "alongside",
                },
                {
                    "id": "det_3",
                    "layout_anchor_id": "right_arm",
                    "lane_side_relation": "right_lane",
                    "heading_relation_to_ego": "crossing",
                    "longitudinal_relation": "aligned",
                },
            ]
        }

        corrected = SceneUnderstandingInterpreter._apply_user_ego_lane_layout_constraints(
            payload,
            "三辆车都在ego前面，左车道一辆，同车道一辆，右车道一辆",
        )

        by_id = {item["id"]: item for item in corrected["traffic_subjects"]}
        self.assertEqual(by_id["det_2"]["layout_anchor_id"], "ego_approach")
        self.assertEqual(by_id["det_3"]["layout_anchor_id"], "ego_approach")
        self.assertEqual(by_id["det_2"]["heading_relation_to_ego"], "same_direction")
        self.assertEqual(by_id["det_3"]["longitudinal_relation"], "ahead")

    def test_merge_infers_right_arm_front_right_as_away_from_junction(self):
        subject = SceneUnderstandingInterpreter._subject_from_detection(
            {
                "id": "det_1",
                "label": "truck",
                "conf": 0.9,
                "bbox_norm": [0.65, 0.35, 0.9, 0.6],
                "center_norm": [0.78, 0.48],
            },
            {
                "visible_end": "side",
                "front_points_image_direction": "right",
                "vehicle_region_hint": "right_arm",
                "junction_travel_direction": "unknown",
                "heading_relation_to_ego": "crossing",
                "motion_state_hint": "stopped",
                "confidence": "high",
                "evidence": "front points image-right on right arm",
            },
            {
                "actor_group_type": "traffic_queue",
                "actor_group_id": "traffic_queue_1",
                "group_order_index": 0,
                "placement_mode_hint": "queue_on_lane",
            },
            is_junction=True,
        )

        self.assertEqual(subject["layout_anchor_id"], "right_arm")
        self.assertEqual(
            subject["anchor_relation"]["travel_direction"],
            "away_from_junction",
        )

    def test_vehicle_position_empty_subjects_is_hard_failure(self):
        interpreter = SceneUnderstandingInterpreter()
        empty_response = json.dumps(
            {
                "traffic_subjects": [],
                "key_pairwise_relations": [],
            }
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            VehiclePositionInterpreter,
            "send_request",
            return_value=empty_response,
        ):
            with self.assertRaisesRegex(RuntimeError, "returned no traffic subjects"):
                interpreter._call_vehicle_position_agent(
                    image_path="unused.jpg",
                    annotated_path="unused.jpg",
                    detections=[
                        {
                            "id": "det_1",
                            "label": "car",
                            "bbox_norm": [0.1, 0.2, 0.3, 0.4],
                            "center_norm": [0.2, 0.3],
                        }
                    ],
                    row_hints=[],
                    road_payload={"road_network": {"map_matching": {"topology_type": "straight_road"}}},
                    orientation_payload={"vehicle_orientation_hints": []},
                    output_fn=str(Path(tmp) / "position.json"),
                    request_timeout=1,
                )

    def test_normalize_accepts_compact_road_network_map_matching(self):
        payload = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_two_way",
                    "junction_visible": False,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "has_crosswalk": True,
                    "curve_direction": "straight",
                }
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)

        self.assertIsNone(error)
        self.assertEqual(
            normalized["road_network"]["map_matching"]["topology_type"],
            "straight_two_way",
        )

    def test_scene_prompts_require_structured_parking_fields(self):
        interpreter = SceneUnderstandingInterpreter()

        for prompt in (interpreter.pre_prompt, interpreter.road_scene_prompt):
            self.assertIn("left_parking_presence", prompt)
            self.assertIn("right_parking_presence", prompt)
            self.assertIn("left_parking_lane_count", prompt)
            self.assertIn("right_parking_lane_count", prompt)
            self.assertIn("curbside parked vehicles", prompt)

    def test_normalize_map_matching_backfills_parking_presence_from_lane_groups(self):
        payload = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 1,
                    }
                ],
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)

        self.assertIsNone(error)
        map_matching = normalized["road_network"]["map_matching"]
        self.assertTrue(map_matching["left_parking_presence"])
        self.assertTrue(map_matching["right_parking_presence"])

    def test_normalize_applies_straight_road_parking_symmetry(self):
        payload = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "has_center_median": False,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 1,
                    }
                ],
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)

        self.assertIsNone(error)
        lane_group = normalized["road_network"]["lane_groups"][0]
        self.assertEqual(lane_group["left_parking_lane_count"], 1)
        self.assertEqual(lane_group["right_parking_lane_count"], 1)
        map_matching = normalized["road_network"]["map_matching"]
        self.assertTrue(map_matching["left_parking_presence"])
        self.assertTrue(map_matching["right_parking_presence"])

    def test_normalize_tolerates_unknown_lane_count_strings(self):
        payload = {
            "traffic_subjects": [
                {
                    "id": "front_car",
                    "category": "car",
                    "heading_relation_to_ego": "same_direction",
                    "lane_side_relation": "same_lane",
                    "lane_index_relation": 0,
                    "count": "unknown",
                }
            ],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "opposing_lane_count": "unknown",
                    "left_parking_lane_count": "unknown",
                    "right_parking_lane_count": "unknown",
                },
                "lane_groups": [
                    {
                        "forward_lane_count": "2",
                        "opposing_lane_count": "unknown",
                        "left_parking_lane_count": "unknown",
                        "right_parking_lane_count": "unknown",
                    }
                ],
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)

        self.assertIsNone(error)
        self.assertIsNotNone(normalized)
        lane_group = normalized["road_network"]["lane_groups"][0]
        self.assertEqual(lane_group["forward_lane_count"], 2)
        self.assertEqual(lane_group["opposing_lane_count"], 1)
        self.assertEqual(lane_group["left_parking_lane_count"], 0)
        self.assertEqual(lane_group["right_parking_lane_count"], 0)
        self.assertEqual(normalized["traffic_subjects"][0]["count"], 1)
        map_matching = normalized["road_network"]["map_matching"]
        self.assertFalse(map_matching["left_parking_presence"])
        self.assertFalse(map_matching["right_parking_presence"])

    def test_normalize_map_matching_canonicalizes_branch_list(self):
        payload = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "signalized_intersection_approach",
                    "junction_visible": True,
                    "junction_type": "signalized_intersection",
                    "junction_branches": [
                        "left_arm",
                        "right_arm",
                        "ahead_arm",
                        "oncoming_arm",
                    ],
                    "target_branch_count": 4,
                    "forward_lane_count": 2,
                    "opposing_lane_count": 2,
                    "driving_lane_count": 4,
                    "has_traffic_light": True,
                    "has_center_median": False,
                }
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)
        self.assertIsNone(error)
        map_matching = normalized["road_network"]["map_matching"]
        self.assertEqual(map_matching["topology_type"], "cross_intersection")
        self.assertEqual(map_matching["junction_type"], "cross_intersection")
        self.assertEqual(
            map_matching["junction_branches"],
            {"ahead": True, "left": True, "right": True, "known": True},
        )

        signature = SceneMapMatcher.build_road_topology_signature(normalized)
        self.assertEqual(signature["topology_type"], "cross_intersection")
        self.assertTrue(signature["junction_branches"]["right"])
        self.assertEqual(signature["target_branch_count"], 4)

    def test_straight_road_with_only_signal_and_crosswalk_does_not_upgrade_to_junction(self):
        payload = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_urban_road_approaching_a_junction",
                    "junction_visible": True,
                    "junction_type": "signalized intersection",
                    "junction_branches": {
                        "ahead": True,
                        "left": False,
                        "right": False,
                        "known": True,
                    },
                    "target_branch_count": 3,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "has_traffic_light": True,
                    "has_crosswalk": True,
                    "ego_to_junction_distance_m": 45,
                }
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(payload)
        self.assertIsNone(error)
        map_matching = normalized["road_network"]["map_matching"]
        self.assertEqual(map_matching["topology_type"], "straight_road")
        self.assertFalse(map_matching["junction_visible"])
        self.assertEqual(map_matching["junction_type"], "none")
        self.assertEqual(
            map_matching["junction_branches"],
            {"ahead": False, "left": False, "right": False, "known": False},
        )

        signature = SceneMapMatcher.build_road_topology_signature(normalized)
        self.assertEqual(signature["topology_type"], "straight_road")
        self.assertFalse(signature["junction_visible"])
        self.assertEqual(signature["junction_branches"]["known"], False)
        self.assertNotIn("lane_markings", normalized["road_network"])
        self.assertNotIn("control_elements", normalized["road_network"])
        self.assertNotIn("lane_groups", normalized["road_network"])

    def test_extract_response_content_rejects_empty_content(self):
        with self.assertRaises(Exception) as context:
            TaskAgent._extract_response_content(
                {"choices": [{"message": {"content": "   "}}]}
            )
        self.assertIn("Empty model response content", str(context.exception))

    def test_send_request_retries_non_json_response_and_reports_preview(self):
        fake_requests = _FakeRequests(
            [
                _FakeResponse(
                    text="<html>bad gateway</html>",
                    headers={"Content-Type": "text/html"},
                    json_error=ValueError("Expecting value"),
                ),
                _FakeResponse(
                    text="<html>bad gateway</html>",
                    headers={"Content-Type": "text/html"},
                    json_error=ValueError("Expecting value"),
                ),
            ]
        )

        with mock.patch.object(task_agent_module, "requests", fake_requests):
            with mock.patch.object(task_agent_module.time, "sleep", lambda _: None):
                with self.assertRaises(Exception) as context:
                    _EchoTaskAgent().send_request(
                        "prompt",
                        {
                            "request_label": "Scene understanding revision",
                            "request_retries": 1,
                        },
                    )

        self.assertEqual(fake_requests.calls, 2)
        message = str(context.exception)
        self.assertIn("Scene understanding revision request failed after 2 attempts", message)
        self.assertIn("status=200", message)
        self.assertIn("content_type=text/html", message)
        self.assertIn("body_preview=<html>bad gateway</html>", message)

    def test_send_request_recovers_after_non_json_retry(self):
        fake_requests = _FakeRequests(
            [
                _FakeResponse(
                    text="temporary proxy error",
                    headers={"Content-Type": "text/plain"},
                    json_error=ValueError("Expecting value"),
                ),
                _FakeResponse(
                    payload={"choices": [{"message": {"content": "ok"}}]},
                    text='{"choices":[{"message":{"content":"ok"}}]}',
                    headers={"Content-Type": "application/json"},
                ),
            ]
        )

        with mock.patch.object(task_agent_module, "requests", fake_requests):
            with mock.patch.object(task_agent_module.time, "sleep", lambda _: None):
                result = _EchoTaskAgent().send_request(
                    "prompt",
                    {
                        "request_label": "Scene understanding revision",
                        "request_retries": 1,
                    },
                )

        self.assertEqual(result, "ok")
        self.assertEqual(fake_requests.calls, 2)

    def test_send_request_retries_rate_limit_http_error(self):
        fake_requests = _FakeRequests(
            [
                _FakeResponse(
                    text='{"error":"rate limit"}',
                    status_code=429,
                    headers={"Retry-After": "0.25"},
                ),
                _FakeResponse(
                    payload={"choices": [{"message": {"content": "ok"}}]},
                    text='{"choices":[{"message":{"content":"ok"}}]}',
                    headers={"Content-Type": "application/json"},
                ),
            ]
        )
        sleeps = []

        with mock.patch.object(task_agent_module, "requests", fake_requests):
            with mock.patch.object(task_agent_module.time, "sleep", sleeps.append):
                result = _EchoTaskAgent().send_request(
                    "prompt",
                    {
                        "request_label": "Scene understanding revision",
                        "request_retries": 1,
                    },
                )

        self.assertEqual(result, "ok")
        self.assertEqual(fake_requests.calls, 2)
        self.assertEqual(sleeps, [0.25])

    def test_send_request_does_not_retry_non_retryable_http_error(self):
        fake_requests = _FakeRequests(
            [
                _FakeResponse(
                    text='{"error":"bad request"}',
                    status_code=400,
                    headers={"Content-Type": "application/json"},
                ),
            ]
        )

        with mock.patch.object(task_agent_module, "requests", fake_requests):
            with self.assertRaises(Exception) as context:
                _EchoTaskAgent().send_request(
                    "prompt",
                    {
                        "request_label": "Scene understanding revision",
                        "request_retries": 2,
                    },
                )

        self.assertEqual(fake_requests.calls, 1)
        self.assertIn("status 400", str(context.exception))

    def test_scene_understanding_revision_falls_back_on_request_failure(self):
        class FailingSceneUnderstandingInterpreter(SceneUnderstandingInterpreter):
            def send_request(self, *args, **kwargs):
                raise Exception("Scene understanding revision request failed after 3 attempts")

        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {"lane_groups": []},
            "general_environment": {},
            "metadata": {"schema_version": "scene-understanding-v1"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "scene_understanding.json"
            result = FailingSceneUnderstandingInterpreter().revise_with_verification_feedback(
                scene_understanding,
                {"passed": False, "repair_hints": ["retry scene understanding"]},
                "",
                str(output_path),
            )

            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIs(result["metadata"]["verification_feedback_applied"], False)
        self.assertIs(result["metadata"]["verification_feedback_revision_failed"], True)
        self.assertIn(
            "Scene understanding revision request failed",
            result["metadata"]["verification_feedback_revision_error"],
        )
        self.assertEqual(written, result)

    def test_extract_split_sections_from_text(self):
        sections = AutoGenerator.extract_split_sections_from_text(SPLIT_TEXT)
        self.assertEqual(
            sections["Road Net Description"],
            "Straight city street with a crosswalk ahead.",
        )
        self.assertIn("scooter is ahead-left", sections["Road Users Description"])
        self.assertIn("traffic light is visible", sections["Static Objects Description"])
        self.assertIn("nearby scooter", sections["Scenario Description"])

    def test_format_scene_context_includes_all_sections(self):
        sections = AutoGenerator.extract_split_sections_from_text(SPLIT_TEXT)
        context = ObstacleGenerator.format_scene_context(
            sections,
            "Map Description:\nMinimal road summary\nNetwork XML:\n<net/>",
            "CARLA Spawn Points:\n- index=0",
        )
        self.assertIn("Road Users Description:", context)
        self.assertIn("Static Objects Description:", context)
        self.assertIn("Generated Road Network Summary:", context)
        self.assertIn("CARLA Spawn Points:", context)

    def test_structured_default_pipeline_writes_core_fallback_artifacts(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "subject_car",
                    "category": "car",
                    "subtype": "car",
                    "appearance": {"color": "blue", "color_rgb": "54,116,168"},
                    "visual_confidence": "high",
                    "motion_state": "moving",
                    "heading_relation_to_ego": "same_direction",
                    "lane_side_relation": "same_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Car ahead in same lane.",
                    "must_reconstruct": True,
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [],
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                        "lane_width_class": "standard",
                    }
                ],
                "junctions": [],
                "special_road_areas": [],
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {"schema_version": "scene-understanding-v1"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "debug_artifacts": False,
                },
            )
            scene_id = "structured_scene"
            relation_dsl = generator.generate_relation_dsl(scene_id, scene_understanding)
            raw = generator.generate_initial_coordinates(scene_id, relation_dsl)
            ordered = generator.apply_pairwise_ordering_step(scene_id, raw, relation_dsl)
            projected = generator.project_entities_step(scene_id, ordered, scene_understanding)
            refined = generator.refine_coordinates_step(scene_id, projected, relation_dsl)
            validation = generator.validate_layout_step(scene_id, relation_dsl, refined)
            match_path = generator.analyze_structured_scene_match(
                scene_id,
                scene_understanding,
                refined,
                validation,
            )
            payload = generator.build_spawn_payload_from_match_or_fallback(
                scene_id,
                refined,
                match_path,
            )
            script_path = generator.generate_final_scene_script(scene_id, match_path)
            py_compile.compile(script_path, doraise=True)

            self.assertTrue(Path(match_path).exists())
            self.assertTrue((Path(tmp) / f"{scene_id}_actors.json").exists())
            self.assertTrue(Path(script_path).exists())
            entity_ids = [entity["id"] for entity in payload["entities"]]
            self.assertEqual(entity_ids[0], "ego")
            self.assertIn("subject_car", entity_ids)
            self.assertIn(
                "# scene_match_status: unavailable",
                Path(script_path).read_text(encoding="utf-8"),
            )
            self.assertIn(
                "_autoscenario_clear_existing_dynamic_actors()",
                Path(script_path).read_text(encoding="utf-8"),
            )
            self.assertFalse((Path(tmp) / f"{scene_id}_relation_dsl.json").exists())

    def test_generate_scene_understanding_skips_merge_for_empty_user_description(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {"lane_groups": []},
            "general_environment": {},
            "metadata": {"schema_version": "scene-understanding-v1"},
        }

        class FakeInterpreter:
            def __init__(self):
                self.merge_called = False

            def call_agent(self, user_request, add_info):
                return dict(scene_understanding)

            def merge_with_user_description(self, *args, **kwargs):
                self.merge_called = True
                raise AssertionError("merge should not be called for empty description")

        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "merge_user_description": True,
                },
            )
            fake = FakeInterpreter()
            generator.scene_understanding_interpreter = fake
            payload = generator.generate_scene_understanding(
                "",
                {
                    "scene_id": "empty_merge",
                    "image_path": "unused.jpg",
                    "user_scene_description": "   ",
                },
            )
            self.assertFalse(fake.merge_called)
            self.assertFalse(payload["metadata"]["user_description_applied"])

    def test_generate_scene_understanding_merges_user_description_without_extra_files(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {"lane_groups": []},
            "general_environment": {},
            "metadata": {"schema_version": "scene-understanding-v1"},
        }
        merged_scene = {
            **scene_understanding,
            "background_traffic": [
                {
                    "id": "right_parked_scooter",
                    "category": "parked_vehicle",
                    "subtype": "motor_scooter",
                    "lane_side_relation": "right_edge",
                }
            ],
            "metadata": {"schema_version": "scene-understanding-v1", "user_description_applied": True},
        }

        class FakeInterpreter:
            def __init__(self):
                self.merge_called = False

            def call_agent(self, user_request, add_info):
                return dict(scene_understanding)

            def merge_with_user_description(self, scene, description, output_fn):
                self.merge_called = True
                self.description = description
                return dict(merged_scene)

        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "merge_user_description": True,
                },
            )
            fake = FakeInterpreter()
            generator.scene_understanding_interpreter = fake
            payload = generator.generate_scene_understanding(
                "",
                {
                    "scene_id": "merged",
                    "image_path": "unused.jpg",
                    "user_scene_description": "右侧停车道近处有一辆停靠摩托车",
                },
            )
            self.assertTrue(fake.merge_called)
            self.assertEqual(fake.description, "右侧停车道近处有一辆停靠摩托车")
            self.assertEqual(payload["background_traffic"][0]["id"], "right_parked_scooter")
            self.assertTrue(payload["metadata"]["user_description_applied"])
            self.assertTrue((Path(tmp) / "merged_su.json").exists())
            self.assertFalse((Path(tmp) / "merged_user_description.txt").exists())

    def test_cone_category_normalizes_to_static_spawn_payload(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "cone_row",
                    "category": "traffic_cone",
                    "subtype": "cone_row",
                    "visual_confidence": "high",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "same_direction",
                    "lane_side_relation": "left_parking_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Cones along the left parking lane.",
                    "must_reconstruct": True,
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        self.assertEqual(normalized["traffic_subjects"][0]["category"], "cone_group")

        payload = build_projected_spawn_payload(
            {
                "entities": [
                    {
                        "id": "cone_row",
                        "category": "cone_group",
                        "spawn_kind": "static",
                        "blueprint_name": "constructioncone",
                        "location": {"x": 1.0, "y": 2.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    }
                ]
            }
        )
        self.assertEqual(payload["entities"][0]["spawn_kind"], "static")
        self.assertEqual(payload["entities"][0]["blueprint_name"], "constructioncone")

    def test_junction_actor_anchor_survives_relation_and_coordinate_stages(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "left_arm_car",
                    "category": "car",
                    "subtype": "sedan",
                    "visual_confidence": "high",
                    "motion_state": "moving",
                    "heading_relation_to_ego": "crossing",
                    "lane_side_relation": "left_lane",
                    "lane_index_relation": -1,
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Vehicle on the left arm of the intersection.",
                    "must_reconstruct": True,
                    "layout_anchor_id": "left_arm",
                    "anchor_relation": {
                        "travel_direction": "toward_junction",
                        "lane_from_right": 1,
                        "position_along_anchor": "near_mouth",
                    },
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 2}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [{"type": "cross_intersection", "location": "near"}],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "actor_layout": {
                "global_anchor": {"type": "junction_center", "confidence": "high"}
            },
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        relation_dsl = build_relation_dsl(normalized, None)
        relation_entity = relation_dsl["ego_relations"][0]
        coordinates = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        coordinate_entity = coordinates["entities"][0]

        self.assertEqual(relation_entity["layout_anchor_id"], "left_arm")
        self.assertEqual(relation_entity["anchor_relation"]["lane_from_right"], 1)
        self.assertEqual(coordinate_entity["layout_anchor_id"], "left_arm")
        self.assertEqual(coordinate_entity["anchor_relation"]["travel_direction"], "toward_junction")
        self.assertEqual(coordinate_entity["longitudinal_relation"], "ahead")
        self.assertEqual(coordinate_entity["longitudinal_proximity"], "near")
        self.assertGreater(coordinate_entity["longitudinal_m"], 0.0)

    def test_soft_aligned_pairwise_refine_updates_longitudinal_m(self):
        anchor = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        projected = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "det_1",
                    "category": "car",
                    "lane_index_relation": 0,
                    "longitudinal_m": 4.0,
                    "location": {"x": 4.0, "y": 0.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "det_2",
                    "category": "truck",
                    "lane_index_relation": -1,
                    "longitudinal_m": 10.0,
                    "location": {"x": 10.0, "y": -8.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
        }
        relation_dsl = {
            "selected_anchor_lane": anchor,
            "lane_context": {},
            "pairwise_relations": [
                {
                    "entity_id": "det_2",
                    "other_entity_id": "det_1",
                    "constraint_strength": "soft",
                    "longitudinal_relation": "aligned_with_other",
                    "lateral_relation": "left_of_other",
                }
            ],
        }

        refined = refine_projected_coordinates_with_pairwise_relations(projected, relation_dsl)
        by_id = {entity["id"]: entity for entity in refined["entities"]}

        self.assertAlmostEqual(by_id["det_2"]["location"]["x"], 4.0)
        self.assertAlmostEqual(by_id["det_2"]["longitudinal_m"], 4.0)

    def test_adjacent_driving_lanes_are_not_collision_spaced_longitudinally(self):
        anchor = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "lane_groups": [{"lane_width_class": "standard"}],
            },
            "general_environment": {},
            "metadata": {},
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "det_1",
                    "category": "car",
                    "lane_index_relation": 0,
                    "lane_side_relation": "same_lane",
                    "heading_relation": "same_direction",
                    "longitudinal_m": 4.0,
                    "location": {"x": 4.0, "y": 0.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "det_2",
                    "category": "truck",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "heading_relation": "same_direction",
                    "longitudinal_m": 4.0,
                    "location": {"x": 4.0, "y": -3.5, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
        }

        projected = project_entities_to_carla_context(
            raw,
            scene_understanding,
            {"topology_sample": [anchor]},
        )
        by_id = {entity["id"]: entity for entity in projected["entities"]}

        self.assertAlmostEqual(by_id["det_1"]["location"]["x"], by_id["det_2"]["location"]["x"])
        self.assertAlmostEqual(by_id["det_2"]["longitudinal_m"], 4.0)

    def test_same_direction_left_lane_uses_same_carriageway_not_opposing_lane(self):
        anchor = {
            "road_id": 10,
            "lane_id": -2,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        topology_sample = [
            anchor,
            {
                "road_id": 10,
                "lane_id": -1,
                "start": {"x": 0.0, "y": -3.5, "z": 0.0, "yaw": 0.0},
                "end": {"x": 50.0, "y": -3.5, "z": 0.0, "yaw": 0.0},
            },
            {
                "road_id": 10,
                "lane_id": -3,
                "start": {"x": 0.0, "y": 3.5, "z": 0.0, "yaw": 0.0},
                "end": {"x": 50.0, "y": 3.5, "z": 0.0, "yaw": 0.0},
            },
            {
                "road_id": 10,
                "lane_id": 1,
                "start": {"x": 0.0, "y": -7.0, "z": 0.0, "yaw": 180.0},
                "end": {"x": -50.0, "y": -7.0, "z": 0.0, "yaw": 180.0},
            },
        ]
        scene_understanding = {
            "traffic_subjects": [],
            "road_network": {"lane_groups": [{"lane_width_class": "standard"}]},
            "general_environment": {},
            "metadata": {},
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "left_same_dir",
                    "category": "car",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "heading_relation": "same_direction",
                    "location": {"x": 12.0, "y": -3.5, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "oncoming",
                    "category": "car",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "heading_relation": "opposite_direction",
                    "flow_compliance": "legal",
                    "location": {"x": 18.0, "y": -7.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                },
            ],
        }

        projected = project_entities_to_carla_context(
            raw,
            scene_understanding,
            {"topology_sample": topology_sample},
        )
        by_id = {entity["id"]: entity for entity in projected["entities"]}

        self.assertEqual(by_id["left_same_dir"]["projected_lane"]["lane_id"], -1)
        self.assertNotEqual(by_id["left_same_dir"]["projected_lane"]["lane_id"], 1)
        self.assertEqual(by_id["oncoming"]["projected_lane"]["lane_id"], 1)

    def test_parking_spawn_payload_preserves_projected_lane(self):
        payload = build_projected_spawn_payload(
            {
                "entities": [
                    {
                        "id": "parked_car",
                        "category": "car",
                        "spawn_kind": "vehicle",
                        "blueprint_name": "car",
                        "lane_side_relation": "right_parking_lane",
                        "lane_index_relation": 1,
                        "heading_relation": "same_direction",
                        "motion_state": "parked",
                        "placement_mode_hint": "parking_lane_actor",
                        "projected_lane": {"road_id": 76, "lane_id": -4, "yaw": 0.0},
                        "location": {"x": 1.0, "y": 2.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    }
                ]
            }
        )

        entity = payload["entities"][0]
        self.assertEqual(entity["placement_mode"], "project_to_parking_lane")
        self.assertEqual(entity["projected_lane"]["road_id"], 76)
        self.assertEqual(entity["projected_lane"]["lane_id"], -4)

    def test_legacy_background_traffic_promotes_to_vehicle_agents(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [
                {
                    "id": "right_curb_row",
                    "category": "parked_vehicle",
                    "subtype": "parked_vehicle_row",
                    "representative_count": 3,
                    "lane_side_relation": "right_edge",
                    "longitudinal_band": "mid",
                    "heading_relation_to_ego": "same_direction",
                    "density_role": "curbside_row",
                    "confidence": "high",
                    "source": "visible right-side vehicle row",
                },
                {
                    "id": "orange_scooter",
                    "category": "parked_scooter",
                    "subtype": "motor_scooter",
                    "representative_count": 1,
                    "lane_side_relation": "right_edge",
                    "longitudinal_band": "near",
                    "confidence": "high",
                },
            ],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        self.assertEqual(normalized["background_traffic"], [])
        by_id = {entity["id"]: entity for entity in normalized["traffic_subjects"]}
        self.assertEqual(by_id["right_curb_row"]["category"], "car")
        self.assertEqual(by_id["right_curb_row"]["count"], 3)
        self.assertEqual(by_id["orange_scooter"]["category"], "motorcycle")

    def test_scene_understanding_background_traffic_is_optional(self):
        scene_understanding = {
            "traffic_subjects": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "one_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 0}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "actor_layout": {
                "global_anchor": {"type": "ego_approach", "confidence": "high"}
            },
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(scene_understanding)

        self.assertIsNone(error)
        self.assertEqual(normalized["background_traffic"], [])
        self.assertEqual(
            normalized["actor_layout"]["global_anchor"]["type"],
            "ego_approach",
        )

    def test_right_edge_vehicle_preserves_edge_semantics(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "right_car",
                    "category": "parked_vehicle",
                    "subtype": "parked_car_row",
                    "visual_confidence": "high",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "same_direction",
                    "lane_side_relation": "right_edge",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Visible right-side car.",
                    "must_reconstruct": True,
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "right_parking_lane_count": 1,
                        "lane_width_class": "standard",
                    }
                ],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        spawn_context = {
            "topology_sample": [
                {
                    "road_id": 1,
                    "lane_id": -1,
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                }
            ],
            "spawn_points": [],
        }
        relation_dsl = build_relation_dsl(normalized, spawn_context)
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        projected = project_entities_to_carla_context(raw, normalized, spawn_context)
        entity = projected["entities"][0]
        self.assertEqual(entity["category"], "car")
        self.assertEqual(entity["lane_side_relation"], "right_edge")
        self.assertEqual(entity["lane_index_relation"], 1)
        self.assertAlmostEqual(entity["location"]["y"], 3.5, places=3)
        payload = build_projected_spawn_payload(projected)
        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_lane")
        self.assertEqual(payload["entities"][0]["lane_index_relation"], 1)
        self.assertNotEqual(payload["entities"][0]["category"], "parked_vehicle")

    def test_vehicle_lane_index_two_projects_to_second_right_lane(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "right_car_2",
                    "category": "car",
                    "subtype": "sedan",
                    "visual_confidence": "high",
                    "heading_relation_to_ego": "same_direction",
                    "lane_index_relation": 2,
                    "lane_side_relation": "right_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Vehicle in second lane to the ego-right.",
                    "must_reconstruct": True,
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "one_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 3,
                        "opposing_lane_count": 0,
                        "lane_width_class": "standard",
                    }
                ],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        spawn_context = {
            "topology_sample": [
                {
                    "road_id": 1,
                    "lane_id": -1,
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                }
            ],
            "spawn_points": [],
        }
        relation_dsl = build_relation_dsl(normalized, spawn_context)
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        projected = project_entities_to_carla_context(raw, normalized, spawn_context)
        entity = projected["entities"][0]
        self.assertEqual(entity["lane_index_relation"], 2)
        self.assertEqual(entity["lane_side_relation"], "right_lane")
        self.assertAlmostEqual(entity["location"]["y"], 7.0, places=3)
        payload = build_projected_spawn_payload(projected)
        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_lane")

    def test_normal_straight_road_pipeline_projects_regular_vehicles(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "lead",
                    "category": "car",
                    "heading_relation_to_ego": "same_direction",
                    "flow_compliance": "legal",
                    "lane_index_relation": 0,
                    "lane_side_relation": "same_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                },
                {
                    "id": "right_car",
                    "category": "car",
                    "heading_relation_to_ego": "same_direction",
                    "flow_compliance": "legal",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                    "count": 1,
                },
                {
                    "id": "oncoming",
                    "category": "car",
                    "heading_relation_to_ego": "opposite_direction",
                    "flow_compliance": "legal",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                    "count": 1,
                },
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "junction_type": "none",
                    "junction_branches": {
                        "ahead": False,
                        "left": False,
                        "right": False,
                        "known": False,
                    },
                    "forward_lane_count": 2,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 3,
                    "curve_direction": "straight",
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 2,
                        "opposing_lane_count": 1,
                        "lane_width_class": "standard",
                    }
                ],
            },
            "actor_layout": {},
            "general_environment": {},
            "metadata": {},
        }
        spawn_context = {
            "topology_sample": [
                {
                    "road_id": 1,
                    "lane_id": -1,
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 80.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                },
                {
                    "road_id": 1,
                    "lane_id": -2,
                    "start": {"x": 0.0, "y": 3.5, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 80.0, "y": 3.5, "z": 0.0, "yaw": 0.0},
                },
                {
                    "road_id": 1,
                    "lane_id": 1,
                    "start": {"x": 80.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
                    "end": {"x": 0.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
                },
            ],
            "dense_local_waypoints": [],
        }

        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        relation_dsl = build_relation_dsl(normalized, spawn_context)
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        projected = project_entities_to_carla_context(raw, normalized, spawn_context)
        payload = build_projected_spawn_payload(projected)
        validation = validate_relation_layout(relation_dsl, projected)

        self.assertEqual(relation_dsl["metadata"]["layout_scene_kind"], "open_road")
        by_id = {entity["id"]: entity for entity in payload["entities"]}
        self.assertEqual(by_id["lead"]["placement_mode"], "project_to_lane")
        self.assertEqual(by_id["right_car"]["placement_mode"], "project_to_lane")
        self.assertEqual(by_id["oncoming"]["placement_mode"], "project_to_opposing_lane")
        self.assertEqual(by_id["lead"]["actor_group_type"], "individual_vehicle")
        self.assertEqual(by_id["right_car"]["placement_mode_hint"], "normal_lane_actor")
        self.assertEqual(by_id["oncoming"]["opposing_lane_from_median"], 1)
        self.assertEqual(validation["summary"]["ego_consistency"]["failed"], 0)
        self.assertEqual(validation["summary"]["pairwise_consistency"]["failed"], 0)
        self.assertEqual(validation["summary"]["lane_legality"]["failed"], 0)

    def test_junction_pipeline_projects_frame_actors_to_junction_lanes(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "right_arm_car",
                    "category": "car",
                    "heading_relation_to_ego": "crossing",
                    "flow_compliance": "legal",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "layout_anchor_id": "right_arm",
                    "anchor_relation": {"travel_direction": "toward_junction"},
                    "count": 1,
                },
                {
                    "id": "oncoming",
                    "category": "car",
                    "heading_relation_to_ego": "opposite_direction",
                    "flow_compliance": "legal",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                    "layout_anchor_id": "oncoming_arm",
                    "anchor_relation": {"travel_direction": "toward_junction"},
                    "count": 1,
                },
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "cross_intersection",
                    "junction_visible": True,
                    "junction_type": "cross_intersection",
                    "junction_branches": {
                        "ahead": True,
                        "left": True,
                        "right": True,
                        "known": True,
                    },
                    "target_branch_count": 4,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 4,
                    "curve_direction": "straight",
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "lane_width_class": "standard",
                    }
                ],
            },
            "actor_layout": {
                "global_anchor": {"type": "junction_center", "confidence": "high"},
                "anchors": [
                    {"id": "right_arm", "type": "junction_arm", "side": "right"},
                    {"id": "oncoming_arm", "type": "junction_arm", "side": "oncoming"},
                ],
            },
            "general_environment": {},
            "metadata": {},
        }
        matched_structure = {
            "kind": "junction",
            "junction_id": 1,
            "leg_count": 4,
            "center": {"x": 100.0, "y": 0.0, "z": 0.0},
            "ego_approach_heading_deg": 0.0,
            "ego_distance_to_center_m": 30.0,
            "legs": [
                {
                    "name": "west",
                    "heading_out_deg": -180.0,
                    "inbound_lanes": [{"road_id": 10, "lane_id": -1, "role": "inbound"}],
                },
                {
                    "name": "east",
                    "heading_out_deg": 0.0,
                    "inbound_lanes": [{"road_id": 20, "lane_id": 1, "role": "inbound"}],
                },
                {
                    "name": "south",
                    "heading_out_deg": 90.0,
                    "inbound_lanes": [{"road_id": 30, "lane_id": 1, "role": "inbound"}],
                },
                {
                    "name": "north",
                    "heading_out_deg": -90.0,
                    "inbound_lanes": [{"road_id": 40, "lane_id": -1, "role": "inbound"}],
                },
            ],
        }
        spawn_context = {
            "topology_sample": [
                {
                    "road_id": 10,
                    "lane_id": -1,
                    "start": {"x": 70.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 100.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                }
            ],
            "matched_structure": matched_structure,
        }

        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        relation_dsl = build_relation_dsl(normalized, spawn_context)
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        reprojected = reproject_actors_for_junction(raw, matched_structure)
        payload = build_projected_spawn_payload(reprojected)

        self.assertEqual(relation_dsl["metadata"]["layout_scene_kind"], "junction")
        by_id = {entity["id"]: entity for entity in payload["entities"]}
        self.assertEqual(by_id["right_arm_car"]["placement_mode"], "project_to_junction_lane")
        self.assertEqual(by_id["oncoming"]["placement_mode"], "project_to_junction_lane")
        self.assertEqual(by_id["right_arm_car"]["actor_group_type"], "individual_vehicle")
        self.assertEqual(by_id["oncoming"]["placement_mode_hint"], "normal_lane_actor")
        self.assertEqual(by_id["right_arm_car"]["junction_leg"], "south")
        self.assertEqual(by_id["oncoming"]["junction_leg"], "east")
        self.assertEqual(by_id["oncoming"]["projected_lane"]["source"], "junction_leg")
        self.assertNotIn("opposing_lane_from_median", by_id["oncoming"])

    def test_open_road_parked_vehicle_uses_normal_lane_projection(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "parked_right",
                    "category": "car",
                    "subtype": "parked_car",
                    "visual_confidence": "high",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "same_direction",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_edge",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "count": 1,
                    "evidence": "Parked curbside vehicle.",
                    "must_reconstruct": True,
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                },
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 0}],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        spawn_context = {
            "topology_sample": [
                {
                    "road_id": 1,
                    "lane_id": -1,
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                    "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
                }
            ]
        }
        relation_dsl = build_relation_dsl(normalized, spawn_context)
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        projected = project_entities_to_carla_context(raw, normalized, spawn_context)
        payload = build_projected_spawn_payload(projected)

        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_lane")
        self.assertEqual(payload["entities"][0]["layout_version"], "v2")

    def test_open_road_crossing_vehicle_is_degraded_to_lane_flow(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "bad_crossing",
                    "category": "car",
                    "heading_relation_to_ego": "crossing",
                    "lane_index_relation": 0,
                    "lane_side_relation": "same_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                }
            ],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                },
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 0}],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        relation_dsl = build_relation_dsl(normalized, {"topology_sample": []})
        relation = relation_dsl["entities"][0]
        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        payload = build_projected_spawn_payload(
            project_entities_to_carla_context(raw, normalized, {"topology_sample": []})
        )

        self.assertEqual(relation["original_heading_relation"], "crossing")
        self.assertEqual(relation["heading_relation"], "unknown")
        self.assertIn("open-road crossing", relation["degraded_reason"])
        self.assertEqual(payload["entities"][0]["heading_relation"], "unknown")
        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_lane")

    def test_wrong_way_vehicle_stays_out_of_opposing_lane_mode(self):
        payload = build_projected_spawn_payload(
            {
                "metadata": {"layout_version": "v2", "layout_scene_kind": "open_road"},
                "entities": [
                    {
                        "id": "wrong_way_bike",
                        "spawn_kind": "vehicle",
                        "category": "motorcycle",
                        "subtype": "motorcycle",
                        "lane_side_relation": "same_lane",
                        "lane_index_relation": 0,
                        "heading_relation": "opposite_direction",
                        "flow_compliance": "wrong_way",
                        "motion_state": "moving",
                        "location": {"x": 8.0, "y": 0.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                    }
                ],
            }
        )

        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_lane")
        self.assertEqual(payload["entities"][0]["flow_compliance"], "wrong_way")

    def test_parking_row_fields_do_not_preserve_xy(self):
        payload = build_projected_spawn_payload(
            {
                "metadata": {"layout_version": "v2", "layout_scene_kind": "open_road"},
                "entities": [
                    {
                        "id": "parked_row_0",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "subtype": "car",
                        "lane_side_relation": "right_edge",
                        "lane_index_relation": 1,
                        "heading_relation": "opposite_direction",
                        "flow_compliance": "legal",
                        "motion_state": "parked",
                        "actor_group_type": "parking_row",
                        "actor_group_id": "parking_row_right",
                        "group_order_index": 0,
                        "placement_mode_hint": "preserve_parking_xy",
                        "location": {"x": 8.0, "y": 3.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                    }
                ],
            }
        )

        entity = payload["entities"][0]
        self.assertEqual(entity["placement_mode"], "project_to_opposing_lane")
        self.assertEqual(entity["heading_relation"], "opposite_direction")
        self.assertEqual(entity["actor_group_type"], "individual_vehicle")
        self.assertEqual(entity["placement_mode_hint"], "normal_lane_actor")

    def test_parking_lane_actor_projects_to_parking_lane(self):
        payload = build_projected_spawn_payload(
            {
                "metadata": {"layout_version": "v2", "layout_scene_kind": "open_road"},
                "entities": [
                    {
                        "id": "right_parked",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "lane_side_relation": "right_parking_lane",
                        "lane_index_relation": 1,
                        "heading_relation": "same_direction",
                        "flow_compliance": "legal",
                        "motion_state": "parked",
                        "actor_group_type": "parking_row",
                        "actor_group_id": "right_parking_row",
                        "placement_mode_hint": "parking_lane_actor",
                        "location": {"x": 8.0, "y": 3.4, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    },
                    {
                        "id": "oncoming",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "lane_side_relation": "left_lane",
                        "lane_index_relation": -1,
                        "heading_relation": "opposite_direction",
                        "flow_compliance": "legal",
                        "motion_state": "moving",
                        "actor_group_type": "individual_vehicle",
                        "placement_mode_hint": "normal_lane_actor",
                        "location": {"x": 18.0, "y": -3.5, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                    },
                ],
            }
        )

        by_id = {entity["id"]: entity for entity in payload["entities"]}
        self.assertEqual(
            by_id["right_parked"]["placement_mode"],
            "project_to_parking_lane",
        )
        self.assertEqual(
            by_id["right_parked"]["placement_mode_hint"],
            "parking_lane_actor",
        )
        self.assertEqual(by_id["oncoming"]["placement_mode"], "project_to_opposing_lane")

    def test_junction_traffic_queue_fields_are_canonicalized_but_keep_junction_projection(self):
        payload = build_projected_spawn_payload(
            {
                "metadata": {"layout_version": "v2", "layout_scene_kind": "junction"},
                "entities": [
                    {
                        "id": "queue_0",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "subtype": "car",
                        "lane_side_relation": "same_lane",
                        "lane_index_relation": 0,
                        "heading_relation": "same_direction",
                        "flow_compliance": "legal",
                        "motion_state": "stopped",
                        "actor_group_type": "traffic_queue",
                        "actor_group_id": "right_arm_queue",
                        "group_order_index": 0,
                        "placement_mode_hint": "queue_on_lane",
                        "junction_placement": "frame",
                        "junction_direction": "ego",
                        "junction_leg": "ego",
                        "junction_motion": "approaching",
                        "projected_lane": {"source": "junction_leg", "yaw": 0.0},
                        "location": {"x": 8.0, "y": 0.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    }
                ],
            }
        )

        entity = payload["entities"][0]
        self.assertEqual(entity["placement_mode"], "project_to_junction_lane")
        self.assertEqual(entity["actor_group_type"], "individual_vehicle")
        self.assertEqual(entity["placement_mode_hint"], "normal_lane_actor")

    def test_parking_row_normalization_uses_standard_heading_correction(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "parked_right",
                    "category": "car",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "opposite_direction",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "actor_group_type": "parking_row",
                    "actor_group_id": "parking_row_right",
                    "group_order_index": 0,
                    "placement_mode_hint": "preserve_parking_xy",
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                },
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 1}],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        subject = normalized["traffic_subjects"][0]
        self.assertEqual(subject["lane_index_relation"], -1)
        self.assertEqual(subject["actor_group_type"], "individual_vehicle")

        relation_dsl = build_relation_dsl(normalized, {"topology_sample": []})
        relation = relation_dsl["entities"][0]
        self.assertEqual(relation["lane_anchor"], "lane_center")
        self.assertEqual(relation["placement_mode_hint"], "normal_lane_actor")

    def test_vehicle_position_parking_lane_semantics_are_preserved(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "curb_car",
                    "category": "car",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "same_direction",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_parking_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "near",
                    "placement_mode_hint": "parking_lane_actor",
                },
                {
                    "id": "lead_car",
                    "category": "car",
                    "motion_state": "parked",
                    "heading_relation_to_ego": "same_direction",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_parking_lane",
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                    "placement_mode_hint": "parking_lane_actor",
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [
                {
                    "entity_id": "curb_car",
                    "other_entity_id": "lead_car",
                    "longitudinal_relation": "behind_other",
                    "lane_relation": "same_parking_lane",
                }
            ],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "right_parking_lane_count": 1,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {},
        }

        normalized, error = normalize_scene_understanding(scene_understanding)

        self.assertIsNone(error)
        subject = normalized["traffic_subjects"][0]
        self.assertEqual(subject["lane_side_relation"], "right_parking_lane")
        self.assertEqual(subject["placement_mode_hint"], "parking_lane_actor")

        relation_dsl = build_relation_dsl(normalized, {"topology_sample": []})
        relation = relation_dsl["entities"][0]
        self.assertEqual(relation["lane_anchor"], "parking_lane")
        self.assertEqual(relation["lateral_mode"], "parking_lane")
        self.assertEqual(relation["lane_side_relation"], "right_parking_lane")
        self.assertEqual(
            relation_dsl["pairwise_relations"][0]["lane_relation"],
            "same_parking_lane",
        )

    def test_left_parking_lane_actor_stays_outside_left_lane_projection(self):
        anchor = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        left_lane = {
            "road_id": 1,
            "lane_id": 1,
            "start": {"x": 0.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
            "end": {"x": 50.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
        }
        relation_dsl = {
            "lane_width_class": "standard",
            "lane_context": {
                "lane_roles": {
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "left_parking_lane_count": 1,
                }
            },
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "entity_id": "det_1",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "lane_anchor": "parking_lane",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_parking_lane",
                    "heading_relation": "unknown",
                    "motion_state": "parked",
                    "order_relation": "alongside",
                    "distance_band": "immediate",
                    "placement_mode_hint": "parking_lane_actor",
                    "actor_group_type": "parking_row",
                    "actor_group_id": "park_left_1",
                }
            ],
        }
        initial = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        expected_parking_y = -2.0 * 3.5
        self.assertAlmostEqual(
            initial["entities"][0]["location"]["y"],
            expected_parking_y,
            places=3,
        )

        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                },
                "lane_groups": [
                    {
                        "lane_width_class": "standard",
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 1,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {},
        }
        projected = project_entities_to_carla_context(
            initial,
            scene_understanding,
            {
                "topology_sample": [anchor, left_lane],
                "dense_local_waypoints": [
                    {"road_id": 1, "lane_id": -1, "x": 4.0, "y": 0.0, "z": 0.1, "yaw": 0.0},
                    {"road_id": 1, "lane_id": 1, "x": 4.0, "y": -3.5, "z": 0.1, "yaw": 180.0},
                ],
            },
        )
        entity = projected["entities"][0]
        self.assertAlmostEqual(entity["location"]["y"], expected_parking_y, places=3)
        self.assertEqual(entity["projected_lane"]["lane_id"], -1)

        validation = validate_relation_layout(relation_dsl, projected)
        self.assertTrue(validation["entity_results"][0]["lateral_check"])

        spawn_payload = build_projected_spawn_payload(projected)
        spawn_entity = spawn_payload["entities"][0]
        self.assertEqual(spawn_entity["placement_mode"], "project_to_parking_lane")
        self.assertEqual(spawn_entity["projected_lane"]["road_id"], 1)
        self.assertEqual(spawn_entity["projected_lane"]["lane_id"], -1)

    def test_parking_lane_actor_collision_spacing_is_compact(self):
        anchor = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "lane_width_class": "standard",
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "right_parking_lane_count": 1,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {},
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "parked_a",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "parking_lane",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_parking_lane",
                    "heading_relation": "same_direction",
                    "motion_state": "parked",
                    "placement_mode_hint": "parking_lane_actor",
                    "actor_group_type": "parking_row",
                    "location": {"x": 8.0, "y": 7.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "parked_b",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "parking_lane",
                    "lane_index_relation": 1,
                    "lane_side_relation": "right_parking_lane",
                    "heading_relation": "same_direction",
                    "motion_state": "parked",
                    "placement_mode_hint": "parking_lane_actor",
                    "actor_group_type": "parking_row",
                    "location": {"x": 8.0, "y": 7.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
        }

        projected = project_entities_to_carla_context(
            raw,
            scene_understanding,
            {"topology_sample": [anchor]},
        )
        a, b = projected["entities"]
        distance = math.hypot(
            a["location"]["x"] - b["location"]["x"],
            a["location"]["y"] - b["location"]["y"],
        )
        self.assertGreaterEqual(distance, 3.5)
        self.assertLess(distance, 4.0)

    def test_oncoming_vehicle_keeps_opposite_yaw_after_dense_snap(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        anchor = {
            "road_id": 1,
            "lane_id": 1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        opposing = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
            "end": {"x": 50.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
        }
        spawn_context = {
            "topology_sample": [anchor, opposing],
            "dense_local_waypoints": [
                {"road_id": 1, "lane_id": -1, "x": 20.0, "y": -3.5, "z": 0.0, "yaw": 180.0}
            ],
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "oncoming",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "center_of_lane",
                    "lane_side_relation": "left_lane",
                    "heading_relation": "opposite_direction",
                    "motion_state": "moving",
                    "location": {"x": 20.0, "y": -3.5, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                }
            ],
        }

        projected = project_entities_to_carla_context(raw, scene_understanding, spawn_context)
        yaw = projected["entities"][0]["rotation"]["yaw"]
        yaw_delta = abs(((yaw - anchor["start"]["yaw"] + 180.0) % 360.0) - 180.0)
        self.assertAlmostEqual(yaw_delta, 180.0, places=3)

        payload = build_projected_spawn_payload(projected)
        entity = payload["entities"][0]
        self.assertEqual(entity["heading_relation"], "opposite_direction")
        self.assertEqual(entity["motion_state"], "moving")
        self.assertEqual(entity["projected_lane"]["lane_id"], -1)

    def test_oncoming_anchor_fallback_lateral_survives_dense_snap(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        anchor = {
            "road_id": 76,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "det_7",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "center_of_lane",
                    "lane_index_relation": -1,
                    "lane_side_relation": "left_lane",
                    "heading_relation": "opposite_direction",
                    "motion_state": "moving",
                    "location": {"x": 22.0, "y": -3.5, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                }
            ],
        }
        projected = project_entities_to_carla_context(
            raw,
            scene_understanding,
            {
                "topology_sample": [anchor],
                "dense_local_waypoints": [
                    {"road_id": 76, "lane_id": -1, "x": 22.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
                ],
            },
        )

        entity = projected["entities"][0]
        self.assertTrue(entity["opposing_lane_fallback_to_anchor"])
        self.assertAlmostEqual(entity["location"]["y"], -3.5, places=3)
        self.assertEqual(entity["projected_lane"]["lane_id"], -1)

    def test_road_reprojection_restores_oncoming_lateral_when_collapsed_to_center(self):
        coordinates = {
            "entities": [
                {
                    "id": "det_7",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "heading_relation": "opposite_direction",
                    "flow_compliance": "unknown",
                    "lane_index_relation": -1,
                    "location": {"x": 22.0, "y": 0.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                }
            ],
            "metadata": {},
        }
        reprojected = reproject_actors_for_structure(
            coordinates,
            {
                "kind": "road_segment",
                "road_id": 76,
                "lane_id": -1,
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward_heading_deg": 0.0,
                "curve_samples": [[0.0, 0.0], [50.0, 0.0]],
            },
        )

        entity = reprojected["entities"][0]
        self.assertAlmostEqual(entity["location"]["y"], -3.5, places=3)
        assignment = reprojected["metadata"]["road_reprojection"]["assignments"][0]
        self.assertEqual(assignment["side"], "opposing")
        self.assertEqual(assignment["lateral_m"], -3.5)

    def test_canonical_turn_intent_normalizes_synonyms(self):
        self.assertEqual(_canonical_turn_intent("Turning Left"), "left")
        self.assertEqual(_canonical_turn_intent("left_turn"), "left")
        self.assertEqual(_canonical_turn_intent("turn_right"), "right")
        self.assertEqual(_canonical_turn_intent("straight"), "through")
        self.assertEqual(_canonical_turn_intent(None), "none")
        self.assertEqual(_canonical_turn_intent("garbage"), "none")

    def test_apply_turn_intent_deflects_by_actor_perspective(self):
        # CARLA frame: driver's right is yaw+90, so left decreases yaw, right increases it.
        self.assertAlmostEqual(_apply_turn_intent(100.0, "left"), 100.0 - TURN_INTENT_YAW_DEG)
        self.assertAlmostEqual(_apply_turn_intent(100.0, "right"), 100.0 + TURN_INTENT_YAW_DEG)
        self.assertAlmostEqual(_apply_turn_intent(100.0, "through"), 100.0)
        self.assertAlmostEqual(_apply_turn_intent(100.0, None), 100.0)

    def test_yaw_for_heading_relation_applies_turn_after_flip(self):
        # An oncoming left-turner: base flip to road_yaw+180, then -45 toward ego's right.
        road_yaw = 91.53
        base = _yaw_for_heading_relation(road_yaw, "opposite_direction")
        turned = _yaw_for_heading_relation(
            road_yaw, "opposite_direction", turn_intent="left"
        )
        self.assertAlmostEqual(base, road_yaw + 180.0)
        self.assertAlmostEqual(turned, road_yaw + 180.0 - TURN_INTENT_YAW_DEG)
        # The turned heading is closer to ego's right (road_yaw+90) than the pure flip.
        ego_right = road_yaw + 90.0
        self.assertLess(abs(turned - ego_right), abs(base - ego_right))

    def test_turn_intent_propagates_through_pipeline_to_yaw(self):
        scene_understanding = {
            "traffic_subjects": [
                {
                    "id": "veh_oncoming_1",
                    "category": "car",
                    "motion_state": "moving",
                    "turn_intent": "turning left",
                    "heading_relation_to_ego": "opposite_direction",
                    "lane_side_relation": "left_lane",
                    "lane_index_relation": -1,
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                }
            ],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, error = normalize_scene_understanding(scene_understanding)
        self.assertIsNone(error)
        # Free-text synonym is canonicalized and preserved on the subject.
        self.assertEqual(
            normalized["traffic_subjects"][0]["turn_intent"], "left"
        )

        anchor = {
            "road_id": 1,
            "lane_id": 1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        opposing = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
            "end": {"x": 50.0, "y": -3.5, "z": 0.0, "yaw": 180.0},
        }
        spawn_context = {"topology_sample": [anchor, opposing]}

        relation_dsl = build_relation_dsl(normalized, spawn_context)
        self.assertEqual(relation_dsl["ego_relations"][0]["turn_intent"], "left")

        raw = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        projected = project_entities_to_carla_context(raw, normalized, spawn_context)
        yaw = projected["entities"][0]["rotation"]["yaw"]
        # Opposing lane points at yaw 180; a left turn deflects it to 180-45=135,
        # i.e. toward ego's right (yaw+90) rather than straight head-on.
        self.assertAlmostEqual((yaw % 360.0), (180.0 - TURN_INTENT_YAW_DEG) % 360.0, places=3)

    def test_oncoming_vehicle_crosses_median_on_divided_road(self):
        # Regression for the divided-expressway bug: an opposite_direction vehicle
        # whose opposing carriageway is NOT present in the local topology sample
        # (e.g. a separate road across a raised median) must be pushed across the
        # median onto the opposing side, not left in ego's adjacent left lane.
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "divided_expressway",
                "directionality": "bidirectional_divided",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 3, "opposing_lane_count": 3}],
                "lane_markings": {},
                "special_road_areas": [
                    {"type": "raised_center_median", "side": "left_of_ego_carriageway"}
                ],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        # Anchor points +x (yaw 0); only the ego carriageway is known locally.
        anchor = {
            "road_id": 54,
            "lane_id": 4,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        spawn_context = {"topology_sample": [anchor]}
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "veh_6",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "center_of_lane",
                    "lane_side_relation": "left_lane",
                    "lane_index_relation": -2,
                    "heading_relation": "opposite_direction",
                    "motion_state": "moving",
                    "location": {"x": 18.0, "y": -7.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                }
            ],
        }

        projected = project_entities_to_carla_context(raw, scene_understanding, spawn_context)
        entity = projected["entities"][0]
        # Right-of-anchor is +y (yaw+90); the opposing carriageway is to the left,
        # so the seed must land well beyond ego's three forward lanes (>10.5m left).
        self.assertLess(entity["location"]["y"], -10.5)
        yaw = entity["rotation"]["yaw"]
        yaw_delta = abs(((yaw - anchor["start"]["yaw"] + 180.0) % 360.0) - 180.0)
        self.assertAlmostEqual(yaw_delta, 180.0, places=3)

        payload = build_projected_spawn_payload(projected)
        # The cross-median seed is snapped onto the opposing carriageway and then
        # walked to the median-adjacent lane at runtime. The nearest oncoming
        # vehicle (both have |index|=2 here) maps to opposing lane 1.
        self.assertEqual(
            payload["entities"][0]["placement_mode"], "project_to_opposing_lane"
        )
        self.assertEqual(payload["entities"][0]["opposing_lane_from_median"], 1)

    def test_opposing_lane_index_counts_from_median(self):
        # Two oncoming vehicles at different ego-relative lane indices map to
        # opposing lanes counted from the median: the nearer one (smaller |index|)
        # becomes lane 1 (median-adjacent), the farther one lane 2.
        def _oncoming(entity_id, lane_index):
            return {
                "id": entity_id,
                "category": "car",
                "spawn_kind": "vehicle",
                "blueprint_name": "car",
                "lane_side_relation": "left_lane",
                "lane_index_relation": lane_index,
                "heading_relation": "opposite_direction",
                "motion_state": "moving",
                "location": {"x": 0.0, "y": -12.0, "z": 0.3},
                "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                "projected_lane": {},
            }

        payload = build_projected_spawn_payload(
            {"entities": [_oncoming("near", -2), _oncoming("far", -3)]}
        )
        by_id = {e["id"]: e for e in payload["entities"]}
        self.assertEqual(by_id["near"]["placement_mode"], "project_to_opposing_lane")
        self.assertEqual(by_id["near"]["opposing_lane_from_median"], 1)
        self.assertEqual(by_id["far"]["opposing_lane_from_median"], 2)

    def test_normalize_corrects_opposite_direction_right_lane_to_left_lane(self):
        # Root-cause regression test: on a curved road the VLM can assign
        # lane_side_relation="right_lane" / lane_index_relation=1 to an oncoming
        # vehicle because the car appears visually to the right in the ego image.
        # normalize_scene_understanding must detect the heading ↔ lane_index
        # inconsistency using the road topology and correct it before the value
        # propagates to the relation DSL or spawn payload.
        payload = {
            "traffic_subjects": [
                {
                    "id": "vehicle_1",
                    "category": "car",
                    "subtype": "sedan",
                    "heading_relation_to_ego": "opposite_direction",
                    "lane_side_relation": "right_lane",
                    "lane_index_relation": 1,
                    "longitudinal_relation": "ahead",
                    "longitudinal_proximity": "mid",
                    "motion_state": "unknown",
                    "visual_confidence": "high",
                    "evidence": "approaching on right side",
                    "count": 1,
                }
            ],
            "background_traffic": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        normalized, err = normalize_scene_understanding(payload)
        self.assertIsNone(err)
        subject = normalized["traffic_subjects"][0]
        # Opposite-direction actor must be assigned to the opposing lane (left).
        self.assertLess(subject["lane_index_relation"], 0,
                        "opposite_direction actor must have negative lane_index after normalization")
        self.assertEqual(subject["lane_side_relation"], "left_lane")

    def test_left_edge_vehicle_spacing_prevents_car_motorcycle_overlap(self):
        scene_understanding = {
            "traffic_subjects": [],
            "background_traffic": [],
            "key_pairwise_relations": [],
            "road_network": {
                "road_type": "urban_straight",
                "directionality": "two_way",
                "road_segments": [{"id": "r0", "geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 1,
                    }
                ],
                "lane_markings": {},
                "special_road_areas": [],
                "junctions": [],
                "roadside_boundaries": {},
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        anchor = {
            "road_id": 1,
            "lane_id": 1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 50.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        raw = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "left_motorcycle",
                    "category": "motorcycle",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "motorcycle",
                    "lane_anchor": "left_curbside",
                    "lane_side_relation": "left_edge",
                    "heading_relation": "same_direction",
                    "location": {"x": 10.0, "y": -3.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
                {
                    "id": "left_car",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "blueprint_name": "car",
                    "lane_anchor": "left_curbside",
                    "lane_side_relation": "left_edge",
                    "heading_relation": "same_direction",
                    "location": {"x": 10.0, "y": -3.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                },
            ],
        }

        projected = project_entities_to_carla_context(
            raw,
            scene_understanding,
            {"topology_sample": [anchor]},
        )
        a, b = projected["entities"]
        distance = math.hypot(
            a["location"]["x"] - b["location"]["x"],
            a["location"]["y"] - b["location"]["y"],
        )
        self.assertGreaterEqual(distance, 5.0)

    def test_spawn_payload_uses_projected_layout_for_matched_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "enable_scene_verify": False,
                },
            )
            scene_id = "projected"
            refined = {
                "selected_anchor_lane": generator._fallback_topology_sample()[0],
                "entities": [
                    {
                        "id": "subject_car",
                        "category": "car",
                        "spawn_kind": "vehicle",
                        "blueprint_name": "car",
                        "location": {"x": 0.0, "y": 0.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                    }
                ],
            }
            match_report = {
                "status": "matched",
                "best_match": {"anchor": {"dx": 10.0, "dy": 20.0}},
                "projected_layout": {
                    "subject_car": {
                        "projected_location": {"x": 12.0, "y": 34.0, "z": 0.0},
                        "projected_rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
                    }
                },
            }
            match_path = Path(tmp) / f"{scene_id}_match.json"
            match_path.write_text(json.dumps(match_report), encoding="utf-8")
            payload = generator.build_spawn_payload_from_match_or_fallback(
                scene_id,
                refined,
                str(match_path),
            )
            subject = next(entity for entity in payload["entities"] if entity["id"] == "subject_car")
            self.assertEqual(subject["location"]["x"], 12.0)
            self.assertEqual(subject["location"]["y"], 34.0)
            self.assertEqual(subject["rotation"]["yaw"], 180.0)

    def test_low_quality_scene_match_is_rejected(self):
        matcher = SceneMapMatcher()
        reason = matcher._match_quality_failure(
            {
                "match_record": {
                    "refined_score": 0.16,
                    "layout_penalties": {"average_vehicle_snap_distance": 1.0},
                    "candidate_features": {
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                    },
                }
            },
            {"road_hints": {"straight_road": True, "near_junction": False}},
        )
        self.assertIn("below threshold", reason)

    def test_scene_verification_agent_parses_fenced_json(self):
        agent = SceneVerificationAgent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verification.json"
            path.write_text(
                "```json\n{\"passed\": true, \"score\": 0.82, "
                "\"hard_failures\": [], \"mismatches\": [], "
                "\"recommended_stage\": \"pass\", \"repair_hints\": []}\n```",
                encoding="utf-8",
            )
            report, error = agent.extract_decision_data(str(path))
            self.assertIsNone(error)
            self.assertTrue(report["passed"])
            self.assertEqual(report["recommended_stage"], "pass")
            self.assertEqual(report["repair_actions"], [])

    def test_scene_verification_agent_filters_repair_actions_by_type(self):
        agent = SceneVerificationAgent()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verification.json"
            path.write_text(
                json.dumps(
                    {
                        "passed": False,
                        "score": 0.4,
                        "hard_failures": [],
                        "mismatches": [],
                        "recommended_stage": "match_spawn",
                        "repair_hints": [],
                        "repair_actions": [
                            {"type": "count_mismatch", "severity": "high"},
                            {"type": "category_mismatch", "severity": "medium"},
                            {
                                "type": "lane_side_mismatch",
                                "entity_id": "left_car",
                                "severity": "high",
                            },
                            {
                                "type": "pairwise_mismatch",
                                "entity_id": "car_a",
                                "severity": "high",
                            },
                            {
                                "type": "pairwise_mismatch",
                                "entity_id": "car_a",
                                "reference_entity_id": "car_b",
                                "target_relation": "ahead_of",
                                "severity": "high",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report, error = agent.extract_decision_data(str(path))
            self.assertIsNone(error)
            self.assertEqual(
                [action["type"] for action in report["repair_actions"]],
                [
                    "count_mismatch",
                    "category_mismatch",
                    "lane_side_mismatch",
                    "pairwise_mismatch",
                ],
            )

    def test_scene_verification_ego_view_prompt_mentions_forward_blind_spot(self):
        prompt = SceneVerificationAgent._build_prompt(
            {
                "capture_mode": "ego_view",
                "scene_understanding": {},
                "scene_match": {},
                "spawn_entities": {},
                "user_scene_description": "",
            }
        )
        self.assertIn("generated CARLA ego-view image", prompt)
        self.assertIn("Do NOT penalize actors not visible in the forward ego-view", prompt)

    def test_scene_verification_bev_prompt_keeps_ego_frame_inference(self):
        prompt = SceneVerificationAgent._build_prompt(
            {
                "capture_mode": "bev_fallback",
                "scene_understanding": {},
                "scene_match": {},
                "spawn_entities": {},
                "user_scene_description": "",
            }
        )
        self.assertIn("generated CARLA BEV", prompt)
        self.assertIn("Use the same ego-centric frame", prompt)

    def test_validate_relation_layout_reports_metric_lateral_diagnostics(self):
        anchor_lane = {
            "start": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end": {"x": 10.0, "y": 0.0, "z": 0.0},
        }
        relation_dsl = {
            "lane_width_class": "narrow",
            "lane_context": {"left_parking_lane_count": 1, "right_parking_lane_count": 1},
            "selected_anchor_lane": anchor_lane,
            "ego_relations": [
                {
                    "entity_id": "left_car",
                    "lane_side_relation": "left_edge",
                    "lane_anchor": "left_curbside",
                    "longitudinal_position_m": 0.0,
                }
            ],
        }
        projected = {
            "selected_anchor_lane": anchor_lane,
            "entities": [
                {
                    "id": "left_car",
                    "category": "car",
                    "location": {"x": 0.0, "y": -3.2, "z": 0.0},
                }
            ],
        }
        validation = validate_relation_layout(relation_dsl, projected)
        result = validation["entity_results"][0]
        self.assertEqual(validation["lane_width_m"], 3.2)
        self.assertEqual(result["expected_lateral_band"], -2.0)
        self.assertEqual(result["expected_lateral_m"], -6.4)
        self.assertEqual(result["actual_lateral_m"], -3.2)
        self.assertEqual(result["lateral_error_m"], 3.2)

    def test_topology_signature_parses_no_junction_and_no_center_median(self):
        scene_understanding = {
            "road_network": {
                "directionality": "two_way_undivided",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 1,
                        "right_parking_lane_count": 1,
                    }
                ],
                "junctions": [{"type": "none_visible_on_current_segment", "confidence": 0.9}],
                "special_road_areas": [
                    {"type": "crosswalk", "position": "immediately_ahead"},
                    {"type": "curbside_parking_strip", "side": "right"},
                ],
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {
                "decisive_map_matching_cues": {
                    "junction_visible": False,
                    "raised_center_median_visible": False,
                    "center_island_visible": False,
                    "simple_straight_street": True,
                    "near_crosswalk": True,
                },
                "ego_path_constraints": ["no center median or center island"],
            },
        }
        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)
        self.assertEqual(signature["topology_type"], "straight_two_way")
        self.assertFalse(signature["junction_visible"])
        self.assertFalse(signature["has_center_median"])
        self.assertTrue(signature["has_crosswalk"])
        self.assertEqual(signature["driving_lane_count"], 2)
        self.assertTrue(signature["right_parking_presence"])

    def test_topology_signature_uses_explicit_parking_strip_area(self):
        scene_understanding = {
            "road_network": {
                "directionality": "two_way_undivided",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                    }
                ],
                "special_road_areas": [
                    {"type": "curbside_parking_strip", "side": "left"}
                ],
            },
            "general_environment": {},
            "metadata": {
                "decisive_map_matching_cues": {
                    "junction_visible": False,
                    "simple_straight_street": True,
                }
            },
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertFalse(signature["left_parking_presence"])
        self.assertFalse(signature["right_parking_presence"])
        self.assertTrue(signature["side_context"]["left_parking_strip"])

    def test_topology_signature_does_not_promote_text_parking_overrides(self):
        scene_understanding = {
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "left_parking_presence": False,
                    "right_parking_presence": False,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {
                "road_scene_brief": {
                    "parking_lane_evidence": (
                        "Parked cars line the right curb, but a clearly separated "
                        "parking lane boundary is not visible; left curbside "
                        "parking lane is not evident."
                    )
                }
            },
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertFalse(signature["left_parking_presence"])
        self.assertFalse(signature["right_parking_presence"])
        self.assertFalse(signature["side_context"]["left_parking_strip"])
        self.assertFalse(signature["side_context"]["right_parking_strip"])
        self.assertTrue(signature["side_context"]["right_parking_from_text"])

    def test_topology_signature_respects_explicit_no_parking_with_left_side_text(self):
        scene_understanding = {
            "actor_layout": {
                "global_anchor": "Ego vehicle is on a left-curving two-way road segment.",
            },
            "road_network": {
                "map_matching": {
                    "topology_type": "curve",
                    "curve_direction": "left",
                    "junction_visible": False,
                    "junction_type": "none",
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "left_parking_presence": False,
                    "right_parking_presence": False,
                    "has_center_median": False,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {
                "road_scene_brief": {
                    "parking_lane_evidence": (
                        "No curbside parked vehicles or marked parking strip are "
                        "visible on either side."
                    )
                },
                "vehicle_position_brief": {
                    "overall_observation": (
                        "A single car is visible ahead on the left side of the "
                        "left-curving undivided road."
                    )
                },
            },
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertFalse(signature["left_parking_presence"])
        self.assertFalse(signature["right_parking_presence"])
        self.assertFalse(signature["side_context"]["left_parking_strip"])
        self.assertFalse(signature["side_context"]["right_parking_strip"])

    def test_topology_signature_does_not_infer_parking_symmetry_from_lane_groups(self):
        scene_understanding = {
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "forward_lane_count": 1,
                    "opposing_lane_count": 1,
                    "driving_lane_count": 2,
                    "has_center_median": False,
                },
                "lane_groups": [
                    {
                        "forward_lane_count": 1,
                        "opposing_lane_count": 1,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 1,
                    }
                ],
            },
            "general_environment": {},
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertFalse(signature["left_parking_presence"])
        self.assertTrue(signature["right_parking_presence"])
        self.assertNotIn("parking_symmetry_inferred", signature)

    def test_topology_signature_detects_planted_median_and_lane_marking_crosswalk(self):
        scene_understanding = {
            "road_network": {
                "directionality": "two_way_divided",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 3, "opposing_lane_count": 2}],
                "junctions": [{"type": "signalized_intersection", "confidence": "high"}],
                "lane_markings": {"zebra_crossing_present": True},
                "special_road_areas": [{"type": "planted_center_median"}],
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertTrue(signature["has_center_median"])
        self.assertTrue(signature["has_crosswalk"])
        self.assertEqual(signature["driving_lane_count"], 5)

    def test_topology_signature_respects_absent_median_from_lane_markings(self):
        scene_understanding = {
            "road_network": {
                "directionality": "divided_by_lane_markings_not_median_visible",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 0}],
                "junctions": [
                    {
                        "type": "signalized_intersection",
                        "position": "ahead",
                        "confidence": "high",
                    },
                    {
                        "type": "side_road_connection",
                        "position": "right_ahead",
                        "confidence": "high",
                    },
                ],
                "special_road_areas": [
                    {"type": "intersection_approach", "presence": True},
                    {"type": "right_side_side_road_mouth", "presence": True},
                    {"type": "center_median", "presence": False},
                ],
                "control_elements": [{"type": "traffic_light", "position": "ahead"}],
            },
            "general_environment": {},
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertFalse(signature["has_center_median"])
        self.assertTrue(signature["junction_branches"]["right"])
        self.assertFalse(signature["junction_branches"]["left"])

    def test_topology_signature_prefers_explicit_map_matching_fields(self):
        scene_understanding = {
            "road_network": {
                "map_matching": {
                    "topology_type": "t_junction",
                    "junction_type": "t_junction",
                    "junction_visible": True,
                    "junction_branches": {
                        "ahead": True,
                        "left": False,
                        "right": True,
                        "known": True,
                    },
                    "forward_lane_count": 2,
                    "opposing_lane_count": 0,
                    "has_center_median": False,
                    "has_crosswalk": True,
                    "has_traffic_light": True,
                    "curve_direction": "right",
                    "ego_lane_from_right": 0,
                    "ego_to_junction_distance_m": 20,
                },
                # Legacy evidence intentionally contains misleading wording; the
                # direct map_matching contract should win.
                "directionality": "two_way_divided",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 4, "opposing_lane_count": 4}],
                "junctions": [{"type": "cross_intersection", "confidence": "high"}],
                "special_road_areas": [{"type": "planted_center_median"}],
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertEqual(signature["topology_type"], "t_junction")
        self.assertEqual(signature["target_branch_count"], 3)
        self.assertEqual(signature["forward_lane_count"], 2)
        self.assertEqual(signature["opposing_lane_count"], 0)
        self.assertEqual(signature["driving_lane_count"], 2)
        self.assertFalse(signature["has_center_median"])
        self.assertEqual(signature["curve_direction"], "right")
        self.assertEqual(signature["ego_lane_from_right"], 0)
        self.assertEqual(signature["ego_to_junction_distance_m"], 20.0)
        self.assertTrue(signature["junction_branches"]["right"])
        self.assertFalse(signature["junction_branches"]["left"])

    def test_topology_signature_uses_metadata_ego_localization(self):
        scene_understanding = {
            "road_network": {
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 1}],
                "junctions": [{"type": "signalized_intersection", "confidence": "high"}],
                "control_elements": [{"type": "traffic_light"}],
            },
            "general_environment": {},
            "metadata": {
                "ego_localization": {
                    "ego_lane_from_right": 0,
                    "ego_to_junction_distance_m": 10,
                }
            },
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertEqual(signature["ego_lane_from_right"], 0)
        self.assertEqual(signature["ego_to_junction_distance_m"], 10.0)

    def test_topology_signature_merges_opposing_lanes_from_secondary_group(self):
        scene_understanding = {
            "road_network": {
                "directionality": "divided_two_way",
                "road_segments": [{"geometry_type": "line"}],
                "lane_groups": [
                    {
                        "forward_lane_count": 3,
                        "opposing_lane_count": 0,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                    },
                    {
                        "forward_lane_count": 0,
                        "opposing_lane_count": 3,
                        "left_parking_lane_count": 0,
                        "right_parking_lane_count": 0,
                    },
                ],
                "junctions": [{"type": "none_clear", "confidence": "medium"}],
                "special_road_areas": [{"type": "raised_center_median"}],
                "control_elements": [{"type": "traffic_signal"}],
            },
            "general_environment": {},
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertEqual(signature["forward_lane_count"], 3)
        self.assertEqual(signature["opposing_lane_count"], 3)
        self.assertEqual(signature["driving_lane_count"], 6)
        self.assertEqual(signature["directionality"], "two_way")
        self.assertEqual(signature["topology_type"], "straight_two_way")
        self.assertTrue(signature["has_center_median"])

    def test_scene_understanding_prompt_requests_divided_road_symmetry_reasoning(self):
        prompt = SceneUnderstandingInterpreter().pre_prompt

        self.assertIn("road-symmetry reasoning", prompt)
        self.assertIn("partial evidence of an opposing roadway or opposing vehicles", prompt)
        self.assertIn("opposing travel-lane count to N", prompt)
        self.assertIn("symmetric estimate", prompt)
        self.assertIn("do not make this symmetry inference", prompt)

    def test_topology_signature_detects_latest_visible_median_wording(self):
        scene_understanding = {
            "road_network": {
                "directionality": "two-way right-hand traffic divided by planted median",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 3, "opposing_lane_count": 2}],
                "junctions": [{"type": "signalized_intersection", "confidence": "high"}],
                "lane_markings": {"zebra_crossing": "visible", "stop_line": "visible"},
                "special_road_areas": [
                    {
                        "type": "planted_median",
                        "description": "raised green median separating opposing traffic",
                    }
                ],
                "control_elements": [{"type": "traffic_signal"}],
            },
            "general_environment": {
                "non_spawnable_landmarks": ["zebra crossings", "planted central median"]
            },
            "metadata": {},
        }

        signature = SceneMapMatcher.build_road_topology_signature(scene_understanding)

        self.assertTrue(signature["has_center_median"])
        self.assertTrue(signature["has_crosswalk"])
        self.assertEqual(signature["topology_type"], "cross_intersection")
        self.assertEqual(signature["target_branch_count"], 4)

    def test_topology_signature_detects_target_environment_context(self):
        urban_scene = {
            "road_network": {
                "road_type": "urban_street",
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "special_road_areas": [],
                "control_elements": [],
                "roadside_boundaries": {
                    "left": "curb and sidewalk with buildings",
                    "right": "storefronts and sidewalk",
                },
            },
            "general_environment": {
                "urban_density": "urban_commercial_street",
                "roadside_context_left": ["building frontage", "sidewalk"],
                "roadside_context_right": ["storefronts", "shops"],
            },
            "metadata": {},
        }
        natural_scene = {
            "road_network": {
                "road_type": "mountain_road",
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "special_road_areas": [],
                "control_elements": [],
                "roadside_boundaries": {
                    "left": "rocky cliff and grass",
                    "right": "water and vegetation",
                },
            },
            "general_environment": {
                "urban_density": "rural_mountain_road",
                "non_spawnable_landmarks": ["rocks", "grass", "water"],
            },
            "metadata": {},
        }

        urban_env = SceneMapMatcher.build_road_topology_signature(urban_scene)[
            "environment_context"
        ]
        natural_env = SceneMapMatcher.build_road_topology_signature(natural_scene)[
            "environment_context"
        ]

        self.assertTrue(urban_env["expects_urban"])
        self.assertTrue(urban_env["expects_buildings"])
        self.assertTrue(urban_env["expects_sidewalks"])
        self.assertTrue(urban_env["avoid_water"])
        self.assertTrue(natural_env["expects_natural"])
        self.assertTrue(natural_env["expects_water"])
        self.assertFalse(natural_env["avoid_water"])

    def test_topology_signature_detects_t_and_cross_junctions(self):
        base = {
            "road_network": {
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 1, "opposing_lane_count": 1}],
                "special_road_areas": [],
                "control_elements": [],
            },
            "general_environment": {},
            "metadata": {},
        }
        t_scene = {
            **base,
            "road_network": {
                **base["road_network"],
                "junctions": [{"type": "t_junction", "confidence": "high"}],
            },
        }
        cross_scene = {
            **base,
            "road_network": {
                **base["road_network"],
                "junctions": [{"type": "cross_intersection", "confidence": "high"}],
            },
        }
        self.assertEqual(
            SceneMapMatcher.build_road_topology_signature(t_scene)["topology_type"],
            "t_junction",
        )
        self.assertEqual(
            SceneMapMatcher.build_road_topology_signature(cross_scene)["topology_type"],
            "cross_intersection",
        )

    def test_topology_candidate_scoring_rejects_complex_region_for_straight_road(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "straight_two_way",
                "target_branch_count": 1,
                "driving_lane_count": 2,
                "left_parking_presence": False,
                "right_parking_presence": True,
                "has_crosswalk": True,
            }
        }
        candidate = {
            "candidate_topology_type": "multi_branch",
            "estimated_junction_degree": 6,
            "same_road_lane_count": 4,
            "junction_waypoint_ratio": 0.24,
            "heading_cluster_count": 6,
            "nearby_road_count": 5,
        }
        score, details = matcher._score_candidate_features(candidate, scene_features)
        self.assertTrue(details["hard_reject"])
        self.assertLessEqual(score, 0.05)

    def test_topology_candidate_scoring_rejects_junction_for_explicit_straight_road(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "straight_road",
                "junction_visible": False,
                "target_branch_count": 1,
                "driving_lane_count": 3,
                "forward_lane_count": 3,
                "opposing_lane_count": 0,
            }
        }
        candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 4,
            "same_direction_lane_count": 4,
            "junction_waypoint_ratio": 0.51,
            "heading_cluster_count": 3,
            "nearby_road_count": 3,
            "is_junction": True,
        }

        score, details = matcher._score_candidate_features(candidate, scene_features)

        self.assertTrue(details["hard_reject"])
        self.assertIn("Straight-road target rejects", details["reject_reason"])
        self.assertLessEqual(score, 0.05)

    def test_topology_candidate_scoring_rejects_overly_wide_straight_road(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "straight_two_way",
                "target_branch_count": 1,
                "driving_lane_count": 2,
            }
        }
        candidate = {
            "candidate_topology_type": "straight_two_way",
            "estimated_junction_degree": 1,
            "same_road_lane_count": 4,
            "junction_waypoint_ratio": 0.0,
            "heading_cluster_count": 2,
            "nearby_road_count": 1,
        }
        score, details = matcher._score_candidate_features(candidate, scene_features)
        self.assertTrue(details["hard_reject"])
        self.assertIn("overly wide", details["reject_reason"])
        self.assertLessEqual(score, 0.05)

    def test_topology_candidate_scoring_prefers_parallel_same_direction_lanes(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 2,
                "forward_lane_count": 2,
                "opposing_lane_count": 0,
                "has_crosswalk": False,
                "has_traffic_light": False,
                "has_traffic_sign": False,
            }
        }
        base_candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 2,
            "junction_waypoint_ratio": 0.3,
            "heading_cluster_count": 2,
            "nearby_road_count": 3,
            "is_junction": True,
        }
        weak_candidate = {
            **base_candidate,
            "same_direction_lane_count": 1,
            "has_parallel_same_direction_lanes": False,
        }
        strong_candidate = {
            **base_candidate,
            "same_direction_lane_count": 2,
            "has_parallel_same_direction_lanes": True,
        }

        weak_score, weak_details = matcher._score_topology_candidate_features(
            weak_candidate,
            scene_features,
        )
        strong_score, strong_details = matcher._score_topology_candidate_features(
            strong_candidate,
            scene_features,
        )

        self.assertTrue(strong_details["expects_parallel_same_direction_lanes"])
        self.assertGreater(
            strong_details["directional_lane_score"],
            weak_details["directional_lane_score"],
        )
        self.assertGreater(strong_score, weak_score)

    def test_junction_target_prefers_real_junction_over_classified_false_positive(self):
        """A junction-target scene must score a genuine junction (or an approach
        lane leading into one) above a non-junction point merely *classified*
        junction-like from nearby road/heading diversity (the Town12 failure
        where a T-junction scene matched a straight road)."""
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 2,
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
            }
        }
        common = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 2,
            "heading_cluster_count": 3,
            "nearby_road_count": 3,
        }
        real_junction = {**common, "is_junction": True, "junction_waypoint_ratio": 0.22,
                         "distance_to_junction_ahead": 5.0}
        approach = {**common, "is_junction": False, "junction_waypoint_ratio": 0.10,
                    "distance_to_junction_ahead": 30.0}
        false_positive = {**common, "is_junction": False, "junction_waypoint_ratio": 0.20,
                          "distance_to_junction_ahead": None}

        s_junc, d_junc = matcher._score_topology_candidate_features(real_junction, scene_features)
        s_appr, d_appr = matcher._score_topology_candidate_features(approach, scene_features)
        s_fp, d_fp = matcher._score_topology_candidate_features(false_positive, scene_features)

        self.assertEqual(d_junc["junction_anchor_quality"], 1.0)
        self.assertEqual(d_appr["junction_anchor_quality"], 0.9)
        self.assertEqual(d_fp["junction_anchor_quality"], 0.30)
        # Both a real junction and an approach lane must beat the false positive.
        self.assertGreater(s_junc, s_fp)
        self.assertGreater(s_appr, s_fp)

    def test_detect_junction_ahead_reports_distance_along_lane(self):
        class FakeWaypoint:
            def __init__(self, is_junction=False):
                self.is_junction = is_junction
                self._next = None

            def next(self, _distance):
                return [self._next] if self._next else []

        # Chain: center -> wp1 -> wp2 -> wp3 -> junction, 5m per step,
        # so the junction is reached after 4 steps == 20m.
        waypoints = [FakeWaypoint() for _ in range(4)]
        junction_wp = FakeWaypoint(is_junction=True)
        chain = waypoints + [junction_wp]
        for current, following in zip(chain, chain[1:]):
            current._next = following

        distance = SceneMapMatcher._detect_junction_ahead(
            waypoints[0], max_distance=80.0, step=5.0
        )
        self.assertEqual(distance, 20.0)

    def test_detect_junction_ahead_returns_none_for_clear_road(self):
        class FakeWaypoint:
            def __init__(self):
                self.is_junction = False
                self._next = None

            def next(self, _distance):
                return [self._next] if self._next else []

        center = FakeWaypoint()
        follower = FakeWaypoint()
        center._next = follower  # lane ends after one step, no junction reached

        self.assertIsNone(
            SceneMapMatcher._detect_junction_ahead(center, max_distance=80.0, step=5.0)
        )

    def test_target_junction_branches_right_stem_t_junction(self):
        scene = {
            "road_network": {
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 2}],
                "control_elements": [
                    {"type": "overhead_traffic_signal", "confidence": "high"}
                ],
                "junctions": [{"type": "t_junction", "confidence": "high"}],
                "special_road_areas": [
                    {
                        "type": "right_side_access",
                        "location": "near-right corner connecting side street to main road",
                        "confidence": "medium",
                    }
                ],
            },
            "general_environment": {"urban_density": "suburban_commercial"},
            "metadata": {},
        }
        signature = SceneMapMatcher.build_road_topology_signature(scene)
        branches = signature["junction_branches"]
        self.assertTrue(branches["known"])
        self.assertTrue(branches["ahead"])
        self.assertTrue(branches["right"])
        self.assertFalse(branches["left"])

    def test_signalized_three_way_not_promoted_to_cross(self):
        # A signalized junction the VLM (told via user text) reports as a
        # three-way with a right branch and an explicit "no left turn" must stay
        # a T-junction (ahead+right), not be promoted to a 4-way cross.
        scene = {
            "road_network": {
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 2}],
                "lane_markings": {"stop_line": "visible", "zebra_crossing": "visible"},
                "control_elements": [{"type": "traffic_signal", "confidence": "high"}],
                "junctions": [
                    {
                        "type": "signalized_intersection",
                        "confidence": "high",
                        "evidence": "three-way junction; right turn into a side "
                        "road is possible, no left turn",
                    }
                ],
                "special_road_areas": [],
            },
            "general_environment": {"urban_density": "suburban_commercial"},
            "metadata": {},
        }
        signature = SceneMapMatcher.build_road_topology_signature(scene)
        self.assertEqual(signature["topology_type"], "t_junction")
        self.assertEqual(signature["target_branch_count"], 3)
        branches = signature["junction_branches"]
        self.assertTrue(branches["ahead"])
        self.assertTrue(branches["right"])
        self.assertFalse(branches["left"])

    def test_classify_junction_branches_from_lane_connectivity(self):
        class FakeWaypoint:
            _counter = 0

            def __init__(self, yaw=0.0, is_junction=False):
                FakeWaypoint._counter += 1
                self.is_junction = is_junction
                self.road_id = FakeWaypoint._counter
                self.lane_id = -1
                self._next = []

                class _Rot:
                    pass

                class _Loc:
                    pass

                rot = _Rot()
                rot.yaw = yaw
                loc = _Loc()
                loc.x = float(FakeWaypoint._counter)
                loc.y = float(FakeWaypoint._counter)
                loc.z = 0.0

                class _Tf:
                    pass

                tf = _Tf()
                tf.rotation = rot
                tf.location = loc
                self.transform = tf

            def next(self, _distance):
                return list(self._next)

        approach = FakeWaypoint(yaw=0.0)
        j_in = FakeWaypoint(yaw=0.0, is_junction=True)
        ex_ahead = FakeWaypoint(yaw=0.0)
        j_right = FakeWaypoint(yaw=45.0, is_junction=True)
        ex_right = FakeWaypoint(yaw=90.0)
        approach._next = [j_in]
        j_in._next = [ex_ahead, j_right]
        j_right._next = [ex_right]

        matcher = SceneMapMatcher()
        dirs = matcher._classify_junction_branches(approach, max_distance=80.0, step=5.0)
        self.assertIsNotNone(dirs)
        self.assertTrue(dirs["ahead"])
        self.assertTrue(dirs["right"])
        self.assertFalse(dirs["left"])
        self.assertEqual(dirs["branch_count"], 2)

    def test_physical_junction_arms_use_whole_junction_legs(self):
        class FakeWaypoint:
            def __init__(self, yaw=0.0, is_junction=False, junction=None):
                self.is_junction = is_junction
                self.road_id = 1
                self.lane_id = -1
                self._junction = junction
                self._next = []
                self.transform = types.SimpleNamespace(
                    rotation=types.SimpleNamespace(yaw=yaw),
                    location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )

            def next(self, _distance):
                return list(self._next)

            def previous(self, _distance):
                return []

            def get_junction(self):
                return self._junction

        class FakeJunction:
            id = 7
            bounding_box = types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                extent=types.SimpleNamespace(x=12.0, y=12.0),
            )

            def get_waypoints(self, _lane_type):
                def wp(yaw):
                    return FakeWaypoint(yaw=yaw)

                return [
                    (wp(180.0), wp(0.0)),
                    (wp(-90.0), wp(90.0)),
                    (wp(0.0), wp(180.0)),
                    (wp(90.0), wp(-90.0)),
                ]

        junction = FakeJunction()
        junction_wp = FakeWaypoint(yaw=0.0, is_junction=True, junction=junction)
        approach = FakeWaypoint(yaw=0.0)
        approach._next = [junction_wp]

        arms = SceneMapMatcher._physical_junction_arms_for_waypoint(approach)

        self.assertTrue(arms["known"])
        self.assertTrue(arms["ahead"])
        self.assertTrue(arms["left"])
        self.assertTrue(arms["right"])
        self.assertEqual(arms["arm_count"], 4)

    def test_branch_direction_scoring_penalizes_wrong_side(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 4,
                "forward_lane_count": 2,
                "opposing_lane_count": 2,
                "junction_branches": {
                    "ahead": True,
                    "left": False,
                    "right": True,
                    "known": True,
                },
            }
        }
        base_candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 4,
            "junction_waypoint_ratio": 0.3,
            "heading_cluster_count": 3,
            "nearby_road_count": 3,
            "is_junction": True,
        }
        right_candidate = {
            **base_candidate,
            "physical_junction_arms": {
                "ahead": True,
                "left": False,
                "right": True,
                "known": True,
                "branch_count": 2,
            },
        }
        left_candidate = {
            **base_candidate,
            "physical_junction_arms": {
                "ahead": True,
                "left": True,
                "right": False,
                "known": True,
                "branch_count": 2,
            },
        }

        right_score, right_details = matcher._score_topology_candidate_features(
            right_candidate, scene_features
        )
        left_score, left_details = matcher._score_topology_candidate_features(
            left_candidate, scene_features
        )

        self.assertEqual(right_details["branch_direction_score"], 1.0)
        self.assertLess(left_details["branch_direction_score"], 1.0)
        self.assertEqual(
            left_details["branch_direction_detail"]["severity"],
            "mirror_branch_direction",
        )
        self.assertEqual(left_details["branch_alignment_factor"], 0.15)
        self.assertTrue(left_details["hard_reject"])
        self.assertEqual(
            left_details["reject_reason"],
            "Branch direction mirror mismatch.",
        )
        self.assertGreater(right_score, left_score)

    def test_branch_direction_missing_candidate_data_penalizes_without_rejecting(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 4,
                "forward_lane_count": 2,
                "opposing_lane_count": 2,
                "junction_branches": {
                    "ahead": True,
                    "left": False,
                    "right": True,
                    "known": True,
                },
            }
        }
        candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 4,
            "junction_waypoint_ratio": 0.3,
            "heading_cluster_count": 3,
            "nearby_road_count": 3,
            "is_junction": True,
        }

        _score, details = matcher._score_topology_candidate_features(
            candidate, scene_features
        )

        self.assertEqual(
            details["branch_direction_detail"]["severity"],
            "candidate_missing",
        )
        self.assertEqual(details["branch_alignment_factor"], 0.55)
        self.assertFalse(details["hard_reject"])

    def test_branch_direction_scoring_penalizes_missing_ahead(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 3,
                "forward_lane_count": 2,
                "opposing_lane_count": 1,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
            }
        }
        candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 5,
            "same_road_lane_count": 2,
            "junction_waypoint_ratio": 0.25,
            "heading_cluster_count": 3,
            "nearby_road_count": 5,
            "is_junction": False,
            "physical_junction_arms": {
                "ahead": False,
                "left": True,
                "right": True,
                "known": True,
                "branch_count": 2,
            },
        }

        _score, details = matcher._score_topology_candidate_features(
            candidate, scene_features
        )

        self.assertEqual(
            details["branch_direction_detail"]["severity"],
            "missing_required_branch",
        )
        self.assertIn(
            "missing_required_ahead_branch",
            details["branch_direction_detail"]["errors"],
        )
        self.assertEqual(details["branch_alignment_factor"], 0.35)

    def test_environment_context_handles_snake_case_enums(self):
        scene = {
            "road_network": {
                "road_type": "urban_arterial",
                "directionality": "two_way",
                "road_segments": [{"geometry_type": "straight"}],
                "lane_groups": [{"forward_lane_count": 2, "opposing_lane_count": 2}],
                "special_road_areas": [],
                "control_elements": [],
            },
            "general_environment": {
                "urban_density": "suburban_commercial",
                "non_spawnable_landmarks": [
                    {"type": "business_signage"},
                    {"type": "utility_poles_and_wires"},
                ],
            },
            "metadata": {},
        }
        env = SceneMapMatcher.build_road_topology_signature(scene)["environment_context"]
        self.assertTrue(env["expects_urban"])
        self.assertTrue(env["expects_buildings"])

    def test_topology_scoring_penalizes_junction_ahead_for_straight_road(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "straight_two_way",
                "junction_visible": False,
                "target_branch_count": 1,
                "driving_lane_count": 2,
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
            }
        }
        base_candidate = {
            "candidate_topology_type": "straight_two_way",
            "estimated_junction_degree": 1,
            "same_road_lane_count": 2,
            "junction_waypoint_ratio": 0.0,
            "heading_cluster_count": 2,
            "nearby_road_count": 1,
        }
        clear_candidate = {**base_candidate, "distance_to_junction_ahead": None}
        junction_ahead_candidate = {
            **base_candidate,
            "distance_to_junction_ahead": 25.0,
        }

        clear_score, clear_details = matcher._score_topology_candidate_features(
            clear_candidate, scene_features
        )
        near_score, near_details = matcher._score_topology_candidate_features(
            junction_ahead_candidate, scene_features
        )

        self.assertEqual(clear_details["junction_ahead_penalty"], 1.0)
        self.assertLess(near_details["junction_ahead_penalty"], 1.0)
        self.assertEqual(near_details["distance_to_junction_ahead"], 25.0)
        self.assertLess(near_score, clear_score)

    def test_auxiliary_scoring_penalizes_unexpected_crosswalk(self):
        signature = {
            "topology_type": "straight_two_way",
            "has_crosswalk": False,
        }
        without_crosswalk = {"has_crosswalk_nearby": False}
        with_crosswalk = {"has_crosswalk_nearby": True}

        self.assertLess(
            SceneMapMatcher._score_auxiliary_context(with_crosswalk, signature),
            SceneMapMatcher._score_auxiliary_context(without_crosswalk, signature),
        )

    def test_auxiliary_scoring_penalizes_unexpected_traffic_light(self):
        signature = {
            "topology_type": "straight_two_way",
            "has_traffic_light": False,
        }
        far_traffic_light = {
            "environment_context": {"nearest_m": {"TrafficLight": 200.0}}
        }
        near_traffic_light = {
            "environment_context": {"nearest_m": {"TrafficLight": 15.0}}
        }

        self.assertLess(
            SceneMapMatcher._score_auxiliary_context(near_traffic_light, signature),
            SceneMapMatcher._score_auxiliary_context(far_traffic_light, signature),
        )

    def test_side_context_scoring_prefers_urban_environment_for_urban_target(self):
        urban_signature = {
            "right_parking_presence": False,
            "left_parking_presence": False,
            "side_context": {
                "left_continuous_buildings": True,
                "right_continuous_buildings": True,
                "tree_lined": True,
            },
            "environment_context": {
                "expects_urban": True,
                "expects_buildings": True,
                "expects_sidewalks": True,
                "expects_natural": False,
                "avoid_water": True,
                "avoid_terrain_dominant": True,
            },
        }
        urban_candidate = {
            "environment_context": {
                "counts": {"Buildings": 10, "Sidewalks": 4, "TrafficSigns": 1},
                "urban_score": 0.9,
                "natural_score": 0.1,
                "buildings_nearby": True,
                "sidewalks_nearby": True,
                "traffic_control_nearby": True,
                "water_nearby": False,
                "environment_class": "urban_like",
            }
        }
        natural_candidate = {
            "environment_context": {
                "counts": {"Terrain": 8, "Vegetation": 12, "Water": 1},
                "urban_score": 0.1,
                "natural_score": 0.95,
                "buildings_nearby": False,
                "sidewalks_nearby": False,
                "traffic_control_nearby": False,
                "water_nearby": True,
                "environment_class": "natural_like",
            }
        }

        self.assertGreater(
            SceneMapMatcher._score_side_context(urban_candidate, urban_signature),
            SceneMapMatcher._score_side_context(natural_candidate, urban_signature),
        )

    def test_side_context_scoring_keeps_old_cache_neutral_and_allows_natural_target(self):
        matcher = SceneMapMatcher()
        natural_signature = {
            "side_context": {},
            "environment_context": {
                "expects_urban": False,
                "expects_buildings": False,
                "expects_sidewalks": False,
                "expects_natural": True,
                "expects_water": False,
                "avoid_water": False,
                "avoid_terrain_dominant": False,
            },
        }
        natural_candidate = {
            "environment_context": {
                "counts": {"Terrain": 6, "Vegetation": 6},
                "urban_score": 0.1,
                "natural_score": 0.8,
                "environment_class": "natural_like",
            }
        }
        old_cache_candidate = {}

        self.assertGreater(
            matcher._score_side_context(natural_candidate, natural_signature),
            matcher._score_side_context(old_cache_candidate, natural_signature),
        )
        self.assertEqual(
            matcher._score_side_context(old_cache_candidate, natural_signature),
            0.5,
        )

    def test_cache_environment_helpers_skip_missing_labels_and_summarize_context(self):
        class FakeWorld:
            def get_environment_objects(self, label):
                if label == "Buildings":
                    return [
                        types.SimpleNamespace(
                            transform=types.SimpleNamespace(
                                location=types.SimpleNamespace(x=0.0, y=10.0, z=0.0)
                            )
                        )
                    ]
                if label == "Water":
                    return [
                        types.SimpleNamespace(
                            transform=types.SimpleNamespace(
                                location=types.SimpleNamespace(x=200.0, y=0.0, z=0.0)
                            )
                        )
                    ]
                raise AssertionError(f"unexpected label {label}")

        fake_carla = types.SimpleNamespace(
            CityObjectLabel=types.SimpleNamespace(Buildings="Buildings", Water="Water")
        )

        points, loaded_counts = cache_map_topology._load_environment_points(
            FakeWorld(),
            fake_carla,
        )
        context = cache_map_topology._summarize_environment_context(
            types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            points,
            60.0,
        )

        self.assertEqual(loaded_counts, {"Buildings": 1, "Water": 1})
        self.assertEqual(context["counts"], {"Buildings": 1})
        self.assertTrue(context["buildings_nearby"])
        self.assertFalse(context["water_nearby"])
        self.assertEqual(context["environment_class"], "urban_like")

    def test_cache_crosswalk_helper_accepts_flat_and_nested_locations(self):
        loc_a = types.SimpleNamespace(x=1.0, y=2.0)
        loc_b = types.SimpleNamespace(x=3.0, y=4.0)
        loc_c = types.SimpleNamespace(x=5.0, y=6.0)

        self.assertEqual(
            cache_map_topology._flatten_crosswalk_locations([loc_a, [loc_b, loc_c]]),
            [loc_a, loc_b, loc_c],
        )

    def test_curve_features_detect_waypoint_bend_from_neighbor_yaw(self):
        class FakeWaypoint:
            def __init__(self, yaw):
                self.transform = types.SimpleNamespace(
                    rotation=types.SimpleNamespace(yaw=yaw)
                )
                self._previous = None
                self._next = None

            def previous(self, _distance):
                return [self._previous] if self._previous else []

            def next(self, _distance):
                return [self._next] if self._next else []

        previous_wp = FakeWaypoint(-12.0)
        center_wp = FakeWaypoint(0.0)
        next_wp = FakeWaypoint(12.0)
        center_wp._previous = previous_wp
        center_wp._next = next_wp

        features = SceneMapMatcher._extract_curve_features(center_wp)

        self.assertTrue(features["is_curve"])
        self.assertEqual(features["curve_direction"], "right")
        self.assertAlmostEqual(features["curve_yaw_delta_deg"], 24.0)

    def test_same_road_lane_count_ignores_non_driving_lanes(self):
        waypoint_type = types.SimpleNamespace
        waypoints = [
            waypoint_type(road_id=1, lane_id=-1, lane_type="Driving"),
            waypoint_type(road_id=1, lane_id=1, lane_type="Driving"),
            waypoint_type(road_id=1, lane_id=2, lane_type="Parking"),
            waypoint_type(road_id=1, lane_id=0, lane_type="Driving"),
            waypoint_type(road_id=2, lane_id=-1, lane_type="Driving"),
        ]
        self.assertEqual(SceneMapMatcher._estimate_same_road_lane_count(waypoints, 1), 2)

    def test_same_direction_lane_features_require_adjacent_same_road_and_yaw(self):
        class FakeWaypoint:
            def __init__(self, lane_id, yaw=0.0, lane_type="Driving", road_id=1):
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.road_id = road_id
                self.transform = types.SimpleNamespace(
                    rotation=types.SimpleNamespace(yaw=yaw)
                )
                self._left = None
                self._right = None

            def get_left_lane(self):
                return self._left

            def get_right_lane(self):
                return self._right

        anchor = FakeWaypoint(1, yaw=0.0)
        adjacent_same_direction = FakeWaypoint(2, yaw=4.0)
        opposite_direction = FakeWaypoint(3, yaw=180.0)
        other_road = FakeWaypoint(4, yaw=0.0, road_id=2)
        anchor._right = adjacent_same_direction
        adjacent_same_direction._right = opposite_direction
        anchor._left = other_road

        features = SceneMapMatcher._estimate_same_direction_lane_features(anchor)

        self.assertEqual(features["same_direction_lane_count"], 2)
        self.assertTrue(features["has_parallel_same_direction_lanes"])
        accepted = [
            lane
            for lane in features["same_direction_lane_evidence"]["inspected_lanes"]
            if lane["accepted"]
        ]
        self.assertEqual([lane["lane_id"] for lane in accepted], [1, 2])

    def test_parking_side_detection_scans_past_opposing_driving_lane(self):
        class FakeWaypoint:
            def __init__(self, lane_id, lane_type="Driving", road_id=76, yaw=0.0):
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.road_id = road_id
                self.transform = types.SimpleNamespace(
                    rotation=types.SimpleNamespace(yaw=yaw)
                )
                self._left = None
                self._right = None

            def get_left_lane(self):
                return self._left

            def get_right_lane(self):
                return self._right

        anchor = FakeWaypoint(-1, yaw=0.0)
        opposing_driving = FakeWaypoint(1, yaw=180.0)
        left_parking = FakeWaypoint(2, "Parking", yaw=180.0)
        right_parking = FakeWaypoint(-2, "Parking")
        anchor._left = opposing_driving
        opposing_driving._right = left_parking
        anchor._right = right_parking

        left, right = SceneMapMatcher._detect_parking_sides(anchor)

        self.assertTrue(left)
        self.assertTrue(right)

    def test_center_median_candidate_detects_separator_on_center_side(self):
        class FakeWaypoint:
            def __init__(self, lane_id, lane_type="Driving", road_id=1):
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.road_id = road_id
                self._left = None
                self._right = None

            def get_left_lane(self):
                return self._left

            def get_right_lane(self):
                return self._right

        lane3 = FakeWaypoint(-3)
        lane2 = FakeWaypoint(-2)
        lane1 = FakeWaypoint(-1)
        median = FakeWaypoint(0, "Sidewalk")
        opposing = FakeWaypoint(1, "Driving")
        lane3._left = lane2
        lane2._left = lane1
        lane1._left = median
        median._left = opposing

        detected, evidence = SceneMapMatcher._detect_center_median_candidate(lane3)

        self.assertTrue(detected)
        self.assertEqual(evidence["side"], "left")
        self.assertTrue(evidence["opposite_driving_lane_found"])
        self.assertIn("Sidewalk", evidence["separator_lane_types"])

    def test_center_median_candidate_ignores_outer_sidewalk(self):
        class FakeWaypoint:
            def __init__(self, lane_id, lane_type="Driving", road_id=1):
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.road_id = road_id
                self._left = None
                self._right = None

            def get_left_lane(self):
                return self._left

            def get_right_lane(self):
                return self._right

        lane = FakeWaypoint(-1)
        outer_sidewalk = FakeWaypoint(-2, "Sidewalk")
        lane._right = outer_sidewalk

        detected, _evidence = SceneMapMatcher._detect_center_median_candidate(lane)

        self.assertFalse(detected)

    def test_center_median_candidate_rejects_inner_shoulder_without_opposing_lane(self):
        class FakeWaypoint:
            def __init__(self, lane_id, lane_type="Driving", road_id=1):
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.road_id = road_id
                self._left = None
                self._right = None

            def get_left_lane(self):
                return self._left

            def get_right_lane(self):
                return self._right

        # One-directional highway: driving lanes then shoulder lanes at the edge,
        # with no opposing driving lane beyond — this is a road edge, not a median.
        lane4 = FakeWaypoint(-4)
        lane3 = FakeWaypoint(-3)
        shoulder_inner = FakeWaypoint(-2, "Shoulder")
        shoulder_outer = FakeWaypoint(-1, "Shoulder")
        lane4._left = lane3
        lane3._left = shoulder_inner
        shoulder_inner._left = shoulder_outer

        detected, _evidence = SceneMapMatcher._detect_center_median_candidate(lane4)

        self.assertFalse(detected)

    def test_true_center_median_interpretation_from_evidence(self):
        shoulder_edge = {
            "has_center_median_candidate": True,
            "center_median_evidence": {
                "opposite_driving_lane_found": False,
                "separator_lane_types": ["Shoulder", "Shoulder"],
            },
        }
        confirmed_opposing = {
            "has_center_median_candidate": True,
            "center_median_evidence": {
                "opposite_driving_lane_found": True,
                "separator_lane_types": ["Sidewalk"],
            },
        }
        explicit_median = {
            "has_center_median_candidate": True,
            "center_median_evidence": {
                "opposite_driving_lane_found": False,
                "separator_lane_types": ["Median"],
            },
        }
        legacy_no_evidence = {"has_center_median_candidate": True}
        no_median = {"has_center_median_candidate": False}

        self.assertFalse(
            SceneMapMatcher._candidate_has_true_center_median(shoulder_edge)
        )
        self.assertTrue(
            SceneMapMatcher._candidate_has_true_center_median(confirmed_opposing)
        )
        self.assertTrue(
            SceneMapMatcher._candidate_has_true_center_median(explicit_median)
        )
        self.assertTrue(
            SceneMapMatcher._candidate_has_true_center_median(legacy_no_evidence)
        )
        self.assertFalse(
            SceneMapMatcher._candidate_has_true_center_median(no_median)
        )

    def test_topology_scoring_penalizes_false_shoulder_median(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "straight_two_way",
                "junction_visible": False,
                "target_branch_count": 1,
                "driving_lane_count": 5,
                "forward_lane_count": 3,
                "opposing_lane_count": 2,
                "has_center_median": True,
            }
        }
        base_candidate = {
            "candidate_topology_type": "straight_two_way",
            "estimated_junction_degree": 1,
            "same_road_lane_count": 5,
            "junction_waypoint_ratio": 0.0,
            "heading_cluster_count": 1,
            "nearby_road_count": 1,
            "has_center_median_candidate": True,
        }
        true_median_candidate = {
            **base_candidate,
            "center_median_evidence": {
                "opposite_driving_lane_found": True,
                "separator_lane_types": ["Median"],
            },
        }
        false_median_candidate = {
            **base_candidate,
            "center_median_evidence": {
                "opposite_driving_lane_found": False,
                "separator_lane_types": ["Shoulder", "Shoulder"],
            },
        }

        true_score, true_details = matcher._score_topology_candidate_features(
            true_median_candidate, scene_features
        )
        false_score, false_details = matcher._score_topology_candidate_features(
            false_median_candidate, scene_features
        )

        self.assertEqual(true_details["median_alignment_adjustment"], "boost")
        self.assertEqual(false_details["median_alignment_adjustment"], "strong_penalty")
        self.assertLess(false_score, true_score)

    def test_center_median_scoring_prefers_candidate_with_cached_median(self):
        signature = {
            "topology_type": "cross_intersection",
            "has_center_median": True,
            "has_crosswalk": False,
            "has_traffic_light": True,
            "has_traffic_sign": True,
        }
        with_median = {
            "is_junction": True,
            "junction_waypoint_ratio": 0.2,
            "nearby_lane_count": 8,
            "has_center_median_candidate": True,
        }
        without_median = {
            **with_median,
            "has_center_median_candidate": False,
        }

        self.assertGreater(
            SceneMapMatcher._score_auxiliary_context(with_median, signature),
            SceneMapMatcher._score_auxiliary_context(without_median, signature),
        )

    def test_center_median_scoring_strongly_penalizes_missing_candidate_median(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "driving_lane_count": 2,
                "forward_lane_count": 2,
                "opposing_lane_count": 0,
                "has_center_median": True,
                "has_crosswalk": False,
                "has_traffic_light": False,
                "has_traffic_sign": False,
            }
        }
        base_candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 2,
            "same_direction_lane_count": 2,
            "junction_waypoint_ratio": 0.4,
            "heading_cluster_count": 2,
            "nearby_road_count": 3,
            "is_junction": True,
        }
        with_median_score, _ = matcher._score_topology_candidate_features(
            {**base_candidate, "has_center_median_candidate": True},
            scene_features,
        )
        without_median_score, details = matcher._score_topology_candidate_features(
            {**base_candidate, "has_center_median_candidate": False},
            scene_features,
        )

        self.assertEqual(details["median_alignment_adjustment"], "strong_penalty")
        self.assertGreater(with_median_score - without_median_score, 0.2)

    def test_topology_match_returns_unmatched_when_all_candidates_rejected(self):
        matcher = SceneMapMatcher()

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def set_timeout(self, timeout):
                pass

            def get_world(self):
                return types.SimpleNamespace(
                    get_map=lambda: types.SimpleNamespace(
                        name="FakeTown",
                        get_spawn_points=lambda: [],
                    )
                )

        previous_carla = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(Client=FakeClient)
        matcher._score_carla_candidates = lambda *args, **kwargs: [
            {
                "score": 0.05,
                "score_details": {
                    "hard_reject": True,
                    "uncapped_total_score": 0.2,
                },
                "hard_reject": True,
                "reject_reason": "bad topology",
                "search_strategy": "mock",
                "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
                "candidate_lane": {"road_id": 1},
            },
            {
                "score": 0.05,
                "score_details": {
                    "hard_reject": True,
                    "uncapped_total_score": 0.6,
                },
                "hard_reject": True,
                "reject_reason": "less bad topology",
                "search_strategy": "mock",
                "location": {"x": 2.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
                "candidate_lane": {"road_id": 2},
            },
        ]
        try:
            result = matcher._match_to_carla(
                {"road_topology_signature": {"topology_type": "straight_two_way"}},
                topology_only=True,
            )
        finally:
            if previous_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = previous_carla
        self.assertEqual(result["status"], "unmatched")
        self.assertIsNone(result["best_match"])
        self.assertEqual(
            result["candidate_debug"]["best_rejected_candidate"]["candidate_lane"]["road_id"],
            2,
        )

    def test_topology_fallback_prefers_non_blacklisted_lane_match(self):
        matcher = SceneMapMatcher()

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def set_timeout(self, timeout):
                pass

            def get_world(self):
                return types.SimpleNamespace(
                    get_map=lambda: types.SimpleNamespace(
                        name="FakeTown",
                        get_spawn_points=lambda: [],
                    )
                )

        previous_carla = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(Client=FakeClient)
        matcher._score_carla_candidates = lambda *args, **kwargs: [
            {
                "score": 0.05,
                "score_details": {
                    "hard_reject": True,
                    "blacklisted": True,
                    "uncapped_total_score": 0.9,
                    "fallback_total_score": 0.9,
                },
                "hard_reject": True,
                "reject_reason": "blacklisted",
                "search_strategy": "mock",
                "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
                "candidate_lane": {"road_id": 1},
            },
            {
                "score": 0.05,
                "score_details": {
                    "hard_reject": True,
                    "blacklisted": False,
                    "uncapped_total_score": 0.5,
                    "fallback_total_score": 0.5,
                },
                "hard_reject": True,
                "reject_reason": "topology mismatch",
                "search_strategy": "mock",
                "location": {"x": 2.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
                "candidate_lane": {"road_id": 2},
            },
        ]
        try:
            result = matcher._match_to_carla(
                {"road_topology_signature": {"topology_type": "straight_two_way"}},
                topology_only=True,
            )
        finally:
            if previous_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = previous_carla
        self.assertEqual(result["status"], "unmatched")
        self.assertIsNone(result["best_match"])
        self.assertEqual(
            result["candidate_debug"]["best_rejected_candidate"]["candidate_lane"]["road_id"],
            2,
        )

    def test_v2_cache_open_road_rejects_junction_and_selects_open_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = {
                "world_name": "FakeTown",
                "candidates": [
                    {
                        "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 10, "lane_id": -1},
                        "candidate_topology_type": "t_junction",
                        "is_junction": True,
                        "junction_waypoint_ratio": 0.5,
                        "heading_cluster_count": 3,
                        "nearby_road_count": 3,
                        "same_road_lane_count": 3,
                        "same_direction_lane_count": 3,
                    },
                    {
                        "location": {"x": 2.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 20, "lane_id": -1},
                        "candidate_topology_type": "straight_two_way",
                        "is_junction": False,
                        "junction_waypoint_ratio": 0.0,
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                        "same_road_lane_count": 3,
                        "same_direction_lane_count": 3,
                        "is_curve": False,
                        "has_center_median_candidate": False,
                    },
                ],
            }
            (Path(tmp) / "FakeTown.json").write_text(json.dumps(cache), encoding="utf-8")
            matcher = SceneMapMatcher(topology_cache_dir=tmp)

            result = matcher._match_from_cache(
                {
                    "road_topology_signature": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                        "target_branch_count": 1,
                        "forward_lane_count": 3,
                        "opposing_lane_count": 0,
                        "driving_lane_count": 3,
                        "has_center_median": False,
                    }
                }
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["best_match"]["search_strategy"], "cache_v2")
        self.assertEqual(result["best_match"]["candidate_lane"]["road_id"], 20)
        self.assertFalse(result["best_match"]["candidate_features"]["is_junction"])
        self.assertLess(
            result["best_match"]["candidate_features"]["junction_waypoint_ratio"],
            0.18,
        )

    def test_v2_cache_open_road_rejects_parking_lane_when_target_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = {
                "world_name": "FakeTown",
                "candidates": [
                    {
                        "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 10, "lane_id": -1},
                        "candidate_topology_type": "straight_two_way",
                        "is_junction": False,
                        "junction_waypoint_ratio": 0.0,
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                        "same_road_lane_count": 2,
                        "same_direction_lane_count": 2,
                        "is_curve": False,
                        "has_center_median_candidate": False,
                        "right_parking_lane_present": True,
                    },
                    {
                        "location": {"x": 2.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 20, "lane_id": -1},
                        "candidate_topology_type": "straight_two_way",
                        "is_junction": False,
                        "junction_waypoint_ratio": 0.0,
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                        "same_road_lane_count": 2,
                        "same_direction_lane_count": 2,
                        "is_curve": False,
                        "has_center_median_candidate": False,
                        "left_parking_lane_present": False,
                        "right_parking_lane_present": False,
                    },
                ],
            }
            (Path(tmp) / "FakeTown.json").write_text(json.dumps(cache), encoding="utf-8")
            matcher = SceneMapMatcher(topology_cache_dir=tmp)

            result = matcher._match_from_cache(
                {
                    "road_topology_signature": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                        "forward_lane_count": 2,
                        "driving_lane_count": 2,
                        "left_parking_presence": False,
                        "right_parking_presence": False,
                        "has_center_median": False,
                    }
                }
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["best_match"]["candidate_lane"]["road_id"], 20)
        self.assertEqual(result["best_match"]["world"], "FakeTown")
        self.assertFalse(
            result["best_match"]["candidate_features"]["right_parking_lane_present"]
        )
        self.assertEqual(
            result["candidate_summary"]["accepted"][0]["world"],
            "FakeTown",
        )
        self.assertEqual(
            result["candidate_summary"]["accepted"][0]["same_direction_lane_count"],
            2,
        )
        self.assertEqual(
            result["candidate_summary"]["rejected"][0]["world"],
            "FakeTown",
        )
        rejected_reasons = (
            result["candidate_debug"]["best_rejected_candidate"]["score_details"][
                "reject_reasons"
            ]
        )
        self.assertIn(
            "open-road target rejects candidate with right parking lane",
            rejected_reasons,
        )

    def test_v2_open_road_lane_mismatch_is_hard_reject(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "straight_road",
                "junction_visible": False,
                "forward_lane_count": 1,
                "driving_lane_count": 2,
                "has_center_median": False,
            },
            {
                "candidate_topology_type": "straight_two_way",
                "is_junction": False,
                "junction_waypoint_ratio": 0.0,
                "heading_cluster_count": 2,
                "nearby_road_count": 1,
                "same_direction_lane_count": 2,
                "same_road_lane_count": 4,
                "has_center_median_candidate": False,
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertTrue(
            any("same-direction lane mismatch" in reason for reason in details["reject_reasons"])
        )
        self.assertTrue(
            any("driving-lane mismatch" in reason for reason in details["reject_reasons"])
        )

    def test_v2_curve_open_road_overwide_lane_mismatch_is_hard_reject(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "curve",
                "curve_direction": "left",
                "junction_visible": False,
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
                "driving_lane_count": 2,
                "has_center_median": False,
            },
            {
                "candidate_topology_type": "curve",
                "is_curve": True,
                "curve_score": 0.8,
                "curve_direction": "left",
                "is_junction": False,
                "junction_waypoint_ratio": 0.0,
                "heading_cluster_count": 2,
                "nearby_road_count": 1,
                "same_direction_lane_count": 2,
                "same_road_lane_count": 4,
                "has_center_median_candidate": False,
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertTrue(
            any("same-direction lane mismatch" in reason for reason in details["reject_reasons"])
        )
        self.assertTrue(
            any("overly wide driving-lane mismatch" in reason for reason in details["reject_reasons"])
        )

    def test_v2_open_road_center_median_mismatch_is_hard_reject(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "straight_road",
                "junction_visible": False,
                "forward_lane_count": 1,
                "driving_lane_count": 2,
                "has_center_median": False,
            },
            {
                "candidate_topology_type": "straight_two_way",
                "is_junction": False,
                "junction_waypoint_ratio": 0.0,
                "heading_cluster_count": 2,
                "nearby_road_count": 1,
                "same_direction_lane_count": 1,
                "same_road_lane_count": 2,
                "has_center_median_candidate": True,
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertTrue(
            any("center-median mismatch" in reason for reason in details["reject_reasons"])
        )

    def test_v2_cache_open_road_prefers_expected_parking_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = {
                "world_name": "FakeTown",
                "candidates": [
                    {
                        "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 10, "lane_id": -1},
                        "candidate_topology_type": "straight_two_way",
                        "is_junction": False,
                        "junction_waypoint_ratio": 0.0,
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                        "same_road_lane_count": 2,
                        "same_direction_lane_count": 2,
                        "is_curve": False,
                        "has_center_median_candidate": False,
                        "left_parking_lane_present": False,
                        "right_parking_lane_present": False,
                    },
                    {
                        "location": {"x": 2.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 20, "lane_id": -1},
                        "candidate_topology_type": "straight_two_way",
                        "is_junction": False,
                        "junction_waypoint_ratio": 0.0,
                        "heading_cluster_count": 1,
                        "nearby_road_count": 1,
                        "same_road_lane_count": 2,
                        "same_direction_lane_count": 2,
                        "is_curve": False,
                        "has_center_median_candidate": False,
                        "left_parking_lane_present": False,
                        "right_parking_lane_present": True,
                    },
                ],
            }
            (Path(tmp) / "FakeTown.json").write_text(json.dumps(cache), encoding="utf-8")
            matcher = SceneMapMatcher(topology_cache_dir=tmp)

            result = matcher._match_from_cache(
                {
                    "road_topology_signature": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                        "forward_lane_count": 2,
                        "driving_lane_count": 2,
                        "left_parking_presence": False,
                        "right_parking_presence": True,
                        "has_center_median": False,
                    }
                }
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["best_match"]["candidate_lane"]["road_id"], 20)
        self.assertTrue(
            result["best_match"]["candidate_features"]["right_parking_lane_present"]
        )
        self.assertIn(
            "parking",
            result["best_match"]["score_details"]["ranking_components"],
        )

    def test_cached_parking_sides_backfill_opposing_outer_curb(self):
        candidates = [
            {
                "candidate_lane": {"road_id": 76, "lane_id": -1},
                "left_parking_lane_present": False,
                "right_parking_lane_present": True,
            },
            {
                "candidate_lane": {"road_id": 76, "lane_id": 1},
                "left_parking_lane_present": False,
                "right_parking_lane_present": True,
            },
        ]

        SceneMapMatcher._normalize_cached_parking_sides(candidates)

        for candidate in candidates:
            self.assertTrue(candidate["left_parking_lane_present"])
            self.assertTrue(candidate["right_parking_lane_present"])

    def test_v2_cache_junction_parking_lane_does_not_hard_reject_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = {
                "world_name": "FakeTown",
                "candidates": [
                    {
                        "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 10, "lane_id": -1},
                        "candidate_topology_type": "t_junction",
                        "is_junction": True,
                        "junction_waypoint_ratio": 0.4,
                        "heading_cluster_count": 3,
                        "nearby_road_count": 3,
                        "estimated_junction_degree": 3,
                        "same_road_lane_count": 2,
                        "same_direction_lane_count": 1,
                        "junction_branch_dirs": {
                            "ahead": True,
                            "left": False,
                            "right": True,
                        },
                        "right_parking_lane_present": True,
                    },
                ],
            }
            (Path(tmp) / "FakeTown.json").write_text(json.dumps(cache), encoding="utf-8")
            matcher = SceneMapMatcher(topology_cache_dir=tmp)

            result = matcher._match_from_cache(
                {
                    "road_topology_signature": {
                        "topology_type": "t_junction",
                        "junction_type": "t_junction",
                        "junction_visible": True,
                        "junction_branches": {
                            "ahead": True,
                            "left": False,
                            "right": True,
                            "known": True,
                        },
                        "forward_lane_count": 1,
                        "driving_lane_count": 2,
                        "left_parking_presence": False,
                        "right_parking_presence": False,
                        "has_center_median": False,
                    }
                }
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["best_match"]["candidate_lane"]["road_id"], 10)
        self.assertEqual(
            result["best_match"]["score_details"]["candidate_kind"],
            "junction",
        )
        self.assertTrue(
            result["best_match"]["candidate_features"]["right_parking_lane_present"]
        )

    def test_v2_physical_junction_branch_mismatch_is_hard_reject(self):
        score, details = score_candidate_v2(
            {
                "topology_type": "cross_intersection",
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 1,
                "driving_lane_count": 2,
            },
            {
                "candidate_topology_type": "cross_intersection",
                "is_junction": False,
                "junction_waypoint_ratio": 0.4,
                "heading_cluster_count": 4,
                "nearby_road_count": 4,
                "estimated_junction_degree": 4,
                "same_direction_lane_count": 1,
                "same_road_lane_count": 2,
                "physical_junction_arms": {
                    "ahead": True,
                    "left": False,
                    "right": True,
                    "known": True,
                },
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertIn(
            "junction target rejects branch mismatch: missing_required_left_branch",
            details["reject_reasons"],
        )
        self.assertLess(score, 0.9)

    def test_v2_ignores_legacy_lane_maneuvers_for_physical_junction_gate(self):
        score, details = score_candidate_v2(
            {
                "topology_type": "cross_intersection",
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 3,
                "driving_lane_count": 6,
            },
            {
                "candidate_topology_type": "cross_intersection",
                "is_junction": True,
                "junction_waypoint_ratio": 0.6,
                "heading_cluster_count": 4,
                "nearby_road_count": 4,
                "estimated_junction_degree": 4,
                "same_direction_lane_count": 3,
                "same_road_lane_count": 6,
                "junction_branch_dirs": {
                    "ahead": True,
                    "left": False,
                    "right": False,
                },
            },
        )

        self.assertFalse(details["hard_reject"])
        self.assertFalse(
            any("branch mismatch" in reason for reason in details["reject_reasons"])
        )
        self.assertGreater(score, 0.5)

    def test_v2_uses_physical_arms_over_lane_maneuvers(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "cross_intersection",
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 3,
                "driving_lane_count": 6,
            },
            {
                "candidate_topology_type": "cross_intersection",
                "is_junction": True,
                "junction_waypoint_ratio": 0.6,
                "heading_cluster_count": 4,
                "nearby_road_count": 4,
                "estimated_junction_degree": 4,
                "same_direction_lane_count": 3,
                "same_road_lane_count": 6,
                "physical_junction_arms": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "junction_branch_dirs": {
                    "ahead": True,
                    "left": False,
                    "right": False,
                },
            },
        )

        self.assertFalse(details["hard_reject"])
        self.assertEqual(details["ranking_components"]["branch_direction"], 1.0)

    def test_v2_junction_lane_mismatch_is_hard_reject(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "cross_intersection",
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 2,
                "driving_lane_count": 4,
            },
            {
                "candidate_topology_type": "cross_intersection",
                "is_junction": False,
                "junction_waypoint_ratio": 0.6,
                "heading_cluster_count": 4,
                "nearby_road_count": 4,
                "estimated_junction_degree": 4,
                "same_direction_lane_count": 1,
                "same_road_lane_count": 2,
                "physical_junction_arms": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertTrue(
            any("too few approach lanes" in reason for reason in details["reject_reasons"])
        )
        self.assertTrue(
            any("severe driving-lane mismatch" in reason for reason in details["reject_reasons"])
        )

    def test_v2_signalized_junction_without_signal_is_hard_reject(self):
        _score, details = score_candidate_v2(
            {
                "topology_type": "cross_intersection",
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 1,
                "driving_lane_count": 2,
                "has_traffic_light": True,
            },
            {
                "candidate_topology_type": "cross_intersection",
                "is_junction": False,
                "junction_waypoint_ratio": 0.6,
                "heading_cluster_count": 4,
                "nearby_road_count": 4,
                "estimated_junction_degree": 4,
                "same_direction_lane_count": 1,
                "same_road_lane_count": 2,
                "physical_junction_arms": {
                    "ahead": True,
                    "left": True,
                    "right": True,
                    "known": True,
                },
                "distance_to_traffic_light_ahead": None,
            },
        )

        self.assertTrue(details["hard_reject"])
        self.assertIn(
            "signalized junction target rejects no-signal candidate",
            details["reject_reasons"],
        )

    def test_v2_cache_does_not_match_when_only_rejected_candidate_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = {
                "world_name": "FakeTown",
                "candidates": [
                    {
                        "location": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "yaw": 0.0,
                        "candidate_lane": {"road_id": 10, "lane_id": -1},
                        "candidate_topology_type": "t_junction",
                        "is_junction": True,
                        "junction_waypoint_ratio": 0.5,
                        "heading_cluster_count": 3,
                        "nearby_road_count": 3,
                        "same_road_lane_count": 3,
                        "same_direction_lane_count": 3,
                    }
                ],
            }
            (Path(tmp) / "FakeTown.json").write_text(json.dumps(cache), encoding="utf-8")
            matcher = SceneMapMatcher(topology_cache_dir=tmp)

            result = matcher._match_from_cache(
                {
                    "road_topology_signature": {
                        "topology_type": "straight_road",
                        "junction_visible": False,
                        "forward_lane_count": 3,
                        "driving_lane_count": 3,
                    }
                }
            )

        self.assertEqual(result["status"], "unmatched")
        self.assertIsNone(result["best_match"])
        self.assertIsNotNone(result["candidate_debug"]["best_rejected_candidate"])

    def test_topology_fallback_penalizes_overly_wide_candidates(self):
        narrow_score = SceneMapMatcher._topology_fallback_score(
            uncapped_total_score=0.40,
            target_topology="straight_two_way",
            candidate_topology="multi_branch",
            target_driving_lanes=2,
            candidate_driving_lanes=2,
            heading_clusters=4,
            nearby_roads=4,
            junction_ratio=0.2,
        )
        wide_score = SceneMapMatcher._topology_fallback_score(
            uncapped_total_score=0.76,
            target_topology="straight_two_way",
            candidate_topology="straight_two_way",
            target_driving_lanes=2,
            candidate_driving_lanes=4,
            heading_clusters=2,
            nearby_roads=1,
            junction_ratio=0.0,
        )
        self.assertGreater(narrow_score, wide_score)

    def test_curve_topology_scoring_prefers_explicit_curve_candidate(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "curve",
                "target_branch_count": 1,
                "driving_lane_count": 2,
                "has_crosswalk": False,
                "has_traffic_light": False,
                "has_traffic_sign": False,
            }
        }
        straight_candidate = {
            "candidate_topology_type": "straight_two_way",
            "estimated_junction_degree": 1,
            "same_road_lane_count": 2,
            "junction_waypoint_ratio": 0.0,
            "heading_cluster_count": 1,
            "nearby_road_count": 1,
            "is_junction": False,
            "is_curve": False,
        }
        curve_candidate = {
            **straight_candidate,
            "candidate_topology_type": "curve",
            "is_curve": True,
            "curve_score": 0.8,
            "curve_direction": "right",
            "curve_yaw_delta_deg": 24.0,
        }

        straight_score, _ = matcher._score_topology_candidate_features(
            straight_candidate, scene_features
        )
        curve_score, details = matcher._score_topology_candidate_features(
            curve_candidate, scene_features
        )

        self.assertGreater(curve_score, straight_score)
        self.assertTrue(details["candidate_is_curve"])

    def test_curve_direction_scoring_prefers_matching_direction(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "curve",
                "target_branch_count": 1,
                "forward_lane_count": 1,
                "opposing_lane_count": 1,
                "driving_lane_count": 2,
                "curve_direction": "right",
            }
        }
        base_candidate = {
            "candidate_topology_type": "curve",
            "estimated_junction_degree": 1,
            "same_road_lane_count": 2,
            "junction_waypoint_ratio": 0.0,
            "heading_cluster_count": 1,
            "nearby_road_count": 1,
            "is_junction": False,
            "is_curve": True,
            "curve_score": 0.8,
        }

        right_score, right_details = matcher._score_topology_candidate_features(
            {**base_candidate, "curve_direction": "right"}, scene_features
        )
        left_score, left_details = matcher._score_topology_candidate_features(
            {**base_candidate, "curve_direction": "left"}, scene_features
        )

        self.assertGreater(right_score, left_score)
        self.assertEqual(right_details["curve_direction_detail"]["status"], "matched")
        self.assertEqual(left_details["curve_direction_detail"]["status"], "mismatch")

    def test_junction_distance_scoring_prefers_matching_ego_distance(self):
        matcher = SceneMapMatcher()
        scene_features = {
            "road_topology_signature": {
                "topology_type": "t_junction",
                "target_branch_count": 3,
                "junction_visible": True,
                "junction_branches": {
                    "ahead": True,
                    "left": False,
                    "right": True,
                    "known": True,
                },
                "forward_lane_count": 2,
                "opposing_lane_count": 0,
                "driving_lane_count": 2,
                "ego_to_junction_distance_m": 20.0,
            }
        }
        base_candidate = {
            "candidate_topology_type": "t_junction",
            "estimated_junction_degree": 3,
            "same_road_lane_count": 2,
            "same_direction_lane_count": 2,
            "junction_waypoint_ratio": 0.3,
            "heading_cluster_count": 3,
            "nearby_road_count": 3,
            "is_junction": False,
            "distance_to_traffic_light_ahead": None,
            "junction_branch_dirs": {
                "ahead": True,
                "left": False,
                "right": True,
            },
        }

        near_score, near_details = matcher._score_topology_candidate_features(
            {**base_candidate, "distance_to_junction_ahead": 22.0}, scene_features
        )
        far_score, far_details = matcher._score_topology_candidate_features(
            {**base_candidate, "distance_to_junction_ahead": 60.0}, scene_features
        )

        self.assertGreater(near_score, far_score)
        self.assertEqual(near_details["junction_distance_detail"]["status"], "matched")
        self.assertEqual(far_details["junction_distance_detail"]["status"], "far")

    def test_topology_match_applies_candidate_lane_to_spawn_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "enable_scene_verify": False,
                },
            )
            candidate_lane = {
                "road_id": 77,
                "lane_id": -1,
                "start": {"x": 10.0, "y": 20.0, "z": 0.0, "yaw": 45.0},
                "end": {"x": 20.0, "y": 30.0, "z": 0.0, "yaw": 45.0},
            }
            match_path = Path(tmp) / "scene_match.json"
            match_path.write_text(
                json.dumps({"best_match": {"candidate_lane": candidate_lane}}),
                encoding="utf-8",
            )
            generator._apply_match_to_spawn_context(str(match_path))
            self.assertEqual(generator.carla_spawn_context["topology_sample"][0]["road_id"], 77)

    def test_cache_best_match_preserves_matched_structure(self):
        matched_structure = {
            "kind": "junction",
            "junction_id": 12,
            "center": {"x": 1.0, "y": 2.0, "z": 0.0},
            "legs": [],
        }
        best_match = SceneMapMatcher._assemble_cache_best_match(
            {
                "score": 0.9,
                "score_details": {},
                "candidate_lane": {"road_id": 1},
                "location": {"x": 1.0, "y": 2.0, "z": 0.0},
                "yaw": 0.0,
                "matched_structure": matched_structure,
            }
        )

        self.assertEqual(best_match["matched_structure"], matched_structure)

    def test_topology_match_applies_matched_structure_to_spawn_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "enable_scene_verify": False,
                },
            )
            matched_structure = {
                "kind": "junction",
                "junction_id": 42,
                "center": {"x": 10.0, "y": 20.0, "z": 0.0},
                "legs": [],
            }
            match_path = Path(tmp) / "scene_match.json"
            match_path.write_text(
                json.dumps(
                    {
                        "best_match": {
                            "candidate_lane": {
                                "road_id": 77,
                                "lane_id": -1,
                                "start": {"x": 10.0, "y": 20.0, "z": 0.0, "yaw": 45.0},
                                "end": {"x": 20.0, "y": 30.0, "z": 0.0, "yaw": 45.0},
                            },
                            "matched_structure": matched_structure,
                        }
                    }
                ),
                encoding="utf-8",
            )

            generator._apply_match_to_spawn_context(str(match_path))

            self.assertEqual(generator.carla_spawn_context["matched_structure"], matched_structure)
            self.assertEqual(generator.carla_spawn_context["matched_structure_source"], "match_report")

    def test_persist_matched_structure_writes_sidecar_and_match_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "enable_scene_verify": False,
                },
            )
            matched_structure = {
                "kind": "junction",
                "junction_id": 42,
                "center": {"x": 10.0, "y": 20.0, "z": 0.0},
                "legs": [
                    {"name": "ego", "heading_out_deg": 180.0},
                    {"name": "right", "heading_out_deg": 90.0},
                    {"name": "opposite", "heading_out_deg": 0.0},
                ],
                "leg_count": 3,
            }
            match_path = Path(tmp) / "scene_match.json"
            match_path.write_text(
                json.dumps({"best_match": {"candidate_lane": {"road_id": 77}}}),
                encoding="utf-8",
            )

            generator._persist_matched_structure(
                "scene",
                str(match_path),
                matched_structure,
                "live_carla",
            )

            sidecar = json.loads((Path(tmp) / "scene_matched_structure.json").read_text())
            updated_match = json.loads(match_path.read_text())
            self.assertEqual(sidecar["matched_structure"], matched_structure)
            self.assertEqual(sidecar["summary"]["source"], "live_carla")
            self.assertTrue(sidecar["summary"]["has_right_leg"])
            self.assertEqual(
                updated_match["best_match"]["matched_structure"],
                matched_structure,
            )
            self.assertEqual(
                updated_match["best_match"]["matched_structure_source"],
                "live_carla",
            )

    def test_crossing_actor_on_straight_road_does_not_require_junction_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = AutoGenerator(
                tmp,
                {
                    "input_type": "image",
                    "require_carla_connection": False,
                    "enable_scene_verify": False,
                },
            )
            refined = {
                "entities": [
                    {
                        "id": "veh_1",
                        "category": "car",
                        "heading_relation": "crossing",
                        "layout_anchor_id": "right_arm",
                        "location": {"x": 0.0, "y": 0.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                    }
                ]
            }

            self.assertIs(generator.reproject_junction_step("scene", refined), refined)

    def test_scene_requires_junction_only_for_topology_evidence(self):
        straight_with_crossing_actor = {
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_road",
                    "junction_visible": False,
                    "junction_type": "none",
                }
            },
            "traffic_subjects": [
                {
                    "category": "car",
                    "heading_relation_to_ego": "crossing",
                    "layout_anchor_id": "right_arm",
                }
            ],
        }
        self.assertFalse(
            AutoGenerator._scene_requires_junction_structure(straight_with_crossing_actor)
        )

        t_junction = {
            "road_network": {
                "map_matching": {
                    "topology_type": "t_junction",
                    "junction_visible": True,
                    "junction_type": "t_junction",
                    "target_branch_count": 3,
                }
            }
        }
        self.assertTrue(AutoGenerator._scene_requires_junction_structure(t_junction))

        branch_evidence = {
            "road_network": {
                "map_matching": {
                    "topology_type": "unknown",
                    "junction_visible": True,
                    "junction_type": "unknown",
                    "junction_branches": {
                        "known": True,
                        "ahead": True,
                        "left": True,
                        "right": False,
                    },
                }
            }
        }
        self.assertTrue(AutoGenerator._scene_requires_junction_structure(branch_evidence))

    def test_junction_side_arm_vehicle_projects_to_declared_lane(self):
        payload = build_projected_spawn_payload(
            {
                "entities": [
                    {
                        "id": "right_arm_car",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "heading_relation": "crossing",
                        "lane_side_relation": "right_lane",
                        "lane_index_relation": 1,
                        "junction_placement": "frame",
                        "junction_direction": "right",
                        "junction_leg": "south",
                        "junction_motion": "approaching",
                        "projected_lane": {
                            "road_id": 622,
                            "lane_id": 1,
                            "role": "inbound",
                            "yaw": 270.153564453125,
                            "source": "junction_leg",
                        },
                        "location": {"x": 85.0, "y": -170.0, "z": 0.3},
                        "rotation": {"pitch": 0.0, "yaw": -90.0, "roll": 0.0},
                    }
                ]
            }
        )

        entity = payload["entities"][0]
        self.assertEqual(entity["placement_mode"], "project_to_junction_lane")
        self.assertEqual(entity["projected_lane"]["road_id"], 622)
        self.assertEqual(entity["projected_lane"]["yaw"], 270.153564453125)
        self.assertEqual(entity["junction_direction"], "right")

    def test_junction_actor_uses_anchor_relation_distance_to_center(self):
        coordinates = {
            "entities": [
                {
                    "id": "right_arm_car",
                    "spawn_kind": "vehicle",
                    "category": "car",
                    "heading_relation": "crossing",
                    "flow_compliance": "legal",
                    "lane_side_relation": "right_lane",
                    "lane_index_relation": 1,
                    "layout_anchor_id": "right_arm",
                    "anchor_relation": {
                        "travel_direction": "toward_junction",
                        "distance_to_junction_m": 24.0,
                        "lane_from_right": 0,
                    },
                    "location": {"x": 0.0, "y": 0.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0},
                }
            ]
        }
        matched_structure = {
            "kind": "junction",
            "junction_id": 7,
            "center": {"x": 10.0, "y": 20.0, "z": 0.0},
            "ego_approach_heading_deg": 0.0,
            "junction_radius_m": 6.0,
            "legs": [
                {
                    "name": "right",
                    "heading_out_deg": 90.0,
                    "inbound_lanes": [
                        {
                            "road_id": 77,
                            "lane_id": -1,
                            "role": "inbound",
                            "yaw": 270.0,
                        }
                    ],
                }
            ],
        }

        reprojected = reproject_actors_for_junction(coordinates, matched_structure)
        entity = reprojected["entities"][0]
        payload = build_projected_spawn_payload(reprojected)

        self.assertEqual(entity["junction_placement"], "frame")
        self.assertEqual(entity["junction_direction"], "right")
        self.assertEqual(entity["junction_motion"], "approaching")
        self.assertAlmostEqual(entity["junction_distance_m"], 24.0)
        self.assertEqual(payload["entities"][0]["placement_mode"], "project_to_junction_lane")

    def test_junction_frame_row_keeps_lane_yaw(self):
        payload = build_projected_spawn_payload(
            {
                "selected_anchor_lane": {
                    "start": {"x": 0.0, "y": 0.0},
                    "end": {"x": 10.0, "y": 0.0},
                },
                "entities": [
                    {
                        "id": "right_arm_truck",
                        "spawn_kind": "vehicle",
                        "category": "truck",
                        "heading_relation": "crossing",
                        "turn_intent": "right",
                        "lane_side_relation": "right_lane",
                        "lane_index_relation": 1,
                        "group_id": "right_arm_queue",
                        "junction_placement": "frame",
                        "junction_direction": "right",
                        "junction_leg": "right",
                        "junction_motion": "leaving",
                        "projected_lane": {
                            "road_id": 2282,
                            "lane_id": 1,
                            "role": "outbound",
                            "yaw": -90.51138305664062,
                            "source": "junction_leg",
                        },
                        "location": {"x": -123.7, "y": 117.6, "z": 0.3},
                        "rotation": {
                            "pitch": 0.0,
                            "yaw": -90.51138305664062,
                            "roll": 0.0,
                        },
                    },
                    {
                        "id": "right_arm_car",
                        "spawn_kind": "vehicle",
                        "category": "car",
                        "heading_relation": "crossing",
                        "turn_intent": "none",
                        "lane_side_relation": "right_lane",
                        "lane_index_relation": 1,
                        "group_id": "right_arm_queue",
                        "junction_placement": "frame",
                        "junction_direction": "right",
                        "junction_leg": "right",
                        "junction_motion": "approaching",
                        "projected_lane": {
                            "road_id": 2292,
                            "lane_id": 1,
                            "role": "inbound",
                            "yaw": -270.5113525390625,
                            "source": "junction_leg",
                        },
                        "location": {"x": -130.7, "y": 117.5, "z": 0.3},
                        "rotation": {
                            "pitch": 0.0,
                            "yaw": -270.5113525390625,
                            "roll": 0.0,
                        },
                    },
                ],
            }
        )

        by_id = {entity["id"]: entity for entity in payload["entities"]}
        self.assertAlmostEqual(
            by_id["right_arm_truck"]["rotation"]["yaw"],
            -90.51138305664062,
        )
        self.assertAlmostEqual(
            by_id["right_arm_car"]["rotation"]["yaw"],
            -270.5113525390625,
        )

if __name__ == "__main__":
    unittest.main()
