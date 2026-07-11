import json
import sys
import tempfile
import types
import unittest
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

from experiments.auto_generate_all_vlm import AutoGenerator


SPLIT_TEXT = """\
## Road Net Description:
A two-lane urban street.

## Road Users Description:
A car is ahead in the same lane.

## Static Objects Description:
Traffic cones on the right shoulder.

## Vehicles' Locations and Behaviors:
The car ahead is moving slowly.

## Scenario Description:
Ego approaches slow-moving traffic near cones.
"""


def _make_generator(tmp, extra=None):
    info = {
        "input_type": "image",
        "require_carla_connection": False,
    }
    if extra:
        info.update(extra)
    return AutoGenerator(tmp, info)


class TestNormalizeCarlaWorldName(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(AutoGenerator._normalize_carla_world_name(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(AutoGenerator._normalize_carla_world_name(""))

    def test_plain_name_unchanged(self):
        self.assertEqual(
            AutoGenerator._normalize_carla_world_name("Town10HD_Opt"),
            "Town10HD_Opt",
        )

    def test_path_returns_last_segment(self):
        self.assertEqual(
            AutoGenerator._normalize_carla_world_name("Carla/Maps/Town10HD_Opt"),
            "Town10HD_Opt",
        )


class TestFallbackTopologySample(unittest.TestCase):
    def test_returns_list_with_one_entry(self):
        sample = AutoGenerator._fallback_topology_sample()
        self.assertIsInstance(sample, list)
        self.assertEqual(len(sample), 1)

    def test_entry_has_required_keys(self):
        entry = AutoGenerator._fallback_topology_sample()[0]
        for key in ("road_id", "lane_id", "start", "end"):
            self.assertIn(key, entry)


class TestLayoutCapture(unittest.TestCase):
    def test_capture_layout_images_prefers_ego_view_and_sets_both_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            generator.require_carla_connection = True
            generator.carla_spawn_context = {"status": "available"}
            seen_env = {}

            def fake_run(*_args, **kwargs):
                seen_env.update(kwargs["env"])
                Path(kwargs["env"]["AUTOSCENARIO_EGO_VIEW_OUTPUT"]).write_bytes(b"ego")
                Path(kwargs["env"]["AUTOSCENARIO_BEV_OUTPUT"]).write_bytes(b"bev")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("experiments.auto_generate_all_vlm.subprocess.run", fake_run):
                capture = generator._capture_layout_images("s0000", "/tmp/final.py", 1)

            self.assertEqual(capture["capture_mode"], "ego_view")
            self.assertEqual(capture["layout_image_path"], capture["ego_view_path"])
            self.assertTrue(Path(capture["bev_path"]).exists())
            self.assertIn("AUTOSCENARIO_EGO_VIEW_OUTPUT", seen_env)
            self.assertIn("AUTOSCENARIO_BEV_OUTPUT", seen_env)
            self.assertIn("AUTOSCENARIO_RENDER_ACTOR_GRAPH_OUTPUT", seen_env)

    def test_capture_layout_images_falls_back_to_bev_when_ego_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            generator.require_carla_connection = True
            generator.carla_spawn_context = {"status": "available"}

            def fake_run(*_args, **kwargs):
                Path(kwargs["env"]["AUTOSCENARIO_BEV_OUTPUT"]).write_bytes(b"bev")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("experiments.auto_generate_all_vlm.subprocess.run", fake_run):
                capture = generator._capture_layout_images("s0000", "/tmp/final.py", 1)

            self.assertEqual(capture["capture_mode"], "bev_fallback")
            self.assertIsNone(capture["ego_view_path"])
            self.assertEqual(capture["layout_image_path"], capture["bev_path"])

    def test_capture_quick_bev_preview_writes_fixed_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            generator.require_carla_connection = True
            generator.carla_spawn_context = {"status": "available"}
            seen_env = {}

            def fake_run(*_args, **kwargs):
                seen_env.update(kwargs["env"])
                Path(kwargs["env"]["AUTOSCENARIO_BEV_OUTPUT"]).write_bytes(b"bev")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("experiments.auto_generate_all_vlm.subprocess.run", fake_run):
                capture = generator._capture_quick_bev_preview("s0000", "/tmp/final.py")

            self.assertIsNone(capture["error"])
            self.assertEqual(capture["bev_path"], str(Path(tmp) / "s0000_quick_bev.png"))
            self.assertTrue(Path(capture["bev_path"]).exists())
            self.assertEqual(Path(tmp, "image.png").read_bytes(), b"bev")
            self.assertIn("AUTOSCENARIO_BEV_OUTPUT", seen_env)
            self.assertNotIn("AUTOSCENARIO_EGO_VIEW_OUTPUT", seen_env)

    def test_capture_quick_bev_preview_skips_without_carla(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)

            capture = generator._capture_quick_bev_preview("s0000", "/tmp/final.py")

            self.assertTrue(capture["enabled"])
            self.assertIsNone(capture["bev_path"])
            self.assertIn("CARLA connection disabled", capture["error"])

    def test_start_and_end_have_coordinate_keys(self):
        entry = AutoGenerator._fallback_topology_sample()[0]
        for coord_key in ("x", "y", "z", "yaw"):
            self.assertIn(coord_key, entry["start"])
            self.assertIn(coord_key, entry["end"])


class TestActorGraphVerifyRepairIntegration(unittest.TestCase):
    def _basic_inputs(self, tmp):
        scene_id = "scene"
        relation_dsl = {
            "ego_relations": [
                {
                    "entity_id": "car_1",
                    "group_id": "car_1",
                    "category": "car",
                    "subtype": "car",
                    "spawn_kind": "vehicle",
                    "lane_side_relation": "same_lane",
                    "order_relation": "ahead",
                    "distance_band": "near",
                    "heading_relation": "same_direction",
                }
            ]
        }
        spawn_payload = {
            "entities": [
                {
                    "id": "car_1",
                    "category": "car",
                    "spawn_kind": "vehicle",
                    "lane_side_relation": "same_lane",
                    "heading_relation": "same_direction",
                    "location": {"x": 10.0, "y": 0.0, "z": 0.3},
                    "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                }
            ]
        }
        match_path = Path(tmp) / f"{scene_id}_match.json"
        match_path.write_text(json.dumps({"status": "matched"}), encoding="utf-8")
        return scene_id, relation_dsl, spawn_payload, str(match_path)

    def test_actor_graph_mode_is_default_verify_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                },
            )

            self.assertEqual(generator.verify_mode, "actor_graph")

    def test_actor_patch_executor_updates_semantic_targets_and_audit_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id, relation_dsl, spawn_payload, _match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload), encoding="utf-8"
            )
            compiled = {
                "schema_version": "actor-repair-v1",
                "patches": [
                    {
                        "op": "set_distance_band",
                        "entity_id": "car_1",
                        "target_band": "mid",
                        "severity": "high",
                    },
                    {
                        "op": "set_heading_relation",
                        "entity_id": "car_1",
                        "target_heading": "opposite_direction",
                        "severity": "medium",
                    },
                ],
                "rejected": [
                    {
                        "requested_patch": {"op": "bad"},
                        "reason": "unsupported_patch_type",
                    }
                ],
                "semantic_unrepairable": [],
            }

            result = generator._apply_actor_repair_patches(scene_id, {}, compiled)
            repaired = json.loads(
                Path(generator._spawn_payload_path(scene_id)).read_text(encoding="utf-8")
            )
            car = repaired["entities"][0]

            self.assertEqual(result["applied_count"], 2)
            self.assertEqual(result["rejected_count"], 1)
            self.assertEqual(car["heading_relation"], "opposite_direction")
            self.assertEqual(car["longitudinal_m"], 12.5)
            self.assertEqual(
                repaired["repair_metadata"]["target_overrides"]["car_1"]["distance_band"],
                "mid",
            )
            self.assertEqual(
                [item["status"] for item in repaired["repair_metadata"]["patch_outcomes"]],
                ["rejected", "applied", "applied"],
            )
            target_graph = generator._write_source_actor_graph(
                scene_id, {}, relation_dsl
            )
            target_actor = target_graph["actors"][0]
            self.assertEqual(target_actor["distance_band"], "mid")
            self.assertEqual(
                target_actor["heading_relation"], "opposite_direction"
            )

    def test_pose_target_uses_actual_render_ego_frame_not_payload_longitudinal(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id, _relation_dsl, spawn_payload, _match_path = self._basic_inputs(tmp)
            spawn_payload["entities"].append(
                {
                    "id": "ego",
                    "category": "car",
                    "location": {"x": 0.0, "y": 0.0, "z": 0.3},
                    "rotation": {"yaw": 0.0},
                }
            )
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload), encoding="utf-8"
            )
            render_graph = {
                "metadata": {
                    "ego": {"location": {"x": 100.0, "y": 200.0, "z": 0.3}, "yaw": 90.0}
                },
                "actors": [
                    {
                        "id": "car_1",
                        "ego_frame": {"longitudinal_m": 30.0},
                        "heading_relation_to_ego": "same_direction",
                    }
                ],
            }
            result = generator._apply_actor_repair_patches(
                scene_id,
                {"lane_width_m": 3.5},
                {
                    "patches": [
                        {
                            "op": "set_pose_target",
                            "entity_id": "car_1",
                            "target_lane": "right_parking_lane",
                            "target_longitudinal_m": 8.0,
                            "target_lateral_offset_m": 1.0,
                            "target_heading_relation": "opposite_direction",
                            "severity": "high",
                        }
                    ],
                    "rejected": [],
                    "semantic_unrepairable": [],
                },
                render_graph=render_graph,
            )
            repaired = json.loads(
                Path(generator._spawn_payload_path(scene_id)).read_text(encoding="utf-8")
            )
            car = next(item for item in repaired["entities"] if item["id"] == "car_1")

            self.assertEqual(result["applied_count"], 1)
            self.assertEqual(car["placement_mode"], "project_to_visual_pose")
            self.assertEqual(car["longitudinal_m"], 8.0)
            self.assertAlmostEqual(car["location"]["x"], 95.5)
            self.assertAlmostEqual(car["location"]["y"], 208.0)
            self.assertEqual(car["heading_relation"], "opposite_direction")
            self.assertEqual(
                car["visual_position_override"]["target_lateral_offset_m"], 1.0
            )

    def test_actor_graph_heading_patch_is_merged_when_vlm_omits_it(self):
        spawn_payload = {
            "entities": [
                {"id": "car_1", "heading_relation": "same_direction"}
            ]
        }
        source_graph = {
            "actors": [
                {
                    "id": "car_1",
                    "heading_relation": "opposite_direction",
                    "distance_band": "near",
                    "lane_side_relation": "same_lane",
                }
            ]
        }
        plan = {
            "issues": [
                {
                    "issue_type": "heading_mismatch",
                    "source_entity_id": "car_1",
                    "severity": "medium",
                }
            ]
        }

        compiled = AutoGenerator._compile_actor_repair_patches(
            {"repair_patches": [], "patch_rejections": [], "semantic_unrepairable": []},
            plan,
            source_graph,
            spawn_payload,
        )

        self.assertEqual(compiled["patches"][0]["op"], "set_heading_relation")
        self.assertEqual(
            compiled["patches"][0]["target_heading"], "opposite_direction"
        )

    def test_live_actor_graph_geometry_compiles_to_pose_target(self):
        compiled = AutoGenerator._compile_actor_repair_patches(
            {"repair_patches": [], "patch_rejections": [], "semantic_unrepairable": []},
            {
                "issues": [
                    {
                        "issue_type": "longitudinal_mismatch",
                        "source_entity_id": "car_1",
                        "severity": "medium",
                        "evidence": "actual pose is too near",
                    }
                ]
            },
            {
                "actors": [
                    {
                        "id": "car_1",
                        "lane_side_relation": "right_lane",
                        "heading_relation": "same_direction",
                        "expected_longitudinal_m": 18.0,
                    }
                ]
            },
            {"entities": [{"id": "car_1"}]},
            render_graph={
                "actors": [
                    {"id": "car_1", "ego_frame": {"longitudinal_m": 8.0}}
                ]
            },
        )

        self.assertEqual(compiled["patches"][0]["op"], "set_pose_target")
        self.assertEqual(compiled["patches"][0]["target_longitudinal_m"], 18.0)
        self.assertEqual(compiled["patches"][0]["target_lane"], "right_lane")

    def test_distance_band_patch_uses_nearest_position_with_minimum_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id = "spacing"
            spawn_payload = {
                "entities": [
                    {
                        "id": "car_1",
                        "category": "car",
                        "lane_index_relation": 0,
                        "location": {"x": 10.0, "y": 0.0, "z": 0.3},
                        "rotation": {"yaw": 0.0},
                    },
                    {
                        "id": "car_2",
                        "category": "car",
                        "lane_index_relation": 0,
                        "longitudinal_m": 12.5,
                        "location": {"x": 12.5, "y": 0.0, "z": 0.3},
                        "rotation": {"yaw": 0.0},
                    },
                ]
            }
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload), encoding="utf-8"
            )
            result = generator._apply_actor_repair_patches(
                scene_id,
                {},
                {
                    "patches": [
                        {
                            "op": "set_distance_band",
                            "entity_id": "car_1",
                            "target_band": "mid",
                            "severity": "medium",
                        }
                    ],
                    "rejected": [],
                    "semantic_unrepairable": [],
                },
            )
            repaired = json.loads(
                Path(generator._spawn_payload_path(scene_id)).read_text(encoding="utf-8")
            )

            self.assertEqual(result["applied_count"], 1)
            self.assertEqual(repaired["entities"][0]["longitudinal_m"], 18.0)

    def test_layered_rollback_compares_fact_vector_before_visual_score(self):
        baseline = {
            "high_fact_issue_count": 0,
            "spawn_failure_count": 0,
            "medium_geometry_issue_count": 1,
            "visual_available": True,
            "visual_score": 0.8,
        }
        better_geometry = {
            **baseline,
            "medium_geometry_issue_count": 0,
            "visual_score": 0.6,
        }
        worse_geometry = {
            **baseline,
            "high_fact_issue_count": 1,
            "visual_score": 0.95,
        }

        self.assertFalse(
            AutoGenerator._evaluation_is_worse(better_geometry, baseline)
        )
        self.assertTrue(AutoGenerator._evaluation_is_worse(worse_geometry, baseline))
        self.assertTrue(
            AutoGenerator._evaluation_is_worse(
                {**baseline, "visual_available": False, "visual_score": None},
                baseline,
            )
        )
        self.assertFalse(
            AutoGenerator._evaluation_is_worse(
                baseline,
                {**baseline, "visual_available": False, "visual_score": None},
            )
        )

    def test_unconfirmed_inventory_feedback_is_rejected_not_repaired(self):
        compiled = AutoGenerator._compile_actor_repair_patches(
            {
                "repair_patches": [],
                "patch_rejections": [],
                "semantic_unrepairable": [
                    {"type": "count_mismatch", "severity": "high"}
                ],
            },
            {"issues": []},
            {"actors": [{"id": "car_1"}]},
            {"entities": [{"id": "car_1"}]},
        )

        self.assertEqual(compiled["semantic_unrepairable"], [])
        self.assertEqual(
            compiled["rejected"][0]["reason"],
            "semantic_issue_not_confirmed_by_actor_graph",
        )

    def test_two_round_regression_reverts_payload_and_selects_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "vlm",
                    "verify_max_rounds": 2,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            payload_path = Path(generator._spawn_payload_path(scene_id))
            payload_path.write_text(json.dumps(spawn_payload), encoding="utf-8")
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": str(Path(tmp) / "bev.png"),
                "capture_mode": "ego_view",
                "error": None,
            }
            baseline_plan = {
                "passed": False,
                "score": 0.8,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "issues": [
                    {
                        "issue_type": "heading_mismatch",
                        "source_entity_id": "car_1",
                        "severity": "medium",
                    }
                ],
                "repair_actions": [],
            }
            regressed_plan = {
                "passed": False,
                "score": 0.0,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "issues": [
                    {
                        "issue_type": "spawn_failure",
                        "source_entity_id": "car_1",
                        "severity": "high",
                    }
                ],
                "repair_actions": [],
            }
            reports = [
                {
                    "passed": False,
                    "score": 0.5,
                    "recommended_stage": "match_spawn",
                    "repair_patches": [
                        {
                            "op": "set_heading_relation",
                            "entity_id": "car_1",
                            "target_heading": "opposite_direction",
                            "severity": "medium",
                        }
                    ],
                },
                {
                    "passed": True,
                    "score": 0.95,
                    "recommended_stage": "pass",
                    "repair_patches": [],
                },
            ]

            with mock.patch.object(
                generator, "_capture_layout_images", return_value=capture
            ), mock.patch.object(
                generator,
                "_write_actor_graph_repair_plan",
                side_effect=[baseline_plan, regressed_plan],
            ), mock.patch.object(
                generator, "_verify_scene_round", side_effect=reports
            ), mock.patch.object(
                generator, "generate_final_scene_script", return_value="final.py"
            ):
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            restored = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["regression_reverted"])
            self.assertEqual(summary["selected_round"], 1)
            self.assertEqual(summary["rounds"][1]["repair"], "reverted_regression")
            self.assertEqual(
                restored["entities"][0]["heading_relation"], "same_direction"
            )
            self.assertNotIn("repair_metadata", restored)

    def test_vlm_mode_writes_actor_graph_artifacts_without_replacing_vlm_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "vlm",
                    "verify_max_rounds": 1,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": None,
                "capture_mode": "ego_view",
                "error": None,
            }
            with mock.patch.object(generator, "_capture_layout_images", return_value=capture), \
                mock.patch.object(
                    generator,
                    "_write_actor_graph_repair_plan",
                    return_value={
                        "passed": True,
                        "score": 1.0,
                        "status": "passed",
                        "truth_source": "carla_actor_transform",
                        "issues": [],
                        "repair_actions": [],
                    },
                ), \
                mock.patch.object(
                    generator,
                    "_verify_scene_round",
                    return_value={
                        "passed": True,
                        "score": 0.9,
                        "recommended_stage": "pass",
                        "repair_patches": [],
                    },
                ) as verify_mock:
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            self.assertEqual(summary["rounds"][0]["repair"], "none")
            verify_mock.assert_called_once()
            self.assertTrue(Path(generator._source_actor_graph_path(scene_id)).exists())
            self.assertTrue(Path(generator._actor_graph_repair_plan_path(scene_id, 1)).exists())

    def test_vlm_mode_still_records_visual_evidence_for_high_fact_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "vlm",
                    "verify_max_rounds": 2,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": None,
                "capture_mode": "ego_view",
                "error": None,
            }
            blocked_plan = {
                "passed": False,
                "score": 0.0,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "stop_repair_loop": False,
                "requires_code_fix": True,
                "issues": [
                    {
                        "issue_type": "systematic_projection_error",
                        "severity": "high",
                        "evidence": "repeated lane-side failure",
                    }
                ],
                "repair_actions": [
                    {
                        "type": "lane_side_mismatch",
                        "entity_id": "car_1",
                        "severity": "medium",
                    }
                ],
            }
            with mock.patch.object(generator, "_capture_layout_images", return_value=capture), \
                mock.patch.object(generator, "_write_actor_graph_repair_plan", return_value=blocked_plan), \
                mock.patch.object(
                    generator,
                    "_verify_scene_round",
                    return_value={
                        "passed": False,
                        "score": 0.2,
                        "recommended_stage": "match_spawn",
                        "repair_patches": [],
                    },
                ) as verify_mock, \
                mock.patch.object(generator, "_apply_actor_repair_patches") as repair_mock:
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            self.assertEqual(summary["rounds"][0]["repair"], "blocked_high_fact_issue")
            verify_mock.assert_called_once()
            repair_mock.assert_not_called()

    def test_vlm_semantic_feedback_never_rewrites_scene_understanding(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "vlm",
                    "verify_max_rounds": 2,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": None,
                "capture_mode": "ego_view",
                "error": None,
            }
            nonblocking_plan = {
                "passed": False,
                "score": 0.5,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "stop_repair_loop": False,
                "requires_code_fix": False,
                "issues": [],
                "repair_actions": [],
            }
            fail_report = {
                "passed": False,
                "score": 0.4,
                "recommended_stage": "scene_understanding",
                "repair_patches": [
                    {
                        "op": "set_lane_target",
                        "entity_id": "car_1",
                        "target_lane": "right_lane",
                        "severity": "high",
                    },
                ],
                "semantic_unrepairable": [
                    {"type": "count_mismatch", "severity": "high"}
                ],
            }
            pass_report = {
                "passed": True,
                "score": 0.9,
                "recommended_stage": "pass",
                "repair_actions": [],
            }
            with mock.patch.object(generator, "_capture_layout_images", return_value=capture), \
                mock.patch.object(generator, "_write_actor_graph_repair_plan", return_value=nonblocking_plan), \
                mock.patch.object(generator, "_verify_scene_round", side_effect=[fail_report, pass_report]), \
                mock.patch.object(
                    generator.scene_understanding_interpreter,
                    "revise_with_verification_feedback",
                    return_value={"revised": True},
                ) as revise_mock, \
                mock.patch.object(
                    generator,
                    "_rerun_structured_tail_no_remap",
                    return_value=(relation_dsl, {}, {}, "final2.py"),
                ), \
                mock.patch.object(
                    generator,
                    "_apply_actor_repair_patches",
                    return_value={
                        "applied_count": 1,
                        "no_op_count": 0,
                        "rejected_count": 0,
                        "blocked_count": 0,
                        "patch_outcomes": [],
                    },
                ) as repair_mock, \
                mock.patch.object(generator, "generate_final_scene_script", return_value="final2.py"):
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            self.assertEqual(summary["rounds"][0]["repair"], "actor_patch_repair")
            revise_mock.assert_not_called()
            repair_mock.assert_called_once()

    def test_actor_graph_mode_stops_on_projection_code_fix_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "actor_graph",
                    "verify_max_rounds": 2,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": None,
                "capture_mode": "ego_view",
                "error": None,
            }
            blocked_plan = {
                "passed": False,
                "score": 0.4,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "stop_repair_loop": True,
                "requires_code_fix": True,
                "issues": [
                    {
                        "issue_type": "systematic_projection_error",
                        "severity": "high",
                        "evidence": "repeated lane-side failure",
                    }
                ],
                "repair_actions": [],
            }
            with mock.patch.object(generator, "_capture_layout_images", return_value=capture), \
                mock.patch.object(generator, "_write_actor_graph_repair_plan", return_value=blocked_plan):
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            self.assertEqual(summary["rounds"][0]["repair"], "blocked_high_fact_issue")

    def test_actor_graph_mode_routes_geometry_actions_to_spawn_payload_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "actor_graph",
                    "verify_max_rounds": 2,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            capture = {
                "layout_image_path": str(Path(tmp) / "ego.png"),
                "ego_view_path": str(Path(tmp) / "ego.png"),
                "bev_path": None,
                "capture_mode": "ego_view",
                "error": None,
            }
            repair_plan = {
                "passed": False,
                "score": 0.6,
                "status": "failed",
                "truth_source": "carla_actor_transform",
                "stop_repair_loop": False,
                "issues": [
                    {
                        "issue_type": "heading_mismatch",
                        "severity": "medium",
                        "source_entity_id": "car_1",
                    }
                ],
                "repair_actions": [
                    {
                        "type": "heading_mismatch",
                        "entity_id": "car_1",
                        "severity": "medium",
                    }
                ],
            }
            pass_plan = {
                "passed": True,
                "score": 1.0,
                "status": "passed",
                "truth_source": "carla_actor_transform",
                "stop_repair_loop": False,
                "issues": [],
                "repair_actions": [],
            }
            with mock.patch.object(generator, "_capture_layout_images", return_value=capture), \
                mock.patch.object(
                    generator,
                    "_write_actor_graph_repair_plan",
                    side_effect=[repair_plan, pass_plan],
                ), \
                mock.patch.object(
                    generator,
                    "_apply_actor_repair_patches",
                    return_value={
                        "applied_count": 1,
                        "no_op_count": 0,
                        "rejected_count": 0,
                        "blocked_count": 0,
                        "patch_outcomes": [],
                    },
                ) as repair_mock, \
                mock.patch.object(generator, "generate_final_scene_script", return_value="final2.py"):
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            repair_mock.assert_called_once()
            self.assertEqual(summary["rounds"][0]["repair"], "actor_patch_repair")
            self.assertEqual(summary["rounds"][1]["repair"], "final_verification")

    def test_actor_graph_mode_repairs_static_overlap_without_carla_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(
                tmp,
                {
                    "enable_scene_verify": True,
                    "verify_mode": "actor_graph",
                    "verify_max_rounds": 1,
                    "require_carla_connection": False,
                },
            )
            scene_id, relation_dsl, spawn_payload, match_path = self._basic_inputs(tmp)
            relation_dsl["ego_relations"].append(
                {
                    **relation_dsl["ego_relations"][0],
                    "entity_id": "car_2",
                    "group_id": "car_2",
                }
            )
            spawn_payload["entities"].append(
                {
                    **spawn_payload["entities"][0],
                    "id": "car_2",
                    "location": {"x": 10.2, "y": 0.0, "z": 0.3},
                }
            )
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )

            with mock.patch.object(generator, "generate_final_scene_script", return_value="final2.py"):
                summary = generator.verify_and_repair_spawn_layout(
                    scene_id,
                    "source.jpg",
                    "",
                    {},
                    relation_dsl,
                    {},
                    match_path,
                    "final.py",
                )

            repaired = json.loads(Path(generator._spawn_payload_path(scene_id)).read_text())
            by_id = {entity["id"]: entity for entity in repaired["entities"]}
            self.assertEqual(
                summary["rounds"][0]["repair"], "blocked_high_fact_issue"
            )
            self.assertAlmostEqual(
                by_id["car_2"]["location"]["x"] - by_id["car_1"]["location"]["x"],
                0.2,
            )
            self.assertNotIn("repair_metadata", repaired)

    def test_pairwise_lateral_repairs_are_aggregated_per_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id, _relation_dsl, spawn_payload, _match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            validation = {
                "pairwise_results": [
                    {
                        "status": "fail",
                        "entity_id": "car_1",
                        "lateral_relation": "left_of_other",
                        "actual_lateral_delta_m": 0.5,
                    },
                    {
                        "status": "fail",
                        "entity_id": "car_1",
                        "lateral_relation": "left_of_other",
                        "actual_lateral_delta_m": 0.5,
                    },
                ]
            }

            result = generator._apply_layout_repair_actions(scene_id, validation, {})
            entity = result["entities"][0]
            repairs = result["repair_metadata"]["last_applied_repairs"]

            self.assertAlmostEqual(entity["location"]["y"], -2.0)
            self.assertEqual(
                [repair["type"] for repair in repairs],
                ["pairwise_lateral"],
            )
            self.assertEqual(repairs[0]["source_count"], 2)

    def test_pairwise_lateral_repair_blocks_when_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id, _relation_dsl, spawn_payload, _match_path = self._basic_inputs(tmp)
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            validation = {
                "pairwise_results": [
                    {
                        "status": "fail",
                        "entity_id": "car_1",
                        "lateral_relation": "left_of_other",
                        "actual_lateral_delta_m": 4.5,
                    }
                ]
            }

            result = generator._apply_layout_repair_actions(scene_id, validation, {})
            entity = result["entities"][0]
            metadata = result["repair_metadata"]

            self.assertAlmostEqual(entity["location"]["y"], 0.0)
            self.assertTrue(metadata["blocked_for_code_fix"])
            self.assertEqual(metadata["blocked_repairs"][0]["entity_id"], "car_1")

    def test_right_parking_actor_ignores_generic_offlane_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            scene_id, _relation_dsl, spawn_payload, _match_path = self._basic_inputs(tmp)
            entity = spawn_payload["entities"][0]
            entity.update(
                {
                    "lane_side_relation": "right_parking_lane",
                    "placement_mode_hint": "parking_lane_actor",
                    "placement_mode": "project_to_parking_lane",
                }
            )
            original_location = dict(entity["location"])
            Path(generator._spawn_payload_path(scene_id)).write_text(
                json.dumps(spawn_payload),
                encoding="utf-8",
            )
            report = {
                "repair_actions": [
                    {
                        "type": "off_lane_spawn",
                        "entity_id": "car_1",
                        "nearest_waypoint": {
                            "x": 99.0,
                            "y": 99.0,
                            "z": 0.0,
                            "yaw": 0.0,
                            "road_id": 76,
                            "lane_id": -1,
                        },
                    }
                ]
            }

            result = generator._apply_layout_repair_actions(scene_id, {}, report)
            repaired = result["entities"][0]

            self.assertEqual(repaired["location"], original_location)
            self.assertEqual(repaired["placement_mode"], "project_to_parking_lane")
            self.assertNotIn(
                "off_lane_snap",
                [
                    item["type"]
                    for item in result["repair_metadata"]["last_applied_repairs"]
                ],
            )

    def test_valid_rightmost_driving_fallback_is_not_offlane(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            generator.carla_spawn_context = {
                "dense_local_waypoints": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "road_id": 76, "lane_id": -2}
                ]
            }
            render_graph = {
                "actors": [
                    {
                        "id": "car_1",
                        "spawn_kind": "vehicle",
                        "spawned": True,
                        "location": {"x": 0.0, "y": 0.0, "z": 0.3},
                        "actual_waypoint": {"road_id": 76, "lane_id": -2},
                        "parking_projection_result": "rightmost_driving_fallback",
                        "parking_projection_lane": {"road_id": 76, "lane_id": -2},
                    }
                ]
            }
            spawn_payload = {
                "entities": [
                    {
                        "id": "car_1",
                        "spawn_kind": "vehicle",
                        "lane_side_relation": "right_parking_lane",
                    }
                ]
            }

            issues = generator._offlane_spawn_issues(
                render_graph,
                spawn_payload,
                {"lane_width_m": 3.5},
            )

            self.assertEqual(issues, [])

    def test_valid_real_parking_projection_is_not_offlane(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _make_generator(tmp)
            generator.carla_spawn_context = {
                "dense_local_waypoints": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "road_id": 76, "lane_id": -1}
                ]
            }
            render_graph = {
                "actors": [
                    {
                        "id": "car_1",
                        "spawn_kind": "vehicle",
                        "spawned": True,
                        "location": {"x": 0.0, "y": 3.5, "z": 0.3},
                        "actual_waypoint": {"road_id": 76, "lane_id": -2},
                        "parking_projection_result": "parking_lane",
                        "parking_projection_lane": {"road_id": 76, "lane_id": -2},
                    }
                ]
            }
            spawn_payload = {
                "entities": [
                    {
                        "id": "car_1",
                        "spawn_kind": "vehicle",
                        "lane_side_relation": "right_parking_lane",
                    }
                ]
            }

            issues = generator._offlane_spawn_issues(
                render_graph, spawn_payload, {"lane_width_m": 3.5}
            )

            self.assertEqual(issues, [])


class TestFallbackCarlaSpawnContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.generator = _make_generator(self.tmp)

    def test_returns_unavailable_status(self):
        ctx = self.generator._fallback_carla_spawn_context("test reason")
        self.assertEqual(ctx["status"], "unavailable")

    def test_contains_failure_reason(self):
        ctx = self.generator._fallback_carla_spawn_context("no carla")
        self.assertEqual(ctx["failure_reason"], "no carla")

    def test_spawn_points_is_empty_list(self):
        ctx = self.generator._fallback_carla_spawn_context("x")
        self.assertEqual(ctx["spawn_points"], [])

    def test_topology_sample_is_non_empty(self):
        ctx = self.generator._fallback_carla_spawn_context("x")
        self.assertIsInstance(ctx["topology_sample"], list)
        self.assertGreater(len(ctx["topology_sample"]), 0)


class TestExtractHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.generator = _make_generator(self.tmp)
        scene_file = Path(self.tmp) / "scene001.txt"
        scene_file.write_text(SPLIT_TEXT, encoding="utf-8")

    def test_extract_net_description(self):
        result = self.generator.extract_net_description("scene001")
        self.assertEqual(result, "A two-lane urban street.")

    def test_extract_scene_description(self):
        result = self.generator.extract_scene_description("scene001")
        self.assertIn("Ego approaches", result)

    def test_extract_scene_sections_returns_all_five(self):
        sections = self.generator.extract_scene_sections("scene001")
        self.assertIn("Road Net Description", sections)
        self.assertIn("Road Users Description", sections)
        self.assertIn("Static Objects Description", sections)
        self.assertIn("Vehicles' Locations and Behaviors", sections)
        self.assertIn("Scenario Description", sections)

    def test_extract_net_description_missing_section_returns_empty(self):
        bad_file = Path(self.tmp) / "bad.txt"
        bad_file.write_text("No sections here.", encoding="utf-8")
        result = self.generator.extract_net_description("bad")
        self.assertEqual(result, "")


class TestAdaptGenerationParams(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.generator = _make_generator(self.tmp)

    def test_default_lane_width_class_is_standard(self):
        params = self.generator.adapt_generation_params_for_scene(
            {"road_network": {"lane_groups": []}, "traffic_subjects": [], "background_traffic": []}
        )
        self.assertEqual(params["lane_width_class"], "standard")

    def test_lane_width_class_from_first_group(self):
        params = self.generator.adapt_generation_params_for_scene(
            {
                "road_network": {
                    "lane_groups": [{"lane_width_class": "narrow"}],
                },
                "traffic_subjects": [],
                "background_traffic": [],
            }
        )
        self.assertEqual(params["lane_width_class"], "narrow")

    def test_roadside_space_false_when_no_edge_relations(self):
        params = self.generator.adapt_generation_params_for_scene(
            {
                "road_network": {"lane_groups": []},
                "traffic_subjects": [{"id": "car", "lane_side_relation": "same_lane"}],
                "background_traffic": [],
            }
        )
        self.assertFalse(params["roadside_space"])

    def test_roadside_space_false_for_right_lane_relation(self):
        params = self.generator.adapt_generation_params_for_scene(
            {
                "road_network": {"lane_groups": []},
                "traffic_subjects": [],
                "background_traffic": [{"id": "scooter", "lane_side_relation": "right_lane"}],
            }
        )
        self.assertFalse(params["roadside_space"])

    def test_roadside_space_true_for_sidewalk_relation(self):
        params = self.generator.adapt_generation_params_for_scene(
            {
                "road_network": {"lane_groups": []},
                "traffic_subjects": [{"id": "ped", "lane_side_relation": "sidewalk_left"}],
                "background_traffic": [],
            }
        )
        self.assertTrue(params["roadside_space"])

    def test_params_stored_on_instance(self):
        self.generator.adapt_generation_params_for_scene(
            {"road_network": {"lane_groups": []}, "traffic_subjects": [], "background_traffic": []}
        )
        self.assertIsNotNone(self.generator.generation_params)


class TestPrependEgoVehicle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.generator = _make_generator(self.tmp)

    def test_ego_inserted_at_index_zero(self):
        coords = {
            "selected_anchor_lane": self.generator._fallback_topology_sample()[0],
            "entities": [
                {"id": "car1", "location": {"x": 5.0, "y": 0.0, "z": 0.3}}
            ],
        }
        self.generator._prepend_ego_vehicle(coords)
        self.assertEqual(coords["entities"][0]["id"], "ego")

    def test_ego_not_duplicated_when_already_present(self):
        coords = {
            "entities": [
                {"id": "ego", "location": {"x": 0.0, "y": 0.0, "z": 0.3}},
                {"id": "car1", "location": {"x": 5.0, "y": 0.0, "z": 0.3}},
            ]
        }
        self.generator._prepend_ego_vehicle(coords)
        ego_ids = [e["id"] for e in coords["entities"] if e["id"] == "ego"]
        self.assertEqual(len(ego_ids), 1)

    def test_ego_vehicle_name_variant_not_duplicated(self):
        coords = {"entities": [{"id": "ego_vehicle", "location": {}}]}
        self.generator._prepend_ego_vehicle(coords)
        self.assertEqual(len(coords["entities"]), 1)

    def test_ego_has_required_spawn_fields(self):
        coords = {"entities": [], "selected_anchor_lane": self.generator._fallback_topology_sample()[0]}
        self.generator._prepend_ego_vehicle(coords)
        ego = coords["entities"][0]
        for field in ("id", "category", "spawn_kind", "location", "rotation"):
            self.assertIn(field, ego)
        self.assertGreaterEqual(ego["location"]["z"], 0.3)

    def test_ego_z_clamps_to_minimum(self):
        anchor = {
            "road_id": 1,
            "lane_id": -1,
            "start": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "end": {"x": 10.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        }
        coords = {"entities": [], "selected_anchor_lane": anchor}
        self.generator._prepend_ego_vehicle(coords)
        self.assertGreaterEqual(coords["entities"][0]["location"]["z"], 0.3)


class TestApplyProjectedLayoutFromMatch(unittest.TestCase):
    def _entity(self, entity_id, x=0.0, y=0.0, z=0.3):
        return {
            "id": entity_id,
            "location": {"x": x, "y": y, "z": z},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        }

    def test_returns_false_when_not_matched(self):
        coords = {"entities": [self._entity("car1")]}
        result = AutoGenerator._apply_projected_layout_from_match(
            coords, {"status": "unmatched", "projected_layout": {}}
        )
        self.assertFalse(result)

    def test_returns_false_when_no_projected_layout(self):
        coords = {"entities": [self._entity("car1")]}
        result = AutoGenerator._apply_projected_layout_from_match(
            coords, {"status": "matched", "projected_layout": {}}
        )
        self.assertFalse(result)

    def test_applies_projected_location_and_rotation(self):
        coords = {"entities": [self._entity("car1")]}
        match_report = {
            "status": "matched",
            "projected_layout": {
                "car1": {
                    "projected_location": {"x": 50.0, "y": 60.0, "z": 0.5},
                    "projected_rotation": {"pitch": 0.0, "yaw": 45.0, "roll": 0.0},
                }
            },
        }
        result = AutoGenerator._apply_projected_layout_from_match(coords, match_report)
        self.assertTrue(result)
        self.assertEqual(coords["entities"][0]["location"]["x"], 50.0)
        self.assertEqual(coords["entities"][0]["rotation"]["yaw"], 45.0)

    def test_z_clamps_to_minimum_03(self):
        coords = {"entities": [self._entity("car1")]}
        match_report = {
            "status": "matched",
            "projected_layout": {
                "car1": {
                    "projected_location": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "projected_rotation": {},
                }
            },
        }
        AutoGenerator._apply_projected_layout_from_match(coords, match_report)
        self.assertGreaterEqual(coords["entities"][0]["location"]["z"], 0.3)

    def test_force_projected_layout_overrides_unmatched_status(self):
        coords = {"entities": [self._entity("car1")]}
        match_report = {
            "status": "unmatched",
            "projected_layout": {
                "car1": {
                    "projected_location": {"x": 99.0, "y": 0.0, "z": 0.5},
                    "projected_rotation": {},
                }
            },
        }
        result = AutoGenerator._apply_projected_layout_from_match(
            coords, match_report, force_projected_layout=True
        )
        self.assertTrue(result)
        self.assertEqual(coords["entities"][0]["location"]["x"], 99.0)


class TestWriteDebugJson(unittest.TestCase):
    def test_does_not_write_when_debug_artifacts_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp, {"debug_artifacts": False})
            gen._write_debug_json("s1", "test_suffix", {"key": "value"})
            self.assertFalse((Path(tmp) / "s1_test_suffix.json").exists())

    def test_writes_file_when_debug_artifacts_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp, {"debug_artifacts": True})
            gen._write_debug_json("s1", "test_suffix", {"key": "value"})
            out = Path(tmp) / "s1_test_suffix.json"
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["key"], "value")


class TestLoadJsonIfExists(unittest.TestCase):
    def test_returns_empty_dict_for_missing_path(self):
        result = AutoGenerator._load_json_if_exists("/nonexistent/path.json")
        self.assertEqual(result, {})

    def test_returns_empty_dict_for_none(self):
        result = AutoGenerator._load_json_if_exists(None)
        self.assertEqual(result, {})

    def test_returns_parsed_json_for_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            p.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
            result = AutoGenerator._load_json_if_exists(str(p))
            self.assertEqual(result["hello"], "world")


class TestWriteVerificationSkipped(unittest.TestCase):
    def test_writes_skipped_report_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp)
            report = gen._write_verification_skipped("scene1", 1, "CARLA unavailable")
            expected_path = Path(tmp) / "scene1_verify_r1.json"
            self.assertTrue(expected_path.exists())
            self.assertTrue(report["skipped"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["score"], 0.0)
            self.assertIn("CARLA unavailable", report["hard_failures"])

    def test_round_index_reflected_in_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp)
            gen._write_verification_skipped("s", 3, "err")
            self.assertTrue((Path(tmp) / "s_verify_r3.json").exists())


class TestVerifyAndRepairSpawnLayoutDisabled(unittest.TestCase):
    def test_returns_disabled_summary_when_verify_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp, {"enable_scene_verify": False})
            result = gen.verify_and_repair_spawn_layout(
                scene_id="s1",
                image_path="/fake/img.jpg",
                user_scene_description="",
                scene_understanding={},
                relation_dsl={},
                validation={},
                match_report_path="/fake/match.json",
                final_scene_path="/fake/scene.py",
            )
            self.assertFalse(result["enabled"])
            self.assertEqual(result["rounds"], [])

    def test_no_repair_summary_file_written_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp, {"enable_scene_verify": False})
            gen.verify_and_repair_spawn_layout(
                scene_id="s1",
                image_path="",
                user_scene_description="",
                scene_understanding={},
                relation_dsl={},
                validation={},
                match_report_path="",
                final_scene_path="",
            )
            self.assertFalse((Path(tmp) / "s1_spawn_repair.json").exists())


class TestAutoGeneratorInit(unittest.TestCase):
    def test_defaults_applied_when_info_dict_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = AutoGenerator(tmp)
            self.assertEqual(gen.input_type, "image")
            self.assertEqual(gen.carla_host, "localhost")
            self.assertEqual(gen.carla_port, 2000)
            self.assertTrue(gen.require_carla_connection)

    def test_output_folder_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "sub" / "output"
            AutoGenerator(str(folder), {"input_type": "image", "require_carla_connection": False})
            self.assertTrue(folder.exists())

    def test_carla_connection_disabled_gives_unavailable_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp)
            self.assertEqual(gen.carla_spawn_context["status"], "unavailable")
            self.assertIn("disabled", gen.carla_spawn_context["failure_reason"])

    def test_custom_spawn_point_limit_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = _make_generator(tmp, {"spawn_point_limit": 5})
            self.assertEqual(gen.spawn_point_limit, 5)


class TestPathHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = _make_generator(self.tmp)

    def test_scene_understanding_path(self):
        p = self.gen._scene_understanding_path("s1")
        self.assertTrue(p.endswith("s1_su.json"))

    def test_scene_match_path(self):
        p = self.gen._scene_match_path("s1")
        self.assertTrue(p.endswith("s1_match.json"))

    def test_spawn_payload_path(self):
        p = self.gen._spawn_payload_path("s1")
        self.assertTrue(p.endswith("s1_actors.json"))

    def test_final_scene_path(self):
        p = self.gen._final_scene_path("s1")
        self.assertTrue(p.endswith("s1_static.py"))

    def test_verification_path_includes_round(self):
        p = self.gen._verification_path("s1", 2)
        self.assertIn("r2", p)

    def test_spawn_layout_repair_summary_path(self):
        p = self.gen._spawn_layout_repair_summary_path("s1")
        self.assertTrue(p.endswith("s1_spawn_repair.json"))



if __name__ == "__main__":
    unittest.main()
