import ast
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

from tools.risk_scenario_runner import RiskScenarioRunner


SPAWN_PAYLOAD = {
    "entities": [
        {
            "id": "ego",
            "category": "car",
            "spawn_kind": "vehicle",
            "blueprint_name": "car",
            "color": "54,116,168",
            "lane_side_relation": "same_lane",
            "placement_mode": "project_to_lane",
            "location": {"x": 0.0, "y": 0.0, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        {
            "id": "ts1",
            "category": "car",
            "spawn_kind": "vehicle",
            "blueprint_name": "car",
            "color": "255,255,255",
            "lane_side_relation": "same_lane",
            "placement_mode": "project_to_lane",
            "location": {"x": 10.0, "y": 0.0, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
    ]
}


RISK_SPEC = {
    "schema_version": "risk-scenario-v1",
    "source_scene_id": "s0000_c0",
    "ego": {
        "actor_id": "ego",
        "controller_mode": "scripted",
        "route_source": "map_forward_waypoints",
        "behavior_profile": "accident_reproduction",
    },
    "risk_candidates": [
        {
            "id": "risk_001",
            "template_id": "lead_vehicle_hard_brake",
            "involved_actor_ids": ["ego", "ts1"],
            "confidence": 0.8,
            "parameter_ranges": {
                "ego_target_speed_mps": [6.0, 6.0],
                "ego_reaction_delay_s": [1.0, 1.0],
                "npc_trigger_distance_m": [10.0, 10.0],
                "npc_brake_intensity": [0.9, 0.9],
            },
        },
        {
            "id": "risk_002",
            "template_id": "door_opening_obstacle",
            "involved_actor_ids": ["ego", "ts1"],
            "confidence": 0.5,
            "parameter_ranges": {
                "ego_target_speed_mps": [6.0, 6.0],
                "ego_reaction_delay_s": [1.0, 1.0],
                "npc_trigger_distance_m": [10.0, 10.0],
                "npc_target_speed_mps": [0.0, 0.0],
                "npc_steer_intensity": [0.2, 0.2],
            },
        },
    ],
    "risk_evidence": {
        "main_objects": [
            {
                "actor_id": "ts1",
                "role": "lead_vehicle",
                "visual_description": "vehicle ahead",
                "matched_from_actor_table": True,
            }
        ],
        "spatial_relations": [
            {
                "subject_actor_id": "ts1",
                "relation": "ahead_of",
                "object_actor_id": "ego",
                "evidence": "positive longitudinal offset",
            }
        ],
        "event_hypotheses": [
            {
                "event": "lead vehicle may brake",
                "actors": ["ego", "ts1"],
                "evidence": "same lane lead actor",
            }
        ],
        "missing_or_uncertain_facts": [],
        "template_mapping": [
            {
                "template_id": "lead_vehicle_hard_brake",
                "actors": ["ego", "ts1"],
                "why_template_fits": "rear-end risk",
                "why_executable": "actors exist",
            }
        ],
    },
    "unsupported_risk_hypotheses": [
        {"description": "falling sign", "reason": "not in catalog"}
    ],
}


RISK_SPEC_WITH_SPEEDS = json.loads(json.dumps(RISK_SPEC))
RISK_SPEC_WITH_SPEEDS["ego"]["speed_hypotheses_mps"] = {
    "low": 4.0,
    "medium": 8.0,
    "high": 12.0,
}


def write_static_outputs(tmp: str) -> str:
    scene_id = "s0000_c0"
    base_id = "s0000"
    root = Path(tmp)
    (root / f"{scene_id}_actors.json").write_text(
        json.dumps(SPAWN_PAYLOAD),
        encoding="utf-8",
    )
    (root / f"{scene_id}_match.json").write_text(
        json.dumps({"status": "matched", "world_name": "Carla/Maps/Town06"}),
        encoding="utf-8",
    )
    (root / f"{base_id}_su.json").write_text(
        json.dumps({"metadata": {"scene_type": "test"}}),
        encoding="utf-8",
    )
    return scene_id


class TestRiskScenarioRunner(unittest.TestCase):
    def test_resolves_base_scene_understanding_for_candidate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            runner = RiskScenarioRunner(tmp, scene_id)
            self.assertTrue(runner._scene_understanding_path().endswith("s0000_su.json"))

    def test_risk_spec_mode_generates_samples_and_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            risk_spec_path = Path(tmp) / "custom_risk_scenario_spec.json"
            risk_spec_path.write_text(json.dumps(RISK_SPEC), encoding="utf-8")
            runner = RiskScenarioRunner(
                tmp,
                scene_id,
                samples_per_candidate=2,
                seed=11,
            )
            summary = runner.run(risk_spec_path=str(risk_spec_path))
            self.assertEqual(len(summary["artifacts"]), 4)
            self.assertEqual(summary["speed_hypotheses_source"], "fixed_10mps")
            self.assertTrue(Path(summary["risk_evidence_path"]).exists())
            self.assertEqual(summary["risk_evidence_warnings"], [])
            implemented = [
                artifact
                for artifact in summary["artifacts"]
                if artifact["execution_status"] == "implemented"
            ]
            planned = [
                artifact
                for artifact in summary["artifacts"]
                if artifact["execution_status"] == "planned_not_implemented"
            ]
            self.assertEqual(len(implemented), 2)
            self.assertEqual(len(planned), 2)
            self.assertIsNone(planned[0]["script_path"])
            self.assertEqual(implemented[0]["ego_speed_label"], "fixed")
            self.assertEqual(implemented[0]["ego_target_speed_mps"], 10.0)
            script = Path(implemented[0]["script_path"]).read_text(encoding="utf-8")
            ast.parse(script)
            self.assertIn("class EgoController", script)
            self.assertIn("class VLAControllerAdapter", script)
            self.assertIn("class NpcRiskController", script)
            self.assertIn("role_name = 'hero'", script)
            self.assertIn("actor_by_id", script)
            self.assertIn("def _autoscenario_hold_vehicle_stationary", script)
            self.assertIn("set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))", script)
            self.assertIn("hand_brake=True", script)
            self.assertIn("def _autoscenario_apply_target_velocity", script)
            self.assertIn("AUTOSCENARIO_USE_VELOCITY_CONTROL", script)
            self.assertIn("math.cos(yaw_radians) * speed", script)
            self.assertLess(
                script.index("_autoscenario_settle_ticks"),
                script.index("_bg_actor.set_autopilot(True"),
            )
            sample_name = Path(implemented[0]["sample_path"]).name
            self.assertIn("r000", sample_name)
            self.assertIn("s000", sample_name)
            self.assertEqual(len(summary["unsupported_risk_hypotheses"]), 1)

    def test_vlm_mode_requires_image_and_passes_it_to_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            image_path = Path(tmp) / "ego.jpg"
            image_path.write_bytes(b"fake-jpeg")
            runner = RiskScenarioRunner(
                tmp,
                scene_id,
                samples_per_candidate=1,
                seed=11,
            )
            captured = {}

            def fake_call_agent(user_request, added_info):
                captured.update(added_info)
                return RISK_SPEC_WITH_SPEEDS

            runner.interpreter.call_agent = fake_call_agent
            summary = runner.run(image_path=str(image_path))
            self.assertEqual(len(summary["artifacts"]), 2)
            self.assertNotIn("actor_bev_path", summary)
            self.assertEqual(captured["image_path"], str(image_path))
            self.assertNotIn("bev_path", captured)
            self.assertNotIn("actor_bev_path", captured)
            self.assertIn("spawn_payload", captured)
            self.assertIn("actor_context", captured)

    def test_vlm_mode_rejects_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_id = write_static_outputs(tmp)
            runner = RiskScenarioRunner(tmp, scene_id, samples_per_candidate=1)
            with self.assertRaisesRegex(ValueError, "image_path is required"):
                runner.run()

    def test_risk_output_folder_keeps_static_folder_clean(self):
        with tempfile.TemporaryDirectory() as static_tmp:
            with tempfile.TemporaryDirectory() as risk_tmp:
                scene_id = write_static_outputs(static_tmp)
                risk_spec_path = Path(static_tmp) / "custom_risk_scenario_spec.json"
                risk_spec_path.write_text(json.dumps(RISK_SPEC), encoding="utf-8")
                runner = RiskScenarioRunner(
                    static_tmp,
                    scene_id,
                    risk_output_folder=risk_tmp,
                    samples_per_candidate=1,
                    seed=11,
                )
                summary = runner.run(risk_spec_path=str(risk_spec_path))
                self.assertEqual(summary["static_output_folder"], static_tmp)
                self.assertEqual(summary["risk_output_folder"], risk_tmp)
                self.assertTrue((Path(risk_tmp) / f"{scene_id}_risk_spec.json").exists())
                self.assertTrue((Path(risk_tmp) / f"{scene_id}_risk_summary.json").exists())
                self.assertFalse((Path(static_tmp) / f"{scene_id}_risk_spec.json").exists())
                implemented = [
                    artifact
                    for artifact in summary["artifacts"]
                    if artifact["execution_status"] == "implemented"
                ]
                script = Path(implemented[0]["script_path"]).read_text(encoding="utf-8")
                self.assertIn(str(Path(static_tmp) / f"{scene_id}_actors.json"), script)
                self.assertIn(str(Path(risk_tmp) / f"{scene_id}_r000_s000.json"), script)


if __name__ == "__main__":
    unittest.main()
