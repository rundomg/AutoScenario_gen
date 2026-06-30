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

from agents.risk_scenario_interpreter import RiskScenarioInterpreter
from tools.accident_template_library import all_templates, executable_template_ids
from tools.risk_scenario_pipeline import (
    build_actor_context,
    expand_candidates_by_speed,
    sample_all_risk_scenarios,
    sample_risk_scenario,
    validate_risk_scenario_spec,
)


SPAWN_PAYLOAD = {
    "entities": [
        {
            "id": "ego",
            "category": "car",
            "spawn_kind": "vehicle",
            "lane_side_relation": "same_lane",
            "location": {"x": 0.0, "y": 0.0, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        {
            "id": "ts1",
            "category": "car",
            "spawn_kind": "vehicle",
            "lane_side_relation": "same_lane",
            "location": {"x": 10.0, "y": 0.0, "z": 0.3},
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
    ]
}


VALID_SPEC = {
    "schema_version": "risk-scenario-v1",
    "source_scene_id": "scene1",
    "ego": {
        "actor_id": "ego",
        "controller_mode": "scripted",
        "route_source": "map_forward_waypoints",
        "behavior_profile": "accident_reproduction",
    },
    "risk_candidates": [
        {
            "id": "risk_001",
            "accident_type": "rear_end",
            "template_id": "lead_vehicle_hard_brake",
            "involved_actor_ids": ["ego", "ts1"],
            "confidence": 0.9,
            "parameter_ranges": {
                "ego_target_speed_mps": [6.0, 13.0],
                "ego_reaction_delay_s": [1.0, 3.0],
                "npc_trigger_distance_m": [8.0, 18.0],
                "npc_brake_intensity": [0.6, 1.0],
            },
        }
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
    "unsupported_risk_hypotheses": [],
}


class TestAccidentTemplateLibrary(unittest.TestCase):
    def test_template_ids_are_unique_and_match_keys(self):
        templates = all_templates()
        ids = [template["template_id"] for template in templates.values()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(templates))

    def test_every_template_has_required_catalog_fields(self):
        for template in all_templates().values():
            for field in (
                "family",
                "external_refs",
                "vlm_selection_cues",
                "required_actor_roles",
                "preconditions",
                "parameter_ranges",
                "controller",
                "metrics",
            ):
                self.assertIn(field, template)
            self.assertIsInstance(template["external_refs"], dict)
            self.assertIsInstance(template["parameter_ranges"], dict)

    def test_executable_template_subset_is_non_empty(self):
        self.assertIn("lead_vehicle_hard_brake", executable_template_ids())


class TestRiskScenarioPipeline(unittest.TestCase):
    def test_validate_accepts_valid_spec(self):
        normalized, error = validate_risk_scenario_spec(VALID_SPEC, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(normalized["ego"]["actor_id"], "ego")
        self.assertEqual(normalized["risk_candidates"][0]["family"], "rear_end")
        self.assertEqual(
            normalized["risk_candidates"][0]["execution_status"],
            "implemented",
        )

    def test_validate_rejects_unknown_template(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["risk_candidates"][0]["template_id"] = "made_up_template"
        _, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIn("Unknown risk template", error)

    def test_validate_requires_ego_in_candidate(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["risk_candidates"][0]["involved_actor_ids"] = ["ts1"]
        _, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIn("must include `ego`", error)

    def test_validate_rejects_unknown_actor_id(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["risk_candidates"][0]["involved_actor_ids"] = ["ego", "missing"]
        _, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIn("unknown actor ids", error)

    def test_sample_is_within_parameter_bounds(self):
        normalized, _ = validate_risk_scenario_spec(VALID_SPEC, SPAWN_PAYLOAD)
        sample = sample_risk_scenario(normalized, sample_index=0, seed=4)
        params = sample["sampled_parameters"]
        self.assertGreaterEqual(params["ego_target_speed_mps"], 6.0)
        self.assertLessEqual(params["ego_target_speed_mps"], 13.0)
        self.assertEqual(sample["risk_actor_id"], "ts1")
        self.assertEqual(sample["candidate_index"], 0)
        self.assertEqual(sample["family"], "rear_end")

    def test_sample_all_generates_each_candidate_times_each_sample(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        second = json.loads(json.dumps(spec["risk_candidates"][0]))
        second["id"] = "risk_002"
        second["template_id"] = "adjacent_vehicle_cut_in"
        second["parameter_ranges"] = {
            "ego_target_speed_mps": [6.0, 6.0],
            "ego_reaction_delay_s": [1.0, 1.0],
            "npc_trigger_distance_m": [8.0, 8.0],
            "npc_target_speed_mps": [4.0, 4.0],
            "npc_steer_intensity": [0.2, 0.2],
            "npc_cut_in_duration_s": [2.0, 2.0],
        }
        spec["risk_candidates"].append(second)
        normalized, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        samples = sample_all_risk_scenarios(
            normalized,
            samples_per_candidate=2,
            seed=10,
        )
        self.assertEqual(len(samples), 4)
        self.assertEqual([sample["candidate_index"] for sample in samples], [0, 0, 1, 1])
        self.assertEqual([sample["sample_index"] for sample in samples], [0, 1, 0, 1])

    def test_expand_candidates_by_speed_uses_fixed_10mps_for_old_spec(self):
        normalized, error = validate_risk_scenario_spec(VALID_SPEC, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        expanded = expand_candidates_by_speed(normalized)
        self.assertEqual(expanded["speed_hypotheses_source"], "fixed_10mps")
        self.assertEqual(expanded["ego"]["target_speed_mps"], 10.0)
        self.assertNotIn("speed_hypotheses_mps", expanded["ego"])
        self.assertEqual(len(expanded["risk_candidates"]), 1)
        labels = [candidate["ego_speed_label"] for candidate in expanded["risk_candidates"]]
        self.assertEqual(labels, ["fixed"])
        speeds = [
            candidate["parameter_ranges"]["ego_target_speed_mps"]
            for candidate in expanded["risk_candidates"]
        ]
        self.assertEqual(speeds, [[10.0, 10.0]])

    def test_sample_exposes_speed_fields_after_expansion(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["ego"]["speed_hypotheses_mps"] = {
            "low": 3.0,
            "medium": 7.0,
            "high": 11.0,
        }
        normalized, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        expanded = expand_candidates_by_speed(normalized)
        sample = sample_risk_scenario(expanded, sample_index=0, seed=4, candidate_index=0)
        self.assertEqual(sample["ego_speed_label"], "fixed")
        self.assertEqual(sample["ego_target_speed_mps"], 10.0)
        self.assertEqual(sample["sampled_parameters"]["ego_target_speed_mps"], 10.0)
        self.assertEqual(sample["source_candidate_index"], 0)

    def test_unsupported_hypotheses_do_not_block_validation(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["unsupported_risk_hypotheses"] = [
            {"description": "falling tree", "reason": "not in vehicle template catalog"}
        ]
        normalized, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(len(normalized["unsupported_risk_hypotheses"]), 1)

    def test_risk_evidence_unknown_actor_is_warning_not_rejection(self):
        spec = json.loads(json.dumps(VALID_SPEC))
        spec["risk_evidence"]["event_hypotheses"][0]["actors"].append("ghost_actor")
        normalized, error = validate_risk_scenario_spec(spec, SPAWN_PAYLOAD)
        self.assertIsNone(error)
        self.assertIn("ghost_actor", normalized["risk_evidence_warnings"][0])

    def test_actor_context_contains_ego_relative_rows(self):
        context = build_actor_context(SPAWN_PAYLOAD)
        ego_row = context["actors"][0]
        self.assertEqual(ego_row["id"], "ego")
        ts1 = [row for row in context["actors"] if row["id"] == "ts1"][0]
        self.assertEqual(ts1["relative_to_ego"]["longitudinal_m"], 10.0)


class TestRiskScenarioInterpreter(unittest.TestCase):
    def test_extract_decision_data_accepts_fenced_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "risk.txt"
            output.write_text(
                "```json\n" + json.dumps(VALID_SPEC) + "\n```",
                encoding="utf-8",
            )
            payload, error = RiskScenarioInterpreter().extract_decision_data(
                str(output),
                SPAWN_PAYLOAD,
            )
            self.assertIsNone(error)
            self.assertEqual(payload["risk_candidates"][0]["template_id"], "lead_vehicle_hard_brake")

    def test_refine_request_includes_raw_image_when_provided(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "ego.jpg"
            image_path.write_bytes(b"fake-jpeg")
            messages = RiskScenarioInterpreter().refine_request(
                "",
                {
                    "scene_id": "s0000_c0",
                    "image_path": str(image_path),
                    "spawn_payload": SPAWN_PAYLOAD,
                    "actor_context": build_actor_context(SPAWN_PAYLOAD),
                    "scene_understanding": {"summary": "reconstructed scene"},
                    "scene_match": {"world_name": "Town10HD_Opt"},
                },
            )
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["type"], "text")
        self.assertEqual(messages[1]["type"], "image_url")
        prompt = messages[0]["text"]
        self.assertIn("CARLA spawn payload", prompt)
        self.assertIn("Use the raw image only", prompt)
        self.assertIn("fixed 10.0 m/s", prompt)
        self.assertIn("risk_evidence", prompt)
        self.assertIn("image_url", json.dumps(messages))

    def test_extract_decision_data_ignores_legacy_speed_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "risk.txt"
            output.write_text(json.dumps(VALID_SPEC), encoding="utf-8")
            payload, error = RiskScenarioInterpreter().extract_decision_data(
                str(output),
                SPAWN_PAYLOAD,
                require_speed_hypotheses_flag=True,
            )
            self.assertIsNone(error)
            self.assertEqual(payload["risk_candidates"][0]["id"], "risk_001")

    def test_extract_decision_data_requires_risk_evidence_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "risk.txt"
            spec = json.loads(json.dumps(VALID_SPEC))
            spec["ego"]["speed_hypotheses_mps"] = {
                "low": 4.0,
                "medium": 8.0,
                "high": 12.0,
            }
            spec.pop("risk_evidence", None)
            output.write_text(json.dumps(spec), encoding="utf-8")
            _payload, error = RiskScenarioInterpreter().extract_decision_data(
                str(output),
                SPAWN_PAYLOAD,
                require_speed_hypotheses_flag=True,
                require_risk_evidence_flag=True,
            )
            self.assertIn("risk_evidence", error)


if __name__ == "__main__":
    unittest.main()
