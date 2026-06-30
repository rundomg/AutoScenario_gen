import base64
import json
import mimetypes
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent
from tools.accident_template_library import template_summary_for_prompt
from tools.risk_scenario_pipeline import (
    RISK_SCHEMA_VERSION,
    build_actor_context,
    require_risk_evidence,
    validate_risk_scenario_spec,
)
from tools.utils import read_file, write_to_file


class RiskScenarioInterpreter(TaskAgent):
    """VLM interpreter that maps a reconstructed static scene to risk templates."""

    def __init__(self) -> None:
        super().__init__()
        template_summary = template_summary_for_prompt()
        self.pre_prompt = f"""
        You convert a reconstructed CARLA static scene and its original camera image into
        ego-related risk scenario templates.

        Use CARLA actors and scene match artifacts as the executable source of truth.
        Use the raw image only to identify salient road users and visual traffic cues.
        Do not use the raw image to override CARLA actor positions or invent executable
        actors. Do not use BEV renders.

        Output only JSON. Do not output markdown fences unless absolutely necessary.
        The output schema_version must be "{RISK_SCHEMA_VERSION}".

        Important ego rule:
        - The ego vehicle is the reconstructed CARLA actor with id "ego".
        - Every risk candidate must directly involve "ego".
        - Do not choose a visible non-ego vehicle as the ego vehicle.

        You must choose template_id values only from this accident template library.
        Do not invent accident types, template ids, trajectories, or CARLA code.

        Accident template library:
        {template_summary}

        Top-level JSON schema:
        {{
          "schema_version": "risk-scenario-v1",
          "source_scene_id": "...",
          "ego": {{
            "actor_id": "ego",
            "controller_mode": "scripted",
            "route_source": "map_forward_waypoints",
            "behavior_profile": "accident_reproduction"
          }},
          "risk_evidence": {{
            "main_objects": [
              {{
                "actor_id": "ts1",
                "role": "lead_vehicle",
                "visual_description": "brief visual cue",
                "matched_from_actor_table": true
              }}
            ],
            "spatial_relations": [
              {{
                "subject_actor_id": "ts1",
                "relation": "ahead_of",
                "object_actor_id": "ego",
                "evidence": "positive longitudinal offset in actor table"
              }}
            ],
            "event_hypotheses": [
              {{
                "event": "lead vehicle may brake hard",
                "actors": ["ego", "ts1"],
                "evidence": "same-lane lead actor and traffic context"
              }}
            ],
            "missing_or_uncertain_facts": [
              {{
                "fact": "future lead vehicle braking intensity",
                "impact": "affects collision timing",
                "modeled_as": "parameter_range"
              }}
            ],
            "template_mapping": [
              {{
                "template_id": "lead_vehicle_hard_brake",
                "actors": ["ego", "ts1"],
                "why_template_fits": "lead actor is ahead of ego",
                "why_executable": "both actor ids exist in actor table"
              }}
            ]
          }},
          "risk_candidates": [
            {{
              "id": "risk_001",
              "accident_type": "rear_end",
              "template_id": "lead_vehicle_hard_brake",
              "involved_actor_ids": ["ego", "ts1"],
              "confidence": 0.0,
              "rationale": "brief visual/geometric reason",
              "preconditions": ["brief condition"],
              "parameter_ranges": {{
                "ego_reaction_delay_s": [1.0, 3.0],
                "npc_trigger_distance_m": [8.0, 18.0],
                "npc_brake_intensity": [0.6, 1.0]
              }}
            }}
          ],
          "unsupported_risk_hypotheses": [
            {{
              "description": "brief description of a plausible risk not covered by the template library",
              "reason": "why no available template fits"
            }}
          ]
        }}

        Rules:
        - Produce 1 to 5 risk_candidates.
        - Prefer candidates that are physically executable from the current CARLA actor layout.
        - involved_actor_ids must use ids from the provided actor table.
        - Do not include ego.speed_hypotheses_mps. The pipeline will set ego speed
          to a fixed 10.0 m/s for every risk candidate.
        - risk_evidence is required. Before risk_candidates, explain main_objects,
          spatial_relations, event_hypotheses, missing_or_uncertain_facts, and
          template_mapping using actor ids from the actor table whenever an executable
          actor is referenced.
        - Do not include ego_target_speed_mps in risk candidate parameter_ranges; the pipeline
          will inject a fixed 10.0 m/s ego speed for each candidate.
        - Use conservative confidence when the evidence is ambiguous.
        - Parameters are ranges, not single deterministic values.
        - Use parameter_ranges, not free-form trajectory descriptions.
        - Use unsupported_risk_hypotheses when a plausible ego-related risk cannot be
          mapped to existing actor ids or does not fit the catalog.
        - Do not output CARLA Python code.
        """

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing risk scenario add_info."
        scene_id = str(add_info.get("scene_id") or "")
        spawn_payload = add_info.get("spawn_payload") or {}
        actor_context = add_info.get("actor_context") or build_actor_context(spawn_payload)
        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nUser request:\n{user_request}"
        prompt += (
            "\n\nsource_scene_id:\n"
            f"{scene_id}\n\n"
            "Actor table from CARLA spawn payload:\n"
            f"{json.dumps(actor_context, indent=2, sort_keys=True)}\n\n"
            "Scene understanding summary:\n"
            f"{json.dumps(add_info.get('scene_understanding') or {}, indent=2, sort_keys=True)[:12000]}\n\n"
            "Scene match summary:\n"
            f"{json.dumps(add_info.get('scene_match') or {}, indent=2, sort_keys=True)[:8000]}\n\n"
            "CARLA spawn payload:\n"
            f"{json.dumps(spawn_payload, indent=2, sort_keys=True)[:12000]}\n"
        )

        messages = [{"type": "text", "text": prompt}]
        image_path = add_info.get("image_path")
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{self._mime_type(image_path)};base64,"
                            f"{self._encode_image(image_path)}"
                        )
                    },
                }
            )

        return messages

    def call_agent(self, user_request, added_info):
        output_fn = added_info["output_fn"]
        spawn_payload = added_info.get("spawn_payload") or {}
        attempts = 0
        while True:
            self.send_request(
                user_request,
                {
                    **added_info,
                    "request_label": "Risk scenario interpretation",
                    "request_timeout": 180,
                    "request_max_tokens": 8000,
                },
            )
            payload, validation_error = self.extract_decision_data(
                output_fn,
                spawn_payload,
                require_speed_hypotheses_flag=False,
                require_risk_evidence_flag=True,
            )
            attempts += 1
            if validation_error is None:
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Risk scenario interpretation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            print(f"Regenerating risk scenario interpretation... Attempt {attempts + 1}")

    def extract_decision_data(
        self,
        file_path: str,
        spawn_payload: Dict[str, Any],
        require_speed_hypotheses_flag: bool = False,
        require_risk_evidence_flag: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        text = read_file(file_path)
        payload, parse_error = self._parse_json_object(text)
        if parse_error:
            return None, parse_error
        if require_speed_hypotheses_flag:
            pass
        if require_risk_evidence_flag:
            evidence_error = require_risk_evidence(payload)
            if evidence_error:
                return None, evidence_error
        return validate_risk_scenario_spec(payload, spawn_payload)

    @staticmethod
    def _parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(text, str) or not text.strip():
            return None, "Risk scenario output is empty."
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"Risk scenario output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Risk scenario output must be a JSON object."
        return payload, None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_type(image_path: str) -> str:
        mime_type, _encoding = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"
