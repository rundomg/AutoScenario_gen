"""Stage 2a of the pure-LLM (no template library) risk experiment.

Given one predicted accident candidate (from ``LlmAccidentPredictor``) plus the
reconstructed executable actor table (``actors.json`` / spawn_payload), an LLM
emits a small, structured scenario DSL (``risk-dsl-v1``). Stage 2b
(``ExistingWorldScenarioGenerator.build_dsl_risk_scene_script``) then turns that
DSL into an executable CARLA Python script deterministically.

Unlike ``agents/risk_scenario_interpreter.py`` this agent is intentionally NOT
constrained by ``tools/accident_template_library.py``. The accident is described
by the upstream VLM in free text; this stage's only job is to map the predicted
participants onto the *real* spawned actor ids and express the accident as a list
of trigger/action events that the deterministic generator can run.

Mirrors the structure of ``agents/llm_scenic_generator.py`` (prior_attempt +
error fed back for a repair loop), but the repair signal is a DSL schema error
(from ``tools/risk_dsl.validate_risk_dsl``) rather than a Scenic compiler error.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent


# Concise DSL reference so the LLM does not have to guess the schema. Every
# trigger/action type below is one the deterministic Stage-2b generator can
# execute (see tools/risk_dsl.py TRIGGER_TYPES / ACTION_TYPES).
DSL_TUTORIAL = """
The risk DSL is a JSON object that describes ONE accident as the ego drives
forward. Schema:

{
  "schema_version": "risk-dsl-v1",
  "scene_id": "<id>",
  "accident_type": "<free text, copy from the prediction>",
  "ego": {"actor_id": "ego", "target_speed_mps": <float>, "behavior": "drive_forward"},
  "actors": [{"actor_id": "<real id from the actor table>", "role": "<short label>"}],
  "events": [
    {"actor_id": "<real non-ego id>",
     "trigger": {...},
     "action": {...}}
  ],
  "duration_s": <float>,
  "metrics": ["collision", "min_ttc_s", "min_distance_m"]
}

trigger.type is exactly one of:
  - {"type": "immediate"}                              fires from t=0
  - {"type": "time_elapsed_above", "value_s": <float>} fires after value_s seconds
  - {"type": "distance_to_ego_below", "value_m": <float>} fires when the actor is
    within value_m metres of the ego

action.type is exactly one of:
  - {"type": "brake", "intensity": <0..1>}             hard/soft braking
  - {"type": "set_speed", "speed_mps": <float>}        hold a constant speed
  - {"type": "accelerate", "speed_mps": <float>}       speed up to a target speed
  - {"type": "steer", "steer": <-1..1>, "throttle": <0..1>, "duration_s": <float>}
    lateral push / cut-in; after duration_s the actor follows the lane again
  - {"type": "cross", "steer": <-1..1>, "throttle": <0..1>, "duration_s": <float>}
    cross the ego's path; after duration_s the actor follows the lane again
  - {"type": "stop"}                                   come to a stop

Rules:
- Use ONLY actor ids that appear in the actor table. Map each predicted
  participant to the table row whose relative_to_ego geometry and category best
  match it (e.g. a "lead vehicle ahead" maps to the actor with the largest
  positive longitudinal_m and small lateral_m).
- "ego" is the only ego; events must target non-ego actors.
- Keep ego.target_speed_mps at the given fixed speed unless told otherwise.
- Prefer time_elapsed_above for all collision-onset events. Do not use
  distance_to_ego_below for lead-vehicle braking, cut-in, or sideswipe events:
  choose a time delay that lets the actors start moving before the risky action.
- For steer/cross actions, set duration_s around 0.8-1.5 seconds so the actor
  performs a lane change or encroachment, then lets the controller recover
  along the lane instead of continuously turning.
- Output ONLY the JSON object. No markdown fences, no commentary.
""".strip()


EXAMPLE_DSL = """
{
  "schema_version": "risk-dsl-v1",
  "scene_id": "s0000_c0",
  "accident_type": "rear-end",
  "ego": {"actor_id": "ego", "target_speed_mps": 10.0, "behavior": "drive_forward"},
  "actors": [{"actor_id": "ts1", "role": "lead_vehicle"}],
  "events": [
    {"actor_id": "ts1",
     "trigger": {"type": "time_elapsed_above", "value_s": 1.0},
     "action": {"type": "brake", "intensity": 0.9}}
  ],
  "duration_s": 12.0,
  "metrics": ["collision", "min_ttc_s", "min_distance_m"]
}
""".strip()


class LlmRiskDslGenerator(TaskAgent):
    """LLM that writes a ``risk-dsl-v1`` document for one predicted accident."""

    def __init__(self, ego_speed_mps: float = 10.0) -> None:
        super().__init__()
        self.ego_speed_mps = float(ego_speed_mps)

    def generate(
        self,
        *,
        scene_id: str,
        candidate: Dict[str, Any],
        actor_context: Dict[str, Any],
        prior_attempt: Optional[str] = None,
        schema_error: Optional[str] = None,
    ) -> str:
        add_info = {
            "scene_id": scene_id,
            "candidate": candidate,
            "actor_context": actor_context,
            "prior_attempt": prior_attempt,
            "schema_error": schema_error,
            "request_label": "Risk DSL generation",
            "request_timeout": 180,
            "request_max_tokens": 3000,
        }
        response = self.send_request("", add_info)
        return self._strip_json(response)

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing risk DSL generator add_info."
        scene_id = str(add_info.get("scene_id") or "")
        candidate = add_info.get("candidate") or {}
        actor_context = add_info.get("actor_context") or {}

        prompt = (
            "You convert ONE predicted accident into an executable risk DSL on a "
            "reconstructed scene.\n\n"
            f"The ego drives forward at a fixed {self.ego_speed_mps:.1f} m/s unless "
            "it must react.\n\n"
            f"DSL reference:\n{DSL_TUTORIAL}\n\n"
            f"Example DSL (style reference, different scene):\n{EXAMPLE_DSL}\n\n"
            "Actor table (the real spawned actors with ego-relative geometry; "
            "longitudinal_m ahead/+, lateral_m right/+). Use these ids:\n"
            f"{json.dumps(actor_context, indent=2, sort_keys=True)}\n\n"
            "Predicted accident to reproduce:\n"
            f"{json.dumps(candidate, indent=2, sort_keys=True)}\n\n"
            "Instructions:\n"
            f'- Set scene_id to "{scene_id}".\n'
            f"- Set ego.target_speed_mps to {self.ego_speed_mps:.1f}.\n"
            "- Map the predicted participants to the real actor ids in the table "
            "and list them in `actors`.\n"
            "- Express the accident mechanism as one or more `events` whose "
            "trigger/action types come ONLY from the reference above.\n"
            "- Use `time_elapsed_above` triggers for lead-vehicle braking, lane "
            "cut-in, and adjacent-lane sideswipe events. Avoid distance triggers "
            "because they can fire immediately when the reconstructed actors are "
            "already close at t=0.\n"
            "- Output ONLY the JSON object, no fences, no commentary.\n"
        )

        prior_attempt = add_info.get("prior_attempt")
        schema_error = add_info.get("schema_error")
        if prior_attempt and schema_error:
            prompt += (
                "\nYour previous DSL failed validation. Fix it.\n"
                "Previous DSL:\n"
                f"{prior_attempt}\n\n"
                "Validation error:\n"
                f"{schema_error}\n\n"
                "Output a corrected DSL that passes. Preserve the same actors and "
                "accident intent.\n"
            )

        return [{"type": "text", "text": prompt}]

    @staticmethod
    def _strip_json(response: str) -> str:
        if not isinstance(response, str):
            return ""
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        if not (text.startswith("{") and text.endswith("}")):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                text = text[start : end + 1]
        return text.strip()
