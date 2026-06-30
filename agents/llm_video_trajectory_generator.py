"""Stage B of the dynamic-reconstruction pipeline.

Given the structured video understanding (``VideoAccidentInterpreter``) plus the
reconstructed actor table (``actors.json`` / spawn_payload), an LLM emits a
``video-trajectory-dsl-v1`` document: for each real spawned actor, a time-ordered
list of control segments. That DSL is validated
(``tools/video_trajectory_dsl.validate_video_trajectory_dsl``) and then lowered to
``risk-dsl-v1`` so the existing deterministic CARLA generator runs it.

Mirrors ``agents/llm_risk_dsl_generator.py`` (prior_attempt + schema error fed
back for a repair loop). The LLM only chooses timings, speeds and longitudinal /
soft-lateral actions; it never writes CARLA Python.
"""

import json
import os
import re
import sys
from typing import Any, Dict, Optional


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent


DSL_TUTORIAL = """
The video trajectory DSL is a JSON object that reconstructs an observed accident
video as per-actor timelines on an already-reconstructed scene. Schema:

{
  "schema_version": "video-trajectory-dsl-v1",
  "scene_id": "<id>",
  "accident_type": "<copy from the understanding>",
  "ego": {"actor_id": "ego", "target_speed_mps": <float>},
  "duration_s": <float>,
  "trajectories": [
    {
      "actor_id": "<real id from the actor table>",
      "role": "<short label>",
      "segments": [
        {"start_s": <float>, "end_s": <float>, "action": "<action>", ...params,
         "confidence": <0..1>, "source_frames": ["start","end"]}
      ]
    }
  ]
}

Each actor's segments are a time-ordered timeline (start_s ascending, no overlap).
The earliest segment of an actor defines its initial speed at t=0.

action is exactly one of (LONGITUDINAL / soft-lateral only in this version):
  - {"action": "lane_follow_speed", "target_speed_mps": <float>}  cruise at a speed
  - {"action": "brake", "intensity": <0..1>}                       decelerate
  - {"action": "stop"}                                             come to a stop
  - {"action": "hold_position"}                                    stay stopped
  - {"action": "steer_offset", "steer": <-1..1>, "throttle": <0..1>} brief lateral nudge

NOT available (no path-tracking controller): "lane_change", "target_waypoint".
To reproduce a cut-in, place/keep the actor in the adjacent lane and use
lane_follow_speed to approach; use steer_offset only for a brief encroachment.

Rules:
- Use ONLY actor ids from the actor table. Map each video participant (by its
  mapping_hint and geometry) to the table row whose relative_to_ego best matches.
- "ego" usually just drives forward; give it segments only if the ego itself
  brakes/decelerates in the video.
- Translate the coarse start/mid/end timeline into concrete start_s/end_s within
  duration_s. A typical reconstruction is 6-12 s.
- Make the mechanism reproducible: e.g. a lead vehicle does
  lane_follow_speed -> brake -> stop so the forward-driving ego rear-ends it.
- Output ONLY the JSON object. No markdown fences, no commentary.
""".strip()


EXAMPLE_DSL = """
{
  "schema_version": "video-trajectory-dsl-v1",
  "scene_id": "s0000_c0",
  "accident_type": "rear-end (lead vehicle stops)",
  "ego": {"actor_id": "ego", "target_speed_mps": 10.0},
  "duration_s": 8.0,
  "trajectories": [
    {
      "actor_id": "veh_1",
      "role": "lead_vehicle",
      "segments": [
        {"start_s": 0.0, "end_s": 2.5, "action": "lane_follow_speed",
         "target_speed_mps": 5.0, "confidence": 0.8, "source_frames": ["start"]},
        {"start_s": 2.5, "end_s": 4.0, "action": "brake", "intensity": 0.9,
         "confidence": 0.7},
        {"start_s": 4.0, "end_s": 8.0, "action": "stop", "confidence": 0.7,
         "source_frames": ["end"]}
      ]
    }
  ]
}
""".strip()


class LlmVideoTrajectoryGenerator(TaskAgent):
    """LLM that writes a ``video-trajectory-dsl-v1`` document from a video understanding."""

    def __init__(self, ego_speed_mps: float = 10.0) -> None:
        super().__init__()
        self.ego_speed_mps = float(ego_speed_mps)

    def generate(
        self,
        *,
        scene_id: str,
        understanding: Dict[str, Any],
        actor_context: Dict[str, Any],
        prior_attempt: Optional[str] = None,
        schema_error: Optional[str] = None,
    ) -> str:
        add_info = {
            "scene_id": scene_id,
            "understanding": understanding,
            "actor_context": actor_context,
            "prior_attempt": prior_attempt,
            "schema_error": schema_error,
            "request_label": "Video trajectory DSL generation",
            "request_timeout": 180,
            "request_max_tokens": 3500,
        }
        response = self.send_request("", add_info)
        return self._strip_json(response)

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing video trajectory generator add_info."
        scene_id = str(add_info.get("scene_id") or "")
        understanding = add_info.get("understanding") or {}
        actor_context = add_info.get("actor_context") or {}

        prompt = (
            "You reconstruct an observed accident video as executable per-actor "
            "timelines on an already-reconstructed static scene.\n\n"
            f"The ego drives forward at about {self.ego_speed_mps:.1f} m/s unless the "
            "video shows it braking.\n\n"
            f"DSL reference:\n{DSL_TUTORIAL}\n\n"
            f"Example DSL (style reference, different scene):\n{EXAMPLE_DSL}\n\n"
            "Actor table (the real spawned actors with ego-relative geometry; "
            "longitudinal_m ahead/+, lateral_m right/+). Use these ids:\n"
            f"{json.dumps(actor_context, indent=2, sort_keys=True)}\n\n"
            "Structured video understanding to reproduce:\n"
            f"{json.dumps(understanding, indent=2, sort_keys=True)}\n\n"
            "Instructions:\n"
            f'- Set scene_id to "{scene_id}".\n'
            f"- Set ego.target_speed_mps to {self.ego_speed_mps:.1f}.\n"
            "- Map each video participant to a real actor id from the table.\n"
            "- Turn the coarse start/mid/end event_sequence into concrete "
            "start_s/end_s segments per actor.\n"
            "- Use ONLY the actions in the reference. For lateral maneuvers, "
            "approximate longitudinally (the cut-in actor is already in its lane).\n"
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
