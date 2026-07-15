"""Stage B of the dynamic-reconstruction pipeline.

Given the structured video understanding (``VideoAccidentInterpreter``) plus the
reconstructed actor table (``actors.json`` / spawn_payload), an LLM emits a
``video-trajectory-dsl-v2`` document: for each real spawned actor, a time-ordered
list of control segments. That DSL is validated
(``tools/video_trajectory_dsl.validate_video_trajectory_dsl``) and then lowered to
``risk-dsl-v1`` so the existing deterministic CARLA generator runs it.

Mirrors ``agents/llm_risk_dsl_generator.py`` (prior_attempt + schema error fed
back for a repair loop). The LLM only chooses timings, speeds and longitudinal /
validated atomic vehicle behaviors; it never writes CARLA Python.
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
from tools.vehicle_atomic_behaviors import behavior_catalog_for_prompt


DSL_TUTORIAL = """
The video trajectory DSL is a JSON object that reconstructs an observed accident
video as per-actor timelines on an already-reconstructed scene. Schema:

{
  "schema_version": "video-trajectory-dsl-v2",
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

Atomic behavior catalog:
__ATOMIC_BEHAVIOR_CATALOG__

Required parameters by behavior:
- lane_follow_speed: target_speed_mps
- accelerate_to_speed: target_speed_mps; optional max_accel_mps2
- decelerate_to_speed: target_speed_mps; optional max_decel_mps2
- brake: intensity 0..1
- emergency_brake, coast: no behavior-specific parameters
- stop, hold_position: optional hand_brake
- steer_offset: steer -1..1, throttle 0..1; optional brake 0..1
- lane_change: direction left|right; optional lane_count, target_speed_mps,
  transition_distance_m
- junction_maneuver: direction left|right|straight|u_turn, target_speed_mps
- drive_to_location: target {x,y,z}, target_speed_mps, acceptance_radius_m
- reverse: target_speed_mps (max 10), steer -1..1
- follow_actor: target_actor_id, desired_gap_m, max_speed_mps
- approach_actor: target_actor_id, target_gap_m (0 permits impact),
  approach_speed_mps; optional travel_direction forward|reverse and steer
- yield_to_actor: target_actor_id, yield_distance_m, resume_speed_mps
- raw_vehicle_control: throttle, steer, brake, hand_brake, reverse,
  manual_gear_shift, gear. Use only when no safer atom expresses the motion.

Any segment may add a `lights` list, e.g. ["left_blinker"] or
["brake", "right_blinker"]. Lights are a side effect and do not replace motion.

Rules:
- Use ONLY actor ids from the actor table. Map each video participant by its
  matched_actor_id when valid, then verify mapping_hint and geometry against the
  table row whose relative_to_ego best matches.
- Emit exactly one trajectory for EVERY non-ego vehicle in the actor table,
  including vehicles that are not accident participants. For a non-participant,
  only classify whether it moves during the observed clip: use one full-duration
  lane_follow_speed segment if it moves, otherwise one full-duration
  hold_position segment. Parked/stopped actor-table states are strong evidence
  for hold_position. If neither the frames nor actor table show movement, choose
  hold_position conservatively; never omit a vehicle.
- Use understanding.actor_motion_states as the primary moving/stationary result
  from the frame-reading VLM. `moving` means lane following; `stationary` means
  hold_position. Resolve `uncertain` conservatively using actor-table motion_state,
  defaulting to hold_position when there is no positive evidence of movement.
- "ego" usually just drives forward; give it segments only if the ego itself
  brakes/decelerates in the video.
- Translate the coarse start/mid/end timeline into concrete start_s/end_s within
  duration_s. A typical reconstruction is 6-12 s.
- Make the mechanism reproducible: e.g. a lead vehicle does
  lane_follow_speed -> brake -> stop so the forward-driving ego rear-ends it.
- Keep acceleration segments physically self-consistent: when an actor starts
  from rest and must reach target_speed_mps within the segment, choose
  max_accel_mps2 >= target_speed_mps / segment_duration_s. If the user says a
  following vehicle is faster, its reachable speed (not merely its requested
  target) must exceed the lead vehicle's reachable speed.
- Complex behavior MUST be composed from atomic segments. Examples:
  cut-in = lane_follow_speed -> lane_change -> lane_follow_speed;
  overtake = lane_change(left) -> accelerate_to_speed -> lane_change(right);
  rear-end = approach_actor(target_gap_m=0);
  loss-of-control = steer_offset/raw_vehicle_control -> emergency_brake.
- Prefer semantic closed-loop atoms over raw_vehicle_control. Never output
  Python, CARLA APIs, teleportation, impulses, or unlisted actions.
- Output ONLY the JSON object. No markdown fences, no commentary.
""".replace("__ATOMIC_BEHAVIOR_CATALOG__", behavior_catalog_for_prompt()).strip()


EXAMPLE_DSL = """
{
  "schema_version": "video-trajectory-dsl-v2",
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
    """LLM that writes a validated atomic vehicle behavior program."""

    def __init__(self, ego_speed_mps: float = 10.0) -> None:
        super().__init__()
        self.ego_speed_mps = float(ego_speed_mps)

    def generate(
        self,
        *,
        scene_id: str,
        understanding: Dict[str, Any],
        actor_context: Dict[str, Any],
        user_request: str = "",
        prior_attempt: Optional[str] = None,
        schema_error: Optional[str] = None,
    ) -> str:
        add_info = {
            "scene_id": scene_id,
            "understanding": understanding,
            "actor_context": actor_context,
            "user_request": str(user_request or ""),
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
        user_request = str(add_info.get("user_request") or "").strip()

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
            "User-specified reconstruction constraints (when non-empty, these "
            "override coarse or inferred timing/speed choices in the structured "
            "video understanding):\n"
            f"{user_request or '(none)'}\n\n"
            "Instructions:\n"
            f'- Set scene_id to "{scene_id}".\n'
            f"- Set ego.target_speed_mps to {self.ego_speed_mps:.1f}.\n"
            "- Map each video participant to a real actor id from the table.\n"
            "- Prefer participants[].matched_actor_id when it exists in the actor "
            "table, but reject it when its type or geometry clearly conflicts.\n"
            "- Turn the coarse start/mid/end event_sequence into concrete "
            "start_s/end_s segments per actor.\n"
            "- Treat explicit user timings, initial motion states, actor ordering, "
            "relative speeds, and collision intent as binding constraints. Preserve "
            "numerical values exactly when the DSL can express them.\n"
            "- Include exactly one trajectory for EVERY non-ego vehicle in the "
            "actor table, even when it is not an accident participant. For each "
            "non-participant, make only a moving/stationary decision from the "
            "frames and actor-table motion_state: moving => one full-duration "
            "lane_follow_speed segment; parked/stopped => one full-duration "
            "hold_position segment. Unknown without visible motion => "
            "hold_position. Never omit a surrounding vehicle.\n"
            "- Treat understanding.actor_motion_states as the VLM's primary "
            "motion classification for every vehicle. Use actor-table "
            "motion_state only to resolve uncertain cases, conservatively "
            "defaulting to hold_position without positive motion evidence.\n"
            "- Use ONLY atomic behaviors from the catalog. Express every complex "
            "maneuver as a time-ordered composition of atomic segments.\n"
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
