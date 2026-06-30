"""Stage 1 of the pure-LLM (no template library) accident experiment.

Given the original camera image and a fixed ego speed, a VLM freely identifies
the road participants and predicts plausible accidents. Unlike
``agents/risk_scenario_interpreter.py`` this agent is intentionally NOT
constrained by ``tools/accident_template_library.py`` -- the whole point of the
experiment branch is to test what a pure LLM produces without the template
catalogue.

Output: ``{scene_id}_candidates.json``.
"""

import base64
import json
import mimetypes
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent
from tools.utils import write_to_file


DEFAULT_EGO_SPEED_MPS = 10.0


class LlmAccidentPredictor(TaskAgent):
    """VLM that freely identifies road users and predicts possible accidents."""

    def __init__(self, ego_speed_mps: float = DEFAULT_EGO_SPEED_MPS) -> None:
        super().__init__()
        self.ego_speed_mps = float(ego_speed_mps)
        self.pre_prompt = f"""
        You are analysing a single driving camera image to predict possible
        traffic accidents involving the ego vehicle.

        Assume the ego vehicle is the camera car, driving straight forward at a
        fixed speed of {self.ego_speed_mps:.1f} m/s unless it must react.

        Your job has two parts:
        1. Freely identify the salient road participants visible in the image
           (vehicles, pedestrians, cyclists, motorcyclists, static obstacles).
           You are NOT given a fixed actor list -- read them directly from the
           image.
        2. Predict the plausible accidents that could occur as the ego keeps
           driving, in priority order. Do not restrict yourself to a fixed
           catalogue of accident types; describe whatever the scene supports.

        Output ONLY JSON. Do not output markdown fences. Use this schema:
        {{
          "scene_id": "...",
          "ego": {{
            "speed_mps": {self.ego_speed_mps:.1f},
            "assumed_motion": "drives straight forward in its lane"
          }},
          "participants": [
            {{
              "id": "p1",
              "type": "car|truck|bus|motorcycle|bicycle|pedestrian|obstacle",
              "visual_description": "short visual cue",
              "position_relative_to_ego": "e.g. ahead in same lane ~10 m / left adjacent lane / right curb",
              "heading_relative_to_ego": "same_direction|oncoming|crossing|stationary"
            }}
          ],
          "accident_candidates": [
            {{
              "id": "acc_001",
              "accident_type": "free-text, e.g. rear-end, pedestrian crossing, cut-in",
              "involved": ["ego", "p1"],
              "mechanism": "how the accident develops as ego keeps driving",
              "ego_dynamics": "what the ego does (e.g. keeps 10 m/s then brakes late)",
              "npc_dynamics": "what the other participant does (e.g. brakes hard / cuts in / crosses)",
              "severity": "low|medium|high",
              "confidence": 0.0
            }}
          ]
        }}

        Rules:
        - Produce 1 to 5 accident_candidates, most plausible first.
        - Every accident_candidate.involved must include "ego".
        - involved ids should reference ids you declared in participants
          (besides "ego").
        - Keep descriptions concrete but brief. No CARLA or Scenic code.
        """

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing accident predictor add_info."
        scene_id = str(add_info.get("scene_id") or "")
        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nUser request:\n{user_request}"
        prompt += f"\n\nsource_scene_id:\n{scene_id}\n"

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

    def call_agent(self, user_request, added_info) -> Dict[str, Any]:
        output_fn = added_info["output_fn"]
        attempts = 0
        while True:
            self.send_request(
                user_request,
                {
                    **added_info,
                    "request_label": "Accident prediction",
                    "request_timeout": 180,
                    "request_max_tokens": 4000,
                },
            )
            payload, error = self.extract_decision_data(output_fn)
            attempts += 1
            if error is None:
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Accident prediction failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {error}"
                )
            print(f"Regenerating accident prediction... Attempt {attempts + 1}")

    def extract_decision_data(
        self, file_path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        with open(file_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        payload, parse_error = self._parse_json_object(text)
        if parse_error:
            return None, parse_error
        return self._validate(payload)

    @staticmethod
    def _validate(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        candidates = payload.get("accident_candidates")
        if not isinstance(candidates, list) or not candidates:
            return None, "`accident_candidates` must be a non-empty list."
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                return None, f"accident_candidates[{index}] must be an object."
            candidate.setdefault("id", f"acc_{index + 1:03d}")
            involved = candidate.get("involved")
            if not isinstance(involved, list) or "ego" not in [str(x) for x in involved]:
                return None, f"{candidate['id']} must include `ego` in involved."
        participants = payload.get("participants")
        if participants is not None and not isinstance(participants, list):
            return None, "`participants` must be a list when present."
        return payload, None

    @staticmethod
    def _parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(text, str) or not text.strip():
            return None, "Accident prediction output is empty."
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"Accident prediction output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Accident prediction output must be a JSON object."
        return payload, None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_type(image_path: str) -> str:
        mime_type, _encoding = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"
