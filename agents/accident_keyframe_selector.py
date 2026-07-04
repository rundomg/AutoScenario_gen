"""VLM selector for accident timing."""

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


class AccidentKeyframeSelector(TaskAgent):
    """Ask a VLM to estimate the accident/collision second."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
        You are analysing frames sampled from a traffic accident video.
        The frames are provided in chronological order. Each image is preceded by
        metadata containing sample_index, timestamp_s, second, and frame_index.

        Your task:
        Estimate the second when the accident/collision most likely happens.

        Accident timing frames should prioritize:
        - the closest visual evidence of vehicle contact or near-contact;
        - abrupt relative motion, blocked path, collision aftermath, or sudden
          posture change;
        - the ego vehicle is the camera vehicle when visible from context.

        Output ONLY JSON (no markdown fences) using this schema:
        {
          "collision_second": 0.0
        }

        Rules:
        - Choose seconds only from the provided frame metadata timestamp_s values.
        - If the exact impact frame is uncertain, choose the nearest sampled timestamp.
        - Do not output explanations, confidence, frame objects, arrays, markdown, or extra keys.
        """

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing keyframe selector add_info."
        scene_id = str(add_info.get("scene_id") or "")
        frames = list(add_info.get("frames") or [])
        if not frames:
            raise ValueError("No sampled frames provided to keyframe selector.")

        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nUser request:\n{user_request}"
        prompt += f"\n\nsource_scene_id:\n{scene_id}\n"

        messages: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            image_path = frame.get("path")
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Frame image not found: {image_path}")
            metadata = {
                "sample_index": frame.get("sample_index"),
                "timestamp_s": frame.get("timestamp_s"),
                "second": frame.get("second"),
                "frame_index": frame.get("frame_index"),
            }
            messages.append({"type": "text", "text": f"Frame metadata: {json.dumps(metadata)}"})
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
                    "request_label": "Accident keyframe selection",
                    "request_timeout": 240,
                    "request_max_tokens": 1000,
                },
            )
            payload, error = self.extract_decision_data(output_fn)
            attempts += 1
            if error is None:
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Accident keyframe selection failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {error}"
                )
            print(f"Regenerating accident keyframe selection... Attempt {attempts + 1}")

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
        if not isinstance(payload.get("collision_second"), (int, float)):
            return None, "`collision_second` must be numeric."
        allowed_keys = {"collision_second"}
        extra_keys = set(payload) - allowed_keys
        if extra_keys:
            return None, f"Unexpected keys in output: {sorted(extra_keys)}."
        return payload, None

    @staticmethod
    def _parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(text, str) or not text.strip():
            return None, "Keyframe selector output is empty."
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"Keyframe selector output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Keyframe selector output must be a JSON object."
        return payload, None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_type(image_path: str) -> str:
        mime_type, _encoding = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"
