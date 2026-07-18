"""VLM fact extractor for ACRS background-environment evaluation."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from agents.task_agent import TaskAgent


class ACRSVisualExtractor(TaskAgent):
    """Extract controlled environment facts; never asks the model for a score."""

    PROMPT_VERSION = "acrs-environment-facts-v1"
    ENUMS = {
        "weather": {"clear", "overcast", "rain", "fog", "snow", "unknown"},
        "lighting": {"daylight", "night", "twilight", "unknown"},
        "time_of_day": {"day", "night", "dawn", "dusk", "unknown"},
        "road_surface": {"dry", "wet", "snow", "unknown"},
        "urban_density": {"urban", "urban_like", "suburban", "rural", "mixed", "natural", "unknown"},
    }
    CONFIDENCE = {"low", "medium", "high", "unknown"}
    ACTOR_ONLY_TAGS = {
        "car", "cars", "vehicle", "vehicles", "traffic_participant",
        "traffic_participants", "parkingcar", "parking_car", "parking_cars",
        "parked_car", "parked_cars", "parked_vehicle", "parked_vehicles",
    }
    TAG_ALIASES = {
        "residential_house": "residential_building",
        "residential_houses": "residential_building",
        "residential_buildings": "residential_building",
    }

    @staticmethod
    def _encode(path: str) -> str:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("ascii")

    @staticmethod
    def _mime(path: str) -> str:
        return "image/png" if path.lower().endswith(".png") else "image/jpeg"

    def refine_request(self, user_request=None, add_info=None):
        add_info = add_info or {}
        image_paths = [path for path in add_info.get("image_paths", []) if path and os.path.isfile(path)]
        if not image_paths:
            raise ValueError("At least one CARLA render image is required.")
        prompt = f"""
You extract observable environment facts from CARLA render images for a static-scene
evaluation. Images may include an ego camera view and a bird's-eye view of the SAME
generated scene. Inspect only these generated images. Do not compare them with a
source photograph, do not judge similarity, and do not output a score.

Return exactly one JSON object with this schema:
{{
  "schema_version": "acrs-visual-observation-v1",
  "weather": "clear|overcast|rain|fog|snow|unknown",
  "lighting": "daylight|night|twilight|unknown",
  "time_of_day": "day|night|dawn|dusk|unknown",
  "road_surface": "dry|wet|snow|unknown",
  "urban_density": "urban|urban_like|suburban|rural|mixed|natural|unknown",
  "roadside_context_left": ["building|residential_building|apartment_building|sidewalk|vegetation|commercial_frontage|industrial_area|open_space|water|other stable tags"],
  "roadside_context_right": ["same controlled environment tags"],
  "landmarks_and_controls": ["traffic_light|street_light|building|vegetation|sidewalk|water|other stable tags"],
  "confidence": {{
    "weather": "low|medium|high|unknown",
    "lighting": "low|medium|high|unknown",
    "time_of_day": "low|medium|high|unknown",
    "road_surface": "low|medium|high|unknown",
    "urban_density": "low|medium|high|unknown",
    "roadside_context_left": "low|medium|high|unknown",
    "roadside_context_right": "low|medium|high|unknown",
    "landmarks_and_controls": "low|medium|high|unknown"
  }},
  "evidence": ["short factual observations"]
}}

Use unknown rather than guessing. Left and right are relative to the ego vehicle's
forward direction, not image-pixel coordinates. Cars, parked cars, pedestrians,
cyclists, and other traffic participants are actors, not roadside environment;
never include them in roadside_context or landmarks_and_controls. Treat
residential_houses and residential_building as the single general tag
residential_building. JSON only.
""".strip()
        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{self._mime(path)};base64,{self._encode(path)}"},
            })
        return content

    @staticmethod
    def _json_text(text: str) -> str:
        stripped = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("VLM response does not contain a JSON object.")
        return stripped[start:end + 1]

    @classmethod
    def normalize(cls, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Visual observation must be an object.")
        normalized: Dict[str, Any] = {"schema_version": "acrs-visual-observation-v1"}
        for field, allowed in cls.ENUMS.items():
            value = str(payload.get(field) or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
            normalized[field] = value if value in allowed else "unknown"
        for field in ("roadside_context_left", "roadside_context_right", "landmarks_and_controls"):
            values = payload.get(field) if isinstance(payload.get(field), list) else []
            slugs = {
                str(value).strip().lower().replace("-", "_").replace(" ", "_")
                for value in values if str(value).strip()
            }
            normalized[field] = sorted({
                cls.TAG_ALIASES.get(value, value)
                for value in slugs if value not in cls.ACTOR_ONLY_TAGS
            })
        raw_confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
        normalized["confidence"] = {}
        for field in cls.ENUMS.keys() | {"roadside_context_left", "roadside_context_right", "landmarks_and_controls"}:
            value = str(raw_confidence.get(field) or "unknown").lower()
            normalized["confidence"][field] = value if value in cls.CONFIDENCE else "unknown"
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
        normalized["evidence"] = [str(item) for item in evidence[:10]]
        normalized["prompt_version"] = cls.PROMPT_VERSION
        return normalized

    @classmethod
    def parse_response(cls, text: str) -> dict:
        return cls.normalize(json.loads(cls._json_text(text)))

    def extract(self, image_paths: list, **request_options) -> Tuple[Optional[dict], Optional[str]]:
        try:
            response = self.send_request(
                "extract ACRS environment facts",
                {"image_paths": image_paths, "request_label": "ACRS visual extraction", **request_options},
            )
            return self.parse_response(response), None
        except Exception as exc:
            return None, str(exc)
