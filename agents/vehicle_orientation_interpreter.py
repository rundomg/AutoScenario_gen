import base64
import json
import os
import re
import sys

import cv2


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent


class VehicleOrientationInterpreter(TaskAgent):
    """VLM helper focused only on detected vehicle orientation."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
You inspect detected vehicles in one traffic image.
Your task is actor-level vehicle orientation using two kinds of visual input.
Do not do CARLA map matching or final actor layout. You MAY use the annotated full image
to infer coarse image-region context such as whether a vehicle is on the right-side arm,
left-side arm, ahead arm, ego approach, or unknown.

Inputs:
1. The first image is the annotated full image. It shows the whole scene and detector ids.
   Use it to understand global context: where the junction center/stop line is, which side
   of the scene each detected vehicle occupies, and how each crop relates to the full image.
2. The following images are vehicle crops in the same order as the detected-vehicle list.
   Use crops to inspect local vehicle details: front, rear, side profile, grille/headlights,
   tail lights, truck cab/bed, wheel direction, and body orientation.
3. Combine both: crop tells where the vehicle front/rear points; full image tells whether
   that direction is toward or away from the visible junction/road arm.

Output JSON only:
{
  "vehicle_orientation_brief": {
    "overall_observation": "short summary of visible vehicle orientation patterns",
    "uncertainties": ["short uncertainty notes"]
  },
  "vehicle_orientation_hints": [
    {
      "det_id": "det_1",
      "visible_end": "front | rear | side | unclear",
      "front_points_image_direction": "left | right | toward_camera | away_from_camera | unclear",
      "vehicle_region_hint": "right_arm | left_arm | ahead_arm | ego_approach | unknown",
      "junction_center_relative_to_vehicle": "left | right | ahead | behind | unknown",
      "heading_relation_to_ego": "same_direction | opposite_direction | crossing | unknown",
      "junction_travel_direction": "toward_junction | away_from_junction | unknown",
      "confidence": "high | medium | low",
      "evidence": "short visual evidence"
    }
  ]
}

Rules:
- Use the crop to judge front/rear/side details. Use the annotated full image to keep
  det id, vehicle region, and junction-center direction grounded.
- Do not set junction_travel_direction=unknown merely because you are not doing map matching.
  If the full image gives a clear coarse arm/region, use that image geometry.
- For side-profile vehicles on the right-side arm/right side road:
  front_points_image_direction=right usually means away_from_junction, and left usually
  means toward_junction, unless the full image clearly shows the junction center on the
  opposite side.
- For side-profile vehicles on the left-side arm/left side road:
  front_points_image_direction=left usually means away_from_junction, and right usually
  means toward_junction, unless contradicted by the full image.
- For vehicles on the ahead/oncoming arm, front facing the ego/camera usually means
  toward_junction; rear facing the ego/camera usually means away_from_junction.
- If the vehicle is on a junction arm, front facing the junction means toward_junction;
  front pointing away from the junction means away_from_junction.
- If it appears in an opposing travel lane and faces the ego/camera, use opposite_direction unless there is clear wrong-way evidence.
- Use unknown/low confidence only when both crop detail and full-image region are ambiguous.
        """

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def refine_request(self, user_request=None, add_info=None):
        assert add_info and "image_path" in add_info, "Missing image_path"
        detections = add_info.get("detections") or []
        annotated_path = add_info.get("annotated_path") or add_info["image_path"]
        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nAdditional context:\n{user_request}"
        prompt += "\n\nDetected vehicles to inspect:\n"
        for det in detections:
            crop_path = det.get("crop_path")
            prompt += (
                f"- {det.get('id')} label={det.get('label')} conf={det.get('conf')} "
                f"bbox={det.get('bbox_norm')} crop={os.path.basename(crop_path) if crop_path else 'none'}\n"
            )

        content = [{"type": "text", "text": prompt}]
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._image_to_base64(annotated_path)}"
                },
            }
        )
        for det in detections:
            crop_path = det.get("crop_path")
            if crop_path and os.path.exists(crop_path):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(crop_path)}"
                        },
                    }
                )
        return content

    @staticmethod
    def extract_orientation_payload(response_text: str) -> dict:
        text = str(response_text or "").strip()
        if not text:
            return {"vehicle_orientation_brief": {}, "vehicle_orientation_hints": []}
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return {"vehicle_orientation_brief": {}, "vehicle_orientation_hints": []}
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {"vehicle_orientation_brief": {}, "vehicle_orientation_hints": []}
        if not isinstance(payload, dict):
            return {"vehicle_orientation_brief": {}, "vehicle_orientation_hints": []}
        brief = payload.get("vehicle_orientation_brief")
        if not isinstance(brief, dict):
            brief = {}
        hints = payload.get("vehicle_orientation_hints") if isinstance(payload, dict) else None
        if not isinstance(hints, list):
            return {"vehicle_orientation_brief": brief, "vehicle_orientation_hints": []}
        normalized = []
        for hint in hints:
            if not isinstance(hint, dict):
                continue
            det_id = str(hint.get("det_id") or "").strip()
            if not det_id:
                continue
            normalized.append(
                {
                    "det_id": det_id,
                    "visible_end": _choice(
                        hint.get("visible_end"),
                        {"front", "rear", "side", "unclear"},
                        "unclear",
                    ),
                    "front_points_image_direction": _choice(
                        hint.get("front_points_image_direction"),
                        {"left", "right", "toward_camera", "away_from_camera", "unclear"},
                        "unclear",
                    ),
                    "vehicle_region_hint": _choice(
                        hint.get("vehicle_region_hint"),
                        {
                            "right_arm",
                            "left_arm",
                            "ahead_arm",
                            "ego_approach",
                            "unknown",
                        },
                        "unknown",
                    ),
                    "junction_center_relative_to_vehicle": _choice(
                        hint.get("junction_center_relative_to_vehicle"),
                        {"left", "right", "ahead", "behind", "unknown"},
                        "unknown",
                    ),
                    "heading_relation_to_ego": _choice(
                        hint.get("heading_relation_to_ego"),
                        {"same_direction", "opposite_direction", "crossing", "unknown"},
                        "unknown",
                    ),
                    "junction_travel_direction": _choice(
                        hint.get("junction_travel_direction"),
                        {"toward_junction", "away_from_junction", "unknown"},
                        "unknown",
                    ),
                    "confidence": _choice(
                        hint.get("confidence"),
                        {"high", "medium", "low"},
                        "low",
                    ),
                    "evidence": str(hint.get("evidence") or ""),
                }
            )
        return {"vehicle_orientation_brief": brief, "vehicle_orientation_hints": normalized}

    @staticmethod
    def extract_orientation_hints(response_text: str) -> list:
        return VehicleOrientationInterpreter.extract_orientation_payload(response_text)[
            "vehicle_orientation_hints"
        ]


def _choice(value, allowed: set, default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default
