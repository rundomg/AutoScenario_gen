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
Do not do CARLA map matching or final actor layout. The road-scene agent has
already classified the road topology; follow that classification exactly.

Inputs:
1. The first image is the annotated full image. It shows the whole scene and detector ids.
   Use it to understand global context: where the junction center/stop line is, which side
   of the scene each detected vehicle occupies, and how each crop relates to the full image.
2. The following images are vehicle crops in the same order as the detected-vehicle list.
   Use crops to inspect local vehicle details: front, rear, side profile, grille/headlights,
   tail lights, truck cab/bed, wheel direction, and body orientation.
3. Combine both: crop tells where the vehicle front/rear points; full image keeps
   the detector id and lane/arm context grounded.

General rules:
- Use the crop to judge front/rear/side details.
- Use the annotated full image only for detector identity and coarse road-region context.
- If a detected vehicle is clearly outside the drivable/reconstructable road scene,
  a duplicate/false positive, or a staged roadside object with no scene role, put
  its det_id in ignored_detections. This is only a soft candidate for downstream
  review, not a final deletion decision.
- If it appears in an opposing travel lane and faces the ego/camera, use
  opposite_direction unless there is clear wrong-way evidence.
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
        road_scene = add_info.get("road_scene") or {}
        is_junction = _is_junction_scene(road_scene)
        prompt += "\n\nRoad scene JSON from the road-scene agent (authoritative):\n"
        prompt += json.dumps(road_scene, ensure_ascii=False, indent=2)
        prompt += _orientation_schema_block(is_junction)
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
            return _empty_orientation_payload()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return _empty_orientation_payload()
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return _empty_orientation_payload()
        if not isinstance(payload, dict):
            return _empty_orientation_payload()
        brief = payload.get("vehicle_orientation_brief")
        if not isinstance(brief, dict):
            brief = {}
        ignored = _normalize_ignored_detections(payload.get("ignored_detections"))
        hints = payload.get("vehicle_orientation_hints") if isinstance(payload, dict) else None
        if not isinstance(hints, list):
            return {
                "vehicle_orientation_brief": brief,
                "ignored_detections": ignored,
                "vehicle_orientation_hints": [],
            }
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
                    "road_region_hint": _choice(
                        hint.get("road_region_hint"),
                        {
                            "ego_lane",
                            "left_lane",
                            "right_lane",
                            "opposing_lane",
                            "left_parking_lane",
                            "right_parking_lane",
                            "roadside",
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
        return {
            "vehicle_orientation_brief": brief,
            "ignored_detections": ignored,
            "vehicle_orientation_hints": normalized,
        }

    @staticmethod
    def sanitize_for_road_scene(payload: dict, road_scene: dict) -> dict:
        if _is_junction_scene(road_scene):
            return payload
        sanitized = {
            "vehicle_orientation_brief": payload.get("vehicle_orientation_brief", {}),
            "ignored_detections": _normalize_ignored_detections(
                payload.get("ignored_detections")
            ),
            "vehicle_orientation_hints": [],
        }
        for hint in payload.get("vehicle_orientation_hints") or []:
            if not isinstance(hint, dict):
                continue
            cleaned = dict(hint)
            cleaned.pop("vehicle_region_hint", None)
            cleaned.pop("junction_center_relative_to_vehicle", None)
            cleaned.pop("junction_travel_direction", None)
            sanitized["vehicle_orientation_hints"].append(cleaned)
        return sanitized

    @staticmethod
    def extract_orientation_hints(response_text: str) -> list:
        return VehicleOrientationInterpreter.extract_orientation_payload(response_text)[
            "vehicle_orientation_hints"
        ]


def _choice(value, allowed: set, default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _empty_orientation_payload() -> dict:
    return {
        "vehicle_orientation_brief": {},
        "ignored_detections": [],
        "vehicle_orientation_hints": [],
    }


def _normalize_ignored_detections(value) -> list:
    if not isinstance(value, list):
        return []
    normalized = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            det_id = item.strip()
            reason = ""
        elif isinstance(item, dict):
            det_id = str(item.get("det_id") or item.get("id") or "").strip()
            reason = str(item.get("reason") or item.get("evidence") or "").strip()
        else:
            continue
        if not det_id or det_id in seen:
            continue
        seen.add(det_id)
        normalized.append({"det_id": det_id, "reason": reason})
    return normalized


def _is_junction_scene(road_scene: dict) -> bool:
    map_matching = ((road_scene or {}).get("road_network") or {}).get("map_matching") or {}
    topology = str(map_matching.get("topology_type") or "").lower()
    return bool(map_matching.get("junction_visible")) or topology in {
        "t_junction",
        "cross_intersection",
        "multi_branch",
        "roundabout",
        "signalized_intersection",
    }


def _orientation_schema_block(is_junction: bool) -> str:
    if is_junction:
        return """



Junction rules:
- Treat ignored_detections as soft ignore candidates for downstream review.
- Use ignored_detections only when a detector id is clearly outside the
  reconstructable road scene, duplicate/false positive, or has no visible scene role.
- Assign arm labels only because the road-scene agent classified this as a junction.
- Do not use image-left/image-right or bbox center-x alone to choose left_arm/right_arm.
- A vehicle ahead of ego in an ego-approach left/right/same lane is still
  vehicle_region_hint=ego_approach, not left_arm/right_arm.
- Use left_arm/right_arm only for vehicles physically on the cross street or side-road
  branch, with visible road/lane geometry supporting that branch membership.
- For side-profile vehicles on the right-side arm, front_points_image_direction=right
  usually means away_from_junction, and left usually means toward_junction.
- For side-profile vehicles on the left-side arm, front_points_image_direction=left
  usually means away_from_junction, and right usually means toward_junction.
- For vehicles on the ahead/oncoming arm, front facing ego/camera usually means
  toward_junction; rear facing ego/camera usually means away_from_junction.

Road-scene branch: JUNCTION. Output JSON only:
{
  "vehicle_orientation_brief": {
    "overall_observation": "short summary of visible vehicle orientation patterns",
    "uncertainties": ["short uncertainty notes"]
  },
  "ignored_detections": [
    {
      "det_id": "det_2",
      "reason": "duplicate, false positive, outside drivable scene, or no reconstructable scene role"
    }
  ],
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
"""
    return """

Road-scene branch: OPEN_ROAD / STRAIGHT_OR_CURVE. Output JSON only:
{
  "vehicle_orientation_brief": {
    "overall_observation": "short summary of visible vehicle orientation patterns",
    "uncertainties": ["short uncertainty notes"]
  },
  "ignored_detections": [
    {
      "det_id": "det_2",
      "reason": "duplicate, false positive, outside drivable scene, or no reconstructable scene role"
    }
  ],
  "vehicle_orientation_hints": [
    {
      "det_id": "det_1",
      "visible_end": "front | rear | side | unclear",
      "front_points_image_direction": "left | right | toward_camera | away_from_camera | unclear",
      "road_region_hint": "ego_lane | left_lane | right_lane | opposing_lane | left_parking_lane | right_parking_lane | roadside | unknown",
      "heading_relation_to_ego": "same_direction | opposite_direction | crossing | unknown",
      "confidence": "high | medium | low",
      "evidence": "short visual evidence"
    }
  ]
}

Open-road rules:
- Treat ignored_detections as soft ignore candidates for downstream review.
- Use ignored_detections only when a detector id is clearly outside the
  reconstructable road scene, duplicate/false positive, or has no visible scene role.
- Do not output vehicle_region_hint, junction_center_relative_to_vehicle, or
  junction_travel_direction.
- Do not use arm labels such as right_arm, left_arm, ahead_arm, or ego_approach.
- Use road_region_hint only for lane/parking/opposing-road context.
"""
