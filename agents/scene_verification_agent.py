import base64
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from agents.task_agent import TaskAgent
from tools.utils import read_file


class SceneVerificationAgent(TaskAgent):
    """VLM judge for comparing the source image with generated CARLA layout views."""

    DEFAULT_REPORT = {
        "passed": False,
        "score": 0.0,
        "hard_failures": [],
        "mismatches": [],
        "recommended_stage": "match_spawn",
        "repair_hints": [],
        "repair_actions": [],
    }

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_type(path: str) -> str:
        lowered = path.lower()
        if lowered.endswith(".png"):
            return "image/png"
        return "image/jpeg"

    def refine_request(self, user_request=None, add_info=None):
        add_info = add_info or {}
        source_image_path = add_info["source_image_path"]
        layout_image_path = add_info.get("layout_image_path") or add_info["bev_image_path"]
        prompt = self._build_prompt(add_info)
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{self._mime_type(source_image_path)};base64,"
                        f"{self._encode_image(source_image_path)}"
                    )
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{self._mime_type(layout_image_path)};base64,"
                        f"{self._encode_image(layout_image_path)}"
                    )
                },
            },
        ]

    @staticmethod
    def _build_prompt(add_info: Dict[str, Any]) -> str:
        scene_understanding = add_info.get("scene_understanding") or {}
        scene_match = add_info.get("scene_match") or {}
        spawn_entities = add_info.get("spawn_entities") or {}
        user_description = str(add_info.get("user_scene_description") or "").strip()
        capture_mode = str(add_info.get("capture_mode") or "bev_fallback")
        if capture_mode == "ego_view":
            view_intro = (
                "You are verifying whether the actor layout in a generated CARLA "
                "ego-view image correctly reconstructs the traffic participants "
                "from the original image. The first image is the original ego-view input. "
                "The second image is the generated CARLA ego-view image.\n\n"
                "Use the two ego-view images directly for front/left/right/near/far "
                "comparison. Do NOT penalize actors not visible in the forward ego-view "
                "field of view. Only judge actors that appear in both images or are "
                "expected to appear in the forward cone based on the scene description.\n\n"
            )
        else:
            view_intro = (
                "You are verifying whether the actor layout in a generated CARLA "
                "bird's-eye-view image correctly reconstructs the traffic participants "
                "from the original image. The first image is the original ego-view input. "
                "The second image is the generated CARLA BEV.\n\n"
                "Use the same ego-centric frame for the original image and the BEV: the "
                "ego vehicle's forward travel direction is positive longitudinal/ahead, "
                "ego-left is negative lateral, and ego-right is positive lateral. The BEV "
                "may be rotated or cropped, so do not judge left/right/ahead/behind from "
                "screen pixel directions. Use the ego actor and spawn_entities yaw/location "
                "to infer the BEV ego frame before comparing actor positions.\n\n"
            )
        return (
            f"{view_intro}"
            "IMPORTANT: The road map and environment (road type, urban/rural context, "
            "sidewalks, buildings, crosswalks) are already fixed and cannot be changed "
            "at this stage. Do NOT penalize score for road topology differences, map "
            "type mismatches, or missing environmental features.\n\n"
            "Judge ONLY the actor layout. Focus on:\n"
            "- Actor count: does the number of vehicles/pedestrians/cyclists match the source?\n"
            "- Actor categories: are vehicle types (car, truck, motorcycle, pedestrian) correct?\n"
            "- Relative spatial relations: ego-centric left/right and near/far positions between actors.\n"
            "- Lane side relations: same lane, adjacent lane, opposing lane, roadside/parked.\n"
            "- Heading/yaw: are actors facing the correct direction relative to the road?\n"
            "- Overlaps: are any actors unrealistically overlapping each other?\n"
            "- Moving vs parked: does each actor's motion state match the source?\n\n"
            "Output only JSON with these keys: passed, score, hard_failures, "
            "mismatches, recommended_stage, repair_hints, repair_actions. "
            "score must be 0.0-1.0. "
            "recommended_stage must be one of: match_spawn, scene_understanding, pass.\n"
            "Use recommended_stage='scene_understanding' only when actor count or "
            "categories are wrong. Use 'match_spawn' when positions/relations are wrong.\n"
            "Use passed=true only when score >= 0.70 and no hard failure remains.\n"
            "repair_actions must be a list. Supported action types are: "
            "lane_side_mismatch, pairwise_mismatch, category_mismatch, count_mismatch, "
            "overlap, heading_mismatch. For lane_side_mismatch, overlap, and "
            "heading_mismatch include entity_id. For pairwise_mismatch include "
            "entity_id, reference_entity_id, and target_relation. For count_mismatch "
            "and category_mismatch entity_id is optional. Each action should include "
            "severity and evidence.\n\n"
            f"User description, if any:\n{user_description or '(none)'}\n\n"
            "Current scene_understanding JSON:\n"
            f"{json.dumps(scene_understanding, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current scene_match JSON:\n"
            f"{json.dumps(scene_match, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current spawn_entities JSON:\n"
            f"{json.dumps(spawn_entities, ensure_ascii=False, sort_keys=True)}"
        )

    @classmethod
    def _extract_json_text(cls, text: str) -> str:
        stripped = str(text or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.S | re.I)
        if fenced:
            return fenced.group(1).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return stripped

    @classmethod
    def normalize_report(cls, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            payload = {}
        report = dict(cls.DEFAULT_REPORT)
        report.update(payload)
        try:
            report["score"] = max(0.0, min(1.0, float(report.get("score", 0.0))))
        except (TypeError, ValueError):
            report["score"] = 0.0
        report["passed"] = bool(report.get("passed")) and report["score"] >= 0.70
        report["hard_failures"] = (
            report["hard_failures"] if isinstance(report.get("hard_failures"), list) else []
        )
        report["mismatches"] = (
            report["mismatches"] if isinstance(report.get("mismatches"), list) else []
        )
        report["repair_hints"] = (
            report["repair_hints"] if isinstance(report.get("repair_hints"), list) else []
        )
        report["repair_actions"] = cls._normalize_repair_actions(report.get("repair_actions"))
        recommended_stage = str(report.get("recommended_stage") or "match_spawn")
        if recommended_stage not in {"match_spawn", "scene_understanding", "pass"}:
            recommended_stage = "match_spawn"
        if report["passed"]:
            recommended_stage = "pass"
        report["recommended_stage"] = recommended_stage
        return report

    @staticmethod
    def _normalize_repair_actions(value: Any) -> list:
        if not isinstance(value, list):
            return []
        requires_entity = {
            "lane_side_mismatch",
            "pairwise_mismatch",
            "overlap",
            "heading_mismatch",
        }
        optional_entity = {"count_mismatch", "category_mismatch"}
        normalized = []
        for item in value:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("type") or "").strip()
            if action_type not in requires_entity and action_type not in optional_entity:
                continue
            severity = str(item.get("severity") or "").strip()
            if not severity:
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            if action_type in requires_entity and not entity_id:
                continue
            reference_id = str(item.get("reference_entity_id") or "").strip()
            if action_type == "pairwise_mismatch" and not reference_id:
                continue
            action = dict(item)
            action["type"] = action_type
            action["severity"] = severity
            if entity_id:
                action["entity_id"] = entity_id
            elif "entity_id" in action:
                action.pop("entity_id", None)
            if reference_id:
                action["reference_entity_id"] = reference_id
            normalized.append(action)
        return normalized

    def extract_decision_data(self, file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        text = read_file(file_path)
        try:
            payload = json.loads(self._extract_json_text(text))
        except json.JSONDecodeError as exc:
            return None, f"Invalid verification JSON: {exc.msg}"
        return self.normalize_report(payload), None

    def call_agent(self, add_info: Dict[str, Any]) -> Dict[str, Any]:
        output_fn = add_info["output_fn"]
        attempts = 0
        while True:
            self.send_request(
                "",
                {
                    **add_info,
                    "request_label": "Scene verification",
                    "request_timeout": max(180, add_info.get("request_timeout", 180)),
                },
            )
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                report = dict(self.DEFAULT_REPORT)
                report["hard_failures"] = [validation_error]
                return report
            print(f"Regenerating scene verification... Attempt {attempts + 1}")
