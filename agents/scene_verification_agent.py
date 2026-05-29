import base64
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from agents.task_agent import TaskAgent
from tools.utils import read_file


class SceneVerificationAgent(TaskAgent):
    """VLM judge for comparing the source image with the generated CARLA BEV."""

    DEFAULT_REPORT = {
        "passed": False,
        "score": 0.0,
        "hard_failures": [],
        "mismatches": [],
        "recommended_stage": "match_spawn",
        "repair_hints": [],
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
        bev_image_path = add_info["bev_image_path"]
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
                        f"data:{self._mime_type(bev_image_path)};base64,"
                        f"{self._encode_image(bev_image_path)}"
                    )
                },
            },
        ]

    @staticmethod
    def _build_prompt(add_info: Dict[str, Any]) -> str:
        if add_info.get("verification_task") == "map_match":
            return SceneVerificationAgent._build_map_match_prompt(add_info)
        scene_understanding = add_info.get("scene_understanding") or {}
        scene_match = add_info.get("scene_match") or {}
        spawn_entities = add_info.get("spawn_entities") or {}
        user_description = str(add_info.get("user_scene_description") or "").strip()
        return (
            "You are verifying whether the actor layout in a generated CARLA "
            "bird's-eye-view image correctly reconstructs the traffic participants "
            "from the original image. The first image is the original ego-view input. "
            "The second image is the generated CARLA BEV.\n\n"
            "IMPORTANT: The road map and environment (road type, urban/rural context, "
            "sidewalks, buildings, crosswalks) are already fixed and cannot be changed "
            "at this stage. Do NOT penalize score for road topology differences, map "
            "type mismatches, or missing environmental features. Those are evaluated "
            "separately before this stage.\n\n"
            "Judge ONLY the actor layout. Focus on:\n"
            "- Actor count: does the number of vehicles/pedestrians/cyclists match the source?\n"
            "- Actor categories: are vehicle types (car, truck, motorcycle, pedestrian) correct?\n"
            "- Relative spatial relations: left/right and near/far positions between actors.\n"
            "- Lane side relations: same lane, adjacent lane, opposing lane, roadside/parked.\n"
            "- Heading/yaw: are actors facing the correct direction relative to the road?\n"
            "- Overlaps: are any actors unrealistically overlapping each other?\n"
            "- Moving vs parked: does each actor's motion state match the source?\n\n"
            "Output only JSON with these keys: passed, score, hard_failures, "
            "mismatches, recommended_stage, repair_hints. score must be 0.0-1.0. "
            "recommended_stage must be one of: match_spawn, scene_understanding, pass.\n"
            "Use recommended_stage='scene_understanding' only when actor count or "
            "categories are wrong. Use 'match_spawn' when positions/relations are wrong.\n"
            "Use passed=true only when score >= 0.70 and no hard failure remains.\n\n"
            f"User description, if any:\n{user_description or '(none)'}\n\n"
            "Current scene_understanding JSON:\n"
            f"{json.dumps(scene_understanding, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current scene_match JSON:\n"
            f"{json.dumps(scene_match, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current spawn_entities JSON:\n"
            f"{json.dumps(spawn_entities, ensure_ascii=False, sort_keys=True)}"
        )

    @staticmethod
    def _build_map_match_prompt(add_info: Dict[str, Any]) -> str:
        topology_signature = add_info.get("road_topology_signature") or {}
        scene_match = add_info.get("scene_match") or {}
        user_description = str(add_info.get("user_scene_description") or "").strip()
        return (
            "You are verifying whether a clean CARLA map bird's-eye-view region "
            "matches the road topology of the original traffic image. The first image "
            "is the original ego-view input. The second image is the clean CARLA map BEV "
            "before scenario vehicles are added.\n\n"
            "Judge the road region only. Do not penalize missing target vehicles, "
            "pedestrians, cones, or parked cars, because those will be added later. "
            "Focus primarily on topology: straight road, T-junction, cross intersection, "
            "multi-branch junction, curve, one-way/two-way organization, visible median "
            "or island, and branch geometry. Use side context such as buildings, shops, "
            "trees, sidewalks, and curbside parking as auxiliary evidence. Crosswalks, "
            "traffic lights, and traffic signs are also auxiliary evidence.\n\n"
            "Output only JSON with these keys: passed, score, road_topology_mismatches, "
            "side_context_mismatches, auxiliary_mismatches, rematch_hints. score must be "
            "0.0-1.0. Use passed=true only when score >= 0.70 and the road topology is "
            "similar enough for later actor placement.\n\n"
            f"User description, if any:\n{user_description or '(none)'}\n\n"
            "Road topology signature:\n"
            f"{json.dumps(topology_signature, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current scene_match JSON:\n"
            f"{json.dumps(scene_match, ensure_ascii=False, sort_keys=True)}"
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
        report["road_topology_mismatches"] = (
            report["road_topology_mismatches"]
            if isinstance(report.get("road_topology_mismatches"), list)
            else []
        )
        report["side_context_mismatches"] = (
            report["side_context_mismatches"]
            if isinstance(report.get("side_context_mismatches"), list)
            else []
        )
        report["auxiliary_mismatches"] = (
            report["auxiliary_mismatches"]
            if isinstance(report.get("auxiliary_mismatches"), list)
            else []
        )
        report["rematch_hints"] = (
            report["rematch_hints"] if isinstance(report.get("rematch_hints"), list) else []
        )
        recommended_stage = str(report.get("recommended_stage") or "match_spawn")
        if recommended_stage not in {"match_spawn", "scene_understanding", "pass"}:
            recommended_stage = "match_spawn"
        if report["passed"]:
            recommended_stage = "pass"
        report["recommended_stage"] = recommended_stage
        return report

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
