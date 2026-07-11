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
        "repair_patches": [],
        "patch_rejections": [],
        "semantic_unrepairable": [],
    }

    PATCH_SCHEMA_VERSION = "actor-repair-v2"
    PATCH_TYPES = {
        "set_lane_target",
        "set_distance_band",
        "set_heading_relation",
        "set_pairwise_relation",
        "resolve_overlap",
        "set_pose_target",
    }
    LANE_TARGETS = {
        "same_lane",
        "left_lane",
        "right_lane",
        "left_parking_lane",
        "right_parking_lane",
        "opposing_lane",
    }
    DISTANCE_BANDS = {"alongside", "near", "mid", "far"}
    HEADING_TARGETS = {"same_direction", "opposite_direction", "crossing"}
    PAIRWISE_TARGETS = {"ahead_of", "behind_other", "left_of_other", "right_of_other"}

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
        bev_image_path = add_info.get("bev_image_path")
        include_bev = bool(
            bev_image_path
            and os.path.isfile(bev_image_path)
            and os.path.abspath(bev_image_path) != os.path.abspath(layout_image_path)
        )
        prompt = self._build_prompt({**add_info, "bev_image_included": include_bev})
        messages = [
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
        if include_bev:
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{self._mime_type(bev_image_path)};base64,"
                            f"{self._encode_image(bev_image_path)}"
                        )
                    },
                }
            )
        return messages

    @staticmethod
    def _build_prompt(add_info: Dict[str, Any]) -> str:
        scene_understanding = add_info.get("scene_understanding") or {}
        scene_match = add_info.get("scene_match") or {}
        spawn_entities = add_info.get("spawn_entities") or {}
        actor_graph_evidence = add_info.get("actor_graph_evidence") or {}
        user_description = str(add_info.get("user_scene_description") or "").strip()
        capture_mode = str(add_info.get("capture_mode") or "bev_fallback")
        bev_image_included = bool(add_info.get("bev_image_included"))
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
            if bev_image_included:
                view_intro += (
                    "The third image is the generated CARLA bird's-eye view (BEV). "
                    "Use the second image for visual framing, visibility, and appearance. "
                    "Use the third image only as supporting evidence for actor existence, "
                    "lane side, and relative spatial layout. Do NOT compare source-image "
                    "appearance directly against the BEV, and do NOT report count_mismatch "
                    "merely because an actor is outside the second image's field of view "
                    "when it is present in the BEV.\n\n"
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
            "mismatches, recommended_stage, repair_hints, repair_schema_version, "
            "repair_patches, semantic_unrepairable. "
            "score must be 0.0-1.0. "
            "recommended_stage must be one of: match_spawn, semantic_unrepairable, pass.\n"
            "Never request a rewrite of scene_understanding. Actor creation, deletion, and "
            "category changes are forbidden. Put confirmed inventory/category problems in "
            "semantic_unrepairable instead of repair_patches.\n"
            "Use passed=true only when score >= 0.70 and no hard failure remains.\n"
            "Set repair_schema_version='actor-repair-v2'. Repair only existing actors. "
            "Each repair_patches item must use set_pose_target and include entity_id, "
            "target_lane, target_longitudinal_m, target_lateral_offset_m, severity, and evidence. "
            "Example: {\"op\":\"set_pose_target\",\"entity_id\":\"det_1\","
            "\"target_lane\":\"right_parking_lane\",\"target_longitudinal_m\":1.5,"
            "\"target_lateral_offset_m\":1.2,\"target_heading_relation\":\"same_direction\","
            "\"severity\":\"high\",\"evidence\":\"near right vehicle is cropped by the camera\"}. "
            "target_longitudinal_m is an exact signed metre distance in the current CARLA ego frame: "
            "ahead is positive and behind is negative. target_lateral_offset_m is a signed metre "
            "offset from the target lane centre: right is positive and left is negative. It is allowed "
            "to place a vehicle near the curb or outside the lane if that best matches the image. "
            "target_anchor_id is optional and may only be one of the actor_graph_evidence anchors. "
            "target_heading_relation is optional; omit it when heading should not change. "
            "Allowed target_lane values: same_lane, left_lane, right_lane, "
            "left_parking_lane, right_parking_lane, opposing_lane. "
            "Allowed target_heading_relation values: same_direction, opposite_direction, crossing. "
            "severity must be high, medium, or low. Do not output a world XY "
            "coordinate, distance bands, pairwise actions, or a source field; the system records provenance itself. "
            "When a vehicle exists in actor_graph_evidence or the BEV but is outside the "
            "ego-view, use set_pose_target or report a visibility mismatch; never report "
            "a count mismatch.\n\n"
            f"User description, if any:\n{user_description or '(none)'}\n\n"
            "Current scene_understanding JSON:\n"
            f"{json.dumps(scene_understanding, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current scene_match JSON:\n"
            f"{json.dumps(scene_match, ensure_ascii=False, sort_keys=True)}\n\n"
            "Current spawn_entities JSON:\n"
            f"{json.dumps(spawn_entities, ensure_ascii=False, sort_keys=True)}\n\n"
            "Authoritative actor_graph_evidence JSON:\n"
            f"{json.dumps(actor_graph_evidence, ensure_ascii=False, sort_keys=True)}"
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
        raw_patches = report.get("repair_patches")
        if raw_patches is None:
            raw_patches = report.get("repair_actions")
        patches, rejections, semantic_unrepairable = cls._normalize_repair_patches(
            raw_patches
        )
        report["repair_schema_version"] = cls.PATCH_SCHEMA_VERSION
        report["repair_patches"] = patches
        report["repair_actions"] = patches
        existing_rejections = report.get("patch_rejections")
        if not isinstance(existing_rejections, list):
            existing_rejections = []
        report["patch_rejections"] = existing_rejections + rejections
        existing_semantic = report.get("semantic_unrepairable")
        if not isinstance(existing_semantic, list):
            existing_semantic = []
        report["semantic_unrepairable"] = existing_semantic + semantic_unrepairable
        recommended_stage = str(report.get("recommended_stage") or "match_spawn")
        if report["semantic_unrepairable"]:
            recommended_stage = "semantic_unrepairable"
        elif recommended_stage not in {"match_spawn", "semantic_unrepairable", "pass"}:
            recommended_stage = "match_spawn"
        if report["passed"]:
            recommended_stage = "pass"
        report["recommended_stage"] = recommended_stage
        return report

    @classmethod
    def _normalize_repair_patches(cls, value: Any) -> tuple:
        if not isinstance(value, list):
            return [], [], []
        normalized = []
        rejected = []
        semantic_unrepairable = []
        for item in value:
            if not isinstance(item, dict):
                rejected.append({"requested_patch": item, "reason": "patch_not_object"})
                continue
            requested = dict(item)
            patch_type = str(item.get("op") or item.get("type") or "").strip()
            if patch_type in {"count_mismatch", "category_mismatch"}:
                semantic_unrepairable.append(
                    {
                        "type": patch_type,
                        "severity": str(item.get("severity") or "high"),
                        "evidence": str(item.get("evidence") or ""),
                        "reason": "actor_inventory_mutation_forbidden",
                    }
                )
                continue
            if patch_type not in cls.PATCH_TYPES:
                rejected.append({"requested_patch": requested, "reason": "unsupported_patch_type"})
                continue
            severity = str(item.get("severity") or "").strip()
            if not severity:
                rejected.append({"requested_patch": requested, "reason": "missing_severity"})
                continue
            if severity not in {"high", "medium", "low"}:
                rejected.append({"requested_patch": requested, "reason": "invalid_severity"})
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            if not entity_id:
                rejected.append({"requested_patch": requested, "reason": "missing_entity_id"})
                continue
            patch = {
                "op": patch_type,
                "entity_id": entity_id,
                "severity": severity,
                "evidence": str(item.get("evidence") or ""),
                "source": "vlm",
            }
            reason = None
            if patch_type == "set_lane_target":
                target = str(item.get("target_lane") or "").strip()
                if target not in cls.LANE_TARGETS:
                    reason = "invalid_target_lane"
                else:
                    patch["target_lane"] = target
            elif patch_type == "set_distance_band":
                target = str(item.get("target_band") or "").strip()
                if target not in cls.DISTANCE_BANDS:
                    reason = "invalid_target_band"
                else:
                    patch["target_band"] = target
            elif patch_type == "set_heading_relation":
                target = str(item.get("target_heading") or "").strip()
                if target not in cls.HEADING_TARGETS:
                    reason = "invalid_target_heading"
                else:
                    patch["target_heading"] = target
            elif patch_type == "set_pairwise_relation":
                reference_id = str(item.get("reference_entity_id") or "").strip()
                target = str(item.get("target_relation") or "").strip()
                if not reference_id:
                    reason = "missing_reference_entity_id"
                elif target not in cls.PAIRWISE_TARGETS:
                    reason = "invalid_target_relation"
                else:
                    patch["reference_entity_id"] = reference_id
                    patch["target_relation"] = target
            elif patch_type == "resolve_overlap":
                reference_id = str(item.get("reference_entity_id") or "").strip()
                if reference_id:
                    patch["reference_entity_id"] = reference_id
            elif patch_type == "set_pose_target":
                lane = str(item.get("target_lane") or "").strip()
                if lane not in cls.LANE_TARGETS:
                    reason = "invalid_target_lane"
                else:
                    try:
                        longitudinal_m = float(item.get("target_longitudinal_m"))
                        lateral_offset_m = float(item.get("target_lateral_offset_m"))
                    except (TypeError, ValueError):
                        reason = "invalid_pose_target_coordinates"
                    else:
                        patch["target_lane"] = lane
                        patch["target_longitudinal_m"] = longitudinal_m
                        patch["target_lateral_offset_m"] = lateral_offset_m
                        anchor_id = str(item.get("target_anchor_id") or "").strip()
                        if anchor_id:
                            patch["target_anchor_id"] = anchor_id
                        heading = str(item.get("target_heading_relation") or "").strip()
                        if heading:
                            if heading not in cls.HEADING_TARGETS:
                                reason = "invalid_target_heading"
                            else:
                                patch["target_heading_relation"] = heading
            if reason:
                rejected.append({"requested_patch": requested, "reason": reason})
                continue
            normalized.append(patch)
        return normalized, rejected, semantic_unrepairable

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
