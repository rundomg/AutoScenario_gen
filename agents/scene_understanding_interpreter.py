import base64
import json
import os
import re
import sys
from copy import deepcopy

import cv2


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent
from tools.utils import read_file, write_to_file
from tools.structured_pipeline import normalize_scene_understanding
from tools import vehicle_detector
from tools import depth_estimator


class SceneUnderstandingInterpreter(TaskAgent):
    """Image interpreter that emits the structured Scene Understanding DSL."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
You convert one traffic image into Scene Understanding JSON for CARLA existing-map reconstruction.
The JSON is consumed by two downstream modules: map matching and actor layout. Do not write a generic scene caption.

Output JSON only. No markdown fences, no prose.

Top-level schema:
{
  "traffic_subjects": [],
  "key_pairwise_relations": [],
  "road_network": {},
  "actor_layout": {},
  "general_environment": {},
  "metadata": {}
}

1) Map Matching Contract
`road_network.map_matching` is the primary map-matching signature. Fill it directly from visible evidence:
{
  "topology_type": "straight_road | straight_two_way | curve | t_junction | cross_intersection | multi_branch | unknown",
  "junction_visible": true/false,
  "junction_type": "none | t_junction | cross_intersection | roundabout | multi_branch | signalized_intersection | unknown",
  "junction_branches": {"ahead": true/false, "left": true/false, "right": true/false, "known": true/false},
  "target_branch_count": 1/3/4/5,
  "forward_lane_count": integer,
  "opposing_lane_count": integer,
  "driving_lane_count": integer,
  "has_center_median": true/false,
  "has_crosswalk": true/false,
  "has_traffic_light": true/false,
  "has_traffic_sign": true/false,
  "curve_direction": "straight | left | right | unknown",
  "ego_lane_from_right": integer or null,
  "ego_to_junction_distance_m": coarse integer or null
}
Keep `road_network` compact. Prefer the `map_matching` object above; add legacy evidence fields only when they carry visible information needed by downstream placement: road_type, directionality, road_segments, lane_groups, lane_markings, special_road_areas, junctions, roadside_boundaries, control_elements.
For a simple straight road, output at most one `lane_groups` item unless distinct carriageways, branches, bus/parking lanes, or separated road sections are visibly present. Do not create duplicate lane groups just because lane count is uncertain.
Do not infer a raised center median, center island, left branch, right branch, or cross intersection unless it is visible or explicitly stated. Painted lane markings are not a physical center median.
Use road-symmetry reasoning only when the ego-side carriageway has N clearly visible same-direction lanes and there is partial evidence of an opposing roadway or opposing vehicles: set the opposing travel-lane count to N, mark the lane_count_evidence as a symmetric estimate, and do not make this symmetry inference when no opposing-roadway evidence exists.
For straight_road / straight_two_way, set curve_direction="straight" or "unknown" and ego_to_junction_distance_m=null unless a junction/stop line/crosswalk/traffic light is visibly ahead. For curve, set curve_direction to left/right only when the road itself bends; otherwise unknown.
For ego_lane_from_right, count only same-direction travel lanes on ego's own carriageway from the right/curb side: 0=rightmost, 1=next lane left, 2=third lane from right. Never count side-road mouths, crossing-road lanes, parking lanes/rows, shoulders, bus bays, driveways, or opposing lanes.

2) Actor Layout Contract
`traffic_subjects` contains every clearly visible spawnable actor or vehicle row. Each item uses:
id, category, subtype, appearance, visual_confidence, motion_state, turn_intent, heading_relation_to_ego, flow_compliance, lane_index_relation, lane_side_relation, longitudinal_relation, longitudinal_proximity, count, evidence, must_reconstruct, layout_anchor_id, anchor_relation.
Use categories: car, truck, bus, motorcycle, bicycle, pedestrian, cone_group, barrier_group. Do not use parked_vehicle.
Preserve visible vehicle type: box trucks, delivery trucks, lorries, cargo trucks, construction trucks, and semi/tractor units must be category=truck, not generic vehicle or car.
Set must_reconstruct=true for every traffic_subject.

For straight roads, ego-relative fields remain useful:
- lane_index_relation: ego lane=0, first ego-right lane=+1, first ego-left lane=-1.
- lane_side_relation: same_lane, left_lane, right_lane, sidewalk_left, sidewalk_right, crosswalk.
- longitudinal_relation/proximity: ahead/behind/aligned and immediate/near/mid/far.

For junctions, do NOT rely only on ego as the anchor. Add `actor_layout` and per-actor anchors:
{
  "global_anchor": {"type": "ego_approach | junction_center | stop_line | crosswalk", "confidence": "high | medium | low"},
  "anchors": [
    {"id": "ego_approach", "type": "lane_approach"},
    {"id": "junction_center", "type": "junction_center"},
    {"id": "left_arm", "type": "junction_arm", "side": "left"},
    {"id": "right_arm", "type": "junction_arm", "side": "right"},
    {"id": "ahead_arm", "type": "junction_arm", "side": "ahead"},
    {"id": "oncoming_arm", "type": "junction_arm", "side": "oncoming"}
  ]
}
For each actor at or beyond a junction, set layout_anchor_id to the relevant arm and use:
anchor_relation = {"position_along_anchor": "near_mouth | entering | mid_arm | far_arm", "lateral": "same_lane | left_lane | right_lane", "heading": "same_direction | opposite_direction | crossing | turning_left | turning_right", "travel_direction": "toward_junction | away_from_junction | unknown", "longitudinal_order_from_junction": integer or null}.
For actors on a junction arm, infer travel_direction from the actor's visible facing on that arm: front/facing toward the junction means toward_junction; rear/facing away from the junction means away_from_junction. This is what selects the inbound vs outbound lane on that arm, so do not omit it when the vehicle orientation is visible.
When multiple actors occupy the same junction arm or side-road mouth, preserve their queue/row structure: use different position_along_anchor values when visible, include lane_from_right when there are multiple lanes, set longitudinal_order_from_junction with 0 closest to the junction and increasing away from the junction, and add key_pairwise_relations for staggered/overlapping vehicles so downstream placement does not collapse them into one spot.

3) Ego Localization
metadata.ego_localization is strongly preferred:
{
  "forward_reference": {"type": "junction | traffic_light | stop_line | crosswalk | landmark | none", "distance_m": coarse integer, "confidence": "high | medium | low"},
  "ego_to_junction_distance_m": coarse integer or null,
  "ego_lane_from_right": integer,
  "ego_lane_confidence": "high | medium | low"
}
Count only same-direction lanes on ego's own carriageway for ego_lane_from_right. Do not count side roads, driveways, parked rows, or the opposing carriageway.

4) Actor Semantics
- Use an ego-centric frame: ego forward is ahead; ego-left is left; ego-right is right. Do not use image-pixel left/right.
- heading_relation_to_ego: same_direction, opposite_direction, crossing, unknown. A normal oncoming vehicle is opposite_direction and legal.
- In right-hand traffic, a vehicle in the opposing travel lane/left opposing carriageway that is facing the camera/ego must be opposite_direction, not same_direction left_lane. If the front of the vehicle is visible in the opposing lane, treat it as legal oncoming traffic unless there is clear wrong-way evidence.
- turn_intent: left, right, through, none. Judge from the actor's own driving perspective.
- Do not mark stopped or parked vehicles near a side-road mouth as turning_right/turning_left unless their current lane position and heading clearly show an active turn. Static roadside/side-arm vehicles should usually use turn_intent none or through with their own arm anchor.
- flow_compliance: legal, wrong_way, unknown. Use wrong_way only for illegal travel against the occupied lane band.
- motion_state: parked, stopped, moving, unknown.
- Include appearance.color only when visible.
- Do not invent exact actor distances; keep actor positions qualitative. The ego forward reference may use a coarse metre estimate.

5) Pairwise Relations
Use key_pairwise_relations only for high-confidence relations that actor layout needs:
entity_id, other_entity_id, longitudinal_relation, optional longitudinal_gap_band, optional lateral_relation, optional lane_relation, optional constraint_strength, optional confidence, optional evidence.

6) Environment
general_environment is context only: weather_hint, lighting_hint, time_of_day_hint, urban_density, roadside_context_left, roadside_context_right, occlusion_notes, non_spawnable_landmarks. Do not put spawnable actors here.
Use uncertainty conservatively: when evidence is insufficient, use unknown/low confidence instead of guessing.
        """

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _detection_prompt_block(detections: list) -> str:
        """Instruction block describing the detector boxes appended to the prompt."""
        listing = vehicle_detector.format_detections_for_prompt(detections)
        has_depth = any("depth_m" in d for d in detections)
        depth_note = (
            "- `depth≈Nm` is a coarse monocular estimate of how far that vehicle "
            "is from the ego camera (metres, larger = farther). Use it to judge "
            "longitudinal ordering and how far ahead a vehicle is — prefer it over "
            "how low the box sits in the image (a box lower in the image is NOT "
            "necessarily closer). It is approximate: keep `longitudinal_relation` / "
            "`longitudinal_proximity` qualitative; do not copy it as an exact distance.\n"
            if has_depth else
            "- Do NOT use these pixel boxes to infer how far ahead a vehicle is — "
            "a box lower in the image is not necessarily closer. Judge "
            "`longitudinal_relation` / `longitudinal_proximity` from perspective.\n"
        )
        return (
            "\n\nDetected vehicles from an object detector "
            "(reliable evidence of vehicle PRESENCE and image position; "
            "normalized [0,1] coordinates, origin = top-left, "
            "bbox=[x1,y1,x2,y2], center=[cx,cy]). The provided image is annotated "
            "with the same ids:\n"
            f"{listing}\n\n"
            "How to use these detections:\n"
            "- Ensure `traffic_subjects` covers at least every detected vehicle above.\n"
            "- You MAY correct a class label from the image, merge obvious duplicate "
            "boxes on one vehicle, and add clearly visible vehicles the detector missed.\n"
            f"{depth_note}"
            "- The boxes give horizontal image position, not lane membership. Still "
            "infer `lane_index_relation` / `lane_side_relation` / "
            "`heading_relation_to_ego` from perspective as instructed above.\n"
        )

    def refine_request(self, user_request, add_info=None):
        if add_info and add_info.get("merge_scene_understanding"):
            return add_info["merge_prompt"]

        assert add_info and "image_path" in add_info, "Missing image_path"
        image_path = add_info["image_path"]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nUser request:\n{user_request}"

        user_scene_description = str(add_info.get("user_scene_description") or "").strip()
        if user_scene_description:
            prompt += (
                "\n\nAuthoritative user description of this scene "
                "(override your visual judgment for any fact explicitly stated here; "
                "use the image only to fill gaps not covered by this description):\n"
                f"{user_scene_description}"
            )

        # Optional detector pass: feed reliable vehicle boxes + an annotated image
        # to the VLM. Falls back to the original image-only flow when the detector
        # is unavailable or finds nothing.
        vlm_image_path = image_path
        if add_info.get("use_vehicle_detector", True):
            detections = vehicle_detector.detect_vehicles(image_path)
            if detections:
                depth_estimator.attach_depth(image_path, detections)
                prompt += self._detection_prompt_block(detections)
                annotated_path = self._derive_annotated_path(add_info, image_path)
                if vehicle_detector.annotate_image(image_path, detections, annotated_path):
                    vlm_image_path = annotated_path

        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._image_to_base64(vlm_image_path)}"
                },
            },
        ]

    @staticmethod
    def _derive_annotated_path(add_info: dict, image_path: str) -> str:
        """Place the annotated image next to the scene-understanding output when an
        output path is known, otherwise alongside the source image."""
        output_fn = add_info.get("output_fn")
        if output_fn:
            base = output_fn[:-8] if output_fn.endswith("_su.json") else os.path.splitext(output_fn)[0]
            return f"{base}_detections.jpg"
        base, _ = os.path.splitext(image_path)
        return f"{base}_detections.jpg"

    def merge_with_user_description(
        self,
        scene_understanding,
        user_description: str,
        output_fn: str,
    ):
        description = str(user_description or "").strip()
        if not description:
            return scene_understanding

        attempts = 0
        merge_prompt = self._build_merge_prompt(scene_understanding, description)
        while True:
            try:
                self.send_request(
                    "",
                    {
                        "output_fn": output_fn,
                        "merge_scene_understanding": True,
                        "merge_prompt": merge_prompt,
                        "request_label": "Scene understanding merge",
                        "request_timeout": 180,
                        "request_max_tokens": 8000,
                    },
                )
            except Exception as exc:
                fallback = deepcopy(scene_understanding)
                fallback.setdefault("metadata", {})["user_description_applied"] = False
                fallback.setdefault("metadata", {})["user_description_merge_failed"] = True
                fallback.setdefault("metadata", {})["user_description_merge_error"] = str(exc)
                write_to_file(
                    output_fn,
                    json.dumps(fallback, indent=2, sort_keys=True, ensure_ascii=False),
                )
                print(
                    "Scene understanding merge request failed; continuing with "
                    f"the original VLM scene understanding. Error: {exc}"
                )
                return fallback
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                payload.setdefault("metadata", {})["user_description_applied"] = True
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                fallback = deepcopy(scene_understanding)
                fallback.setdefault("metadata", {})["user_description_applied"] = False
                fallback.setdefault("metadata", {})["user_description_merge_failed"] = True
                fallback.setdefault("metadata", {})["user_description_merge_error"] = (
                    validation_error
                )
                write_to_file(
                    output_fn,
                    json.dumps(fallback, indent=2, sort_keys=True, ensure_ascii=False),
                )
                print(
                    "Scene understanding merge failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts; continuing with "
                    "the original VLM scene understanding."
                )
                return fallback
            print(f"Regenerating scene understanding merge... Attempt {attempts + 1}")

    def revise_with_verification_feedback(
        self,
        scene_understanding,
        verification_report,
        user_description: str,
        output_fn: str,
    ):
        attempts = 0
        revision_prompt = self._build_revision_prompt(
            scene_understanding,
            verification_report,
            user_description,
        )
        while True:
            try:
                self.send_request(
                    "",
                    {
                        "output_fn": output_fn,
                        "merge_scene_understanding": True,
                        "merge_prompt": revision_prompt,
                        "request_label": "Scene understanding revision",
                        "request_timeout": 180,
                        "request_max_tokens": 8000,
                    },
                )
            except Exception as exc:
                fallback = deepcopy(scene_understanding)
                fallback.setdefault("metadata", {})[
                    "verification_feedback_applied"
                ] = False
                fallback.setdefault("metadata", {})[
                    "verification_feedback_revision_failed"
                ] = True
                fallback.setdefault("metadata", {})[
                    "verification_feedback_revision_error"
                ] = str(exc)
                write_to_file(
                    output_fn,
                    json.dumps(fallback, indent=2, sort_keys=True, ensure_ascii=False),
                )
                print(
                    "Scene understanding revision request failed; continuing with "
                    f"the previous scene understanding. Error: {exc}"
                )
                return fallback
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                payload.setdefault("metadata", {})["verification_feedback_applied"] = True
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                fallback = deepcopy(scene_understanding)
                fallback.setdefault("metadata", {})[
                    "verification_feedback_applied"
                ] = False
                fallback.setdefault("metadata", {})[
                    "verification_feedback_revision_failed"
                ] = True
                fallback.setdefault("metadata", {})[
                    "verification_feedback_revision_error"
                ] = validation_error
                write_to_file(
                    output_fn,
                    json.dumps(fallback, indent=2, sort_keys=True, ensure_ascii=False),
                )
                print(
                    "Scene understanding revision failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts; continuing with "
                    "the previous scene understanding."
                )
                return fallback
            print(f"Regenerating scene understanding revision... Attempt {attempts + 1}")

    def revise_spawn_coordinates_with_feedback(
        self,
        spawn_payload: dict,
        validation: dict,
        verification_report: dict,
        user_description: str,
        output_fn: str,
    ) -> dict:
        """Ask the LLM to adjust entity coordinates in spawn_payload based on
        constraint violations and VLM repair hints.

        Only location.x/y and rotation.yaw are modified; all other fields
        (blueprint_name, spawn_kind, color, etc.) are preserved unchanged.
        Returns the updated spawn_payload, or the original on repeated failure.
        """
        revision_prompt = self._build_spawn_coordinate_revision_prompt(
            spawn_payload, validation, verification_report, user_description
        )
        attempts = 0
        while True:
            try:
                self.send_request(
                    "",
                    {
                        "output_fn": output_fn,
                        "merge_scene_understanding": True,
                        "merge_prompt": revision_prompt,
                        "request_label": "Spawn coordinate revision",
                        "request_timeout": 120,
                        "request_max_tokens": 4000,
                    },
                )
            except Exception as exc:
                print(
                    "Spawn coordinate revision request failed; keeping current "
                    f"coordinates. Error: {exc}"
                )
                return spawn_payload
            payload, error = self._extract_spawn_coordinate_patch(output_fn, spawn_payload)
            attempts += 1
            if error is None:
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                print(
                    f"Spawn coordinate revision failed after {self.MAX_REGENERATE_ATTEMPTS} "
                    "attempts; keeping current coordinates."
                )
                return spawn_payload
            print(f"Regenerating spawn coordinate revision... Attempt {attempts + 1}")

    @staticmethod
    def _extract_spawn_coordinate_patch(
        file_path: str, original_payload: dict
    ) -> tuple:
        """Parse the LLM response and apply coordinate patches to original_payload."""
        text = read_file(file_path)
        import re as _re
        match = _re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None, "No JSON object found in spawn coordinate revision response."
        try:
            patch = json.loads(match.group())
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON in spawn coordinate revision: {exc}"
        patch_entities = patch.get("entities")
        if not isinstance(patch_entities, list):
            return None, "Spawn coordinate revision response missing 'entities' list."

        patched = dict(original_payload)
        entities_by_id = {
            str(e.get("id")): dict(e)
            for e in (original_payload.get("entities") or [])
        }
        for patch_entity in patch_entities:
            eid = str(patch_entity.get("id") or "")
            if eid not in entities_by_id:
                continue
            target = entities_by_id[eid]
            new_loc = patch_entity.get("location") or {}
            new_rot = patch_entity.get("rotation") or {}
            loc = dict(target.get("location") or {})
            rot = dict(target.get("rotation") or {})
            if "x" in new_loc:
                loc["x"] = float(new_loc["x"])
            if "y" in new_loc:
                loc["y"] = float(new_loc["y"])
            if "z" in new_loc:
                loc["z"] = float(new_loc["z"])
            if "yaw" in new_rot:
                rot["yaw"] = float(new_rot["yaw"])
            target["location"] = loc
            target["rotation"] = rot

        patched["entities"] = list(entities_by_id.values())
        return patched, None

    @staticmethod
    def _build_spawn_coordinate_revision_prompt(
        spawn_payload: dict,
        validation: dict,
        verification_report: dict,
        user_description: str,
    ) -> str:
        failed_pairs = [
            r for r in (validation.get("pairwise_results") or [])
            if str(r.get("status")) == "fail"
        ]
        repair_hints = verification_report.get("repair_hints") or []
        mismatches = verification_report.get("mismatches") or []
        entities_summary = json.dumps(
            spawn_payload.get("entities") or [], indent=2, sort_keys=True,
            ensure_ascii=False,
        )
        failed_summary = json.dumps(failed_pairs, indent=2, sort_keys=True, ensure_ascii=False)
        return (
            "You are adjusting spawn coordinates for a CARLA simulation scene.\n"
            "The current entity placement violates spatial constraints or does not "
            "match the source image.\n\n"
            "Output ONLY a JSON object of this form — no prose, no markdown fences:\n"
            '{"entities": [{"id": "<id>", "location": {"x": <float>, "y": <float>}, '
            '"rotation": {"yaw": <float>}}, ...]}\n\n'
            "Rules:\n"
            "1. Include ONLY entities whose coordinates need to change.\n"
            "2. Modify ONLY location.x, location.y, and rotation.yaw.\n"
            "3. Respect failed pairwise constraints: if entity A must be AHEAD of B "
            "by at least N metres, shift A forward (positive x/y along road direction) "
            "or B backward until the gap is satisfied.\n"
            "4. Do NOT add new entities or change blueprint_name / spawn_kind / color.\n"
            "5. Keep adjustments minimal — correct violations, do not redesign the layout.\n\n"
            f"User description (authoritative reference):\n"
            f"{str(user_description or '').strip() or '(none)'}\n\n"
            f"Constraint violations (pairwise_results with status==fail):\n"
            f"{failed_summary}\n\n"
            f"VLM repair hints:\n"
            f"{json.dumps(repair_hints, ensure_ascii=False)}\n\n"
            f"VLM mismatches:\n"
            f"{json.dumps(mismatches, ensure_ascii=False)}\n\n"
            f"Current entity coordinates:\n"
            f"{entities_summary}"
        )

    @staticmethod
    def _apply_confidence_filter(
        scene_understanding: dict,
        user_description: str,
    ) -> dict:
        """Drop low-confidence background entities not evidenced by user description."""
        if not user_description:
            return scene_understanding
        desc_lower = user_description.lower()

        def _mentioned(entity: dict) -> bool:
            for key in ("category", "subtype"):
                val = str(entity.get(key) or "").lower()
                if val and val in desc_lower:
                    return True
            return False

        filtered = deepcopy(scene_understanding)
        filtered["background_traffic"] = [
            e for e in filtered.get("background_traffic", [])
            if str(e.get("confidence") or "medium") != "low" or _mentioned(e)
        ]
        return filtered

    @staticmethod
    def _build_merge_prompt(scene_understanding, user_description: str) -> str:
        low_conf_ids = [
            e.get("id") for e in scene_understanding.get("traffic_subjects", [])
            if str(e.get("visual_confidence") or "medium") in {"low", "uncertain"}
        ]
        low_conf_ids = [str(i) for i in low_conf_ids if i]

        low_conf_note = (
            f"Low-confidence VLM entities (user description takes full precedence "
            f"for these): {', '.join(low_conf_ids)}\n\n"
            if low_conf_ids else ""
        )

        return (
            "You are merging a VLM-generated traffic scene JSON with a user-provided "
            "description of the same image.\n\n"
            "Output only a complete JSON object using the Scene Understanding DSL schema. "
            "Do not output markdown fences or prose.\n\n"
            "Keep evidence/source text short so the full JSON fits in the response. "
            "Use compact values instead of long explanations.\n\n"
            f"{low_conf_note}"
	            "Priority rules:\n"
	            "1. The user description is authoritative when it conflicts with the VLM JSON.\n"
	            "2. For low-confidence VLM entities listed above, apply user description "
	            "unconditionally — do not preserve the VLM value even for minor attributes "
	            "(color, count, lane_side_relation, heading).\n"
	            "3. Preserve VLM road_network, general_environment, and high-confidence actors "
	            "that the user does not contradict.\n"
	            "4. Add or correct visible vehicles, curbside rows, motorcycles, scooters, pedestrians, "
	            "cones, left/right relations, ahead/behind relations, lane-side relations, and "
	            "pairwise relations explicitly mentioned by the user.\n"
	            "5. Use the ego-centric frame for all left/right/ahead/behind relations: "
	            "ego forward is positive longitudinal, ego-left is negative lateral, and "
	            "ego-right is positive lateral. Do not use image pixel left/right when it "
	            "conflicts with the ego driver's perspective.\n"
	            "6. Preserve decisive road-shape constraints: near crosswalk/zebra crossing, "
	            "absence/presence of a raised center median or center island, straight-road vs "
	            "junction evidence, and continuous curbside parking lanes.\n"
	            "7. Use conservative uncertainty where the user and VLM are both ambiguous.\n"
	            "8. Do not store or repeat the raw user description in metadata.\n\n"
            "Required top-level keys: traffic_subjects, key_pairwise_relations, "
            "road_network, actor_layout, general_environment, metadata. Put all "
            "visible spawnable vehicles in traffic_subjects using real categories "
            "such as car or motorcycle, not parked_vehicle.\n\n"
            "User description (authoritative):\n"
            f"{user_description.strip()}\n\n"
            "VLM scene understanding JSON (reference only — defer to user description "
            "on any conflict):\n"
            f"{json.dumps(scene_understanding, indent=2, sort_keys=True, ensure_ascii=False)}"
        )

    @staticmethod
    def _build_revision_prompt(
        scene_understanding,
        verification_report,
        user_description: str,
    ) -> str:
        return (
            "You are revising a Scene Understanding JSON after comparing the generated "
            "CARLA BEV with the original image. Output only a complete JSON object using "
            "the Scene Understanding DSL schema. Do not output markdown fences or prose.\n\n"
            "Keep evidence/source text short so the full JSON fits in the response. "
            "Use compact values instead of long explanations.\n\n"
	            "Revision rules:\n"
	            "1. Treat user description and verification hard_failures as authoritative.\n"
	            "2. Correct only scene-understanding information that caused visible mismatch.\n"
	            "3. Keep valid road_network, general_environment, and actors not contradicted "
	            "by the verification report.\n"
	            "4. Explicitly encode near crosswalks, absence/presence of center median or "
	            "center island, curbside vehicle rows, cones, and pairwise left/right/near/far "
	            "relations when mentioned.\n"
	            "5. Keep all left/right/ahead/behind revisions in the ego-centric frame: "
	            "ego forward is positive longitudinal, ego-left is negative lateral, and "
	            "ego-right is positive lateral. Do not switch to BEV image or screen coordinates.\n"
	            "6. Do not store or repeat the raw user description in metadata.\n\n"
            "Required top-level keys: traffic_subjects, key_pairwise_relations, "
            "road_network, actor_layout, general_environment, metadata. Put all "
            "visible spawnable vehicles in traffic_subjects using real categories "
            "such as car or motorcycle, not parked_vehicle.\n\n"
            "User description:\n"
            f"{str(user_description or '').strip() or '(none)'}\n\n"
            "Verification report:\n"
            f"{json.dumps(verification_report, indent=2, sort_keys=True, ensure_ascii=False)}\n\n"
            "Current scene understanding JSON:\n"
            f"{json.dumps(scene_understanding, indent=2, sort_keys=True, ensure_ascii=False)}"
        )

    def call_agent(self, user_request, add_info):
        output_fn = add_info["output_fn"]
        attempts = 0
        while True:
            self.send_request(
                user_request,
                {
                    **add_info,
                    "request_label": "Scene understanding",
                    "request_timeout": max(180, add_info.get("request_timeout", 180)),
                    "request_max_tokens": max(
                        4000, int(add_info.get("request_max_tokens", 0) or 0)
                    ),
                },
            )
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Scene understanding generation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            print(f"Regenerating scene understanding... Attempt {attempts + 1}")

    def extract_decision_data(self, file_path: str):
        text = read_file(file_path)
        payload_text = self._extract_json_payload(text)
        if payload_text is None:
            return None, "Scene understanding response does not contain a JSON object."
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            context = self._json_error_context(payload_text, exc.pos)
            return (
                None,
                "Scene understanding JSON is invalid: "
                f"{exc.msg} at line {exc.lineno}, column {exc.colno}. "
                f"Context near error: {context}",
            )

        normalized, validation_error = normalize_scene_understanding(payload)
        if validation_error is not None:
            return None, validation_error

        write_to_file(file_path, json.dumps(normalized, indent=2, sort_keys=True))
        return normalized, None

    @staticmethod
    def _extract_json_payload(text: str) -> str | None:
        fenced = re.search(r"```json\s+(.*?)\s+```", text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return stripped[start : end + 1]
        return None

    @staticmethod
    def _json_error_context(text: str, pos: int, radius: int = 160) -> str:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        snippet = text[start:end].replace("\n", "\\n")
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        return snippet
