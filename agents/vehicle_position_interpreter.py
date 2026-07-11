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


class VehiclePositionInterpreter(TaskAgent):
    """VLM helper focused on compact actor position and key layout relations."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
You inspect one traffic image with detected vehicle boxes after the road topology
has already been classified. Your task is vehicle POSITION and key placement
relations, not road topology and not detailed vehicle orientation.

Inputs:
1. Annotated full image with detector ids.
2. Detector list with normalized bbox=[x1,y1,x2,y2] and center=[cx,cy].
3. Road-scene map_matching / lane_groups from the road topology pass.
4. Vehicle orientation payload from the crop pass, including ignored_detections
   soft candidates.

When the detector list is empty, inspect the original full image anyway. Add each
clearly visible vehicle as vlm_1, vlm_2, ... with visual_confidence=low or medium.
An empty detector list is not evidence that the road contains no vehicles.

Output JSON only. Keep it compact. Include vehicle_position_brief first as a
short visible reasoning summary for debugging; do not include hidden
step-by-step reasoning, graph nodes/edges, or long evidence prose:
{
  "vehicle_position_brief": {
    "overall_observation": "one-sentence layout summary",
    "lane_assignment_basis": "short cue summary for lane/parking bands",
    "pairwise_relation_basis": "short cue summary for key relation ordering",
    "uncertainties": ["short uncertainty notes"]
  },
  "ego_lane_review": {
    "ego_lane_from_right": 0,
    "confidence": "high, medium, or low",
    "evidence": "short lane-boundary evidence",
    "agrees_with_road_scene": true
  },
  "traffic_subjects": [
    {
      "id": "det_1",
      "category": "car | truck | bus | motorcycle | bicycle",
      "motion_state": "parked | stopped | moving | unknown",
      "heading_relation_to_ego": "same_direction | opposite_direction | crossing | unknown",
      "lane_side_relation": "same_lane | left_lane | right_lane | opposing_lane | left_parking_lane | right_parking_lane | left_edge | right_edge | crosswalk",
      "lane_index_relation": -1,
      "longitudinal_relation": "ahead | behind | aligned | alongside",
      "longitudinal_proximity": "immediate | near | mid | far",
      "placement_mode_hint": "normal_lane_actor | parking_lane_actor",
      "actor_group_id": "det_1",
      "actor_group_type": "individual_vehicle | parking_row | oncoming_flow",
      "group_order_index": 0,
      "visual_confidence": "high | medium | low"
    }
  ],
  "key_pairwise_relations": [
    {
      "entity_id": "det_2",
      "other_entity_id": "det_3",
      "longitudinal_relation": "ahead_of_other | behind_other | aligned_with_other",
      "longitudinal_gap_band": "overlap | tight | near | mid | far",
      "lateral_relation": "left_of_other | right_of_other | same_lateral_band",
      "lane_relation": "same_lane | same_parking_lane | adjacent_left_lane | adjacent_right_lane | cross_lane"
    }
  ]
}

Rules:
- By default, cover every detector id exactly once in traffic_subjects. You may
  omit a detector only when it appears in ignored_detections.
- Do not omit curbside parked vehicles, motorcycles, or bicycles merely because
  the orientation payload says parked/no impact; keep them when they define
  parking rows, road-edge activity, traffic context, or any plausible scene role.
- Omit fields that are empty/default. For open roads, do not output
  layout_anchor_id or anchor_relation. For junction vehicles, include only
  non-empty layout_anchor_id and anchor_relation.
- Use bbox coordinates as evidence for image ordering. On open/straight roads,
  use bbox center-x and horizontal overlap as auxiliary cues for pairwise
  vehicle left/right ordering, then express the result in the ego-centric road
  frame. Still infer lane/parking bands from the annotated full image,
  road-scene topology, and orientation hints.
- Output only pairwise relations that affect placement: same parking row/queue
  order, overlap/tight spacing, adjacent-lane constraints, or important
  ahead/behind constraints. Do not build a complete all-pairs relation set.
- Do not call a curbside parking row same_lane unless it is truly in a travel lane.
- Use orientation_hints only for heading; do not let them override road topology.
- Independently review ego_lane_from_right from the right road edge, dashed lane
  dividers, and vanishing point. When uncertain, do not infer it from actor positions. Use null
  and low confidence when lane boundaries are unclear.
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
        annotated_path = add_info.get("annotated_path") or add_info["image_path"]
        road_scene = add_info.get("road_scene") or {}
        prompt = self.pre_prompt
        prompt += "\n\nRoad scene JSON:\n"
        prompt += json.dumps(road_scene, ensure_ascii=False, indent=2)
        prompt += _position_branch_block(_is_junction_scene(road_scene))
        prompt += "\n\nDetected vehicles:\n"
        detections = add_info.get("detections") or []
        for det in detections:
            prompt += (
                f"- {det.get('id')} label={det.get('label')} conf={det.get('conf')} "
                f"bbox={det.get('bbox_norm')} center={det.get('center_norm')}\n"
            )
        if not detections:
            prompt += (
                "- No detector boxes survived the balanced detector passes. "
                "Perform a full-image vehicle inventory and assign vlm_N ids.\n"
            )
        prompt += "\n\nOrientation payload including ignored_detections soft candidates:\n"
        prompt += json.dumps(add_info.get("orientation_payload") or {}, ensure_ascii=False, indent=2)
        prompt += "\n\nDetector row hints (hints only; do not blindly turn them into same_lane):\n"
        prompt += json.dumps(add_info.get("row_hints") or [], ensure_ascii=False, indent=2)
        if user_request:
            prompt += f"\nAdditional context:\n{user_request}"

        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._image_to_base64(annotated_path)}"
                },
            },
        ]

    @staticmethod
    def extract_position_payload(response_text: str) -> dict:
        text = str(response_text or "").strip()
        if not text:
            return _empty_position_payload()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return _empty_position_payload()
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return _empty_position_payload()
        if not isinstance(payload, dict):
            return _empty_position_payload()
        subjects = [
            _compact_subject(item)
            for item in payload.get("traffic_subjects", [])
            if isinstance(item, dict)
        ]
        relations = [
            _compact_relation(item)
            for item in payload.get("key_pairwise_relations", [])
            if isinstance(item, dict)
        ]
        if not relations:
            relations = _relations_from_legacy_graph(payload.get("vehicle_position_graph"))
        return {
            "vehicle_position_brief": _compact_position_brief(
                payload.get("vehicle_position_brief")
            ),
            "traffic_subjects": subjects,
            "key_pairwise_relations": relations,
            "ego_lane_review": _compact_ego_lane_review(
                payload.get("ego_lane_review")
            ),
        }


def _empty_position_payload() -> dict:
    return {
        "vehicle_position_brief": {},
        "traffic_subjects": [],
        "key_pairwise_relations": [],
        "ego_lane_review": {},
    }


def _compact_ego_lane_review(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {
        key: value.get(key)
        for key in (
            "ego_lane_from_right",
            "confidence",
            "evidence",
            "agrees_with_road_scene",
        )
        if key in value
    }
    _drop_empty_defaults(result)
    _truncate_evidence(result, max_words=14)
    return result


def _compact_position_brief(brief: object) -> dict:
    if not isinstance(brief, dict):
        return {}
    keep = (
        "scene_kind",
        "overall_observation",
        "lane_assignment_basis",
        "pairwise_relation_basis",
        "parking_row_order",
        "uncertainties",
    )
    compact = {key: brief.get(key) for key in keep if key in brief}
    for key in list(compact):
        value = compact[key]
        if value in ("", None, {}, [], "none"):
            compact.pop(key, None)
            continue
        if isinstance(value, list):
            compact[key] = [
                _truncate_words(str(item), 18)
                for item in value[:4]
                if str(item or "").strip()
            ]
            if not compact[key]:
                compact.pop(key, None)
        else:
            compact[key] = _truncate_words(str(value), 24)
    return compact


def _compact_subject(item: dict) -> dict:
    keep = {
        "id",
        "category",
        "motion_state",
        "heading_relation_to_ego",
        "lane_side_relation",
        "lane_index_relation",
        "longitudinal_relation",
        "longitudinal_proximity",
        "placement_mode_hint",
        "actor_group_id",
        "actor_group_type",
        "group_order_index",
        "group_order_rule",
        "visual_confidence",
        "layout_anchor_id",
        "anchor_relation",
        "evidence",
    }
    compact = {key: item.get(key) for key in keep if key in item}
    _drop_empty_defaults(compact)
    _truncate_evidence(compact)
    return compact


def _compact_relation(item: dict) -> dict:
    keep = {
        "entity_id",
        "other_entity_id",
        "longitudinal_relation",
        "longitudinal_gap_band",
        "lateral_relation",
        "lane_relation",
        "evidence",
    }
    compact = {key: item.get(key) for key in keep if key in item}
    _drop_empty_defaults(compact)
    _truncate_evidence(compact)
    return compact


def _drop_empty_defaults(item: dict) -> None:
    for key in list(item):
        value = item.get(key)
        if value in ("", None, {}, [], "none"):
            item.pop(key, None)


def _truncate_evidence(item: dict, max_words: int = 8) -> None:
    evidence = str(item.get("evidence") or "").strip()
    if not evidence:
        item.pop("evidence", None)
        return
    item["evidence"] = _truncate_words(evidence, max_words)


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return " ".join(words)


def _relations_from_legacy_graph(graph: dict) -> list:
    if not isinstance(graph, dict):
        return []
    relations = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source_id") or "").strip()
        target_id = str(edge.get("target_id") or "").strip()
        if not source_id or not target_id:
            continue
        relations.append(
            _compact_relation(
                {
                    "entity_id": source_id,
                    "other_entity_id": target_id,
                    "longitudinal_relation": edge.get("longitudinal_relation"),
                    "lateral_relation": edge.get("lateral_relation"),
                    "lane_relation": edge.get("lane_relation"),
                    "evidence": edge.get("evidence"),
                }
            )
        )
    return relations


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


def _position_branch_block(is_junction: bool) -> str:
    if is_junction:
        return """

Road-scene branch: JUNCTION. Position logic:
- Assign each vehicle to an anchor arm only because the road-scene agent classified
  this as a junction: ego_approach, right_arm, left_arm, ahead_arm, or oncoming_arm.
- Do not use image-left/image-right or bbox center-x alone to choose left_arm/right_arm.
- Vehicles ahead of ego in the ego approach's adjacent lanes remain layout_anchor_id=ego_approach
  with lane_side_relation=left_lane/right_lane/same_lane; they are not left_arm/right_arm.
- Use left_arm/right_arm only when the vehicle is physically on the cross street or side-road
  branch, with lane markings/curbs/road continuation showing that branch membership.
- Include non-empty layout_anchor_id and anchor_relation for vehicles on junction arms.
- Use anchor_relation.position_along_anchor as near_mouth, mid_arm, or far_arm when visible.
- For actors outside ego_approach, set anchor_relation.lane_from_right in the
  actor arm's local travel frame. Do not reinterpret that slot as ego-left/right.
- Key pairwise relations should focus on same-arm queue order or overlap/tight spacing.
"""
    return """

Road-scene branch: OPEN_ROAD / STRAIGHT_OR_CURVE. Position logic:
- Do not output layout_anchor_id or anchor_relation.
- Do not output arm labels such as right_arm, left_arm, ahead_arm, ego_approach,
  or oncoming_arm.
- Use lane_side_relation/lane_index_relation for lane placement.
- Preserve curbside/parking semantics. Vehicles lined along a curb are parking-lane
  vehicles, not normal same-lane actors.
- For curbside parked vehicles, prefer left_parking_lane/right_parking_lane with
  placement_mode_hint=parking_lane_actor.
- Use detected bbox center-x and horizontal overlap to help decide pairwise
  vehicle left/right relations on the straight/open road, but keep the final
  lateral_relation and lane_side_relation ego-centric.
- Key pairwise relations should focus on parking-row/queue order or overlap/tight spacing.
"""
