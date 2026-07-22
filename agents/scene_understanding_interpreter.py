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
from agents.vehicle_orientation_interpreter import VehicleOrientationInterpreter
from agents.vehicle_position_interpreter import VehiclePositionInterpreter
from tools.utils import read_file, write_to_file
from tools.structured_pipeline import normalize_scene_understanding
from tools.user_text_constraints import (
    apply_user_constraints_to_scene,
    normalize_user_constraint_bundle,
)
from tools import vehicle_detector


class SceneUnderstandingInterpreter(TaskAgent):
    """Image interpreter that emits the structured Scene Understanding DSL."""

    def __init__(self) -> None:
        super().__init__()
        self._last_detection_metadata = {}
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
  "ego_to_junction_distance_m": coarse integer or null,
  "left_parking_presence": true/false,
  "right_parking_presence": true/false
}
Keep `road_network` compact. Prefer the `map_matching` object above; add legacy evidence fields only when they carry visible information needed by downstream placement: road_type, directionality, road_segments, lane_groups, lane_markings, special_road_areas, junctions, roadside_boundaries, control_elements.
For a simple straight road, output at most one `lane_groups` item unless distinct carriageways, branches, bus/parking lanes, or separated road sections are visibly present. Do not create duplicate lane groups just because lane count is uncertain.
Do not infer a raised center median, center island, left branch, right branch, or cross intersection unless it is visible or explicitly stated. Painted lane markings are not a physical center median.
Use junction only for real road-topology intersections or branches, such as T-junctions, cross intersections, roundabouts, or multi-branch road joins. A crossing/sideways vehicle, a crosswalk, a stop line, a traffic light, a driveway, or a roadside entrance does not by itself make the road a junction.
Use road-symmetry reasoning on ordinary straight/open roads. When the ego-side carriageway has N clearly visible same-direction lanes and there is partial evidence of an opposing roadway or opposing vehicles, set the opposing travel-lane count to N and mark the lane_count_evidence as a symmetric estimate; do not make this symmetry inference when no opposing-roadway evidence exists. For undivided two-way straight/open urban roads, also apply parking symmetry: if curbside parked vehicles establish a parking lane/strip on one curb, assume the opposite curb has the same parking lane/strip unless there is clear evidence that the opposite curb cannot support parking. "Not evident" or weak visibility is uncertainty, not a hard absence.
For straight_road / straight_two_way, set curve_direction="straight" or "unknown" and ego_to_junction_distance_m=null unless a junction/stop line/crosswalk/traffic light is visibly ahead. For curve, set curve_direction to left/right only when the road itself bends; otherwise unknown.
On ordinary straight/open roads, curbside parked vehicles lined along the road edge are evidence of a curbside parking lane/parking strip on that side even if lane markings are weak: set the matching left_parking_lane_count or right_parking_lane_count in lane_groups and left_parking_presence or right_parking_presence in map_matching. When no curbside parked vehicles, parking lane, or parking strip is visible on a side, set that side's parking count to 0 and parking_presence=false. Do not treat moving/stopped traffic queues, ordinary shoulders, building edges, bus bays, or driveways as parking lanes.
For ego_lane_from_right, count only same-direction travel lanes on ego's own carriageway from the right/curb side: 0=rightmost, 1=next lane left, 2=third lane from right. Never count side-road mouths, crossing-road lanes, parking lanes/rows, shoulders, bus bays, driveways, or opposing lanes.
Infer ego_lane_from_right from lane boundaries, road edge, and the vanishing point,
not from where other vehicles happen to sit. On a two-lane same-direction road,
if the camera is between the right road edge and the first dashed divider, use 0.
When those boundaries are not visible, use null and ego_lane_confidence=low.

2) Actor Layout Contract
`traffic_subjects` contains every clearly visible spawnable actor or vehicle row. Each item uses:
id, category, subtype, appearance, visual_confidence, motion_state, turn_intent, heading_relation_to_ego, flow_compliance, lane_index_relation, lane_side_relation, longitudinal_relation, longitudinal_proximity, count, evidence, must_reconstruct, layout_anchor_id, anchor_relation, actor_group_type, actor_group_id, group_order_index, group_order_rule, placement_mode_hint.
Use categories: car, truck, bus, motorcycle, bicycle, pedestrian, cone_group, barrier_group. Do not use parked_vehicle.
Preserve visible vehicle type: box trucks, delivery trucks, lorries, cargo trucks, construction trucks, and semi/tractor units must be category=truck, not generic vehicle or car.
Set must_reconstruct=true for every traffic_subject.
Use actor_group_type=individual_vehicle and placement_mode_hint=normal_lane_actor for vehicles.

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
anchor_relation = {"travel_direction": "toward_junction | away_from_junction | unknown"}.
For actors on a junction arm, infer travel_direction from the actor's visible facing on that arm: front/facing toward the junction means toward_junction; rear/facing away from the junction means away_from_junction. This is what selects the inbound vs outbound lane on that arm, so do not omit it when the vehicle orientation is visible.
For side-arm actors, lane_index_relation is not an ego-frame lateral offset.
Put the arm-local lane slot in anchor_relation.lane_from_right, counted from the
actor's travel-right edge: 0=rightmost lane on that arm, 1=one lane left.
When multiple actors occupy the same junction arm or side-road mouth, add key_pairwise_relations for staggered/overlapping vehicles so downstream placement does not collapse them into one spot.

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
- Do not mark vehicles near a side-road mouth as turning_right/turning_left unless their current lane position and heading clearly show an active turn.
- flow_compliance: legal, wrong_way, unknown. Use wrong_way only for illegal travel against the occupied lane band.
- motion_state: parked, stopped, moving, unknown.
- Include appearance.color only when visible.
- Do not invent exact actor distances; keep actor positions qualitative. The ego forward reference may use a coarse metre estimate.

5) Pairwise Relations
Use key_pairwise_relations only for high-confidence relations that actor layout needs:
entity_id, other_entity_id, longitudinal_relation, optional longitudinal_gap_band, optional lateral_relation, optional lane_relation, optional constraint_strength, optional confidence, optional evidence.

6) Environment
general_environment is context only: weather_hint, lighting_hint, time_of_day_hint, urban_density, roadside_context_left, roadside_context_right, occlusion_notes, non_spawnable_landmarks. Do not put spawnable actors here.
Read weather and lighting from the actual image. Night scenes must not default to
day/daylight. Wet reflective pavement should set road_surface_hint=wet even when
active rainfall is uncertain. Use unknown instead of a daylight/clear guess.
Use uncertainty conservatively: when evidence is insufficient, use unknown/low confidence instead of guessing.
        """
        self.road_scene_prompt = """
You inspect one traffic image for road-scene structure.



Road-network contract:
- Fill road_network.map_matching with topology_type, junction_visible, junction_type,
  junction_branches, target_branch_count, forward_lane_count, opposing_lane_count,
  driving_lane_count, has_center_median, has_crosswalk, has_traffic_light,
  has_street_lights,
  curve_direction, ego_lane_from_right, ego_to_junction_distance_m,
  left_parking_presence, and right_parking_presence.
- Fill road_network.lane_groups with the same travel-lane counts plus
  left_parking_lane_count and right_parking_lane_count. Keep map_matching
  parking presence consistent with lane_groups: count > 0 means presence=true;
  count == 0 means presence=false.
- Classify junction only from visible road-geometry branches. Do not classify as
  junction because of traffic lights, crosswalks, stop lines, traffic signs, lane
  arrows, roadside entrances, driveways, parking-lot mouths, or crossing vehicles.
- If the main visible road body is straight and no left/right branch road is clearly
  visible, set topology_type=straight_road, junction_visible=false, junction_type=none,
  and junction_branches={"ahead": false, "left": false, "right": false, "known": false},
  even if traffic lights or crosswalks are visible ahead.
- Do not upgrade a straight road into a junction without visible branch-road geometry.
- On undivided two-way straight/open urban roads, apply parking symmetry: if one
  curb has curbside parked vehicles, set both left_parking_lane_count and
  right_parking_lane_count to 1 unless the opposite curb clearly cannot support
  parking. Do not treat "not evident", weak visibility, or occlusion as hard
  absence.
- When no curbside parking lane is visible, explicitly set
  left_parking_lane_count=0, right_parking_lane_count=0,
  left_parking_presence=false, and right_parking_presence=false.
- Set has_street_lights=true only when one or more roadside street lamps are
  visibly present. Do not count vehicle lamps or traffic signals as street lights.

Actor-layout frame contract:
- Do not output vehicle actors in traffic_subjects.
- Do not output vehicle orientation; that is handled by the vehicle crop pass.
- For junctions, output actor_layout anchors for junction_center, ego_approach,
  left_arm, right_arm, ahead_arm, and oncoming_arm only when applicable.
- For open roads, actor_layout may describe global_anchor and lane-frame hints only.
- metadata.ego_localization is strongly preferred.
- metadata.ego_localization must include ego_lane_confidence and a short
  ego_lane_evidence string. Use ego_lane_from_right=null when lane boundaries
  cannot support a confident count.


Output JSON only:
{
  "road_scene_brief": {
    "scene_kind_observation": "short road topology observation",
    "lane_evidence": ["visible lane-count or divider evidence"],
    "junction_evidence": ["visible branch/signal/stop-line evidence"],
    "parking_lane_evidence": "short parking/curbside observation",
    "uncertainties": ["short uncertainty notes"]
  },
  "traffic_subjects": [],
  "key_pairwise_relations": [],
  "road_network": {
    "map_matching": {
      "topology_type": "straight_road | straight_two_way | curve | t_junction | cross_intersection | multi_branch | unknown",
      "junction_visible": false,
      "junction_type": "none | t_junction | cross_intersection | multi_branch | unknown",
      "junction_branches": {"ahead": false, "left": false, "right": false, "known": false},
      "target_branch_count": 1,
      "forward_lane_count": 1,
      "opposing_lane_count": 1,
      "driving_lane_count": 2,
      "has_center_median": false,
      "has_crosswalk": false,
      "has_traffic_light": false,
      "has_street_lights": false,
      "curve_direction": "straight | left | right | unknown",
      "ego_lane_from_right": 0,
      "ego_to_junction_distance_m": null,
      "left_parking_presence": false,
      "right_parking_presence": false
    },
    "lane_groups": [
      {
        "forward_lane_count": 1,
        "opposing_lane_count": 1,
        "left_parking_lane_count": 0,
        "right_parking_lane_count": 0,
        "lane_count_evidence": "short evidence for travel and parking lane counts",
        "lane_count_confidence": "low | medium | high"
      }
    ]
  },
  "actor_layout": {},
  "general_environment": {
    "weather_hint": "clear | rain | fog | snow",
    "lighting_hint": "daylight | night | twilight",
    "time_of_day_hint": "day | night | dawn | dusk",
    "road_surface_hint": "dry | wet | snow",
    "urban_density": "urban | suburban | rural"
  },
  "metadata": {}
}
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
            "- Do NOT use these pixel boxes to infer exact depth. Judge "
            "`longitudinal_relation` / `longitudinal_proximity` from perspective, "
            "queue order, lane context, stop lines, and junction anchors.\n"
            "- The boxes give horizontal image position, not lane membership. Still "
            "infer `lane_index_relation` / `lane_side_relation` / "
            "`heading_relation_to_ego` from perspective as instructed above.\n"
        )

    @staticmethod
    def _orientation_prompt_block(orientation_hints: list, row_hints: list) -> str:
        if not orientation_hints and not row_hints:
            return ""
        block = "\n\nVehicle orientation and row/group hints from a dedicated crop pass:\n"
        if orientation_hints:
            block += json.dumps(
                {"vehicle_orientation_hints": orientation_hints},
                ensure_ascii=False,
                indent=2,
            )
            block += "\n"
        if row_hints:
            block += (
                "\nDetector row candidates (hints only for relative ordering; keep vehicles as normal actors):\n"
                f"{vehicle_detector.format_row_hints_for_prompt(row_hints)}\n"
            )
        block += (
            "\nHow to use these hints:\n"
            "- For detected actors with matching det ids, prefer high/medium confidence "
            "orientation hints over a fresh guess from the full image.\n"
            "- Keep vehicles as actor_group_type=individual_vehicle and "
            "placement_mode_hint=normal_lane_actor.\n"
            "- Keep uncertain conflicts explicit with visual_confidence=low or "
            "heading_relation_to_ego=unknown instead of forcing a guess.\n"
            "- Preserve these hints in metadata.vehicle_orientation_hints when possible.\n"
        )
        return block

    def refine_request(self, user_request, add_info=None):
        if add_info and add_info.get("merge_scene_understanding"):
            return add_info["merge_prompt"]

        assert add_info and "image_path" in add_info, "Missing image_path"
        image_path = add_info["image_path"]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        road_scene_only = bool(add_info.get("road_scene_only"))
        prompt = self.road_scene_prompt if road_scene_only else self.pre_prompt
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
        if (not road_scene_only) and add_info.get("use_vehicle_detector", True):
            detections, _detection_metadata = vehicle_detector.detect_vehicles_balanced(
                image_path
            )
            if detections:
                detections = vehicle_detector.select_representative_detections(detections)
                row_hints = vehicle_detector.infer_row_group_hints(detections)
                annotated_path = self._derive_annotated_path(add_info, image_path)
                if vehicle_detector.annotate_image(image_path, detections, annotated_path):
                    vlm_image_path = annotated_path
                crop_dir, crop_base = self._derive_crop_output(add_info, image_path)
                detections_with_crops = vehicle_detector.write_detection_crops(
                    image_path,
                    detections,
                    output_dir=crop_dir,
                    base_name=crop_base,
                )
                if detections_with_crops:
                    detections = detections_with_crops
                orientation_hints = self._build_vehicle_orientation_hints(
                    image_path=image_path,
                    annotated_path=vlm_image_path,
                    detections=detections,
                )
                prompt += self._detection_prompt_block(detections)
                prompt += self._orientation_prompt_block(orientation_hints, row_hints)

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

    @staticmethod
    def _derive_crop_output(add_info: dict, image_path: str) -> tuple[str, str]:
        output_fn = add_info.get("output_fn")
        if output_fn:
            base = output_fn[:-8] if output_fn.endswith("_su.json") else os.path.splitext(output_fn)[0]
            return os.path.dirname(base) or ".", os.path.basename(base)
        return os.path.dirname(image_path) or ".", os.path.splitext(os.path.basename(image_path))[0]

    @staticmethod
    def _build_vehicle_orientation_hints(
        *,
        image_path: str,
        annotated_path: str,
        detections: list,
    ) -> list:
        if not detections:
            return []
        try:
            agent = VehicleOrientationInterpreter()
            response = agent.send_request(
                "",
                {
                    "image_path": image_path,
                    "annotated_path": annotated_path,
                    "detections": detections,
                    "request_label": "vehicle-orientation",
                    "request_max_tokens": 1200,
                },
            )
            return VehicleOrientationInterpreter.extract_orientation_hints(response)
        except Exception as exc:  # noqa: BLE001 - orientation is optional evidence
            print(f"[vehicle_orientation] skipped orientation hints: {exc}")
            return []

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
                metadata = payload.setdefault("metadata", {})
                bundle, constraint_errors = normalize_user_constraint_bundle(
                    metadata.get("user_constraints")
                )
                if constraint_errors or not bundle.get("constraints"):
                    detail = "; ".join(constraint_errors) or "constraint list is empty"
                    validation_error = (
                        "User-description merge did not preserve explicit statements in "
                        f"metadata.user_constraints: {detail}"
                    )
                else:
                    metadata["user_constraints"] = bundle
                    payload, application = apply_user_constraints_to_scene(payload)
                    if application.get("unresolved"):
                        validation_error = (
                            "User-description constraints reference unresolved targets: "
                            + ", ".join(application["unresolved"])
                        )
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
                original_constraints = (
                    (scene_understanding.get("metadata") or {}).get("user_constraints")
                )
                if original_constraints:
                    payload.setdefault("metadata", {})["user_constraints"] = deepcopy(
                        original_constraints
                    )
                    payload, _application = apply_user_constraints_to_scene(payload)
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
            "4. Add or correct visible vehicles, motorcycles, scooters, pedestrians, "
	            "cones, left/right relations, ahead/behind relations, lane-side relations, and "
	            "pairwise relations explicitly mentioned by the user.\n"
            "5. Use the ego-centric frame for all left/right/ahead/behind relations: "
            "ego forward is positive longitudinal, ego-left is negative lateral, and "
            "ego-right is positive lateral. Do not use image pixel left/right when it "
            "conflicts with the ego driver's perspective.\n"
            "6. If the user says actors are in ego's same/left/right lane or ahead of ego, "
            "keep them on layout_anchor_id=ego_approach in junction scenes; encode their "
            "lateral position with lane_side_relation/lane_index_relation, not with "
            "left_arm/right_arm. Use left_arm/right_arm only for actors physically on a "
            "cross street or side-road branch.\n"
	            "7. Preserve decisive road-shape constraints: near crosswalk/zebra crossing, "
	            "absence/presence of a raised center median or center island, straight-road vs "
	            "junction evidence, and continuous curbside parking lanes.\n"
	            "8. Use conservative uncertainty where the user and VLM are both ambiguous.\n"
	            "9. Do not store or repeat the raw user description in metadata.\n"
            "10. Preserve every explicit user statement independently in "
            "metadata.user_constraints. This side-channel is mandatory and must contain "
            "only facts stated by the user, never facts inferred only from the image. "
            "Use source=user_text and strength=hard for definite statements; use soft only "
            "for explicitly uncertain wording. Intermediate VLM confidence is not a gate "
            "for these constraints.\n\n"
            "metadata.user_constraints schema:\n"
            '{"schema_version":"user-constraints-v1","source":"user_text",'
            '"constraints":[{"id":"user_...","target":"scene|entity|relation",'
            '"entity_id":"det id when target is entity/relation",'
            '"reference_entity_id":"ego or det id when target is relation",'
            '"path":"canonical field path","value":"JSON value",'
            '"strength":"hard|soft","evidence":"short user-stated phrase",'
            '"tolerance":1.5}]}\n'
            "For scene facts use paths under road_network.map_matching, "
            "road_network.lane_groups[0], actor_layout, or general_environment. "
            "For actor facts use category, subtype, count, appearance.color, motion_state, "
            "turn_intent, heading_relation_to_ego, lane_side_relation, "
            "lane_index_relation, longitudinal_relation, longitudinal_proximity, "
            "longitudinal_m, layout_anchor_id, or anchor_relation.lane_from_right. "
            "Encode an exact user distance such as '5m ahead' as entity path "
            "longitudinal_m with numeric value 5 and tolerance 1.5. When the user gives "
            "absolute right-to-left lane slots, preserve ego as scene path "
            "road_network.map_matching.ego_lane_from_right and each actor as entity path "
            "anchor_relation.lane_from_right. Ego-relative left is lane_index_relation=-1 "
            "and right is +1. Pairwise facts that cannot be represented as entity fields "
            "use target=relation.\n\n"
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
        if add_info.get("split_scene_agents", True) and not add_info.get("merge_scene_understanding"):
            return self._call_split_scene_agents(user_request, add_info)
        return self._call_legacy_scene_agent(user_request, add_info)

    def _call_legacy_scene_agent(self, user_request, add_info):
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

    def _call_split_scene_agents(self, user_request, add_info):
        output_fn = add_info["output_fn"]
        image_path = add_info["image_path"]
        base = output_fn[:-8] if output_fn.endswith("_su.json") else os.path.splitext(output_fn)[0]
        road_output_fn = f"{base}_road_scene.json"
        orientation_output_fn = f"{base}_orientation.json"
        position_output_fn = f"{base}_position.json"

        detections, row_hints, annotated_path = self._prepare_detection_evidence(
            image_path=image_path,
            output_fn=output_fn,
            add_info=add_info,
        )
        road_payload = self._call_road_scene_agent(
            user_request,
            {
                **add_info,
                "output_fn": road_output_fn,
                "use_vehicle_detector": False,
                "road_scene_only": True,
            },
        )
        orientation_payload = self._call_vehicle_orientation_agent(
            user_request=user_request,
            image_path=image_path,
            annotated_path=annotated_path,
            detections=detections,
            road_payload=road_payload,
            output_fn=orientation_output_fn,
            request_timeout=max(180, add_info.get("request_timeout", 180)),
        )
        position_payload = self._call_vehicle_position_agent(
            user_request=user_request,
            image_path=image_path,
            annotated_path=annotated_path,
            detections=detections,
            row_hints=row_hints,
            road_payload=road_payload,
            orientation_payload=orientation_payload,
            output_fn=position_output_fn,
            request_timeout=max(180, add_info.get("request_timeout", 180)),
        )
        final_payload = self._merge_split_scene_payloads(
            road_payload=road_payload,
            detections=detections,
            row_hints=row_hints,
            orientation_payload=orientation_payload,
            position_payload=position_payload,
            detection_metadata=self._last_detection_metadata,
        )
        normalized, validation_error = normalize_scene_understanding(final_payload)
        if validation_error is not None:
            raise RuntimeError(f"Split scene understanding merge failed: {validation_error}")
        write_to_file(output_fn, json.dumps(normalized, indent=2, sort_keys=True))
        return normalized

    def _call_road_scene_agent(self, user_request, add_info):
        output_fn = add_info["output_fn"]
        attempts = 0
        while True:
            self.send_request(
                user_request,
                {
                    **add_info,
                    "request_label": "Road scene understanding",
                    "request_timeout": max(180, add_info.get("request_timeout", 180)),
                    "request_max_tokens": max(
                        3500, int(add_info.get("request_max_tokens", 0) or 0)
                    ),
                },
            )
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Road scene understanding failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            print(f"Regenerating road scene understanding... Attempt {attempts + 1}")

    def _prepare_detection_evidence(self, *, image_path: str, output_fn: str, add_info: dict):
        if not add_info.get("use_vehicle_detector", True):
            return [], [], image_path
        detections, detection_metadata = vehicle_detector.detect_vehicles_balanced(
            image_path
        )
        self._last_detection_metadata = detection_metadata
        if not detections:
            return [], [], image_path
        detections = vehicle_detector.select_representative_detections(detections)
        self._last_detection_metadata["selected_detection_count"] = len(detections)
        self._last_detection_metadata["selected_detections"] = [
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "conf": item.get("conf"),
                "bbox_norm": item.get("bbox_norm"),
            }
            for item in detections
        ]
        row_hints = vehicle_detector.infer_row_group_hints(detections)
        annotated_path = self._derive_annotated_path({"output_fn": output_fn}, image_path)
        if not vehicle_detector.annotate_image(image_path, detections, annotated_path):
            annotated_path = image_path
        crop_dir, crop_base = self._derive_crop_output({"output_fn": output_fn}, image_path)
        detections_with_crops = vehicle_detector.write_detection_crops(
            image_path,
            detections,
            output_dir=crop_dir,
            base_name=crop_base,
        )
        if detections_with_crops:
            detections = detections_with_crops
        return detections, row_hints, annotated_path

    def _call_vehicle_orientation_agent(
        self,
        *,
        user_request: str = "",
        image_path: str,
        annotated_path: str,
        detections: list,
        road_payload: dict,
        output_fn: str,
        request_timeout: float,
    ) -> dict:
        if not detections:
            payload = {
                "vehicle_orientation_brief": {},
                "ignored_detections": [],
                "vehicle_orientation_hints": [],
            }
            write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
            return payload
        try:
            agent = VehicleOrientationInterpreter()
            response = agent.send_request(
                user_request or "",
                {
                    "image_path": image_path,
                    "annotated_path": annotated_path,
                    "detections": detections,
                    "road_scene": road_payload,
                    "request_label": "vehicle-orientation",
                    "request_timeout": request_timeout,
                    "request_max_tokens": 1800,
                },
            )
            payload = VehicleOrientationInterpreter.extract_orientation_payload(response)
            payload = VehicleOrientationInterpreter.sanitize_for_road_scene(
                payload,
                road_payload,
            )
        except Exception as exc:  # noqa: BLE001 - keep road-only reconstruction alive
            print(f"[vehicle_orientation] skipped orientation agent: {exc}")
            payload = {
                "vehicle_orientation_brief": {
                    "overall_observation": "",
                    "uncertainties": [str(exc)],
                },
                "ignored_detections": [],
                "vehicle_orientation_hints": [],
            }
        write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def _call_vehicle_position_agent(
        self,
        *,
        user_request: str = "",
        image_path: str,
        annotated_path: str,
        detections: list,
        row_hints: list,
        road_payload: dict,
        orientation_payload: dict,
        output_fn: str,
        request_timeout: float,
    ) -> dict:
        try:
            agent = VehiclePositionInterpreter()
            response = agent.send_request(
                user_request or "",
                {
                    "image_path": image_path,
                    "annotated_path": annotated_path,
                    "detections": detections,
                    "road_scene": road_payload,
                    "orientation_payload": orientation_payload,
                    "row_hints": row_hints,
                    "inventory_without_detections": not bool(detections),
                    "request_label": "vehicle-position",
                    "request_timeout": request_timeout,
                    "request_max_tokens": 3200,
                },
            )
            raw_output_fn = f"{os.path.splitext(output_fn)[0]}_raw.txt"
            write_to_file(raw_output_fn, response)
            payload = VehiclePositionInterpreter.extract_position_payload(response)
            if detections and not payload.get("traffic_subjects"):
                raise RuntimeError(
                    "Vehicle position agent returned no traffic subjects; "
                    "aborting instead of falling back to detector geometry."
                )
        except Exception as exc:  # noqa: BLE001 - keep reconstruction alive
            if "Vehicle position agent returned no traffic subjects" in str(exc):
                raise
            print(f"[vehicle_position] skipped position agent: {exc}")
            payload = self._fallback_vehicle_position_payload(
                detections=detections,
                row_hints=row_hints,
                orientation_payload=orientation_payload,
                road_payload=road_payload,
            )
        payload = self._apply_user_ego_lane_layout_constraints(
            payload,
            user_request,
        )
        write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
        return payload

    @staticmethod
    def _merge_split_scene_payloads(
        *,
        road_payload: dict,
        detections: list,
        row_hints: list,
        orientation_payload: dict,
        position_payload: dict,
        detection_metadata: dict = None,
    ) -> dict:
        merged = deepcopy(road_payload)
        hints = orientation_payload.get("vehicle_orientation_hints") or []
        hints_by_id = {str(h.get("det_id")): h for h in hints if isinstance(h, dict)}
        ignored_candidates = _normalize_ignored_detection_candidates(
            orientation_payload.get("ignored_detections")
        )
        road_network = merged.get("road_network") or {}
        map_matching = road_network.get("map_matching") or {}
        is_junction = bool(map_matching.get("junction_visible")) or str(
            map_matching.get("topology_type") or ""
        ) in {"t_junction", "cross_intersection", "multi_branch", "roundabout"}
        has_signal = bool(map_matching.get("has_traffic_light"))
        row_membership = SceneUnderstandingInterpreter._row_membership(
            row_hints,
            hints_by_id,
            is_junction=is_junction,
            has_signal=has_signal,
        )
        subjects = [
            item
            for item in (position_payload or {}).get("traffic_subjects", [])
            if isinstance(item, dict)
        ]
        if not subjects:
            subjects = SceneUnderstandingInterpreter._fallback_subjects_from_detections(
                detections,
                hints_by_id,
                row_membership,
                is_junction=is_junction,
            )
        if is_junction:
            # The position pass owns arm membership and arm-local position, while
            # the orientation pass owns whether an actor travels toward or away
            # from the junction.  Preserve the position fields but always carry a
            # concrete orientation result into the relation consumed by junction
            # lane reprojection.  Without this merge, right/ahead arms fall back
            # to opposite motion defaults and select the wrong inbound/outbound
            # lane band.
            for subject in subjects:
                subject_id = str(subject.get("id") or "").strip()
                hint = hints_by_id.get(subject_id) or {}
                travel_direction = str(
                    hint.get("junction_travel_direction") or ""
                ).strip().lower()
                if travel_direction not in {
                    "toward_junction",
                    "away_from_junction",
                }:
                    continue
                anchor_relation = subject.get("anchor_relation")
                if not isinstance(anchor_relation, dict):
                    anchor_relation = {}
                    subject["anchor_relation"] = anchor_relation
                anchor_relation["travel_direction"] = travel_direction
        merged["traffic_subjects"] = subjects
        subject_ids = {
            str(item.get("id") or "").strip()
            for item in subjects
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        }
        position_relations = [
            item
            for item in (position_payload or {}).get("key_pairwise_relations", [])
            if isinstance(item, dict)
        ]
        if not position_relations:
            position_relations = SceneUnderstandingInterpreter._relations_from_position_graph(
                (position_payload or {}).get("vehicle_position_graph")
            )
        position_relations = [
            relation
            for relation in position_relations
            if _relation_refs_existing_subjects(relation, subject_ids)
        ]
        row_relations = [
            relation
            for relation in SceneUnderstandingInterpreter._pairwise_from_rows(row_membership)
            if _relation_refs_existing_subjects(relation, subject_ids)
        ]
        for relation in position_relations:
            relation.setdefault("constraint_strength", "soft")
        merged["key_pairwise_relations"] = (
            position_relations
            if position_relations
            else row_relations
        )
        merged.setdefault("background_traffic", [])
        merged.setdefault("actor_layout", {})
        merged.setdefault("general_environment", {})
        metadata = merged.setdefault("metadata", {})
        if "road_scene_brief" in merged:
            metadata["road_scene_brief"] = merged.pop("road_scene_brief")
        else:
            metadata.setdefault(
                "road_scene_brief",
                (road_payload.get("metadata") or {}).get("road_scene_brief", {}),
            )
        metadata["vehicle_orientation_brief"] = orientation_payload.get(
            "vehicle_orientation_brief", {}
        )
        metadata["vehicle_orientation_hints"] = hints
        metadata["ignored_detection_candidates"] = ignored_candidates
        metadata["vehicle_position_brief"] = (position_payload or {}).get(
            "vehicle_position_brief", {}
        )
        ego_lane_review = (position_payload or {}).get("ego_lane_review") or {}
        metadata["ego_lane_review"] = deepcopy(ego_lane_review)
        if isinstance(ego_lane_review, dict):
            reviewed_lane = ego_lane_review.get("ego_lane_from_right")
            reviewed_confidence = str(
                ego_lane_review.get("confidence") or ""
            ).lower()
            road_lane = map_matching.get("ego_lane_from_right")
            ego_localization = metadata.get("ego_localization")
            if not isinstance(ego_localization, dict):
                ego_localization = {}
                metadata["ego_localization"] = ego_localization
            if (
                reviewed_lane is not None
                and reviewed_confidence in {"high", "medium"}
            ):
                try:
                    lanes_agree = int(reviewed_lane) == int(road_lane)
                except (TypeError, ValueError):
                    lanes_agree = road_lane is None
                if lanes_agree:
                    ego_localization["ego_lane_from_right"] = int(reviewed_lane)
                    ego_localization["ego_lane_confidence"] = reviewed_confidence
                    ego_localization["ego_lane_evidence"] = str(
                        ego_lane_review.get("evidence") or "position-agent agreement"
                    )
                else:
                    map_matching["ego_lane_from_right"] = None
                    ego_localization["ego_lane_from_right"] = None
                    ego_localization["ego_lane_confidence"] = "low"
                    ego_localization["ego_lane_evidence"] = (
                        f"road/position disagreement: {road_lane} vs {reviewed_lane}"
                    )
        metadata["vehicle_position_graph"] = (
            SceneUnderstandingInterpreter._derive_vehicle_position_graph(
                subjects,
                merged["key_pairwise_relations"],
            )
        )
        metadata["detector_row_hints"] = row_hints
        metadata["vehicle_detection"] = deepcopy(detection_metadata or {})
        metadata["prompt_version"] = "vlm-static-v2-balanced"
        metadata["split_scene_agents"] = True
        return merged

    @staticmethod
    def _merge_road_and_vehicle_orientation(
        *,
        road_payload: dict,
        detections: list,
        row_hints: list,
        orientation_payload: dict,
    ) -> dict:
        position_payload = SceneUnderstandingInterpreter._fallback_vehicle_position_payload(
            detections=detections,
            row_hints=row_hints,
            orientation_payload=orientation_payload,
            road_payload=road_payload,
        )
        return SceneUnderstandingInterpreter._merge_split_scene_payloads(
            road_payload=road_payload,
            detections=detections,
            row_hints=row_hints,
            orientation_payload=orientation_payload,
            position_payload=position_payload,
        )

    @staticmethod
    def _user_describes_ego_lane_layout(user_request: str) -> bool:
        text = str(user_request or "").lower()
        compact = "".join(text.split())
        if not compact:
            return False
        front_terms = (
            "ego前",
            "ego车前",
            "ego前方",
            "自车前",
            "自车前方",
            "本车前",
            "本车前方",
            "主车前",
            "主车前方",
            "在前面",
            "前方",
            "ahead of ego",
            "in front of ego",
            "in ego front",
        )
        lane_terms = (
            "同车道",
            "左车道",
            "右车道",
            "ego左",
            "ego右",
            "same_lane",
            "left_lane",
            "right_lane",
            "same lane",
            "left lane",
            "right lane",
        )
        has_front = any(term in compact or term in text for term in front_terms)
        has_lane = any(term in compact or term in text for term in lane_terms)
        return has_front and has_lane

    @staticmethod
    def _apply_user_ego_lane_layout_constraints(payload: dict, user_request: str) -> dict:
        if not SceneUnderstandingInterpreter._user_describes_ego_lane_layout(user_request):
            return payload
        if not isinstance(payload, dict):
            return payload
        subjects = payload.get("traffic_subjects")
        if not isinstance(subjects, list):
            return payload
        ego_lane_sides = {"same_lane", "left_lane", "right_lane"}
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            lane_side = str(subject.get("lane_side_relation") or "").strip()
            layout_anchor = str(subject.get("layout_anchor_id") or "").strip()
            if lane_side not in ego_lane_sides:
                continue
            if layout_anchor in {"left_arm", "right_arm"}:
                subject["layout_anchor_id"] = "ego_approach"
            if subject.get("heading_relation_to_ego") == "crossing":
                subject["heading_relation_to_ego"] = "same_direction"
            if subject.get("longitudinal_relation") in {"alongside", "aligned"}:
                subject["longitudinal_relation"] = "ahead"
        return payload

    @staticmethod
    def _fallback_subjects_from_detections(
        detections: list,
        hints_by_id: dict,
        row_membership: dict,
        *,
        is_junction: bool,
    ) -> list:
        subjects = []
        for det in detections:
            det_id = str(det.get("id") or "")
            if not det_id:
                continue
            subjects.append(
                SceneUnderstandingInterpreter._subject_from_detection(
                    det,
                    hints_by_id.get(det_id, {}),
                    row_membership.get(det_id, {}),
                    is_junction=is_junction,
                )
            )
        return subjects

    @staticmethod
    def _fallback_vehicle_position_payload(
        *,
        detections: list,
        row_hints: list,
        orientation_payload: dict,
        road_payload: dict,
    ) -> dict:
        hints = (orientation_payload or {}).get("vehicle_orientation_hints") or []
        hints_by_id = {str(h.get("det_id")): h for h in hints if isinstance(h, dict)}
        map_matching = ((road_payload or {}).get("road_network") or {}).get("map_matching") or {}
        is_junction = bool(map_matching.get("junction_visible")) or str(
            map_matching.get("topology_type") or ""
        ) in {"t_junction", "cross_intersection", "multi_branch", "roundabout"}
        row_membership = SceneUnderstandingInterpreter._row_membership(
            row_hints,
            hints_by_id,
            is_junction=is_junction,
            has_signal=bool(map_matching.get("has_traffic_light")),
        )
        subjects = SceneUnderstandingInterpreter._fallback_subjects_from_detections(
            detections,
            hints_by_id,
            row_membership,
            is_junction=is_junction,
        )
        relations = SceneUnderstandingInterpreter._pairwise_from_rows(row_membership)
        return {
            "vehicle_position_brief": {
                "scene_kind": "junction" if is_junction else "open_road",
                "overall_observation": "fallback detector geometry layout",
                "uncertainties": ["vehicle position agent unavailable"],
            },
            "traffic_subjects": subjects,
            "key_pairwise_relations": relations,
        }

    @staticmethod
    def _relations_from_position_graph(graph: dict) -> list:
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
                {
                    "entity_id": source_id,
                    "other_entity_id": target_id,
                    "longitudinal_relation": edge.get("longitudinal_relation"),
                    "lateral_relation": edge.get("lateral_relation"),
                    "lane_relation": edge.get("lane_relation"),
                    "constraint_strength": "soft",
                    "confidence": str(edge.get("confidence") or "medium"),
                    "evidence": str(edge.get("evidence") or "vehicle position graph"),
                }
            )
        return relations

    @staticmethod
    def _derive_vehicle_position_graph(subjects: list, relations: list) -> dict:
        def compact(item: dict) -> dict:
            return {key: value for key, value in item.items() if value not in ("", None, {}, [])}

        nodes = []
        for subject in subjects or []:
            if not isinstance(subject, dict):
                continue
            det_id = str(subject.get("id") or "").strip()
            if not det_id:
                continue
            nodes.append(
                compact(
                    {
                        "det_id": det_id,
                        "lane_band": subject.get("layout_anchor_id")
                        or subject.get("lane_side_relation"),
                        "motion_state": subject.get("motion_state"),
                        "longitudinal_band": subject.get("longitudinal_proximity"),
                        "confidence": subject.get("visual_confidence"),
                        "evidence": subject.get("evidence"),
                    }
                )
            )

        edges = []
        for relation in relations or []:
            if not isinstance(relation, dict):
                continue
            source_id = str(relation.get("entity_id") or "").strip()
            target_id = str(relation.get("other_entity_id") or "").strip()
            if not source_id or not target_id:
                continue
            edges.append(
                compact(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "longitudinal_relation": relation.get("longitudinal_relation"),
                        "lateral_relation": relation.get("lateral_relation"),
                        "lane_relation": relation.get("lane_relation"),
                        "confidence": relation.get("confidence"),
                        "evidence": relation.get("evidence"),
                    }
                )
            )
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _row_membership(row_hints: list, hints_by_id: dict, *, is_junction: bool, has_signal: bool) -> dict:
        membership = {}
        for index, row in enumerate(row_hints or []):
            det_ids = [str(v) for v in row.get("det_ids", []) if v]
            if len(det_ids) < 2:
                continue
            group_type = "individual_vehicle"
            placement = "normal_lane_actor"
            group_id = f"{group_type}_{index + 1}"
            for order, det_id in enumerate(det_ids):
                membership[det_id] = {
                    "actor_group_type": group_type,
                    "actor_group_id": group_id,
                    "group_order_index": order,
                    "group_order_rule": row.get("order_rule_hint") or "",
                    "placement_mode_hint": placement,
                }
        return membership

    @staticmethod
    def _subject_from_detection(det: dict, hint: dict, group_info: dict, *, is_junction: bool) -> dict:
        det_id = str(det.get("id") or "")
        label = str(det.get("label") or "car")
        bbox = det.get("bbox_norm") or [0.4, 0.4, 0.6, 0.6]
        center = det.get("center_norm") or [
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        ]
        cx, cy = float(center[0]), float(center[1])
        lane_index = -1 if cx < 0.38 else 1 if cx > 0.62 else 0
        lane_side = "same_lane" if lane_index == 0 else "left_lane" if lane_index < 0 else "right_lane"
        actor_group_type = group_info.get("actor_group_type") or "individual_vehicle"
        placement_hint = group_info.get("placement_mode_hint") or "normal_lane_actor"
        motion = "unknown"
        heading = str(hint.get("heading_relation_to_ego") or "unknown")
        proximity = "immediate" if cy > 0.78 else "near" if cy > 0.58 else "mid" if cy > 0.36 else "far"
        layout_anchor_id = (
            SceneUnderstandingInterpreter._layout_anchor_for_detection(cx, cy, is_junction)
        )
        region_hint = str(hint.get("vehicle_region_hint") or "unknown")
        if is_junction and region_hint in {"right_arm", "left_arm", "ahead_arm", "ego_approach"}:
            layout_anchor_id = region_hint
        anchor_relation = {}
        travel = SceneUnderstandingInterpreter._infer_junction_travel_direction(
            hint,
            layout_anchor_id,
        )
        if is_junction:
            anchor_relation["travel_direction"] = travel
        subject = {
            "id": det_id,
            "category": label if label in {"car", "truck", "bus", "motorcycle", "bicycle"} else "car",
            "subtype": label,
            "appearance": {},
            "visual_confidence": str(hint.get("confidence") or "medium"),
            "motion_state": motion,
            "turn_intent": "none",
            "heading_relation_to_ego": heading,
            "flow_compliance": "legal",
            "lane_index_relation": lane_index,
            "lane_side_relation": lane_side,
            "longitudinal_relation": "ahead",
            "longitudinal_proximity": proximity,
            "count": 1,
            "evidence": str(hint.get("evidence") or f"detector bbox {det.get('bbox_norm')}"),
            "must_reconstruct": True,
            "layout_anchor_id": layout_anchor_id,
            "anchor_relation": anchor_relation,
            "actor_group_type": actor_group_type,
            "actor_group_id": group_info.get("actor_group_id") or det_id,
            "group_order_index": group_info.get("group_order_index"),
            "group_order_rule": group_info.get("group_order_rule") or "",
            "placement_mode_hint": placement_hint,
        }
        return subject

    @staticmethod
    def _infer_junction_travel_direction(hint: dict, layout_anchor_id: str) -> str:
        explicit = str(hint.get("junction_travel_direction") or "unknown").lower()
        if explicit in {"toward_junction", "away_from_junction"}:
            return explicit
        front_dir = str(hint.get("front_points_image_direction") or "unclear").lower()
        visible_end = str(hint.get("visible_end") or "unclear").lower()
        anchor = str(layout_anchor_id or "").lower()
        if anchor == "right_arm":
            if front_dir == "right":
                return "away_from_junction"
            if front_dir == "left":
                return "toward_junction"
        if anchor == "left_arm":
            if front_dir == "left":
                return "away_from_junction"
            if front_dir == "right":
                return "toward_junction"
        if anchor in {"ahead_arm", "oncoming_arm"}:
            if visible_end == "front" or front_dir == "toward_camera":
                return "toward_junction"
            if visible_end == "rear" or front_dir == "away_from_camera":
                return "away_from_junction"
        return "unknown"

    @staticmethod
    def _layout_anchor_for_detection(cx: float, cy: float, is_junction: bool) -> str:
        if not is_junction:
            return ""
        # Detector bbox position alone is not topology evidence. A high/low image
        # center may simply be a far/near vehicle on ego's approach; only explicit
        # VLM arm hints should move fallback actors to ahead/left/right arms.
        return "ego_approach"

    @staticmethod
    def _pairwise_from_rows(row_membership: dict) -> list:
        by_group = {}
        for det_id, info in row_membership.items():
            group_id = info.get("actor_group_id")
            if not group_id:
                continue
            by_group.setdefault(group_id, []).append((det_id, info))
        relations = []
        for group_id, members in by_group.items():
            members.sort(key=lambda item: int(item[1].get("group_order_index") or 0))
            for (front_id, _front), (rear_id, _rear) in zip(members, members[1:]):
                relations.append(
                    {
                        "entity_id": rear_id,
                        "other_entity_id": front_id,
                        "longitudinal_relation": "behind_other",
                        "longitudinal_gap_band": "near",
                        "lane_relation": "same_lane",
                        "constraint_strength": "soft",
                        "confidence": "medium",
                        "evidence": f"detector row group {group_id}",
                    }
                )
        return relations

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
        if isinstance(payload.get("road_scene_brief"), dict):
            normalized.setdefault("metadata", {})["road_scene_brief"] = payload[
                "road_scene_brief"
            ]

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


def _normalize_ignored_detection_candidates(value) -> list:
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


def _relation_refs_existing_subjects(relation: dict, subject_ids: set) -> bool:
    if not subject_ids:
        return False
    entity_id = str(relation.get("entity_id") or "").strip()
    other_id = str(relation.get("other_entity_id") or "").strip()
    return bool(entity_id and other_id and entity_id in subject_ids and other_id in subject_ids)
