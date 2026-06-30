import base64
import json
import os
import re
import sys

import cv2


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent
from tools.utils import read_file, write_to_file
from opendrive_experiment.tools.structured_pipeline import normalize_scene_understanding


class SceneUnderstandingInterpreter(TaskAgent):
    """Image interpreter that emits the structured Scene Understanding DSL."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
        You are an assistant for faithfully converting a single traffic image into a structured Scene Understanding DSL for OpenDRIVE reconstruction.

        Output only JSON. Do not output markdown fences unless absolutely necessary.
        Do not narrate. Do not add prose outside the JSON object.

        Top-level JSON schema:
        {
          "traffic_subjects": [],
          "background_traffic": [],
          "key_pairwise_relations": [],
          "road_network": {},
          "general_environment": {},
          "metadata": {}
        }

        Requirements:
        - `traffic_subjects` must include only the key visible actors or obstacle groups that directly constrain the ego vehicle.
        - `background_traffic` may include representative traffic inferred from clearly visible context such as a curbside parking row, sparse opposing flow, or sidewalk pedestrian group.
        - `road_network` must include only road geometry and control/layout evidence needed to build OpenDRIVE.
        - `general_environment` is contextual only; do not include spawnable traffic actors there.
        - Preserve left/right and ahead/behind relationships faithfully.
        - Use uncertainty conservatively: only when the image evidence is genuinely insufficient.

        `traffic_subjects` entity fields:
        - id
        - category
        - subtype
        - appearance
        - visual_confidence
        - motion_state
        - heading_relation_to_ego
        - lane_side_relation
        - longitudinal_relation
        - longitudinal_proximity
        - count
        - evidence
        - must_reconstruct

        `background_traffic` entity fields:
        - id
        - category
        - appearance
        - source
        - representative_count
        - lane_side_relation
        - longitudinal_band
        - motion_bias
        - heading_relation_to_ego
        - density_role
        - confidence
        - spawn_priority

        `road_network` fields:
        - road_type
        - directionality
        - road_segments
        - lane_groups
        - lane_markings
        - special_road_areas
        - junctions
        - roadside_boundaries
        - control_elements

        `general_environment` fields:
        - weather_hint
        - lighting_hint
        - time_of_day_hint
        - urban_density
        - roadside_context_left
        - roadside_context_right
        - occlusion_notes
        - non_spawnable_landmarks

        Constraints:
        - Use `must_reconstruct=true` for every traffic_subject.
        - Keep `background_traffic` to a small representative set.
        - `key_pairwise_relations` is optional and should include only the most useful, high-confidence pairwise relations between visible actors.
        - Do not invent exact metric distances.
        - Enumerate clearly visible vehicle-like actors before adding generic background activity. This includes cars, motorcycles, scooters, bicycles, and parked two-wheelers.
        - If a scooter or motorcycle is stopped or parked in a curbside parking lane/bay, represent it as `category="parked_vehicle"` with `subtype="motorcycle"` or `subtype="motor_scooter"`, not as pedestrian or sidewalk activity.
        - Do not add spawnable pedestrians unless human bodies are clearly visible and relevant to the ego path. Storefront customers, distant silhouettes, or ambiguous sidewalk activity should be contextual environment, not `background_traffic`.
        - For curbside parking rows, include the row as background traffic and preserve the visible side (`right_edge` or `left_edge`). If an individual nearby parked motorcycle/scooter is distinct from the row, include it separately.
        - Determine lane count from visible ground evidence first: lane lines, edge lines, parking-lane separators, curb-adjacent boundaries, and other painted ground markings.
        - Do not infer lane count only from how many rows of vehicles are present.
        - If a parking lane is visibly continuous and separated from the adjacent driving lane by a clear boundary, count it as its own lane in `road_network.lane_groups`.
        - When ground markings conflict with vehicle placement or roadside semantics, prefer the ground markings.
        - Example: one driving lane in each direction plus one parking lane on each outer side should be represented as 2 driving lanes + 2 parking lanes, not as 2 driving lanes plus a vague parking strip.
        - Prefer structured categorical relations such as `left_lane`, `right_edge`, `same_lane`, `sidewalk_left`, `crosswalk`, `ahead`, `near`, `far`.
        - For vehicle-like actors, include `appearance.color` when the vehicle body color is visible, using simple labels such as `white`, `black`, `gray`, `silver`, `blue`, `red`, `green`, `yellow`, `orange`, or `brown`.
        - Only include `appearance.color` when there is visual evidence; otherwise omit it instead of guessing.
        - For `key_pairwise_relations`, use fields: `entity_id`, `other_entity_id`, `longitudinal_relation`, optional `longitudinal_gap_band`, optional `lateral_relation`, optional `lane_relation`, optional `constraint_strength`, optional `confidence`, optional `evidence`.
        - Each `lane_groups[]` item may include `forward_lane_count` and `opposing_lane_count` for driving lanes, plus optional `left_parking_lane_count`, `right_parking_lane_count`, `lane_count_evidence`, and `lane_count_confidence` when parking lanes are visibly present.
        """

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

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
        return [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._image_to_base64(image_path)}"
                },
            },
        ]

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
            self.send_request(
                "",
                {
                    "output_fn": output_fn,
                    "merge_scene_understanding": True,
                    "merge_prompt": merge_prompt,
                    "request_label": "Scene understanding merge",
                    "request_timeout": 180,
                },
            )
            payload, validation_error = self.extract_decision_data(output_fn)
            attempts += 1
            if validation_error is None:
                payload.setdefault("metadata", {})["user_description_applied"] = True
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Scene understanding merge failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            print(f"Regenerating scene understanding merge... Attempt {attempts + 1}")

    @staticmethod
    def _build_merge_prompt(scene_understanding, user_description: str) -> str:
        return (
            "You are merging a VLM-generated traffic scene JSON with a user-provided "
            "description of the same image.\n\n"
            "Output only a complete JSON object using the Scene Understanding DSL schema. "
            "Do not output markdown fences or prose.\n\n"
            "Priority rules:\n"
            "1. The user description is authoritative when it conflicts with the VLM JSON.\n"
            "2. Preserve VLM road_network, general_environment, and visible actors that the "
            "user does not contradict.\n"
            "3. Add or correct vehicles, parked rows, motorcycles, scooters, pedestrians, "
            "cones, left/right relations, ahead/behind relations, lane-side relations, and "
            "pairwise relations explicitly mentioned by the user.\n"
            "4. Use conservative uncertainty where the user and VLM are both ambiguous.\n"
            "5. Do not store or repeat the raw user description in metadata.\n\n"
            "Required top-level keys: traffic_subjects, background_traffic, "
            "key_pairwise_relations, road_network, general_environment, metadata.\n\n"
            "User description:\n"
            f"{user_description.strip()}\n\n"
            "VLM scene understanding JSON:\n"
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
            return None, f"Scene understanding JSON is invalid: {exc.msg}."

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
