"""Structured video accident understanding (replaces the free-text branch).

Stage A of the dynamic-reconstruction pipeline. Given the anchor frames of an
accident video (start frame + optional context frames + end-anchor frame), a VLM
emits a *structured* JSON understanding instead of the free-text ``## Decision``
blob produced by ``agents/video_interpreter.py``.

Output (``{scene_id}_video_understanding.json``) is consumed by
``agents/llm_video_trajectory_generator.py`` to write the executable trajectory
DSL, and its ``end_state`` block is reused as the symbolic end-anchor reference
for the soft-constraint check (see ``tools/video_reconstruction_runner.py``).

Mirrors ``agents/llm_accident_predictor.py`` (VLM + image messages + JSON
extraction with a regenerate loop).
"""

import base64
import json
import mimetypes
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.task_agent import TaskAgent
from tools.utils import write_to_file


class VideoAccidentInterpreter(TaskAgent):
    """VLM that turns accident anchor frames into a structured understanding JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_prompt = """
        You are analysing an accident captured by a forward-facing vehicle camera.
        You are given sampled frames in chronological order from alert to event.
        Every image label includes available source time, time relative to START,
        sampling progress/phase, and source frame index. Use changes across the
        whole ordered sequence to infer each actor's motion; do not treat the
        images as unrelated views. The ego vehicle is the camera car.

        Produce a STRUCTURED understanding of how the accident unfolds, in terms of
        the visible road participants and a coarse timeline. Do NOT output CARLA or
        Scenic code. Output ONLY JSON (no markdown fences), using this schema:

        {
          "scene_id": "...",
          "accident_type": "free text, e.g. rear-end / cut-in / pedestrian crossing",
          "summary": "one or two sentences on the accident mechanism",
          "participants": [
            {
              "ref": "v1",
              "type": "car|truck|bus|motorcycle|bicycle|pedestrian|obstacle",
              "visual_description": "short visual cue (colour, position)",
              "start_position_relative_to_ego": "e.g. ahead same lane ~12 m / left lane",
              "role": "lead_vehicle|cut_in_vehicle|crossing|oncoming|...",
              "mapping_hint": "short visual/geometric reason for the match",
              "matched_actor_id": "real id from the supplied actor table, or null",
              "match_confidence": 0.0
            }
          ],
          "actor_motion_states": [
            {
              "actor_id": "every non-ego vehicle id from the actor table",
              "motion_state": "moving|stationary|uncertain",
              "evidence": "short cross-frame visual reason"
            }
          ],
          "event_sequence": [
            {
              "t_rel": "start|mid|end (coarse ordering)",
              "actor_ref": "v1 or ego",
              "action": "decelerate|brake|stop|cut_in|cross|lane_change|accelerate|maintain",
              "description": "what happens"
            }
          ],
          "end_state": {
            "collision": true,
            "collision_pair": ["ego", "v1"],
            "relative_order": "v1 ahead of ego in same lane",
            "final_motion": "both stopped"
          },
          "uncertainty": "what is hard to tell from the frames"
        }

        Rules:
        - Use ego-relative descriptions; longitudinal ahead is positive.
        - Keep the timeline coarse (start/mid/end), do not invent exact metres or
          seconds you cannot see.
        - For lateral maneuvers, still describe them, but note that placement may be
          approximated by the downstream reconstruction.
        - participants[].ref ids must be referenced consistently in event_sequence
          and end_state.
        - A compact table of the real CARLA actors is supplied below. Match every
          visible participant to an actor id using type, lane relation, heading,
          and ego-relative geometry. Never invent an actor id. Use null when no
          table row is defensible, and explain the ambiguity in uncertainty.
        - match_confidence must be between 0 and 1.
        - Inspect EVERY non-ego vehicle from the actor table across the ordered
          frames and include it exactly once in actor_motion_states, even when it
          is not an accident participant. This is a simple moving/stationary
          classification, not a detailed trajectory. A vehicle whose image
          position remains consistent with the static scene is stationary.
          Use the actor-table motion_state as supporting prior evidence. Choose
          uncertain only when occlusion or inconsistent matching truly prevents
          a decision; explain why in evidence.
        """

    def refine_request(self, user_request, add_info=None):
        assert add_info is not None, "Missing video interpreter add_info."
        scene_id = str(add_info.get("scene_id") or "")
        prompt = self.pre_prompt
        if user_request:
            prompt += f"\nUser request:\n{user_request}"
        prompt += f"\n\nsource_scene_id:\n{scene_id}\n"
        actor_context = add_info.get("actor_context") or {}
        prompt += (
            "\nCompact CARLA actor table at the START frame. Coordinates x/y/yaw "
            "are CARLA world values; relative_to_ego is the preferred evidence "
            "for matching visible participants:\n"
            f"{json.dumps(actor_context, indent=2, sort_keys=True)}\n"
        )

        messages: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for label, image_path in self._ordered_frames(add_info):
            if not image_path:
                continue
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Frame image not found: {image_path}")
            messages.append({"type": "text", "text": f"Frame: {label}"})
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{self._mime_type(image_path)};base64,"
                            f"{self._encode_image(image_path)}"
                        )
                    },
                }
            )
        return messages

    @staticmethod
    def _ordered_frames(add_info: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
        sequence = add_info.get("frame_sequence") or []
        if sequence:
            ordered: List[Tuple[str, Optional[str]]] = []
            count = len(sequence)
            for index, frame in enumerate(sequence):
                role = (
                    "START"
                    if index == 0
                    else "END"
                    if index == count - 1
                    else "CONTEXT"
                )
                details = [f"FRAME {index + 1}/{count}", role]
                if frame.get("timestamp_s") is not None:
                    details.append(f"source_t={float(frame['timestamp_s']):.3f}s")
                if frame.get("relative_timestamp_s") is not None:
                    details.append(f"t_rel={float(frame['relative_timestamp_s']):.3f}s")
                if frame.get("window_progress") is not None:
                    details.append(f"progress={float(frame['window_progress']):.4f}")
                if frame.get("phase"):
                    details.append(f"phase={frame['phase']}")
                if frame.get("frame_index") is not None:
                    details.append(f"source_frame={frame['frame_index']}")
                ordered.append((" | ".join(details), frame.get("path")))
            return ordered

        frames: List[Tuple[str, Optional[str]]] = [
            ("START (before accident)", add_info.get("start_frame_path"))
        ]
        for index, path in enumerate(add_info.get("context_frame_paths") or []):
            frames.append((f"CONTEXT {index}", path))
        frames.append(("END (accident outcome)", add_info.get("end_anchor_frame_path")))
        return frames

    def call_agent(self, user_request, added_info) -> Dict[str, Any]:
        output_fn = added_info["output_fn"]
        attempts = 0
        while True:
            self.send_request(
                user_request,
                {
                    **added_info,
                    "request_label": "Video accident understanding",
                    "request_timeout": 180,
                    "request_max_tokens": 4000,
                },
            )
            payload, error = self.extract_decision_data(output_fn)
            if error is None:
                error = self._validate_actor_matches(
                    payload,
                    added_info.get("actor_context") or {},
                )
            attempts += 1
            if error is None:
                write_to_file(output_fn, json.dumps(payload, indent=2, sort_keys=True))
                return payload
            if attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Video understanding failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {error}"
                )
            print(f"Regenerating video understanding... Attempt {attempts + 1}")

    @staticmethod
    def _validate_actor_matches(
        payload: Dict[str, Any], actor_context: Dict[str, Any]
    ) -> Optional[str]:
        actor_ids = {
            str(row.get("id"))
            for row in actor_context.get("actors", [])
            if isinstance(row, dict) and row.get("id") is not None
        }
        for index, participant in enumerate(payload.get("participants") or []):
            matched_id = participant.get("matched_actor_id")
            if matched_id is not None and str(matched_id) not in actor_ids:
                return (
                    f"`participants[{index}].matched_actor_id` references unknown "
                    f"actor id `{matched_id}`; allowed ids: {sorted(actor_ids)}."
                )
        motion_rows = payload.get("actor_motion_states")
        if not isinstance(motion_rows, list):
            return "`actor_motion_states` must be a list covering every non-ego vehicle."
        required_vehicle_ids = {
            str(row.get("id"))
            for row in actor_context.get("actors", [])
            if isinstance(row, dict)
            and row.get("id") is not None
            and str(row.get("id")) != "ego"
            and VideoAccidentInterpreter._is_vehicle_actor_row(row)
        }
        seen_motion_ids = set()
        for index, motion_row in enumerate(motion_rows):
            if not isinstance(motion_row, dict):
                return f"`actor_motion_states[{index}]` must be an object."
            actor_id = str(motion_row.get("actor_id") or "")
            if actor_id not in required_vehicle_ids:
                return (
                    f"`actor_motion_states[{index}].actor_id` references unknown "
                    f"non-ego vehicle `{actor_id}`; allowed ids: "
                    f"{sorted(required_vehicle_ids)}."
                )
            if actor_id in seen_motion_ids:
                return f"Duplicate actor_motion_states entry for `{actor_id}`."
            seen_motion_ids.add(actor_id)
            if motion_row.get("motion_state") not in {
                "moving",
                "stationary",
                "uncertain",
            }:
                return (
                    f"`actor_motion_states[{index}].motion_state` must be "
                    "moving, stationary, or uncertain."
                )
        missing_motion_ids = sorted(required_vehicle_ids - seen_motion_ids)
        if missing_motion_ids:
            return (
                "Missing actor_motion_states entries for non-ego vehicles: "
                + ", ".join(missing_motion_ids)
            )
        return None

    @staticmethod
    def _is_vehicle_actor_row(row: Dict[str, Any]) -> bool:
        spawn_kind = str(row.get("spawn_kind") or "").lower()
        category = str(row.get("category") or "").lower()
        if spawn_kind:
            return spawn_kind == "vehicle"
        if category:
            return category in {
                "car",
                "truck",
                "bus",
                "van",
                "motorcycle",
                "bicycle",
                "vehicle",
            }
        # Compact legacy actor tables contained ids/geometry only and represented
        # vehicles; preserve that interpretation.
        return True

    def extract_decision_data(
        self, file_path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        with open(file_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        payload, parse_error = self._parse_json_object(text)
        if parse_error:
            return None, parse_error
        return self._validate(payload)

    @staticmethod
    def _validate(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(payload.get("accident_type"), str) or not payload["accident_type"]:
            return None, "`accident_type` must be a non-empty string."
        participants = payload.get("participants")
        if not isinstance(participants, list) or not participants:
            return None, "`participants` must be a non-empty list."
        for index, participant in enumerate(participants):
            if not isinstance(participant, dict):
                return None, f"`participants[{index}]` must be an object."
            confidence = participant.get("match_confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0
            ):
                return None, (
                    f"`participants[{index}].match_confidence` must be between 0 and 1."
                )
        events = payload.get("event_sequence")
        if not isinstance(events, list) or not events:
            return None, "`event_sequence` must be a non-empty list."
        end_state = payload.get("end_state")
        if not isinstance(end_state, dict):
            return None, "`end_state` must be an object."
        return payload, None

    @staticmethod
    def _parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(text, str) or not text.strip():
            return None, "Video understanding output is empty."
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, f"Video understanding output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Video understanding output must be a JSON object."
        return payload, None

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    def _mime_type(image_path: str) -> str:
        mime_type, _encoding = mimetypes.guess_type(image_path)
        return mime_type or "image/jpeg"
