"""Orchestrator for the accident video -> CARLA dynamic-reconstruction pipeline.

    adaptive frames_manifest.json (preferred), or video + (start_s, end_s)
      --> load sampled frames, or extract_anchor_frames for legacy callers
      --> [reuse Layer-1 static recon]  --> {scene_id}_actors.json + _match.json
      --> VideoAccidentInterpreter      --> {scene_id}_video_understanding.json   (Stage A)
      --> LlmVideoTrajectoryGenerator   --> {scene_id}_video_trajectory_dsl.json  (Stage B)
            (validate video-trajectory-dsl-v1 + repair loop)
      --> lower_to_risk_dsl             --> {scene_id}_video_risk_dsl.json
      --> build_dsl_risk_scene_script   --> {scene_id}_dynamic_reconstruction.py
      --> check_python_compile          --> compile report
      --> evaluate_end_anchor           --> symbolic soft-constraint check
      --> {scene_id}_video_reconstruction_summary.json

Consistent with the rest of Layer-2 (see PureLlmRiskRunner): it REUSES the
already-reconstructed ``{scene_id}_actors.json`` / ``{scene_id}_match.json`` and
never re-derives geometry. The start frame is the spawn anchor; run Layer-1
(``experiments/auto_generate_all_vlm.py``) on the saved start frame first if
those files do not exist yet.
"""

import json
import os
from os.path import join
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.existing_world_scenario_generator import ExistingWorldScenarioGenerator
from agents.llm_video_trajectory_generator import LlmVideoTrajectoryGenerator
from agents.video_accident_interpreter import VideoAccidentInterpreter
from tools.python_compile import check_python_compile
from tools.risk_scenario_pipeline import build_actor_context
from tools.utils import read_file, write_to_file
from tools.video_frames import extract_anchor_frames
from tools.video_trajectory_dsl import (
    lower_to_risk_dsl,
    validate_video_trajectory_dsl,
)


class VideoReconstructionRunner:
    """Accident video -> structured understanding -> trajectory DSL -> CARLA Python."""

    def __init__(
        self,
        output_folder: str,
        scene_id: str,
        video_path: Optional[str] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        frames_manifest_path: Optional[str] = None,
        risk_output_folder: Optional[str] = None,
        ego_speed_mps: float = 10.0,
        context_sample_rate_s: Optional[float] = None,
        max_retries: int = 2,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        enable_compile: bool = True,
        interpreter: Optional[VideoAccidentInterpreter] = None,
        generator: Optional[LlmVideoTrajectoryGenerator] = None,
        codegen: Optional[ExistingWorldScenarioGenerator] = None,
        compile_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        frame_extractor: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.output_folder = output_folder
        self.risk_output_folder = risk_output_folder or output_folder
        self.scene_id = scene_id
        self.video_path = video_path
        self.start_s = float(start_s) if start_s is not None else None
        self.end_s = float(end_s) if end_s is not None else None
        self.frames_manifest_path = frames_manifest_path
        self.ego_speed_mps = float(ego_speed_mps)
        self.context_sample_rate_s = context_sample_rate_s
        self.max_retries = max(0, int(max_retries))
        self.carla_host = carla_host
        self.carla_port = int(carla_port)
        self.enable_compile = enable_compile
        os.makedirs(self.risk_output_folder, exist_ok=True)
        self.interpreter = interpreter or VideoAccidentInterpreter()
        self.generator = generator or LlmVideoTrajectoryGenerator(self.ego_speed_mps)
        self.codegen = codegen or ExistingWorldScenarioGenerator()
        self.compile_fn = compile_fn or check_python_compile
        self.frame_extractor = frame_extractor or extract_anchor_frames

    def run(self, user_request: str = "") -> Dict[str, Any]:
        frames_manifest, effective_manifest_path = self._prepare_frames()
        ordered_frames = frames_manifest["frames"]

        spawn_payload = self._load_json(self._spawn_payload_path())
        match_report = self._load_json(self._scene_match_path())
        actor_context = self._build_actor_context(spawn_payload)
        write_to_file(
            self._actor_context_path(),
            json.dumps(actor_context, indent=2, sort_keys=True),
        )

        understanding = self.interpreter.call_agent(
            user_request,
            {
                "output_fn": self._understanding_path(),
                "scene_id": self.scene_id,
                "actor_context": actor_context,
                "start_frame_path": frames_manifest.get("start_frame_path"),
                "end_anchor_frame_path": frames_manifest.get("end_anchor_frame_path"),
                "frame_sequence": ordered_frames,
                "context_frame_paths": [
                    frame["path"]
                    for frame in frames_manifest.get("frames", [])
                    if not frame.get("is_start") and not frame.get("is_end_anchor")
                ],
            },
        )

        traj_dsl, attempts, schema_error = self._generate_with_repair(
            understanding=understanding,
            actor_context=actor_context,
            spawn_payload=spawn_payload,
            user_request=user_request,
        )

        # The map is fixed by Layer-1: actors.json coordinates are projected onto
        # the SceneMapMatcher-chosen world. It is intentionally NOT overridable
        # here -- loading a different map would break the actor geometry.
        map_name = self._normalize_world_name(match_report.get("world_name"))

        artifact: Dict[str, Any] = {
            "attempts": attempts,
            "dsl_valid": traj_dsl is not None,
            "schema_error": schema_error,
            "trajectory_dsl_path": None,
            "risk_dsl_path": None,
            "script_path": None,
            "metrics_path": None,
            "compile_report_path": None,
            "compiled": None,
            "end_anchor_check": None,
        }

        if traj_dsl is not None:
            traj_path = self._trajectory_dsl_path()
            write_to_file(traj_path, json.dumps(traj_dsl, indent=2, sort_keys=True))
            artifact["trajectory_dsl_path"] = traj_path

            risk_dsl = lower_to_risk_dsl(traj_dsl, ego_speed_mps=self.ego_speed_mps)
            risk_dsl_path = self._risk_dsl_path()
            write_to_file(risk_dsl_path, json.dumps(risk_dsl, indent=2, sort_keys=True))
            artifact["risk_dsl_path"] = risk_dsl_path

            metrics_path = self._metrics_path()
            script = self.codegen.build_dsl_risk_scene_script(
                spawn_payload_filename=os.path.abspath(self._spawn_payload_path()),
                dsl_filename=os.path.abspath(risk_dsl_path),
                risk_metrics_filename=os.path.abspath(metrics_path),
                carla_host=self.carla_host,
                carla_port=self.carla_port,
                carla_map=map_name,
                scene_match_status=match_report.get("status"),
                scene_match_reason=match_report.get("reason"),
            )
            script_path = self._script_path()
            write_to_file(script_path, script)
            artifact["script_path"] = script_path
            artifact["metrics_path"] = metrics_path

            if self.enable_compile:
                compile_result = self.compile_fn(script_path)
                compile_path = self._compile_report_path()
                write_to_file(
                    compile_path, json.dumps(compile_result, indent=2, sort_keys=True)
                )
                artifact["compile_report_path"] = compile_path
                artifact["compiled"] = compile_result.get("compiled")

            artifact["end_anchor_check"] = evaluate_end_anchor(
                understanding.get("end_state") or {},
                self._load_json_optional(metrics_path),
            )

        summary = {
            "enabled": True,
            "experiment": "video_dynamic_reconstruction",
            "scene_id": self.scene_id,
            "static_output_folder": self.output_folder,
            "risk_output_folder": self.risk_output_folder,
            "video_path": os.path.abspath(self.video_path) if self.video_path else None,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "ego_speed_mps": self.ego_speed_mps,
            "map_name": map_name,
            "compile_enabled": self.enable_compile,
            "frames_manifest_path": effective_manifest_path,
            "frame_source": (
                "adaptive_manifest" if self.frames_manifest_path else "video_extraction"
            ),
            "frame_count": len(ordered_frames),
            "user_request": user_request,
            "understanding_path": self._understanding_path(),
            "actor_context_path": self._actor_context_path(),
            "artifact": artifact,
        }
        write_to_file(
            self._summary_path(), json.dumps(summary, indent=2, sort_keys=True)
        )
        return summary

    def _prepare_frames(self) -> Tuple[Dict[str, Any], str]:
        """Load pre-sampled frames, falling back to legacy video extraction."""
        if self.frames_manifest_path:
            manifest_path = os.path.abspath(self.frames_manifest_path)
            if os.path.isdir(manifest_path):
                manifest_path = join(manifest_path, "frames_manifest.json")
            manifest = self._load_json(manifest_path)
            frames = manifest.get("frames")
            if not isinstance(frames, list) or len(frames) < 2:
                raise ValueError(
                    "Adaptive frames manifest must contain at least two frames."
                )
            frames = sorted(
                frames,
                key=lambda row: (
                    row.get("sample_index", float("inf")),
                    row.get("timestamp_s", float("inf")),
                    row.get("frame_index", float("inf")),
                ),
            )
            normalized_frames: List[Dict[str, Any]] = []
            manifest_dir = os.path.dirname(manifest_path)
            first_timestamp = float(frames[0].get("timestamp_s", 0.0))
            for index, source in enumerate(frames):
                if not isinstance(source, dict) or not source.get("path"):
                    raise ValueError(f"Manifest frame {index} has no image path.")
                row = dict(source)
                path = str(row["path"])
                if not os.path.isabs(path):
                    path = os.path.abspath(join(manifest_dir, path))
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Sampled frame image not found: {path}")
                row["path"] = path
                row["is_start"] = index == 0
                row["is_end_anchor"] = index == len(frames) - 1
                if row.get("timestamp_s") is not None:
                    row["relative_timestamp_s"] = round(
                        float(row["timestamp_s"]) - first_timestamp, 3
                    )
                normalized_frames.append(row)

            self.video_path = self.video_path or manifest.get("video_path")
            self.start_s = float(
                manifest.get("time_of_alert_s", frames[0].get("timestamp_s", 0.0))
            )
            self.end_s = float(
                manifest.get("time_of_event_s", frames[-1].get("timestamp_s", 0.0))
            )
            manifest = dict(manifest)
            manifest["frames"] = normalized_frames
            manifest["start_frame_path"] = normalized_frames[0]["path"]
            manifest["end_anchor_frame_path"] = normalized_frames[-1]["path"]
            return manifest, manifest_path

        if not self.video_path or self.start_s is None or self.end_s is None:
            raise ValueError(
                "Provide frames_manifest_path, or provide video_path + start_s + end_s."
            )
        frames_dir = join(self.risk_output_folder, "frames")
        manifest = self.frame_extractor(
            self.video_path,
            self.start_s,
            self.end_s,
            frames_dir,
            self.scene_id,
            context_sample_rate_s=self.context_sample_rate_s,
        )
        return manifest, join(frames_dir, "frames.json")

    def _generate_with_repair(
        self,
        understanding: Dict[str, Any],
        actor_context: Dict[str, Any],
        spawn_payload: Dict[str, Any],
        user_request: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
        prior_attempt: Optional[str] = None
        schema_error: Optional[str] = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            dsl_text = self.generator.generate(
                scene_id=self.scene_id,
                understanding=understanding,
                actor_context=actor_context,
                user_request=user_request,
                prior_attempt=prior_attempt,
                schema_error=schema_error,
            )
            parsed, parse_error = self._parse_json_object(dsl_text)
            if parse_error is None:
                normalized, validate_error = validate_video_trajectory_dsl(
                    parsed,
                    spawn_payload,
                    require_all_vehicle_actors=True,
                )
                if validate_error is None:
                    return normalized, attempts, None
                schema_error = validate_error
            else:
                schema_error = parse_error
            prior_attempt = dsl_text
        return None, attempts, schema_error

    def _build_actor_context(self, spawn_payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_context = build_actor_context(spawn_payload)
        metadata = {
            str(entity.get("id")): entity
            for entity in (spawn_payload.get("entities") or [])
            if isinstance(entity, dict) and entity.get("id") is not None
        }
        for row in actor_context.get("actors", []):
            entity = metadata.get(str(row.get("id"))) or {}
            row["blueprint_name"] = entity.get("blueprint_name")
            row["heading_relation"] = entity.get("heading_relation")
            row["motion_state"] = entity.get("motion_state")
        return actor_context

    @staticmethod
    def _parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not isinstance(text, str) or not text.strip():
            return None, "Trajectory DSL output is empty."
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"Trajectory DSL output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Trajectory DSL output must be a JSON object."
        return payload, None

    @staticmethod
    def _normalize_world_name(world_name: Optional[str]) -> Optional[str]:
        if not world_name:
            return None
        return str(world_name).split("/")[-1]

    def _spawn_payload_path(self) -> str:
        return self._first_existing_path(
            f"{self.scene_id}_actors.json",
            f"{self.scene_id}_spawn_entities.json",
        )

    def _scene_match_path(self) -> str:
        return self._first_existing_path(
            f"{self.scene_id}_match.json",
            f"{self.scene_id}_scene_match.json",
        )

    def _understanding_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_understanding.json")

    def _actor_context_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_actors.json")

    def _trajectory_dsl_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_trajectory_dsl.json")

    def _risk_dsl_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_risk_dsl.json")

    def _script_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_dynamic_reconstruction.py")

    def _metrics_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_metrics.json")

    def _compile_report_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_video_compile.json")

    def _summary_path(self) -> str:
        return join(
            self.risk_output_folder, f"{self.scene_id}_video_reconstruction_summary.json"
        )

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Required JSON file not found: {path}. Run Layer-1 static "
                "reconstruction on the start frame first."
            )
        return json.loads(read_file(path))

    @staticmethod
    def _load_json_optional(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        try:
            return json.loads(read_file(path))
        except (ValueError, OSError):
            return None

    def _first_existing_path(self, *relative_paths: str) -> str:
        for relative_path in relative_paths:
            path = join(self.output_folder, relative_path)
            if os.path.exists(path):
                return path
        return join(self.output_folder, relative_paths[0])


def evaluate_end_anchor(
    expected_end_state: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Symbolic soft-constraint check of the simulated end against the video end frame.

    Per the plan review (P1#4): compare *discrete symbols* only -- did a collision
    happen, and (when known) between whom -- never pixels. Returns ``status``:
    ``pending`` (no CARLA metrics yet), ``pass`` or ``mismatch`` with a
    ``repair_hint`` the trajectory generator can act on.
    """
    expected_collision = bool(expected_end_state.get("collision"))
    if metrics is None:
        return {
            "status": "pending",
            "reason": "No CARLA metrics yet; run the generated script to evaluate.",
            "expected_collision": expected_collision,
        }

    collision_events = metrics.get("collision_events") or metrics.get("collisions") or []
    observed_collision = len(collision_events) > 0
    result: Dict[str, Any] = {
        "status": "pass" if observed_collision == expected_collision else "mismatch",
        "expected_collision": expected_collision,
        "observed_collision": observed_collision,
        "min_distance_m": metrics.get("min_distance_m"),
        "min_ttc_s": metrics.get("min_ttc_s"),
    }
    if result["status"] == "mismatch":
        if expected_collision and not observed_collision:
            result["repair_hint"] = (
                "Video shows a collision but the simulation avoided it. Make the "
                "lead/risk actor brake harder or stop earlier, or reduce its cruise "
                "speed so the forward-driving ego closes the gap."
            )
        else:
            result["repair_hint"] = (
                "Simulation collided but the video shows no contact. Delay or soften "
                "the risk actor's braking, or raise its cruise speed."
            )
    return result
