"""Orchestrator for the pure-LLM (no template library) risk experiment.

Mirrors ``tools/pure_llm_scenic_runner.py`` but emits executable CARLA Python
(via an intermediate DSL) instead of Scenic:

    image --> LlmAccidentPredictor   --> {scene_id}_candidates.json
           --> LlmRiskDslGenerator   --> {scene_id}_rNNN_dsl.json   (Stage 2a)
           --> build_dsl_risk_scene_script --> {scene_id}_rNNN.py    (Stage 2b, deterministic)
           --> check_python_compile  --> {scene_id}_rNNN_compile.json
           --> {scene_id}_risk_summary.json

It reuses the already-reconstructed ``{scene_id}_actors.json`` (spawn_payload,
real CARLA world coordinates) and ``{scene_id}_match.json`` (map / world_name).
The automatic repair loop targets Stage 2a (DSL schema validation via
``validate_risk_dsl``); the generated Python is deterministic, so
``check_python_compile`` is a final syntax sanity gate, not part of the loop.
"""

import json
import os
from os.path import join
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.existing_world_scenario_generator import ExistingWorldScenarioGenerator
from agents.llm_accident_predictor import LlmAccidentPredictor
from agents.llm_risk_dsl_generator import LlmRiskDslGenerator
from tools.python_compile import check_python_compile
from tools.risk_dsl import validate_risk_dsl
from tools.risk_scenario_pipeline import build_actor_context
from tools.utils import read_file, write_to_file


class PureLlmRiskRunner:
    """Two-stage pure-LLM risk generator (accident -> DSL -> CARLA Python)."""

    def __init__(
        self,
        output_folder: str,
        scene_id: str,
        risk_output_folder: Optional[str] = None,
        ego_speed_mps: float = 10.0,
        num_accidents: int = 1,
        max_retries: int = 2,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
        enable_compile: bool = True,
        predictor: Optional[LlmAccidentPredictor] = None,
        generator: Optional[LlmRiskDslGenerator] = None,
        codegen: Optional[ExistingWorldScenarioGenerator] = None,
        compile_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.output_folder = output_folder
        self.risk_output_folder = risk_output_folder or output_folder
        self.scene_id = scene_id
        self.ego_speed_mps = float(ego_speed_mps)
        self.num_accidents = max(1, int(num_accidents))
        self.max_retries = max(0, int(max_retries))
        self.carla_host = carla_host
        self.carla_port = int(carla_port)
        self.carla_map = carla_map
        self.enable_compile = enable_compile
        os.makedirs(self.risk_output_folder, exist_ok=True)
        self.predictor = predictor or LlmAccidentPredictor(self.ego_speed_mps)
        self.generator = generator or LlmRiskDslGenerator(self.ego_speed_mps)
        self.codegen = codegen or ExistingWorldScenarioGenerator()
        self.compile_fn = compile_fn or check_python_compile

    def run(
        self,
        candidates_path: Optional[str] = None,
        user_request: str = "",
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        spawn_payload = self._load_json(self._spawn_payload_path())
        match_report = self._load_json(self._scene_match_path())
        actor_context = self._build_actor_context(spawn_payload)
        write_to_file(
            self._actor_context_path(),
            json.dumps(actor_context, indent=2, sort_keys=True),
        )

        if candidates_path:
            candidates = self._load_json(candidates_path)
        else:
            if not image_path:
                raise ValueError("image_path is required when candidates_path is not provided.")
            candidates = self.predictor.call_agent(
                user_request,
                {
                    "output_fn": self._candidates_path(),
                    "scene_id": self.scene_id,
                    "image_path": image_path,
                },
            )

        map_name = self._normalize_world_name(
            self.carla_map or match_report.get("world_name")
        )

        accident_candidates = (candidates.get("accident_candidates") or [])[: self.num_accidents]
        artifacts: List[Dict[str, Any]] = []
        for index, candidate in enumerate(accident_candidates):
            normalized_dsl, attempts, schema_error = self._generate_with_repair(
                candidate=candidate,
                actor_context=actor_context,
                spawn_payload=spawn_payload,
            )

            artifact: Dict[str, Any] = {
                "candidate_index": index,
                "candidate_id": candidate.get("id"),
                "accident_type": candidate.get("accident_type"),
                "attempts": attempts,
                "dsl_valid": normalized_dsl is not None,
                "schema_error": schema_error,
                "dsl_path": None,
                "script_path": None,
                "metrics_path": None,
                "compile_report_path": None,
                "compiled": None,
            }

            if normalized_dsl is not None:
                dsl_path = self._dsl_path(index)
                write_to_file(dsl_path, json.dumps(normalized_dsl, indent=2, sort_keys=True))
                artifact["dsl_path"] = dsl_path

                script_path = self._script_path(index)
                metrics_path = self._metrics_path(index)
                script = self.codegen.build_dsl_risk_scene_script(
                    spawn_payload_filename=os.path.abspath(self._spawn_payload_path()),
                    dsl_filename=os.path.abspath(dsl_path),
                    risk_metrics_filename=os.path.abspath(metrics_path),
                    carla_host=self.carla_host,
                    carla_port=self.carla_port,
                    carla_map=map_name,
                    scene_match_status=match_report.get("status"),
                    scene_match_reason=match_report.get("reason"),
                )
                write_to_file(script_path, script)
                artifact["script_path"] = script_path
                artifact["metrics_path"] = metrics_path

                if self.enable_compile:
                    compile_result = self.compile_fn(script_path)
                    compile_path = self._compile_report_path(index)
                    write_to_file(
                        compile_path,
                        json.dumps(compile_result, indent=2, sort_keys=True),
                    )
                    artifact["compile_report_path"] = compile_path
                    artifact["compiled"] = compile_result.get("compiled")

            artifacts.append(artifact)

        summary = {
            "enabled": True,
            "experiment": "pure_llm_risk",
            "scene_id": self.scene_id,
            "static_output_folder": self.output_folder,
            "risk_output_folder": self.risk_output_folder,
            "ego_speed_mps": self.ego_speed_mps,
            "num_accidents": self.num_accidents,
            "map_name": map_name,
            "compile_enabled": self.enable_compile,
            "candidates_path": self._candidates_path(),
            "actor_context_path": self._actor_context_path(),
            "num_candidates": len(accident_candidates),
            "num_dsl_valid": sum(1 for a in artifacts if a.get("dsl_valid")),
            "num_compiled": sum(1 for a in artifacts if a.get("compiled")),
            "artifacts": artifacts,
        }
        write_to_file(
            self._summary_path(),
            json.dumps(summary, indent=2, sort_keys=True),
        )
        return summary

    def _generate_with_repair(
        self,
        candidate: Dict[str, Any],
        actor_context: Dict[str, Any],
        spawn_payload: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
        prior_attempt: Optional[str] = None
        schema_error: Optional[str] = None
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            dsl_text = self.generator.generate(
                scene_id=self.scene_id,
                candidate=candidate,
                actor_context=actor_context,
                prior_attempt=prior_attempt,
                schema_error=schema_error,
            )
            parsed, parse_error = self._parse_json_object(dsl_text)
            if parse_error is None:
                normalized, validate_error = validate_risk_dsl(parsed, spawn_payload)
                if validate_error is None:
                    normalized = self._convert_distance_triggers_to_time(
                        normalized,
                        actor_context,
                    )
                    return normalized, attempts, None
                schema_error = validate_error
            else:
                schema_error = parse_error
            prior_attempt = dsl_text
        return None, attempts, schema_error

    def _convert_distance_triggers_to_time(
        self,
        dsl: Dict[str, Any],
        actor_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert distance triggers to time triggers to avoid t=0 braking.

        Reconstructed traffic can already be inside a distance threshold when
        the risk script starts. In that case distance_to_ego_below fires on the
        first tick and the risk actor appears stationary. Time triggers let the
        actor receive its flying-start speed first, then perform the risky
        braking / cut-in / sideswipe action.
        """
        normalized = json.loads(json.dumps(dsl))
        actor_rows = {
            str(row.get("id")): row
            for row in (actor_context or {}).get("actors", [])
            if isinstance(row, dict)
        }
        converted = []
        for event in normalized.get("events") or []:
            trigger = event.get("trigger") or {}
            if trigger.get("type") != "distance_to_ego_below":
                continue
            value_s = self._time_trigger_for_event(event, actor_rows)
            event["trigger"] = {
                "type": "time_elapsed_above",
                "value_s": value_s,
            }
            converted.append({
                "actor_id": event.get("actor_id"),
                "from": {
                    "type": "distance_to_ego_below",
                    "value_m": trigger.get("value_m"),
                },
                "to": event["trigger"],
                "action_type": (event.get("action") or {}).get("type"),
            })
        if converted:
            normalized.setdefault("metadata", {})
            normalized["metadata"]["distance_triggers_converted_to_time"] = converted
        return normalized

    def _time_trigger_for_event(
        self,
        event: Dict[str, Any],
        actor_rows: Dict[str, Dict[str, Any]],
    ) -> float:
        actor_id = str(event.get("actor_id") or "")
        actor = actor_rows.get(actor_id) or {}
        relative = actor.get("relative_to_ego") or {}
        try:
            longitudinal = float(relative.get("longitudinal_m", 12.0))
        except (TypeError, ValueError):
            longitudinal = 12.0
        action_type = str((event.get("action") or {}).get("type") or "")
        ego_speed = max(0.1, float(self.ego_speed_mps))

        # Same-lane lead braking should happen shortly after the flying start,
        # but not at t=0. Farther lead vehicles can wait longer.
        if action_type in {"brake", "stop", "set_speed"}:
            target_gap_m = 6.0
            delay_s = (max(0.0, longitudinal) - target_gap_m) / ego_speed
            return round(min(3.0, max(0.8, delay_s)), 2)

        # Cut-in / sideswipe actors need enough time to be visibly moving before
        # steering into the ego lane.
        if action_type in {"steer", "cross"}:
            target_gap_m = 8.0
            delay_s = (max(0.0, longitudinal) - target_gap_m) / ego_speed
            return round(min(2.5, max(0.8, delay_s)), 2)

        return 1.0

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
            return None, "Risk DSL output is empty."
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"Risk DSL output is not valid JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "Risk DSL output must be a JSON object."
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

    def _candidates_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_candidates.json")

    def _actor_context_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_actors.json")

    def _dsl_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_r{index:03d}_dsl.json")

    def _script_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_r{index:03d}.py")

    def _metrics_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_r{index:03d}_metrics.json")

    def _compile_report_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_r{index:03d}_compile.json")

    def _summary_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_summary.json")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return json.loads(read_file(path))

    def _first_existing_path(self, *relative_paths: str) -> str:
        for relative_path in relative_paths:
            path = join(self.output_folder, relative_path)
            if os.path.exists(path):
                return path
        return join(self.output_folder, relative_paths[0])
