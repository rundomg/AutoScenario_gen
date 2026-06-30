import json
import os
import re
from os.path import join
from typing import Any, Dict, Optional

from agents.existing_world_scenario_generator import ExistingWorldScenarioGenerator
from agents.risk_scenario_interpreter import RiskScenarioInterpreter
from tools.risk_scenario_pipeline import (
    build_actor_context,
    expand_candidates_by_speed,
    require_risk_evidence,
    sample_all_risk_scenarios,
    validate_risk_scenario_spec,
)
from tools.utils import read_file, write_to_file


class RiskScenarioRunner:
    """Standalone risk scenario generator for completed static reconstruction output."""

    def __init__(
        self,
        output_folder: str,
        scene_id: str,
        risk_output_folder: Optional[str] = None,
        samples_per_candidate: int = 3,
        seed: int = 0,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
    ) -> None:
        self.output_folder = output_folder
        self.risk_output_folder = risk_output_folder or output_folder
        self.scene_id = scene_id
        self.samples_per_candidate = max(1, int(samples_per_candidate))
        self.seed = int(seed)
        self.carla_host = carla_host
        self.carla_port = int(carla_port)
        self.carla_map = carla_map
        os.makedirs(self.risk_output_folder, exist_ok=True)
        self.interpreter = RiskScenarioInterpreter()
        self.scenario_generator = ExistingWorldScenarioGenerator()

    def run(
        self,
        risk_spec_path: Optional[str] = None,
        user_request: str = "",
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        spawn_payload = self._load_json(self._spawn_payload_path())
        match_report = self._load_json(self._scene_match_path())
        scene_understanding = self._load_json_if_exists(self._scene_understanding_path())
        actor_context = build_actor_context(spawn_payload)
        write_to_file(
            self._risk_actor_context_path(),
            json.dumps(actor_context, indent=2, sort_keys=True),
        )

        if risk_spec_path:
            risk_spec = self._load_json(risk_spec_path)
        else:
            if not image_path:
                raise ValueError("image_path is required when risk_spec_path is not provided.")
            risk_spec = self.interpreter.call_agent(
                user_request,
                {
                    "output_fn": self._risk_scenario_spec_path(),
                    "scene_id": self.scene_id,
                    "image_path": image_path,
                    "scene_understanding": scene_understanding,
                    "scene_match": match_report,
                    "spawn_payload": spawn_payload,
                    "actor_context": actor_context,
                },
            )
            evidence_error = require_risk_evidence(risk_spec)
            if evidence_error:
                raise ValueError(f"Invalid VLM risk scenario spec: {evidence_error}")

        return self.build_artifacts(risk_spec, match_report, spawn_payload)

    def build_artifacts(
        self,
        risk_spec: Dict[str, Any],
        match_report: Dict[str, Any],
        spawn_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_spec, validation_error = validate_risk_scenario_spec(
            risk_spec,
            spawn_payload,
        )
        if validation_error:
            raise ValueError(f"Invalid risk scenario spec: {validation_error}")

        expanded_spec = expand_candidates_by_speed(normalized_spec)
        expanded_spec, validation_error = validate_risk_scenario_spec(
            expanded_spec,
            spawn_payload,
        )
        if validation_error:
            raise ValueError(f"Invalid expanded risk scenario spec: {validation_error}")

        write_to_file(
            self._risk_scenario_spec_path(),
            json.dumps(expanded_spec, indent=2, sort_keys=True),
        )
        risk_evidence = expanded_spec.get("risk_evidence")
        risk_evidence_path = None
        if risk_evidence is not None:
            risk_evidence_path = self._risk_evidence_path()
            write_to_file(
                risk_evidence_path,
                json.dumps(risk_evidence, indent=2, sort_keys=True),
            )

        samples = sample_all_risk_scenarios(
            expanded_spec,
            samples_per_candidate=self.samples_per_candidate,
            seed=self.seed,
        )
        map_name = self._normalize_carla_world_name(
            self.carla_map or match_report.get("world_name")
        )

        artifacts = []
        for sample in samples:
            candidate_index = int(sample["candidate_index"])
            sample_index = int(sample["sample_index"])
            sample_path = self._risk_sample_path(candidate_index, sample_index)
            write_to_file(sample_path, json.dumps(sample, indent=2, sort_keys=True))

            artifact = {
                "candidate_index": candidate_index,
                "sample_index": sample_index,
                "risk_candidate_id": sample.get("risk_candidate_id"),
                "family": sample.get("family"),
                "template_id": sample.get("template_id"),
                "execution_status": sample.get("execution_status"),
                "ego_speed_label": sample.get("ego_speed_label"),
                "ego_target_speed_mps": sample.get("ego_target_speed_mps"),
                "source_candidate_index": sample.get("source_candidate_index"),
                "source_candidate_id": sample.get("source_candidate_id"),
                "sample_path": sample_path,
                "script_path": None,
                "metrics_path": None,
            }

            if sample.get("execution_status") == "implemented":
                metrics_path = self._risk_metrics_path(candidate_index, sample_index)
                script_path = self._risk_scene_path(candidate_index, sample_index)
                script = self.scenario_generator.build_dynamic_risk_scene_script(
                    spawn_payload_filename=os.path.abspath(self._spawn_payload_path()),
                    risk_sample_filename=os.path.abspath(sample_path),
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

            artifacts.append(artifact)

        summary = {
            "enabled": True,
            "scene_id": self.scene_id,
            "static_output_folder": self.output_folder,
            "risk_output_folder": self.risk_output_folder,
            "risk_spec_path": self._risk_scenario_spec_path(),
            "risk_evidence_path": risk_evidence_path,
            "risk_evidence_warnings": expanded_spec.get("risk_evidence_warnings", []),
            "actor_context_path": self._risk_actor_context_path(),
            "samples_per_candidate": self.samples_per_candidate,
            "seed": self.seed,
            "speed_hypotheses_source": expanded_spec.get("speed_hypotheses_source"),
            "unsupported_risk_hypotheses": normalized_spec.get(
                "unsupported_risk_hypotheses",
                [],
            ),
            "artifacts": artifacts,
        }
        write_to_file(
            self._risk_generation_summary_path(),
            json.dumps(summary, indent=2, sort_keys=True),
        )
        return summary

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

    def _scene_understanding_path(self) -> str:
        direct = self._first_existing_path(
            f"{self.scene_id}_su.json",
            f"{self.scene_id}_scene_understanding.json",
        )
        if os.path.exists(direct):
            return direct
        base_scene_id = re.sub(r"(_cand\d+|_c\d+)$", "", self.scene_id)
        return self._first_existing_path(
            f"{base_scene_id}_su.json",
            f"{base_scene_id}_scene_understanding.json",
        )

    def _risk_scenario_spec_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_spec.json")

    def _risk_evidence_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_evidence.json")

    def _risk_actor_context_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_actors.json")

    def _risk_sample_path(self, candidate_index: int, sample_index: int) -> str:
        return join(
            self.risk_output_folder,
            f"{self.scene_id}_r{candidate_index:03d}_s{sample_index:03d}.json",
        )

    def _risk_scene_path(self, candidate_index: int, sample_index: int) -> str:
        return join(
            self.risk_output_folder,
            f"{self.scene_id}_r{candidate_index:03d}_s{sample_index:03d}.py",
        )

    def _risk_metrics_path(self, candidate_index: int, sample_index: int) -> str:
        return join(
            self.risk_output_folder,
            f"{self.scene_id}_r{candidate_index:03d}_m{sample_index:03d}.json",
        )

    def _risk_generation_summary_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_risk_summary.json")

    @staticmethod
    def _normalize_carla_world_name(map_name: Optional[str]) -> Optional[str]:
        if not map_name:
            return None
        return str(map_name).split("/")[-1]

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Required JSON file not found: {path}")
        return json.loads(read_file(path))

    @staticmethod
    def _load_json_if_exists(path: str) -> Dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        return json.loads(read_file(path))

    def _first_existing_path(self, *relative_paths: str) -> str:
        for relative_path in relative_paths:
            path = join(self.output_folder, relative_path)
            if os.path.exists(path):
                return path
        return join(self.output_folder, relative_paths[0])
