"""Orchestrator for the pure-LLM (no template library) Scenic experiment.

Mirrors ``tools/risk_scenario_runner.py`` but swaps the template-based risk spec +
CARLA Python backend for a two-stage pure-LLM flow that emits Scenic code:

    image --> LlmAccidentPredictor  --> {scene_id}_candidates.json
           --> LlmScenicGenerator   --> {scene_id}_cand{N}.scenic
           --> scenic compile check --> {scene_id}_cand{N}_compile.json
           --> {scene_id}_pure_llm_summary.json

It reuses the already-reconstructed ``{scene_id}_actors.json`` (spawn_payload, CARLA
world coordinates) and ``{scene_id}_match.json`` (map / world_name).
"""

import json
import os
from os.path import join
from typing import Any, Callable, Dict, List, Optional

from agents.llm_accident_predictor import LlmAccidentPredictor
from agents.llm_scenic_generator import LlmScenicGenerator
from tools.risk_scenario_pipeline import build_actor_context
from tools.scenic_compile import check_scenic_compile
from tools.utils import read_file, write_to_file


DEFAULT_CARLA_MAPS_DIR = (
    "/home/zx/code/autodirve/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive"
)


class PureLlmScenicRunner:
    def __init__(
        self,
        output_folder: str,
        scene_id: str,
        risk_output_folder: Optional[str] = None,
        ego_speed_mps: float = 10.0,
        carla_maps_dir: Optional[str] = None,
        max_retries: int = 2,
        carla_map: Optional[str] = None,
        scenic_conda_env: str = "scenicNL",
        enable_compile: bool = True,
        predictor: Optional[LlmAccidentPredictor] = None,
        generator: Optional[LlmScenicGenerator] = None,
        compile_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.output_folder = output_folder
        self.risk_output_folder = risk_output_folder or output_folder
        self.scene_id = scene_id
        self.ego_speed_mps = float(ego_speed_mps)
        self.carla_maps_dir = (
            carla_maps_dir
            or os.getenv("CARLA_MAPS_DIR")
            or DEFAULT_CARLA_MAPS_DIR
        )
        self.max_retries = max(0, int(max_retries))
        self.carla_map = carla_map
        self.scenic_conda_env = scenic_conda_env
        self.enable_compile = enable_compile
        os.makedirs(self.risk_output_folder, exist_ok=True)
        self.predictor = predictor or LlmAccidentPredictor(self.ego_speed_mps)
        self.generator = generator or LlmScenicGenerator(self.ego_speed_mps)
        self.compile_fn = compile_fn or check_scenic_compile

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
        map_header = self._build_map_header(map_name)

        artifacts: List[Dict[str, Any]] = []
        accident_candidates = candidates.get("accident_candidates") or []
        for index, candidate in enumerate(accident_candidates):
            scenic_path = self._scenic_path(index)
            compile_path = self._compile_report_path(index)
            body, compile_result, attempts = self._generate_with_repair(
                candidate=candidate,
                actor_context=actor_context,
                map_name=map_name,
                map_header=map_header,
                scenic_path=scenic_path,
            )
            write_to_file(compile_path, json.dumps(compile_result, indent=2, sort_keys=True))
            artifacts.append(
                {
                    "candidate_index": index,
                    "candidate_id": candidate.get("id"),
                    "accident_type": candidate.get("accident_type"),
                    "scenic_path": scenic_path,
                    "compile_report_path": compile_path,
                    "compiled": compile_result.get("compiled"),
                    "attempts": attempts,
                }
            )

        summary = {
            "enabled": True,
            "experiment": "pure_llm_scenic",
            "scene_id": self.scene_id,
            "static_output_folder": self.output_folder,
            "risk_output_folder": self.risk_output_folder,
            "ego_speed_mps": self.ego_speed_mps,
            "map_name": map_name,
            "carla_maps_dir": self.carla_maps_dir,
            "scenic_conda_env": self.scenic_conda_env,
            "compile_enabled": self.enable_compile,
            "candidates_path": self._candidates_path(),
            "actor_context_path": self._actor_context_path(),
            "num_candidates": len(accident_candidates),
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
        map_name: str,
        map_header: str,
        scenic_path: str,
    ):
        prior_attempt: Optional[str] = None
        compile_error: Optional[str] = None
        compile_result: Dict[str, Any] = {"compiled": None, "error": None, "skipped": True}
        body = ""
        attempts = 0
        for attempt in range(self.max_retries + 1):
            attempts = attempt + 1
            body = self.generator.generate(
                scene_id=self.scene_id,
                candidate=candidate,
                actor_context=actor_context,
                map_name=map_name,
                prior_attempt=prior_attempt,
                compile_error=compile_error,
            )
            full_program = f"{map_header}\n\n{body}\n"
            write_to_file(scenic_path, full_program)

            if not self.enable_compile:
                compile_result = {"compiled": None, "error": None, "skipped": True, "attempt": attempts}
                break

            result = self.compile_fn(scenic_path, conda_env=self.scenic_conda_env)
            compile_result = {**result, "attempt": attempts}
            if result.get("compiled"):
                break
            prior_attempt = body
            compile_error = result.get("error")
        return body, compile_result, attempts

    def _build_actor_context(self, spawn_payload: Dict[str, Any]) -> Dict[str, Any]:
        actor_context = build_actor_context(spawn_payload)
        blueprints = {
            str(entity.get("id")): entity.get("blueprint_name")
            for entity in (spawn_payload.get("entities") or [])
            if isinstance(entity, dict) and entity.get("id") is not None
        }
        for row in actor_context.get("actors", []):
            row["blueprint_name"] = blueprints.get(str(row.get("id")))
        return actor_context

    def _build_map_header(self, map_name: str) -> str:
        xodr_path = join(self.carla_maps_dir, f"{map_name}.xodr")
        return (
            f'param map = "{xodr_path}"\n'
            f"param carla_map = '{map_name}'\n"
            "model scenic.simulators.carla.model"
        )

    @staticmethod
    def _normalize_world_name(world_name: Optional[str]) -> str:
        if not world_name:
            return "Town01"
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
        return join(self.risk_output_folder, f"{self.scene_id}_scenic_actors.json")

    def _scenic_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_cand{index:03d}.scenic")

    def _compile_report_path(self, index: int) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_cand{index:03d}_compile.json")

    def _summary_path(self) -> str:
        return join(self.risk_output_folder, f"{self.scene_id}_pure_llm_summary.json")

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
