import os
import re
import sys
import json
import math
import shutil
import argparse
import subprocess
from copy import deepcopy
from datetime import datetime
from os.path import join
from typing import Optional
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import read_file, write_to_file
from agents.net_generator import NetGenerator
from agents.obstacle_generator import ObstacleGenerator
from agents.universal_interpreter import UniInterpreter
from agents.scenario_generator import ScenarioGenerator
from tools.scene_map_matcher import SceneMapMatcher
from agents.scene_understanding_interpreter import SceneUnderstandingInterpreter
from agents.existing_world_scenario_generator import ExistingWorldScenarioGenerator
from agents.scene_verification_agent import SceneVerificationAgent
from tools.structured_pipeline import (
    apply_pairwise_ordering,
    build_projected_spawn_payload,
    build_relation_dsl,
    generate_initial_coordinates_from_relation_dsl,
    project_entities_to_carla_context,
    refine_projected_coordinates_with_pairwise_relations,
    validate_relation_layout,
    _infer_ego_lane_offset,
    _scene_forward_lane_count,
    _find_lane_in_dense,
    _select_sibling_lane_cache_only,
    _next_wp_along_lane,
    _infer_ego_junction_target,
    _measure_forward_junction_distance,
    _slide_along_dense_waypoints,
    compute_cache_longitudinal_slide,
    slide_anchor_along_segment,
)
from tools.junction_placement import (
    reproject_actors_for_structure,
    validate_structural_reprojection,
)
from tools.actor_graph_verifier import (
    build_render_actor_graph_from_spawn_payload,
    build_source_actor_graph,
    compare_actor_graphs,
)

load_dotenv()


class AutoGenerator:
    """
    A class responsible for generating road network descriptions, obstacles, and full simulation scenarios.
    """

    def __init__(self, output_folder, info_dict=None):
        """
        Initialize the AutoGenerator with required components.
        """
        if info_dict is None:
            info_dict = {"input_type": "image"}
        self.input_type = info_dict["input_type"]

        os.makedirs(output_folder, exist_ok=True)

        self.carla_host = info_dict.get("carla_host", "localhost")
        self.carla_port = info_dict.get("carla_port", 2000)
        self.carla_timeout = info_dict.get("carla_timeout", 10.0)
        self.carla_map = info_dict.get("carla_map")
        self.spawn_point_limit = info_dict.get("spawn_point_limit", 12)
        self.require_carla_connection = info_dict.get("require_carla_connection", True)
        self.debug_artifacts = info_dict.get("debug_artifacts", False)
        self.merge_user_description = info_dict.get("merge_user_description", True)
        self.enable_scene_verify = info_dict.get("enable_scene_verify", False)
        self.generate_quick_bev_preview = info_dict.get("generate_quick_bev_preview", True)
        self.legacy_actor_layout = bool(info_dict.get("legacy_actor_layout", False))
        self.verify_max_rounds = int(info_dict.get("verify_max_rounds", 2))
        self.verify_min_score = float(info_dict.get("verify_min_score", 0.70))
        self.verify_mode = info_dict.get("verify_mode", "actor_graph")
        self.verify_script_timeout = float(info_dict.get("verify_script_timeout", 120.0))
        self.map_match_blacklist_radius_m = float(
            info_dict.get("map_match_blacklist_radius_m", 35.0)
        )
        self.topology_cache_dir = info_dict.get("topology_cache_dir")

        self.scene_understanding_interpreter = SceneUnderstandingInterpreter()
        self.scene_verification_agent = SceneVerificationAgent()
        self.interpreter = UniInterpreter(info_dict["input_type"])
        self.net_generator = NetGenerator(output_folder)
        self.obstacle_generator = ObstacleGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.structured_scenario_generator = ExistingWorldScenarioGenerator()
        self.enable_scene_match = info_dict.get("enable_scene_match", True)
        self.scene_matcher = SceneMapMatcher(
            host=self.carla_host,
            port=self.carla_port,
            timeout=self.carla_timeout,
            load_world_name=self._normalize_carla_world_name(self.carla_map),
            topology_weight=float(info_dict.get("map_match_topology_weight", 0.70)),
            side_context_weight=float(info_dict.get("map_match_side_context_weight", 0.20)),
            auxiliary_weight=float(info_dict.get("map_match_auxiliary_weight", 0.10)),
            blacklist_radius_m=self.map_match_blacklist_radius_m,
            topology_cache_dir=self.topology_cache_dir,
        )
        self.carla_spawn_context = self._load_carla_spawn_context()
        actual_map_name = self._normalize_carla_world_name(
            (self.carla_spawn_context or {}).get("map_name")
        )
        if actual_map_name:
            self.carla_map = actual_map_name
            self.scene_matcher.load_world_name = actual_map_name
        self.output_folder = output_folder

    @staticmethod
    def _normalize_carla_world_name(map_name):
        if not map_name:
            return None
        return map_name.split("/")[-1]

    @staticmethod
    def _entity_requires_junction_structure(entity: dict) -> bool:
        """Return true only for actors already expressed in a junction frame.

        A crossing heading or a left/right arm hint from the VLM is not enough to
        prove the road topology is a junction; accident frames often contain
        sideways/rotated vehicles on ordinary roads.  Junction structure should
        be required by road topology evidence, not by actor pose alone.
        """
        if entity.get("junction_leg") or entity.get("junction_direction"):
            return True
        return str(entity.get("junction_placement") or "").strip().lower() == "frame"

    @classmethod
    def _scene_requires_junction_structure(cls, scene_understanding: dict) -> bool:
        road_network = scene_understanding.get("road_network") or {}
        map_matching = road_network.get("map_matching") or {}
        topology_type = str(map_matching.get("topology_type") or "").strip().lower()
        junction_type = str(map_matching.get("junction_type") or "").strip().lower()
        strong_junction_types = {
            "junction",
            "multi_branch",
            "t_junction",
            "cross_intersection",
            "intersection",
            "signalized_intersection",
            "roundabout",
        }
        if topology_type in strong_junction_types or junction_type in strong_junction_types:
            return True
        target_branch_count = map_matching.get("target_branch_count")
        if isinstance(target_branch_count, (int, float)) and target_branch_count >= 3:
            return True
        branches = map_matching.get("junction_branches")
        if isinstance(branches, dict) and bool(branches.get("known")):
            visible_branches = sum(
                1 for key in ("ahead", "left", "right") if bool(branches.get(key))
            )
            if visible_branches >= 2 and bool(map_matching.get("junction_visible")):
                return True
        for area in road_network.get("special_road_areas") or []:
            area_type = str((area or {}).get("type") or "").strip().lower()
            if area_type in {
                "junction",
                "intersection",
                "t_junction",
                "cross_intersection",
                "multi_branch",
            }:
                return True
        return False

    @classmethod
    def _coordinates_require_junction_structure(cls, coordinates: dict) -> bool:
        return any(
            cls._entity_requires_junction_structure(entity)
            for entity in coordinates.get("entities") or []
        )

    def _require_junction_structure(self, context: str) -> dict:
        matched_structure = (self.carla_spawn_context or {}).get("matched_structure")
        if not isinstance(matched_structure, dict) or matched_structure.get("error"):
            source = (self.carla_spawn_context or {}).get("matched_structure_source") or "missing"
            raise RuntimeError(
                f"{context} requires matched_structure for junction placement, "
                f"but none is available (source={source})."
            )
        if str(matched_structure.get("kind") or "").lower() != "junction":
            raise RuntimeError(
                f"{context} requires junction matched_structure, "
                f"but got kind={matched_structure.get('kind')!r}."
            )
        return matched_structure

    @staticmethod
    def _matched_structure_summary(matched_structure: dict, source: str = None) -> dict:
        if not isinstance(matched_structure, dict):
            return {"available": False, "source": source or "missing"}
        legs = matched_structure.get("legs") or []
        leg_names = [
            str(leg.get("name") or "")
            for leg in legs
            if isinstance(leg, dict) and leg.get("name")
        ]
        return {
            "available": not bool(matched_structure.get("error")),
            "source": source or matched_structure.get("source") or "unknown",
            "kind": matched_structure.get("kind"),
            "error": matched_structure.get("error"),
            "junction_id": matched_structure.get("junction_id"),
            "leg_count": matched_structure.get("leg_count") or len(legs),
            "leg_names": leg_names,
            "has_ego_leg": "ego" in leg_names,
            "has_right_leg": "right" in leg_names,
            "has_left_leg": "left" in leg_names,
            "has_opposite_leg": "opposite" in leg_names,
            "ego_approach_heading_deg": matched_structure.get("ego_approach_heading_deg"),
            "ego_distance_to_center_m": matched_structure.get("ego_distance_to_center_m"),
            "junction_radius_m": matched_structure.get("junction_radius_m"),
        }

    def _persist_matched_structure(
        self,
        scene_id: str,
        match_report_path: str,
        matched_structure: dict,
        source: str,
    ) -> None:
        summary = self._matched_structure_summary(matched_structure, source)
        write_to_file(
            self._matched_structure_path(scene_id),
            json.dumps(
                {
                    "source": source,
                    "summary": summary,
                    "matched_structure": matched_structure
                    if isinstance(matched_structure, dict)
                    else None,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        match_report = self._load_json_if_exists(match_report_path)
        if not match_report:
            return
        best_match = match_report.setdefault("best_match", {})
        best_match["matched_structure"] = matched_structure
        best_match["matched_structure_source"] = source
        best_match["matched_structure_summary"] = summary
        write_to_file(
            match_report_path,
            json.dumps(match_report, indent=2, sort_keys=True),
        )

    def _load_carla_spawn_context(self):
        if not self.require_carla_connection:
            return self._fallback_carla_spawn_context("CARLA connection disabled.")

        # When an offline topology cache is present, map matching runs entirely
        # without CARLA.  The matched anchor lane is injected later by
        # _apply_match_to_spawn_context, and dense waypoints are sampled by
        # _sample_dense_local_waypoints — so an initial CARLA connection here
        # is not needed and would produce a misleading "spawn points cached"
        # message.  Use a "deferred" status so BEV capture is not blocked.
        if self.topology_cache_dir and os.path.isdir(self.topology_cache_dir):
            return {
                "status": "deferred",
                "map_name": None,
                "spawn_points": [],
                "topology_sample": self._fallback_topology_sample(),
            }

        try:
            import carla
        except ImportError as exc:
            return self._fallback_carla_spawn_context(
                f"CARLA Python API is not available: {exc}"
            )

        try:
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.carla_timeout)
            world = (
                client.load_world(self.carla_map)
                if self.carla_map
                else client.get_world()
            )
            world_map = world.get_map()
            spawn_points = world_map.get_spawn_points()
        except Exception as exc:
            return self._fallback_carla_spawn_context(
                f"Failed to connect to CARLA before object generation: {exc}"
            )

        if not spawn_points:
            return self._fallback_carla_spawn_context(
                f"Connected to CARLA map {world_map.name}, but no spawn points were found."
            )

        sampled_spawn_points = []
        for index, transform in enumerate(spawn_points[: self.spawn_point_limit]):
            sampled_spawn_points.append(
                {
                    "index": index,
                    "location": {
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z,
                    },
                    "rotation": {
                        "pitch": transform.rotation.pitch,
                        "yaw": transform.rotation.yaw,
                        "roll": transform.rotation.roll,
                    },
                }
            )

        topology_sample = []
        try:
            topology = world_map.get_topology()
        except Exception:
            topology = []
        for start_wp, end_wp in topology[: max(1, self.spawn_point_limit)]:
            topology_sample.append(
                {
                    "road_id": int(start_wp.road_id),
                    "lane_id": int(start_wp.lane_id),
                    "start": {
                        "x": float(start_wp.transform.location.x),
                        "y": float(start_wp.transform.location.y),
                        "z": float(start_wp.transform.location.z),
                        "yaw": float(start_wp.transform.rotation.yaw),
                        "is_junction": bool(start_wp.is_junction),
                    },
                    "end": {
                        "x": float(end_wp.transform.location.x),
                        "y": float(end_wp.transform.location.y),
                        "z": float(end_wp.transform.location.z),
                        "yaw": float(end_wp.transform.rotation.yaw),
                        "is_junction": bool(end_wp.is_junction),
                    },
                }
            )

        print(f"Connected to CARLA map {world_map.name}.")
        return {
            "status": "available",
            "map_name": world_map.name,
            "spawn_points": sampled_spawn_points,
            "topology_sample": topology_sample or self._fallback_topology_sample(),
        }

    @staticmethod
    def _fallback_topology_sample():
        return [
            {
                "road_id": 1,
                "lane_id": -1,
                "start": {
                    "x": 0.0,
                    "y": -1.75,
                    "z": 0.0,
                    "yaw": 0.0,
                    "is_junction": False,
                },
                "end": {
                    "x": 80.0,
                    "y": -1.75,
                    "z": 0.0,
                    "yaw": 0.0,
                    "is_junction": False,
                },
            }
        ]

    def _fallback_carla_spawn_context(self, failure_reason):
        return {
            "status": "unavailable",
            "map_name": None,
            "spawn_points": [],
            "topology_sample": self._fallback_topology_sample(),
            "failure_reason": failure_reason,
        }

    def _scene_understanding_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_su.json")

    def _scene_match_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_match.json")

    def _spawn_payload_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_actors.json")

    def _matched_structure_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_matched_structure.json")

    def _final_scene_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_static.py")

    def _verification_path(self, scene_id: str, round_index: int) -> str:
        return join(self.output_folder, f"{scene_id}_verify_r{round_index}.json")

    def _bev_path(self, scene_id: str, round_index: int) -> str:
        return join(self.output_folder, f"{scene_id}_bev_r{round_index}.png")

    def _quick_bev_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_quick_bev.png")

    def _ego_view_path(self, scene_id: str, round_index: int) -> str:
        return join(self.output_folder, f"{scene_id}_ego_r{round_index}.png")

    def _source_actor_graph_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_source_actor_graph.json")

    def _render_actor_graph_path(self, scene_id: str, round_index: int) -> str:
        return join(self.output_folder, f"{scene_id}_render_actor_graph_r{round_index}.json")

    def _actor_graph_repair_plan_path(self, scene_id: str, round_index: int) -> str:
        return join(self.output_folder, f"{scene_id}_actor_graph_repair_plan_r{round_index}.json")

    def _write_debug_json(self, scene_id: str, suffix: str, payload: dict) -> None:
        if not self.debug_artifacts:
            return
        write_to_file(
            join(self.output_folder, f"{scene_id}_{suffix}.json"),
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def adapt_generation_params_for_scene(self, scene_understanding):
        road_network = scene_understanding.get("road_network", {})
        lane_groups = road_network.get("lane_groups", []) or []
        params = {"lane_width_class": "standard", "roadside_space": False}
        if lane_groups and isinstance(lane_groups[0], dict):
            params["lane_width_class"] = lane_groups[0].get(
                "lane_width_class", "standard"
            )

        roadside_relations = []
        for entity in scene_understanding.get("traffic_subjects", []) or []:
            if isinstance(entity, dict):
                roadside_relations.append(str(entity.get("lane_side_relation") or ""))
        for entity in scene_understanding.get("background_traffic", []) or []:
            if isinstance(entity, dict):
                roadside_relations.append(str(entity.get("lane_side_relation") or ""))
        params["roadside_space"] = any(
            relation in {"sidewalk_left", "sidewalk_right"}
            for relation in roadside_relations
        )
        self.generation_params = params
        return params

    def extract_net_description(self, scene_id):
        """
        Extract the road network description from a stored text file.
        """
        text = read_file(join(self.output_folder, f"{scene_id}.txt"))
        match = re.search(
            r"##\s*Road Net Description\s*:\s*(.*?)\s*##",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    def extract_scene_description(self, scene_id):
        """
        Extract the scenario description from a stored text file.
        """
        text = read_file(join(self.output_folder, f"{scene_id}.txt"))
        match = re.search(r"## Scenario Description:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def extract_split_sections_from_text(text):
        section_names = (
            "Road Net Description",
            "Road Users Description",
            "Static Objects Description",
            "Vehicles' Locations and Behaviors",
            "Scenario Description",
        )
        sections = {}
        for index, section_name in enumerate(section_names):
            next_name = section_names[index + 1] if index + 1 < len(section_names) else None
            if next_name:
                pattern = (
                    rf"##\s*{re.escape(section_name)}\s*:\s*(.*?)"
                    rf"\s*##\s*{re.escape(next_name)}\s*:"
                )
            else:
                pattern = rf"##\s*{re.escape(section_name)}\s*:\s*(.*)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            sections[section_name] = match.group(1).strip() if match else ""
        return sections

    def extract_scene_sections(self, scene_id):
        text = read_file(join(self.output_folder, f"{scene_id}.txt"))
        return self.extract_split_sections_from_text(text)

    def generate_net(self, scene_id, road_net_description):
        """
        Generate a road network file using the NetGenerator component.
        """
        print("Generating road network.......")
        return self.net_generator.call_agent(
            road_net_description,
            scene_id,
            {"output_fn": join(self.output_folder, f"{scene_id}_net.txt")},
        )

    def generate_objects(self, scene_id, scenario_description):
        """
        Generate objects within the simulation scene.
        """
        scene_sections = self.extract_scene_sections(scene_id)
        net_info = self.obstacle_generator.extract_network_info_with_xml(
            join(self.output_folder, f"{scene_id}_net.txt"),
            join(self.output_folder, f"{scene_id}.net.xml"),
        )
        spawn_points_info = self.obstacle_generator.format_carla_spawn_points_info(
            self.carla_spawn_context
        )
        final_request = self.obstacle_generator.format_scene_context(
            scene_sections, net_info, spawn_points_info
        )
        print("Generating objects .......")
        return self.obstacle_generator.call_agent(
            final_request, scene_id, self.output_folder
        )

    def generate_scene(self, scene_id, scenario_description):
        """
        Generate the full simulation scene file.
        """
        print("Generating full scene .......")
        self.scenario_generator.run_scenario_generation(
            join(self.output_folder, f"{scene_id}_scene.py"),
            scene_id,
            scenario_description,
            self.output_folder,
        )

    def analyze_scene_match(self, scene_id):
        """
        Match the generated scene to a region in the current CARLA world.
        """
        if not self.enable_scene_match:
            return None

        print("Analyzing scene-to-CARLA match .......")
        try:
            report_path = self.scene_matcher.analyze_scene_assets(
                scene_id, self.output_folder
            )
            print(f"Scene match report saved to {report_path}")
            matched_scene_path = self.scene_matcher.apply_match_to_scene_script(
                scene_id, self.output_folder
            )
            if matched_scene_path:
                print(f"Matched scene script saved to {matched_scene_path}")
            return report_path
        except Exception as exc:
            print(f"Scene matching skipped for {scene_id}: {exc}")
            return None

    def generate_scene_understanding(self, user_request, input_dict):
        print("Generating structured scene understanding.......")
        scene_id = input_dict["scene_id"]
        output_fn = self._scene_understanding_path(scene_id)
        scene_understanding = self.scene_understanding_interpreter.call_agent(
            user_request,
            {
                **input_dict,
                "output_fn": output_fn,
            },
        )
        user_scene_description = str(input_dict.get("user_scene_description") or "").strip()
        if self.merge_user_description and user_scene_description:
            print("Merging user scene description with VLM scene understanding.......")
            scene_understanding = (
                self.scene_understanding_interpreter.merge_with_user_description(
                    scene_understanding,
                    user_scene_description,
                    output_fn,
                )
            )
        else:
            scene_understanding.setdefault("metadata", {})[
                "user_description_applied"
            ] = False

        write_to_file(
            output_fn,
            json.dumps(scene_understanding, indent=2, sort_keys=True),
        )
        self.adapt_generation_params_for_scene(scene_understanding)
        return scene_understanding

    def generate_relation_dsl(self, scene_id, scene_understanding):
        print("Generating relation DSL.......")
        relation_dsl = build_relation_dsl(
            scene_understanding,
            self.carla_spawn_context,
            legacy_actor_layout=self.legacy_actor_layout,
        )
        self._write_debug_json(scene_id, "relation_dsl", relation_dsl)
        return relation_dsl

    def generate_initial_coordinates(self, scene_id, relation_dsl):
        print("Generating initial coordinates.......")
        raw_initial = generate_initial_coordinates_from_relation_dsl(relation_dsl)
        self._write_debug_json(scene_id, "coordinates_raw_initial", raw_initial)
        return raw_initial

    def apply_pairwise_ordering_step(self, scene_id, raw_initial, relation_dsl):
        print("Applying pairwise ordering.......")
        ordered = apply_pairwise_ordering(raw_initial, relation_dsl)
        self._write_debug_json(scene_id, "coordinates_raw_ordered", ordered)
        return ordered

    def project_entities_step(self, scene_id, ordered, scene_understanding):
        print("Projecting entities onto CARLA map context.......")
        projected = project_entities_to_carla_context(
            ordered,
            scene_understanding,
            self.carla_spawn_context,
        )
        self._write_debug_json(scene_id, "coordinates_projected", projected)
        return projected

    def refine_coordinates_step(self, scene_id, projected, relation_dsl):
        print("Refining coordinates with pairwise relations.......")
        refined = refine_projected_coordinates_with_pairwise_relations(
            projected,
            relation_dsl,
        )
        self._write_debug_json(scene_id, "coordinates_projected_refined", refined)
        return refined

    def reproject_junction_step(self, scene_id, refined):
        """Re-place actors inside the matched structural reference frame (Phase 4).

        Junction scenes are re-placed onto real legs (cross/oncoming/turning
        actors land on the correct arm facing the right way); straight/curved
        roads are re-laid along the real lane centreline so a curve bends instead
        of drifting off a fixed axis, with heading taken from the lane tangent.
        No-op when no structural description was matched.
        """
        if self.legacy_actor_layout:
            refined.setdefault("metadata", {})["layout_version"] = "legacy"
            return refined
        matched_structure = (self.carla_spawn_context or {}).get("matched_structure")
        if not isinstance(matched_structure, dict):
            return refined
        kind = str(matched_structure.get("kind") or "").lower()
        if kind == "junction":
            print("Re-placing actors in junction reference frame.......")
        elif kind in ("road_segment", "road", "straight", "curve"):
            print("Re-laying actors along matched road centreline.......")
        else:
            if self._coordinates_require_junction_structure(refined):
                self._require_junction_structure("Junction actor layout")
            return refined
        refined = reproject_actors_for_structure(refined, matched_structure)
        structural_validation = validate_structural_reprojection(refined)
        refined.setdefault("metadata", {})["structural_validation"] = {
            "status": structural_validation.get("status"),
            "summary": structural_validation.get("summary", {}),
        }
        if structural_validation.get("status") == "fail":
            print(
                "  [structural_validation] failed: "
                f"{structural_validation.get('summary', {}).get('failed', 0)} issue(s)"
            )
        self._write_debug_json(scene_id, "structural_validation", structural_validation)
        self._write_debug_json(scene_id, "coordinates_structural_reprojected", refined)
        return refined

    def validate_layout_step(self, scene_id, relation_dsl, refined):
        print("Validating relation layout.......")
        validation = validate_relation_layout(relation_dsl, refined)
        self._write_debug_json(scene_id, "relation_validation", validation)
        return validation

    def analyze_structured_scene_match(
        self,
        scene_id,
        scene_understanding,
        refined_coordinates,
        validation,
    ):
        print("Analyzing structured scene-to-CARLA match.......")
        if not self.enable_scene_match:
            report = {
                "scene_id": scene_id,
                "source_files": {},
                "status": "skipped",
                "world_name": None,
                "scene_features": {},
                "candidate_summary": {},
                "best_match": None,
                "projected_layout": {},
                "reason": "Scene matching disabled.",
            }
            report_path = self._scene_match_path(scene_id)
            write_to_file(report_path, json.dumps(report, indent=2, sort_keys=True))
            return report_path

        report_path = self.scene_matcher.analyze_structured_scene_assets(
            scene_id=scene_id,
            output_folder=self.output_folder,
            scene_understanding=scene_understanding,
            refined_coordinates=refined_coordinates,
            validation=validation,
            spawn_context=self.carla_spawn_context,
        )
        print(f"Scene match report saved to {report_path}")
        return report_path

    def analyze_topology_scene_match(
        self,
        scene_id: str,
        scene_understanding: dict,
        blacklist_locations=None,
        image_path: str = None,
    ) -> str:
        print("Analyzing topology-first scene-to-CARLA match.......")
        if not self.enable_scene_match:
            report = {
                "scene_id": scene_id,
                "source_files": {},
                "status": "skipped",
                "world_name": None,
                "scene_features": {},
                "candidate_summary": {},
                "best_match": None,
                "projected_layout": {},
                "reason": "Scene matching disabled.",
            }
            report_path = self._scene_match_path(scene_id)
            write_to_file(report_path, json.dumps(report, indent=2, sort_keys=True))
            return report_path
        report_path = self.scene_matcher.analyze_topology_scene_assets(
            scene_id=scene_id,
            output_folder=self.output_folder,
            scene_understanding=scene_understanding,
            spawn_context=self.carla_spawn_context,
            blacklist_locations=blacklist_locations or [],
            image_path=image_path,
        )
        print(f"Topology scene match report saved to {report_path}")
        return report_path

    def _apply_match_to_spawn_context(self, match_report_path: str) -> None:
        match_report = self._load_json_if_exists(match_report_path)
        best_match = match_report.get("best_match") or {}
        candidate_lane = best_match.get("candidate_lane")
        if not isinstance(candidate_lane, dict):
            return
        # Carry the candidate's cached distance-to-junction so cache-only ego
        # longitudinal alignment can run without a live CARLA connection.
        candidate_lane = dict(candidate_lane)
        candidate_features = best_match.get("candidate_features") or {}
        if "distance_to_junction_ahead" not in candidate_lane:
            dist = candidate_features.get("distance_to_junction_ahead")
            if isinstance(dist, (int, float)):
                candidate_lane["distance_to_junction_ahead"] = float(dist)
        # Carry the sibling-lane chain (with baked start geometry) so cache-only
        # ego lateral alignment can hop to the correct driving lane without a
        # live CARLA connection.
        if "same_direction_lane_evidence" not in candidate_lane:
            evidence = candidate_features.get("same_direction_lane_evidence")
            if isinstance(evidence, dict):
                candidate_lane["same_direction_lane_evidence"] = evidence
        topology_sample = list((self.carla_spawn_context or {}).get("topology_sample") or [])
        self.carla_spawn_context["topology_sample"] = [candidate_lane] + topology_sample
        # Carry the structural description (junction centre+legs / road segment)
        # for reference-frame actor placement (Phase 4).
        matched_structure = best_match.get("matched_structure")
        if isinstance(matched_structure, dict) and not matched_structure.get("error"):
            self.carla_spawn_context["matched_structure"] = matched_structure
            self.carla_spawn_context["matched_structure_source"] = "match_report"
        else:
            self.carla_spawn_context.pop("matched_structure", None)
            self.carla_spawn_context["matched_structure_source"] = "missing"

    def build_spawn_payload_from_match_or_fallback(
        self,
        scene_id: str,
        refined_coordinates: dict,
        match_report_path: str,
        force_projected_layout: bool = False,
    ) -> dict:
        print("Building final spawn payload.......")
        match_report = json.loads(read_file(match_report_path))
        coordinates_for_payload = deepcopy(refined_coordinates)
        self._prepend_ego_vehicle(coordinates_for_payload)
        used_projected_layout = self._apply_projected_layout_from_match(
            coordinates_for_payload,
            match_report,
            force_projected_layout=force_projected_layout,
        )
        best_match = match_report.get("best_match") or {}
        anchor = best_match.get("anchor") or {}
        if used_projected_layout and anchor:
            dx = float(anchor.get("dx", 0.0) or 0.0)
            dy = float(anchor.get("dy", 0.0) or 0.0)
            projected_layout = match_report.get("projected_layout") or {}
            for entity in coordinates_for_payload.get("entities", []) or []:
                if str(entity.get("id")) in projected_layout:
                    continue
                location = entity.get("location") or {}
                location["x"] = float(location.get("x", 0.0)) + dx
                location["y"] = float(location.get("y", 0.0)) + dy
        elif match_report.get("status") == "matched" and anchor:
            dx = float(anchor.get("dx", 0.0) or 0.0)
            dy = float(anchor.get("dy", 0.0) or 0.0)
            for entity in coordinates_for_payload.get("entities", []) or []:
                location = entity.get("location") or {}
                location["x"] = float(location.get("x", 0.0)) + dx
                location["y"] = float(location.get("y", 0.0)) + dy

        payload = build_projected_spawn_payload(coordinates_for_payload)
        write_to_file(
            self._spawn_payload_path(scene_id),
            json.dumps(payload, indent=2, sort_keys=True),
        )
        return payload

    @staticmethod
    def _apply_projected_layout_from_match(
        coordinates: dict,
        match_report: dict,
        force_projected_layout: bool = False,
    ) -> bool:
        if match_report.get("status") != "matched" and not force_projected_layout:
            return False
        projected_layout = match_report.get("projected_layout") or {}
        if not projected_layout:
            return False
        used = False
        for entity in coordinates.get("entities", []) or []:
            projected = projected_layout.get(str(entity.get("id")))
            if not projected:
                continue
            projected_location = projected.get("projected_location") or {}
            projected_rotation = projected.get("projected_rotation") or {}
            if projected_location:
                current_location = entity.setdefault("location", {})
                current_location["x"] = float(projected_location.get("x", current_location.get("x", 0.0)))
                current_location["y"] = float(projected_location.get("y", current_location.get("y", 0.0)))
                current_location["z"] = max(
                    0.3,
                    float(projected_location.get("z", current_location.get("z", 0.3)) or 0.3),
                )
                used = True
            if projected_rotation:
                current_rotation = entity.setdefault("rotation", {})
                current_rotation["pitch"] = float(
                    projected_rotation.get("pitch", current_rotation.get("pitch", 0.0))
                )
                current_rotation["yaw"] = float(
                    projected_rotation.get("yaw", current_rotation.get("yaw", 0.0))
                )
                current_rotation["roll"] = float(
                    projected_rotation.get("roll", current_rotation.get("roll", 0.0))
                )
                used = True
        return used

    def _prepend_ego_vehicle(self, coordinates: dict) -> None:
        entities = coordinates.setdefault("entities", [])
        if any(str(entity.get("id")) in {"ego", "ego_vehicle"} for entity in entities):
            return

        anchor_lane = (
            coordinates.get("selected_anchor_lane")
            or self._fallback_topology_sample()[0]
        )
        start = anchor_lane.get("start") or {}
        end = anchor_lane.get("end") or {}
        start_x = float(start.get("x", 0.0) or 0.0)
        start_y = float(start.get("y", 0.0) or 0.0)
        start_z = float(start.get("z", 0.0) or 0.0)
        if "yaw" in start:
            yaw = float(start.get("yaw", 0.0) or 0.0)
        else:
            yaw = math.degrees(
                math.atan2(
                    float(end.get("y", start_y) or start_y) - start_y,
                    float(end.get("x", start_x) or start_x) - start_x,
                )
            )

        entities.insert(
            0,
            {
                "id": "ego",
                "group_id": "ego",
                "source": "ego",
                "priority": "ego",
                "category": "car",
                "subtype": "car",
                "spawn_kind": "vehicle",
                "blueprint_name": "car",
                "motion_state": "stopped",
                "lane_side_relation": "same_lane",
                "heading_relation": "same_direction",
                "road_id": anchor_lane.get("road_id"),
                "appearance": {"color": "blue", "color_rgb": "54,116,168"},
                "location": {"x": start_x, "y": start_y, "z": max(start_z, 0.3)},
                "rotation": {"pitch": 0.0, "yaw": yaw, "roll": 0.0},
            },
        )

    def _index_ego_on_candidate_lane(self, scene_understanding: dict) -> None:
        """Adjust ego's position on the matched candidate lane in two steps.

        Step 1 (lateral): move ego to the correct driving lane within the road
        cross-section, inferred from traffic_subjects' lane_index_relation.

        Step 2 (longitudinal): slide ego forward/backward along the lane so
        its distance to the nearest forward junction matches the qualitative
        junction-visibility cues in the scene understanding.

        Both steps are best-effort: if dense_local_waypoints is empty (CARLA
        unavailable) or the required data is absent, the method returns without
        modifying spawn_context.
        """
        topology_sample = (self.carla_spawn_context or {}).get("topology_sample") or []
        dense_wps = (self.carla_spawn_context or {}).get("dense_local_waypoints") or []
        if not topology_sample:
            return
        if not dense_wps:
            # Cache-only mode (no live CARLA): use the sibling-lane geometry
            # baked into the candidate to hop ego to the correct driving lane
            # (lateral), then slide it longitudinally using the candidate's own
            # cached distance_to_junction_ahead and forward direction.
            self._index_ego_lateral_cache_only(scene_understanding, topology_sample)
            self._index_ego_longitudinal_cache_only(scene_understanding, topology_sample)
            return

        candidate_lane = topology_sample[0]

        # ------------------------------------------------------------------
        # Step 1: lateral – select the driving lane ego belongs to.
        # ------------------------------------------------------------------
        lane_offset = _infer_ego_lane_offset(scene_understanding)
        corrected_wp = _find_lane_in_dense(candidate_lane, lane_offset, dense_wps)
        if corrected_wp is not None:
            end_wp = _next_wp_along_lane(corrected_wp, dense_wps, lookahead_m=20.0)
            end_dict = {
                "x": float(end_wp.get("x", corrected_wp.get("x", 0.0))),
                "y": float(end_wp.get("y", corrected_wp.get("y", 0.0))),
                "z": float(end_wp.get("z", corrected_wp.get("z", 0.0))),
                "yaw": float(end_wp.get("yaw", corrected_wp.get("yaw", 0.0))),
                "is_junction": bool(end_wp.get("is_junction", False)),
            }
            start_dict = {
                "x": float(corrected_wp.get("x", 0.0)),
                "y": float(corrected_wp.get("y", 0.0)),
                "z": float(corrected_wp.get("z", 0.0)),
                "yaw": float(corrected_wp.get("yaw", 0.0)),
                "is_junction": bool(corrected_wp.get("is_junction", False)),
            }
            candidate_lane = {
                **candidate_lane,
                "road_id": corrected_wp.get("road_id", candidate_lane.get("road_id")),
                "lane_id": corrected_wp["lane_id"],
                "start": start_dict,
                "end": end_dict,
            }
            print(
                f"  [ego_indexer] lateral: lane_offset={lane_offset} → "
                f"lane_id={corrected_wp['lane_id']}"
            )

        # ------------------------------------------------------------------
        # Step 2: longitudinal – slide to correct junction distance.
        # ------------------------------------------------------------------
        constrain, target_m = _infer_ego_junction_target(scene_understanding)
        if constrain:
            start = candidate_lane.get("start") or {}
            end = candidate_lane.get("end") or {}
            actual_m = _measure_forward_junction_distance(start, end, dense_wps)
            if actual_m is not None:
                slide_m = actual_m - target_m
                if abs(slide_m) >= 5.0:
                    new_start = _slide_along_dense_waypoints(
                        start, end, slide_m, dense_wps, candidate_lane
                    )
                    if new_start is not None:
                        new_start_dict = {
                            "x": float(new_start.get("x", 0.0)),
                            "y": float(new_start.get("y", 0.0)),
                            "z": float(new_start.get("z", 0.0)),
                            "yaw": float(new_start.get("yaw", 0.0)),
                            "is_junction": bool(new_start.get("is_junction", False)),
                        }
                        candidate_lane = {**candidate_lane, "start": new_start_dict}
                        print(
                            f"  [ego_indexer] longitudinal: actual={actual_m:.1f}m "
                            f"target={target_m:.1f}m slide={slide_m:+.1f}m"
                        )

        self.carla_spawn_context["topology_sample"][0] = candidate_lane

    def _index_ego_lateral_cache_only(
        self, scene_understanding: dict, topology_sample: list
    ) -> None:
        """Hop ego to the correct driving lane without live CARLA.

        Uses the sibling-lane ``start`` geometry baked into the candidate's
        ``same_direction_lane_evidence`` (cache-only analogue of Step 1's
        ``_find_lane_in_dense``). No-ops when the cache carries no sibling
        geometry (older cache), the road is single-lane, or the inferred offset
        is out of range. Both endpoints translate by the same lateral delta so
        the start→end forward direction (used for relative actor placement) is
        preserved.
        """
        candidate_lane = topology_sample[0]
        if not isinstance(candidate_lane, dict):
            return
        lane_offset = _infer_ego_lane_offset(scene_understanding)
        target = _select_sibling_lane_cache_only(candidate_lane, lane_offset)
        if target is None:
            return
        old_start = candidate_lane.get("start") or {}
        old_end = candidate_lane.get("end") or {}
        target_start = target.get("start") or {}
        new_start = {
            "x": float(target_start.get("x", 0.0)),
            "y": float(target_start.get("y", 0.0)),
            "z": float(target_start.get("z", old_start.get("z", 0.0))),
            "yaw": float(target_start.get("yaw", old_start.get("yaw", 0.0))),
            "is_junction": bool(old_start.get("is_junction", False)),
        }
        dx = new_start["x"] - float(old_start.get("x", 0.0))
        dy = new_start["y"] - float(old_start.get("y", 0.0))
        new_end = {
            **old_end,
            "x": float(old_end.get("x", 0.0)) + dx,
            "y": float(old_end.get("y", 0.0)) + dy,
        }
        updated = {
            **candidate_lane,
            "road_id": target.get("road_id", candidate_lane.get("road_id")),
            "lane_id": target.get("lane_id", candidate_lane.get("lane_id")),
            "start": new_start,
            "end": new_end,
        }
        topology_sample[0] = updated
        self.carla_spawn_context["topology_sample"][0] = updated
        print(
            f"  [ego_indexer] lateral(cache): offset={lane_offset} → "
            f"lane_id={updated['lane_id']}"
        )

    def _index_ego_longitudinal_cache_only(
        self, scene_understanding: dict, topology_sample: list
    ) -> None:
        """Slide ego to the target junction distance without live CARLA.

        Uses the candidate lane's cached ``distance_to_junction_ahead`` as the
        current distance and the candidate lane's start→end vector as the forward
        direction. No-ops when the scene has no constrainable forward junction or
        the candidate lane carries no junction distance.
        """
        candidate_lane = topology_sample[0]
        if not isinstance(candidate_lane, dict):
            return
        constrain, target_m = _infer_ego_junction_target(scene_understanding)
        if not constrain:
            return
        actual_m = candidate_lane.get("distance_to_junction_ahead")
        slide_m = compute_cache_longitudinal_slide(actual_m, target_m)
        if slide_m == 0.0:
            return
        start = candidate_lane.get("start") or {}
        end = candidate_lane.get("end") or {}
        # Translate BOTH endpoints by the same delta so the start→end forward
        # direction (used for relative actor placement) is preserved; sliding
        # only `start` could move it past `end` and flip the scene heading.
        new_start = slide_anchor_along_segment(start, end, slide_m)
        dx = new_start["x"] - float(start.get("x", 0.0))
        dy = new_start["y"] - float(start.get("y", 0.0))
        new_end = {
            **end,
            "x": float(end.get("x", 0.0)) + dx,
            "y": float(end.get("y", 0.0)) + dy,
        }
        updated = {
            **candidate_lane,
            "start": new_start,
            "end": new_end,
            # Reflect the post-slide distance so downstream consumers/debug agree.
            "distance_to_junction_ahead": max(0.0, float(actual_m) - slide_m),
        }
        self.carla_spawn_context["topology_sample"][0] = updated
        print(
            f"  [ego_indexer] longitudinal (cache): actual={float(actual_m):.1f}m "
            f"target={target_m:.1f}m slide={slide_m:+.1f}m"
        )

    def _ego_lane_anchor_warning(
        self, scene_id: str, scene_understanding: dict
    ) -> Optional[dict]:
        """Best-effort sanity check: did ego land in the lane its relations imply?

        ego anchors every relative actor placement, so a mis-anchored ego
        cannot be repaired downstream (see results/auto_result_20260617_220440).
        This compares the lane offset ego *should* occupy (from
        ``_infer_ego_lane_offset``) against the offset it *actually* occupies on
        the matched CARLA road (nearest dense waypoint, using the actual map
        cross-section). A mismatch surfaces residual risks the lane-offset fix
        cannot remove — e.g. the VLM's forward_lane_count disagreeing with the
        matched road. Recorded as a warning only; ego is not auto-corrected.

        Returns a warning dict on discrepancy, else None (also None when CARLA
        waypoints or the ego transform are unavailable).
        """
        dense_wps = (self.carla_spawn_context or {}).get("dense_local_waypoints") or []
        if not dense_wps:
            return None
        payload = self._load_json_if_exists(self._spawn_payload_path(scene_id))
        ego = next(
            (
                entity
                for entity in (payload.get("entities") or [])
                if str(entity.get("id")) in {"ego", "ego_vehicle"}
            ),
            None,
        )
        if not ego:
            return None
        loc = ego.get("location") or {}
        ex, ey = float(loc.get("x", 0.0)), float(loc.get("y", 0.0))

        projected_lane = ego.get("projected_lane") if isinstance(ego.get("projected_lane"), dict) else {}
        try:
            projected_lane_id = int(projected_lane.get("lane_id"))
        except (TypeError, ValueError):
            projected_lane_id = None
        projected_sign = (
            1 if projected_lane_id and projected_lane_id > 0
            else -1 if projected_lane_id and projected_lane_id < 0
            else None
        )

        driving = [
            wp for wp in dense_wps
            if isinstance(wp.get("lane_id"), int) and wp["lane_id"] < 0
        ]
        if projected_sign is not None:
            driving = [
                wp for wp in dense_wps
                if isinstance(wp.get("lane_id"), int)
                and wp["lane_id"] != 0
                and (1 if wp["lane_id"] > 0 else -1) == projected_sign
            ]
        if not driving:
            return None
        nearest = min(
            driving,
            key=lambda wp: math.hypot(wp.get("x", 0.0) - ex, wp.get("y", 0.0) - ey),
        )
        road_id = nearest.get("road_id")
        actual_lane_id = nearest["lane_id"]

        # Actual cross-section on ego's road, sorted rightmost-first by
        # descending |lane_id| (curb lane = largest |id|) — same ordering as
        # _find_lane_in_dense. Raw -lane_id would rank -1 (the LEFT lane) first.
        best_dist_by_lane: dict = {}
        for wp in driving:
            if wp.get("road_id") != road_id:
                continue
            lid = wp["lane_id"]
            dist = math.hypot(wp.get("x", 0.0) - ex, wp.get("y", 0.0) - ey)
            if lid not in best_dist_by_lane or dist < best_dist_by_lane[lid]:
                best_dist_by_lane[lid] = dist
        sorted_lane_ids = sorted(best_dist_by_lane.keys(), key=lambda l: -abs(l))
        try:
            actual_offset = sorted_lane_ids.index(actual_lane_id)
        except ValueError:
            return None

        expected_offset = _infer_ego_lane_offset(scene_understanding)
        if actual_offset == expected_offset:
            return None
        return {
            "type": "ego_lane_anchor_warning",
            "message": (
                "Rendered ego lane differs from the lane its actor relations "
                "imply; ego is the placement anchor and cannot be repaired "
                "downstream."
            ),
            "expected_offset_from_right": expected_offset,
            "actual_offset_from_right": actual_offset,
            "actual_lane_id": actual_lane_id,
            "actual_road_id": road_id,
            "actual_driving_lane_count": len(sorted_lane_ids),
            "vlm_forward_lane_count": _scene_forward_lane_count(scene_understanding),
            "lateral_correction_m": round((actual_offset - expected_offset) * 3.5, 3),
        }

    @staticmethod
    def _nearest_dense_waypoint(location: dict, dense_wps: list) -> Optional[dict]:
        if not location or not dense_wps:
            return None
        try:
            x = float(location.get("x", 0.0))
            y = float(location.get("y", 0.0))
        except (TypeError, ValueError):
            return None
        return min(
            dense_wps,
            key=lambda wp: math.hypot(
                float(wp.get("x", 0.0)) - x,
                float(wp.get("y", 0.0)) - y,
            ),
        )

    def _offlane_spawn_issues(
        self,
        render_graph: dict,
        spawn_payload: dict,
        validation: dict,
    ) -> list:
        dense_wps = (self.carla_spawn_context or {}).get("dense_local_waypoints") or []
        if not dense_wps:
            return []
        lane_width = float((validation or {}).get("lane_width_m") or 3.5)
        threshold = max(2.2, lane_width * 0.65)
        entities_by_id = {
            str(entity.get("id")): entity
            for entity in (spawn_payload or {}).get("entities", [])
        }
        issues = []
        for actor in (render_graph or {}).get("actors", []):
            if not actor.get("spawned", True):
                continue
            entity_id = str(actor.get("id") or "")
            entity = entities_by_id.get(entity_id) or actor
            spawn_kind = str(entity.get("spawn_kind") or actor.get("spawn_kind") or "vehicle")
            if spawn_kind != "vehicle":
                continue
            nearest = self._nearest_dense_waypoint(actor.get("location") or {}, dense_wps)
            if nearest is None:
                continue
            location = actor.get("location") or {}
            dist = math.hypot(
                float(location.get("x", 0.0)) - float(nearest.get("x", 0.0)),
                float(location.get("y", 0.0)) - float(nearest.get("y", 0.0)),
            )
            if dist <= threshold:
                continue
            issues.append(
                {
                    "issue_type": "off_lane_spawn",
                    "target_artifact": "spawn_payload",
                    "operation": "snap_to_nearest_driving_lane",
                    "source_entity_id": entity_id,
                    "render_entity_id": entity_id,
                    "severity": "high",
                    "evidence": (
                        f"{entity_id} is {dist:.2f}m from the nearest driving lane "
                        f"center, above threshold {threshold:.2f}m"
                    ),
                    "nearest_waypoint": {
                        "x": float(nearest.get("x", 0.0)),
                        "y": float(nearest.get("y", 0.0)),
                        "z": float(nearest.get("z", 0.0)),
                        "yaw": float(nearest.get("yaw", 0.0)),
                        "road_id": nearest.get("road_id"),
                        "lane_id": nearest.get("lane_id"),
                    },
                    "distance_to_lane_center_m": round(dist, 3),
                }
            )
        return issues

    @staticmethod
    def _repair_action_from_extra_issue(issue: dict) -> Optional[dict]:
        issue_type = str(issue.get("issue_type") or "")
        entity_id = issue.get("source_entity_id") or issue.get("render_entity_id")
        if issue_type == "ego_lane_mismatch" and entity_id:
            return {
                "type": "ego_lane_mismatch",
                "entity_id": entity_id,
                "severity": issue.get("severity", "high"),
                "lateral_correction_m": issue.get("lateral_correction_m"),
                "evidence": issue.get("evidence", ""),
            }
        if issue_type == "off_lane_spawn" and entity_id:
            return {
                "type": "off_lane_spawn",
                "entity_id": entity_id,
                "severity": issue.get("severity", "high"),
                "nearest_waypoint": issue.get("nearest_waypoint") or {},
                "evidence": issue.get("evidence", ""),
            }
        return None

    def _augment_actor_graph_plan(
        self,
        scene_id: str,
        plan: dict,
        render_graph: dict,
        validation: dict,
        scene_understanding: dict,
    ) -> dict:
        augmented = deepcopy(plan or {})
        extra_issues = []
        ego_warning = self._ego_lane_anchor_warning(scene_id, scene_understanding)
        if ego_warning:
            extra_issues.append(
                {
                    "issue_type": "ego_lane_mismatch",
                    "target_artifact": "spawn_payload",
                    "operation": "shift_scene_to_expected_ego_lane",
                    "source_entity_id": "ego",
                    "render_entity_id": "ego",
                    "severity": "high",
                    "evidence": ego_warning.get("message", "Ego lane mismatch."),
                    "lateral_correction_m": ego_warning.get("lateral_correction_m"),
                    "ego_lane_warning": ego_warning,
                }
            )
        spawn_payload = self._load_json_if_exists(self._spawn_payload_path(scene_id))
        extra_issues.extend(
            self._offlane_spawn_issues(render_graph, spawn_payload, validation)
        )
        if not extra_issues:
            return augmented
        issues = list(augmented.get("issues") or [])
        issues.extend(extra_issues)
        repair_actions = list(augmented.get("repair_actions") or [])
        for issue in extra_issues:
            action = self._repair_action_from_extra_issue(issue)
            if action:
                repair_actions.append(action)
        augmented["issues"] = issues
        augmented["repair_actions"] = repair_actions
        augmented["passed"] = False
        augmented["status"] = "failed"
        base_score = float(augmented.get("score", 1.0) or 0.0)
        augmented["score"] = max(0.0, base_score - 0.12 * len(extra_issues))
        augmented["extra_verification_checks"] = {
            "ego_lane_checked": ego_warning is not None,
            "off_lane_issue_count": sum(
                1 for issue in extra_issues if issue.get("issue_type") == "off_lane_spawn"
            ),
        }
        return augmented

    def generate_final_scene_script(self, scene_id: str, match_report_path: str):
        print("Generating deterministic CARLA scene script.......")
        match_report = json.loads(read_file(match_report_path))
        carla_map = self._normalize_carla_world_name(
            match_report.get("world_name") or self.carla_map
        )
        script = self.structured_scenario_generator.build_existing_world_scene_script(
            spawn_payload_filename=os.path.basename(self._spawn_payload_path(scene_id)),
            carla_host=self.carla_host,
            carla_port=self.carla_port,
            carla_map=carla_map,
            scene_match_status=match_report.get("status"),
            scene_match_reason=match_report.get("reason"),
        )
        output_path = self._final_scene_path(scene_id)
        write_to_file(output_path, script)
        return output_path

    @staticmethod
    def _load_json_if_exists(path: str) -> dict:
        if not path or not os.path.exists(path):
            return {}
        return json.loads(read_file(path))

    def _capture_layout_images(
        self,
        scene_id: str,
        final_scene_path: str,
        round_index: int,
    ) -> dict:
        ego_view_path = self._ego_view_path(scene_id, round_index)
        bev_path = self._bev_path(scene_id, round_index)
        render_actor_graph_path = self._render_actor_graph_path(scene_id, round_index)
        if not self.require_carla_connection:
            return {
                "layout_image_path": None,
                "ego_view_path": None,
                "bev_path": None,
                "render_actor_graph_path": None,
                "capture_mode": None,
                "error": "CARLA connection disabled (require_carla_connection=False).",
            }
        if (self.carla_spawn_context or {}).get("status") == "unavailable":
            return {
                "layout_image_path": None,
                "ego_view_path": None,
                "bev_path": None,
                "render_actor_graph_path": None,
                "capture_mode": None,
                "error": (self.carla_spawn_context or {}).get(
                    "failure_reason",
                    "CARLA connection unavailable.",
                ),
            }
        env = os.environ.copy()
        env["AUTOSCENARIO_EGO_VIEW_OUTPUT"] = ego_view_path
        env["AUTOSCENARIO_BEV_OUTPUT"] = bev_path
        env["AUTOSCENARIO_RENDER_ACTOR_GRAPH_OUTPUT"] = render_actor_graph_path
        env.setdefault("AUTOSCENARIO_EGO_VIEW_SIZE", "1024")
        env.setdefault("AUTOSCENARIO_EGO_VIEW_FOV", "90")
        env.setdefault("AUTOSCENARIO_BEV_SIZE", "1024")
        env.setdefault("AUTOSCENARIO_BEV_HEIGHT", "80")
        for stale_path in (ego_view_path, bev_path, render_actor_graph_path):
            try:
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            except Exception:
                pass
        try:
            completed = subprocess.run(
                [sys.executable, final_scene_path],
                cwd=self.output_folder,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.verify_script_timeout,
            )
        except Exception as exc:
            return {
                "layout_image_path": None,
                "ego_view_path": None,
                "bev_path": None,
                "render_actor_graph_path": None,
                "capture_mode": None,
                "error": f"Failed to run final scene script for layout capture: {exc}",
            }
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            return {
                "layout_image_path": None,
                "ego_view_path": None,
                "bev_path": None,
                "render_actor_graph_path": None,
                "capture_mode": None,
                "error": f"Final scene script exited with {completed.returncode}: {stderr}",
            }

        if os.path.exists(ego_view_path):
            shutil.copyfile(ego_view_path, join(self.output_folder, "image.png"))
            return {
                "layout_image_path": ego_view_path,
                "ego_view_path": ego_view_path,
                "bev_path": bev_path if os.path.exists(bev_path) else None,
                "render_actor_graph_path": render_actor_graph_path if os.path.exists(render_actor_graph_path) else None,
                "capture_mode": "ego_view",
                "error": None,
            }
        if os.path.exists(bev_path):
            shutil.copyfile(bev_path, join(self.output_folder, "image.png"))
            return {
                "layout_image_path": bev_path,
                "ego_view_path": None,
                "bev_path": bev_path,
                "render_actor_graph_path": render_actor_graph_path if os.path.exists(render_actor_graph_path) else None,
                "capture_mode": "bev_fallback",
                "error": None,
            }
        return {
            "layout_image_path": None,
            "ego_view_path": None,
            "bev_path": None,
            "render_actor_graph_path": render_actor_graph_path if os.path.exists(render_actor_graph_path) else None,
            "capture_mode": None,
            "error": "Final scene script completed but did not produce ego-view or BEV image.",
        }

    def _capture_quick_bev_preview(
        self,
        scene_id: str,
        final_scene_path: str,
    ) -> dict:
        preview_path = self._quick_bev_path(scene_id)
        if not self.generate_quick_bev_preview:
            return {
                "enabled": False,
                "bev_path": None,
                "error": "Quick BEV preview disabled.",
            }
        if not self.require_carla_connection:
            return {
                "enabled": True,
                "bev_path": None,
                "error": "CARLA connection disabled (require_carla_connection=False).",
            }
        if (self.carla_spawn_context or {}).get("status") == "unavailable":
            return {
                "enabled": True,
                "bev_path": None,
                "error": (self.carla_spawn_context or {}).get(
                    "failure_reason",
                    "CARLA connection unavailable.",
                ),
            }

        try:
            if os.path.exists(preview_path):
                os.remove(preview_path)
        except Exception:
            pass

        env = os.environ.copy()
        env["AUTOSCENARIO_BEV_OUTPUT"] = preview_path
        env.setdefault("AUTOSCENARIO_BEV_SIZE", "1024")
        env.setdefault("AUTOSCENARIO_BEV_HEIGHT", "80")
        try:
            completed = subprocess.run(
                [sys.executable, final_scene_path],
                cwd=self.output_folder,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.verify_script_timeout,
            )
        except Exception as exc:
            return {
                "enabled": True,
                "bev_path": None,
                "error": f"Failed to run final scene script for quick BEV preview: {exc}",
            }

        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            return {
                "enabled": True,
                "bev_path": None,
                "error": f"Final scene script exited with {completed.returncode}: {stderr}",
            }
        if not os.path.exists(preview_path):
            return {
                "enabled": True,
                "bev_path": None,
                "error": "Final scene script completed but did not produce quick BEV preview.",
            }
        shutil.copyfile(preview_path, join(self.output_folder, "image.png"))
        return {
            "enabled": True,
            "bev_path": preview_path,
            "error": None,
        }

    def _write_verification_skipped(
        self,
        scene_id: str,
        round_index: int,
        reason: str,
    ) -> dict:
        report = {
            "passed": False,
            "score": 0.0,
            "hard_failures": [reason],
            "mismatches": [],
            "recommended_stage": "match_spawn",
            "repair_hints": [],
            "repair_actions": [],
            "skipped": True,
        }
        write_to_file(
            self._verification_path(scene_id, round_index),
            json.dumps(report, indent=2, sort_keys=True),
        )
        return report

    def _verify_scene_round(
        self,
        scene_id: str,
        round_index: int,
        image_path: str,
        layout_image_path: str,
        capture_mode: str,
        user_scene_description: str,
        scene_understanding: dict,
        match_report_path: str,
    ) -> dict:
        verification_path = self._verification_path(scene_id, round_index)
        match_report = self._load_json_if_exists(match_report_path)
        spawn_payload = self._load_json_if_exists(self._spawn_payload_path(scene_id))
        if self.verify_mode != "vlm":
            report = {
                "passed": False,
                "score": 0.0,
                "hard_failures": ["Only VLM verification mode is currently implemented."],
                "mismatches": [],
                "recommended_stage": "match_spawn",
                "repair_hints": [],
                "repair_actions": [],
            }
        else:
            report = self.scene_verification_agent.call_agent(
                {
                    "output_fn": verification_path,
                    "source_image_path": image_path,
                    "layout_image_path": layout_image_path,
                    "bev_image_path": layout_image_path,
                    "capture_mode": capture_mode,
                    "scene_understanding": scene_understanding,
                    "scene_match": match_report,
                    "spawn_entities": spawn_payload,
                    "user_scene_description": user_scene_description,
                }
            )
        report["capture_mode"] = capture_mode
        report["layout_image_path"] = layout_image_path
        report["passed"] = bool(report.get("passed")) and float(report.get("score", 0.0)) >= self.verify_min_score
        score = float(report.get("score", 0.0))
        passed = report.get("passed")
        stage = report.get("recommended_stage", "match_spawn")
        print(
            f"  Spawn layout verification score: {score:.2f} "
            f"({'PASS' if passed else 'FAIL'}, recommended_stage={stage})"
        )
        write_to_file(
            verification_path,
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        )
        return report

    def _write_source_actor_graph(
        self,
        scene_id: str,
        scene_understanding: dict,
        relation_dsl: dict,
    ) -> dict:
        graph = build_source_actor_graph(scene_understanding, relation_dsl)
        write_to_file(
            self._source_actor_graph_path(scene_id),
            json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False),
        )
        return graph

    def _load_or_build_render_actor_graph(
        self,
        scene_id: str,
        round_index: int,
    ) -> dict:
        render_graph_path = self._render_actor_graph_path(scene_id, round_index)
        graph = self._load_json_if_exists(render_graph_path)
        if graph:
            return graph
        spawn_payload = self._load_json_if_exists(self._spawn_payload_path(scene_id))
        graph = build_render_actor_graph_from_spawn_payload(spawn_payload)
        write_to_file(
            render_graph_path,
            json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False),
        )
        return graph

    def _write_actor_graph_repair_plan(
        self,
        scene_id: str,
        round_index: int,
        source_graph: dict,
        render_graph: dict,
        validation: dict,
        previous_plan=None,
    ) -> dict:
        plan = compare_actor_graphs(
            source_graph,
            render_graph,
            validation=validation,
            previous_plan=previous_plan,
        )
        write_to_file(
            self._actor_graph_repair_plan_path(scene_id, round_index),
            json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False),
        )
        return plan

    @staticmethod
    def _verification_report_from_actor_graph_plan(plan: dict) -> dict:
        issues = plan.get("issues") or []
        hard_failures = [
            str(issue.get("evidence") or issue.get("issue_type"))
            for issue in issues
            if str(issue.get("severity") or "") == "high"
        ]
        repair_actions = plan.get("repair_actions") or []
        semantic_types = {"count_mismatch", "category_mismatch", "missing_actor"}
        recommended_stage = "match_spawn"
        if any(str(issue.get("issue_type")) in semantic_types for issue in issues):
            recommended_stage = "scene_understanding"
        if plan.get("passed"):
            recommended_stage = "pass"
        return {
            "passed": bool(plan.get("passed")),
            "score": float(plan.get("score")) if plan.get("score") is not None else 0.0,
            "hard_failures": hard_failures,
            "mismatches": issues,
            "recommended_stage": recommended_stage,
            "repair_hints": [
                str(issue.get("evidence") or issue.get("issue_type"))
                for issue in issues
            ],
            "repair_actions": repair_actions,
            "actor_graph_plan": plan,
        }

    def _ensure_matched_world_loaded(self, match_report: dict) -> None:
        """Load the matched CARLA world if the server is on a different map.

        The large-map cache flow matches without loading any world, so the
        currently-loaded map may differ from the match. All subsequent live
        geometry must come from the matched map.
        """
        world_name = self._normalize_carla_world_name(
            match_report.get("world_name") or self.carla_map
        )
        if not world_name:
            return
        try:
            import carla

            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(max(self.carla_timeout, 60.0))
            current = self._normalize_carla_world_name(
                client.get_world().get_map().name
            )
            if current == world_name:
                return
            print(f"Loading matched world {world_name} (was {current}).......")
            client.load_world(world_name)
            self.carla_map = world_name
        except Exception as exc:
            print(f"Could not load matched world {world_name}: {exc}")

    def _ensure_matched_structure(
        self,
        scene_id: str,
        match_report_path: str,
        anchor_location_dict: dict,
        best_match: dict,
    ) -> dict:
        """Build the structural description (junction centre+legs / road segment)
        from the loaded matched world, for reference-frame reprojection.

        No-op when a structure is already present (live match path emits it).
        Recovers the candidate waypoint at the matched anchor and delegates to
        tools.map_structure.
        """
        if self.carla_spawn_context is None:
            self.carla_spawn_context = {}
        existing = self.carla_spawn_context.get("matched_structure")
        if isinstance(existing, dict) and not existing.get("error"):
            source = self.carla_spawn_context.get("matched_structure_source") or "match_report"
            self._persist_matched_structure(scene_id, match_report_path, existing, source)
            return existing
        try:
            import carla

            from tools.map_structure import build_matched_structure_from_waypoint

            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.carla_timeout)
            world_map = client.get_world().get_map()
            waypoint = world_map.get_waypoint(
                carla.Location(
                    float(anchor_location_dict.get("x", 0.0)),
                    float(anchor_location_dict.get("y", 0.0)),
                    float(anchor_location_dict.get("z", 0.0)),
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is None:
                structure = {
                    "error": "no_driving_waypoint_at_anchor",
                    "kind": None,
                }
                self.carla_spawn_context["matched_structure_source"] = "missing"
                self._persist_matched_structure(
                    scene_id, match_report_path, structure, "missing"
                )
                return structure
            ego_yaw = best_match.get("yaw")
            structure = build_matched_structure_from_waypoint(
                waypoint,
                ego_inbound_yaw=float(ego_yaw) if isinstance(ego_yaw, (int, float)) else None,
            )
            self.carla_spawn_context["matched_structure"] = structure
            self.carla_spawn_context["matched_structure_source"] = "live_carla"
            self._persist_matched_structure(
                scene_id, match_report_path, structure, "live_carla"
            )
            print(
                f"Built matched_structure ({structure.get('kind')}) from loaded "
                "world for reference-frame reprojection."
            )
            return structure
        except Exception as exc:
            print(f"matched_structure build skipped: {exc}")
            structure = {
                "error": str(exc),
                "kind": None,
            }
            self.carla_spawn_context["matched_structure_source"] = "missing"
            self._persist_matched_structure(
                scene_id, match_report_path, structure, "missing"
            )
            return structure

    def _sample_dense_local_waypoints(
        self,
        anchor_location_dict: dict,
        radius_m: float = 80.0,
        step_m: float = 2.0,
    ) -> None:
        """BFS-sample CARLA waypoints within radius_m of the matched anchor location.

        Called once after map matching.  Results are stored in
        spawn_context["dense_local_waypoints"] and used by
        project_entities_to_carla_context for higher-accuracy projection
        (correct z, curve-aware yaw/x/y) without repeated API calls.
        """
        try:
            import carla
        except ImportError:
            print("Dense waypoint sampling skipped: CARLA Python API not available.")
            return
        try:
            client = carla.Client(self.carla_host, self.carla_port)
            client.set_timeout(self.carla_timeout)
            world_map = client.get_world().get_map()
        except Exception as exc:
            print(f"Dense waypoint sampling skipped: {exc}")
            return

        anchor_loc = carla.Location(
            float(anchor_location_dict.get("x", 0.0)),
            float(anchor_location_dict.get("y", 0.0)),
            float(anchor_location_dict.get("z", 0.0)),
        )
        seed_wp = world_map.get_waypoint(
            anchor_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if seed_wp is None:
            print("Dense waypoint sampling skipped: no drivable road found at anchor.")
            return

        visited: set = set()
        dense: list = []
        queue: list = [seed_wp]
        while queue:
            wp = queue.pop(0)
            key = (wp.road_id, wp.lane_id, round(wp.s))
            if key in visited:
                continue
            visited.add(key)
            dense.append({
                "road_id": wp.road_id,
                "lane_id": wp.lane_id,
                "x": wp.transform.location.x,
                "y": wp.transform.location.y,
                "z": wp.transform.location.z,
                "yaw": wp.transform.rotation.yaw,
                "lane_width": wp.lane_width,
                "is_junction": wp.is_junction,
            })
            if anchor_loc.distance(wp.transform.location) > radius_m:
                continue
            for nwp in (wp.next(step_m) or []):
                queue.append(nwp)
            for pwp in (wp.previous(step_m) or []):
                queue.append(pwp)
            for adj in filter(None, [wp.get_left_lane(), wp.get_right_lane()]):
                if adj.lane_type == carla.LaneType.Driving:
                    queue.append(adj)

        if self.carla_spawn_context is None:
            self.carla_spawn_context = {}
        self.carla_spawn_context["dense_local_waypoints"] = dense
        print(
            f"Sampled {len(dense)} local waypoints "
            f"(radius={radius_m}m, step={step_m}m)."
        )

    def _rerun_structured_tail_no_remap(
        self,
        scene_id: str,
        scene_understanding: dict,
        match_report_path: str,
    ) -> tuple:
        """Re-run DSL → coordinates → spawn_payload without redoing map matching.

        The existing match_report_path (and its anchor) is reused so that
        entity positions change but the map location stays fixed.
        Returns (relation_dsl, refined_coordinates, validation, final_scene_path).
        """
        relation_dsl = self.generate_relation_dsl(scene_id, scene_understanding)
        raw_initial = self.generate_initial_coordinates(scene_id, relation_dsl)
        ordered = self.apply_pairwise_ordering_step(scene_id, raw_initial, relation_dsl)
        projected = self.project_entities_step(scene_id, ordered, scene_understanding)
        refined = self.refine_coordinates_step(scene_id, projected, relation_dsl)
        validation = self.validate_layout_step(scene_id, relation_dsl, refined)
        refined = self.reproject_junction_step(scene_id, refined)
        self.build_spawn_payload_from_match_or_fallback(scene_id, refined, match_report_path)
        final_scene_path = self.generate_final_scene_script(scene_id, match_report_path)
        return relation_dsl, refined, validation, final_scene_path

    @staticmethod
    def _layout_frame_from_validation(validation: dict, fallback_anchor: dict) -> tuple:
        anchor_lane = (
            (validation or {}).get("selected_anchor_lane")
            or (validation or {}).get("anchor_lane")
            or fallback_anchor
            or {}
        )
        start = anchor_lane.get("start") or {}
        end = anchor_lane.get("end") or {}
        ex = float(end.get("x", 1.0)) - float(start.get("x", 0.0))
        ey = float(end.get("y", 0.0)) - float(start.get("y", 0.0))
        length = math.sqrt(ex * ex + ey * ey) or 1.0
        fwd_x, fwd_y = ex / length, ey / length
        right_x, right_y = -fwd_y, fwd_x
        return anchor_lane, fwd_x, fwd_y, right_x, right_y

    @staticmethod
    def _repair_action_types(report: dict) -> tuple:
        actions = (report or {}).get("repair_actions") or []
        geometry_types = {
            "ego_lane_mismatch",
            "off_lane_spawn",
            "lane_side_mismatch",
            "pairwise_mismatch",
            "longitudinal_mismatch",
            "heading_mismatch",
            "overlap",
            "junction_lane_mismatch",
        }
        semantic_types = {"count_mismatch", "category_mismatch"}
        geometry = [a for a in actions if str(a.get("type")) in geometry_types]
        semantic = [a for a in actions if str(a.get("type")) in semantic_types]
        return geometry, semantic

    @staticmethod
    def _has_high_severity_semantic_action(actions: list) -> bool:
        semantic_types = {"count_mismatch", "category_mismatch"}
        return any(
            str(action.get("type") or "") in semantic_types
            and str(action.get("severity") or "").lower() == "high"
            for action in actions or []
        )

    @staticmethod
    def _validation_has_repairable_failures(validation: dict) -> bool:
        for entity in (validation or {}).get("entity_results") or []:
            if entity.get("status") == "fail" and entity.get("lateral_check") is False:
                return True
        for pair in (validation or {}).get("pairwise_results") or []:
            if pair.get("status") == "fail":
                return True
        return False

    def _apply_layout_repair_actions(
        self,
        scene_id: str,
        current_validation: dict,
        report: dict,
    ) -> dict:
        """Apply deterministic lane-side and pairwise repairs to spawn payload."""
        spawn_payload_path = self._spawn_payload_path(scene_id)
        spawn_payload = self._load_json_if_exists(spawn_payload_path)
        entities = spawn_payload.get("entities") or []
        entities_by_id = {str(e.get("id")): e for e in entities}

        fallback_anchor = ((self.carla_spawn_context or {}).get("topology_sample") or [{}])[0]
        _, fwd_x, fwd_y, right_x, right_y = self._layout_frame_from_validation(
            current_validation,
            fallback_anchor,
        )
        report = report or {"repair_actions": []}
        actions = report.get("repair_actions") or []
        action_types = {str(action.get("type")) for action in actions}
        applied_repairs = []
        blocked_repairs = []
        blocked_entity_ids = set()
        max_lateral_correction_m = 3.5

        def block_lateral_repair(entity_id: str, correction: float, source: str) -> None:
            blocked_entity_ids.add(str(entity_id))
            blocked_repairs.append({
                "type": "blocked_for_code_fix",
                "reason": "lateral_correction_exceeds_limit",
                "source": source,
                "entity_id": entity_id,
                "lateral_correction_m": round(correction, 3),
                "max_lateral_correction_m": max_lateral_correction_m,
            })

        lane_side_action_ids = {
            str(action.get("entity_id"))
            for action in actions
            if str(action.get("type") or "") == "lane_side_mismatch"
        }

        for action in actions:
            if str(action.get("type") or "") != "ego_lane_mismatch":
                continue
            try:
                correction = float(action.get("lateral_correction_m"))
            except (TypeError, ValueError):
                continue
            if abs(correction) <= 0.1:
                continue
            if abs(correction) > max_lateral_correction_m * 2:
                block_lateral_repair("ego", correction, "ego_lane_mismatch")
                continue
            for entity in entities:
                loc = entity.setdefault("location", {})
                loc["x"] = float(loc.get("x", 0.0)) + right_x * correction
                loc["y"] = float(loc.get("y", 0.0)) + right_y * correction
            applied_repairs.append({
                "type": "ego_lane_scene_shift",
                "entity_id": "ego",
                "lateral_correction_m": round(correction, 3),
            })

        # Lane-side repair from validation is deterministic and does not rely on VLM text.
        for entity_result in (current_validation or {}).get("entity_results") or []:
            entity_id = str(entity_result.get("entity_id") or "")
            should_repair = (
                entity_result.get("status") == "fail"
                and entity_result.get("lateral_check") is False
            ) or entity_id in lane_side_action_ids
            if not should_repair:
                continue
            entity = entities_by_id.get(entity_id)
            if entity is None:
                continue
            try:
                actual = float(entity_result.get("actual_lateral_m"))
                expected = float(entity_result.get("expected_lateral_m"))
            except (TypeError, ValueError):
                continue
            correction = expected - actual
            if abs(correction) <= 0.1:
                continue
            if abs(correction) > max_lateral_correction_m:
                block_lateral_repair(entity_id, correction, "lane_side_validation")
                continue
            loc = entity.setdefault("location", {})
            loc["x"] = float(loc.get("x", 0.0)) + right_x * correction
            loc["y"] = float(loc.get("y", 0.0)) + right_y * correction
            entity.setdefault("repair_metadata", {})["applied_repair"] = "lane_index_lateral_offset"
            entity["repair_metadata"]["old_lateral_m"] = round(actual, 3)
            entity["repair_metadata"]["new_lateral_m"] = round(expected, 3)
            entity["repair_metadata"]["lateral_correction_m"] = round(correction, 3)
            applied_repairs.append({
                "type": "lane_side_validation",
                "entity_id": entity.get("id"),
                "lateral_correction_m": round(correction, 3),
            })

        pairwise_corrections = {}
        for pair in ((current_validation or {}).get("pairwise_results") or []):
            if str(pair.get("status")) != "fail":
                continue
            lon_delta = float(pair.get("actual_longitudinal_delta_m") or 0.0)
            lat_delta = float(pair.get("actual_lateral_delta_m") or 0.0)
            lon_rel = str(pair.get("longitudinal_relation") or "")
            lat_rel = str(pair.get("lateral_relation") or "")
            eid = str(pair.get("entity_id") or "")
            entity = entities_by_id.get(eid)
            if entity is None:
                continue

            pending = pairwise_corrections.setdefault(
                eid, {"longitudinal": [], "lateral": []}
            )
            if lon_rel == "ahead_of_other" and lon_delta < 0:
                pending["longitudinal"].append(-lon_delta + 0.5)
            elif lon_rel == "behind_other" and lon_delta > 0:
                pending["longitudinal"].append(-(lon_delta + 0.5))
            lateral_correction = 0.0
            if lat_rel == "left_of_other" and lat_delta >= -1.0:
                lateral_correction = -1.5 - lat_delta
            elif lat_rel == "right_of_other" and lat_delta <= 1.0:
                lateral_correction = 1.5 - lat_delta
            elif lat_rel == "same_lateral_band" and abs(lat_delta) > 1.0:
                lateral_correction = -lat_delta * 0.5
            if abs(lateral_correction) > 0.1:
                pending["lateral"].append(lateral_correction)

        for eid, pending in pairwise_corrections.items():
            entity = entities_by_id.get(eid)
            if entity is None:
                continue
            loc = entity.setdefault("location", {})
            lon_values = pending.get("longitudinal") or []
            if lon_values:
                longitudinal_correction = sum(lon_values) / len(lon_values)
                if abs(longitudinal_correction) > 0.1:
                    loc["x"] = float(loc.get("x", 0.0)) + fwd_x * longitudinal_correction
                    loc["y"] = float(loc.get("y", 0.0)) + fwd_y * longitudinal_correction
                    applied_repairs.append({
                        "type": "pairwise_longitudinal",
                        "entity_id": eid,
                        "longitudinal_correction_m": round(longitudinal_correction, 3),
                        "source_count": len(lon_values),
                    })
            lat_values = pending.get("lateral") or []
            if lat_values:
                lateral_correction = sum(lat_values) / len(lat_values)
                if abs(lateral_correction) <= 0.1:
                    continue
                if abs(lateral_correction) > max_lateral_correction_m:
                    block_lateral_repair(eid, lateral_correction, "pairwise_validation")
                    continue
                loc["x"] = float(loc.get("x", 0.0)) + right_x * lateral_correction
                loc["y"] = float(loc.get("y", 0.0)) + right_y * lateral_correction
                applied_repairs.append({
                    "type": "pairwise_lateral",
                    "entity_id": eid,
                    "lateral_correction_m": round(lateral_correction, 3),
                    "source_count": len(lat_values),
                })

        for action in actions:
            action_type = str(action.get("type") or "")
            entity = entities_by_id.get(str(action.get("entity_id") or ""))
            if entity is None:
                continue
            if str(entity.get("id") or "") in blocked_entity_ids:
                continue
            loc = entity.setdefault("location", {})
            if action_type == "pairwise_mismatch":
                other = entities_by_id.get(str(action.get("reference_entity_id") or ""))
                if other is None:
                    continue
                other_loc = other.get("location") or {}
                target = str(action.get("target_relation") or "")
                spacing = max(self._spawn_min_spacing(entity), self._spawn_min_spacing(other)) + 0.5
                if target in {"ahead_of", "ahead_of_other"}:
                    loc["x"] = float(other_loc.get("x", 0.0)) + fwd_x * spacing
                    loc["y"] = float(other_loc.get("y", 0.0)) + fwd_y * spacing
                elif target in {"behind_of", "behind_other", "behind_of_other"}:
                    loc["x"] = float(other_loc.get("x", 0.0)) - fwd_x * spacing
                    loc["y"] = float(other_loc.get("y", 0.0)) - fwd_y * spacing
                elif target in {"left_of", "left_of_other"}:
                    loc["x"] = float(other_loc.get("x", 0.0)) - right_x * spacing
                    loc["y"] = float(other_loc.get("y", 0.0)) - right_y * spacing
                elif target in {"right_of", "right_of_other"}:
                    loc["x"] = float(other_loc.get("x", 0.0)) + right_x * spacing
                    loc["y"] = float(other_loc.get("y", 0.0)) + right_y * spacing
                else:
                    continue
                applied_repairs.append({"type": "pairwise_action", "entity_id": entity.get("id")})
            elif action_type == "overlap":
                other = entities_by_id.get(str(action.get("reference_entity_id") or ""))
                try:
                    min_spacing = float(action.get("min_spacing_m"))
                except (TypeError, ValueError):
                    min_spacing = self._spawn_min_spacing(entity)
                spacing = max(min_spacing, self._spawn_min_spacing(entity)) + 0.5
                if other is not None:
                    other_loc = other.get("location") or {}
                    loc["x"] = float(other_loc.get("x", 0.0)) + fwd_x * spacing
                    loc["y"] = float(other_loc.get("y", 0.0)) + fwd_y * spacing
                else:
                    loc["x"] = float(loc.get("x", 0.0)) + fwd_x * spacing
                    loc["y"] = float(loc.get("y", 0.0)) + fwd_y * spacing
                applied_repairs.append({
                    "type": "overlap_action",
                    "entity_id": entity.get("id"),
                    "reference_entity_id": action.get("reference_entity_id"),
                    "spacing_m": round(spacing, 3),
                })
            elif action_type == "longitudinal_mismatch":
                try:
                    delta = float(action.get("delta_m", 3.0) or 3.0)
                except (TypeError, ValueError):
                    delta = 3.0
                direction = str(action.get("direction") or "")
                sign = 1.0
                if "->" in direction:
                    source_band, render_band = [
                        part.strip() for part in direction.split("->", 1)
                    ]
                    band_order = {"near": 0, "mid": 1, "far": 2}
                    source_rank = band_order.get(source_band)
                    render_rank = band_order.get(render_band)
                    if source_rank is not None and render_rank is not None:
                        sign = -1.0 if render_rank > source_rank else 1.0
                elif "behind" in direction:
                    sign = -1.0
                loc["x"] = float(loc.get("x", 0.0)) + fwd_x * delta * sign
                loc["y"] = float(loc.get("y", 0.0)) + fwd_y * delta * sign
                applied_repairs.append({
                    "type": "longitudinal_action",
                    "entity_id": entity.get("id"),
                    "delta_m": round(delta * sign, 3),
                })
            elif action_type == "off_lane_spawn":
                nearest = action.get("nearest_waypoint") or {}
                if not nearest:
                    continue
                loc["x"] = float(nearest.get("x", loc.get("x", 0.0)))
                loc["y"] = float(nearest.get("y", loc.get("y", 0.0)))
                loc["z"] = float(nearest.get("z", loc.get("z", 0.0)))
                rotation = entity.setdefault("rotation", {})
                rotation["yaw"] = float(nearest.get("yaw", rotation.get("yaw", 0.0)))
                entity["placement_mode"] = "project_to_lane"
                entity.setdefault("repair_metadata", {})["applied_repair"] = "snap_to_driving_lane"
                applied_repairs.append({
                    "type": "off_lane_snap",
                    "entity_id": entity.get("id"),
                    "road_id": nearest.get("road_id"),
                    "lane_id": nearest.get("lane_id"),
                })
            elif action_type == "junction_lane_mismatch":
                entity["placement_mode"] = "preserve_xy"
                entity.setdefault("repair_metadata", {})["applied_repair"] = (
                    "preserve_junction_seed_after_lane_mismatch"
                )
                entity["repair_metadata"]["expected_lane"] = action.get("expected_lane") or {}
                entity["repair_metadata"]["actual_waypoint"] = action.get("actual_waypoint") or {}
                applied_repairs.append({
                    "type": "junction_lane_action",
                    "entity_id": entity.get("id"),
                    "new_placement_mode": "preserve_xy",
                })
            elif action_type == "heading_mismatch":
                rotation = entity.setdefault("rotation", {})
                try:
                    yaw = float(rotation.get("yaw", 0.0))
                except Exception:
                    yaw = 0.0
                rotation["yaw"] = self._normalize_yaw(yaw + 180.0)
                applied_repairs.append({"type": "heading_action", "entity_id": entity.get("id")})

        spawn_payload["entities"] = list(entities_by_id.values())
        spawn_payload.setdefault("repair_metadata", {})
        spawn_payload["repair_metadata"]["last_applied_repairs"] = applied_repairs
        spawn_payload["repair_metadata"]["blocked_repairs"] = blocked_repairs
        spawn_payload["repair_metadata"]["blocked_for_code_fix"] = bool(blocked_repairs)
        spawn_payload["repair_metadata"]["action_types"] = sorted(action_types)
        self._post_repair_spawn_payload_layout(spawn_payload)
        write_to_file(
            spawn_payload_path,
            json.dumps(spawn_payload, indent=2, sort_keys=True),
        )
        return spawn_payload

    @staticmethod
    def _angle_distance(yaw_a: float, yaw_b: float) -> float:
        delta = (float(yaw_a) - float(yaw_b) + 180.0) % 360.0 - 180.0
        return abs(delta)

    @staticmethod
    def _normalize_yaw(yaw: float) -> float:
        while yaw <= -180.0:
            yaw += 360.0
        while yaw > 180.0:
            yaw -= 360.0
        return yaw

    @staticmethod
    def _spawn_min_spacing(entity: dict) -> float:
        category = str(entity.get("category") or "")
        if AutoGenerator._is_parking_spawn_entity(entity):
            if category in {"truck", "bus"}:
                return 5.0
            if category in {"motorcycle", "bicycle"}:
                return 2.0
            return 3.5
        if category in {"truck", "bus"}:
            return 7.0
        if category == "car":
            return 5.0
        if category in {"motorcycle", "bicycle"}:
            return 3.0
        if category == "pedestrian":
            return 0.8
        return 2.5

    @staticmethod
    def _is_parking_spawn_entity(entity: dict) -> bool:
        lane_side = str(entity.get("lane_side_relation") or "")
        placement_hint = str(entity.get("placement_mode_hint") or "")
        placement_mode = str(entity.get("placement_mode") or "")
        actor_group_type = str(entity.get("actor_group_type") or "")
        motion_state = str(entity.get("motion_state") or "")
        return (
            "parking" in lane_side
            or placement_hint == "parking_lane_actor"
            or placement_mode == "project_to_parking_lane"
            or actor_group_type == "parking_row"
            or (motion_state == "parked" and lane_side in {"left_edge", "right_edge"})
        )

    def _post_repair_spawn_payload_layout(self, spawn_payload: dict) -> None:
        entities = spawn_payload.get("entities") or []
        ego = next(
            (
                entity
                for entity in entities
                if str(entity.get("id")) in {"ego", "ego_vehicle"}
            ),
            None,
        )
        ego_yaw = None
        if ego is not None:
            try:
                ego_yaw = float((ego.get("rotation") or {}).get("yaw"))
            except Exception:
                ego_yaw = None

        if ego_yaw is not None:
            for entity in entities:
                rotation = entity.get("rotation") or {}
                try:
                    yaw = float(rotation.get("yaw", 0.0))
                except Exception:
                    continue
                heading_relation = str(entity.get("heading_relation") or "unknown")
                if (
                    heading_relation == "opposite_direction"
                    and self._angle_distance(yaw, ego_yaw) < 90.0
                ):
                    rotation["yaw"] = self._normalize_yaw(yaw + 180.0)
                elif (
                    heading_relation == "same_direction"
                    and self._angle_distance(yaw, ego_yaw) > 90.0
                ):
                    rotation["yaw"] = self._normalize_yaw(yaw + 180.0)
                entity["rotation"] = rotation

        anchor_lane = (
            (self.carla_spawn_context or {}).get("topology_sample") or [{}]
        )[0]
        start = anchor_lane.get("start") or {}
        end = anchor_lane.get("end") or {}
        ex = float(end.get("x", 1.0)) - float(start.get("x", 0.0))
        ey = float(end.get("y", 0.0)) - float(start.get("y", 0.0))
        length = math.sqrt(ex * ex + ey * ey) or 1.0
        fwd_x, fwd_y = ex / length, ey / length

        lane_index_entities = [
            entity
            for entity in entities
            if str(entity.get("spawn_kind") or "vehicle") == "vehicle"
            and int(float(entity.get("lane_index_relation", 0) or 0)) != 0
        ]
        for index, entity in enumerate(lane_index_entities):
            loc = entity.get("location") or {}
            for other in lane_index_entities[:index]:
                if int(float(entity.get("lane_index_relation", 0) or 0)) != int(
                    float(other.get("lane_index_relation", 0) or 0)
                ):
                    continue
                other_loc = other.get("location") or {}
                dx = float(loc.get("x", 0.0)) - float(other_loc.get("x", 0.0))
                dy = float(loc.get("y", 0.0)) - float(other_loc.get("y", 0.0))
                distance = math.sqrt(dx * dx + dy * dy)
                threshold = max(
                    self._spawn_min_spacing(entity),
                    self._spawn_min_spacing(other),
                )
                if distance >= threshold:
                    continue
                shift = threshold - distance + 0.2
                loc["x"] = float(loc.get("x", 0.0)) + fwd_x * shift
                loc["y"] = float(loc.get("y", 0.0)) + fwd_y * shift
            entity["location"] = loc

    def _spawn_layout_repair_summary_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_spawn_repair.json")

    def verify_and_repair_spawn_layout(
        self,
        scene_id: str,
        image_path: str,
        user_scene_description: str,
        scene_understanding: dict,
        relation_dsl: dict,
        validation: dict,
        match_report_path: str,
        final_scene_path: str,
    ) -> dict:
        """Verify spawned actor layout and repair with v3 deterministic priority."""
        if not self.enable_scene_verify:
            return {"enabled": False, "rounds": []}

        matched_structure = (self.carla_spawn_context or {}).get("matched_structure")
        matched_structure_source = (
            (self.carla_spawn_context or {}).get("matched_structure_source")
            or "missing"
        )
        repair_summary: dict = {
            "enabled": True,
            "rounds": [],
            "matched_structure_summary": self._matched_structure_summary(
                matched_structure,
                matched_structure_source,
            ),
            "matched_structure_path": self._matched_structure_path(scene_id),
        }
        ego_lane_warning = self._ego_lane_anchor_warning(scene_id, scene_understanding)
        if ego_lane_warning:
            repair_summary["ego_lane_anchor_warning"] = ego_lane_warning
            print(
                "  [ego_lane_warning] expected offset "
                f"{ego_lane_warning['expected_offset_from_right']} != actual "
                f"{ego_lane_warning['actual_offset_from_right']} "
                f"(lane_id={ego_lane_warning['actual_lane_id']}, "
                f"vlm_forward={ego_lane_warning['vlm_forward_lane_count']}, "
                f"actual_lanes={ego_lane_warning['actual_driving_lane_count']})"
            )
        current_scene_understanding = scene_understanding
        current_relation_dsl = relation_dsl
        current_validation = validation
        current_final_path = final_scene_path
        previous_actor_graph_plan = None
        previous_score = None
        last_good_spawn_payload_text = None

        for round_index in range(1, max(1, self.verify_max_rounds) + 1):
            print(f"Spawn layout verify-repair round {round_index}.......")
            capture = self._capture_layout_images(
                scene_id, current_final_path, round_index
            )
            source_graph = self._write_source_actor_graph(
                scene_id,
                current_scene_understanding,
                current_relation_dsl,
            )
            render_graph = self._load_or_build_render_actor_graph(scene_id, round_index)
            actor_graph_plan = self._write_actor_graph_repair_plan(
                scene_id,
                round_index,
                source_graph,
                render_graph,
                current_validation,
                previous_plan=previous_actor_graph_plan,
            )
            actor_graph_plan = self._augment_actor_graph_plan(
                scene_id,
                actor_graph_plan,
                render_graph,
                current_validation,
                current_scene_understanding,
            )
            write_to_file(
                self._actor_graph_repair_plan_path(scene_id, round_index),
                json.dumps(actor_graph_plan, indent=2, sort_keys=True, ensure_ascii=False),
            )
            if capture.get("error"):
                report = self._verification_report_from_actor_graph_plan(actor_graph_plan)
                report["capture_mode"] = capture.get("capture_mode")
                report["layout_image_path"] = capture.get("layout_image_path")
                report["hard_failures"] = list(report.get("hard_failures") or [])
                report["hard_failures"].append(str(capture.get("error")))
                geometry_actions, semantic_actions = self._repair_action_types(report)
                actionable_without_capture = bool(geometry_actions) and not semantic_actions
                if actionable_without_capture:
                    print(
                        "Layout capture failed, but static actor-graph repair_actions "
                        "are available; applying deterministic layout repair."
                    )
                    last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                    previous_score = float(report.get("score", 0.0) or 0.0)
                    self._apply_layout_repair_actions(scene_id, current_validation, report)
                    current_final_path = self.generate_final_scene_script(
                        scene_id, match_report_path
                    )
                    repair_summary["rounds"].append({
                        "round": round_index,
                        "verification": report,
                        "repair": "deterministic_layout_repair",
                        "repair_actions": report.get("repair_actions") or [],
                        "capture_mode": capture.get("capture_mode"),
                        "layout_image_path": capture.get("layout_image_path"),
                        "ego_view_path": capture.get("ego_view_path"),
                        "bev_path": capture.get("bev_path"),
                        "render_actor_graph_path": self._render_actor_graph_path(scene_id, round_index),
                        "actor_graph_repair_plan_path": self._actor_graph_repair_plan_path(scene_id, round_index),
                        "actor_graph_repair_plan": actor_graph_plan,
                    })
                    if round_index < max(1, self.verify_max_rounds):
                        previous_actor_graph_plan = actor_graph_plan
                        continue
                    break
                report = self._write_verification_skipped(
                    scene_id, round_index, str(capture.get("error"))
                )
                repair_summary["rounds"].append({
                    "round": round_index,
                    "verification": report,
                    "repair": "skipped",
                    "capture_mode": capture.get("capture_mode"),
                    "layout_image_path": capture.get("layout_image_path"),
                    "ego_view_path": capture.get("ego_view_path"),
                    "bev_path": capture.get("bev_path"),
                    "render_actor_graph_path": self._render_actor_graph_path(scene_id, round_index),
                    "actor_graph_repair_plan_path": self._actor_graph_repair_plan_path(scene_id, round_index),
                    "actor_graph_repair_plan": actor_graph_plan,
                })
                break

            actor_graph_blocked = (
                bool(actor_graph_plan.get("stop_repair_loop"))
                or bool(actor_graph_plan.get("requires_code_fix"))
            )
            if actor_graph_blocked:
                report = self._verification_report_from_actor_graph_plan(actor_graph_plan)
                report["capture_mode"] = capture.get("capture_mode")
                report["layout_image_path"] = capture.get("layout_image_path")
                write_to_file(
                    self._verification_path(scene_id, round_index),
                    json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
                )
                print("Actor graph verification detected a systematic issue; stopping repair loop.")
            elif self.verify_mode == "actor_graph":
                report = self._verification_report_from_actor_graph_plan(actor_graph_plan)
                report["capture_mode"] = capture.get("capture_mode")
                report["layout_image_path"] = capture.get("layout_image_path")
                write_to_file(
                    self._verification_path(scene_id, round_index),
                    json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
                )
                score = float(report.get("score", 0.0))
                print(
                    f"  Actor graph verification score: {score:.2f} "
                    f"({'PASS' if report.get('passed') else 'FAIL'}, "
                    f"status={actor_graph_plan.get('status')})"
                )
            else:
                report = self._verify_scene_round(
                    scene_id,
                    round_index,
                    image_path,
                    capture.get("layout_image_path"),
                    capture.get("capture_mode"),
                    user_scene_description,
                    current_scene_understanding,
                    match_report_path,
                )
            round_summary: dict = {
                "round": round_index,
                "layout_image_path": capture.get("layout_image_path"),
                "ego_view_path": capture.get("ego_view_path"),
                "bev_path": capture.get("bev_path"),
                "render_actor_graph_path": self._render_actor_graph_path(scene_id, round_index),
                "source_actor_graph_path": self._source_actor_graph_path(scene_id),
                "actor_graph_repair_plan_path": self._actor_graph_repair_plan_path(scene_id, round_index),
                "actor_graph_repair_plan": actor_graph_plan,
                "capture_mode": capture.get("capture_mode"),
                "verification": report,
            }

            score = float(report.get("score", 0.0) or 0.0)
            if (
                round_index > 1
                and previous_score is not None
                and score < float(previous_score) - 1e-9
                and last_good_spawn_payload_text
            ):
                print(
                    "Repair regression detected "
                    f"({score:.2f} < {float(previous_score):.2f}); reverting spawn payload."
                )
                write_to_file(self._spawn_payload_path(scene_id), last_good_spawn_payload_text)
                current_final_path = self.generate_final_scene_script(
                    scene_id, match_report_path
                )
                round_summary["repair"] = "reverted_regression"
                round_summary["reverted_to_previous_score"] = float(previous_score)
                repair_summary["rounds"].append(round_summary)
                repair_summary["regression_reverted"] = True
                break

            if actor_graph_blocked:
                round_summary["repair"] = "blocked_for_code_fix"
                repair_summary["rounds"].append(round_summary)
                break

            if report.get("passed"):
                round_summary["repair"] = "none"
                repair_summary["rounds"].append(round_summary)
                break

            geometry_actions, semantic_actions = self._repair_action_types(report)
            recommended_stage = str(report.get("recommended_stage") or "match_spawn")
            validation_repairable = self._validation_has_repairable_failures(current_validation)

            if self.verify_mode == "actor_graph" and actor_graph_plan.get("stop_repair_loop"):
                round_summary["repair"] = "blocked_for_code_fix"
                repair_summary["rounds"].append(round_summary)
                print("Actor graph verification requested repair-loop stop.")
                break

            if (
                self.verify_mode == "actor_graph"
                and actor_graph_plan.get("truth_unavailable")
                and not geometry_actions
                and not validation_repairable
            ):
                round_summary["repair"] = "truth_unavailable_static_check_only"
                repair_summary["rounds"].append(round_summary)
                print("Actor graph verification has no CARLA truth; stopping after static checks.")
                break

            if round_index >= max(1, self.verify_max_rounds) and (
                geometry_actions or validation_repairable
            ):
                print(
                    "Applying final deterministic layout repair before leaving "
                    f"verify-repair at max rounds ({self.verify_max_rounds})."
                )
                last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                previous_score = score
                self._apply_layout_repair_actions(
                    scene_id,
                    current_validation,
                    report if geometry_actions else {"repair_actions": []},
                )
                current_final_path = self.generate_final_scene_script(
                    scene_id, match_report_path
                )
                round_summary["repair"] = (
                    "deterministic_layout_repair"
                    if geometry_actions
                    else "validation_deterministic_layout_repair"
                )
                round_summary["repair_actions"] = report.get("repair_actions") or []
                round_summary["deterministic_fallback"] = not bool(geometry_actions)
                repair_summary["rounds"].append(round_summary)
                break

            if round_index >= max(1, self.verify_max_rounds):
                round_summary["repair"] = "max_rounds_reached"
                repair_summary["rounds"].append(round_summary)
                print(
                    f"Spawn layout verify-repair reached max rounds ({self.verify_max_rounds}). "
                    "Continuing with last result."
                )
                break

            scene_understanding_first = (
                recommended_stage == "scene_understanding"
                or self._has_high_severity_semantic_action(semantic_actions)
            )
            round_summary["repair_actions"] = report.get("repair_actions") or []
            round_summary["deterministic_fallback"] = False

            if scene_understanding_first:
                print("Repairing scene_understanding before layout repair.......")
                last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                previous_score = score
                output_fn = self._scene_understanding_path(scene_id)
                revised = self.scene_understanding_interpreter.revise_with_verification_feedback(
                    current_scene_understanding,
                    report,
                    user_scene_description,
                    output_fn,
                )
                current_scene_understanding = revised
                (
                    current_relation_dsl,
                    _refined,
                    current_validation,
                    current_final_path,
                ) = self._rerun_structured_tail_no_remap(
                    scene_id, current_scene_understanding, match_report_path
                )
                round_summary["repair"] = "scene_understanding_revision"

            elif geometry_actions:
                print("Applying deterministic repair_actions to spawn payload.......")
                last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                previous_score = score
                self._apply_layout_repair_actions(scene_id, current_validation, report)
                current_final_path = self.generate_final_scene_script(
                    scene_id, match_report_path
                )
                round_summary["repair"] = "deterministic_layout_repair"

            elif semantic_actions:
                print("Repairing scene_understanding based on VLM feedback.......")
                last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                previous_score = score
                output_fn = self._scene_understanding_path(scene_id)
                revised = self.scene_understanding_interpreter.revise_with_verification_feedback(
                    current_scene_understanding,
                    report,
                    user_scene_description,
                    output_fn,
                )
                current_scene_understanding = revised
                (
                    current_relation_dsl,
                    _refined,
                    current_validation,
                    current_final_path,
                ) = self._rerun_structured_tail_no_remap(
                    scene_id, current_scene_understanding, match_report_path
                )
                round_summary["repair"] = "scene_understanding_revision"

            elif validation_repairable:
                print("Applying validation-driven deterministic layout repair.......")
                last_good_spawn_payload_text = read_file(self._spawn_payload_path(scene_id))
                previous_score = score
                self._apply_layout_repair_actions(
                    scene_id,
                    current_validation,
                    {"repair_actions": []},
                )
                current_final_path = self.generate_final_scene_script(
                    scene_id, match_report_path
                )
                round_summary["repair"] = "validation_deterministic_layout_repair"
                round_summary["deterministic_fallback"] = True

            else:
                round_summary["repair"] = "no_repair_action"
                repair_summary["rounds"].append(round_summary)
                print("Spawn layout verification failed but produced no actionable repair.")
                break

            repair_summary["rounds"].append(round_summary)
            previous_actor_graph_plan = actor_graph_plan

        write_to_file(
            self._spawn_layout_repair_summary_path(scene_id),
            json.dumps(repair_summary, indent=2, sort_keys=True, ensure_ascii=False),
        )
        return repair_summary

    def _build_scenario_for_match(
        self,
        scene_id: str,
        match_report_path: str,
        scene_understanding: dict,
        image_path: str,
        user_scene_description: str,
    ) -> None:
        """Run the full downstream pipeline for one matched map region.

        Assumes self.carla_spawn_context already has this candidate's
        topology_sample prepended (via _apply_match_to_spawn_context).
        """
        match_report = self._load_json_if_exists(match_report_path)
        best_match = match_report.get("best_match") or {}
        anchor_loc = best_match.get("location") or {}
        if self.require_carla_connection:
            # Cache-based matching never loads the matched world, so the matched
            # map must be loaded here before any live geometry (dense waypoints,
            # structural reprojection, spawn verification) is read from it.
            self._ensure_matched_world_loaded(match_report)
        if anchor_loc and self.require_carla_connection:
            self._sample_dense_local_waypoints(anchor_loc)
            self._ensure_matched_structure(
                scene_id,
                match_report_path,
                anchor_loc,
                best_match,
            )
        self._index_ego_on_candidate_lane(scene_understanding)
        if self._scene_requires_junction_structure(scene_understanding):
            self._require_junction_structure("Junction scene")
        matched_structure = (self.carla_spawn_context or {}).get("matched_structure")
        if isinstance(matched_structure, dict):
            self._persist_matched_structure(
                scene_id,
                match_report_path,
                matched_structure,
                (self.carla_spawn_context or {}).get("matched_structure_source")
                or "unknown",
            )

        relation_dsl = self.generate_relation_dsl(scene_id, scene_understanding)
        raw_initial = self.generate_initial_coordinates(scene_id, relation_dsl)
        ordered = self.apply_pairwise_ordering_step(scene_id, raw_initial, relation_dsl)
        projected = self.project_entities_step(scene_id, ordered, scene_understanding)
        refined = self.refine_coordinates_step(scene_id, projected, relation_dsl)
        validation = self.validate_layout_step(scene_id, relation_dsl, refined)
        refined = self.reproject_junction_step(scene_id, refined)
        self.build_spawn_payload_from_match_or_fallback(scene_id, refined, match_report_path)
        final_scene_path = self.generate_final_scene_script(scene_id, match_report_path)
        repair_summary = None
        if self.enable_scene_verify:
            repair_summary = self.verify_and_repair_spawn_layout(
                scene_id,
                image_path,
                user_scene_description,
                scene_understanding,
                relation_dsl,
                validation,
                match_report_path,
                final_scene_path,
            )
        else:
            print("Spawn layout verify-repair skipped.")
        quick_bev = self._capture_quick_bev_preview(scene_id, final_scene_path)
        print(f"  Scene match report: {match_report_path}")
        print(f"  Spawn script:       {final_scene_path}")
        if self.enable_scene_verify:
            print(f"  Repair summary:     {self._spawn_layout_repair_summary_path(scene_id)}")
        if quick_bev.get("bev_path"):
            print(f"  Quick BEV preview:  {quick_bev['bev_path']}")
        elif quick_bev.get("error"):
            print(f"  Quick BEV preview:  skipped ({quick_bev['error']})")
        if repair_summary is not None:
            repair_summary["quick_bev_preview"] = quick_bev
            write_to_file(
                self._spawn_layout_repair_summary_path(scene_id),
                json.dumps(repair_summary, indent=2, sort_keys=True, ensure_ascii=False),
            )

    def generate_interpretation(self, user_request, input_dict):
        """
        Process a user request and generate structured output.
        """
        print("Generating scene description.......")
        result = self.interpreter.call_agent(user_request, input_dict)
        if self.input_type == "image":
            return result
        return self.interpreter.structure_output(input_dict["output_fn"])

    def fetch_interpretation(self, input_dict):
        """
        Retrieve road network and scenario descriptions from stored files.
        """
        scene_id = input_dict["scene_id"]
        return self.extract_net_description(scene_id), self.extract_scene_description(
            scene_id
        )


if __name__ == "__main__":
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Layer-1 structured static reconstruction (image -> CARLA scene).",
    )
    parser.add_argument(
        "--image-path",
        default=os.path.join(os.getcwd(), "data", "0107.jpg"),
        help="Input camera image to reconstruct (default: data/0107.jpg).",
    )
    parser.add_argument(
        "--output-folder",
        default=os.path.join(os.getcwd(), f"results/auto_result_{run_timestamp}"),
        help="Output folder for this run (default: results/auto_result_<timestamp>).",
    )
    parser.add_argument(
        "--user-input",
        default="",
        help="Optional extra scene description merged into the interpreter request.",
    )
    args = parser.parse_args()

    output_folder = args.output_folder
    image_path = args.image_path
    user_input = args.user_input
    if not os.path.exists(image_path):
        raise SystemExit(f"Input image not found: {image_path}")
    os.makedirs(output_folder, exist_ok=True)

    num_generated_scenes = 1
    mode = "FullPipeline"  #  "AfterInterpreter" #"AfterNet","AfterObject"
    input_type = "image"
    input_info = {
        "generation_mode": "generation",
        "input_type": input_type,
        "require_carla_connection": True,
        "spawn_point_limit": 12,
        "enable_scene_match": True,
        "enable_scene_verify": False,
        "verify_max_rounds": 2,
        "verify_min_score": 0.70,
        "verify_mode": "actor_graph",
        "map_match_blacklist_radius_m": 35.0,
        "map_match_topology_weight": 0.70,
        "map_match_side_context_weight": 0.20,
        "map_match_auxiliary_weight": 0.10,
        "debug_artifacts": False,
        "merge_user_description": True,
        "topology_cache_dir": os.path.join(os.getcwd(), "data", "map_cache"),
    }
    auto_generator = AutoGenerator(output_folder, input_info)

    for i in range(num_generated_scenes):
        scene_id = f"s{i:04d}"
        additional_info = {
            "output_fn": auto_generator._scene_understanding_path(scene_id),
            "scene_id": scene_id,
            "image_path": image_path,
        }
        if user_input.strip():
            additional_info["user_scene_description"] = user_input.strip()

        if input_type == "request":
            user_request = input_info["input_data"]
        else:
            # or add additional customized request
            user_request = ""

        if mode == "FullPipeline":
            scene_understanding = auto_generator.generate_scene_understanding(
                user_request,
                additional_info,
            )
            print(f"Generated scene understanding: {auto_generator._scene_understanding_path(scene_id)}")

            # Collect candidate map regions. Increase num_candidates for alternatives.
            num_candidates = 1
            candidate_matches: list = []  # list of (cand_scene_id, match_report_path)
            blacklist: list = []
            user_desc = additional_info.get("user_scene_description", "")
            for cand_idx in range(num_candidates):
                cand_scene_id = f"{scene_id}_c{cand_idx}"
                try:
                    match_path = auto_generator.analyze_topology_scene_match(
                        cand_scene_id,
                        scene_understanding,
                        blacklist_locations=blacklist,
                        image_path=image_path,
                    )
                except RuntimeError as exc:
                    print(f"Candidate {cand_idx} map match failed: {exc}")
                    break
                candidate_matches.append((cand_scene_id, match_path))
                report = auto_generator._load_json_if_exists(match_path)
                # Blacklist the entire world so the next candidate comes from a different map.
                world_name = report.get("world_name")
                if world_name:
                    blacklist.append({
                        "world": world_name,
                        "reason": f"Reserved for candidate {cand_idx}.",
                    })

            if not candidate_matches:
                print("No valid map candidates found. Exiting.")
                sys.exit(1)

            # Save spawn context before per-candidate mutations.
            base_spawn_context = deepcopy(auto_generator.carla_spawn_context)

            for cand_scene_id, match_path in candidate_matches:
                print(f"\n=== Building scenario for {cand_scene_id} ===")
                # Restore base context, then inject this candidate's anchor lane.
                auto_generator.carla_spawn_context = deepcopy(base_spawn_context)
                auto_generator._apply_match_to_spawn_context(match_path)
                auto_generator._build_scenario_for_match(
                    cand_scene_id,
                    match_path,
                    scene_understanding,
                    image_path,
                    user_desc,
                )

        elif mode == "AfterInterpreter":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)
            auto_generator.analyze_scene_match(scene_id)

        elif mode == "AfterNet":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)
            auto_generator.analyze_scene_match(scene_id)

        elif mode == "AfterObject":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_scene(scene_id, scenario_description)
            auto_generator.analyze_scene_match(scene_id)
