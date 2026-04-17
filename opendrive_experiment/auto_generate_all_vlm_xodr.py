import json
import os
import sys
from os.path import join

from dotenv import load_dotenv


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

from tools.utils import read_file, write_to_file
from opendrive_experiment.agents import (
    OpendriveGenerator,
    SceneUnderstandingInterpreter,
    XODRScenarioGenerator,
)
from opendrive_experiment.tools import (
    DEFAULT_GENERATION_PARAMS,
    apply_pairwise_ordering,
    build_relation_dsl,
    build_road_generation_request,
    evaluate_coordinate_program,
    load_xodr_world,
    project_entities_to_xodr,
    refine_projected_coordinates_with_pairwise_relations,
    validate_relation_layout,
    validate_pairwise_relations,
    write_coordinate_program,
)


load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")


class AutoGenerator:
    """Structured image-to-OpenDRIVE pipeline for the OpenDRIVE experiment."""

    def __init__(self, output_folder, info_dict=None):
        if info_dict is None:
            info_dict = {"input_type": "image"}
        self.input_type = info_dict["input_type"]
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        self.carla_host = info_dict.get("carla_host", "localhost")
        self.carla_port = info_dict.get("carla_port", 2000)
        self.carla_timeout = info_dict.get("carla_timeout", 10.0)
        self.spawn_point_limit = info_dict.get("spawn_point_limit", 12)
        self.require_carla_connection = info_dict.get("require_carla_connection", True)
        self.opendoive_stage = info_dict.get("opendrive_stage", "stage2").lower()
        self.generation_params = dict(DEFAULT_GENERATION_PARAMS)
        self.generation_params.update(info_dict.get("generation_params", {}))

        self.scene_understanding_interpreter = SceneUnderstandingInterpreter()
        self.opendrive_generator = OpendriveGenerator(
            output_folder,
            generation_stage=self.opendoive_stage,
        )
        self.scenario_generator = XODRScenarioGenerator()
        self.carla_spawn_context = None

    def adapt_generation_params_for_scene(self, scene_understanding):
        road_network = scene_understanding.get("road_network", {})
        lane_groups = road_network.get("lane_groups", []) or []
        current_width = float(self.generation_params.get("additional_width", 1.5))
        target_width = current_width

        for lane_group in lane_groups:
            if not isinstance(lane_group, dict):
                continue
            if int(lane_group.get("left_parking_lane_count", 0) or 0) > 0:
                target_width = max(target_width, 4.0)
            if int(lane_group.get("right_parking_lane_count", 0) or 0) > 0:
                target_width = max(target_width, 4.0)

        roadside_relations = []
        for entity in scene_understanding.get("traffic_subjects", []) or []:
            if isinstance(entity, dict):
                roadside_relations.append(str(entity.get("lane_side_relation") or ""))
        for entity in scene_understanding.get("background_traffic", []) or []:
            if isinstance(entity, dict):
                roadside_relations.append(str(entity.get("lane_side_relation") or ""))

        if any(
            relation in {"left_edge", "right_edge", "sidewalk_left", "sidewalk_right"}
            for relation in roadside_relations
        ):
            target_width = max(target_width, 6.0)

        control_elements = road_network.get("control_elements", []) or []
        if any(
            isinstance(control, dict)
            and "cone" in str(control.get("type") or "").lower()
            for control in control_elements
        ):
            target_width = max(target_width, 6.0)

        self.generation_params["additional_width"] = target_width
        return self.generation_params

    def _scene_understanding_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_scene_understanding.json")

    def _relation_dsl_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_relation_dsl.json")

    def _coordinate_program_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_coordinate_program.py")

    def _raw_initial_coordinates_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_coordinates_raw_initial.json")

    def _raw_ordered_coordinates_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_coordinates_raw_ordered.json")

    def _projected_coordinates_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_coordinates_projected.json")

    def _projected_refined_coordinates_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_coordinates_projected_refined.json")

    def _relation_validation_path(self, scene_id: str) -> str:
        return join(self.output_folder, f"{scene_id}_relation_validation.json")

    def generate_scene_understanding(self, user_request, input_dict):
        print("Generating structured scene understanding.......")
        output_fn = input_dict.get("output_fn") or self._scene_understanding_path(
            input_dict["scene_id"]
        )
        scene_understanding = self.scene_understanding_interpreter.call_agent(
            user_request,
            {
                **input_dict,
                "output_fn": output_fn,
            },
        )
        self.adapt_generation_params_for_scene(scene_understanding)
        return scene_understanding

    def read_scene_understanding(self, scene_id: str):
        return json.loads(read_file(self._scene_understanding_path(scene_id)))

    def generate_opendrive_from_road_network(self, scene_id, scene_understanding):
        print("Generating OpenDRIVE map from structured road network.......")
        request = build_road_generation_request(scene_understanding)
        return self.opendrive_generator.call_agent(
            request,
            scene_id,
            {"output_fn": join(self.output_folder, f"{scene_id}_xodr.txt")},
        )

    def load_xodr_world(self, road_artifact):
        if not self.require_carla_connection:
            return None
        print("Loading generated OpenDRIVE world in CARLA.......")
        self.carla_spawn_context = load_xodr_world(
            road_artifact["xodr_path"],
            carla_host=self.carla_host,
            carla_port=self.carla_port,
            generation_params=self.generation_params,
            timeout=self.carla_timeout,
            spawn_point_limit=self.spawn_point_limit,
        )
        return self.carla_spawn_context

    def generate_relation_dsl(self, scene_id, scene_understanding, road_artifact):
        print("Generating relation DSL.......")
        relation_dsl = build_relation_dsl(
            scene_understanding,
            self.carla_spawn_context,
            road_artifact=road_artifact,
        )
        write_to_file(
            self._relation_dsl_path(scene_id),
            json.dumps(relation_dsl, indent=2, sort_keys=True),
        )
        return relation_dsl

    def generate_initial_coordinates(self, scene_id):
        print("Generating initial coordinate program.......")
        return write_coordinate_program(
            self._relation_dsl_path(scene_id),
            self._coordinate_program_path(scene_id),
            self._raw_initial_coordinates_path(scene_id),
        )

    def evaluate_initial_coordinates(self, scene_id):
        print("Evaluating initial coordinate program.......")
        raw_initial = evaluate_coordinate_program(
            self._coordinate_program_path(scene_id),
            cwd=self.output_folder,
            output_path=self._raw_initial_coordinates_path(scene_id),
        )
        write_to_file(
            self._raw_initial_coordinates_path(scene_id),
            json.dumps(raw_initial, indent=2, sort_keys=True),
        )
        return raw_initial

    def apply_pairwise_ordering(self, scene_id, raw_initial, relation_dsl):
        print("Applying pairwise ordering.......")
        ordered = apply_pairwise_ordering(raw_initial, relation_dsl)
        write_to_file(
            self._raw_ordered_coordinates_path(scene_id),
            json.dumps(ordered, indent=2, sort_keys=True),
        )
        return ordered

    def project_entities_to_xodr(self, scene_id, ordered_coordinates, scene_understanding):
        print("Projecting entities onto OpenDRIVE surface.......")
        projected = project_entities_to_xodr(
            ordered_coordinates,
            scene_understanding,
            self.carla_spawn_context,
        )
        write_to_file(
            self._projected_coordinates_path(scene_id),
            json.dumps(projected, indent=2, sort_keys=True),
        )
        return projected

    def refine_projected_coordinates(self, scene_id, projected_coordinates, relation_dsl):
        print("Refining projected coordinates with pairwise relations.......")
        refined = refine_projected_coordinates_with_pairwise_relations(
            projected_coordinates,
            relation_dsl,
        )
        write_to_file(
            self._projected_refined_coordinates_path(scene_id),
            json.dumps(refined, indent=2, sort_keys=True),
        )
        return refined

    def validate_relation_layout(self, scene_id, relation_dsl, refined_coordinates):
        print("Validating ego, pairwise, and lane-aware relations.......")
        validation = validate_relation_layout(
            relation_dsl,
            refined_coordinates,
        )
        write_to_file(
            self._relation_validation_path(scene_id),
            json.dumps(validation, indent=2, sort_keys=True),
        )
        return validation

    def generate_scene_script_from_projected_coordinates(
        self,
        scene_id,
        refined_coordinates,
        road_artifact,
    ):
        print("Generating deterministic OpenDRIVE scene script.......")
        final_scene_path = self.scenario_generator.build_scene_script_from_projected_coordinates(
            scene_id=scene_id,
            output_folder=self.output_folder,
            projected_coordinates=refined_coordinates,
            projected_coordinates_path=self._projected_refined_coordinates_path(scene_id),
            xodr_path=road_artifact["xodr_path"],
            generation_params=self.generation_params,
        )
        print(f"Generated OpenDRIVE scene script: {final_scene_path}")
        return final_scene_path


if __name__ == "__main__":
    output_folder = os.path.join(os.getcwd(), "auto_result_xodr")
    image_path = os.path.join(os.getcwd(), "data", "020.jpg")

    os.makedirs(output_folder, exist_ok=True)

    num_generated_scenes = 1
    mode = "FullPipeline"
    input_type = "image"
    input_info = {
        "generation_mode": "generation",
        "input_type": input_type,
        "require_carla_connection": True,
        "spawn_point_limit": 12,
        "opendrive_stage": "stage2",
        "generation_params": {
            "wall_height": 0.0,
            "additional_width": 1.5,
        },
    }
    auto_generator = AutoGenerator(output_folder, input_info)

    for i in range(num_generated_scenes):
        scene_id = f"{input_type}_interpreter_{i:04d}_split"
        scene_understanding_path = auto_generator._scene_understanding_path(scene_id)
        additional_info = {
            "output_fn": scene_understanding_path,
            "scene_id": scene_id,
            "image_path": image_path,
        }

        user_request = ""

        if mode == "FullPipeline":
            scene_understanding = auto_generator.generate_scene_understanding(
                user_request,
                additional_info,
            )
            road_artifact = auto_generator.generate_opendrive_from_road_network(
                scene_id,
                scene_understanding,
            )
            auto_generator.load_xodr_world(road_artifact)
            relation_dsl = auto_generator.generate_relation_dsl(
                scene_id,
                scene_understanding,
                road_artifact,
            )
            auto_generator.generate_initial_coordinates(scene_id)
            raw_initial_coordinates = auto_generator.evaluate_initial_coordinates(scene_id)
            raw_ordered_coordinates = auto_generator.apply_pairwise_ordering(
                scene_id,
                raw_initial_coordinates,
                relation_dsl,
            )
            projected_coordinates = auto_generator.project_entities_to_xodr(
                scene_id,
                raw_ordered_coordinates,
                scene_understanding,
            )
            refined_coordinates = auto_generator.refine_projected_coordinates(
                scene_id,
                projected_coordinates,
                relation_dsl,
            )
            auto_generator.validate_relation_layout(
                scene_id,
                relation_dsl,
                refined_coordinates,
            )
            final_scene_path = auto_generator.generate_scene_script_from_projected_coordinates(
                scene_id,
                refined_coordinates,
                road_artifact,
            )
            print(f"Generated OpenDRIVE file: {road_artifact['xodr_path']}")
            print(f"Generated scene understanding: {scene_understanding_path}")
            print(f"Run this script to spawn the scene: {final_scene_path}")
