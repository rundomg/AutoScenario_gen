import os
import re
import sys
from os.path import join
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.utils import read_file
from agents.net_generator import NetGenerator
from agents.obstacle_generator import ObstacleGenerator
from agents.universal_interpreter import UniInterpreter
from agents.scenario_generator import ScenarioGenerator
from tools.scene_map_matcher import SceneMapMatcher

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")


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

        self.interpreter = UniInterpreter(info_dict["input_type"])
        self.net_generator = NetGenerator(output_folder)
        self.obstacle_generator = ObstacleGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.enable_scene_match = info_dict.get("enable_scene_match", True)
        self.scene_matcher = SceneMapMatcher(
            host=self.carla_host,
            port=self.carla_port,
            timeout=self.carla_timeout,
            load_world_name=self._normalize_carla_world_name(self.carla_map),
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

    def _load_carla_spawn_context(self):
        if not self.require_carla_connection:
            return None

        try:
            import carla
        except ImportError as exc:
            raise RuntimeError(
                f"CARLA Python API is required before object generation: {exc}"
            ) from exc

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
            raise RuntimeError(
                f"Failed to connect to CARLA before object generation: {exc}"
            ) from exc

        if not spawn_points:
            raise RuntimeError(
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

        print(
            f"Connected to CARLA map {world_map.name} and cached "
            f"{len(sampled_spawn_points)} spawn points for coordinate initialization."
        )
        return {
            "map_name": world_map.name,
            "spawn_points": sampled_spawn_points,
        }

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
        net_info = self.obstacle_generator.extract_network_info_with_xml(
            join(self.output_folder, f"{scene_id}_net.txt"),
            join(self.output_folder, f"{scene_id}.net.xml"),
        )
        spawn_points_info = self.obstacle_generator.format_carla_spawn_points_info(
            self.carla_spawn_context
        )
        final_request_parts = [scenario_description, net_info]
        if spawn_points_info:
            final_request_parts.append(spawn_points_info)
        final_request = "\n".join(final_request_parts)
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
    output_folder = os.path.join(os.getcwd(), "auto_result")
    image_path = os.path.join(os.getcwd(), "data", "0107.jpg")

    os.makedirs(output_folder, exist_ok=True)

    num_generated_scenes = 1
    mode = "FullPipeline"  #  "AfterInterpreter" #"AfterNet","AfterObject"
    input_type = "image"
    input_info = {
        "generation_mode": "generation",
        "input_type": input_type,
        "require_carla_connection": True,
        "spawn_point_limit": 12,
    }
    auto_generator = AutoGenerator(output_folder, input_info)

    for i in range(num_generated_scenes):
        scene_id = f"{input_type}_interpreter_{i:04d}_split"
        additional_info = {
            "output_fn": join(output_folder, f"{scene_id}.txt"),
            "scene_id": scene_id,
            "image_path": image_path,
        }

        if input_type == "request":
            user_request = input_info["input_data"]
        else:
            # or add additional customized request
            user_request = ""

        if mode == "FullPipeline":
            standard_description = auto_generator.generate_interpretation(
                user_request, additional_info
            )
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)
            auto_generator.analyze_scene_match(scene_id)

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
