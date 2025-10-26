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

        self.interpreter = UniInterpreter(info_dict["input_type"])
        self.net_generator = NetGenerator(output_folder)
        self.obstacle_generator = ObstacleGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.output_folder = output_folder

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
        final_request = scenario_description + "\n" + net_info
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

    def generate_interpretation(self, user_request, input_dict):
        """
        Process a user request and generate structured output.
        """
        print(f"Generating scene description.......")
        self.interpreter.call_agent(user_request, input_dict)
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
    video_path = os.path.join(os.getcwd(), "data", "crash1.mp4")
    os.makedirs(output_folder, exist_ok=True)

    num_generated_scenes = 1
    # mode = "AfterInterpreter"  # [AfterInterpreter] # "FullPipeline"# GenerateInterpretation
    mode = "FullPipeline"
    input_type = "video"
    input_info = {"generation_mode": "generation", "input_type": input_type}
    auto_generator = AutoGenerator(output_folder, input_info)

    for i in range(num_generated_scenes):
        scene_id = f"{input_type}_interpreter_{i:04d}_split"
        additional_info = {
            "output_fn": join(output_folder, f"{input_type}_interpreter_{i:04d}.txt"),
            "scene_id": scene_id,
            "video_path": video_path,
        }

        if input_type == "request":
            user_request = input_info["input_data"]
        else:
            # or add additional customized request
            user_request = ""

        if mode == "GenerateInterpretation":
            standard_description = auto_generator.generate_interpretation(
                user_request, additional_info
            )
        elif mode == "FullPipeline":
            standard_description = auto_generator.generate_interpretation(
                user_request, additional_info
            )
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterInterpreter":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterNet":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterObject":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(additional_info)
            )
            auto_generator.generate_scene(scene_id, scenario_description)
