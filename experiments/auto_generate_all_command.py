import os
import re
import sys
import xml.etree.ElementTree as ET
from os.path import join
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.utils import  read_file
from agents.net_generator import NetGenerator
from agents.obstacle_generator import ObstacleGenerator
from agents.universal_interpreter import UniInterpreter
from agents.scenario_generator import ScenarioGenerator
from agents.rou_generator import RouteGenerator

# Load API Key from environment variables
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

class AutoGenerator:
    """
    A class responsible for generating road network descriptions, obstacles, and full simulation scenarios.
    """
    def __init__(self,  output_folder, info_dict=None):
        """
        Initialize the AutoGenerator with required components.
        """
        if info_dict is None:
            info_dict = {"input_type": "image"}
        
        self.interpreter = UniInterpreter(info_dict["input_type"])
        self.net_generator = NetGenerator(output_folder)
        self.obstacle_generator = ObstacleGenerator()
        self.rou_generator = RouteGenerator()
        self.scenario_generator = ScenarioGenerator()
        self.output_folder = output_folder
        self.gps_info = info_dict.get("GPS_info", None)

    def extract_net_description(self, scene_id):
        """
        Extract the road network description from a stored text file.
        """
        text = read_file(join(self.output_folder, f"{scene_id}.txt"))
        match = re.search(r"##\s*Road Net Description\s*:\s*(.*?)\s*##", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_scene_description(self, scene_id):
        """
        Extract the scenario description from a stored text file.
        """
        text = read_file(join(self.output_folder, f"{scene_id}.txt"))
        #match = re.search(r"## Road Net Description:\s*(.*)", text, re.DOTALL)
        match = re.search(r"## Scenario Description:\s*(.*)", text, re.DOTALL) 
        return match.group(1).strip() if match else ""

    def generate_net(self, scene_id, road_net_description):
        """
        Generate a road network file using the NetGenerator component.
        """
        if self.gps_info is not None:
            self.net_generator.prepare_net_based_on_gps(self.output_folder, scene_id, self.gps_info)
        else:
            return self.net_generator.call_agent(road_net_description, scene_id, {"output_fn": join(self.output_folder, f"{scene_id}_net.txt")})

    def generate_objects(self, scene_id, scenario_description, generation_item="object"):
        """
        Generate objects within the simulation scene.
        """
        if generation_item == "object":
            if self.gps_info is not None:
                net_info = self.obstacle_generator.extract_network_info_with_gps(
                    join(self.output_folder, f"{scene_id}.txt"),
                    join(self.output_folder, f"{scene_id}.net.xml")
                )
            else:
                net_info = self.obstacle_generator.extract_network_info_with_xml(
                    join(self.output_folder, f"{scene_id}_net.txt"),
                    join(self.output_folder, f"{scene_id}.net.xml")
                )
            final_request = scenario_description + "\n" + net_info
            return self.obstacle_generator.call_agent(final_request, scene_id, self.output_folder)
        else:
            return self.generate_routes(scene_id, scenario_description)
        
    def generate_routes(self, scene_id, scenario_description):
        """
        Generate objects within the simulation scene.
        """
        net_info = self.obstacle_generator.extract_network_info_with_xml(
            join(self.output_folder, f"{scene_id}_net.txt"),
            join(self.output_folder, f"{scene_id}.net.xml"),
        )
        final_request = scenario_description + "\n" + net_info
        return self.rou_generator.call_agent(
            final_request, scene_id, self.output_folder
        )

    def generate_scene(self, scene_id, scenario_description):
        """
        Generate the full simulation scene file.
        """
        self.scenario_generator.run_scenario_generation(
            join(self.output_folder, f"{scene_id}_scene.py"),
            scene_id, scenario_description, self.output_folder
        )

    def generate_interpretation(self, user_request, input_dict):
        """
        Process a user request and generate structured output.
        """
        self.interpreter.call_agent(user_request, input_dict)
        return self.interpreter.structure_output(input_dict["output_fn"])

    def fetch_interpretation(self, input_dict):
        """
        Retrieve road network and scenario descriptions from stored files.
        """
        scene_id = input_dict["scene_id"]
        return self.extract_net_description(scene_id), self.extract_scene_description(scene_id)
    

if __name__ == "__main__":
    output_folder = os.path.join(os.getcwd(), "auto_result")
    os.makedirs(output_folder, exist_ok=True)

    num_generated_scenes = 1
    mode = "FullPipeline" # "AfterInterpreter" #"AfterNet","AfterObject"
    input_type = "request"  # crash_report, image, video, request
    request_string = "Generate something dangerous."

    GPS_info = {"lat":40.0034, "lon":116.3269} # None 
    if GPS_info is not None:
        lat = GPS_info["lat"]
        lon = GPS_info["lon"]
        request_string += f" GPS location: ({lat}, {lon})."
    
    input_info = {"generation_mode": "generation", "input_type": input_type, "input_data": request_string, "GPS_info":GPS_info}
    auto_generator = AutoGenerator(output_folder, input_info)

    
    for i in range(num_generated_scenes):
        scene_id = f"request_interpreter_{i:04d}_split"
        output_info = {"output_fn": join(output_folder, f"request_interpreter_{i:04d}.txt"), "scene_id": scene_id}

        if input_type == "request":
            user_request = input_info["input_data"]
        else:
            # or add additional customized request
            user_request = ""
    
        if mode == "FullPipeline":
            standard_description = auto_generator.generate_interpretation(
                user_request, output_info
            )
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(output_info)
            )
        
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterInterpreter":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(output_info)
            )
            auto_generator.generate_net(scene_id, road_net_description)
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterNet":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(output_info)
            )
            auto_generator.generate_objects(scene_id, scenario_description)
            auto_generator.generate_scene(scene_id, scenario_description)

        elif mode == "AfterObject":
            road_net_description, scenario_description = (
                auto_generator.fetch_interpretation(output_info)
            )
            auto_generator.generate_scene(scene_id, scenario_description)