import os
import subprocess
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from tools.utils import extract_text_section, read_file, write_to_file


class ScenarioGenerator(TaskAgent):

    def __init__(self):
        super().__init__()
        SYSTEM_PROMPT = """
        You are GPT-4o, a large multi-modal model trained by OpenAI. Now you act as a mature scenario generator, who can understand user's testing request and design the correspondinng testing scenarios.
        The senario is built in Carla Simulator which uses Unreal Engine 4, so you will need to use the PythonAPI of Carla Simulator
        The user will give you a descripton of the scene, the loacation and the rotation of the vehicles and static objects in the scene.
        Your mission is to accurately understand the scene description provided by the user, identify the object layout of the scene, select appropriate objects and spawn them with proper location and roatation.
        The objects in the senario can be divided into two types: the static objects including construction objects like construction cones or Street Barrier, and the dynamic objects including vehicles.
        There are some world constraints in World setting part and objects constraints in Object part, these constraints can not be broken. The constriant with * is the most important.
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.

        World constraints
        1. Choose vehicle and static objects according to the given information. The vehicles can only be chosen from bike, car, jeep, motorcycle, suv, truck and van. The static objects can only be chosen from warningconstruction, streetbarrier, constructioncone, warningaccident.
        2. Generate vehicle colors
        3. If the weather is involved in description, use carla API and simulate the weather, or set the weather as ClearNoon
        4. The color of the vehicle is adjusted by 'R,G,B', not words like 'white' 'black' or 'random_color'


        Your answer should follow this format:
        ## Description
        Your description of the user request.
        ## Reasoning
        Reasoning based on user request, identify the type of obstalces and put at the designated coordinate. Tell the details how you calculate the coordinates of the objects.
        ## Decision
        The python code you generated. It should be like this format:
        Here is an example for the code, follow this format exactly:
        import carla
        import time
        import math
        import random

        # Connect to Carla Server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # Blueprint library
        blueprint_library = world.get_blueprint_library()

        # Function to spawn a static prop
        def spawn_static_prop(blueprint_name, location, rotation):
            blueprint = blueprint_library.find(f'static.prop.{blueprint_name}')
            transform = carla.Transform(location, rotation)
            static = world.try_spawn_actor(blueprint, transform)
            if static is not None:
                static.set_simulate_physics(True)
            return static

        # Function to spawn a dynamic vehicle
        def spawn_vehicle(blueprint_name, location, rotation, color=None):
            blueprint = blueprint_library.find(f'vehicle.omni.{blueprint_name}')
            if color:
                    blueprint.set_attribute('color', color)
            transform = carla.Transform(location, rotation)
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                vehicle.set_autopilot(False)  # Control manually
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
            return vehicle

        # Function to spawn a pedestrian
        def spawn_pedestrian(location, rotation):
            blueprint = random.choice(blueprint_library.filter('walker.pedestrian.*'))
            transform = carla.Transform(location, rotation)
            pedestrian = world.try_spawn_actor(blueprint, transform)

        # Load generated agents and objects

        # Change the weather
        weather = carla.WeatherParameters()
        world.set_weather(weather)

        # Spawn into Carla
        # The code to spawn vehicles
        spawn_vehicle(vehicle_type, loc, rot)
        v1_color = 'R,G,B'

        # The code to spawn the static objects
        spawn_static_prop(type, loc, rot)

        # The code to spawn pedestrian
        spawn_pedestrian(p1_loc, p1_rot)

        # Check if there is any objects

        # Allow time for the scenario to load
        time.sleep(2)

        """
        self.pre_prompt = SYSTEM_PROMPT
        self.obstalce_postition = None
        self.onstalce_type = None

    def refine_request(self, user_request, add_info=None):
        """Based on the scene description, add the description of the obstacle."""
        assert (
            add_info and "object_info" in add_info
        ), "Missing required object_info in add_info."
        return f"{self.pre_prompt}\nThe scene description is:\n{user_request}\nObject and agent information:\n{add_info['object_info']}"

    def run_scenario_generation(
        self, obj_code_fn, scenario_id, scenario_description, output_folder
    ):
        """Executes the generated scenario script; use obj location and scenario description to refine Carla scene."""
        command = ["python", obj_code_fn]
        result = subprocess.run(command, capture_output=True, text=True)
        obj_dict_str = result.stdout
        result_fn = obj_code_fn.replace(".py", "_obj.txt")
        write_to_file(result_fn, obj_dict_str)
        self.call_agent(scenario_description, scenario_id, output_folder, obj_dict_str)

    def call_agent(self, user_request, scenario_id, output_folder, object_info):
        """Sends requests to the agent until a valid response is obtained."""
        success_bool = False
        attempts = 0
        output_file = os.path.join(output_folder, f"{scenario_id}_scene_final.txt")
        while not success_bool:
            self.send_request(
                user_request, {"output_fn": output_file, "object_info": object_info}
            )
            success_bool = self.extract_decision_data(
                scenario_id, output_folder=output_folder
            )
            attempts += 1
            if attempts > 1:
                print(f"Regenerating scene... Attempt {attempts}")
        return attempts

    def extract_decision_data(self, scenario_id, output_folder=None, response=None):
        """Extracts decision data and generates the Python script for CARLA simulation."""
        if output_folder is None:
            text = response
        else:
            file_path = os.path.join(output_folder, f"{scenario_id}_scene_final.txt")
            text = read_file(file_path)

        decision_content = extract_text_section(text, r"## Decision\s+(.*?)(?=\s+##|$)")
        python_code = (
            extract_text_section(decision_content, r"```python\s+(.*?)\s+```")
            if decision_content
            else None
        )
        if python_code is None:
            return False

        write_to_file(
            os.path.join(output_folder, f"{scenario_id}_scene_final.py"), python_code
        )
        return True


if __name__ == "__main__":
    scenegen = ScenarioGenerator()
