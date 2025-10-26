import re
import sys

sys.path.insert(0, "../")
from agents.task_agent import TaskAgent
from tools.evaluation_metrics import compute_entropy


class CommandInterpreter(TaskAgent):

    def __init__(self) -> None:
        super().__init__()

        SYSTEM_PROMPT = """
        You are GPT-4V(ision), a large multi-modal model trained by OpenAI. Now you act as a mature autonomous driving tester, who can understand user's testing request and design the correspondinng testing scenarios. 
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.

        Your answer should follow this format:
        ## Description
        Your description of the user request.
        ## Reasoning
        reasoning based on user request, what is the testing goal and what are the best testing scenarios. Try to create complex road network with varying road types, road sturcture and connections.  The generated road network should be *very detailed and shorter than 100m*. Provide detailed description in "road geometry" part in the `## Decision` section.
        ## Decision
        Should start with answer directly no other words.
        Output a string list ["road description:", "traffic density:", "test goal:"], and a vector (number of lanes, number of vehicles) for each scenario. SHOULD BE exactly same and no other words!
        Test goal should be one of ["scene_orientated", "av_behavior", " bv_behavior", "general"]. For a given goal, provide a detailed description, such as turning behavior. 
        """
        road_description = """If a GPS location is provided, generate a brief road description based on the latitude and longitude. """

        type_instruction = """In terms of road geometry, a realistic intersection structure takes into account several factors to ensure safe and efficient vehicle and pedestrian movement. Here are the key elements:
        1. Lane Width
        Standard Lane Width: Typically ranges from 3.0 to 3.6 meters (10 to 12 feet).
        Turn Lanes: May be slightly wider to accommodate larger turning vehicles.
        2. Turn Radii
        Minimum Turn Radius: Should accommodate the largest vehicles expected to use the intersection, such as buses or trucks. A typical minimum radius is around 10.7 meters (35 feet) for tight turns, but larger for more comfortable turns.
        3. Intersection Angles
        Right Angles: Intersections ideally meet at right angles (90 degrees) to minimize the complexity of vehicle movements.
        Skewed Intersections: Should be avoided or minimized, as they can complicate traffic movements and reduce visibility.
        4. Crosswalks and Pedestrian Facilities
        Crosswalk Width: Typically ranges from 2.4 to 3.6 meters (8 to 12 feet).
        5. Intersection Corner Design
        Curb Radii: Typically ranges from 4.6 to 9.1 meters (15 to 30 feet) for urban intersections, but can be larger for high-speed or high-volume intersections.
        Tight Corners: Slower traffic down and protect pedestrians, while larger radii accommodate higher speed and larger vehicles.
        6. Sight Distance
        Adequate Sight Lines: Ensure that drivers can see oncoming traffic, pedestrians, and any obstacles.
        Sight Triangles: Area at corners where obstructions are minimized to ensure visibility.
        """

        additional_hints = "Typical intersection types: Crossroad, X-intersection, Y-intersection, T-intersection, Misaligned intersection, Ramp merge, Deformed intersection etc."

        FORMAT_PROMT = """Try to describe from following perspectives: 1. General Layout. Overall Structure: Grid, radial, organic, or a mix. Extent and Coverage: Geographic area covered, including urban, suburban, and rural areas. Key Features: Major roads, highways, and interchanges.
        2. Road Types and Configurations Road Types: Highways, arterial roads, local streets, alleys, and service roads. 3. Number of lanes for each road.
        For exmaple, one answer is: Road Description: At the heart of this network, Central Square Intersection itself is a four-way junction where the city's two major arterial roads, Grand Avenue and Main Street, converge.
        On the northern approach, Grand Avenue features four lanes that split into dedicated left-turn, straight, and right-turn lanes as it nears the intersection. This road is flanked by parallel service lanes that provide access to local businesses and residential areas, ensuring that through traffic remains unobstructed. To the south, Grand Avenue continues with similar lane configurations, but also includes a bus lane that integrates seamlessly with the city's public transport system.Main Street, running east to west, mirrors this complexity. The western approach of Main Street includes three lanes for straight-going traffic, complemented by additional lanes for left and right turns. The eastern stretch of Main Street also features multiple lanes, with a dedicated tram line running down the center, offering another layer of public transportation connectivity.
        Adjacent to the intersection are several critical connectors. On the northwest corner, Elm Street branches off from Grand Avenue, acting as a key feeder road for the residential neighborhood it serves. This street features traffic calming measures such as speed bumps and narrow lanes to maintain safe speeds. On the southeast side, Oak Road provides a direct route to the nearby commercial district, with wide lanes to accommodate delivery trucks and heavy traffic", "traffic density: medium", "test goal: av_behavior: turning"], [2, 5]"""

        self.pre_prompt = (
            SYSTEM_PROMPT
            + road_description
            + type_instruction
            + additional_hints
            + FORMAT_PROMT
        )

    def task_summary(self, results):
        num_scenarios = len(results)
        print("num_scenarios", num_scenarios)
        lane_num_list = [int(res["number of lanes"]) for res in results]
        vehicle_num_list = [int(res["number of vehicles"]) for res in results]
        print(
            "entropy of lanes and vehicles are: ",
            compute_entropy(lane_num_list),
            "   ",
            compute_entropy(vehicle_num_list),
        )
        return lane_num_list, vehicle_num_list

    def call_agent(self, user_request, input_dict):
        answer_not_right = True
        generation_cnt = 0
        output_fn = input_dict["output_fn"]

        while answer_not_right:
            self.send_request(user_request, input_dict)
            result, answer_not_right = self.extract_decision_data(output_fn)
            generation_cnt += 1
            if generation_cnt > 1:
                print(
                    "Regenerating the scene interpretation! Generation Round:",
                    generation_cnt,
                )
        return result

    def refine_request(self, user_request, add_info=None):
        final_request = self.pre_prompt + f"\nUser request is : {user_request}"
        return final_request

    def extract_decision_data(self, file_path):
        """Return extracted structured answer and Need to regenerate or not."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        decision_text = re.search(r"## Decision\n(.+)", text, re.S)
        if decision_text:
            decision_text = decision_text.group(1)
        else:
            return "No decision section found."

        # Extract data within square brackets
        pattern = re.compile(r'\["(.*?)", "(.*?)", "(.*?)"\], \[(\d+), (\d+)\]')
        matches = pattern.findall(decision_text)
        if len(matches) == 0:
            return None, True

        result = []

        for match in matches:
            road_geometry = re.search(r":\s*(.*)", match[0]).group(1)
            traffic_density = re.sub(r"^.*: ", "", match[1])
            test_goal = re.sub(r"^.*: ", "", match[2])
            result.append(
                {
                    "road description": road_geometry,
                    "traffic density": traffic_density,
                    "test goal": test_goal,
                    "number of lanes": int(match[3]),
                    "number of vehicles": int(match[4]),
                }
            )
        if len(result) == 0:
            return result, True
        return result, False
