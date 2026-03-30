import os
from typing import Optional
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from tools.utils import read_sumo_file, extract_text_section, read_file


class ObstacleGenerator(TaskAgent):
    """
    Obstacle Generator for controlled scenario generation in SUMO simulation.
    """

    def __init__(self):
        super().__init__()
        SYSTEM_PROMPT = """
        You are GPT-4o, a large multi-modal model trained by OpenAI. Now you act as a mature senario generator, who can understand user's testing request and design the correspondinng testing scenarios.
        The user will give you an existing map in sumo format containing node and edge, a description of the map and a description of the scenario.
        Your mission is to accurately understand the scene description provided by the user, identify the object layout of the scene, select appropriate objects and spawn them with proper location and roatation.
        The objects in the senario can be divided into two types: the static objects including construction objects like construction cones or Street Barrier, and the dynamic objects including vehicles.
        There are some world constraints in World setting part and objects constraints in Object part, these constraints can not be broken. The constriant with * is the most important.
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.

        World constraints
        1. Generate both the location and rotation for the objects, and list the reasons.
        2. Both the Y-axis value and yaw-value of the objects need to be negated when spawning.
        3. Unless otherwise specified, before assign the rotation of the vehicle, use the function to carlculate the road direction and then let the vehicle direction is the same as the road.
        4. Save the information of the vehicles and objects in the format of following code.
        5. Don't omit repeated code
        6. If CARLA spawn points are provided, choose one spawn point as the anchor and initialize all generated coordinates near that anchor instead of inventing arbitrary small coordinates far away from the spawn-point range.

        Object constraints
        1. Choose static objects from these: warningconstruction, streetbarrier, constructioncone, warningaccident. Do not choose other objects!
        2. Choose vehicles from these: bike, car, jeep, motorcycle, suv, truck and van. Do not choose other vehicles!
        3. Pedestrian usually show up around crosswalk.
        4. Spawn the construnctioncone at z=0.5, streetbarrier at z=1, the warningstruction at z=1, the vehicles at z=2, the pedestrian at z=1.
        5. The vehicles must be spawned on the road
        *6. The distance between the vehicles and other objects must not be less than 8 meters. The distance between other objects must not be less than 0.5 meters.
        7. Constructioncones are usually placed near the roadside, while streetbarriers are usually placed in the middle of the road.
        8. Streetbarrier and warningconstruction need to be perpendicular to the direction of the road.
        9. When mentioned about "wait at a intersection", do not put the vehicle in the middle of the intersection. They should be put on one road near intersection.
        10. If an accident happened, the vehicles in accident are always closed to each other and static objects are always placed closed around accident car. But at least put them 1.5 meters apart.
        11. You can use for loop to simulate high density traffic.
        12. Do not spawn the car at the edge of the road
        14. Crosswalk at intersections are typically located along the edges of the intersection where pedestrian pathways meet and cross the roads.
        15. If an accident happened when the pedestrian is crossing the roadway, put them near the crosswalk.
        16. When a car turn right, it must be at the rightest lane of the road. And for turn left, it must be put at the leftest lane of the road.


        Your answer should follow this format:
        ## Description
        Your description of the user request.
        ## Reasoning
        Reasoning based on user request, identify the type of obstalces and put at the designated coordinate. Tell the details how you calculate the coordinates of the objects.
        ## Decision
        The python code you generated. It should be like this format:
        Here is an example for the code, follow this format exactly:
        import time
        import math
        import random

        class Location:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

            def __add__(self, other):
                return Location(self.x + other.x, self.y + other.y, self.z + other.z)

        class Rotation:
            def __init__(self, pitch, yaw, roll):
                self.pitch = pitch
                self.yaw = yaw
                self.roll = roll

        # Function to negate the y axis
        def nege_y(location)

        # Function to negate the yaw axis
        def nege_yaw(rotation)

        # Function to calculate the road direction (angle) from start to end
        def road_direction(start, end):
            # Extract coordinates from the start and end points
            x1, y1 = start.x, start.y
            x2, y2 = end.x, end.y

            # Calculate the angle in radians
            angle_radians = math.atan2(y2 - y1, x2 - x1)

            # Convert radians to degrees
            angle_degrees = math.degrees(angle_radians)

            return yaw

        # Get the node
        node1 = Location(x, y, 0)
        node2 = ...

        # Function to negate the y axis
        def nege_y(location)

        # Function to negate the yaw axis
        def nege_yaw(rotation)

        # Ensure all objects, including statics and vehicles' location is 5 meters apart from each other
        # The loc and rot of static objects, use nege_y
        s1_loc = nege_y(Location(x, y, z))
        road_angle = road_direction(start_node, end_node)
        s1_rot = Rotation(x, road_angle + 90, z)
        s2 = ...

        # All the car must be on the road
        # The loc and rot of vehicle, use nege_y and road_direction
        v1_loc = nege_y(Location(x, y, z))
        road_angle = road_direction(start_node, end_node)
        v1_rot = nege_yaw(Rotation(x, road_angle, z))

        v2_loc = nege_y(Location(x, y, z))
        road_angle = road_direction(start_node, end_node)
        v2_rot = nege_yaw(Rotation(x, road_angle, z))

        # save all agents location, rotation, type into a dictionary with key as agent name. Type includes pedestrain, bike, car, jeep, motorcycle, suv, truck and van
        agent_dict = {"v1": {"location": (v1_loc.x, v1_loc.y, v1_loc.z), "rotation":(v1_rot.x, v1_rot.y, v1_rot.z), "type": "vehicle"}, "v2":{"location":, "rotation":, "type": "vehicle"}}
        # save all objects location, rotation, type into a dictionary with key as object name. Type includes warningconstruction, streetbarrier, constructioncone, warningaccident
        object_dict = {"s1": {"location": (s1_loc.x, s1_loc.y, s1_loc.z), "rotation":(s1_rot.x, s1_rot.y, s1_rot.z), "type": "streetbarrier"},}

        # Print the dictionaries
        
        
        """
        self.pre_prompt = SYSTEM_PROMPT

    def refine_request(self, user_request, add_info=None):
        """Formats user request by appending the system prompt."""
        return f"{self.pre_prompt}\nScenario Description:\n{user_request}"

    def format_carla_spawn_points_info(self, spawn_context: dict) -> str:
        """Formats CARLA spawn point metadata so the LLM can initialize coordinates near a real anchor."""
        if not spawn_context:
            return ""

        map_name = spawn_context.get("map_name", "unknown")
        spawn_points = spawn_context.get("spawn_points", [])
        if not spawn_points:
            return f"CARLA Map: {map_name}\nCARLA Spawn Points: unavailable"

        lines = [
            f"CARLA Map: {map_name}",
            "CARLA Spawn Points (use one of these as the coordinate anchor for initialization):",
        ]
        for point in spawn_points:
            location = point["location"]
            rotation = point["rotation"]
            lines.append(
                "  - "
                f"index={point['index']}, "
                f"location=({location['x']:.6f}, {location['y']:.6f}, {location['z']:.6f}), "
                f"rotation=({rotation['pitch']:.6f}, {rotation['yaw']:.6f}, {rotation['roll']:.6f})"
            )
        lines.append(
            "Initialization rule: pick one spawn point as the anchor and keep all generated object coordinates in that same neighborhood."
        )
        return "\n".join(lines)

    def extract_network_info(
        self, txt_file_path: str, include_node_edge: Optional[bool] = False
    ) -> str:
        """Extracts network description and SUMO specifications(Node and Edge) from the given text file."""
        with open(txt_file_path, "r") as file:
            content = file.read()
        description = extract_text_section(content, r"## Description\s+(.*?)\s+##")
        if include_node_edge:
            net_info = extract_text_section(
                content, r"## SUMO Files Specification\s+(.*)"
            )
            return f"Map Description:\n{description}\nSUMO Network Info:\n{net_info}"
        return f"Map Description:\n{description}"

    def extract_network_info_with_gps(
        self, txt_file_path: str, xml_file_path: str
    ) -> str:
        with open(txt_file_path, "r") as file:
            content = file.read()
        description = extract_text_section(
            content, r"#{2,3}\s*Road Net Description:\s*\n(.*?)(?=\n#{2,3}|\Z)"
        )
        sumo_net = read_sumo_file(xml_file_path)
        return f"{description}\nNetwork XML:\n{sumo_net}"

    def extract_network_info_with_xml(
        self,
        txt_file_path: str,
        xml_file_path: str,
        include_node_edge: Optional[bool] = False,
    ) -> str:
        """Extracts network description and SUMO Net XML information."""
        description = self.extract_network_info(txt_file_path, include_node_edge)
        sumo_net = read_sumo_file(xml_file_path)
        return f"{description}\nNetwork XML:\n{sumo_net}"

    def extract_network_info_with_offset(
        self, txt_file_path: str, xml_file_path: str
    ) -> str:
        """Extracts network description, SUMO network data, and netOffset value."""
        description = self.extract_network_info(txt_file_path)
        net_offset = self._extract_net_offset(xml_file_path)

        return f"{description}\nNet Offset: {net_offset}\n(Ensure to consider netOffset when calculating coordinates.)"

    def _extract_net_offset(self, xml_file_path: str) -> str:
        """Extracts netOffset value from SUMO XML file."""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            location = root.find("location")
            return (
                location.attrib.get("netOffset", "Not found")
                if location is not None
                else "Not found"
            )
        except ET.ParseError:
            return "XML parsing error"

    def call_agent(
        self, user_request: str, scenario_id: str, output_folder: str
    ) -> int:
        """Generates an obstacle and extracts the decision data."""
        attempt_count = 0
        success = False
        output_file = os.path.join(output_folder, f"{scenario_id}_scene.txt")

        while not success:
            self.send_request(user_request, {"output_fn": output_file})
            success = self.extract_decision_data(scenario_id, output_folder)
            attempt_count += 1
            if attempt_count > 1:
                print(f"Regenerating obstacle... Attempt {attempt_count}")

        return attempt_count

    def extract_decision_data(self, scenario_id: str, output_folder: str) -> bool:
        """Extracts the generated Python code from the Decision section of the output."""
        file_path = os.path.join(output_folder, f"{scenario_id}_scene.txt")
        text = read_file(file_path)

        decision_content = extract_text_section(text, r"## Decision\s+(.*?)(?=\s+##|$)")
        if decision_content is None:
            return False

        python_code = extract_text_section(decision_content, r"```python\s+(.*?)\s+```")
        if python_code is None:
            return False

        with open(
            os.path.join(output_folder, f"{scenario_id}_scene.py"),
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(python_code)
        return True


if __name__ == "__main__":

    file_folder = os.path.join(os.getcwd(), "auto_result", "module_test")
    obstaclegen = ObstacleGenerator()
    scene_id = f"crash_report_interpreter_0000_split"
    subfolder_path = os.path.join(file_folder, scene_id)
    net_xml_path = os.path.join(subfolder_path, f"{scene_id}.net.xml")
    txt_xml_path = os.path.join(subfolder_path, f"{scene_id}_net.txt")
    net_info = obstaclegen.extract_network_info_with_offset(txt_xml_path, net_xml_path)
