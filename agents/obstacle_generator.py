import ast
import json
import os
import subprocess
import sys
from typing import Optional
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from tools.utils import read_sumo_file, extract_text_section, read_file, write_to_file


OBJECT_INFO_PREFIX = "__AUTOSCENARIO_OBJECT_INFO__="


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
        request = f"{self.pre_prompt}\nScenario Description:\n{user_request}"
        if add_info and add_info.get("validation_error"):
            request += (
                "\n\nPrevious generation failed validation and must be regenerated."
                "\nValidation failure:\n"
                f"{add_info['validation_error']}\n"
                "Regeneration requirements:\n"
                "1. Return executable Python code inside the ## Decision python fence.\n"
                "2. Ensure the script defines agent_dict and object_dict as dictionaries.\n"
                "3. Each entry must include location, rotation, and type.\n"
                "4. The script must execute successfully and print valid object info."
            )
        return request

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
        validation_error = None
        output_file = os.path.join(output_folder, f"{scenario_id}_scene.txt")

        while not success:
            request_info = {"output_fn": output_file}
            if validation_error:
                request_info["validation_error"] = validation_error

            self.send_request(user_request, request_info)
            success, validation_error = self.extract_decision_data(
                scenario_id, output_folder
            )
            attempt_count += 1
            if attempt_count > 1:
                print(f"Regenerating obstacle... Attempt {attempt_count}")

        return attempt_count

    def extract_decision_data(
        self, scenario_id: str, output_folder: str
    ) -> tuple[bool, Optional[str]]:
        """Extracts the generated Python code from the Decision section of the output."""
        file_path = os.path.join(output_folder, f"{scenario_id}_scene.txt")
        text = read_file(file_path)

        decision_content = extract_text_section(text, r"## Decision\s+(.*?)(?=\s+##|$)")
        if decision_content is None:
            return False, "Missing ## Decision section in obstacle generator output."

        python_code = extract_text_section(decision_content, r"```python\s+(.*?)\s+```")
        if python_code is None:
            return False, "Missing executable python fenced block in ## Decision."

        normalized_code, normalize_error = self._normalize_generated_code(python_code)
        if normalize_error:
            return False, normalize_error

        output_path = os.path.join(output_folder, f"{scenario_id}_scene.py")
        write_to_file(output_path, normalized_code)
        return self._validate_generated_script(output_path)

    def _normalize_generated_code(
        self, python_code: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Normalizes generated code and injects compatibility shims for known issues."""
        normalized_code = python_code.strip()
        try:
            syntax_tree = ast.parse(normalized_code)
        except SyntaxError as exc:
            return (
                None,
                "Generated python code has syntax errors before validation: "
                f"{exc.msg} at line {exc.lineno}.",
            )

        normalized_code = self._inject_class_compatibility_shims(
            normalized_code, syntax_tree
        )
        normalized_code = (
            f"{normalized_code}\n\n{self._build_object_info_emitter_block()}\n"
        )
        return normalized_code, None

    def _inject_class_compatibility_shims(
        self, python_code: str, syntax_tree: ast.AST
    ) -> str:
        """Patches frequently missing compatibility helpers directly into class bodies."""
        lines = python_code.splitlines()
        injections = []

        for node in getattr(syntax_tree, "body", []):
            if not isinstance(node, ast.ClassDef):
                continue

            class_indent = self._detect_class_body_indent(lines, node)

            if node.name == "Location" and not self._class_has_method(node, "__sub__"):
                injections.append(
                    (
                        node.end_lineno,
                        [
                            "",
                            f"{class_indent}def __sub__(self, other):",
                            (
                                f"{class_indent}    return Location("
                                "self.x - other.x, self.y - other.y, self.z - other.z)"
                            ),
                        ],
                    )
                )

            if node.name == "Rotation":
                alias_specs = [("x", "pitch"), ("y", "yaw"), ("z", "roll")]
                alias_lines = []
                for alias_name, source_name in alias_specs:
                    if self._class_needs_rotation_alias(node, alias_name, source_name):
                        alias_lines.extend(
                            [
                                "",
                                f"{class_indent}@property",
                                f"{class_indent}def {alias_name}(self):",
                                f"{class_indent}    return self.{source_name}",
                            ]
                        )
                if alias_lines:
                    injections.append((node.end_lineno, alias_lines))

        for line_number, block in sorted(injections, reverse=True):
            lines[line_number:line_number] = block

        return "\n".join(lines)

    @staticmethod
    def _detect_class_body_indent(lines: list[str], node: ast.ClassDef) -> str:
        if node.body:
            body_line = lines[node.body[0].lineno - 1]
            return body_line[: len(body_line) - len(body_line.lstrip())]

        class_line = lines[node.lineno - 1]
        class_indent = class_line[: len(class_line) - len(class_line.lstrip())]
        return f"{class_indent}    "

    @staticmethod
    def _class_has_method(node: ast.ClassDef, method_name: str) -> bool:
        return any(
            isinstance(child, ast.FunctionDef) and child.name == method_name
            for child in node.body
        )

    def _class_needs_rotation_alias(
        self, node: ast.ClassDef, alias_name: str, source_name: str
    ) -> bool:
        if self._class_has_method(node, alias_name):
            return False
        if self._class_assigns_attribute(node, alias_name):
            return False
        return self._class_assigns_attribute(node, source_name)

    @staticmethod
    def _class_assigns_attribute(node: ast.ClassDef, attr_name: str) -> bool:
        for child in ast.walk(node):
            target = None
            if isinstance(child, ast.Assign):
                for assign_target in child.targets:
                    if (
                        isinstance(assign_target, ast.Attribute)
                        and isinstance(assign_target.value, ast.Name)
                        and assign_target.value.id == "self"
                        and assign_target.attr == attr_name
                    ):
                        return True
            elif isinstance(child, ast.AnnAssign):
                target = child.target
            elif isinstance(child, ast.AugAssign):
                target = child.target

            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == attr_name
            ):
                return True
        return False

    @staticmethod
    def _build_object_info_emitter_block() -> str:
        return f"""import json


def _autoscenario_to_plain_data(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {{str(key): _autoscenario_to_plain_data(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_autoscenario_to_plain_data(item) for item in value]
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [value.x, value.y, value.z]
    if hasattr(value, "pitch") and hasattr(value, "yaw") and hasattr(value, "roll"):
        return [value.pitch, value.yaw, value.roll]
    return str(value)


def _autoscenario_emit_object_info():
    payload = {{
        "agent_dict": _autoscenario_to_plain_data(globals().get("agent_dict", {{}})),
        "object_dict": _autoscenario_to_plain_data(globals().get("object_dict", {{}})),
    }}
    print("{OBJECT_INFO_PREFIX}" + json.dumps(payload, sort_keys=True))


_autoscenario_emit_object_info()"""

    def _validate_generated_script(self, script_path: str) -> tuple[bool, Optional[str]]:
        """Runs the generated script and validates the emitted object info payload."""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(script_path),
            )
        except subprocess.TimeoutExpired:
            return False, "Generated python script timed out during validation."

        if result.returncode != 0:
            error_output = (result.stderr or result.stdout).strip()
            if not error_output:
                error_output = f"Exited with status code {result.returncode}."
            return (
                False,
                "Generated python script failed during validation:\n"
                f"{self._truncate_validation_message(error_output)}",
            )

        payload, payload_error = self._extract_object_info_payload(result.stdout)
        if payload_error:
            return False, payload_error

        is_valid, validation_error = self._validate_object_info_payload(payload)
        if not is_valid:
            return False, validation_error

        return True, None

    def _extract_object_info_payload(
        self, stdout: str
    ) -> tuple[Optional[dict], Optional[str]]:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(OBJECT_INFO_PREFIX):
                continue
            payload_text = line[len(OBJECT_INFO_PREFIX) :].strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                return (
                    None,
                    "Generated python script printed malformed object info payload: "
                    f"{exc.msg}.",
                )
            if not isinstance(payload, dict):
                return None, "Generated object info payload is not a dictionary."
            return payload, None

        return (
            None,
            "Generated python script did not print the required structured object info payload.",
        )

    def _validate_object_info_payload(
        self, payload: dict
    ) -> tuple[bool, Optional[str]]:
        agent_dict = payload.get("agent_dict")
        object_dict = payload.get("object_dict")

        if not isinstance(agent_dict, dict):
            return False, "agent_dict is missing or is not a dictionary."
        if not isinstance(object_dict, dict):
            return False, "object_dict is missing or is not a dictionary."
        if not agent_dict and not object_dict:
            return False, "Both agent_dict and object_dict are empty."

        for collection_name, collection in (
            ("agent_dict", agent_dict),
            ("object_dict", object_dict),
        ):
            is_valid, error = self._validate_entity_collection(collection_name, collection)
            if not is_valid:
                return False, error

        return True, None

    def _validate_entity_collection(
        self, collection_name: str, collection: dict
    ) -> tuple[bool, Optional[str]]:
        for entity_name, entity_info in collection.items():
            if not isinstance(entity_info, dict):
                return (
                    False,
                    f"{collection_name}.{entity_name} is not a dictionary entry.",
                )
            for key in ("location", "rotation", "type"):
                if key not in entity_info:
                    return False, f"{collection_name}.{entity_name} is missing {key}."

            is_valid, error = self._validate_vector(
                entity_info["location"], f"{collection_name}.{entity_name}.location"
            )
            if not is_valid:
                return False, error

            is_valid, error = self._validate_vector(
                entity_info["rotation"], f"{collection_name}.{entity_name}.rotation"
            )
            if not is_valid:
                return False, error

            if not isinstance(entity_info["type"], str) or not entity_info["type"].strip():
                return False, f"{collection_name}.{entity_name}.type must be a non-empty string."

        return True, None

    @staticmethod
    def _validate_vector(value, field_name: str) -> tuple[bool, Optional[str]]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return False, f"{field_name} must be a length-3 list or tuple."
        if not all(isinstance(item, (int, float)) for item in value):
            return False, f"{field_name} must contain only numeric values."
        return True, None

    @staticmethod
    def _truncate_validation_message(message: str, limit: int = 1200) -> str:
        if len(message) <= limit:
            return message
        return f"{message[:limit]}..."


if __name__ == "__main__":

    file_folder = os.path.join(os.getcwd(), "auto_result", "module_test")
    obstaclegen = ObstacleGenerator()
    scene_id = f"crash_report_interpreter_0000_split"
    subfolder_path = os.path.join(file_folder, scene_id)
    net_xml_path = os.path.join(subfolder_path, f"{scene_id}.net.xml")
    txt_xml_path = os.path.join(subfolder_path, f"{scene_id}_net.txt")
    net_info = obstaclegen.extract_network_info_with_offset(txt_xml_path, net_xml_path)
