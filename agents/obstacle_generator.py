import ast
import json
import os
import subprocess
import sys
from typing import Optional
import xml.etree.ElementTree as ET
from agents.task_agent import OPENAI_TIMEOUT, TaskAgent
from tools.utils import read_sumo_file, extract_text_section, read_file, write_to_file


OBJECT_INFO_PREFIX = "__AUTOSCENARIO_OBJECT_INFO__="


class ObstacleGenerator(TaskAgent):
    """
    Obstacle Generator for controlled scenario generation in SUMO simulation.
    """

    def __init__(self):
        super().__init__()
        SYSTEM_PROMPT = """
        You generate object placement code for a scene reconstructed from a single traffic image and a simple road-network summary.
        Use only the visible scene evidence and the provided road-network context.
        Do not invent extra actors or enrich the scene.
        Unsupported visual objects such as traffic lights, road signs, storefronts, trees, or benches are layout references only. Do not include them in object_dict.

        Required output:
        - Include `## Description`, `## Reasoning`, and `## Decision`
        - Keep `## Description` and `## Reasoning` brief and focused
        - `## Decision` must contain exactly one executable ```python fenced block
        - The code must define `agent_dict` and `object_dict`

        Spawnable types:
        - Static objects: warningconstruction, streetbarrier, constructioncone, warningaccident
        - Vehicles: bike, car, jeep, motorcycle, suv, truck, van
        - Pedestrians may appear in `agent_dict` when clearly supported

        Placement constraints:
        - Generate both location and rotation for every spawned entity
        - Negate Y coordinates and yaw before spawning
        - Keep vehicles on the road and align them with the road direction unless the input clearly requires otherwise
        - If CARLA spawn points are provided, choose one as the anchor and keep all coordinates in that neighborhood
        - If the scene is uncertain, spawn fewer objects rather than hallucinating more
        - Vehicle/object distance must be at least 8 meters; object/object distance must be at least 0.5 meters
        - constructioncone z=0.5, streetbarrier z=1, warningconstruction z=1, vehicles z=2, pedestrians z=1
        - Use crosswalk context for pedestrians when clearly supported

        Code requirements:
        - Return the simplest executable Python code possible
        - Save spawned agents in `agent_dict`
        - Save spawned static objects in `object_dict`
        - Define `agent_dict` and `object_dict` directly as plain Python dictionaries
        - Store `location` and `rotation` as numeric length-3 lists
        - Avoid custom classes, CARLA imports, and extra helper functions unless absolutely necessary
        """
        self.pre_prompt = SYSTEM_PROMPT

    def refine_request(self, user_request, add_info=None):
        """Formats user request by appending the system prompt."""
        request = f"{self.pre_prompt}\nVisible Scene and Network Inputs:\n{user_request}"
        if add_info and add_info.get("validation_error"):
            request += (
                "\n\nPrevious generation failed validation and must be regenerated."
                "\nValidation failure:\n"
                f"{add_info['validation_error']}\n"
                "Regeneration requirements:\n"
                "1. Return executable Python code inside the ## Decision python fence.\n"
                "2. Ensure the script defines agent_dict and object_dict as dictionaries.\n"
                "3. Each entry must include location, rotation, and type.\n"
                "4. Use plain numeric lists for location and rotation.\n"
                "5. Keep the script minimal and executable without unnecessary helpers.\n"
                "6. The script must execute successfully and print valid object info."
            )
        return request

    @staticmethod
    def format_scene_context(scene_sections: dict, net_info: str, spawn_points_info: str) -> str:
        ordered_sections = [
            "Road Net Description",
            "Road Users Description",
            "Static Objects Description",
            "Vehicles' Locations and Behaviors",
            "Scenario Description",
        ]
        parts = []
        for section_name in ordered_sections:
            section_content = scene_sections.get(section_name, "").strip()
            if section_content:
                parts.append(f"{section_name}:\n{section_content}")
        if net_info:
            parts.append(f"Generated Road Network Summary:\n{net_info}")
        if spawn_points_info:
            parts.append(spawn_points_info)
        return "\n\n".join(parts)

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
        for point in spawn_points[:5]:
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
        """Extracts a compact network summary from text metadata and SUMO Net XML."""
        description = self.extract_network_info(txt_file_path, include_node_edge)
        compact_summary = self._summarize_network_xml(xml_file_path)
        return f"{description}\nCompact Network Summary:\n{compact_summary}"

    def _summarize_network_xml(self, xml_file_path: str) -> str:
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
        except ET.ParseError:
            return "Network summary unavailable because XML parsing failed."

        edge_lines = []
        for edge in root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            lanes = edge.findall("lane")
            num_lanes = len(lanes)
            lane = lanes[0] if lanes else None
            length = lane.attrib.get("length", "unknown") if lane is not None else "unknown"
            shape = lane.attrib.get("shape", "") if lane is not None else ""
            edge_lines.append(
                f"- edge_id={edge.attrib.get('id')}, from={edge.attrib.get('from')}, "
                f"to={edge.attrib.get('to')}, num_lanes={num_lanes}, length={length}, "
                f"shape={shape}"
            )

        if not edge_lines:
            return "No non-internal edges found."
        return "\n".join(edge_lines[:8])

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
            request_info = {
                "output_fn": output_file,
                "request_timeout": max(OPENAI_TIMEOUT, 180),
                "request_retries": 2,
                "request_label": "Obstacle generation",
            }
            if validation_error:
                request_info["validation_error"] = validation_error

            print(f"Obstacle generation attempt {attempt_count + 1}")
            try:
                self.send_request(user_request, request_info)
            except Exception as exc:
                validation_error = str(exc)
                attempt_count += 1
                print(f"Obstacle generation failed: {validation_error}")
                if attempt_count >= self.MAX_REGENERATE_ATTEMPTS:
                    raise RuntimeError(
                        "Obstacle generation failed after "
                        f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                    ) from exc
                print(f"Regenerating obstacle... Attempt {attempt_count + 1}")
                continue

            success, validation_error = self.extract_decision_data(
                scenario_id, output_folder
            )
            attempt_count += 1
            if not success and validation_error:
                print(f"Obstacle generation failed: {validation_error}")
            if not success and attempt_count >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Obstacle generation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            if attempt_count > 1:
                print(f"Regenerating obstacle... Attempt {attempt_count}")

        return attempt_count

    def extract_decision_data(
        self, scenario_id: str, output_folder: str
    ) -> tuple[bool, Optional[str]]:
        """Extracts the generated Python code from the Decision section of the output."""
        file_path = os.path.join(output_folder, f"{scenario_id}_scene.txt")
        text = read_file(file_path)
        if not text.strip():
            return False, "Empty obstacle generator output."

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
