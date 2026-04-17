import ast
import os
import re
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from tools.utils import extract_text_section, read_file, write_to_file


SCENE_HELPER_MARKER = "# __AUTOSCENARIO_SCENE_HELPERS__"
SCENE_ACTIVATION_MARKER = "# __AUTOSCENARIO_ACTIVATE_HELPERS__"
SCENE_SPECTATOR_MARKER = "# __AUTOSCENARIO_FOCUS_SPECTATOR__"


class ScenarioGenerator(TaskAgent):

    def __init__(self):
        super().__init__()
        SYSTEM_PROMPT = """
        You are GPT-4o, a large multi-modal model trained by OpenAI. Now you act as a mature scenario generator, who can understand user's testing request and design the correspondinng testing scenarios.
        The scenario is built in Carla Simulator which uses Unreal Engine 4, so you will need to use the PythonAPI of Carla Simulator
        The user will give you a descripton of the scene, the loacation and the rotation of the vehicles and static objects in the scene.
        Your mission is to accurately understand the scene description provided by the user, identify the object layout of the scene, select appropriate objects and spawn them with proper location and roatation.
        The objects in the scenario can be divided into two types: the static objects including construction objects like construction cones or Street Barrier, and the dynamic objects including vehicles.
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
        try:
            result = subprocess.run(
                [sys.executable, obj_code_fn],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(obj_code_fn) or None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Scene object script timed out during execution: {obj_code_fn}"
            ) from exc

        if result.returncode != 0:
            error_output = (result.stderr or result.stdout).strip()
            if not error_output:
                error_output = f"Exited with status code {result.returncode}."
            raise RuntimeError(
                "Scene object script failed before scene generation:\n"
                f"{self._truncate_subprocess_output(error_output)}"
            )

        obj_dict_str = result.stdout.strip()
        if not obj_dict_str:
            raise RuntimeError(
                f"Scene object script produced empty stdout: {obj_code_fn}"
            )

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
            success_bool, error_message = self.extract_decision_data(
                scenario_id, output_folder=output_folder
            )
            attempts += 1
            if not success_bool and error_message:
                print(f"Scene normalization failed: {error_message}")
            if not success_bool and attempts >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Scene generation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {error_message}"
                )
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
        if decision_content is None:
            return False, "Missing ## Decision section in scenario generator output."

        python_code = (
            extract_text_section(decision_content, r"```python\s+(.*?)\s+```")
            if decision_content
            else None
        )
        if python_code is None:
            return False, "Missing executable python fenced block in ## Decision."

        normalized_code, normalize_error = self._normalize_generated_scene_code(
            python_code
        )
        if normalize_error:
            return False, normalize_error

        write_to_file(
            os.path.join(output_folder, f"{scenario_id}_scene_final.py"),
            normalized_code,
        )
        return True, None

    @staticmethod
    def _truncate_subprocess_output(output: str, limit: int = 1200) -> str:
        output = output.strip()
        if len(output) <= limit:
            return output
        return f"{output[:limit].rstrip()}..."

    def _normalize_generated_scene_code(
        self, python_code: str
    ) -> tuple[str | None, str | None]:
        normalized_code = python_code.strip()
        try:
            ast.parse(normalized_code)
        except SyntaxError as exc:
            return (
                None,
                "Generated scene python code has syntax errors before normalization: "
                f"{exc.msg} at line {exc.lineno}.",
            )

        normalized_code = self._replace_weather_calls(normalized_code)
        normalized_code = self._inject_helper_activation(normalized_code)
        normalized_code = self._inject_spectator_focus(normalized_code)
        if SCENE_HELPER_MARKER not in normalized_code:
            normalized_code = f"{self._build_scene_helper_block()}\n\n{normalized_code}"

        try:
            ast.parse(normalized_code)
        except SyntaxError as exc:
            return (
                None,
                "Generated scene python code has syntax errors after normalization: "
                f"{exc.msg} at line {exc.lineno}.",
            )

        return normalized_code, None

    def _replace_weather_calls(self, python_code: str) -> str:
        lines = python_code.splitlines()
        replaced_lines = []

        for line in lines:
            match = re.match(r"^(\s*)world\.set_weather\((.+)\)\s*$", line)
            if match:
                indent, weather_expr = match.groups()
                replaced_lines.append(
                    f"{indent}_autoscenario_apply_weather({weather_expr.strip()})"
                )
                continue
            replaced_lines.append(line)

        return "\n".join(replaced_lines)

    def _inject_helper_activation(self, python_code: str) -> str:
        if SCENE_ACTIVATION_MARKER in python_code:
            return python_code

        lines = python_code.splitlines()
        insert_index = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# Change the weather"):
                insert_index = index
                break
            if stripped.startswith("weather ="):
                insert_index = index
                break
            if stripped.startswith("# Spawn into Carla"):
                insert_index = index
                break
            if (
                ("spawn_vehicle(" in stripped and not stripped.startswith("def "))
                or (
                    "spawn_static_prop(" in stripped
                    and not stripped.startswith("def ")
                )
                or (
                    "spawn_pedestrian(" in stripped
                    and not stripped.startswith("def ")
                )
            ):
                insert_index = index
                break

        if insert_index is None:
            lines.extend(
                [
                    "",
                    SCENE_ACTIVATION_MARKER,
                    "spawn_static_prop = _autoscenario_spawn_static_prop",
                    "spawn_vehicle = _autoscenario_spawn_vehicle",
                    "spawn_pedestrian = _autoscenario_spawn_pedestrian",
                    '_autoscenario_apply_weather(globals().get("weather"))',
                ]
            )
        else:
            indent = lines[insert_index][: len(lines[insert_index]) - len(lines[insert_index].lstrip())]
            lines[insert_index:insert_index] = [
                "",
                f"{indent}{SCENE_ACTIVATION_MARKER}",
                f"{indent}spawn_static_prop = _autoscenario_spawn_static_prop",
                f"{indent}spawn_vehicle = _autoscenario_spawn_vehicle",
                f"{indent}spawn_pedestrian = _autoscenario_spawn_pedestrian",
                f'{indent}_autoscenario_apply_weather(globals().get("weather"))',
                "",
            ]

        return "\n".join(lines)

    def _inject_spectator_focus(self, python_code: str) -> str:
        if SCENE_SPECTATOR_MARKER in python_code:
            return python_code

        lines = python_code.splitlines()
        for index, line in enumerate(lines):
            if "time.sleep(" in line:
                indent = line[: len(line) - len(line.lstrip())]
                lines[index:index] = [
                    "",
                    f"{indent}{SCENE_SPECTATOR_MARKER}",
                    f"{indent}_autoscenario_focus_spectator()",
                ]
                return "\n".join(lines)

        lines.extend(["", SCENE_SPECTATOR_MARKER, "_autoscenario_focus_spectator()"])
        return "\n".join(lines)

    @staticmethod
    def _build_scene_helper_block() -> str:
        return textwrap.dedent(
            f"""
            {SCENE_HELPER_MARKER}
            import math
            import random

            _AUTOSCENARIO_SPAWNED_ACTORS = []
            _AUTOSCENARIO_FOCUS_POINTS = []

            _AUTOSCENARIO_VEHICLE_BLUEPRINTS = {{
                "bike": [
                    "vehicle.bh.crossbike",
                    "vehicle.diamondback.century",
                    "vehicle.gazelle.omafiets",
                ],
                "car": [
                    "vehicle.tesla.model3",
                    "vehicle.audi.a2",
                    "vehicle.citroen.c3",
                    "vehicle.mini.cooper_s",
                ],
                "jeep": [
                    "vehicle.jeep.wrangler_rubicon",
                    "vehicle.nissan.patrol",
                ],
                "motorcycle": [
                    "vehicle.harley-davidson.low_rider",
                    "vehicle.kawasaki.ninja",
                    "vehicle.yamaha.yzf",
                ],
                "suv": [
                    "vehicle.audi.etron",
                    "vehicle.nissan.patrol",
                    "vehicle.tesla.cybertruck",
                ],
                "truck": [
                    "vehicle.carlamotors.carlacola",
                    "vehicle.tesla.cybertruck",
                    "vehicle.ford.ambulance",
                ],
                "van": [
                    "vehicle.mercedes.sprinter",
                    "vehicle.ford.ambulance",
                ],
            }}

            _AUTOSCENARIO_STATIC_BLUEPRINTS = {{
                "warningconstruction": ["static.prop.warningconstruction"],
                "streetbarrier": ["static.prop.streetbarrier"],
                "constructioncone": ["static.prop.constructioncone"],
                "warningaccident": ["static.prop.warningaccident"],
            }}


            def _autoscenario_normalize_key(value):
                return "".join(ch for ch in str(value).lower() if ch.isalnum())


            def _autoscenario_to_location(location):
                if hasattr(location, "x") and hasattr(location, "y") and hasattr(location, "z"):
                    return carla.Location(float(location.x), float(location.y), float(location.z))
                if isinstance(location, (list, tuple)) and len(location) >= 3:
                    return carla.Location(float(location[0]), float(location[1]), float(location[2]))
                raise ValueError(f"Unsupported location value: {{location!r}}")


            def _autoscenario_to_rotation(rotation):
                if hasattr(rotation, "pitch") and hasattr(rotation, "yaw") and hasattr(rotation, "roll"):
                    return carla.Rotation(
                        float(rotation.pitch), float(rotation.yaw), float(rotation.roll)
                    )
                if hasattr(rotation, "x") and hasattr(rotation, "y") and hasattr(rotation, "z"):
                    return carla.Rotation(float(rotation.x), float(rotation.y), float(rotation.z))
                if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
                    return carla.Rotation(float(rotation[0]), float(rotation[1]), float(rotation[2]))
                return carla.Rotation()


            def _autoscenario_record_focus_point(location):
                try:
                    _AUTOSCENARIO_FOCUS_POINTS.append(_autoscenario_to_location(location))
                except Exception:
                    return


            def _autoscenario_pick_blueprint(category, semantic_name):
                normalized_name = _autoscenario_normalize_key(semantic_name)
                if category == "vehicle":
                    candidate_ids = list(_AUTOSCENARIO_VEHICLE_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = ["vehicle.*"]
                elif category == "static":
                    candidate_ids = list(_AUTOSCENARIO_STATIC_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = [
                        f"static.prop.*{{normalized_name}}*",
                        "static.prop.*",
                    ]
                else:
                    candidate_ids = []
                    fallback_patterns = []

                if semantic_name and "." in str(semantic_name):
                    candidate_ids.insert(0, str(semantic_name))
                elif semantic_name:
                    prefix = "vehicle." if category == "vehicle" else "static.prop."
                    candidate_ids.extend(
                        [
                            f"{{prefix}}{{semantic_name}}",
                            f"{{prefix}}{{normalized_name}}",
                        ]
                    )

                seen = set()
                for blueprint_id in candidate_ids:
                    if not blueprint_id or blueprint_id in seen:
                        continue
                    seen.add(blueprint_id)
                    try:
                        return blueprint_library.find(blueprint_id)
                    except Exception:
                        continue

                for pattern in fallback_patterns:
                    try:
                        matches = list(blueprint_library.filter(pattern))
                    except Exception:
                        matches = []
                    if matches:
                        return random.choice(matches)

                return None


            def _autoscenario_try_spawn(blueprint, location, rotation, actor_kind):
                if blueprint is None:
                    return None

                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                yaw_radians = math.radians(base_rotation.yaw)
                forward_x = math.cos(yaw_radians)
                forward_y = math.sin(yaw_radians)
                right_x = -math.sin(yaw_radians)
                right_y = math.cos(yaw_radians)

                if actor_kind == "vehicle":
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (4.0, 0.0, 0.35),
                        (-4.0, 0.0, 0.35),
                        (8.0, 0.0, 0.35),
                        (-8.0, 0.0, 0.35),
                        (0.0, 2.5, 0.35),
                        (0.0, -2.5, 0.35),
                    ]
                elif actor_kind == "walker":
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (1.0, 0.0, 0.35),
                        (-1.0, 0.0, 0.35),
                        (0.0, 1.0, 0.35),
                        (0.0, -1.0, 0.35),
                    ]
                else:
                    offsets = [
                        (0.0, 0.0, 0.35),
                        (0.8, 0.0, 0.35),
                        (-0.8, 0.0, 0.35),
                        (0.0, 0.8, 0.35),
                        (0.0, -0.8, 0.35),
                    ]

                for forward_offset, right_offset, dz in offsets:
                    try_location = carla.Location(
                        base_location.x
                        + forward_x * forward_offset
                        + right_x * right_offset,
                        base_location.y
                        + forward_y * forward_offset
                        + right_y * right_offset,
                        base_location.z + dz,
                    )
                    actor = world.try_spawn_actor(
                        blueprint, carla.Transform(try_location, base_rotation)
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    if actor_kind == "vehicle":
                        try:
                            actor.set_autopilot(False)
                        except Exception:
                            pass
                        try:
                            actor.apply_control(carla.VehicleControl(brake=1.0))
                        except Exception:
                            pass
                    elif actor_kind == "static":
                        try:
                            actor.set_simulate_physics(True)
                        except Exception:
                            pass
                    return actor

                return None


            def _autoscenario_apply_vehicle_color(blueprint, color):
                if blueprint is None or not color:
                    return
                try:
                    has_color = blueprint.has_attribute("color")
                except Exception:
                    has_color = False
                if not has_color:
                    return

                if isinstance(color, (list, tuple)) and len(color) >= 3:
                    color_value = ",".join(str(int(channel)) for channel in color[:3])
                else:
                    color_value = str(color).strip()
                if not color_value:
                    return

                try:
                    blueprint.set_attribute("color", color_value)
                except Exception:
                    return


            def _autoscenario_spawn_static_prop(blueprint_name, location, rotation):
                _autoscenario_record_focus_point(location)
                blueprint = _autoscenario_pick_blueprint("static", blueprint_name)
                return _autoscenario_try_spawn(
                    blueprint,
                    location,
                    rotation,
                    "static",
                )


            def _autoscenario_spawn_vehicle(blueprint_name, location, rotation, color=None):
                _autoscenario_record_focus_point(location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                return _autoscenario_try_spawn(
                    blueprint,
                    location,
                    rotation,
                    "vehicle",
                )


            def _autoscenario_spawn_pedestrian(location, rotation):
                _autoscenario_record_focus_point(location)
                try:
                    pedestrian_blueprints = list(
                        blueprint_library.filter("walker.pedestrian.*")
                    )
                except Exception:
                    pedestrian_blueprints = []
                if not pedestrian_blueprints:
                    return None
                return _autoscenario_try_spawn(
                    random.choice(pedestrian_blueprints),
                    location,
                    rotation,
                    "walker",
                )


            def _autoscenario_default_weather():
                try:
                    return carla.WeatherParameters.ClearNoon
                except Exception:
                    return carla.WeatherParameters()


            def _autoscenario_apply_weather(weather_value=None):
                if weather_value is None:
                    world.set_weather(_autoscenario_default_weather())
                    return

                if isinstance(weather_value, str):
                    preset = getattr(carla.WeatherParameters, weather_value, None)
                    if preset is not None:
                        world.set_weather(preset)
                        return
                    world.set_weather(_autoscenario_default_weather())
                    return

                if isinstance(weather_value, dict):
                    weather = carla.WeatherParameters()
                    for key, value in weather_value.items():
                        if hasattr(weather, key):
                            setattr(weather, key, value)
                    world.set_weather(weather)
                    return

                world.set_weather(weather_value)


            def _autoscenario_focus_spectator():
                try:
                    spectator = world.get_spectator()
                except Exception:
                    return

                focus_locations = []
                for actor in _AUTOSCENARIO_SPAWNED_ACTORS:
                    if actor is None:
                        continue
                    try:
                        focus_locations.append(actor.get_transform().location)
                    except Exception:
                        continue

                if not focus_locations:
                    focus_locations = list(_AUTOSCENARIO_FOCUS_POINTS)
                if not focus_locations:
                    return

                avg_x = sum(location.x for location in focus_locations) / len(focus_locations)
                avg_y = sum(location.y for location in focus_locations) / len(focus_locations)
                avg_z = sum(location.z for location in focus_locations) / len(focus_locations)
                spectator_location = carla.Location(avg_x, avg_y, avg_z + 18.0)
                spectator_rotation = carla.Rotation(pitch=-55.0, yaw=0.0, roll=0.0)

                try:
                    spectator.set_transform(
                        carla.Transform(spectator_location, spectator_rotation)
                    )
                except Exception:
                    return
            """
        ).strip()


if __name__ == "__main__":
    scenegen = ScenarioGenerator()
