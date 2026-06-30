import json
import os
import re
import textwrap
from typing import Any, Dict, Optional

from agents.obstacle_generator import OBJECT_INFO_PREFIX
from agents.scenario_generator import ScenarioGenerator
from opendrive_experiment.tools.structured_pipeline import build_projected_spawn_payload
from tools.utils import extract_text_section, read_file, write_to_file


XODR_WORLD_MARKER = "# __AUTOSCENARIO_XODR_WORLD__"


class XODRScenarioGenerator(ScenarioGenerator):
    """Scenario generator that injects XODR world bootstrapping into final scripts."""

    def build_scene_script_from_projected_coordinates(
        self,
        scene_id: str,
        output_folder: str,
        projected_coordinates: Dict[str, Any],
        projected_coordinates_path: str,
        xodr_path: str,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = build_projected_spawn_payload(projected_coordinates)
        write_to_file(
            projected_coordinates_path,
            json.dumps(projected_coordinates, indent=2, sort_keys=True),
        )
        spawn_payload_path = os.path.join(
            output_folder, f"{scene_id}_spawn_entities.json"
        )
        write_to_file(
            spawn_payload_path,
            json.dumps(payload, indent=2, sort_keys=True),
        )

        script = self._build_deterministic_scene_script(
            spawn_payload_filename=os.path.basename(spawn_payload_path),
            xodr_filename=os.path.basename(xodr_path),
            generation_params=generation_params or {},
        )
        output_path = os.path.join(output_folder, f"{scene_id}_scene_final_xodr.py")
        write_to_file(output_path, script)
        return output_path

    @staticmethod
    def _build_scene_helper_block() -> str:
        base_block = ScenarioGenerator._build_scene_helper_block()
        xodr_block = textwrap.dedent(
            """


            def _autoscenario_blueprint_matches_category(blueprint, category, semantic_name):
                if blueprint is None or category != "vehicle":
                    return blueprint is not None

                normalized_name = _autoscenario_normalize_key(semantic_name)
                try:
                    if blueprint.has_attribute("number_of_wheels"):
                        wheels_attr = blueprint.get_attribute("number_of_wheels")
                        wheels = int(getattr(wheels_attr, "as_int", lambda: wheels_attr)())
                    else:
                        wheels = None
                except Exception:
                    wheels = None

                if wheels is None:
                    return True
                if normalized_name in {"bike", "motorcycle"}:
                    return wheels <= 2
                return wheels >= 4


            def _autoscenario_pick_blueprint(category, semantic_name):
                normalized_name = _autoscenario_normalize_key(semantic_name)
                if category == "vehicle":
                    candidate_ids = list(_AUTOSCENARIO_VEHICLE_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = ["vehicle.*"]
                elif category == "static":
                    candidate_ids = list(_AUTOSCENARIO_STATIC_BLUEPRINTS.get(normalized_name, []))
                    fallback_patterns = [
                        f"static.prop.*{normalized_name}*",
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
                            f"{prefix}{semantic_name}",
                            f"{prefix}{normalized_name}",
                        ]
                    )

                seen = set()
                for blueprint_id in candidate_ids:
                    if not blueprint_id or blueprint_id in seen:
                        continue
                    seen.add(blueprint_id)
                    try:
                        blueprint = blueprint_library.find(blueprint_id)
                    except Exception:
                        continue
                    if _autoscenario_blueprint_matches_category(
                        blueprint, category, semantic_name
                    ):
                        return blueprint

                for pattern in fallback_patterns:
                    try:
                        matches = list(blueprint_library.filter(pattern))
                    except Exception:
                        matches = []
                    filtered = [
                        blueprint
                        for blueprint in matches
                        if _autoscenario_blueprint_matches_category(
                            blueprint, category, semantic_name
                        )
                    ]
                    if filtered:
                        return random.choice(filtered)

                return None


            def _autoscenario_normalize_yaw(yaw_value):
                while yaw_value <= -180.0:
                    yaw_value += 360.0
                while yaw_value > 180.0:
                    yaw_value -= 360.0
                return yaw_value


            def _autoscenario_angle_distance(yaw_a, yaw_b):
                return abs(_autoscenario_normalize_yaw(yaw_a - yaw_b))


            def _autoscenario_project_vehicle_to_lane(location, rotation):
                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                try:
                    world_map = world.get_map()
                except Exception:
                    return base_location, base_rotation

                waypoint = None
                lane_type = getattr(carla.LaneType, "Driving", None)
                try:
                    if lane_type is not None:
                        waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                            lane_type=lane_type,
                        )
                    else:
                        waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                        )
                except TypeError:
                    try:
                        waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        waypoint = None
                except Exception:
                    waypoint = None

                if waypoint is None:
                    return base_location, base_rotation

                snapped_location = waypoint.transform.location
                waypoint_yaw = float(waypoint.transform.rotation.yaw)
                candidate_yaws = [waypoint_yaw, waypoint_yaw + 180.0]
                snapped_yaw = min(
                    candidate_yaws,
                    key=lambda yaw_value: _autoscenario_angle_distance(
                        yaw_value,
                        float(base_rotation.yaw),
                    ),
                )

                projected_location = carla.Location(
                    snapped_location.x,
                    snapped_location.y,
                    snapped_location.z + 0.35,
                )
                projected_rotation = carla.Rotation(
                    float(base_rotation.pitch),
                    _autoscenario_normalize_yaw(snapped_yaw),
                    float(base_rotation.roll),
                )
                return projected_location, projected_rotation


            def _autoscenario_collect_vehicle_spawn_candidates(location, rotation):
                base_location = _autoscenario_to_location(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                try:
                    world_map = world.get_map()
                except Exception:
                    return [(base_location, base_rotation)]

                lane_type = getattr(carla.LaneType, "Driving", None)
                try:
                    if lane_type is not None:
                        base_waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                            lane_type=lane_type,
                        )
                    else:
                        base_waypoint = world_map.get_waypoint(
                            base_location,
                            project_to_road=True,
                        )
                except TypeError:
                    try:
                        base_waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        base_waypoint = None
                except Exception:
                    base_waypoint = None

                if base_waypoint is None:
                    return [(base_location, base_rotation)]

                def _candidate_from_waypoint(waypoint):
                    location = waypoint.transform.location
                    waypoint_yaw = float(waypoint.transform.rotation.yaw)
                    candidate_yaws = [waypoint_yaw, waypoint_yaw + 180.0]
                    snapped_yaw = min(
                        candidate_yaws,
                        key=lambda yaw_value: _autoscenario_angle_distance(
                            yaw_value,
                            float(base_rotation.yaw),
                        ),
                    )
                    return (
                        carla.Location(location.x, location.y, location.z + 0.6),
                        carla.Rotation(
                            float(base_rotation.pitch),
                            _autoscenario_normalize_yaw(snapped_yaw),
                            float(base_rotation.roll),
                        ),
                    )

                candidates = []
                seen = set()

                def _append_waypoint(waypoint):
                    if waypoint is None:
                        return
                    key = (
                        round(float(waypoint.transform.location.x), 2),
                        round(float(waypoint.transform.location.y), 2),
                        int(waypoint.road_id),
                        int(waypoint.lane_id),
                    )
                    if key in seen:
                        return
                    seen.add(key)
                    candidates.append(_candidate_from_waypoint(waypoint))

                _append_waypoint(base_waypoint)

                for distance in (8.0, 16.0, 24.0, 32.0):
                    try:
                        next_waypoints = base_waypoint.next(distance)
                    except Exception:
                        next_waypoints = []
                    for waypoint in next_waypoints or []:
                        _append_waypoint(waypoint)

                    previous_method = getattr(base_waypoint, "previous", None)
                    if previous_method is None:
                        continue
                    try:
                        previous_waypoints = previous_method(distance)
                    except Exception:
                        previous_waypoints = []
                    for waypoint in previous_waypoints or []:
                        _append_waypoint(waypoint)

                if not candidates:
                    candidates.append((base_location, base_rotation))
                return candidates


            def _autoscenario_try_spawn_vehicle_actor(blueprint, location, rotation):
                if blueprint is None:
                    return None

                for candidate_location, candidate_rotation in _autoscenario_collect_vehicle_spawn_candidates(
                    location,
                    rotation,
                ):
                    actor = world.try_spawn_actor(
                        blueprint,
                        carla.Transform(candidate_location, candidate_rotation),
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    try:
                        actor.set_autopilot(False)
                    except Exception:
                        pass
                    try:
                        actor.apply_control(carla.VehicleControl(brake=1.0))
                    except Exception:
                        pass
                    return actor
                return None


            def _autoscenario_project_static_to_ground(location):
                base_location = _autoscenario_to_location(location)
                ground_z = max(0.02, float(base_location.z))
                try:
                    world_map = world.get_map()
                    waypoint = world_map.get_waypoint(
                        base_location,
                        project_to_road=True,
                    )
                except TypeError:
                    try:
                        waypoint = world_map.get_waypoint(base_location, True)
                    except Exception:
                        waypoint = None
                except Exception:
                    waypoint = None

                if waypoint is not None:
                    ground_z = max(0.02, float(waypoint.transform.location.z) + 0.02)
                return carla.Location(float(base_location.x), float(base_location.y), ground_z)


            def _autoscenario_try_spawn_static(blueprint, location, rotation):
                if blueprint is None:
                    return None

                base_location = _autoscenario_project_static_to_ground(location)
                base_rotation = _autoscenario_to_rotation(rotation)
                offsets = [
                    (0.0, 0.0, 0.0),
                    (0.3, 0.0, 0.0),
                    (-0.3, 0.0, 0.0),
                    (0.0, 0.3, 0.0),
                    (0.0, -0.3, 0.0),
                ]

                for dx, dy, dz in offsets:
                    try_location = carla.Location(
                        base_location.x + dx,
                        base_location.y + dy,
                        base_location.z + dz,
                    )
                    actor = world.try_spawn_actor(
                        blueprint, carla.Transform(try_location, base_rotation)
                    )
                    if actor is None:
                        continue
                    _AUTOSCENARIO_SPAWNED_ACTORS.append(actor)
                    try:
                        actor.set_simulate_physics(True)
                    except Exception:
                        pass
                    return actor
                return None


            def _autoscenario_spawn_static_prop(blueprint_name, location, rotation):
                snapped_location = _autoscenario_project_static_to_ground(location)
                _autoscenario_record_focus_point(snapped_location)
                blueprint = _autoscenario_pick_blueprint("static", blueprint_name)
                return _autoscenario_try_spawn_static(
                    blueprint,
                    snapped_location,
                    rotation,
                )


            def _autoscenario_spawn_vehicle(blueprint_name, location, rotation, color=None):
                snapped_location, snapped_rotation = _autoscenario_project_vehicle_to_lane(location, rotation)
                _autoscenario_record_focus_point(snapped_location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                return _autoscenario_try_spawn_vehicle_actor(
                    blueprint,
                    snapped_location,
                    snapped_rotation,
                )


            def _autoscenario_spawn_vehicle_direct(blueprint_name, location, rotation, color=None):
                _autoscenario_record_focus_point(location)
                blueprint = _autoscenario_pick_blueprint("vehicle", blueprint_name)
                _autoscenario_apply_vehicle_color(blueprint, color)
                return _autoscenario_try_spawn(
                    blueprint,
                    location,
                    rotation,
                    "vehicle",
                )
            """
        )
        return base_block + xodr_block

    @staticmethod
    def _apply_coordinate_negation(stdout_text: str) -> str:
        """Negate Y-coordinate and yaw for every entity in the AUTOSCENARIO payload.

        The XODRObstacleGenerator no longer asks the LLM to perform this
        transformation.  Instead, we apply it once here, deterministically,
        to the raw coordinates the LLM outputs.

        CARLA uses a left-handed coordinate system where Y increases to the
        right; the DSL/image coordinate system has Y increasing upward.
        Negating Y (and the yaw heading) converts between them.
        """
        lines = stdout_text.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith(OBJECT_INFO_PREFIX):
                continue
            payload_text = line[len(OBJECT_INFO_PREFIX):]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                return stdout_text  # Can't parse — leave unchanged
            for collection in (
                payload.get("agent_dict", {}),
                payload.get("object_dict", {}),
            ):
                for entity in collection.values():
                    loc = entity.get("location", [])
                    if len(loc) == 3:
                        entity["location"] = [loc[0], -float(loc[1]), loc[2]]
                    rot = entity.get("rotation", [])
                    if len(rot) == 3:
                        entity["rotation"] = [rot[0], -float(rot[1]), rot[2]]
            lines[i] = OBJECT_INFO_PREFIX + json.dumps(payload, sort_keys=True)
            return "\n".join(lines)
        return stdout_text

    def run_scenario_generation(
        self,
        obj_code_fn,
        scenario_id,
        scenario_description,
        output_folder,
        xodr_path,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        try:
            result = __import__("subprocess").run(
                [__import__("sys").executable, obj_code_fn],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(obj_code_fn) or None,
            )
        except __import__("subprocess").TimeoutExpired as exc:
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

        obj_dict_str = self._apply_coordinate_negation(obj_dict_str)

        result_fn = obj_code_fn.replace(".py", "_obj.txt")
        write_to_file(result_fn, obj_dict_str)
        self.call_agent(
            scenario_description,
            scenario_id,
            output_folder,
            obj_dict_str,
            xodr_path=xodr_path,
            generation_params=generation_params,
        )

    def call_agent(
        self,
        user_request,
        scenario_id,
        output_folder,
        object_info,
        xodr_path,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        success_bool = False
        attempts = 0
        output_file = os.path.join(output_folder, f"{scenario_id}_scene_final.txt")
        while not success_bool:
            self.send_request(
                user_request,
                {
                    "output_fn": output_file,
                    "object_info": object_info,
                },
            )
            success_bool, error_message = self.extract_decision_data(
                scenario_id,
                output_folder=output_folder,
                xodr_path=xodr_path,
                generation_params=generation_params,
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

    def extract_decision_data(
        self,
        scenario_id,
        output_folder=None,
        response=None,
        xodr_path: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ):
        if response is not None:
            text = response
        elif output_folder is None:
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
        if not xodr_path:
            return False, "Missing xodr_path for OpenDRIVE scene generation."

        normalized_code, normalize_error = self._normalize_generated_scene_code(
            python_code
        )
        if normalize_error:
            return False, normalize_error

        normalized_code = self._inject_xodr_world_bootstrap(
            normalized_code, xodr_path, generation_params or {}
        )

        write_to_file(
            os.path.join(output_folder, f"{scenario_id}_scene_final_xodr.py"),
            normalized_code,
        )
        return True, None

    def _inject_xodr_world_bootstrap(
        self,
        python_code: str,
        xodr_path: str,
        generation_params: Dict[str, Any],
    ) -> str:
        code = python_code
        if XODR_WORLD_MARKER not in code:
            code = (
                f"{self._build_xodr_loader_block(os.path.basename(xodr_path), generation_params)}\n\n"
                f"{code}"
            )

        lines = code.splitlines()
        replaced = False
        world_patterns = (
            r"^(\s*)world\s*=\s*client\.get_world\(\)\s*$",
            r"^(\s*)world\s*=\s*client\.load_world\(.*\)\s*$",
        )
        for index, line in enumerate(lines):
            for pattern in world_patterns:
                match = re.match(pattern, line)
                if match:
                    indent = match.group(1)
                    lines[index] = f"{indent}world = _autoscenario_load_xodr_world(client)"
                    replaced = True
                    break
            if replaced:
                break

        if not replaced:
            for index, line in enumerate(lines):
                if re.match(r"^\s*blueprint_library\s*=", line):
                    indent = line[: len(line) - len(line.lstrip())]
                    lines[index:index] = [
                        "",
                        f"{indent}world = _autoscenario_load_xodr_world(client)",
                    ]
                    replaced = True
                    break

        if not replaced:
            lines.extend(["", "world = _autoscenario_load_xodr_world(client)"])

        return "\n".join(lines)

    @staticmethod
    def _build_xodr_loader_block(
        xodr_filename: str, generation_params: Dict[str, Any]
    ) -> str:
        params = {
            "vertex_distance": float(generation_params.get("vertex_distance", 2.0)),
            "max_road_length": float(generation_params.get("max_road_length", 50.0)),
            "wall_height": float(generation_params.get("wall_height", 0.0)),
            "additional_width": float(generation_params.get("additional_width", 1.5)),
            "smooth_junctions": bool(generation_params.get("smooth_junctions", True)),
            "enable_mesh_visibility": bool(
                generation_params.get("enable_mesh_visibility", True)
            ),
        }
        return f"""{XODR_WORLD_MARKER}
import os

_AUTOSCENARIO_XODR_FILENAME = {xodr_filename!r}
_AUTOSCENARIO_XODR_GENERATION_PARAMS = {params!r}


def _autoscenario_load_xodr_world(client):
    xodr_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_XODR_FILENAME)
    with open(xodr_path, "r", encoding="utf-8") as file:
        xodr_content = file.read()
    if not xodr_content.strip():
        raise RuntimeError(f"OpenDRIVE file is empty: {{xodr_path}}")
    if not hasattr(carla, "OpendriveGenerationParameters"):
        raise RuntimeError("Current CARLA Python API does not expose OpendriveGenerationParameters.")

    params = carla.OpendriveGenerationParameters(
        vertex_distance=float(_AUTOSCENARIO_XODR_GENERATION_PARAMS["vertex_distance"]),
        max_road_length=float(_AUTOSCENARIO_XODR_GENERATION_PARAMS["max_road_length"]),
        wall_height=float(_AUTOSCENARIO_XODR_GENERATION_PARAMS["wall_height"]),
        additional_width=float(_AUTOSCENARIO_XODR_GENERATION_PARAMS["additional_width"]),
        smooth_junctions=bool(_AUTOSCENARIO_XODR_GENERATION_PARAMS["smooth_junctions"]),
        enable_mesh_visibility=bool(_AUTOSCENARIO_XODR_GENERATION_PARAMS["enable_mesh_visibility"]),
    )
    world = client.generate_opendrive_world(xodr_content, params)
    try:
        world.wait_for_tick()
    except Exception:
        time.sleep(1.0)
    return world"""

    @staticmethod
    def _build_spawn_payload_loop() -> str:
        return (
            "def _autoscenario_load_spawn_payload():\n"
            "    payload_path = os.path.join(os.path.dirname(__file__), _AUTOSCENARIO_SPAWN_PAYLOAD)\n"
            "    with open(payload_path, 'r', encoding='utf-8') as file:\n"
            "        return json.load(file)\n\n"
            "def _autoscenario_payload_ego_yaw(spawn_payload):\n"
            "    for payload_entity in spawn_payload.get('entities', []):\n"
            "        if str(payload_entity.get('id')) in {'ego', 'ego_vehicle'}:\n"
            "            try:\n"
            "                return float((payload_entity.get('rotation') or {}).get('yaw'))\n"
            "            except Exception:\n"
            "                return None\n"
            "    return None\n\n"
            "def _autoscenario_apply_heading_relation(entity, rotation, ego_yaw):\n"
            "    if ego_yaw is None:\n"
            "        return rotation\n"
            "    relation = str(entity.get('heading_relation') or 'unknown')\n"
            "    yaw = float(rotation.yaw)\n"
            "    if relation == 'opposite_direction' and _autoscenario_angle_distance(yaw, ego_yaw) < 90.0:\n"
            "        rotation.yaw = _autoscenario_normalize_yaw(yaw + 180.0)\n"
            "    elif relation == 'same_direction' and _autoscenario_angle_distance(yaw, ego_yaw) > 90.0:\n"
            "        rotation.yaw = _autoscenario_normalize_yaw(yaw + 180.0)\n"
            "    return rotation\n\n"
            "def _autoscenario_clear_existing_vehicles():\n"
            "    try:\n"
            "        existing_vehicles = list(world.get_actors().filter('vehicle.*'))\n"
            "    except Exception:\n"
            "        existing_vehicles = []\n"
            "    for actor in existing_vehicles:\n"
            "        try:\n"
            "            actor.destroy()\n"
            "        except Exception:\n"
            "            pass\n"
            "    if existing_vehicles:\n"
            "        try:\n"
            "            world.wait_for_tick()\n"
            "        except Exception:\n"
            "            time.sleep(0.5)\n\n"
            "_autoscenario_apply_weather(carla.WeatherParameters.ClearNoon)\n"
            "_autoscenario_clear_existing_vehicles()\n"
            "spawn_payload = _autoscenario_load_spawn_payload()\n"
            "_autoscenario_ego_yaw = _autoscenario_payload_ego_yaw(spawn_payload)\n"
            "for entity in spawn_payload.get('entities', []):\n"
            "    location = carla.Location(\n"
            "        x=float(entity['location']['x']),\n"
            "        y=float(entity['location']['y']),\n"
            "        z=float(entity['location']['z']),\n"
            "    )\n"
            "    rotation = carla.Rotation(\n"
            "        pitch=float(entity['rotation']['pitch']),\n"
            "        yaw=float(entity['rotation']['yaw']),\n"
            "        roll=float(entity['rotation']['roll']),\n"
            "    )\n"
            "    rotation = _autoscenario_apply_heading_relation(entity, rotation, _autoscenario_ego_yaw)\n"
            "    spawn_kind = str(entity.get('spawn_kind') or 'vehicle')\n"
            "    if spawn_kind == 'pedestrian':\n"
            "        _autoscenario_spawn_pedestrian(location, rotation)\n"
            "        continue\n"
            "    blueprint_name = entity.get('blueprint_name')\n"
            "    if spawn_kind == 'static':\n"
            "        _autoscenario_spawn_static_prop(blueprint_name, location, rotation)\n"
            "        continue\n"
            "    placement_mode = str(entity.get('placement_mode') or 'project_to_lane')\n"
            "    if placement_mode == 'direct':\n"
            "        _autoscenario_spawn_vehicle_direct(\n"
            "            blueprint_name,\n"
            "            location,\n"
            "            rotation,\n"
            "            entity.get('color'),\n"
            "        )\n"
            "        continue\n"
            "    _autoscenario_spawn_vehicle(\n"
            "        blueprint_name,\n"
            "        location,\n"
            "        rotation,\n"
            "        entity.get('color'),\n"
            "    )\n\n"
            "_autoscenario_focus_spectator()\n"
            "time.sleep(2)\n"
        )

    def _build_deterministic_scene_script(
        self,
        spawn_payload_filename: str,
        xodr_filename: str,
        generation_params: Dict[str, Any],
    ) -> str:
        loader_block = self._build_xodr_loader_block(
            xodr_filename,
            generation_params,
        )
        helper_block = self._build_scene_helper_block()
        return (
            f"{loader_block}\n\n"
            f"{helper_block}\n\n"
            "import json\n"
            "import os\n"
            "import carla\n"
            "import time\n\n"
            f"_AUTOSCENARIO_SPAWN_PAYLOAD = {spawn_payload_filename!r}\n\n"
            "client = carla.Client('localhost', 2000)\n"
            "client.set_timeout(10.0)\n"
            "world = _autoscenario_load_xodr_world(client)\n"
            "blueprint_library = world.get_blueprint_library()\n\n"
            f"{self._build_spawn_payload_loop()}"
        )

    def build_existing_world_scene_script(
        self,
        spawn_payload_filename: str,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        carla_map: Optional[str] = None,
        scene_match_status: Optional[str] = None,
        scene_match_reason: Optional[str] = None,
    ) -> str:
        helper_block = self._build_scene_helper_block()
        status_comment = f"# scene_match_status: {scene_match_status or 'unknown'}"
        if scene_match_reason:
            status_comment += f"; reason: {scene_match_reason}"
        world_loader = (
            f"world = client.load_world({carla_map!r})\n"
            if carla_map
            else "world = client.get_world()\n"
        )
        return (
            f"{status_comment}\n"
            f"{helper_block}\n\n"
            "import json\n"
            "import os\n"
            "import carla\n"
            "import time\n\n"
            f"_AUTOSCENARIO_SPAWN_PAYLOAD = {spawn_payload_filename!r}\n\n"
            f"client = carla.Client({carla_host!r}, {int(carla_port)})\n"
            "client.set_timeout(10.0)\n"
            f"{world_loader}"
            "try:\n"
            "    world.wait_for_tick()\n"
            "except Exception:\n"
            "    time.sleep(1.0)\n"
            "blueprint_library = world.get_blueprint_library()\n\n"
            f"{self._build_spawn_payload_loop()}"
        )
