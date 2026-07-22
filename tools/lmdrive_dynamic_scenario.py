"""ScenarioRunner implementation for AutoScenario dynamic XML files."""

from __future__ import print_function

import json
import math
import os
import time
import weakref
import xml.etree.ElementTree as ET

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (
    ActorSpeedAboveThresholdTest,
    CollisionTest,
    InRouteTest,
    OutsideRouteLanesTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
)
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario
from agents.navigation.local_planner import RoadOption


def _speed(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _set_forward_speed(actor, speed_mps):
    try:
        actor.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False)
        )
    except Exception:
        pass
    transform = actor.get_transform()
    radians = math.radians(transform.rotation.yaw)
    actor.set_target_velocity(
        carla.Vector3D(
            math.cos(radians) * max(0.0, speed_mps),
            math.sin(radians) * max(0.0, speed_mps),
            0.0,
        )
    )


def _load_extension(scenario_name):
    config_path = os.environ.get("AUTOSCENARIO_CONFIG_XML")
    if not config_path:
        raise RuntimeError("AUTOSCENARIO_CONFIG_XML must point to the generated XML")
    root = ET.parse(config_path).getroot()
    scenario = next(
        (
            node
            for node in root.iter("scenario")
            if node.attrib.get("name") == scenario_name
        ),
        None,
    )
    if scenario is None:
        raise RuntimeError("Scenario {!r} not found in {}".format(scenario_name, config_path))
    extension = scenario.find("autoscenario")
    if extension is None:
        raise RuntimeError("Generated XML has no <autoscenario> extension")
    actor_ids = [node.attrib.get("id", "") for node in scenario.iter("other_actor")]
    ego_controller = extension.find("ego_controller")
    spectator = extension.find("spectator")
    leaderboard = extension.find("leaderboard")
    notices = [dict(node.attrib) for node in extension.iter("notice")]
    events = [dict(node.attrib) for node in extension.iter("event")]
    trajectories = []
    for node in extension.iter("trajectory"):
        trajectories.append(
            {
                "actor_id": node.attrib.get("actor_id", ""),
                "keyframes": [dict(item.attrib) for item in node.iter("keyframe")],
            }
        )
    return {
        "scene_id": extension.attrib.get("scene_id", scenario_name),
        "duration_s": float(extension.attrib.get("duration_s", 12.0)),
        "metrics_output": extension.attrib.get("metrics_output"),
        "source_spawn_payload": extension.attrib.get("source_spawn_payload"),
        "actor_ids": actor_ids,
        "events": events,
        "trajectories": trajectories,
        "ego_controller": dict(ego_controller.attrib) if ego_controller is not None else {},
        "spectator": dict(spectator.attrib) if spectator is not None else {},
        "leaderboard": dict(leaderboard.attrib) if leaderboard is not None else {},
        "notices": notices,
    }


class _DynamicTimeline(AtomicBehavior):
    def __init__(self, ego, actors_by_id, spec):
        super(_DynamicTimeline, self).__init__("AutoScenario dynamic timeline")
        self._ego = ego
        self._actors_by_id = actors_by_id
        self._spec = spec
        self._start_time = None
        self._min_distance = float("inf")
        self._min_ttc = float("inf")
        self._collision_actor_ids = []
        self._collision_sensor = None
        self._trajectory_origins = {}
        self._spectator = ego.get_world().get_spectator()
        self._realtime = os.environ.get("AUTOSCENARIO_REALTIME", "1") != "0"
        self._finalized = False
        self._event_window_closed = False

    def initialise(self):
        blueprint = self._ego.get_world().get_blueprint_library().find(
            "sensor.other.collision"
        )
        self._collision_sensor = self._ego.get_world().spawn_actor(
            blueprint, carla.Transform(), attach_to=self._ego
        )
        self._collision_sensor.listen(
            lambda event: self._on_collision(weakref.ref(self), event)
        )
        for trajectory in self._spec.get("trajectories") or []:
            actor = self._actors_by_id.get(trajectory.get("actor_id"))
            if actor is not None:
                self._trajectory_origins[trajectory["actor_id"]] = actor.get_transform()
        super(_DynamicTimeline, self).initialise()

    def _takeover_ready(self):
        controller = self._spec.get("ego_controller") or {}
        if controller.get("mode", "lmdrive_agent") != "lmdrive_agent":
            return True
        try:
            return bool(
                py_trees.blackboard.Blackboard().get(
                    "AutoScenarioLMDriveTakeover"
                )
            )
        except (AttributeError, KeyError):
            return False

    def _begin_evaluation(self):
        self._start_time = GameTime.get_time()
        for event in self._spec["events"]:
            actor = self._actors_by_id.get(event.get("actor_id"))
            if actor is None or event.get("action") != "decelerate_to_speed":
                continue
            _set_forward_speed(actor, float(event.get("initial_speed_mps", 4.0)))
        print(
            "LMDrive has control; dynamic event timeline and route evaluation started",
            flush=True,
        )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is not None and event.other_actor is not None:
            actor_id = int(event.other_actor.id)
            if actor_id not in self._collision_actor_ids:
                self._collision_actor_ids.append(actor_id)

    def _apply_event(self, event, elapsed):
        actor = self._actors_by_id.get(event.get("actor_id"))
        if actor is None:
            return
        start_s = float(event.get("start_s", 0.0))
        duration_s = float(event.get("active_duration_s", self._spec["duration_s"]))
        local_time = elapsed - start_s
        if local_time < 0.0 or local_time > duration_s:
            return
        action = event.get("action")
        lights = set(filter(None, event.get("lights", "").split(",")))
        if lights and hasattr(carla, "VehicleLightState"):
            state = carla.VehicleLightState.NONE
            for name, attribute in (
                ("brake", "Brake"),
                ("left_blinker", "LeftBlinker"),
                ("right_blinker", "RightBlinker"),
            ):
                if name in lights and hasattr(carla.VehicleLightState, attribute):
                    state |= getattr(carla.VehicleLightState, attribute)
            actor.set_light_state(carla.VehicleLightState(state))
        if action == "set_speed":
            _set_forward_speed(actor, float(event.get("speed_mps", 0.0)))
        elif action == "steer":
            actor.apply_control(
                carla.VehicleControl(
                    throttle=float(event.get("throttle", 0.0)),
                    steer=float(event.get("steer", 0.0)),
                    brake=float(event.get("brake", 0.0)),
                )
            )
        elif action == "decelerate_to_speed":
            initial = float(event.get("initial_speed_mps", 4.0))
            target = float(event.get("target_speed_mps", 0.0))
            decel = float(event.get("max_decel_mps2", 1.5))
            _set_forward_speed(actor, max(target, initial - decel * local_time))
        elif action in ("stop", "hold_position"):
            _set_forward_speed(actor, 0.0)
            actor.apply_control(
                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=event.get("hand_brake", "false").lower() == "true",
                )
            )

    @staticmethod
    def _interpolated_keyframe(keyframes, elapsed):
        if not keyframes:
            return None
        before = keyframes[0]
        after = keyframes[-1]
        for keyframe in keyframes:
            if float(keyframe.get("t_s", 0.0)) <= elapsed:
                before = keyframe
            if float(keyframe.get("t_s", 0.0)) >= elapsed:
                after = keyframe
                break
        start = float(before.get("t_s", 0.0))
        end = float(after.get("t_s", start))
        alpha = 0.0 if end <= start else min(1.0, max(0.0, (elapsed - start) / (end - start)))
        result = {}
        for key in ("forward_m", "right_m", "heading_delta_deg", "speed_mps"):
            first = float(before.get(key, 0.0))
            second = float(after.get(key, first))
            result[key] = first + (second - first) * alpha
        result["lights"] = before.get("lights", "")
        return result

    def _apply_trajectory(self, trajectory, elapsed):
        actor_id = trajectory.get("actor_id")
        actor = self._actors_by_id.get(actor_id)
        origin = self._trajectory_origins.get(actor_id)
        frame = self._interpolated_keyframe(trajectory.get("keyframes") or [], elapsed)
        if actor is None or origin is None or frame is None:
            return
        yaw = math.radians(origin.rotation.yaw)
        forward = frame["forward_m"]
        right = frame["right_m"]
        location = carla.Location(
            x=origin.location.x + math.cos(yaw) * forward - math.sin(yaw) * right,
            y=origin.location.y + math.sin(yaw) * forward + math.cos(yaw) * right,
            z=origin.location.z,
        )
        rotation = carla.Rotation(
            pitch=origin.rotation.pitch,
            yaw=origin.rotation.yaw + frame["heading_delta_deg"],
            roll=origin.rotation.roll,
        )
        actor.set_transform(carla.Transform(location, rotation))
        _set_forward_speed(actor, frame["speed_mps"])
        lights = set(filter(None, frame.get("lights", "").split(",")))
        if lights and hasattr(carla, "VehicleLightState"):
            state = carla.VehicleLightState.NONE
            light_names = {
                "brake": "Brake",
                "left_blinker": "LeftBlinker",
                "right_blinker": "RightBlinker",
            }
            for name in lights:
                attribute = light_names.get(name)
                if attribute and hasattr(carla.VehicleLightState, attribute):
                    state |= getattr(carla.VehicleLightState, attribute)
            actor.set_light_state(carla.VehicleLightState(state))

    def _sample_metrics(self):
        ego_location = self._ego.get_location()
        ego_velocity = self._ego.get_velocity()
        for actor in self._actors_by_id.values():
            actor_location = actor.get_location()
            dx = actor_location.x - ego_location.x
            dy = actor_location.y - ego_location.y
            distance = math.sqrt(dx * dx + dy * dy)
            self._min_distance = min(self._min_distance, distance)
            if distance <= 1e-3:
                continue
            actor_velocity = actor.get_velocity()
            closing_speed = (
                (ego_velocity.x - actor_velocity.x) * dx
                + (ego_velocity.y - actor_velocity.y) * dy
            ) / distance
            if closing_speed > 1e-3:
                self._min_ttc = min(self._min_ttc, distance / closing_speed)

    def update(self):
        if self._start_time is None:
            if self._takeover_ready():
                self._begin_evaluation()
            else:
                self._update_spectator()
                if self._realtime:
                    time.sleep(0.05)
                return py_trees.common.Status.RUNNING
        elapsed = GameTime.get_time() - self._start_time
        for event in self._spec["events"]:
            # Ego is intentionally owned by autonomous driving in this run.
            if event.get("actor_id") not in ("ego", "ego_vehicle"):
                self._apply_event(event, elapsed)
        for trajectory in self._spec.get("trajectories") or []:
            self._apply_trajectory(trajectory, elapsed)
        self._sample_metrics()
        self._update_spectator()
        if elapsed >= self._spec["duration_s"] and not self._event_window_closed:
            self._event_window_closed = True
            print(
                "Dynamic event window ended at {:.1f}s; continuing until route completion or timeout".format(
                    elapsed
                ),
                flush=True,
            )
        if self._realtime:
            time.sleep(0.05)
        return py_trees.common.Status.RUNNING

    def _update_spectator(self):
        settings = self._spec.get("spectator") or {}
        if settings.get("mode", "chase") != "chase":
            return
        transform = self._ego.get_transform()
        forward = transform.get_forward_vector()
        distance = float(settings.get("distance_m", 9.0))
        height = float(settings.get("height_m", 4.5))
        location = transform.location - forward * distance + carla.Location(z=height)
        rotation = carla.Rotation(
            pitch=float(settings.get("pitch", -15.0)), yaw=transform.rotation.yaw
        )
        self._spectator.set_transform(carla.Transform(location, rotation))

    def _finalize(self, elapsed):
        if self._finalized:
            return
        self._finalized = True
        output = self._spec.get("metrics_output")
        if output:
            payload = {
                "schema_version": "lmdrive-scenario-runner-v1",
                "scene_id": self._spec["scene_id"],
                "duration_s": round(float(elapsed), 3),
                "ego_role_name": "hero",
                "other_role_name": "other_vehicles",
                "collision": bool(self._collision_actor_ids),
                "collision_actor_ids": sorted(self._collision_actor_ids),
                "min_center_distance_m": (
                    None if math.isinf(self._min_distance) else round(self._min_distance, 3)
                ),
                "min_ttc_s": None if math.isinf(self._min_ttc) else round(self._min_ttc, 3),
            }
            with open(output, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print("AutoScenario metrics: {}".format(output))

    def terminate(self, new_status):
        elapsed = (
            0.0
            if self._start_time is None
            else GameTime.get_time() - self._start_time
        )
        self._finalize(elapsed)
        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        super(_DynamicTimeline, self).terminate(new_status)


class AutoScenarioDynamic(BasicScenario):
    """Run generated actors with the ego controlled by the configured agent."""

    timeout = 30

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
    ):
        del randomize
        self._runtime_spec = _load_extension(config.name)
        evaluation_timeout = float(
            (self._runtime_spec.get("leaderboard") or {}).get("timeout_s", 15.0)
        )
        self.timeout = max(1, int(math.ceil(evaluation_timeout)))
        self._traffic_manager = None
        self._actors_by_id = {}
        self._route = []
        self._config = config
        super(AutoScenarioDynamic, self).__init__(
            config.name,
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _initialize_actors(self, config):
        source_payload = self._runtime_spec.get("source_spawn_payload")
        if source_payload and os.path.isfile(source_payload):
            self._spawn_source_actors(source_payload)
        else:
            # CARLA 0.9.15 can segfault inside apply_batch_sync when several
            # actors are spawned on a streamed *_Opt map.
            for actor_config in config.other_actors or []:
                actor = CarlaDataProvider.request_new_actor(
                    actor_config.model,
                    actor_config.transform,
                    actor_config.rolename,
                    autopilot=actor_config.autopilot,
                    random_location=actor_config.random_location,
                    color=actor_config.color,
                    actor_category=actor_config.category,
                )
                self.other_actors.append(actor)
        self._actors_by_id = dict(
            zip(self._runtime_spec["actor_ids"], self.other_actors)
        )
        controller = self._runtime_spec["ego_controller"]
        ego = self.ego_vehicles[0]
        self._route = self._build_route(ego)
        mode = controller.get("mode", "lmdrive_agent")
        if mode == "lmdrive_agent":
            if not getattr(config, "agent", None):
                raise RuntimeError("ego_controller requests LMDrive but no --agent was loaded")
            self._configure_lmdrive_agent(config.agent)
        else:
            self._enable_autopilot(ego, controller)

    def _spawn_source_actors(self, payload_path):
        """Use the same placement helpers as the generated reconstruction."""
        import autoscenario_scene_helpers as helpers

        with open(payload_path) as stream:
            payload = json.load(stream)
        world = CarlaDataProvider.get_world()
        helpers._autoscenario_init(world, world.get_blueprint_library())
        helpers._AUTOSCENARIO_SPAWNED_ACTORS[:] = []
        helpers._AUTOSCENARIO_FOCUS_POINTS[:] = []
        helpers._AUTOSCENARIO_EXISTING_VEHICLES = list(
            world.get_actors().filter("vehicle.*")
        )
        payload_entities = payload.get("entities") or []
        ego_entity = next(
            (
                entity
                for entity in payload_entities
                if str(entity.get("id")) in ("ego", "ego_vehicle")
            ),
            None,
        )
        if ego_entity is not None:
            location_data = ego_entity.get("location") or {}
            rotation_data = ego_entity.get("rotation") or {}
            ego_location = carla.Location(
                x=float(location_data.get("x", 0.0)),
                y=float(location_data.get("y", 0.0)),
                z=float(location_data.get("z", 0.3)),
            )
            ego_rotation = carla.Rotation(
                pitch=float(rotation_data.get("pitch", 0.0)),
                yaw=float(rotation_data.get("yaw", 0.0)),
                roll=float(rotation_data.get("roll", 0.0)),
            )
            if str(ego_entity.get("placement_mode") or "project_to_lane") == "project_to_lane":
                ego_location, ego_rotation = helpers._autoscenario_project_vehicle_to_lane(
                    ego_location, ego_rotation
                )
                self.ego_vehicles[0].set_transform(
                    carla.Transform(ego_location, ego_rotation)
                )

        entities = {
            str(entity.get("id")): entity
            for entity in payload_entities
            if str(entity.get("id")) not in ("ego", "ego_vehicle")
        }
        ego_yaw = helpers._autoscenario_payload_ego_yaw(payload)
        actors_by_id = {}
        for actor_id in self._runtime_spec["actor_ids"]:
            entity = entities.get(actor_id)
            if entity is None:
                raise RuntimeError("Source payload has no actor {!r}".format(actor_id))
            location_data = entity.get("location") or {}
            rotation_data = entity.get("rotation") or {}
            location = carla.Location(
                x=float(location_data.get("x", 0.0)),
                y=float(location_data.get("y", 0.0)),
                z=float(location_data.get("z", 0.3)),
            )
            rotation = carla.Rotation(
                pitch=float(rotation_data.get("pitch", 0.0)),
                yaw=float(rotation_data.get("yaw", 0.0)),
                roll=float(rotation_data.get("roll", 0.0)),
            )
            rotation = helpers._autoscenario_apply_heading_relation(
                entity, rotation, ego_yaw
            )
            blueprint = str(entity.get("blueprint_name") or "")
            color = entity.get("color")
            mode = str(entity.get("placement_mode") or "project_to_lane")
            if mode == "project_to_visual_pose":
                spawn_location, spawn_rotation, _ = (
                    helpers._autoscenario_project_vehicle_to_visual_pose(
                    location,
                    rotation,
                    entity.get("visual_position_override"),
                    entity.get("projected_lane"),
                    entity.get("lane_side_relation"),
                    )
                )
            elif mode in ("exact_world_pose", "direct", "preserve_xy"):
                spawn_location, spawn_rotation = location, rotation
            elif mode == "project_to_opposing_lane":
                spawn_location, spawn_rotation = helpers._autoscenario_project_to_opposing_lane(
                    location,
                    rotation,
                    int(entity.get("opposing_lane_from_median") or 1),
                )
            elif mode == "project_to_parking_lane":
                spawn_location, spawn_rotation = helpers._autoscenario_project_vehicle_to_parking_lane(
                    location,
                    rotation,
                    entity.get("projected_lane"),
                    entity.get("lane_side_relation"),
                )
            elif mode == "project_to_junction_lane":
                spawn_location, spawn_rotation = helpers._autoscenario_project_vehicle_to_specific_lane(
                    location,
                    rotation,
                    entity.get("projected_lane"),
                    entity.get("junction_distance_m"),
                )
                projected_lane = entity.get("projected_lane")
                if isinstance(projected_lane, dict) and projected_lane.get(
                    "preserve_input_yaw"
                ):
                    spawn_rotation.yaw = float(rotation.yaw)
            else:
                spawn_location, spawn_rotation = helpers._autoscenario_project_vehicle_to_lane(
                    location, rotation
                )

            # Generic source labels such as "car" make the reconstruction
            # helper choose a random CARLA blueprint.  Some 0.9.15 blueprints
            # crash natively while spawning on streamed *_Opt maps.  Use the
            # converter's deterministic, known-good fallback while preserving
            # the original transform, color, role and behavior.
            model = blueprint if blueprint.startswith("vehicle.") else "vehicle.tesla.model3"
            print(
                "Restoring actor {} mode={} model={} at ({:.2f}, {:.2f}, {:.2f})".format(
                    actor_id,
                    mode,
                    model,
                    spawn_location.x,
                    spawn_location.y,
                    spawn_location.z,
                ),
                flush=True,
            )
            actor = CarlaDataProvider.request_new_actor(
                model,
                carla.Transform(spawn_location, spawn_rotation),
                "other_vehicles",
                autopilot=False,
                random_location=False,
                color=color,
                actor_category="car",
            )
            self.other_actors.append(actor)
            actors_by_id[actor_id] = actor

        actors_by_id["ego"] = self.ego_vehicles[0]
        weather_name = (payload.get("metadata") or {}).get("carla_weather_preset")
        # Reuse the reconstruction's weather path instead of relying solely on
        # the reduced ScenarioRunner XML weather schema.  This preserves every
        # CARLA preset (including ClearNight/WetNight) and applies the same
        # visibility adjustment used by the generated static scene.
        helpers._autoscenario_apply_weather(weather_name or "ClearNoon")
        helpers._autoscenario_configure_vehicle_lights_for_environment(
            actors_by_id, payload, weather_name
        )

    def _build_route(self, ego):
        settings = self._runtime_spec.get("leaderboard") or {}
        length = float(settings.get("route_length_m", 40.0))
        step = float(settings.get("route_step_m", 1.0))
        waypoint = CarlaDataProvider.get_map().get_waypoint(ego.get_location())
        route = [(waypoint.transform.location, RoadOption.LANEFOLLOW)]
        traveled = 0.0
        while traveled < length:
            candidates = waypoint.next(min(step, length - traveled))
            if not candidates:
                break
            candidates.sort(
                key=lambda item: abs(
                    (item.transform.rotation.yaw - waypoint.transform.rotation.yaw + 180.0)
                    % 360.0
                    - 180.0
                )
            )
            next_waypoint = candidates[0]
            traveled += waypoint.transform.location.distance(next_waypoint.transform.location)
            waypoint = next_waypoint
            route.append((waypoint.transform.location, RoadOption.LANEFOLLOW))
        if len(route) < 2:
            raise RuntimeError("Could not build an LMDrive route from the ego spawn")
        return route

    def _configure_lmdrive_agent(self, agent):
        from leaderboard.utils.route_manipulation import _get_latlon_ref, location_route_to_gps

        world_route = [
            (carla.Transform(location), option) for location, option in self._route
        ]
        world = CarlaDataProvider.get_world()
        if world is None:
            raise RuntimeError("CARLA world is unavailable while configuring LMDrive")
        lat_ref, lon_ref = _get_latlon_ref(world)
        agent.set_global_plan(
            location_route_to_gps(world_route, lat_ref, lon_ref), world_route
        )
        agent.town_id = CarlaDataProvider.get_map().name.rstrip("/").rsplit("/", 1)[-1].replace("_Opt", "")
        controller = self._runtime_spec.get("ego_controller") or {}
        agent._autoscenario_ego_vehicle = self.ego_vehicles[0]
        agent._autoscenario_target_speed_mps = float(
            controller.get("target_speed_mps", 0.0)
        )
        agent._autoscenario_speed_tolerance_mps = float(
            controller.get("takeover_speed_tolerance_mps", 0.2)
        )
        agent._autoscenario_skip_brake_warmup = (
            float(controller.get("warmup_s", 0.0)) <= 0.0
        )
        agent._autoscenario_takeover = False
        py_trees.blackboard.Blackboard().set(
            "AutoScenarioLMDriveTakeover", False, overwrite=True
        )
        if agent._autoscenario_target_speed_mps > 0.0:
            _set_forward_speed(
                self.ego_vehicles[0], agent._autoscenario_target_speed_mps
            )
            print(
                "Seeded ego initial speed at {:.3f}m/s; LMDrive will take over on the first inference frame".format(
                    agent._autoscenario_target_speed_mps
                ),
                flush=True,
            )
        agent.sampled_scenarios = [
            {
                "name": notice.get("name"),
                "trigger_position": {
                    "x": float(notice.get("trigger_x", 0.0)),
                    "y": float(notice.get("trigger_y", 0.0)),
                    "z": float(notice.get("trigger_z", 0.0)),
                },
            }
            for notice in self._runtime_spec.get("notices") or []
        ]
        agent.scenario_cofing_name = self._runtime_spec["scene_id"]

    def _enable_autopilot(self, ego, controller):
        port = CarlaDataProvider.get_traffic_manager_port()
        self._traffic_manager = CarlaDataProvider.get_client().get_trafficmanager(port)
        self._traffic_manager.set_synchronous_mode(CarlaDataProvider.is_sync_mode())
        ego.set_autopilot(True, port)
        self._traffic_manager.auto_lane_change(ego, False)
        self._traffic_manager.distance_to_leading_vehicle(ego, 2.0)
        target_kph = float(controller.get("target_speed_mps", 6.0)) * 3.6
        if hasattr(self._traffic_manager, "set_desired_speed"):
            self._traffic_manager.set_desired_speed(ego, target_kph)

    def _setup_scenario_trigger(self, config):
        del config
        return None

    def _create_behavior(self):
        return _DynamicTimeline(
            self.ego_vehicles[0], self._actors_by_id, self._runtime_spec
        )

    def _create_test_criteria(self):
        ego = self.ego_vehicles[0]
        return [
            RouteCompletionTest(ego, self._route),
            CollisionTest(ego),
            OutsideRouteLanesTest(ego, self._route),
            RunningRedLightTest(ego),
            RunningStopTest(ego),
            InRouteTest(ego, self._route, terminate_on_failure=False),
            ActorSpeedAboveThresholdTest(
                ego,
                speed_threshold=0.1,
                below_threshold_max_time=90.0,
                terminate_on_failure=False,
            ),
        ]

    def remove_all_actors(self):
        ego = self.ego_vehicles[0] if self.ego_vehicles else None
        mode = (self._runtime_spec.get("ego_controller") or {}).get("mode")
        if mode != "lmdrive_agent" and ego is not None and getattr(ego, "is_alive", False):
            try:
                ego.set_autopilot(False)
            except RuntimeError:
                pass
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        super(AutoScenarioDynamic, self).remove_all_actors()

    def __del__(self):
        try:
            self.remove_all_actors()
        except (AttributeError, RuntimeError):
            pass
