#!/usr/bin/env python3
"""Run LMDrive's old ScenarioRunner with modern CARLA map-name handling."""

from __future__ import print_function

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

# This repository also has a top-level ``agents`` package.  LMDrive needs
# CARLA's ``agents.navigation`` package, so give that path explicit priority.
_carla_root = os.environ.get("CARLA_ROOT", "/home/zx/code/autodirve/CARLA_0.9.15")
sys.path.insert(0, os.path.join(_carla_root, "PythonAPI", "carla"))

import scenario_runner
import carla
import py_trees
from srunner.scenariomanager.traffic_events import TrafficEventType


def _map_basename(name):
    return str(name or "").rstrip("/").rsplit("/", 1)[-1]


def _load_and_wait_for_world_compat(self, town, ego_vehicles=None):
    """Backport basename comparison used by newer ScenarioRunner releases."""
    load_name = _map_basename(town)
    if self._args.reloadWorld:
        self.world = self.client.load_world(load_name)
    else:
        ego_vehicle_found = False
        if self._args.waitForEgo:
            while not ego_vehicle_found and not self._shutdown_requested:
                vehicles = self.client.get_world().get_actors().filter("vehicle.*")
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    for vehicle in vehicles:
                        if vehicle.attributes["role_name"] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            break
                    if not ego_vehicle_found:
                        print("Not all ego vehicles ready. Waiting ... ")
                        time.sleep(1)
                        break

    self.world = self.client.get_world()

    if self._args.sync:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        self.world.apply_settings(settings)

    data_provider = scenario_runner.CarlaDataProvider
    data_provider.set_client(self.client)
    data_provider.set_world(self.world)
    data_provider.set_traffic_manager_port(int(self._args.trafficManagerPort))

    if data_provider.is_sync_mode():
        self.world.tick()
    else:
        self.world.wait_for_tick()

    actual_name = data_provider.get_map().name
    if _map_basename(actual_name) != load_name and actual_name != "OpenDriveMap":
        print("The CARLA server uses the wrong map: {}".format(actual_name))
        print("This scenario requires to use map: {}".format(town))
        return False
    return True


def _cleanup_compat(self):
    """Avoid double-destroying actors in this LMDrive ScenarioRunner fork."""
    if (
        self.manager is not None
        and self.manager.get_running_status()
        and self.world is not None
        and self._args.sync
    ):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    # ScenarioManager.cleanup() already calls CarlaDataProvider.cleanup() in
    # this fork.  The original ScenarioRunner calls it a second time and then
    # destroys the same ego actor again, which modern CARLA reports as an error.
    self.manager.cleanup()
    for index, _ego in enumerate(self.ego_vehicles):
        # CarlaDataProvider.cleanup() above has already submitted a batched
        # DestroyActor command for every registered actor, including the ego.
        self.ego_vehicles[index] = None
    self.ego_vehicles = []

    if self.agent_instance:
        self.agent_instance.destroy()
        self.agent_instance = None


scenario_runner.ScenarioRunner._load_and_wait_for_world = (
    _load_and_wait_for_world_compat
)
scenario_runner.ScenarioRunner._cleanup = _cleanup_compat


def _metrics_output(config):
    path = os.environ.get("AUTOSCENARIO_CONFIG_XML")
    if not path:
        return None
    root = ET.parse(path).getroot()
    node = next(
        (item for item in root.iter("scenario") if item.get("name") == config.name),
        None,
    )
    extension = node.find("autoscenario") if node is not None else None
    return extension.get("metrics_output") if extension is not None else None


def _write_leaderboard_metrics(runner, config):
    output = _metrics_output(config)
    if not output or not os.path.isfile(output):
        return
    criteria = runner.manager.scenario.get_criteria()
    events = [event for criterion in criteria for event in criterion.list_traffic_events]
    by_type = {}
    for event in events:
        by_type.setdefault(event.get_type(), []).append(event)

    route_score = 0.0
    if by_type.get(TrafficEventType.ROUTE_COMPLETED):
        route_score = 100.0
    else:
        for event in by_type.get(TrafficEventType.ROUTE_COMPLETION, []):
            route_score = max(route_score, float((event.get_dict() or {}).get("route_completed", 0.0)))

    penalties = {
        TrafficEventType.COLLISION_PEDESTRIAN: 0.50,
        TrafficEventType.COLLISION_VEHICLE: 0.60,
        TrafficEventType.COLLISION_STATIC: 0.65,
        TrafficEventType.TRAFFIC_LIGHT_INFRACTION: 0.70,
        TrafficEventType.STOP_INFRACTION: 0.80,
    }
    penalty = 1.0
    for event in events:
        if event.get_type() in penalties:
            penalty *= penalties[event.get_type()]
        elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
            penalty *= max(0.0, 1.0 - float((event.get_dict() or {}).get("percentage", 0.0)) / 100.0)

    def messages(event_type):
        return [event.get_message() for event in by_type.get(event_type, [])]

    infractions = {
        "collisions_pedestrian": messages(TrafficEventType.COLLISION_PEDESTRIAN),
        "collisions_vehicle": messages(TrafficEventType.COLLISION_VEHICLE),
        "collisions_static": messages(TrafficEventType.COLLISION_STATIC),
        "outside_route_lanes": messages(TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION),
        "red_light": messages(TrafficEventType.TRAFFIC_LIGHT_INFRACTION),
        "stop_sign": messages(TrafficEventType.STOP_INFRACTION),
        "route_deviation": messages(TrafficEventType.ROUTE_DEVIATION),
        "vehicle_blocked": messages(TrafficEventType.VEHICLE_BLOCKED),
    }
    with open(output) as stream:
        payload = json.load(stream)
    payload["leaderboard"] = {
        "status": "Completed" if route_score >= 100.0 else "Partial",
        "scores": {
            "route_completion": round(route_score, 3),
            "infraction_penalty": round(penalty, 6),
            "driving_score": round(route_score * penalty, 3),
        },
        "infractions": infractions,
        "criteria": {
            criterion.name: {
                "status": criterion.test_status,
                "actual_value": criterion.actual_value,
            }
            for criterion in criteria
        },
        "duration_system_s": round(runner.manager.scenario_duration_system, 3),
        "duration_game_s": round(runner.manager.scenario_duration_game, 3),
    }
    with open(output, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("Leaderboard-style metrics: {}".format(output))


_original_analyze_scenario = scenario_runner.ScenarioRunner._analyze_scenario


def _analyze_scenario_compat(self, config):
    result = _original_analyze_scenario(self, config)
    _write_leaderboard_metrics(self, config)
    return result


scenario_runner.ScenarioRunner._analyze_scenario = _analyze_scenario_compat


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="2000")
    parser.add_argument("--timeout", default="60")
    parser.add_argument("--trafficManagerPort", default="8000")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--configFile", required=True)
    parser.add_argument("--additionalScenario", required=True)
    parser.add_argument("--agent")
    parser.add_argument("--agentConfig", default="")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--reloadWorld", action="store_true")
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.openscenario = None
    args.route = None
    args.file = False
    args.junit = False
    args.outputDir = ""
    args.record = ""
    args.randomize = False
    args.repetitions = 1
    args.waitForEgo = False
    if args.agent:
        args.sync = True
    return args


def main():
    args = _arguments()
    runner = None
    try:
        runner = scenario_runner.ScenarioRunner(args)
        if args.agent:
            # ScenarioRunner 0.9.9 derives the wrong class name for LMDrive.
            entry_point = runner.module_agent.get_entry_point()
            legacy_name = runner.module_agent.__name__.title().replace("_", "")
            setattr(runner.module_agent, legacy_name, getattr(runner.module_agent, entry_point))
            # Avoid LMDrive's optional frame dump, which otherwise requires ROUTES.
            runner.module_agent.SAVE_PATH = None

            # LMDrive computes a misleading instruction even when that feature
            # is disabled. Its legacy lookup table has no Town12/13/15 entry;
            # keep real-map navigation and bypass only that unused branch.
            planner_class = runner.module_agent.InstructionPlanner
            original_mislead = planner_class.command2mislead

            def safe_command2mislead(self, town_id, tick_data):
                command = tick_data.get("next_command")
                if command in (1, 2, 3) and town_id not in self.tjunction_mapping:
                    return getattr(self, "prev_mislead", "")
                return original_mislead(self, town_id, tick_data)

            planner_class.command2mislead = safe_command2mislead

            trace_path = os.environ.get("AUTOSCENARIO_AGENT_TRACE")
            agent_class = getattr(runner.module_agent, entry_point)
            original_run_step = agent_class.run_step
            if trace_path:
                with open(trace_path, "w"):
                    pass

            def traced_run_step(self, input_data, timestamp):
                if (
                    not getattr(self, "initialized", False)
                    and getattr(self, "_autoscenario_skip_brake_warmup", False)
                    and not getattr(self, "_autoscenario_takeover", True)
                ):
                    # LMDrive's stock agent spends its first 20 calls braking.
                    # Converted scenes already seed the requested rolling
                    # speed, and AgentWrapper only calls us after every sensor
                    # has delivered data, so start at its first inference step.
                    self.step = 19
                control = original_run_step(self, input_data, timestamp)
                inference_executed = self.step >= 20 and self.step % 2 == 0
                speed_sample = input_data.get("speed")
                if isinstance(speed_sample, (tuple, list)) and len(speed_sample) > 1:
                    speed_sample = speed_sample[1]
                if isinstance(speed_sample, dict):
                    ego_speed = float(speed_sample.get("speed", 0.0))
                else:
                    ego_speed = 0.0

                control_source = "lmdrive"
                target_speed = float(
                    getattr(self, "_autoscenario_target_speed_mps", 0.0)
                )
                takeover = bool(getattr(self, "_autoscenario_takeover", True))
                if not takeover:
                    tolerance = float(
                        getattr(self, "_autoscenario_speed_tolerance_mps", 0.2)
                    )
                    speed_ready = ego_speed >= max(0.0, target_speed - tolerance)
                    inference_ready = self.step >= 20 and self.step % 2 == 0
                    if speed_ready and inference_ready:
                        self._autoscenario_takeover = True
                        takeover = True
                        py_trees.blackboard.Blackboard().set(
                            "AutoScenarioLMDriveTakeover", True, overwrite=True
                        )
                        print(
                            "LMDrive takeover at step={} speed={:.3f}m/s target={:.3f}m/s".format(
                                self.step, ego_speed, target_speed
                            ),
                            flush=True,
                        )
                    else:
                        control_source = "speed_initializer"
                        ego_vehicle = getattr(
                            self, "_autoscenario_ego_vehicle", None
                        )
                        if ego_vehicle is not None and ego_vehicle.is_alive:
                            forward = ego_vehicle.get_transform().get_forward_vector()
                            ego_vehicle.set_target_velocity(
                                carla.Vector3D(
                                    x=forward.x * target_speed,
                                    y=forward.y * target_speed,
                                    z=0.0,
                                )
                            )
                        control = carla.VehicleControl(
                            throttle=0.0, steer=0.0, brake=0.0
                        )
                record = {
                    "timestamp_s": round(float(timestamp), 3),
                    "step": int(self.step),
                    "sensor_warmup": self.step < 20,
                    "inference_executed": inference_executed,
                    "ego_speed_mps": round(ego_speed, 6),
                    "target_speed_mps": round(target_speed, 6),
                    "takeover": takeover,
                    "control_source": control_source,
                    "instruction": getattr(self, "curr_instruction", ""),
                    "notice": getattr(self, "curr_notice", ""),
                    "visual_buffer_frames": len(
                        getattr(self, "visual_feature_buffer", [])
                    ),
                    "control": {
                        "throttle": round(float(control.throttle), 6),
                        "steer": round(float(control.steer), 6),
                        "brake": round(float(control.brake), 6),
                    },
                }
                if trace_path:
                    with open(trace_path, "a") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                if inference_executed:
                    print(
                        "LMDrive inference step={step} source={source} notice={notice!r} "
                        "control=({throttle:.3f},{steer:.3f},{brake:.3f})".format(
                            step=record["step"],
                            source=record["control_source"],
                            notice=record["notice"],
                            **record["control"]
                        )
                    )
                return control

            agent_class.run_step = traced_run_step

            # LMDrive uses Leaderboard pseudo sensors (speedometer), unsupported
            # by ScenarioRunner's older wrapper.
            import srunner.scenariomanager.scenario_manager as manager_module
            from leaderboard.autoagents.agent_wrapper import AgentWrapper

            manager_module.AgentWrapper = AgentWrapper
        return not runner.run()
    finally:
        if runner is not None:
            runner.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
