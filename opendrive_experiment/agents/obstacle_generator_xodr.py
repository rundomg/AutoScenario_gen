import os
import re
from typing import Any, Dict

from agents.obstacle_generator import ObstacleGenerator


class XODRObstacleGenerator(ObstacleGenerator):
    """Obstacle generation wrapper for the OpenDRIVE experiment pipeline.

    Differences from the base ObstacleGenerator:
    - The "Negate Y coordinates and yaw" instruction is removed from the
      system prompt; negation is applied deterministically in code by
      XODRScenarioGenerator._apply_coordinate_negation().
    - The spawn-point anchor rule is tightened to a concrete 30 m radius
      and an explicit heading reference, replacing the vague "neighborhood".
    """

    def __init__(self) -> None:
        super().__init__()
        # Remove the line that asks the LLM to perform coordinate math —
        # Y-negation is handled deterministically in XODRScenarioGenerator.
        self.pre_prompt = self.pre_prompt.replace(
            "        - Negate Y coordinates and yaw before spawning\n", ""
        )
        # In a custom xodr world the road surface is near z=0; z=2 causes
        # vehicles to float above the road and fail to spawn when waypoint
        # projection is unavailable.  Use z=0.3 so spawns land on the
        # surface even without projection.
        self.pre_prompt = self.pre_prompt.replace(
            "constructioncone z=0.5, streetbarrier z=1, warningconstruction z=1, vehicles z=2, pedestrians z=1",
            "constructioncone z=0.5, streetbarrier z=0.5, warningconstruction z=0.5, vehicles z=0.3, pedestrians z=0.3",
        )
        # Replace the conservative "spawn fewer" rule with one that reproduces
        # all clearly described actors, including parked vehicles.
        self.pre_prompt = self.pre_prompt.replace(
            "        - If the scene is uncertain, spawn fewer objects rather than hallucinating more\n",
            "        - Reproduce ALL actors described in the scene (including parked vehicles and pedestrians); omit only actors with no supporting evidence in the description\n",
        )
        self.pre_prompt = self.pre_prompt.replace(
            "        - Reproduce ALL actors described in the scene (including parked vehicles and pedestrians); omit only actors with no supporting evidence in the description\n",
            "        - Reproduce ALL actors described in the scene (including parked vehicles and pedestrians); omit only actors with no supporting evidence in the description\n"
            "        - If the description uses plural evidence such as several, multiple, a row of, or line the curb, spawn a small representative set instead of collapsing them into one actor\n",
        )
        # Dense roadside scenes should preserve nearby parked vehicles and
        # pedestrians. Requiring 8 m clearance makes the LLM over-prune them.
        self.pre_prompt = self.pre_prompt.replace(
            "        - Vehicle/object distance must be at least 8 meters; object/object distance must be at least 0.5 meters\n",
            "        - Avoid physical overlap: use about 1.5 m clearance between parked or queued vehicles when the scene is dense, and at least 0.5 m between small props\n",
        )

    def format_carla_spawn_points_info(self, spawn_context: dict) -> str:
        """Tightened anchor rule: explicit 30 m radius + heading reference."""
        if not spawn_context:
            return ""

        map_name = spawn_context.get("map_name", "unknown")
        spawn_points = spawn_context.get("spawn_points", [])
        if not spawn_points:
            return f"CARLA Map: {map_name}\nCARLA Spawn Points: unavailable"

        lines = [
            f"CARLA Map: {map_name}",
            "CARLA Spawn Points:",
        ]
        for point in spawn_points[:5]:
            location = point["location"]
            rotation = point["rotation"]
            lines.append(
                "  - "
                f"index={point['index']}, "
                f"location=({location['x']:.3f}, {location['y']:.3f}, {location['z']:.3f}), "
                f"rotation=({rotation['pitch']:.3f}, {rotation['yaw']:.3f}, {rotation['roll']:.3f})"
            )

        anchor = spawn_points[0]
        ax = anchor["location"]["x"]
        ay = anchor["location"]["y"]
        az = anchor["location"]["z"]
        a_yaw = anchor["rotation"]["yaw"]
        lines.append(
            f"Anchor (index 0): location=({ax:.3f}, {ay:.3f}, {az:.3f}), yaw={a_yaw:.1f}°.\n"
            "Placement rule:\n"
            f"  1. All object locations must be within 30 m of ({ax:.3f}, {ay:.3f}).\n"
            f"  2. Use {a_yaw:.1f}° as the base heading; align vehicles ±180° of this.\n"
            "  3. Do NOT negate coordinates — output raw values as-is."
        )
        return "\n".join(lines)

    def build_generation_request(
        self,
        scene_sections: Dict[str, str],
        road_artifact: Dict[str, Any],
        spawn_context: Dict[str, Any] | None,
    ) -> str:
        if road_artifact.get("stage") == "stage1":
            # For Stage 1, combine the SUMO-level edge summary with the
            # CARLA topology sample (lane centres, yaw) so the LLM knows
            # exact lane positions — without this it must guess y-offsets.
            sumo_info = self.extract_network_info_with_xml(
                road_artifact["net_prompt_path"],
                road_artifact["debug_net_path"],
            )
            topo_info = self.format_opendrive_topology_info(road_artifact, spawn_context)
            net_info = f"{sumo_info}\n{topo_info}" if topo_info else sumo_info
        else:
            net_info = self.format_opendrive_topology_info(road_artifact, spawn_context)

        spawn_points_info = self.format_carla_spawn_points_info(spawn_context or {})
        request = self.format_scene_context(scene_sections, net_info, spawn_points_info)
        coverage_hints = self._build_scene_coverage_hints(scene_sections)
        if coverage_hints:
            request = f"{request}\n\nScene Coverage Hints:\n{coverage_hints}"
        return request

    @staticmethod
    def _build_scene_coverage_hints(scene_sections: Dict[str, str]) -> str:
        combined_text = "\n".join(scene_sections.values()).lower()
        hints = []

        if re.search(
            r"(multiple|several|row of|line the .* curb).*parked car|parked cars .*line",
            combined_text,
        ):
            hints.append(
                "- Spawn at least 3 representative parked cars along the described curb if space allows."
            )

        if re.search(r"(multiple|several)\s+pedestrian|pedestrians?\s+are\s+visible", combined_text):
            hints.append(
                "- Spawn at least 2 pedestrians on the described sidewalk or crosswalk side when they are clearly visible."
            )

        if re.search(r"(cones?|barriers?).*(line|row|narrow|taper)", combined_text):
            hints.append(
                "- Preserve cone or barrier groups as a short series rather than collapsing them into a single prop."
            )

        return "\n".join(hints)

    def format_opendrive_topology_info(
        self,
        road_artifact: Dict[str, Any],
        spawn_context: Dict[str, Any] | None,
    ) -> str:
        lines = [
            "OpenDRIVE Road Summary:",
            road_artifact.get("summary", "OpenDRIVE summary unavailable."),
        ]

        topology_sample = (spawn_context or {}).get("topology_sample", [])
        if topology_sample:
            lines.append("OpenDRIVE Topology Sample:")
            for item in topology_sample[:8]:
                lines.append(
                    "- "
                    f"road_id={item.get('road_id')}, lane_id={item.get('lane_id')}, "
                    f"start=({item['start']['x']:.3f}, {item['start']['y']:.3f}, yaw={item['start']['yaw']:.3f}), "
                    f"end=({item['end']['x']:.3f}, {item['end']['y']:.3f}, yaw={item['end']['yaw']:.3f}), "
                    f"junction={item['start']['is_junction'] or item['end']['is_junction']}"
                )

        debug_net_path = road_artifact.get("debug_net_path")
        if debug_net_path and os.path.exists(debug_net_path):
            lines.append("Optional debug SUMO summary:")
            lines.append(self._summarize_network_xml(debug_net_path))

        return "\n".join(lines)
