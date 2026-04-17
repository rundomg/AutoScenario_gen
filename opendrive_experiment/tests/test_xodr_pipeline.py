import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
matplotlib_module = types.ModuleType("matplotlib")
matplotlib_pyplot = types.ModuleType("matplotlib.pyplot")
matplotlib_module.pyplot = matplotlib_pyplot
sys.modules.setdefault("matplotlib", matplotlib_module)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_pyplot)

import json

from opendrive_experiment.agents.opendrive_generator import OpendriveGenerator
from opendrive_experiment.agents.scenario_generator_xodr import XODRScenarioGenerator
from opendrive_experiment.tools.xodr_world_manager import (
    DEFAULT_GENERATION_PARAMS,
    load_xodr_world,
)


class _FakeLocation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _FakeRotation:
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class _FakeTransform:
    def __init__(self, location, rotation):
        self.location = location
        self.rotation = rotation


class _FakeWaypoint:
    def __init__(self, road_id, lane_id, x, y, yaw, is_junction=False):
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.transform = _FakeTransform(
            _FakeLocation(x, y, 0.0), _FakeRotation(0.0, yaw, 0.0)
        )


class _FakeMap:
    name = "OpenDriveTestMap"

    def get_spawn_points(self):
        return [
            _FakeTransform(_FakeLocation(1.0, 2.0, 0.0), _FakeRotation(0.0, 90.0, 0.0))
        ]

    def get_topology(self):
        return [
            (
                _FakeWaypoint(1, -1, 0.0, 0.0, 0.0, False),
                _FakeWaypoint(1, -1, 10.0, 0.0, 0.0, False),
            )
        ]


class _FakeWorld:
    def get_map(self):
        return _FakeMap()

    def wait_for_tick(self):
        return None


class _FakeClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.timeout = None
        self.generated = None

    def set_timeout(self, timeout):
        self.timeout = timeout

    def generate_opendrive_world(self, xodr_content, params):
        self.generated = (xodr_content, params)
        return _FakeWorld()


class _FakeOpendriveGenerationParameters:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCarlaModule:
    Client = _FakeClient
    OpendriveGenerationParameters = _FakeOpendriveGenerationParameters


class TestOpenDriveExperiment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workdir = Path(self.temp_dir.name)

    def test_load_xodr_world_uses_generate_opendrive_world_and_returns_context(self):
        xodr_path = self.workdir / "sample.xodr"
        xodr_path.write_text("<OpenDRIVE/>", encoding="utf-8")

        context = load_xodr_world(
            str(xodr_path),
            carla_host="localhost",
            carla_port=2000,
            generation_params={"vertex_distance": 3.0},
            carla_module=_FakeCarlaModule,
        )

        self.assertEqual(context["map_name"], "OpenDriveTestMap")
        self.assertEqual(len(context["spawn_points"]), 1)
        self.assertEqual(context["spawn_points"][0]["location"]["x"], 1.0)
        self.assertEqual(context["topology_sample"][0]["road_id"], 1)
        self.assertEqual(context["generation_params"]["vertex_distance"], 3.0)

    def test_stage1_generator_exports_xodr_after_sumo_generation(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage1")
        scene_id = "scene_0001"
        net_path = self.workdir / f"{scene_id}.net.xml"
        net_path.write_text("<net/>", encoding="utf-8")

        with mock.patch("agents.net_generator.NetGenerator.call_agent", return_value=(True, 1)):
            with mock.patch.object(generator, "convert_sumo_net_to_opendrive") as convert_mock:
                artifact = generator.call_agent(
                    "straight road",
                    scene_id,
                    {"output_fn": str(self.workdir / f"{scene_id}_net.txt")},
                )

        convert_mock.assert_called_once_with(
            str(net_path), str(self.workdir / f"{scene_id}.xodr")
        )
        self.assertEqual(artifact["stage"], "stage1")
        self.assertTrue((self.workdir / f"{scene_id}_xodr_context.json").exists())

    def test_stage2_renderer_writes_valid_opendrive_root(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        dsl = {
            "roads": [
                {
                    "id": "1",
                    "length_m": 40.0,
                    "plan_view": [
                        {
                            "s": 0.0,
                            "x": 0.0,
                            "y": 0.0,
                            "hdg": 0.0,
                            "length": 40.0,
                            "geometry": "line",
                        }
                    ],
                    "lane_sections": [
                        {
                            "s": 0.0,
                            "left": [{"type": "driving", "width": 3.5}],
                            "right": [{"type": "driving", "width": 3.5}],
                        }
                    ],
                    "successor": None,
                    "predecessor": None,
                }
            ],
            "junctions": [],
            "connections": [],
            "metadata": {"assumptions": ["minimal straight road"]},
        }
        xodr_path = self.workdir / "stage2.xodr"
        generator.render_road_dsl_to_xodr(dsl, str(xodr_path))

        content = xodr_path.read_text(encoding="utf-8")
        self.assertIn("<OpenDRIVE>", content)
        self.assertIn('<road name="1" length="40.000" id="1" junction="-1">', content)
        self.assertIn("<planView>", content)
        self.assertIn("<laneSection s=\"0.000\">", content)
        self.assertIn("<geoReference>", content)
        self.assertIn("+proj=tmerc", content)
        self.assertIn('lane id="0" type="none"', content)
        self.assertIn('roadMark sOffset="0.0" type="solid" weight="standard" color="yellow"', content)
        self.assertIn('roadMark sOffset="0.0" type="broken" weight="standard" color="white"', content)
        self.assertIn('<type name="broken_white" width="0.150">', content)
        self.assertIn('<line length="3.000" space="9.000" tOffset="0.0" width="0.150" sOffset="0.0" />', content)

    def test_stage2_renderer_respects_custom_lane_marking_style(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        dsl = {
            "roads": [
                {
                    "id": "1",
                    "length_m": 40.0,
                    "plan_view": [
                        {
                            "s": 0.0,
                            "x": 0.0,
                            "y": 0.0,
                            "hdg": 0.0,
                            "length": 40.0,
                            "geometry": "line",
                        }
                    ],
                    "lane_sections": [
                        {
                            "s": 0.0,
                            "center_road_mark_type": "solid",
                            "center_road_mark_color": "yellow",
                            "left": [
                                {
                                    "type": "driving",
                                    "width": 3.5,
                                    "road_mark_type": "solid",
                                    "road_mark_color": "white",
                                    "lane_change": "none",
                                }
                            ],
                            "right": [
                                {
                                    "type": "driving",
                                    "width": 3.5,
                                    "road_mark_type": "broken",
                                    "road_mark_color": "white",
                                    "road_mark_line_length": 4.0,
                                    "road_mark_space_length": 8.0,
                                }
                            ],
                        }
                    ],
                    "successor": None,
                    "predecessor": None,
                }
            ],
            "junctions": [],
            "connections": [],
            "metadata": {"assumptions": ["custom markings"]},
        }
        xodr_path = self.workdir / "custom_markings.xodr"
        generator.render_road_dsl_to_xodr(dsl, str(xodr_path))

        content = xodr_path.read_text(encoding="utf-8")
        self.assertIn('roadMark sOffset="0.0" type="solid" weight="standard" color="white"', content)
        self.assertIn('roadMark sOffset="0.0" type="broken" weight="standard" color="white"', content)
        self.assertIn('<line length="4.000" space="8.000" tOffset="0.0" width="0.150" sOffset="0.0" />', content)

    def test_stage2_renderer_supports_explicit_parking_lanes(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        dsl = {
            "roads": [
                {
                    "id": "4lane",
                    "length_m": 50.0,
                    "plan_view": [
                        {
                            "s": 0.0,
                            "x": 0.0,
                            "y": 0.0,
                            "hdg": 0.0,
                            "length": 50.0,
                            "geometry": "line",
                        }
                    ],
                    "lane_sections": [
                        {
                            "s": 0.0,
                            "left": [
                                {"type": "driving", "width": 3.5},
                                {"type": "parking", "width": 2.4, "road_mark_type": "solid"},
                            ],
                            "right": [
                                {"type": "driving", "width": 3.5},
                                {"type": "parking", "width": 2.4, "road_mark_type": "solid"},
                            ],
                        }
                    ],
                    "successor": None,
                    "predecessor": None,
                }
            ],
            "junctions": [],
            "connections": [],
            "metadata": {"assumptions": ["explicit parking lanes"]},
        }
        xodr_path = self.workdir / "parking_lanes.xodr"
        generator.render_road_dsl_to_xodr(dsl, str(xodr_path))

        content = xodr_path.read_text(encoding="utf-8")
        self.assertIn('lane id="1" type="driving"', content)
        self.assertIn('lane id="2" type="parking"', content)
        self.assertIn('lane id="-1" type="driving"', content)
        self.assertIn('lane id="-2" type="parking"', content)

    def test_stage2_prompt_mentions_explicit_parking_lane_output(self):
        prompt = OpendriveGenerator.DSL_SYSTEM_PROMPT
        self.assertIn("Lane `type` may include `driving`, `parking`, and `shoulder`", prompt)
        self.assertIn("emit explicit `parking` lanes", prompt)
        self.assertIn("from the road center outward", prompt)

    def test_stage2_extract_artifacts_adds_outer_shoulder_buffers(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        scene_id = "scene_shoulder"
        output_fn = self.workdir / f"{scene_id}_xodr.txt"
        output_fn.write_text(
            """## Description
simple
## Reasoning
simple
## OpenDRIVE DSL
```json
{
  "roads": [
    {
      "id": "1",
      "length_m": 30.0,
      "plan_view": [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 30.0, "geometry": "line"}
      ],
      "lane_sections": [
        {
          "s": 0.0,
          "left": [{"type": "driving", "width": 3.5}],
          "right": [{"type": "driving", "width": 3.5}, {"type": "parking", "width": 2.5}]
        }
      ],
      "successor": null,
      "predecessor": null
    }
  ],
  "junctions": [],
  "connections": [],
  "metadata": {"assumptions": ["simple"]}
}
```
""",
            encoding="utf-8",
        )

        with mock.patch.object(generator, "validate_xodr_output", return_value=None):
            artifact, error = generator.extract_stage2_artifacts(scene_id, str(output_fn))

        self.assertIsNone(error)
        with open(self.workdir / f"{scene_id}.road_dsl.json", "r", encoding="utf-8") as file:
            dsl = json.load(file)
        lane_section = dsl["roads"][0]["lane_sections"][0]
        self.assertEqual(lane_section["left"][-1]["type"], "shoulder")
        self.assertEqual(lane_section["right"][-1]["type"], "shoulder")
        self.assertEqual(lane_section["left"][-1]["width"], 3.0)
        self.assertEqual(lane_section["right"][-1]["width"], 3.0)
        self.assertIn("shoulder buffers", " ".join(dsl["metadata"]["assumptions"]))

    def test_stage2_extract_artifacts_widens_existing_outer_shoulders(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        scene_id = "scene_shoulder_existing"
        output_fn = self.workdir / f"{scene_id}_xodr.txt"
        output_fn.write_text(
            """## Description
simple
## Reasoning
simple
## OpenDRIVE DSL
```json
{
  "roads": [
    {
      "id": "1",
      "length_m": 30.0,
      "plan_view": [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 30.0, "geometry": "line"}
      ],
      "lane_sections": [
        {
          "s": 0.0,
          "left": [{"type": "driving", "width": 3.5}, {"type": "shoulder", "width": 2.0}],
          "right": [{"type": "driving", "width": 3.5}, {"type": "parking", "width": 2.5}, {"type": "shoulder", "width": 2.0}]
        }
      ],
      "successor": null,
      "predecessor": null
    }
  ],
  "junctions": [],
  "connections": [],
  "metadata": {"assumptions": ["simple"]}
}
```
""",
            encoding="utf-8",
        )

        with mock.patch.object(generator, "validate_xodr_output", return_value=None):
            artifact, error = generator.extract_stage2_artifacts(scene_id, str(output_fn))

        self.assertIsNone(error)
        with open(self.workdir / f"{scene_id}.road_dsl.json", "r", encoding="utf-8") as file:
            dsl = json.load(file)
        lane_section = dsl["roads"][0]["lane_sections"][0]
        self.assertEqual(lane_section["left"][-1]["width"], 3.0)
        self.assertEqual(lane_section["right"][-1]["width"], 3.0)

    def test_stage2_extract_artifacts_sanitizes_alias_links_and_drops_empty_junctions(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        scene_id = "scene_alias_links"
        output_fn = self.workdir / f"{scene_id}_xodr.txt"
        output_fn.write_text(
            """## Description
simple
## Reasoning
simple
## OpenDRIVE DSL
```json
{
  "roads": [
    {
      "id": "r1",
      "length_m": 30.0,
      "plan_view": [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 30.0, "geometry": "line"}
      ],
      "lane_sections": [
        {"s": 0.0, "left": [{"type": "driving", "width": 3.5}], "right": [{"type": "driving", "width": 3.5}]}
      ],
      "predecessor": {"type": "none", "id": null},
      "successor": {"type": "junction", "id": "j1"}
    }
  ],
  "junctions": [
    {"id": "j1", "type": "signalized_intersection"}
  ],
  "connections": [],
  "metadata": {"assumptions": ["simple"]}
}
```
""",
            encoding="utf-8",
        )

        with mock.patch.object(generator, "validate_xodr_output", return_value=None):
            artifact, error = generator.extract_stage2_artifacts(scene_id, str(output_fn))

        self.assertIsNone(error)
        self.assertIsNotNone(artifact)
        with open(self.workdir / f"{scene_id}.road_dsl.json", "r", encoding="utf-8") as file:
            dsl = json.load(file)
        road = dsl["roads"][0]
        self.assertIsNone(road["predecessor"])
        self.assertIsNone(road["successor"])
        self.assertEqual(dsl["junctions"], [])

    def test_stage2_extract_artifacts_renders_and_validates_xodr(self):
        generator = OpendriveGenerator(str(self.workdir), generation_stage="stage2")
        scene_id = "scene_0002"
        output_fn = self.workdir / f"{scene_id}_xodr.txt"
        output_fn.write_text(
            """## Description
simple
## Reasoning
simple
## OpenDRIVE DSL
```json
{
  "roads": [
    {
      "id": "1",
      "length_m": 30.0,
      "plan_view": [
        {"s": 0.0, "x": 0.0, "y": 0.0, "hdg": 0.0, "length": 30.0, "geometry": "line"}
      ],
      "lane_sections": [
        {"s": 0.0, "left": [{"type": "driving", "width": 3.5}], "right": [{"type": "driving", "width": 3.5}]}
      ],
      "successor": null,
      "predecessor": null
    }
  ],
  "junctions": [],
  "connections": [],
  "metadata": {"assumptions": ["simple"]}
}
```
""",
            encoding="utf-8",
        )

        with mock.patch.object(generator, "validate_xodr_output", return_value=None):
            artifact, error = generator.extract_stage2_artifacts(scene_id, str(output_fn))

        self.assertIsNone(error)
        self.assertEqual(artifact["stage"], "stage2")
        self.assertTrue((self.workdir / f"{scene_id}.road_dsl.json").exists())
        self.assertTrue((self.workdir / f"{scene_id}.xodr").exists())

    def test_xodr_scene_generator_injects_world_bootstrap(self):
        generator = XODRScenarioGenerator()
        response = """## Decision
```python
import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
time.sleep(2)
```
"""

        success, error = generator.extract_decision_data(
            "scene_0003",
            output_folder=str(self.workdir),
            response=response,
            xodr_path=str(self.workdir / "scene_0003.xodr"),
            generation_params=DEFAULT_GENERATION_PARAMS,
        )

        self.assertTrue(success)
        self.assertIsNone(error)
        scene_path = self.workdir / "scene_0003_scene_final_xodr.py"
        content = scene_path.read_text(encoding="utf-8")
        self.assertIn("_autoscenario_load_xodr_world(client)", content)
        self.assertIn("__AUTOSCENARIO_XODR_WORLD__", content)
        self.assertIn("scene_0003.xodr", content)
        self.assertIn("_autoscenario_project_vehicle_to_lane", content)
        self.assertIn("world_map.get_waypoint", content)
        self.assertIn("_autoscenario_project_static_to_ground", content)
        self.assertIn("_autoscenario_blueprint_matches_category", content)
        self.assertIn("snapped_location", content)
        self.assertIn("snapped_rotation", content)


class TestValidateRoadGeometryContinuity(unittest.TestCase):
    """Unit tests for OpendriveGenerator.validate_road_geometry_continuity."""

    def _straight_road(self, rid, x, y, hdg, length, successor=None, predecessor=None):
        return {
            "id": rid,
            "length_m": length,
            "plan_view": [{"s": 0, "x": x, "y": y, "hdg": hdg, "length": length, "geometry": "line"}],
            "lane_sections": [{"s": 0, "left": [{"type": "driving", "width": 3.5}],
                                "right": [{"type": "driving", "width": 3.5}]}],
            "successor": successor,
            "predecessor": predecessor,
        }

    def _link(self, elem_id, elem_type="road"):
        return {"elementType": elem_type, "elementId": str(elem_id), "contactPoint": "start"}

    def test_connected_roads_aligned(self):
        # Road 1 ends at (40, 0); Road 2 starts at (40, 0) → gap = 0
        dsl = {
            "roads": [
                self._straight_road("1", 0, 0, 0.0, 40.0, successor=self._link("2")),
                self._straight_road("2", 40, 0, 0.0, 20.0, predecessor=self._link("1")),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        self.assertIsNone(OpendriveGenerator.validate_road_geometry_continuity(dsl))

    def test_connected_roads_gap_exceeds_tolerance(self):
        # Road 1 ends at (40, 0); Road 2 starts at (50, 0) → gap = 10 m
        dsl = {
            "roads": [
                self._straight_road("1", 0, 0, 0.0, 40.0, successor=self._link("2")),
                self._straight_road("2", 50, 0, 0.0, 20.0, predecessor=self._link("1")),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_geometry_continuity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("'1'", err)
        self.assertIn("10.0 m", err)

    def test_junction_successor_not_checked(self):
        # Junction successors are skipped — only road→road links are checked
        dsl = {
            "roads": [
                self._straight_road("1", 0, 0, 0.0, 40.0,
                                    successor={"elementType": "junction", "elementId": "J1",
                                               "contactPoint": "start"}),
            ],
            "junctions": [{"id": "J1", "connections": [
                {"id": "1", "incomingRoad": "1", "connectingRoad": "1", "contactPoint": "start"}
            ]}],
            "connections": [], "metadata": {"assumptions": []},
        }
        self.assertIsNone(OpendriveGenerator.validate_road_geometry_continuity(dsl))

    def test_tolerance_boundary(self):
        # Gap of exactly 3.0 m should pass (tolerance = 3.0)
        dsl = {
            "roads": [
                self._straight_road("1", 0, 0, 0.0, 40.0, successor=self._link("2")),
                self._straight_road("2", 43, 0, 0.0, 20.0, predecessor=self._link("1")),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        # gap = 43 - 40 = 3.0 m → borderline, not strictly > 3.0 so should pass
        self.assertIsNone(OpendriveGenerator.validate_road_geometry_continuity(dsl))

    def test_arc_endpoint_computed_correctly(self):
        import math
        # Quarter-circle arc: radius=10, curvature=0.1, length=pi/2*10≈15.708
        # Start: (0, 0, hdg=0). Arc center: (0, 10). End: (10, 10) rotated...
        # Actually: end = (cx + r*cos(angle_end), cy + r*sin(angle_end))
        # cx=0, cy=10, angle_start=-pi/2, angle_span=15.708*0.1≈1.5708≈pi/2
        # angle_end = -pi/2 + pi/2 = 0 → end = (10, 10)
        # Road 2 starts at (10, 10) → gap = 0
        arc_length = math.pi / 2 * 10  # ≈15.708
        dsl = {
            "roads": [
                {
                    "id": "1", "length_m": arc_length,
                    "plan_view": [{"s": 0, "x": 0, "y": 0, "hdg": 0.0,
                                   "length": arc_length, "geometry": "arc", "curvature": 0.1}],
                    "lane_sections": [{"s": 0, "left": [], "right": [{"type": "driving", "width": 3.5}]}],
                    "successor": self._link("2"), "predecessor": None,
                },
                self._straight_road("2", 10, 10, 0.0, 20.0, predecessor=self._link("1")),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        self.assertIsNone(
            OpendriveGenerator.validate_road_geometry_continuity(dsl, tolerance_m=0.1)
        )


class TestApplyCoordinateNegation(unittest.TestCase):
    """Unit tests for XODRScenarioGenerator._apply_coordinate_negation."""

    def _make_stdout(self, agent_dict, object_dict):
        from agents.obstacle_generator import OBJECT_INFO_PREFIX
        payload = {"agent_dict": agent_dict, "object_dict": object_dict}
        return OBJECT_INFO_PREFIX + json.dumps(payload, sort_keys=True)

    def test_negates_y_and_yaw(self):
        stdout = self._make_stdout(
            {"car_1": {"location": [10.0, 20.0, 2.0], "rotation": [0.0, 90.0, 0.0], "type": "car"}},
            {},
        )
        result = XODRScenarioGenerator._apply_coordinate_negation(stdout)
        from agents.obstacle_generator import OBJECT_INFO_PREFIX
        payload_line = next(l for l in result.splitlines() if l.startswith(OBJECT_INFO_PREFIX))
        payload = json.loads(payload_line[len(OBJECT_INFO_PREFIX):])
        car = payload["agent_dict"]["car_1"]
        self.assertAlmostEqual(car["location"][1], -20.0)
        self.assertAlmostEqual(car["rotation"][1], -90.0)
        # x and z unchanged
        self.assertAlmostEqual(car["location"][0], 10.0)
        self.assertAlmostEqual(car["location"][2], 2.0)

    def test_static_objects_also_negated(self):
        stdout = self._make_stdout(
            {},
            {"cone_1": {"location": [5.0, -3.0, 0.5], "rotation": [0.0, 45.0, 0.0], "type": "constructioncone"}},
        )
        result = XODRScenarioGenerator._apply_coordinate_negation(stdout)
        from agents.obstacle_generator import OBJECT_INFO_PREFIX
        payload_line = next(l for l in result.splitlines() if l.startswith(OBJECT_INFO_PREFIX))
        payload = json.loads(payload_line[len(OBJECT_INFO_PREFIX):])
        cone = payload["object_dict"]["cone_1"]
        self.assertAlmostEqual(cone["location"][1], 3.0)
        self.assertAlmostEqual(cone["rotation"][1], -45.0)

    def test_no_payload_line_returns_unchanged(self):
        stdout = "some unrelated output\nno prefix here"
        self.assertEqual(XODRScenarioGenerator._apply_coordinate_negation(stdout), stdout)

    def test_malformed_json_returns_unchanged(self):
        from agents.obstacle_generator import OBJECT_INFO_PREFIX
        stdout = OBJECT_INFO_PREFIX + "{not valid json"
        self.assertEqual(XODRScenarioGenerator._apply_coordinate_negation(stdout), stdout)


class TestXODRObstacleGeneratorPrompt(unittest.TestCase):
    """Verify prompt overrides in XODRObstacleGenerator."""

    def setUp(self):
        from opendrive_experiment.agents.obstacle_generator_xodr import XODRObstacleGenerator
        self.gen = XODRObstacleGenerator()

    def test_no_negate_instruction_in_prompt(self):
        self.assertNotIn("Negate Y", self.gen.pre_prompt)

    def test_vehicle_z_lowered(self):
        self.assertIn("vehicles z=0.3", self.gen.pre_prompt)
        self.assertNotIn("vehicles z=2", self.gen.pre_prompt)

    def test_conservative_spawn_rule_replaced(self):
        self.assertNotIn("spawn fewer objects rather than hallucinating", self.gen.pre_prompt)
        self.assertIn("Reproduce ALL actors", self.gen.pre_prompt)
        self.assertIn("plural evidence", self.gen.pre_prompt)

    def test_dense_spacing_rule_replaced(self):
        self.assertNotIn("Vehicle/object distance must be at least 8 meters", self.gen.pre_prompt)
        self.assertIn("Avoid physical overlap", self.gen.pre_prompt)

    def test_anchor_prompt_contains_radius_constraint(self):
        spawn_context = {
            "map_name": "TestMap",
            "spawn_points": [
                {"index": 0, "location": {"x": 10.0, "y": 20.0, "z": 0.0},
                 "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0}},
            ],
        }
        info = self.gen.format_carla_spawn_points_info(spawn_context)
        self.assertIn("30 m", info)
        self.assertIn("Do NOT negate", info)
        self.assertIn("90.0°", info)

    def test_stage1_build_request_includes_topology(self):
        """Stage 1 requests must include topology_sample alongside SUMO net info."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            net_txt = pathlib.Path(tmp) / "net.txt"
            net_xml = pathlib.Path(tmp) / "net.xml"
            net_txt.write_text(
                "## Description\nStraight road.\n## SUMO Files Specification\n```xml\n```",
                encoding="utf-8",
            )
            net_xml.write_text("<net/>", encoding="utf-8")

            road_artifact = {
                "stage": "stage1",
                "net_prompt_path": str(net_txt),
                "debug_net_path": str(net_xml),
                "summary": "2 roads",
            }
            spawn_context = {
                "map_name": "TestMap",
                "spawn_points": [
                    {"index": 0, "location": {"x": 50.0, "y": -1.6, "z": 0.0},
                     "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}},
                ],
                "topology_sample": [
                    {
                        "road_id": 1, "lane_id": -1,
                        "start": {"x": 0.0, "y": -1.6, "yaw": 0.0, "is_junction": False},
                        "end": {"x": 100.0, "y": -1.6, "yaw": 0.0, "is_junction": False},
                    }
                ],
            }
            request = self.gen.build_generation_request({}, road_artifact, spawn_context)
            # Should contain topology info even for stage1
            self.assertIn("road_id=1", request)
            self.assertIn("lane_id=-1", request)

    def test_build_request_adds_scene_coverage_hints(self):
        road_artifact = {"stage": "stage2", "summary": "simple"}
        scene_sections = {
            "Road Users Description": "Several pedestrians are visible. A parked scooter is near the curb.",
            "Static Objects Description": "Multiple parked cars line the right curb. Cones narrow the roadside.",
        }
        request = self.gen.build_generation_request(scene_sections, road_artifact, None)
        self.assertIn("Scene Coverage Hints:", request)
        self.assertIn("representative parked cars", request)
        self.assertIn("at least 2 pedestrians", request)
        self.assertIn("short series", request)


class TestValidateRoadConnectivity(unittest.TestCase):
    """Unit tests for OpendriveGenerator.validate_road_connectivity."""

    def _make_road(self, rid, successor=None, predecessor=None):
        return {
            "id": rid,
            "length_m": 30.0,
            "plan_view": [{"s": 0, "x": 0, "y": 0, "hdg": 0, "length": 30, "geometry": "line"}],
            "lane_sections": [{"s": 0, "left": [{"type": "driving", "width": 3.5}],
                                "right": [{"type": "driving", "width": 3.5}]}],
            "successor": successor,
            "predecessor": predecessor,
        }

    def _link(self, elem_id, elem_type="road", contact="start"):
        return {"elementType": elem_type, "elementId": str(elem_id), "contactPoint": contact}

    def test_valid_two_road_chain(self):
        dsl = {
            "roads": [
                self._make_road("1", successor=self._link("2")),
                self._make_road("2", predecessor=self._link("1")),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        self.assertIsNone(OpendriveGenerator.validate_road_connectivity(dsl))

    def test_p0_successor_references_missing_road(self):
        dsl = {
            "roads": [self._make_road("1", successor=self._link("99"))],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("99", err)

    def test_p1_missing_reverse_link(self):
        # A → B but B has no predecessor
        dsl = {
            "roads": [
                self._make_road("1", successor=self._link("2")),
                self._make_road("2"),  # no predecessor
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("predecessor", err)

    def test_p1_reverse_link_wrong_target(self):
        # A.successor → B, but B.predecessor → C (not A)
        dsl = {
            "roads": [
                self._make_road("1", successor=self._link("2")),
                self._make_road("2", predecessor=self._link("3")),
                self._make_road("3"),
            ],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("'3'", err)

    def test_p2_duplicate_road_ids(self):
        dsl = {
            "roads": [self._make_road("1"), self._make_road("1")],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("Duplicate road IDs", err)

    def test_p0_junction_connection_missing_road(self):
        dsl = {
            "roads": [
                self._make_road("1", successor=self._link("J1", elem_type="junction")),
            ],
            "junctions": [{"id": "J1", "connections": [
                {"id": "1", "incomingRoad": "1", "connectingRoad": "99", "contactPoint": "start"}
            ]}],
            "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("99", err)

    def test_p1_road_junction_not_mentioned_in_junction(self):
        # Road 1 references junction J1, but J1 has no connection mentioning road 1
        dsl = {
            "roads": [
                self._make_road("1", successor=self._link("J1", elem_type="junction")),
                self._make_road("2"),
            ],
            "junctions": [{"id": "J1", "connections": [
                {"id": "1", "incomingRoad": "2", "connectingRoad": "2", "contactPoint": "start"}
            ]}],
            "connections": [], "metadata": {"assumptions": []},
        }
        err = OpendriveGenerator.validate_road_connectivity(dsl)
        self.assertIsNotNone(err)
        self.assertIn("junction 'J1'", err)

    def test_valid_single_road_no_links(self):
        dsl = {
            "roads": [self._make_road("1")],
            "junctions": [], "connections": [], "metadata": {"assumptions": []},
        }
        self.assertIsNone(OpendriveGenerator.validate_road_connectivity(dsl))


if __name__ == "__main__":
    unittest.main()
