import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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

from agents.obstacle_generator import ObstacleGenerator
from agents.task_agent import TaskAgent
from experiments.auto_generate_all_vlm import AutoGenerator


SPLIT_TEXT = """## Road Net Description:
Straight city street with a crosswalk ahead.

## Road Users Description:
A scooter is ahead-left and pedestrians are near the sidewalk.

## Static Objects Description:
Cones are on the roadside and a traffic light is visible.

## Vehicles' Locations and Behaviors:
The scooter is ahead-left and appears slow or stopped.

## Scenario Description:
The ego vehicle approaches a crosswalk with a nearby scooter and roadside cones.
"""


class TestGenerationPipelineHelpers(unittest.TestCase):
    def test_extract_response_content_rejects_empty_content(self):
        with self.assertRaises(Exception) as context:
            TaskAgent._extract_response_content(
                {"choices": [{"message": {"content": "   "}}]}
            )
        self.assertIn("Empty model response content", str(context.exception))

    def test_extract_split_sections_from_text(self):
        sections = AutoGenerator.extract_split_sections_from_text(SPLIT_TEXT)
        self.assertEqual(
            sections["Road Net Description"],
            "Straight city street with a crosswalk ahead.",
        )
        self.assertIn("scooter is ahead-left", sections["Road Users Description"])
        self.assertIn("traffic light is visible", sections["Static Objects Description"])
        self.assertIn("nearby scooter", sections["Scenario Description"])

    def test_format_scene_context_includes_all_sections(self):
        sections = AutoGenerator.extract_split_sections_from_text(SPLIT_TEXT)
        context = ObstacleGenerator.format_scene_context(
            sections,
            "Map Description:\nMinimal road summary\nNetwork XML:\n<net/>",
            "CARLA Spawn Points:\n- index=0",
        )
        self.assertIn("Road Users Description:", context)
        self.assertIn("Static Objects Description:", context)
        self.assertIn("Generated Road Network Summary:", context)
        self.assertIn("CARLA Spawn Points:", context)


if __name__ == "__main__":
    unittest.main()
