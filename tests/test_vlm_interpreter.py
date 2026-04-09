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

from agents.vlm_interpreter import VLMInterpreter


VALID_OUTPUT = """## Road Net Description:
The image shows a paved urban street with visible lane markings, a zebra crosswalk, and sidewalks on both sides. The full lane count is unclear because the image crops part of the roadway.

## Road Users Description:
A motorcycle is visible ahead of the ego vehicle in the same direction of travel. It matters because it occupies the lane directly in front of the ego vehicle. A pedestrian is visible on the crosswalk ahead, which matters because the crossing area is occupied.

## Static Objects Description:
Storefronts are visible along the right side of the street and trees are visible on the left sidewalk. A blue road sign is visible farther ahead, but the text on the sign is unclear.

## Vehicles' Locations and Behaviors:
A motorcycle is ahead of the ego vehicle near the center-right portion of the roadway. A small car is visible farther ahead in the opposite direction. Their exact speeds are unclear from the single image.

## Scenario Description:
This is an urban street scene with a visible crosswalk, multiple road users ahead of the ego vehicle, and roadside objects that narrow visual attention to the active traffic area.
"""


class TestVLMInterpreter(unittest.TestCase):
    def setUp(self):
        self.interpreter = VLMInterpreter()

    def test_validate_output_structure_accepts_new_template(self):
        self.assertIsNone(self.interpreter.validate_output_structure(VALID_OUTPUT))

    def test_validate_output_structure_rejects_missing_section(self):
        invalid_output = VALID_OUTPUT.replace("## Static Objects Description:\n", "")
        self.assertIsNotNone(
            self.interpreter.validate_output_structure(invalid_output)
        )

    def test_validate_output_structure_rejects_legacy_sections(self):
        legacy_output = """## Description
Old format

## Reasoning
Old format

## Decision
Old format
"""
        self.assertIsNotNone(
            self.interpreter.validate_output_structure(legacy_output)
        )


if __name__ == "__main__":
    unittest.main()
