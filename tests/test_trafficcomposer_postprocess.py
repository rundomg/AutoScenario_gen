import sys
import unittest
from pathlib import Path


TRAFFICCOMPOSER_ROOT = Path(__file__).resolve().parents[2] / "TrafficComposer"
sys.path.insert(0, str(TRAFFICCOMPOSER_ROOT))

from trafficcomposer.gen_textual_ir.gen_textual_ir import post_process


class TrafficComposerPostProcessTests(unittest.TestCase):
    def test_preserves_direct_ego_and_road_relations(self):
        raw = """<YAML>
participant:
  ego_vehicle:
    position_target: intersection
    position_relation: behind
  lead:
    position_target: ego vehicle
    position_relation: front
</YAML>"""
        result = post_process(raw)
        self.assertIn("position_relation: behind", result)
        self.assertIn("position_relation: front", result)

    def test_flattens_one_hop_actor_relation_without_erasing_it(self):
        raw = """<YAML>
participant:
  lead:
    position_target: ego vehicle
    position_relation: front
  truck:
    position_target: lead
    position_relation: right
</YAML>"""
        result = post_process(raw)
        self.assertIn("position_relation: right front", result)
        self.assertIn("position_target: ego vehicle", result)


if __name__ == "__main__":
    unittest.main()
