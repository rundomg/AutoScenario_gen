import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "experiments" / "merge_trafficcomposer_reproducible_ir.py"
SPEC = importlib.util.spec_from_file_location("merge_trafficcomposer_ir", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MergeTrafficComposerIRTests(unittest.TestCase):
    def test_fusion_keeps_text_relation_and_adds_visual_lane(self):
        textual = {
            "road_network": {"road_type": "intersection", "lane_number": None},
            "participant": {
                "ego_vehicle": {"position_relation": "behind"},
                "other_actor_1": {"type": "car", "position_target": "ego vehicle", "position_relation": "front"},
            },
        }
        visual = {
            "road_network": {"lane_number": 2},
            "participant": {
                "ego_vehicle": {"lane_idx": 1},
                "other_actor_1": {"type": "car", "lane_idx": 1, "position_target": "ego_vehicle", "position_relation": "front"},
            },
        }
        fused = MODULE.fuse_ir(textual, visual)
        self.assertEqual(fused["participant"]["other_actor_1"]["position_relation"], "front")
        self.assertEqual(fused["participant"]["other_actor_1"]["lane_idx"], 1)
        self.assertEqual(fused["participant"]["ego_vehicle"]["lane_idx"], 1)
        self.assertEqual(fused["road_network"]["lane_number"], 2)

    def test_unmatched_visual_actor_is_added(self):
        textual = {"participant": {"ego_vehicle": {}}}
        visual = {"participant": {"ego_vehicle": {}, "other_actor_1": {"type": "truck", "position_relation": "right front"}}}
        fused = MODULE.fuse_ir(textual, visual)
        self.assertEqual(fused["participant"]["other_actor_1"]["type"], "truck")
        self.assertEqual(fused["fusion_metadata"]["added_visual_actors"], 1)


if __name__ == "__main__":
    unittest.main()
