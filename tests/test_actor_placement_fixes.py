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

from tools.structured_pipeline import _select_semantic_lane, MAX_LOCAL_LANE_DISTANCE_M


def _lane(road_id, lane_id, x, y):
    return {
        "road_id": road_id, "lane_id": lane_id,
        "start": {"x": x, "y": y, "z": 0.0, "yaw": 0.0},
        "end": {"x": x, "y": y + 20.0, "z": 0.0, "yaw": 0.0},
    }


class LocalLaneGuardTests(unittest.TestCase):
    """A right-lane actor must not snap onto a far leaked fallback lane."""

    def _entity(self):
        return {"lane_side_relation": "right_lane", "lane_index_relation": 1,
                "category": "car", "heading_relation": "unknown"}

    def test_far_lane_rejected_returns_anchor(self):
        anchor = _lane(1, -1, 0.0, 0.0)
        # The matching right lane (lane_id -2) only exists ~190m away (leaked
        # fallback): it must be rejected, falling back to the anchor.
        far = _lane(1, -2, 0.0, -190.0)
        chosen = _select_semantic_lane(self._entity(), [anchor, far], anchor)
        self.assertIs(chosen, anchor)

    def test_local_lane_is_used(self):
        anchor = _lane(1, -1, 0.0, 0.0)
        # Same right lane but genuinely local (within threshold) -> used.
        near = _lane(1, -2, 0.0, -40.0)
        self.assertLess(40.0, MAX_LOCAL_LANE_DISTANCE_M)
        chosen = _select_semantic_lane(self._entity(), [anchor, near], anchor)
        self.assertEqual(chosen["lane_id"], -2)
        self.assertAlmostEqual(chosen["start"]["y"], -40.0)


if __name__ == "__main__":
    unittest.main()
