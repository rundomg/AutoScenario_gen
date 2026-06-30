import math
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

from tools.structured_pipeline import (
    compute_cache_longitudinal_slide,
    slide_anchor_along_segment,
    _infer_ego_junction_target,
    _infer_ego_lane_offset,
    _ego_forward_reference_distance,
    MIN_JUNCTION_CLEARANCE_M,
)


class CacheSlideMathTests(unittest.TestCase):
    def test_slides_forward_to_target(self):
        # candidate is 50m before junction, want ego 28m before -> slide +22
        self.assertAlmostEqual(
            compute_cache_longitudinal_slide(50.0, 28.0), 22.0
        )

    def test_below_threshold_is_zero(self):
        self.assertEqual(compute_cache_longitudinal_slide(30.0, 28.0), 0.0)

    def test_never_overshoots_stop_line(self):
        # target 2m but clearance is 6m -> capped at 50 - 6 = 44
        self.assertAlmostEqual(
            compute_cache_longitudinal_slide(50.0, 2.0), 50.0 - MIN_JUNCTION_CLEARANCE_M
        )

    def test_backward_slide_allowed(self):
        # candidate closer (10m) than target (40m) -> move back (negative)
        self.assertAlmostEqual(compute_cache_longitudinal_slide(10.0, 40.0), -30.0)

    def test_none_distance_is_zero(self):
        self.assertEqual(compute_cache_longitudinal_slide(None, 28.0), 0.0)


class SlideAnchorTests(unittest.TestCase):
    def test_translation_along_forward(self):
        start = {"x": 0.0, "y": 0.0, "z": 0.3, "yaw": 90.0}
        end = {"x": 0.0, "y": 20.0}  # forward = +y
        out = slide_anchor_along_segment(start, end, 22.0)
        self.assertAlmostEqual(out["x"], 0.0)
        self.assertAlmostEqual(out["y"], 22.0)
        self.assertEqual(out["yaw"], 90.0)

    def test_degenerate_segment_no_move(self):
        start = {"x": 5.0, "y": 5.0}
        out = slide_anchor_along_segment(start, dict(start), 10.0)
        self.assertAlmostEqual(out["x"], 5.0)
        self.assertAlmostEqual(out["y"], 5.0)


class ForwardReferenceTests(unittest.TestCase):
    def test_direct_junction_distance(self):
        meta = {"ego_localization": {"ego_to_junction_distance_m": 25}}
        self.assertEqual(_ego_forward_reference_distance(meta), 25.0)

    def test_map_matching_junction_distance_fallback(self):
        self.assertEqual(
            _ego_forward_reference_distance(
                {},
                {"ego_to_junction_distance_m": 22},
            ),
            22.0,
        )

    def test_mappable_reference_type(self):
        meta = {"ego_localization": {"forward_reference": {"type": "traffic_light", "distance_m": 18}}}
        self.assertEqual(_ego_forward_reference_distance(meta), 18.0)

    def test_landmark_is_ignored(self):
        meta = {"ego_localization": {"forward_reference": {"type": "landmark", "distance_m": 30}}}
        self.assertIsNone(_ego_forward_reference_distance(meta))

    def test_missing_block(self):
        self.assertIsNone(_ego_forward_reference_distance({}))


class JunctionTargetPrecedenceTests(unittest.TestCase):
    def test_vlm_metric_takes_priority(self):
        su = {
            "metadata": {"ego_localization": {"ego_to_junction_distance_m": 15}},
            "road_network": {"junctions": [{"location": "ahead", "confidence": "high"}]},
        }
        constrain, target = _infer_ego_junction_target(su)
        self.assertTrue(constrain)
        self.assertEqual(target, 15.0)

    def test_falls_back_to_qualitative(self):
        su = {
            "metadata": {},
            "road_network": {"junctions": [{"location": "ahead", "confidence": "high"}]},
        }
        constrain, target = _infer_ego_junction_target(su)
        self.assertTrue(constrain)
        self.assertEqual(target, 28.0)

    def test_no_junction(self):
        su = {"metadata": {}, "road_network": {"junctions": []}}
        constrain, target = _infer_ego_junction_target(su)
        self.assertFalse(constrain)

    def test_straight_map_matching_ignores_stray_junction_distance(self):
        su = {
            "metadata": {},
            "road_network": {
                "map_matching": {
                    "topology_type": "straight_two_way",
                    "junction_visible": False,
                    "ego_to_junction_distance_m": 20,
                },
                "junctions": [],
            },
        }
        constrain, target = _infer_ego_junction_target(su)
        self.assertFalse(constrain)
        self.assertEqual(target, 0.0)


class EgoLaneFromRightTests(unittest.TestCase):
    def test_explicit_used_when_supported_by_actors(self):
        # explicit=1 and a same-direction actor two lanes right -> heuristic=2,
        # reconciled min(1, 2) = 1.
        su = {
            "metadata": {"ego_localization": {"ego_lane_from_right": 1}},
            "road_network": {"lane_groups": [{"forward_lane_count": 3}]},
            "traffic_subjects": [
                {"heading_relation_to_ego": "same_direction", "lane_index_relation": 2},
            ],
        }
        self.assertEqual(_infer_ego_lane_offset(su, forward_lane_count=3), 1)

    def test_map_matching_ego_lane_from_right_used(self):
        su = {
            "metadata": {},
            "road_network": {
                "map_matching": {"ego_lane_from_right": 1},
                "lane_groups": [{"forward_lane_count": 3}],
            },
            "traffic_subjects": [
                {"heading_relation_to_ego": "same_direction", "lane_index_relation": 2},
            ],
        }
        self.assertEqual(_infer_ego_lane_offset(su, forward_lane_count=3), 1)

    def test_explicit_sparse_junction_lane_is_trusted(self):
        # explicit=1 with no same-direction actors still means ego is one lane
        # left of the rightmost lane; sparse junction scenes often have only
        # crossing / oncoming visible actors.
        su = {
            "metadata": {"ego_localization": {"ego_lane_from_right": 1}},
            "road_network": {"lane_groups": [{"forward_lane_count": 2}]},
            "traffic_subjects": [
                {"heading_relation_to_ego": "crossing", "lane_index_relation": 1},
                {"heading_relation_to_ego": "unknown", "lane_index_relation": 2},
            ],
        }
        self.assertEqual(_infer_ego_lane_offset(su, forward_lane_count=2), 1)

    def test_falls_back_to_heuristic(self):
        # no explicit signal -> uses traffic_subjects max-right
        su = {
            "metadata": {},
            "road_network": {"lane_groups": [{"forward_lane_count": 3}]},
            "traffic_subjects": [
                {"heading_relation_to_ego": "same_direction", "lane_index_relation": 1},
            ],
        }
        self.assertEqual(_infer_ego_lane_offset(su, forward_lane_count=3), 1)


class CacheOnlyIndexMethodTests(unittest.TestCase):
    """_index_ego_longitudinal_cache_only slides both endpoints, keeps heading."""

    def _make_generator(self, spawn_context):
        # Avoid AutoGenerator.__init__ (constructs many agents); we only exercise
        # one method that reads/writes self.carla_spawn_context.
        for mod in ("matplotlib", "matplotlib.pyplot"):
            sys.modules.setdefault(mod, types.ModuleType(mod))
        from experiments.auto_generate_all_vlm import AutoGenerator

        gen = AutoGenerator.__new__(AutoGenerator)
        gen.carla_spawn_context = spawn_context
        return gen

    def test_forward_slide_preserves_direction(self):
        su = {
            "metadata": {"ego_localization": {"ego_to_junction_distance_m": 28}},
            "road_network": {"junctions": [{"location": "ahead", "confidence": "high"}]},
        }
        sample = [{
            "road_id": 30, "lane_id": -2,
            "start": {"x": 0.0, "y": 0.0, "z": 0.3, "yaw": 90.0},
            "end": {"x": 0.0, "y": 20.0, "z": 0.3, "yaw": 90.0},
            "distance_to_junction_ahead": 50.0,
        }]
        gen = self._make_generator({"topology_sample": sample})
        gen._index_ego_longitudinal_cache_only(su, sample)

        updated = gen.carla_spawn_context["topology_sample"][0]
        # 50 -> 28 means slide +22 along +y.
        self.assertAlmostEqual(updated["start"]["y"], 22.0)
        self.assertAlmostEqual(updated["end"]["y"], 42.0)
        # Forward direction (end-start) unchanged: still +y, magnitude 20.
        fwd = updated["end"]["y"] - updated["start"]["y"]
        self.assertAlmostEqual(fwd, 20.0)
        self.assertAlmostEqual(updated["distance_to_junction_ahead"], 28.0)

    def test_noop_when_no_junction(self):
        su = {"metadata": {}, "road_network": {"junctions": []}}
        sample = [{
            "start": {"x": 0.0, "y": 0.0}, "end": {"x": 0.0, "y": 20.0},
            "distance_to_junction_ahead": 50.0,
        }]
        gen = self._make_generator({"topology_sample": sample})
        gen._index_ego_longitudinal_cache_only(su, sample)
        self.assertAlmostEqual(
            gen.carla_spawn_context["topology_sample"][0]["start"]["y"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
