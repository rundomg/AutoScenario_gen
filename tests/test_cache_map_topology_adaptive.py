import unittest
from types import SimpleNamespace

from tools.cache_map_topology import (
    _adaptive_waypoint_sample,
    compress_structural_candidates,
)


def _wp(s, *, road=1, lane=-1, section=0, x=None, y=0.0, yaw=0.0, junction=False):
    location = SimpleNamespace(x=float(s if x is None else x), y=float(y), z=0.0)
    rotation = SimpleNamespace(yaw=float(yaw))
    return SimpleNamespace(
        road_id=road,
        lane_id=lane,
        section_id=section,
        s=float(s),
        is_junction=junction,
        transform=SimpleNamespace(location=location, rotation=rotation),
    )


class AdaptiveWaypointSamplingTests(unittest.TestCase):
    def test_long_straight_uses_coarse_spacing_and_keeps_endpoints(self):
        points = [_wp(s) for s in range(0, 205, 5)]
        selected, stats = _adaptive_waypoint_sample(points, junction_radius_m=20.0)
        self.assertEqual([point.s for point in selected], [0, 50, 100, 150, 200])
        self.assertEqual(stats["endpoints"], 2)

    def test_junction_neighborhood_stays_dense(self):
        points = [_wp(s, junction=(s == 100)) for s in range(0, 205, 5)]
        selected, _ = _adaptive_waypoint_sample(points, junction_radius_m=20.0)
        selected_s = {point.s for point in selected}
        self.assertTrue(set(range(80, 125, 5)).issubset(selected_s))

    def test_curve_uses_medium_spacing(self):
        points = [_wp(s, yaw=s * 0.5) for s in range(0, 105, 5)]
        selected, stats = _adaptive_waypoint_sample(
            points,
            junction_radius_m=20.0,
            curve_yaw_threshold_deg=8.0,
        )
        self.assertGreater(stats["curve"], 0)
        gaps = [b.s - a.s for a, b in zip(selected, selected[1:])]
        self.assertLessEqual(max(gaps), 10.0)

    def test_each_lane_run_keeps_its_ends(self):
        points = [_wp(s, lane=-1) for s in (0, 5, 10)]
        points += [_wp(s, lane=-2) for s in (0, 5, 10)]
        selected, _ = _adaptive_waypoint_sample(points)
        self.assertEqual(len(selected), 4)

    def test_structural_compression_keeps_diverse_regions(self):
        candidates = []
        for x in (0, 20, 220, 420, 620):
            candidates.append(
                {
                    "location": {"x": x, "y": 0},
                    "yaw": 0,
                    "candidate_topology_type": "straight_two_way",
                    "same_direction_lane_count": 1,
                    "same_road_lane_count": 2,
                    "curve_direction": "straight",
                    "environment_context": {"environment_class": "urban_like"},
                }
            )
        selected, stats = compress_structural_candidates(
            candidates, max_representatives=3, region_size_m=200
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(stats["signature_count"], 1)
        selected_x = {item["location"]["x"] for item in selected}
        self.assertIn(0, selected_x)
        self.assertIn(620, selected_x)


if __name__ == "__main__":
    unittest.main()
