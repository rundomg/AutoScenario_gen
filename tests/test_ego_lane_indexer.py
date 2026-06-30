"""Unit tests for ego lane indexing helpers in tools/structured_pipeline.py.

All tests use plain Python dicts — no numpy, cv2, carla, or network access required.
"""

import math
import sys
import unittest
import unittest.mock as mock

# Stub heavy optional deps so structured_pipeline can be imported in any env.
for _mod in ("numpy", "cv2", "dotenv", "requests", "PIL", "Pillow"):
    sys.modules.setdefault(_mod, mock.MagicMock())

from tools.structured_pipeline import (  # noqa: E402
    _find_lane_in_dense,
    _infer_ego_junction_target,
    _infer_ego_lane_offset,
    _measure_forward_junction_distance,
    _next_wp_along_lane,
    _scene_forward_lane_count,
    _scene_has_center_median,
    _slide_along_dense_waypoints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wp(x=0.0, y=0.0, z=0.0, yaw=0.0, road_id=1, lane_id=-1, is_junction=False):
    return {
        "x": x, "y": y, "z": z, "yaw": yaw,
        "road_id": road_id, "lane_id": lane_id,
        "lane_width": 3.5, "is_junction": is_junction,
    }


def _candidate(road_id=1, lane_id=-1, sx=0.0, sy=0.0, ex=20.0, ey=0.0):
    return {
        "road_id": road_id,
        "lane_id": lane_id,
        "start": {"x": sx, "y": sy, "z": 0.0, "yaw": 0.0, "is_junction": False},
        "end":   {"x": ex, "y": ey, "z": 0.0, "yaw": 0.0, "is_junction": False},
    }


# ---------------------------------------------------------------------------
# _infer_ego_lane_offset
# ---------------------------------------------------------------------------

class TestInferEgoLaneOffset(unittest.TestCase):
    # These cases supply no road_network, so forward_lane_count is unknown and
    # the function falls back to legacy max-right behaviour (clamped at 0).

    def test_no_subjects_returns_zero(self):
        su = {"traffic_subjects": []}
        self.assertEqual(_infer_ego_lane_offset(su), 0)

    def test_missing_traffic_subjects_key(self):
        self.assertEqual(_infer_ego_lane_offset({}), 0)

    def test_none_traffic_subjects(self):
        self.assertEqual(_infer_ego_lane_offset({"traffic_subjects": None}), 0)

    def test_single_right_vehicle(self):
        su = {"traffic_subjects": [{"lane_index_relation": 1}]}
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_multiple_right_vehicles_takes_max(self):
        su = {"traffic_subjects": [
            {"lane_index_relation": 1},
            {"lane_index_relation": 2},
            {"lane_index_relation": 0},
        ]}
        self.assertEqual(_infer_ego_lane_offset(su), 2)

    def test_left_vehicles_ignored(self):
        su = {"traffic_subjects": [
            {"lane_index_relation": -1},
            {"lane_index_relation": -2},
        ]}
        self.assertEqual(_infer_ego_lane_offset(su), 0)

    def test_non_integer_lane_index_skipped(self):
        su = {"traffic_subjects": [
            {"lane_index_relation": "n/a"},
            {"lane_index_relation": None},
            {"lane_index_relation": 2},
        ]}
        self.assertEqual(_infer_ego_lane_offset(su), 2)

    def test_string_integer_coerced(self):
        su = {"traffic_subjects": [{"lane_index_relation": "3"}]}
        self.assertEqual(_infer_ego_lane_offset(su), 3)

    def test_mixed_lane_indices(self):
        su = {"traffic_subjects": [
            {"lane_index_relation": -1},
            {"lane_index_relation": 0},
            {"lane_index_relation": 1},
        ]}
        self.assertEqual(_infer_ego_lane_offset(su), 1)


# ---------------------------------------------------------------------------
# _infer_ego_lane_offset — reconciliation with forward_lane_count + median
# ---------------------------------------------------------------------------

class TestInferEgoLaneOffsetReconciled(unittest.TestCase):

    @staticmethod
    def _su(subjects, forward_lane_count=3, divided=True, special=None):
        road_network = {
            "directionality": "divided_two_way" if divided else "one_way",
            "lane_groups": [{"forward_lane_count": forward_lane_count}],
        }
        if special is not None:
            road_network["special_road_areas"] = special
        return {"road_network": road_network, "traffic_subjects": subjects}

    def test_regression_gore_van_does_not_push_ego_off_middle(self):
        # Reproduces results/auto_result_20260617_220440: ego is in the middle
        # of 3 forward lanes (left neighbour at -1, right neighbour at +1), but a
        # van in the right-side gore area reports +2. With a center median the
        # left neighbour anchors ego → offset 1 (middle), not 2 (leftmost).
        su = self._su([
            {"lane_index_relation": 0, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": 2, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": -1, "heading_relation_to_ego": "same_direction"},
        ])
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_ego_leftmost_when_no_left_neighbour(self):
        # L=0, N=3, median → offset (3-1)-0 = 2 (leftmost forward lane).
        su = self._su([
            {"lane_index_relation": 0, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": 2, "heading_relation_to_ego": "same_direction"},
        ])
        self.assertEqual(_infer_ego_lane_offset(su), 2)

    def test_opposite_direction_not_counted_as_left(self):
        # Opposing actors at -2 sit across the median and must not increase L.
        su = self._su([
            {"lane_index_relation": -1, "heading_relation_to_ego": "same_direction"},
            {"lane_index_relation": -2, "heading_relation_to_ego": "opposite_direction"},
            {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
        ])
        # L=1 (only same-direction), N=3, median → offset (3-1)-1 = 1.
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_no_median_uses_right_neighbour_clamped(self):
        # Without a median the right-neighbour count is used, clamped to N-1.
        su = self._su(
            [
                {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
                {"lane_index_relation": 5, "heading_relation_to_ego": "same_direction"},
            ],
            forward_lane_count=3,
            divided=False,
        )
        self.assertEqual(_infer_ego_lane_offset(su), 2)  # clamp(5, 0, 2)

    def test_gore_subject_excluded_by_evidence(self):
        su = self._su(
            [
                {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
                {
                    "lane_index_relation": 3,
                    "heading_relation_to_ego": "same_direction",
                    "evidence": "van in the right-side gore / merge area",
                },
            ],
            forward_lane_count=4,
            divided=False,
        )
        # The +3 gore subject is dropped → max_right = 1.
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_alongside_subject_excluded(self):
        su = self._su(
            [
                {"lane_index_relation": 1, "heading_relation_to_ego": "same_direction"},
                {
                    "lane_index_relation": 3,
                    "heading_relation_to_ego": "same_direction",
                    "longitudinal_relation": "alongside",
                },
            ],
            forward_lane_count=4,
            divided=False,
        )
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_special_area_median_detected(self):
        su = self._su(
            [
                {"lane_index_relation": 0, "heading_relation_to_ego": "same_direction"},
                {"lane_index_relation": 2, "heading_relation_to_ego": "same_direction"},
                {"lane_index_relation": -1, "heading_relation_to_ego": "same_direction"},
            ],
            divided=False,
            special=[{"type": "raised_center_median_barrier", "side": "left"}],
        )
        # Median detected via special_road_areas → left-anchored: (3-1)-1 = 1.
        self.assertEqual(_infer_ego_lane_offset(su), 1)

    def test_explicit_forward_lane_count_param_overrides(self):
        su = {
            "road_network": {"directionality": "divided_two_way", "lane_groups": []},
            "traffic_subjects": [
                {"lane_index_relation": -1, "heading_relation_to_ego": "same_direction"},
            ],
        }
        # No lane_groups → derived N=0, but explicit N=3 activates reconciliation.
        self.assertEqual(_infer_ego_lane_offset(su, forward_lane_count=3), 1)


class TestSceneMedianAndLaneCountHelpers(unittest.TestCase):

    def test_divided_directionality_is_median(self):
        self.assertTrue(
            _scene_has_center_median({"road_network": {"directionality": "divided_two_way"}})
        )

    def test_lane_marking_divided_directionality_is_not_median(self):
        su = {
            "road_network": {
                "directionality": "divided_by_lane_markings_not_median_visible",
                "special_road_areas": [{"type": "center_median", "presence": False}],
            }
        }
        self.assertFalse(_scene_has_center_median(su))

    def test_one_way_is_not_median(self):
        self.assertFalse(
            _scene_has_center_median({"road_network": {"directionality": "one_way"}})
        )

    def test_special_area_barrier_is_median(self):
        su = {"road_network": {"special_road_areas": [{"type": "concrete_center_divider"}]}}
        self.assertTrue(_scene_has_center_median(su))

    def test_decisive_cue_median(self):
        su = {"metadata": {"decisive_map_matching_cues": {"has_center_median": True}}}
        self.assertTrue(_scene_has_center_median(su))

    def test_forward_lane_count_reads_first_group(self):
        su = {"road_network": {"lane_groups": [{"forward_lane_count": 4}]}}
        self.assertEqual(_scene_forward_lane_count(su), 4)

    def test_forward_lane_count_missing_returns_zero(self):
        self.assertEqual(_scene_forward_lane_count({}), 0)


# ---------------------------------------------------------------------------
# _find_lane_in_dense
# ---------------------------------------------------------------------------

class TestFindLaneInDense(unittest.TestCase):

    def _three_lane_wps(self):
        # Three driving lanes on road 1, each with two waypoints spaced 2m apart.
        # Candidate forward is +x (yaw 0); CARLA right is +y, so increasing y is
        # toward the curb: lane -1 is innermost (left), lane -3 is rightmost.
        return [
            _wp(x=0.0,  y=0.0,  road_id=1, lane_id=-1),
            _wp(x=2.0,  y=0.0,  road_id=1, lane_id=-1),
            _wp(x=0.0,  y=3.5,  road_id=1, lane_id=-2),
            _wp(x=2.0,  y=3.5,  road_id=1, lane_id=-2),
            _wp(x=0.0,  y=7.0,  road_id=1, lane_id=-3),
            _wp(x=2.0,  y=7.0,  road_id=1, lane_id=-3),
        ]

    def test_offset_0_returns_rightmost(self):
        # Rightmost = largest |lane_id| = curb lane -3, NOT the innermost -1.
        wp = _find_lane_in_dense(_candidate(road_id=1, lane_id=-1), 0, self._three_lane_wps())
        self.assertIsNotNone(wp)
        self.assertEqual(wp["lane_id"], -3)

    def test_offset_1_returns_second_lane(self):
        wp = _find_lane_in_dense(_candidate(road_id=1, lane_id=-1), 1, self._three_lane_wps())
        self.assertIsNotNone(wp)
        self.assertEqual(wp["lane_id"], -2)

    def test_offset_2_returns_third_lane(self):
        # Leftmost (innermost) forward lane = |id|=1.
        wp = _find_lane_in_dense(_candidate(road_id=1, lane_id=-1), 2, self._three_lane_wps())
        self.assertIsNotNone(wp)
        self.assertEqual(wp["lane_id"], -1)

    def test_offset_beyond_available_returns_none(self):
        wp = _find_lane_in_dense(_candidate(road_id=1, lane_id=-1), 5, self._three_lane_wps())
        self.assertIsNone(wp)

    def test_different_road_id_returns_none(self):
        wp = _find_lane_in_dense(_candidate(road_id=99, lane_id=-1), 0, self._three_lane_wps())
        self.assertIsNone(wp)

    def test_positive_lane_ids_excluded(self):
        # Positive lane_ids = opposing direction, should be filtered out.
        wps = [
            _wp(x=0.0, y=0.0, road_id=1, lane_id=1),   # opposing
            _wp(x=0.0, y=3.5, road_id=1, lane_id=-1),  # driving
        ]
        wp = _find_lane_in_dense(_candidate(road_id=1, lane_id=-1), 0, wps)
        self.assertEqual(wp["lane_id"], -1)

    def test_deduplicates_by_closest_to_start(self):
        # Two waypoints for lane_id=-1; one closer (5m), one farther (50m).
        wps = [
            _wp(x=5.0,  y=0.0, road_id=1, lane_id=-1),   # 5m from start
            _wp(x=50.0, y=0.0, road_id=1, lane_id=-1),   # 50m from start
        ]
        cand = _candidate(road_id=1, lane_id=-1, sx=0.0, sy=0.0)
        wp = _find_lane_in_dense(cand, 0, wps)
        self.assertAlmostEqual(wp["x"], 5.0)

    def test_empty_dense_wps_returns_none(self):
        wp = _find_lane_in_dense(_candidate(), 0, [])
        self.assertIsNone(wp)

    def test_positive_anchor_keeps_same_direction_no_flip(self):
        # When the matcher anchors on a POSITIVE lane_id (its driving direction),
        # ego must stay on the positive carriageway — selecting negative lanes
        # would flip ego 180° onto the opposing direction (junction-behind bug).
        wps = [
            _wp(x=0.0, y=0.0, road_id=42, lane_id=1, yaw=7.0),    # anchor lane (driving)
            _wp(x=0.0, y=3.5, road_id=42, lane_id=2, yaw=7.0),    # lane to the right (driving)
            _wp(x=0.0, y=-3.5, road_id=42, lane_id=-1, yaw=187.0),  # opposing carriageway
        ]
        cand = _candidate(road_id=42, lane_id=1, sx=0.0, sy=0.0, ex=20.0, ey=0.0)
        # offset 0 = rightmost same-direction lane (largest id on this carriageway).
        wp0 = _find_lane_in_dense(cand, 0, wps)
        self.assertEqual(wp0["lane_id"], 2)
        wp1 = _find_lane_in_dense(cand, 1, wps)
        self.assertEqual(wp1["lane_id"], 1)
        # The opposing (negative) lane must never be selected for a positive anchor.
        self.assertNotIn(_find_lane_in_dense(cand, 0, wps)["lane_id"], (-1,))


# ---------------------------------------------------------------------------
# _next_wp_along_lane
# ---------------------------------------------------------------------------

class TestNextWpAlongLane(unittest.TestCase):

    def _lane_wps(self, road_id=1, lane_id=-1, yaw=0.0):
        # Waypoints spaced 5m ahead (positive x = forward for yaw=0).
        return [
            _wp(x=5.0,  y=0.0, road_id=road_id, lane_id=lane_id, yaw=yaw),
            _wp(x=10.0, y=0.0, road_id=road_id, lane_id=lane_id, yaw=yaw),
            _wp(x=15.0, y=0.0, road_id=road_id, lane_id=lane_id, yaw=yaw),
            _wp(x=20.0, y=0.0, road_id=road_id, lane_id=lane_id, yaw=yaw),
            _wp(x=25.0, y=0.0, road_id=road_id, lane_id=lane_id, yaw=yaw),
        ]

    def test_returns_closest_to_lookahead(self):
        origin = _wp(x=0.0, y=0.0, road_id=1, lane_id=-1, yaw=0.0)
        wps = self._lane_wps()
        result = _next_wp_along_lane(origin, wps, lookahead_m=20.0)
        self.assertAlmostEqual(result["x"], 20.0)

    def test_no_forward_candidates_falls_back_to_self(self):
        origin = _wp(x=100.0, y=0.0, road_id=1, lane_id=-1, yaw=0.0)
        wps = self._lane_wps()
        result = _next_wp_along_lane(origin, wps, lookahead_m=20.0)
        # All wps are behind origin; should return origin itself.
        self.assertAlmostEqual(result["x"], 100.0)

    def test_empty_dense_falls_back_to_self(self):
        origin = _wp(x=0.0, y=0.0, road_id=1, lane_id=-1, yaw=0.0)
        result = _next_wp_along_lane(origin, [], lookahead_m=20.0)
        self.assertIs(result, origin)

    def test_different_lane_ignored(self):
        origin = _wp(x=0.0, y=0.0, road_id=1, lane_id=-1, yaw=0.0)
        wps = [_wp(x=20.0, y=0.0, road_id=1, lane_id=-2, yaw=0.0)]
        result = _next_wp_along_lane(origin, wps, lookahead_m=20.0)
        self.assertIs(result, origin)


# ---------------------------------------------------------------------------
# _infer_ego_junction_target
# ---------------------------------------------------------------------------

class TestInferEgoJunctionTarget(unittest.TestCase):

    def _su_with_junction(self, location="ahead", confidence=None):
        j = {"location": location}
        if confidence is not None:
            j["confidence"] = confidence
        return {"road_network": {"junctions": [j]}}

    def test_no_junction_returns_false(self):
        constrain, dist = _infer_ego_junction_target({"road_network": {"junctions": []}})
        self.assertFalse(constrain)
        self.assertEqual(dist, 0.0)

    def test_missing_road_network(self):
        constrain, _ = _infer_ego_junction_target({})
        self.assertFalse(constrain)

    def test_immediate_location_maps_to_8m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("immediate"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 8.0)

    def test_at_junction_location_maps_to_8m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("at junction"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 8.0)

    def test_entering_maps_to_8m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("entering"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 8.0)

    def test_close_maps_to_18m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("close"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 18.0)

    def test_approaching_maps_to_18m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("approaching"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 18.0)

    def test_generic_ahead_maps_to_28m(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("ahead"))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 28.0)

    def test_far_junction_skipped(self):
        constrain, _ = _infer_ego_junction_target(self._su_with_junction("far ahead"))
        self.assertFalse(constrain)

    def test_distant_junction_skipped(self):
        constrain, _ = _infer_ego_junction_target(self._su_with_junction("distant"))
        self.assertFalse(constrain)

    def test_low_confidence_junction_skipped(self):
        constrain, _ = _infer_ego_junction_target(self._su_with_junction("ahead", confidence="low"))
        self.assertFalse(constrain)

    def test_numeric_low_confidence_skipped(self):
        constrain, _ = _infer_ego_junction_target(self._su_with_junction("ahead", confidence=0.3))
        self.assertFalse(constrain)

    def test_high_confidence_not_skipped(self):
        constrain, dist = _infer_ego_junction_target(self._su_with_junction("ahead", confidence=0.9))
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 28.0)

    def test_decisive_cue_false_overrides_junctions(self):
        su = {
            "road_network": {"junctions": [{"location": "ahead"}]},
            "metadata": {"decisive_map_matching_cues": {"junction_visible": False}},
        }
        constrain, _ = _infer_ego_junction_target(su)
        self.assertFalse(constrain)

    def test_decisive_cue_true_fallback(self):
        # junction_visible=True but no junction objects → default 28m.
        su = {
            "road_network": {"junctions": []},
            "metadata": {"decisive_map_matching_cues": {"junction_visible": True}},
        }
        constrain, dist = _infer_ego_junction_target(su)
        self.assertTrue(constrain)
        self.assertAlmostEqual(dist, 28.0)


# ---------------------------------------------------------------------------
# _measure_forward_junction_distance
# ---------------------------------------------------------------------------

class TestMeasureForwardJunctionDistance(unittest.TestCase):

    def _start_end(self, x=0.0, y=0.0, fwd_x=30.0, fwd_y=0.0):
        return (
            {"x": x,     "y": y,     "yaw": 0.0},
            {"x": fwd_x, "y": fwd_y, "yaw": 0.0},
        )

    def test_junction_directly_ahead(self):
        start, end = self._start_end()
        wps = [_wp(x=25.0, y=0.0, is_junction=True)]
        dist = _measure_forward_junction_distance(start, end, wps)
        self.assertAlmostEqual(dist, 25.0, places=1)

    def test_junction_behind_excluded(self):
        start, end = self._start_end()
        wps = [_wp(x=-10.0, y=0.0, is_junction=True)]
        dist = _measure_forward_junction_distance(start, end, wps)
        self.assertIsNone(dist)

    def test_no_junction_waypoints_returns_none(self):
        start, end = self._start_end()
        wps = [_wp(x=10.0, y=0.0, is_junction=False)]
        dist = _measure_forward_junction_distance(start, end, wps)
        self.assertIsNone(dist)

    def test_returns_nearest_of_multiple_junctions(self):
        start, end = self._start_end()
        wps = [
            _wp(x=40.0, y=0.0, is_junction=True),
            _wp(x=15.0, y=0.0, is_junction=True),
            _wp(x=60.0, y=0.0, is_junction=True),
        ]
        dist = _measure_forward_junction_distance(start, end, wps)
        self.assertAlmostEqual(dist, 15.0, places=1)

    def test_lateral_junction_still_counted(self):
        # Junction is slightly to the side; longitudinal component still positive.
        start, end = self._start_end()
        wps = [_wp(x=20.0, y=5.0, is_junction=True)]
        dist = _measure_forward_junction_distance(start, end, wps)
        # Longitudinal component along x-axis = 20.0.
        self.assertAlmostEqual(dist, 20.0, places=1)

    def test_empty_wps_returns_none(self):
        start, end = self._start_end()
        dist = _measure_forward_junction_distance(start, end, [])
        self.assertIsNone(dist)

    def test_degenerate_start_end_uses_yaw(self):
        # start == end; function falls back to start yaw.
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end = {"x": 0.0, "y": 0.0}
        wps = [_wp(x=30.0, y=0.0, is_junction=True)]
        dist = _measure_forward_junction_distance(start, end, wps)
        self.assertAlmostEqual(dist, 30.0, places=1)


# ---------------------------------------------------------------------------
# _slide_along_dense_waypoints
# ---------------------------------------------------------------------------

class TestSlideAlongDenseWaypoints(unittest.TestCase):

    def _make_lane_wps(self, road_id=1, lane_id=-1, n=20, spacing=2.0):
        """n waypoints spaced spacing metres apart along +x."""
        wps = []
        for i in range(n):
            x = float(i) * spacing
            wps.append(_wp(x=x, y=0.0, road_id=road_id, lane_id=lane_id))
        return wps

    def test_slide_forward(self):
        wps = self._make_lane_wps()
        cand = _candidate(road_id=1, lane_id=-1, sx=0.0, sy=0.0, ex=20.0, ey=0.0)
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 20.0, "y": 0.0, "yaw": 0.0}
        result = _slide_along_dense_waypoints(start, end, 10.0, wps, cand)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["x"], 10.0, delta=2.0)

    def test_slide_backward(self):
        wps = self._make_lane_wps()
        cand = _candidate(road_id=1, lane_id=-1, sx=10.0, sy=0.0, ex=30.0, ey=0.0)
        start = {"x": 10.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 30.0, "y": 0.0, "yaw": 0.0}
        # Slide backward by 8m from x=10 → target x ≈ 2.
        result = _slide_along_dense_waypoints(start, end, -8.0, wps, cand)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["x"], 2.0, delta=2.0)

    def test_junction_clearance_enforced(self):
        # Junction at x=20; clearance=5m; trying to slide to x=18 should be blocked.
        wps = self._make_lane_wps()
        wps.append(_wp(x=20.0, y=0.0, road_id=1, lane_id=-1, is_junction=True))
        cand = _candidate(road_id=1, lane_id=-1, sx=0.0, sy=0.0, ex=20.0, ey=0.0)
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 20.0, "y": 0.0, "yaw": 0.0}
        # Trying to land at x=18 (2m from junction) should be rejected.
        result = _slide_along_dense_waypoints(start, end, 18.0, wps, cand)
        if result is not None:
            dist_to_junction = 20.0 - result["x"]
            self.assertGreaterEqual(dist_to_junction, 5.0)

    def test_wrong_road_id_returns_none(self):
        wps = self._make_lane_wps(road_id=1)
        cand = _candidate(road_id=99, lane_id=-1)
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 20.0, "y": 0.0, "yaw": 0.0}
        result = _slide_along_dense_waypoints(start, end, 5.0, wps, cand)
        self.assertIsNone(result)

    def test_wrong_lane_id_returns_none(self):
        wps = self._make_lane_wps(lane_id=-2)
        cand = _candidate(road_id=1, lane_id=-1)
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 20.0, "y": 0.0, "yaw": 0.0}
        result = _slide_along_dense_waypoints(start, end, 5.0, wps, cand)
        self.assertIsNone(result)

    def test_empty_wps_returns_none(self):
        cand = _candidate()
        start = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        end   = {"x": 20.0, "y": 0.0, "yaw": 0.0}
        result = _slide_along_dense_waypoints(start, end, 5.0, [], cand)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
