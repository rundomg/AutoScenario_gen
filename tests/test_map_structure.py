"""Mock-based unit tests for tools.map_structure (no CARLA required).

CARLA-dependent paths are validated against a live server separately; these
tests lock the structure-building *logic* (leg clustering, curve sampling,
junction-ahead dispatch) so it stays correct in CI where CARLA is absent.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.map_structure import (  # noqa: E402
    _circular_mean,
    _cluster_headings,
    _leg_relation_to_ego,
    build_road_structure,
    build_matched_structure_from_waypoint,
    enumerate_junction_legs,
)
from tools.reference_frame import build_reference_frame, angle_difference  # noqa: E402


# --------------------------------------------------------------------------- #
# Minimal CARLA-shaped fakes
# --------------------------------------------------------------------------- #
class _Vec:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Rot:
    def __init__(self, yaw):
        self.yaw = yaw


class _Tf:
    def __init__(self, x, y, yaw, z=0.0):
        self.location = _Vec(x, y, z)
        self.rotation = _Rot(yaw)


class _BB:
    def __init__(self, x, y, z=0.0, ex=8.0, ey=8.0):
        self.location = _Vec(x, y, z)
        self.extent = _Vec(ex, ey, 0.0)


class _WP:
    def __init__(
        self,
        x,
        y,
        yaw,
        road_id=1,
        lane_id=-1,
        is_junction=False,
        chain=None,
        previous_chain=None,
    ):
        self.transform = _Tf(x, y, yaw)
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self._chain = chain or []
        self._previous_chain = previous_chain or []
        self._junction = None

    def next(self, step):
        return [self._chain.pop(0)] if self._chain else []

    def previous(self, step):
        return [self._previous_chain.pop(0)] if self._previous_chain else []

    def get_junction(self):
        return self._junction


class _Junction:
    def __init__(self, jid, center_xy, pairs):
        self.id = jid
        self.bounding_box = _BB(*center_xy)
        self._pairs = pairs

    def get_waypoints(self, lane_type):
        return self._pairs


class PureHelperTests(unittest.TestCase):
    def test_circular_mean_wraps(self):
        self.assertAlmostEqual(_circular_mean([170.0, -170.0]), 180.0, places=3)

    def test_cluster_merges_close_and_splits_far(self):
        clusters = _cluster_headings(
            [(0.0, "in"), (5.0, "out"), (90.0, "in"), (-178.0, "in"), (179.0, "out")]
        )
        # 0/5 merge, 90 alone, -178/179 merge (wrap) => 3 legs.
        self.assertEqual(len(clusters), 3)

    def test_leg_relation_to_ego(self):
        # Ego inbound yaw 0 (east) => ego leg points west (-180).
        self.assertEqual(_leg_relation_to_ego(-180.0, 0.0, -180.0), "ego")
        self.assertEqual(_leg_relation_to_ego(0.0, 0.0, -180.0), "opposite")
        self.assertEqual(_leg_relation_to_ego(-90.0, 0.0, -180.0), "left")
        self.assertEqual(_leg_relation_to_ego(90.0, 0.0, -180.0), "right")


class JunctionLegEnumerationTests(unittest.TestCase):
    def test_four_way_legs_from_entry_exit_pairs(self):
        center = (0.0, 0.0)
        # A pair is one path THROUGH the junction: entry yaw points inward (leg
        # out = inbound + 180), exit yaw is already the leg's outward heading.
        # Provide entries/exits covering four legs whose out headings are
        # {0, 90, 180, -90}.
        def wp(yaw):
            return _WP(0, 0, yaw)
        pairs = [
            (wp(180.0), wp(0.0)),    # in from east leg(out 0) -> out east-opposite leg(out 0)/...
            (wp(-90.0), wp(90.0)),
            (wp(0.0), wp(180.0)),
            (wp(90.0), wp(-90.0)),
        ]
        j = _Junction(7, center, pairs)
        c, legs = enumerate_junction_legs(j, lane_type="Driving")
        outs = sorted(round(l["heading_out_deg"]) for l in legs)
        self.assertEqual(outs, [-90, 0, 90, 180])
        for leg in legs:
            self.assertTrue(leg["has_inbound"] and leg["has_outbound"])

    def test_lane_records_include_arm_anchors(self):
        entry_anchor = _WP(-20, 0, 0.0, road_id=10, lane_id=1)
        exit_anchor = _WP(20, 0, 180.0, road_id=11, lane_id=-1)
        entry = _WP(
            -8,
            0,
            0.0,
            road_id=10,
            lane_id=1,
            previous_chain=[entry_anchor],
        )
        exit_wp = _WP(
            8,
            0,
            180.0,
            road_id=11,
            lane_id=-1,
            chain=[exit_anchor],
        )
        j = _Junction(8, (0.0, 0.0), [(entry, exit_wp)])

        _, legs = enumerate_junction_legs(j, lane_type="Driving")

        inbound = next(lane for leg in legs for lane in leg["inbound_lanes"])
        outbound = next(lane for leg in legs for lane in leg["outbound_lanes"])
        self.assertEqual(inbound["anchor"]["x"], -20.0)
        self.assertEqual(outbound["anchor"]["x"], 20.0)
        self.assertAlmostEqual(inbound["anchor"]["distance_from_center_m"], 20.0)


class RoadStructureTests(unittest.TestCase):
    def test_curve_samples_follow_chain(self):
        # A bend: yaw ramps 0 -> -10 -> -20 along a linked next() chain.
        w2 = _WP(0, 0, -20.0)
        w1 = _WP(0, 0, -10.0, chain=[w2])
        head = _WP(10.0, 5.0, 0.0, road_id=42, lane_id=-2, chain=[w1])
        rs = build_road_structure(head, lookahead_m=10.0, step_m=5.0)
        self.assertEqual(rs["kind"], "road_segment")
        self.assertEqual(rs["road_id"], 42)
        self.assertEqual(rs["lane_id"], -2)
        self.assertEqual([round(s[1]) for s in rs["curve_samples"]], [0, -10, -20])
        # Round-trips into a usable curved reference frame.
        frame = build_reference_frame(rs)
        far = frame.place(longitudinal_m=10.0, lateral_m=0.0)
        self.assertAlmostEqual(angle_difference(far.yaw, -20.0), 0.0, places=3)

    def test_road_chain_stops_at_junction(self):
        tail = [_WP(0, 0, 0.0, is_junction=True)]
        head = _WP(0.0, 0.0, 0.0, chain=tail)
        rs = build_road_structure(head, lookahead_m=30.0, step_m=5.0)
        self.assertEqual(rs["s_range"][1], 5.0)  # stopped after stepping into junction


class DispatchTests(unittest.TestCase):
    def test_dispatch_to_junction_ahead(self):
        # A junction 10 m ahead of a plain approach waypoint.
        jwp = _WP(0, 0, 0.0, is_junction=True)
        jwp._junction = _Junction(
            9, (10.0, 0.0),
            [(_WP(0, 0, 0.0), _WP(0, 0, 0.0)), (_WP(0, 0, 90.0), _WP(0, 0, 90.0))],
        )
        approach2 = _WP(5.0, 0.0, 0.0, chain=[jwp])
        approach1 = _WP(0.0, 0.0, 0.0, chain=[approach2])
        structure = build_matched_structure_from_waypoint(
            approach1, junction_lookahead_m=20.0, lane_type="Driving"
        )
        self.assertEqual(structure["kind"], "junction")
        self.assertIn("ego_distance_to_center_m", structure)

    def test_dispatch_to_road_when_no_junction(self):
        wp = _WP(0.0, 0.0, 0.0, road_id=3, lane_id=-1, chain=[])
        structure = build_matched_structure_from_waypoint(wp)
        self.assertEqual(structure["kind"], "road_segment")


if __name__ == "__main__":
    unittest.main()
