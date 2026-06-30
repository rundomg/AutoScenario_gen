"""Unit tests for tools.reference_frame.

These exercise the four canonical scene structures (cross intersection,
T-junction, two-way straight, curve) plus the two failure modes the structural
redesign exists to fix:

  * a cross-leg vehicle must land ON its side leg facing along that street, not
    sideways in front of ego (the old single-axis "横放" bug);
  * a genuine wrong-way actor (common for motorcycles) must keep its observed
    against-flow heading rather than being snapped legal.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reference_frame import (  # noqa: E402
    JunctionLeg,
    build_reference_frame,
    check_heading_consistency,
    heading_for,
    normalize_angle,
    angle_difference,
)


def _cross_intersection():
    # Ego approaches travelling east (heading 0). Four legs leave the centre at
    # E(0), N(-90 i.e. -y is north), W(180), S(90).
    return build_reference_frame(
        {
            "kind": "junction",
            "center": {"x": 100.0, "y": 0.0, "z": 0.0},
            "ego_approach_heading_deg": 0.0,
            "legs": [
                {"name": "east", "heading_out_deg": 0.0},
                {"name": "north", "heading_out_deg": -90.0},
                {"name": "west", "heading_out_deg": 180.0},
                {"name": "south", "heading_out_deg": 90.0},
            ],
        }
    )


class HeadingForTests(unittest.TestCase):
    def test_legal_follows_lane(self):
        self.assertAlmostEqual(heading_for(30.0, "legal"), 30.0)
        self.assertAlmostEqual(heading_for(30.0, "unknown"), 30.0)
        self.assertAlmostEqual(heading_for(30.0, None), 30.0)

    def test_wrong_way_flips_180(self):
        self.assertAlmostEqual(abs(normalize_angle(heading_for(30.0, "wrong_way") - 30.0)), 180.0)


class CrossIntersectionTests(unittest.TestCase):
    def test_ego_placed_back_along_approach_leg_facing_center(self):
        frame = _cross_intersection()
        # Ego comes in from the west leg (the leg whose outbound heading is 180).
        ego = frame.place(direction="ego", distance_m=40.0, motion="approaching")
        # 40 m back along the west leg from the centre at x=100 -> x=60.
        self.assertAlmostEqual(ego.location["x"], 60.0, places=3)
        self.assertAlmostEqual(ego.location["y"], 0.0, places=3)
        # Facing toward the centre = east = heading 0.
        self.assertAlmostEqual(normalize_angle(ego.yaw - 0.0), 0.0, places=3)

    def test_right_leg_car_is_on_the_right_street_not_sideways_in_front_of_ego(self):
        frame = _cross_intersection()
        ego = frame.place(direction="ego", distance_m=40.0, motion="approaching")
        car = frame.place(direction="right", distance_m=30.0, motion="approaching")
        self.assertIsNotNone(car)
        # Right of an east-bound ego is south (+y). The car must sit on the south
        # leg axis (x == centre x), i.e. genuinely on the cross street...
        self.assertAlmostEqual(car.location["x"], 100.0, places=3)
        self.assertAlmostEqual(car.location["y"], 30.0, places=3)
        # ...and NOT parked just ahead of ego on ego's own axis.
        self.assertGreater(car.location["x"] - ego.location["x"], 20.0)
        self.assertGreater(abs(car.location["y"] - ego.location["y"]), 20.0)
        # Heading runs along the cross street (north/south), ~90 deg off ego.
        self.assertAlmostEqual(angle_difference(car.yaw, ego.yaw), 90.0, places=3)

    def test_oncoming_through_car_faces_ego(self):
        frame = _cross_intersection()
        ego = frame.place(direction="ego", distance_m=40.0, motion="approaching")
        car = frame.place(direction="opposite", distance_m=40.0, motion="approaching")
        # On the east leg, approaching the centre => heading west (180), opposing ego.
        self.assertAlmostEqual(car.location["x"], 140.0, places=3)
        self.assertAlmostEqual(angle_difference(car.yaw, ego.yaw), 180.0, places=3)


class TJunctionTests(unittest.TestCase):
    def test_missing_left_arm_returns_none(self):
        # T-junction with arms east(ahead-through), west(ego), south(right) only.
        frame = build_reference_frame(
            {
                "kind": "junction",
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "ego_approach_heading_deg": 0.0,
                "legs": [
                    {"name": "east", "heading_out_deg": 0.0},
                    {"name": "west", "heading_out_deg": 180.0},
                    {"name": "south", "heading_out_deg": 90.0},
                ],
            }
        )
        self.assertIsNotNone(frame.assign_leg("right"))   # south exists
        self.assertEqual(frame.assign_leg("right").name, "south")
        self.assertIsNone(frame.assign_leg("left"))       # no north arm


class TwoWayStraightTests(unittest.TestCase):
    def _frame(self):
        # Ego at origin heading east along a straight road.
        return build_reference_frame(
            {
                "kind": "road_segment",
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward_heading_deg": 0.0,
            }
        )

    def test_same_direction_vehicle_follows_ego(self):
        car = self._frame().place(longitudinal_m=20.0, lateral_m=0.0, side="same")
        self.assertAlmostEqual(car.location["x"], 20.0, places=3)
        self.assertAlmostEqual(normalize_angle(car.yaw), 0.0, places=3)

    def test_legal_oncoming_is_in_opposing_lane_facing_back(self):
        # Opposing carriageway is to ego's left (-y). Legal oncoming faces west.
        car = self._frame().place(longitudinal_m=20.0, lateral_m=-3.5, side="opposing")
        self.assertAlmostEqual(car.location["y"], -3.5, places=3)
        self.assertAlmostEqual(angle_difference(car.yaw, 0.0), 180.0, places=3)

    def test_wrong_way_motorcycle_in_ego_lane_keeps_against_flow_heading(self):
        # A ghost-riding motorcycle sits in ego's own lane band (same side) but
        # faces back toward ego. It must NOT be moved to the opposing lane, and
        # its heading must stay opposed to ego.
        moto = self._frame().place(
            longitudinal_m=15.0, lateral_m=0.0, side="same", flow_compliance="wrong_way"
        )
        self.assertAlmostEqual(moto.location["y"], 0.0, places=3)  # still ego's band
        self.assertAlmostEqual(angle_difference(moto.yaw, 0.0), 180.0, places=3)
        # Distinct from a legal oncoming car which lives in the opposing band.
        self.assertEqual(moto.flow_compliance, "wrong_way")


class CurveTests(unittest.TestCase):
    def test_yaw_follows_local_tangent(self):
        # Road bends left: tangent 0 deg at s=0, ramping to -45 deg by s=20.
        frame = build_reference_frame(
            {
                "kind": "road_segment",
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "forward_heading_deg": 0.0,
                "curve_samples": [[0.0, 0.0], [10.0, -22.5], [20.0, -45.0]],
            }
        )
        near = frame.place(longitudinal_m=0.0, lateral_m=0.0)
        far = frame.place(longitudinal_m=20.0, lateral_m=0.0)
        self.assertAlmostEqual(normalize_angle(near.yaw), 0.0, places=3)
        self.assertAlmostEqual(normalize_angle(far.yaw), -45.0, places=3)
        # The far point must have bent off the straight x-axis (y < 0 since the
        # road curves toward -y), proving placement follows the bend.
        self.assertLess(far.location["y"], -3.0)


class ConsistencyGuardTests(unittest.TestCase):
    def test_legal_oncoming_facing_ego_passes(self):
        ok, _ = check_heading_consistency(
            yaw=180.0, ego_heading=0.0, heading_relation="opposite_direction",
            flow_compliance="legal",
        )
        self.assertTrue(ok)

    def test_legal_oncoming_facing_same_as_ego_is_flagged(self):
        # Pipeline bug: an oncoming car ended up following ego -> caught.
        ok, reason = check_heading_consistency(
            yaw=0.0, ego_heading=0.0, heading_relation="opposite_direction",
            flow_compliance="legal",
        )
        self.assertFalse(ok)
        self.assertIn("oncoming", reason)

    def test_wrong_way_is_never_corrected(self):
        # Even though it "violates" the legal expectation, a wrong_way actor is
        # accepted as observed.
        ok, reason = check_heading_consistency(
            yaw=0.0, ego_heading=0.0, heading_relation="same_direction",
            flow_compliance="wrong_way",
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "wrong_way_preserved")


if __name__ == "__main__":
    unittest.main()
