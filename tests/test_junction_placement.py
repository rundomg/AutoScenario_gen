"""Unit tests for tools.junction_placement (offline, no CARLA)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.junction_placement import (  # noqa: E402
    direction_and_motion_for_entity,
    reproject_actors_for_junction,
    reproject_actors_for_road,
    reproject_actors_for_structure,
)
from tools.reference_frame import angle_difference  # noqa: E402


def _cross_structure():
    # Centre at (100,0); ego approaches heading east (0). Legs: west(ego, out
    # -180), east(opposite, out 0), south(right, out 90), north(left, out -90).
    return {
        "kind": "junction",
        "junction_id": 1,
        "leg_count": 4,
        "center": {"x": 100.0, "y": 0.0, "z": 0.0},
        "ego_approach_heading_deg": 0.0,
        "ego_distance_to_center_m": 30.0,
        "legs": [
            {"name": "west", "heading_out_deg": -180.0},
            {"name": "east", "heading_out_deg": 0.0},
            {"name": "south", "heading_out_deg": 90.0},
            {"name": "north", "heading_out_deg": -90.0},
        ],
    }


class DirectionMappingTests(unittest.TestCase):
    def test_ego(self):
        self.assertEqual(direction_and_motion_for_entity({"id": "ego"}), ("ego", "approaching"))

    def test_oncoming(self):
        d, m = direction_and_motion_for_entity({"heading_relation": "opposite_direction"})
        self.assertEqual((d, m), ("opposite", "approaching"))

    def test_crossing_uses_turn_then_lane_index(self):
        self.assertEqual(
            direction_and_motion_for_entity({"heading_relation": "crossing", "turn_intent": "left"}),
            ("left", "approaching"),
        )
        self.assertEqual(
            direction_and_motion_for_entity(
                {"heading_relation": "crossing", "lane_index_relation": -1}
            ),
            ("left", "approaching"),
        )

    def test_same_direction_turning_leaves_on_side_leg(self):
        self.assertEqual(
            direction_and_motion_for_entity(
                {"heading_relation": "same_direction", "turn_intent": "right"}
            ),
            ("right", "leaving"),
        )


class ReprojectionTests(unittest.TestCase):
    def _coords(self):
        return {
            "entities": [
                {"id": "ego", "priority": "ego", "heading_relation": "same_direction",
                 "location": {"x": 0, "y": 0, "z": 0.3}},
                {"id": "onc", "heading_relation": "opposite_direction", "longitudinal_m": 25.0,
                 "flow_compliance": "legal", "location": {"x": 0, "y": 0, "z": 0.3}},
                {"id": "cross_r", "heading_relation": "crossing", "lane_index_relation": 1,
                 "longitudinal_m": 20.0, "location": {"x": 0, "y": 0, "z": 0.3}},
                {"id": "moto", "heading_relation": "opposite_direction", "longitudinal_m": 15.0,
                 "flow_compliance": "wrong_way", "category": "motorcycle",
                 "location": {"x": 0, "y": 0, "z": 0.3}},
            ]
        }

    def test_ego_on_approach_leg_facing_center(self):
        out = reproject_actors_for_junction(self._coords(), _cross_structure())
        ego = next(e for e in out["entities"] if e["id"] == "ego")
        self.assertEqual(ego["junction_leg"], "west")
        self.assertAlmostEqual(ego["location"]["x"], 70.0, places=3)  # 30 m back from centre
        self.assertAlmostEqual(angle_difference(ego["rotation"]["yaw"], 0.0), 0.0, places=3)

    def test_cross_traffic_on_side_leg_not_in_front_of_ego(self):
        out = reproject_actors_for_junction(self._coords(), _cross_structure())
        ego = next(e for e in out["entities"] if e["id"] == "ego")
        cross = next(e for e in out["entities"] if e["id"] == "cross_r")
        self.assertEqual(cross["junction_direction"], "right")
        self.assertEqual(cross["junction_leg"], "south")
        # On the south cross street (x ~ centre x, within a half-lane lateral),
        # well off ego's axis.
        self.assertLess(abs(cross["location"]["x"] - 100.0), 2.0)
        self.assertGreater(abs(cross["location"]["y"] - ego["location"]["y"]), 15.0)
        # Heading ~90 deg off ego, i.e. along the cross street, not sideways-in-front.
        self.assertAlmostEqual(angle_difference(cross["rotation"]["yaw"], ego["rotation"]["yaw"]), 90.0, places=3)

    def test_oncoming_faces_ego(self):
        out = reproject_actors_for_junction(self._coords(), _cross_structure())
        ego = next(e for e in out["entities"] if e["id"] == "ego")
        onc = next(e for e in out["entities"] if e["id"] == "onc")
        self.assertEqual(onc["junction_leg"], "east")
        self.assertAlmostEqual(angle_difference(onc["rotation"]["yaw"], ego["rotation"]["yaw"]), 180.0, places=3)

    def test_wrong_way_motorcycle_preserved(self):
        out = reproject_actors_for_junction(self._coords(), _cross_structure())
        ego = next(e for e in out["entities"] if e["id"] == "ego")
        moto = next(e for e in out["entities"] if e["id"] == "moto")
        # On the opposite (east) leg like a normal oncoming, BUT facing WITH ego
        # (against that leg's inbound flow) because it is wrong_way.
        self.assertEqual(moto["junction_leg"], "east")
        self.assertAlmostEqual(angle_difference(moto["rotation"]["yaw"], ego["rotation"]["yaw"]), 0.0, places=3)
        self.assertEqual(moto["heading_consistency"], "wrong_way_preserved")

    def test_layout_anchor_overrides_ego_relative_guess_and_picks_rightmost_lane(self):
        struct = _cross_structure()
        for leg in struct["legs"]:
            if leg["name"] == "north":
                leg["inbound_lanes"] = [
                    {"road_id": 20, "lane_id": -1, "role": "inbound"},
                    {"road_id": 20, "lane_id": -2, "role": "inbound"},
                ]
        coords = {"entities": [
            {
                "id": "anchored",
                "heading_relation": "same_direction",
                "turn_intent": "none",
                "layout_anchor_id": "left_arm",
                "anchor_relation": {
                    "travel_direction": "toward_junction",
                },
                "location": {"x": 0, "y": 0, "z": 0.3},
            }
        ]}

        out = reproject_actors_for_junction(coords, struct)

        actor = out["entities"][0]
        self.assertEqual(actor["junction_leg"], "north")
        self.assertEqual(actor["junction_motion"], "approaching")
        # With no explicit lane band, the rightmost travel lane is used.
        self.assertEqual(actor["projected_lane"]["lane_id"], -2)
        self.assertEqual(actor["projected_lane"]["yaw"], None)

    def test_junction_lane_index_changes_lateral_band(self):
        coords = {"entities": [
            {
                "id": "left_car",
                "heading_relation": "same_direction",
                "lane_index_relation": -1,
                "location": {"x": 0, "y": 0, "z": 0.3},
            },
            {
                "id": "right_car",
                "heading_relation": "same_direction",
                "lane_index_relation": 1,
                "location": {"x": 0, "y": 0, "z": 0.3},
            },
        ]}

        out = reproject_actors_for_junction(coords, _cross_structure())

        left = next(e for e in out["entities"] if e["id"] == "left_car")
        right = next(e for e in out["entities"] if e["id"] == "right_car")
        self.assertEqual(left["junction_leg"], right["junction_leg"])
        self.assertAlmostEqual(left["junction_distance_m"], right["junction_distance_m"])
        self.assertGreater(abs(left["location"]["y"] - right["location"]["y"]), 6.0)

    def test_picks_rightmost_approach_lane(self):
        struct = _cross_structure()
        for leg in struct["legs"]:
            if leg["name"] == "west":
                leg["inbound_lanes"] = [
                    {"road_id": 10, "lane_id": 1, "role": "inbound"},
                    {"road_id": 10, "lane_id": 2, "role": "inbound"},
                ]
        coords = {"entities": [
            {
                "id": "ego",
                "priority": "ego",
                "heading_relation": "same_direction",
                "location": {"x": 0, "y": 0, "z": 0.3},
            }
        ]}

        out = reproject_actors_for_junction(coords, struct)

        ego = out["entities"][0]
        self.assertEqual(ego["projected_lane"]["lane_id"], 2)

    def test_same_arm_near_mouth_vehicles_are_queued_not_overlapped(self):
        struct = _cross_structure()
        for leg in struct["legs"]:
            if leg["name"] == "south":
                leg["inbound_lanes"] = [{"road_id": 30, "lane_id": 1, "role": "inbound"}]
        coords = {"entities": [
            {
                "id": "truck",
                "category": "truck",
                "heading_relation": "same_direction",
                "layout_anchor_id": "right_arm",
                "anchor_relation": {"position_along_anchor": "near_mouth"},
                "location": {"x": 0, "y": 0, "z": 0.3},
            },
            {
                "id": "car_1",
                "category": "car",
                "heading_relation": "same_direction",
                "layout_anchor_id": "right_arm",
                "anchor_relation": {"position_along_anchor": "near_mouth"},
                "location": {"x": 0, "y": 0, "z": 0.3},
            },
            {
                "id": "car_2",
                "category": "car",
                "heading_relation": "same_direction",
                "lane_index_relation": 2,
                "layout_anchor_id": "right_arm",
                "anchor_relation": {"position_along_anchor": "near_mouth"},
                "location": {"x": 0, "y": 0, "z": 0.3},
            },
        ]}

        out = reproject_actors_for_junction(coords, struct)

        actors = out["entities"]
        distances = [actor["junction_distance_m"] for actor in actors]
        self.assertEqual([actor["junction_leg"] for actor in actors], ["south", "south", "south"])
        self.assertGreaterEqual(distances[1] - distances[0], 5.5)
        self.assertAlmostEqual(distances[2], distances[0])
        self.assertGreater(
            abs(actors[2]["location"]["x"] - actors[0]["location"]["x"]),
            6.0,
        )

    def test_missing_leg_left_unassigned(self):
        # T-junction without a north(left) arm: a left-crossing actor is left as-is.
        struct = _cross_structure()
        struct["legs"] = [l for l in struct["legs"] if l["name"] != "north"]
        struct["leg_count"] = 3
        coords = {"entities": [
            {"id": "x", "heading_relation": "crossing", "turn_intent": "left",
             "location": {"x": 5, "y": 5, "z": 0.3}},
        ]}
        out = reproject_actors_for_junction(coords, struct)
        e = out["entities"][0]
        self.assertEqual(e["junction_placement"], "unassigned")
        self.assertEqual(e["location"], {"x": 5, "y": 5, "z": 0.3})  # untouched

    def test_non_junction_structure_is_noop(self):
        coords = {"entities": [{"id": "a", "location": {"x": 1, "y": 2, "z": 0.3}}]}
        out = reproject_actors_for_junction(coords, {"kind": "road_segment"})
        self.assertEqual(out["entities"][0]["location"], {"x": 1, "y": 2, "z": 0.3})


class RoadReprojectionTests(unittest.TestCase):
    def _straight(self):
        # Ego at origin heading east (0). Same-dir car ahead+right, oncoming
        # ahead+left (opposing band), wrong-way moto ahead in ego band.
        return {
            "kind": "road_segment",
            "road_id": 5,
            "lane_id": -1,
            "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
            "forward_heading_deg": 0.0,
            "curve_samples": [[0.0, 0.0]],
        }

    def _coords(self):
        return {"entities": [
            {"id": "lead", "heading_relation": "same_direction", "lane_index_relation": 0,
             "location": {"x": 20.0, "y": 0.0, "z": 0.3}},
            {"id": "onc", "heading_relation": "opposite_direction", "flow_compliance": "legal",
             "location": {"x": 25.0, "y": -3.5, "z": 0.3}},
            {"id": "moto", "heading_relation": "opposite_direction", "flow_compliance": "wrong_way",
             "lane_index_relation": 0, "location": {"x": 15.0, "y": -3.5, "z": 0.3}},
        ]}

    def test_same_direction_keeps_position_and_heading(self):
        out = reproject_actors_for_road(self._coords(), self._straight())
        lead = next(e for e in out["entities"] if e["id"] == "lead")
        self.assertAlmostEqual(lead["location"]["x"], 20.0, places=3)
        self.assertAlmostEqual(angle_difference(lead["rotation"]["yaw"], 0.0), 0.0, places=3)

    def test_legal_oncoming_faces_back_on_opposing_side(self):
        out = reproject_actors_for_road(self._coords(), self._straight())
        onc = next(e for e in out["entities"] if e["id"] == "onc")
        self.assertLess(onc["location"]["y"], -1.0)  # stays on opposing (left) band
        self.assertAlmostEqual(angle_difference(onc["rotation"]["yaw"], 0.0), 180.0, places=3)

    def test_wrong_way_moto_pulled_into_ego_band_facing_back(self):
        out = reproject_actors_for_road(self._coords(), self._straight())
        moto = next(e for e in out["entities"] if e["id"] == "moto")
        # Pulled back into ego's own band (lane_index 0 -> lateral 0), not the
        # opposing offset, but still facing back toward ego.
        self.assertAlmostEqual(moto["location"]["y"], 0.0, places=3)
        self.assertAlmostEqual(angle_difference(moto["rotation"]["yaw"], 0.0), 180.0, places=3)
        self.assertEqual(moto["heading_consistency"], "wrong_way_preserved")

    def test_curve_bends_layout_off_straight_axis(self):
        struct = self._straight()
        # Road bends toward -y: tangent ramps 0 -> -30 by s=20.
        struct["curve_samples"] = [[0.0, 0.0], [10.0, -15.0], [20.0, -30.0]]
        coords = {"entities": [
            {"id": "lead", "heading_relation": "same_direction", "lane_index_relation": 0,
             "location": {"x": 20.0, "y": 0.0, "z": 0.3}},
        ]}
        out = reproject_actors_for_road(coords, struct)
        lead = out["entities"][0]
        # 20 m along a road bending to -y must leave the straight x-axis.
        self.assertLess(lead["location"]["y"], -3.0)
        self.assertAlmostEqual(angle_difference(lead["rotation"]["yaw"], -30.0), 0.0, places=3)

    def test_dispatcher_routes_by_kind(self):
        # road kind -> road reprojection (adds road_placement marker)
        out = reproject_actors_for_structure(self._coords(), self._straight())
        self.assertEqual(out["entities"][0]["road_placement"], "frame")

    def test_ego_is_skipped(self):
        coords = {"entities": [{"id": "ego", "priority": "ego",
                                "location": {"x": 1.0, "y": 2.0, "z": 0.3}}]}
        out = reproject_actors_for_road(coords, self._straight())
        self.assertEqual(out["entities"][0]["location"], {"x": 1.0, "y": 2.0, "z": 0.3})


if __name__ == "__main__":
    unittest.main()
