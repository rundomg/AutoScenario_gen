import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import autoscenario_scene_helpers as helpers  # noqa: E402


class _FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _FakeRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class _FakeCarla:
    Location = _FakeLocation
    Rotation = _FakeRotation
    class LaneType:
        Driving = "Driving"
        Parking = "Parking"


class _FakeTransform:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.location = _FakeLocation(x, y, z)
        self.rotation = _FakeRotation(yaw=yaw)


class _FakeWaypoint:
    def __init__(
        self,
        road_id,
        lane_id,
        nxt=None,
        *,
        lane_type="Driving",
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=0.0,
        left=None,
        right=None,
    ):
        self.road_id = road_id
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.transform = _FakeTransform(x, y, z, yaw)
        self._next = nxt
        self._previous = nxt
        self._left = left
        self._right = right

    def next(self, _distance):
        return [] if self._next is None else [self._next]

    def previous(self, _distance):
        return [] if self._previous is None else [self._previous]

    def get_left_lane(self):
        return self._left

    def get_right_lane(self):
        return self._right


class SceneHelperTests(unittest.TestCase):
    def setUp(self):
        self._old_carla = helpers.carla
        helpers.carla = _FakeCarla

    def tearDown(self):
        helpers.carla = self._old_carla

    def test_strict_lane_spawn_candidates_do_not_shift_laterally(self):
        location = _FakeLocation(10.0, 20.0, 0.35)
        rotation = _FakeRotation(yaw=90.0)

        candidates = helpers._autoscenario_collect_strict_lane_spawn_candidates(
            location, rotation
        )

        self.assertGreater(len(candidates), 1)
        for candidate_location, candidate_rotation in candidates:
            self.assertAlmostEqual(candidate_location.x, 10.0, places=6)
            self.assertAlmostEqual(candidate_rotation.yaw, 90.0, places=6)
            offset_y = candidate_location.y - 20.0
            self.assertTrue(
                any(math.isclose(offset_y, expected, abs_tol=1e-6) for expected in {
                    0.0, 5.5, -5.5, 11.0, -11.0, 16.5, -16.5
                })
            )

    def test_same_lane_step_rejects_road_or_lane_jump(self):
        jumped = _FakeWaypoint(17, 1)
        start = _FakeWaypoint(1708, 1, nxt=jumped)

        result = helpers._autoscenario_step_waypoint_same_lane(
            start,
            6.0,
            forward=True,
            target_road_id=1708,
            target_lane_id=1,
        )

        self.assertIs(result, start)

    def test_parking_lane_spawn_uses_strict_lane_candidates(self):
        location = _FakeLocation(10.0, 20.0, 0.35)
        rotation = _FakeRotation(yaw=90.0)
        blueprint = object()
        actor = object()

        with mock.patch.object(
            helpers,
            "_autoscenario_project_vehicle_to_parking_lane",
            return_value=(location, rotation),
        ), mock.patch.object(
            helpers,
            "_autoscenario_record_focus_point",
        ), mock.patch.object(
            helpers,
            "_autoscenario_pick_blueprint",
            return_value=blueprint,
        ), mock.patch.object(
            helpers,
            "_autoscenario_apply_vehicle_color",
        ), mock.patch.object(
            helpers,
            "_autoscenario_apply_role_name",
        ), mock.patch.object(
            helpers,
            "_autoscenario_try_spawn_vehicle_actor",
        ) as normal_spawn, mock.patch.object(
            helpers,
            "_autoscenario_try_spawn_vehicle_actor_strict_lane",
            return_value=actor,
        ) as strict_spawn:
            result = helpers._autoscenario_spawn_vehicle_parking_lane(
                "vehicle.tesla.model3",
                location,
                rotation,
                {"road_id": 76, "lane_id": -2},
            )

        self.assertIs(result, actor)
        normal_spawn.assert_not_called()
        strict_spawn.assert_called_once_with(blueprint, location, rotation, 3.5)

    def test_parking_projection_prefers_real_parking_lane(self):
        location = _FakeLocation(10.0, 20.0, 0.0)
        rotation = _FakeRotation(yaw=90.0)
        driving = _FakeWaypoint(76, -1, x=10.0, y=20.0, yaw=90.0)
        parking = _FakeWaypoint(
            76,
            -2,
            lane_type="Parking",
            x=11.0,
            y=20.0,
            yaw=90.0,
        )

        def get_waypoint(_location, lane_type):
            return parking if lane_type == _FakeCarla.LaneType.Parking else driving

        with mock.patch.object(
            helpers,
            "_autoscenario_get_waypoint_for_lane_type",
            side_effect=get_waypoint,
        ):
            snapped_location, _ = helpers._autoscenario_project_vehicle_to_parking_lane(
                location,
                rotation,
                {"road_id": 76, "lane_id": -1},
                lane_side_relation="right_parking_lane",
            )

        self.assertAlmostEqual(snapped_location.x, 11.0)
        self.assertEqual(
            helpers._AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT["result"],
            "parking_lane",
        )
        self.assertEqual(
            helpers._AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT["lane"],
            {"road_id": 76, "lane_id": -2},
        )

    def test_right_parking_falls_back_to_rightmost_driving_lane(self):
        location = _FakeLocation(10.0, 20.0, 0.0)
        rotation = _FakeRotation(yaw=90.0)
        outer = _FakeWaypoint(76, -2, x=12.0, y=20.0, yaw=90.0)
        inner = _FakeWaypoint(
            76,
            -1,
            x=10.0,
            y=20.0,
            yaw=90.0,
            right=outer,
        )

        def get_waypoint(_location, lane_type):
            return None if lane_type == _FakeCarla.LaneType.Parking else inner

        with mock.patch.object(
            helpers,
            "_autoscenario_get_waypoint_for_lane_type",
            side_effect=get_waypoint,
        ):
            snapped_location, _ = helpers._autoscenario_project_vehicle_to_parking_lane(
                location,
                rotation,
                {"road_id": 76, "lane_id": -1},
                lane_side_relation="right_parking_lane",
            )

        self.assertAlmostEqual(snapped_location.x, 12.0)
        self.assertEqual(
            helpers._AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT["result"],
            "rightmost_driving_fallback",
        )
        self.assertEqual(
            helpers._AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT["lane"],
            {"road_id": 76, "lane_id": -2},
        )

    def test_left_parking_does_not_use_right_driving_fallback(self):
        location = _FakeLocation(10.0, 20.0, 0.0)
        rotation = _FakeRotation(yaw=90.0)
        driving = _FakeWaypoint(76, -1, x=12.0, y=20.0, yaw=90.0)

        def get_waypoint(_location, lane_type):
            return None if lane_type == _FakeCarla.LaneType.Parking else driving

        with mock.patch.object(
            helpers,
            "_autoscenario_get_waypoint_for_lane_type",
            side_effect=get_waypoint,
        ):
            snapped_location, _ = helpers._autoscenario_project_vehicle_to_parking_lane(
                location,
                rotation,
                {"road_id": 76, "lane_id": -1},
                lane_side_relation="left_parking_lane",
            )

        self.assertAlmostEqual(snapped_location.x, 10.0)
        self.assertEqual(
            helpers._AUTOSCENARIO_LAST_PARKING_PROJECTION_RESULT["result"],
            "unavailable",
        )


if __name__ == "__main__":
    unittest.main()
