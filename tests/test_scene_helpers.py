import math
import os
import sys
import unittest

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


class _FakeWaypoint:
    def __init__(self, road_id, lane_id, nxt=None):
        self.road_id = road_id
        self.lane_id = lane_id
        self._next = nxt
        self._previous = nxt

    def next(self, _distance):
        return [] if self._next is None else [self._next]

    def previous(self, _distance):
        return [] if self._previous is None else [self._previous]


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


if __name__ == "__main__":
    unittest.main()
