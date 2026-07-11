import unittest

from tools.actor_graph_verifier import (
    build_render_actor_graph_from_carla_records,
    build_render_actor_graph_from_spawn_payload,
    build_source_actor_graph,
    compare_actor_graphs,
)


def _relation_dsl(*entities):
    return {"ego_relations": list(entities)}


def _source_entity(
    entity_id,
    category="car",
    lane="same_lane",
    distance="near",
    heading="same_direction",
    lane_index=None,
):
    payload = {
        "entity_id": entity_id,
        "group_id": entity_id.split("_")[0],
        "category": category,
        "subtype": category,
        "spawn_kind": "vehicle",
        "lane_side_relation": lane,
        "order_relation": "ahead",
        "distance_band": distance,
        "heading_relation": heading,
        "visual_confidence": "high",
    }
    if lane_index is not None:
        payload["lane_index_relation"] = lane_index
    return payload


def _render_actor(actor_id, category="car", lane="same_lane", distance="near", heading="same_direction"):
    return {
        "id": actor_id,
        "category": category,
        "spawned": True,
        "truth_source": "carla_actor_transform",
        "ego_frame": {
            "lane_side_relation": lane,
            "distance_band": distance,
            "longitudinal_m": 10.0,
            "lateral_m": 0.0,
        },
        "heading_relation_to_ego": heading,
        "location": {"x": 10.0, "y": 0.0, "z": 0.3},
        "yaw": 0.0,
        "placement_mode": "preserve_xy",
    }


def _render_graph(*actors):
    return {
        "schema_version": "actor-graph-v1",
        "graph_type": "render_actor_graph",
        "truth_source": "carla_actor_transform",
        "position_checks_enabled": True,
        "spawn_checks_enabled": True,
        "actors": list(actors),
        "ignored_actors": [],
    }


class TestActorGraphVerifier(unittest.TestCase):
    def test_live_heading_classifies_perpendicular_actor_as_crossing(self):
        graph = build_render_actor_graph_from_carla_records(
            {
                "entities": [
                    {"id": "ego", "category": "car"},
                    {"id": "cross", "category": "car", "rotation": {"yaw": 90.0}},
                ]
            },
            {
                "ego": {"location": {"x": 0.0, "y": 0.0, "z": 0.3}, "yaw": 0.0},
                "cross": {"location": {"x": 8.0, "y": 0.0, "z": 0.3}, "yaw": 90.0},
            },
        )

        self.assertEqual(graph["actors"][0]["heading_relation_to_ego"], "crossing")

    def test_junction_road_id_transition_with_same_lane_and_yaw_is_valid(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                {
                    **_source_entity("car_1", heading="crossing"),
                    "layout_anchor_id": "left_arm",
                    "anchor_relation": {"lane_from_right": 0},
                }
            ),
        )
        render_actor = _render_actor("car_1", heading="crossing")
        render_actor["placement_mode"] = "project_to_junction_lane"
        render_actor["projected_lane"] = {"road_id": 75, "lane_id": 1, "yaw": -90.0}
        render_actor["actual_waypoint"] = {"road_id": 25, "lane_id": 1, "yaw": 270.0}

        plan = compare_actor_graphs(source, _render_graph(render_actor))

        self.assertNotIn(
            "junction_lane_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_side_arm_lane_is_not_compared_in_ego_lateral_frame(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                {
                    **_source_entity("car_1", lane="right_lane", lane_index=1),
                    "layout_anchor_id": "right_arm",
                    "anchor_relation": {"lane_from_right": 0},
                }
            ),
        )
        render_actor = _render_actor("car_1", lane="left_lane")
        render_actor["ego_frame"]["lane_index_relation"] = -4
        render_actor["ego_frame"]["lateral_m"] = -14.0

        plan = compare_actor_graphs(source, _render_graph(render_actor))

        self.assertNotIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_build_source_actor_graph_keeps_expanded_ids_and_ignores_nonvehicles(self):
        graph = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_row_0_0", "car"),
                _source_entity("car_row_0_1", "car"),
                _source_entity("ped_1", "pedestrian"),
            ),
        )

        self.assertEqual([actor["id"] for actor in graph["actors"]], ["car_row_0_0", "car_row_0_1"])
        self.assertEqual(graph["ignored_actors"][0]["id"], "ped_1")
        self.assertTrue(graph["metadata"]["id_chain_valid"])

    def test_fallback_render_graph_marks_truth_unavailable(self):
        graph = build_render_actor_graph_from_spawn_payload(
            {
                "entities": [
                    {"id": "ego", "category": "car"},
                    {
                        "id": "car_1",
                        "category": "car",
                        "spawn_kind": "vehicle",
                        "lane_side_relation": "right_lane",
                        "rotation": {"yaw": 0.0},
                        "location": {"x": 1.0, "y": 2.0, "z": 0.3},
                    },
                ]
            }
        )

        self.assertEqual(graph["truth_source"], "spawn_payload_fallback")
        self.assertFalse(graph["position_checks_enabled"])
        self.assertFalse(graph["spawn_checks_enabled"])
        self.assertEqual(graph["actors"][0]["id"], "car_1")

    def test_fallback_compare_does_not_strong_pass(self):
        source = build_source_actor_graph({}, _relation_dsl(_source_entity("car_1")))
        render = build_render_actor_graph_from_spawn_payload(
            {"entities": [{"id": "car_1", "category": "car"}]}
        )

        plan = compare_actor_graphs(source, render)

        self.assertEqual(plan["status"], "static_checked")
        self.assertFalse(plan["passed"])
        self.assertIsNone(plan["score"])
        self.assertTrue(plan["truth_unavailable"])

    def test_project_to_junction_lane_mismatch_is_repairable(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1", lane="right_lane", lane_index=1)),
        )
        render_actor = _render_actor("car_1", lane="right_lane")
        render_actor["placement_mode"] = "project_to_junction_lane"
        render_actor["projected_lane"] = {
            "road_id": 622,
            "lane_id": 1,
            "yaw": 270.0,
            "source": "junction_leg",
        }
        render_actor["actual_waypoint"] = {
            "road_id": 631,
            "lane_id": 1,
            "yaw": 90.0,
        }
        render = _render_graph(render_actor)

        plan = compare_actor_graphs(source, render)

        self.assertIn(
            "junction_lane_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )
        self.assertIn(
            "junction_lane_mismatch",
            [action["type"] for action in plan["repair_actions"]],
        )
        self.assertFalse(plan["requires_code_fix"])

    def test_fallback_compare_detects_static_spawn_overlap(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1"), _source_entity("car_2")),
        )
        render = build_render_actor_graph_from_spawn_payload(
            {
                "entities": [
                    {"id": "car_1", "category": "car", "location": {"x": 10.0, "y": 0.0}},
                    {"id": "car_2", "category": "car", "location": {"x": 10.5, "y": 0.0}},
                ]
            }
        )

        plan = compare_actor_graphs(source, render)

        self.assertEqual(plan["status"], "static_failed")
        self.assertIn("overlap", [issue["issue_type"] for issue in plan["issues"]])
        self.assertTrue(any(action["type"] == "overlap" for action in plan["repair_actions"]))

    def test_detects_count_and_category_mismatches(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1", "car"), _source_entity("truck_1", "truck")),
        )
        render = _render_graph(_render_actor("car_1", "car"), _render_actor("truck_1", "car"))

        plan = compare_actor_graphs(source, render)
        issue_types = [issue["issue_type"] for issue in plan["issues"]]

        self.assertIn("count_mismatch", issue_types)
        self.assertIn("category_mismatch", issue_types)
        self.assertTrue(any(action["type"] == "category_mismatch" for action in plan["repair_actions"]))

    def test_detects_lane_longitudinal_and_heading_mismatches(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_1", lane="right_edge", distance="far", heading="opposite_direction")
            ),
        )
        render = _render_graph(
            _render_actor("car_1", lane="same_lane", distance="near", heading="same_direction")
        )

        plan = compare_actor_graphs(source, render)
        issue_types = [issue["issue_type"] for issue in plan["issues"]]

        self.assertIn("lane_side_mismatch", issue_types)
        self.assertIn("longitudinal_mismatch", issue_types)
        self.assertIn("heading_mismatch", issue_types)

    def test_visual_position_override_skips_lane_checks_and_validates_target(self):
        source = build_source_actor_graph(
            {}, _relation_dsl(_source_entity("car_1", lane="same_lane"))
        )
        source_actor = source["actors"][0]
        source_actor["visual_position_override"] = {
            "target_longitudinal_m": 10.0,
            "target_lateral_offset_m": 4.0,
            "target_heading_relation": "same_direction",
            "requested_world_location": {"x": 10.0, "y": 4.0, "z": 0.3},
            "longitudinal_tolerance_m": 1.5,
            "lateral_tolerance_m": 0.75,
        }
        render_actor = _render_actor("car_1", lane="left_lane")
        render_actor["ego_frame"]["lane_index_relation"] = -2
        render_actor["ego_frame"]["lateral_m"] = -7.0
        render_actor["location"] = {"x": 10.0, "y": 4.0, "z": 0.3}
        render = _render_graph(render_actor)

        plan = compare_actor_graphs(source, render)

        issue_types = [issue["issue_type"] for issue in plan["issues"]]
        self.assertNotIn("lane_side_mismatch", issue_types)
        self.assertNotIn("longitudinal_mismatch", issue_types)
        self.assertNotIn("visual_target_deviation", issue_types)

    def test_right_edge_source_is_compatible_with_right_lane_render(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1", lane="right_edge")),
        )
        render_actor = _render_actor("car_1", lane="right_lane")
        render_actor["ego_frame"]["lane_index_relation"] = 1
        render_actor["ego_frame"]["lateral_m"] = 3.5
        render = _render_graph(render_actor)

        plan = compare_actor_graphs(source, render)

        self.assertNotIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_right_parking_driving_fallback_is_not_lane_mismatch(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_1", lane="right_parking_lane", lane_index=1)
            ),
        )
        render_actor = _render_actor("car_1", lane="same_lane")
        render_actor["ego_frame"]["lane_index_relation"] = 0
        render_actor["ego_frame"]["lateral_m"] = 0.0
        render_actor["parking_projection_result"] = "rightmost_driving_fallback"
        render_actor["parking_projection_lane"] = {"road_id": 76, "lane_id": -2}
        render_actor["actual_waypoint"] = {"road_id": 76, "lane_id": -2}

        plan = compare_actor_graphs(source, _render_graph(render_actor))

        self.assertNotIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_real_parking_lane_is_not_lane_mismatch(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_1", lane="right_parking_lane", lane_index=1)
            ),
        )
        render_actor = _render_actor("car_1", lane="same_lane")
        render_actor["ego_frame"]["lane_index_relation"] = 0
        render_actor["parking_projection_result"] = "parking_lane"
        render_actor["parking_projection_lane"] = {"road_id": 76, "lane_id": -2}
        render_actor["actual_waypoint"] = {
            "road_id": 76,
            "lane_id": -2,
            "lane_type": "LaneType.Parking",
        }

        plan = compare_actor_graphs(source, _render_graph(render_actor))

        self.assertNotIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_render_graph_preserves_parking_semantic_relation(self):
        spawn_payload = {
            "entities": [
                {
                    "id": "car_1",
                    "category": "car",
                    "lane_side_relation": "right_parking_lane",
                }
            ]
        }
        graph = build_render_actor_graph_from_carla_records(
            spawn_payload,
            {
                "ego": {
                    "location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "yaw": 0.0,
                },
                "car_1": {
                    "location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "yaw": 0.0,
                    "actual_waypoint": {"road_id": 76, "lane_id": -2},
                }
            },
        )

        self.assertEqual(
            graph["actors"][0]["lane_side_relation"],
            "right_parking_lane",
        )

    def test_parking_fallback_requires_actor_on_recorded_lane(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_1", lane="right_parking_lane", lane_index=1)
            ),
        )
        render_actor = _render_actor("car_1", lane="same_lane")
        render_actor["ego_frame"]["lane_index_relation"] = 0
        render_actor["ego_frame"]["lateral_m"] = 0.0
        render_actor["parking_projection_result"] = "rightmost_driving_fallback"
        render_actor["parking_projection_lane"] = {"road_id": 76, "lane_id": -2}
        render_actor["actual_waypoint"] = {"road_id": 76, "lane_id": -1}

        plan = compare_actor_graphs(source, _render_graph(render_actor))

        self.assertIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_oncoming_actor_on_opposing_side_not_flagged_as_lane_mismatch(self):
        # An oncoming vehicle is placed across the median onto the opposing
        # carriageway, so its rendered ego-frame lane index (e.g. -5) no longer
        # equals the source lane index (-2). Validation is by side, not exact
        # index, so this must NOT raise a lane_side_mismatch.
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity(
                    "veh_6", lane="left_lane", heading="opposite_direction", lane_index=-2
                )
            ),
        )
        render_actor = _render_actor("veh_6", lane="left_lane", heading="opposite_direction")
        render_actor["ego_frame"]["lane_index_relation"] = -5
        render_actor["ego_frame"]["lateral_m"] = -19.25
        render_actor["placement_mode"] = "project_to_lane"
        render = _render_graph(render_actor)

        plan = compare_actor_graphs(source, render)

        self.assertNotIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_oncoming_actor_on_ego_right_side_is_flagged(self):
        # An oncoming vehicle that ends up on ego's RIGHT (same/right carriageway)
        # is genuinely wrong-sided and must be flagged.
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity(
                    "veh_6", lane="left_lane", heading="opposite_direction", lane_index=-2
                )
            ),
        )
        render_actor = _render_actor("veh_6", lane="right_lane", heading="opposite_direction")
        render_actor["ego_frame"]["lane_index_relation"] = 1
        render_actor["ego_frame"]["lateral_m"] = 3.5
        render = _render_graph(render_actor)

        plan = compare_actor_graphs(source, render)

        self.assertIn(
            "lane_side_mismatch",
            [issue["issue_type"] for issue in plan["issues"]],
        )

    def test_detects_spawn_failure_and_requires_code_fix(self):
        source = build_source_actor_graph({}, _relation_dsl(_source_entity("car_1")))
        render = _render_graph(
            {
                "id": "car_1",
                "category": "car",
                "spawned": False,
                "spawn_failure_reason": "try_spawn_actor_returned_none_or_collision",
            }
        )

        plan = compare_actor_graphs(source, render)

        self.assertIn("spawn_failure", [issue["issue_type"] for issue in plan["issues"]])
        self.assertTrue(plan["requires_code_fix"])

    def test_detects_overlap(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1"), _source_entity("car_2")),
        )
        render = _render_graph(
            _render_actor("car_1"),
            {**_render_actor("car_2"), "location": {"x": 10.2, "y": 0.0, "z": 0.3}},
        )

        plan = compare_actor_graphs(source, render)

        self.assertIn("overlap", [issue["issue_type"] for issue in plan["issues"]])

    def test_detects_systematic_error_for_three_same_direction_issues(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(
                _source_entity("car_1", lane="right_lane", lane_index=2),
                _source_entity("car_2", lane="right_lane", lane_index=2),
                _source_entity("car_3", lane="right_lane", lane_index=2),
            ),
        )
        render = _render_graph(
            _render_actor("car_1", lane="same_lane"),
            _render_actor("car_2", lane="same_lane"),
            _render_actor("car_3", lane="same_lane"),
        )
        for actor in render["actors"]:
            actor["placement_mode"] = "project_to_lane"

        plan = compare_actor_graphs(source, render)

        self.assertTrue(plan["systematic_error"])
        self.assertTrue(plan["requires_code_fix"])
        self.assertTrue(plan["stop_repair_loop"])

    def test_detects_repeated_issue_from_previous_plan(self):
        source = build_source_actor_graph(
            {},
            _relation_dsl(_source_entity("car_1", lane="right_lane", lane_index=2)),
        )
        render = _render_graph(_render_actor("car_1", lane="same_lane"))
        previous = {
            "issues": [
                {
                    "issue_type": "lane_side_mismatch",
                    "source_entity_id": "car_1",
                    "direction": "2->0",
                }
            ]
        }

        plan = compare_actor_graphs(source, render, previous_plan=previous)

        self.assertTrue(plan["systematic_error"])
        self.assertTrue(plan["requires_code_fix"])


if __name__ == "__main__":
    unittest.main()
