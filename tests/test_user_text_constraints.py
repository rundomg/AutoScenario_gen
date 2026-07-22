import unittest

from tools.user_text_constraints import (
    apply_user_constraints_to_relation_dsl,
    apply_user_constraints_to_scene,
    evaluate_user_constraints,
    normalize_user_constraint_bundle,
)
from tools.structured_pipeline import (
    apply_pairwise_ordering,
    generate_initial_coordinates_from_relation_dsl,
    refine_projected_coordinates_with_pairwise_relations,
)


def _bundle(constraints):
    return {
        "schema_version": "user-constraints-v1",
        "source": "user_text",
        "constraints": constraints,
    }


class TestUserTextConstraints(unittest.TestCase):
    def test_hard_ego_and_actor_lanes_override_conflicting_intermediate_values(self):
        scene = {
            "traffic_subjects": [
                {
                    "id": "det_5",
                    "lane_side_relation": "right_lane",
                    "lane_index_relation": 1,
                    "anchor_relation": {"lane_from_right": 0},
                }
            ],
            "road_network": {
                "map_matching": {
                    "ego_lane_from_right": 1,
                    "forward_lane_count": 2,
                }
            },
            "actor_layout": {"ego_approach": {"ego_lane_from_right": 1}},
            "metadata": {
                "user_constraints": _bundle(
                    [
                        {
                            "id": "ego_rightmost",
                            "target": "scene",
                            "path": "road_network.map_matching.ego_lane_from_right",
                            "value": 0,
                            "strength": "hard",
                        },
                        {
                            "id": "waiting_car_left",
                            "target": "entity",
                            "entity_id": "det_5",
                            "path": "anchor_relation.lane_from_right",
                            "value": 1,
                            "strength": "hard",
                        },
                    ]
                )
            },
        }

        applied, report = apply_user_constraints_to_scene(scene)

        self.assertEqual(
            applied["road_network"]["map_matching"]["ego_lane_from_right"], 0
        )
        self.assertEqual(
            applied["actor_layout"]["ego_approach"]["ego_lane_from_right"], 0
        )
        self.assertEqual(
            applied["metadata"]["ego_localization"]["ego_lane_source"],
            "user_text",
        )
        actor = applied["traffic_subjects"][0]
        self.assertEqual(actor["anchor_relation"]["lane_from_right"], 1)
        self.assertEqual(actor["lane_index_relation"], -1)
        self.assertEqual(actor["lane_side_relation"], "left_lane")
        self.assertFalse(report["unresolved"])

    def test_bundle_rejects_unsafe_paths(self):
        bundle, errors = normalize_user_constraint_bundle(
            _bundle(
                [
                    {
                        "target": "scene",
                        "path": "road_network.__class__['x']",
                        "value": 1,
                    }
                ]
            )
        )
        self.assertFalse(bundle["constraints"])
        self.assertTrue(errors)

    def test_exact_user_distance_reaches_coordinate_ir(self):
        scene = {
            "metadata": {
                "user_constraints": _bundle(
                    [
                        {
                            "id": "lead_distance",
                            "target": "entity",
                            "entity_id": "det_3",
                            "path": "longitudinal_m",
                            "value": 5,
                            "strength": "hard",
                        }
                    ]
                )
            }
        }
        relation_dsl = {
            "ego_relations": [
                {
                    "entity_id": "det_3",
                    "group_id": "det_3",
                    "distance_band": "far",
                }
            ],
            "entities": [],
            "metadata": {},
        }

        constrained = apply_user_constraints_to_relation_dsl(scene, relation_dsl)
        self.assertEqual(
            constrained["ego_relations"][0]["user_target_longitudinal_m"], 5.0
        )
        constrained["selected_anchor_lane"] = {
            "start": {"x": 0.0, "y": 0.0},
            "end": {"x": 20.0, "y": 0.0},
        }
        constrained["lane_width_class"] = "standard"
        coordinates = generate_initial_coordinates_from_relation_dsl(constrained)
        self.assertEqual(coordinates["entities"][0]["longitudinal_m"], 5.0)

    def test_final_validation_prefers_rendered_carla_actor_transform(self):
        scene = {
            "metadata": {
                "user_constraints": _bundle(
                    [
                        {
                            "id": "lead_distance",
                            "target": "entity",
                            "entity_id": "det_3",
                            "path": "longitudinal_m",
                            "value": 5,
                            "strength": "hard",
                            "tolerance": 1.0,
                        }
                    ]
                )
            }
        }
        payload = {
            "entities": [
                {"id": "det_3", "longitudinal_m": 5.0},
            ]
        }
        rendered = {
            "truth_source": "carla_actor_transform",
            "actors": [
                {
                    "id": "det_3",
                    "ego_frame": {"longitudinal_m": 9.0},
                }
            ],
        }

        report = evaluate_user_constraints(
            scene,
            payload,
            render_graph=rendered,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["results"][0]["actual"], 9.0)
        self.assertEqual(
            report["results"][0]["actual_source"], "carla_actor_transform"
        )

    def test_final_junction_lane_uses_rendered_junction_slot_when_generic_slot_is_null(self):
        scene = {
            "metadata": {
                "user_constraints": _bundle(
                    [
                        {
                            "id": "bike_rightmost",
                            "target": "entity",
                            "entity_id": "bike",
                            "path": "anchor_relation.lane_from_right",
                            "value": 0,
                            "strength": "hard",
                        }
                    ]
                )
            }
        }
        payload = {"entities": [{"id": "bike", "lane_from_right": 0}]}
        rendered = {
            "truth_source": "carla_actor_transform",
            "actors": [
                {
                    "id": "bike",
                    "spawned": True,
                    "lane_from_right": None,
                    "junction_lane_from_right": 0,
                }
            ],
        }

        report = evaluate_user_constraints(scene, payload, render_graph=rendered)

        self.assertEqual(report["status"], "satisfied")
        self.assertEqual(report["results"][0]["actual"], 0)

    def test_pairwise_solver_moves_other_actor_not_exact_user_target(self):
        anchor = {
            "start": {"x": 0.0, "y": 0.0},
            "end": {"x": 20.0, "y": 0.0},
        }
        coordinates = {
            "selected_anchor_lane": anchor,
            "entities": [
                {
                    "id": "locked",
                    "category": "car",
                    "priority": "subject",
                    "lane_side_relation": "same_lane",
                    "lane_index_relation": 0,
                    "longitudinal_m": 5.0,
                    "user_target_longitudinal_m": 5.0,
                    "location": {"x": 5.0, "y": 0.0, "z": 0.3},
                    "rotation": {"yaw": 0.0},
                },
                {
                    "id": "movable",
                    "category": "car",
                    "priority": "subject",
                    "lane_side_relation": "same_lane",
                    "lane_index_relation": 0,
                    "longitudinal_m": 10.0,
                    "location": {"x": 10.0, "y": 0.0, "z": 0.3},
                    "rotation": {"yaw": 0.0},
                },
            ],
        }
        relation_dsl = {
            "selected_anchor_lane": anchor,
            "lane_context": {},
            "pairwise_relations": [
                {
                    "entity_id": "locked",
                    "other_entity_id": "movable",
                    "constraint_strength": "hard",
                    "longitudinal_relation": "aligned_with_other",
                    "longitudinal_gap_band": "near",
                    "lateral_relation": "same_lateral_band",
                }
            ],
        }

        ordered = apply_pairwise_ordering(coordinates, relation_dsl)
        refined = refine_projected_coordinates_with_pairwise_relations(
            ordered, relation_dsl
        )

        locked = next(item for item in refined["entities"] if item["id"] == "locked")
        self.assertEqual(locked["longitudinal_m"], 5.0)
        self.assertEqual(locked["location"]["x"], 5.0)


if __name__ == "__main__":
    unittest.main()
