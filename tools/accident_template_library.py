from copy import deepcopy
from typing import Any, Dict, List, Optional


DEFAULT_METRICS = ["collision", "min_ttc_s", "min_distance_m", "impact_speed_mps"]


COMMON_EGO = ["car", "truck", "bus"]
COMMON_VEHICLE = ["car", "truck", "bus", "motorcycle"]
COMMON_VRU = ["pedestrian", "bicycle", "motorcycle"]


def _template(
    template_id: str,
    family: str,
    description: str,
    external_refs: Dict[str, List[str]],
    vlm_selection_cues: List[str],
    required_actor_roles: Dict[str, List[str]],
    preconditions: List[str],
    parameter_ranges: Dict[str, List[float]],
    controller: str,
    execution_status: str = "planned_not_implemented",
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "template_id": template_id,
        "family": family,
        "description": description,
        "external_refs": external_refs,
        "vlm_selection_cues": vlm_selection_cues,
        "required_actor_roles": required_actor_roles,
        "preconditions": preconditions,
        "parameter_ranges": parameter_ranges,
        "controller": controller,
        "execution_status": execution_status,
        "metrics": metrics or list(DEFAULT_METRICS),
    }


REAR_END_PARAMS = {
    "ego_target_speed_mps": [5.0, 13.0],
    "ego_reaction_delay_s": [0.8, 3.0],
    "npc_trigger_distance_m": [8.0, 20.0],
    # Lead vehicle cruises forward before the risk trigger. Without this the
    # controller falls back to ~2 m/s, so the lead actor barely moves and the
    # ego closes the initial gap almost instantly.
    "npc_target_speed_mps": [4.0, 9.0],
}

STEER_PARAMS = {
    "ego_target_speed_mps": [5.0, 13.0],
    "ego_reaction_delay_s": [0.8, 2.8],
    "npc_trigger_distance_m": [5.0, 18.0],
    "npc_target_speed_mps": [2.0, 11.0],
    "npc_steer_intensity": [0.12, 0.45],
}

CROSSING_PARAMS = {
    "ego_target_speed_mps": [4.0, 12.0],
    "ego_reaction_delay_s": [0.8, 3.0],
    "npc_trigger_distance_m": [8.0, 24.0],
    "npc_target_speed_mps": [2.0, 12.0],
    "npc_steer_intensity": [0.08, 0.35],
}


ACCIDENT_TEMPLATE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "lead_vehicle_stopped": _template(
        "lead_vehicle_stopped",
        "rear_end",
        "Ego closes on a stopped lead vehicle.",
        {
            "nhtsa": ["Rear-end: lead vehicle stopped"],
            "carla": ["FollowLeadingVehicle"],
            "euro_ncap": ["Car-to-Car Rear stationary"],
        },
        ["lead actor is stopped", "lead actor is ahead of ego", "same lane"],
        {"ego": COMMON_EGO, "lead": COMMON_VEHICLE},
        ["lead actor longitudinal_m > 0", "abs(lead actor lateral_m) <= same_lane_threshold"],
        {**REAR_END_PARAMS, "npc_target_speed_mps": [0.0, 1.0]},
        "lead_vehicle_stopped",
    ),
    "lead_vehicle_slower_constant_speed": _template(
        "lead_vehicle_slower_constant_speed",
        "rear_end",
        "Ego closes on a slower lead vehicle.",
        {
            "nhtsa": ["Rear-end: lead vehicle moving at constant, slower speed"],
            "carla": ["OtherLeadingVehicle"],
            "euro_ncap": ["Car-to-Car Rear moving"],
        },
        ["lead actor is ahead", "same lane", "queue or slow traffic evidence"],
        {"ego": COMMON_EGO, "lead": COMMON_VEHICLE},
        ["lead actor longitudinal_m > 0", "abs(lead actor lateral_m) <= same_lane_threshold"],
        {**REAR_END_PARAMS, "npc_target_speed_mps": [1.0, 6.0]},
        "lead_vehicle_slower_constant_speed",
    ),
    "lead_vehicle_hard_brake": _template(
        "lead_vehicle_hard_brake",
        "rear_end",
        "Lead vehicle brakes hard in front of ego.",
        {
            "nhtsa": ["Rear-end: lead vehicle decelerating"],
            "carla": ["Longitudinal control after leading vehicle's brake"],
            "euro_ncap": ["Car-to-Car Rear braking"],
        },
        ["risk actor is ahead of ego", "same lane or near same lane", "traffic queue, red light, stop line, or obstacle ahead"],
        {"ego": COMMON_EGO, "lead": COMMON_VEHICLE},
        ["lead actor longitudinal_m > 0", "abs(lead actor lateral_m) <= same_lane_threshold"],
        {**REAR_END_PARAMS, "npc_brake_intensity": [0.6, 1.0]},
        "lead_vehicle_hard_brake",
        execution_status="implemented",
    ),
    "lead_vehicle_cut_out_reveals_obstacle": _template(
        "lead_vehicle_cut_out_reveals_obstacle",
        "rear_end",
        "Lead vehicle leaves lane and reveals a stopped obstacle.",
        {
            "nhtsa": ["Rear-end: following vehicle changing lanes"],
            "carla": ["Obstacle avoidance with prior action - vehicle"],
            "euro_ncap": ["Car-to-Car Rear stationary"],
        },
        ["lead actor ahead", "possible obstacle or stopped traffic ahead", "adjacent lane available"],
        {"ego": COMMON_EGO, "lead": COMMON_VEHICLE, "obstacle": COMMON_VEHICLE},
        ["lead actor longitudinal_m > 0", "obstacle actor longitudinal_m > lead actor longitudinal_m"],
        {**STEER_PARAMS, "npc_cut_in_duration_s": [1.2, 3.5]},
        "lead_vehicle_cut_out_reveals_obstacle",
    ),
    "adjacent_vehicle_cut_in": _template(
        "adjacent_vehicle_cut_in",
        "lane_change_and_cut_in",
        "Adjacent vehicle cuts into ego lane.",
        {
            "nhtsa": ["Lane change: 1 vehicle going straight and another changing lanes"],
            "carla": ["Static cut-in"],
            "euro_ncap": ["Car-to-Car Front Head-On lane change"],
        },
        ["actor in adjacent lane", "actor near or slightly ahead of ego", "ego lane conflict possible"],
        {"ego": COMMON_EGO, "cut_in": COMMON_VEHICLE},
        ["abs(cut_in actor lateral_m) > same_lane_threshold", "cut_in actor longitudinal_m > -rear_buffer_m"],
        {**STEER_PARAMS, "npc_cut_in_duration_s": [1.5, 4.0]},
        "adjacent_vehicle_cut_in",
        execution_status="implemented",
    ),
    "static_queue_cut_in": _template(
        "static_queue_cut_in",
        "lane_change_and_cut_in",
        "Vehicle cuts out from static traffic into the ego lane.",
        {"nhtsa": ["Lane change: 1 vehicle going straight and another changing lanes"], "carla": ["Static cut-in"], "euro_ncap": []},
        ["static or queued vehicle in neighboring lane", "gap ahead of ego", "urban queue context"],
        {"ego": COMMON_EGO, "cut_in": COMMON_VEHICLE},
        ["cut_in actor lane_side_relation in adjacent_lane_set"],
        {**STEER_PARAMS, "npc_cut_in_duration_s": [1.2, 3.5]},
        "static_queue_cut_in",
    ),
    "roadside_vehicle_pull_out": _template(
        "roadside_vehicle_pull_out",
        "lane_change_and_cut_in",
        "Roadside vehicle pulls out into ego path.",
        {"nhtsa": ["Lane change: 1 vehicle going straight and another entering or leaving parking position"], "carla": ["Parking Cut-in"], "euro_ncap": []},
        ["actor at road edge or parking lane", "ego path passes nearby", "pull-out conflict possible"],
        {"ego": COMMON_EGO, "pull_out": COMMON_VEHICLE},
        ["pull_out actor lane_side_relation in edge_lane_set"],
        STEER_PARAMS,
        "roadside_vehicle_pull_out",
        execution_status="implemented",
    ),
    "lateral_encroachment": _template(
        "lateral_encroachment",
        "lane_change_and_cut_in",
        "Nearby actor drifts laterally toward ego.",
        {"nhtsa": ["Lane change: 2 vehicles going straight and 1 vehicle encroaching in same lane"], "carla": ["Vehicle invading lane"], "euro_ncap": []},
        ["side-by-side vehicle", "narrow lateral gap", "lane marking ambiguity or squeeze"],
        {"ego": COMMON_EGO, "encroaching": COMMON_VEHICLE},
        ["abs(encroaching actor lateral_m) <= adjacent_lane_threshold"],
        STEER_PARAMS,
        "lateral_encroachment",
        execution_status="implemented",
    ),
    "vehicle_entering_from_parking": _template(
        "vehicle_entering_from_parking",
        "lane_change_and_cut_in",
        "Parked vehicle enters traffic flow near ego.",
        {"nhtsa": ["Lane change: 1 vehicle going straight and another entering or leaving parking position"], "carla": ["Parking Exit"], "euro_ncap": []},
        ["parked vehicle row", "edge actor near ego", "available curbside parking context"],
        {"ego": COMMON_EGO, "parking_actor": COMMON_VEHICLE},
        ["parking_actor lane_side_relation in edge_lane_set"],
        STEER_PARAMS,
        "vehicle_entering_from_parking",
    ),
    "straight_crossing_path": _template(
        "straight_crossing_path",
        "intersection_crossing",
        "Ego and target move on perpendicular straight crossing paths.",
        {"nhtsa": ["Crossing paths: straight crossing paths (SCP)"], "carla": ["Crossing traffic running a red light"], "euro_ncap": ["Car-to-car Crossing Straight Crossing Path"]},
        ["intersection or junction", "cross traffic actor", "ego path crosses target path"],
        {"ego": COMMON_EGO, "cross_traffic": COMMON_VEHICLE},
        ["scene has junction", "cross_traffic actor is not same lane"],
        CROSSING_PARAMS,
        "straight_crossing_path",
        execution_status="implemented",
    ),
    "cross_traffic_red_light": _template(
        "cross_traffic_red_light",
        "intersection_crossing",
        "Cross traffic violates signal and enters ego path.",
        {"nhtsa": ["Crossing paths: straight crossing paths (SCP)"], "carla": ["Crossing traffic running a red light"], "euro_ncap": ["Car-to-car Crossing Straight Crossing Path"]},
        ["traffic signal visible", "red-light or signalized intersection context", "cross traffic actor"],
        {"ego": COMMON_EGO, "cross_traffic": COMMON_VEHICLE},
        ["scene has signalized junction", "cross_traffic actor is not same lane"],
        CROSSING_PARAMS,
        "cross_traffic_red_light",
        execution_status="implemented",
    ),
    "left_turn_across_opposite": _template(
        "left_turn_across_opposite",
        "intersection_crossing",
        "Turning actor crosses the path of opposing traffic.",
        {"nhtsa": ["Crossing paths: left turn across path from opposite direction (LTAP/OD)"], "carla": ["SignalizedJunctionLeftTurn"], "euro_ncap": ["Car-to-Car Front turn-across-path"]},
        ["intersection", "oncoming or opposite-direction actor", "left-turn conflict"],
        {"ego": COMMON_EGO, "turning_actor": COMMON_VEHICLE},
        ["scene has junction", "turning_actor heading_relation opposite or crossing"],
        CROSSING_PARAMS,
        "left_turn_across_opposite",
        execution_status="implemented",
    ),
    "right_turn_into_cross_traffic": _template(
        "right_turn_into_cross_traffic",
        "intersection_crossing",
        "Right-turning actor enters a crossing path conflict.",
        {"nhtsa": ["Crossing paths: right turn into path (RTIP)"], "carla": ["SignalizedJunctionRightTurn"], "euro_ncap": []},
        ["right turn area", "cross traffic actor", "ego path near junction"],
        {"ego": COMMON_EGO, "turning_actor": COMMON_VEHICLE},
        ["scene has junction"],
        CROSSING_PARAMS,
        "right_turn_into_cross_traffic",
    ),
    "opposing_turn_conflict": _template(
        "opposing_turn_conflict",
        "intersection_crossing",
        "Opposing actor turns across the ego path.",
        {"nhtsa": ["Crossing paths: left turn across path from opposite direction (LTAP/OD)"], "carla": ["SignalizedJunctionLeftTurn"], "euro_ncap": ["Car-to-Car Front turn-across-path"]},
        ["opposing actor", "turn-across-path possibility", "junction ahead"],
        {"ego": COMMON_EGO, "opposing_turner": COMMON_VEHICLE},
        ["opposing_turner actor longitudinal_m > 0"],
        CROSSING_PARAMS,
        "opposing_turn_conflict",
        execution_status="implemented",
    ),
    "pedestrian_nearside_crossing": _template(
        "pedestrian_nearside_crossing",
        "vru_conflict",
        "Pedestrian crosses ego path from the near side.",
        {"nhtsa": ["Pedestrian: vehicle going straight and pedestrian crossing road"], "carla": ["Pedestrian emerging from behind parked vehicle"], "euro_ncap": ["Car-to-Pedestrian Nearside Adult"]},
        ["pedestrian near roadside or crosswalk", "ego path crosses pedestrian path"],
        {"ego": COMMON_EGO, "vru": ["pedestrian"]},
        ["vru actor category pedestrian"],
        CROSSING_PARAMS,
        "pedestrian_nearside_crossing",
        execution_status="implemented",
    ),
    "pedestrian_farside_crossing": _template(
        "pedestrian_farside_crossing",
        "vru_conflict",
        "Pedestrian crosses ego path from the far side.",
        {"nhtsa": ["Pedestrian: vehicle going straight and pedestrian crossing road"], "carla": ["Obstacle avoidance without prior action"], "euro_ncap": ["Car-to-Pedestrian Farside Adult"]},
        ["pedestrian near far-side curb or crosswalk", "crossing path possible"],
        {"ego": COMMON_EGO, "vru": ["pedestrian"]},
        ["vru actor category pedestrian"],
        CROSSING_PARAMS,
        "pedestrian_farside_crossing",
        execution_status="implemented",
    ),
    "pedestrian_occluded_from_parked_vehicle": _template(
        "pedestrian_occluded_from_parked_vehicle",
        "vru_conflict",
        "Pedestrian emerges from behind an occluding parked vehicle.",
        {"nhtsa": ["Pedestrian: vehicle going straight and pedestrian darting onto road"], "carla": ["Pedestrian emerging from behind parked vehicle"], "euro_ncap": ["Car-to-Pedestrian Nearside Child Obstructed"]},
        ["parked vehicle row", "occlusion note", "pedestrian or potential pedestrian conflict"],
        {"ego": COMMON_EGO, "vru": ["pedestrian"], "occluder": COMMON_VEHICLE},
        ["occluder actor near edge", "vru actor category pedestrian"],
        CROSSING_PARAMS,
        "pedestrian_occluded_from_parked_vehicle",
    ),
    "bicyclist_crossing": _template(
        "bicyclist_crossing",
        "vru_conflict",
        "Bicyclist crosses ego path.",
        {"nhtsa": ["Pedalcyclist: vehicle starting in traffic lane on crossing paths"], "carla": ["Crossing with oncoming bicycles"], "euro_ncap": ["Car-to-Bicyclist Nearside Adult", "Car-to-Bicyclist Farside Adult"]},
        ["bicycle or cyclist near junction/crosswalk", "crossing trajectory possible"],
        {"ego": COMMON_EGO, "vru": ["bicycle"]},
        ["vru actor category bicycle"],
        CROSSING_PARAMS,
        "bicyclist_crossing",
        execution_status="implemented",
    ),
    "motorcyclist_or_scooter_crossing": _template(
        "motorcyclist_or_scooter_crossing",
        "vru_conflict",
        "Motorcyclist or scooter crosses ego path.",
        {"nhtsa": ["Pedalcyclist: vehicle starting in traffic lane on crossing paths"], "carla": ["Crossing traffic"], "euro_ncap": ["Car-to-Motorcyclist Crossing straight crossing path"]},
        ["motorcycle or scooter near intersection", "crossing or weaving path possible"],
        {"ego": COMMON_EGO, "vru": ["motorcycle"]},
        ["vru actor category motorcycle"],
        CROSSING_PARAMS,
        "motorcyclist_or_scooter_crossing",
        execution_status="implemented",
    ),
    "vru_longitudinal_same_direction": _template(
        "vru_longitudinal_same_direction",
        "vru_conflict",
        "Ego closes on a same-direction VRU.",
        {"nhtsa": ["Pedestrian: vehicle going straight and pedestrian walking along road", "Pedalcyclist: vehicle going straight on parallel paths"], "carla": ["Slow moving hazard at lane edge"], "euro_ncap": ["Car-to-Bicyclist Longitudinal Adult"]},
        ["VRU ahead or lane-edge", "same direction", "ego closes from behind"],
        {"ego": COMMON_EGO, "vru": COMMON_VRU},
        ["vru actor longitudinal_m > 0"],
        {**REAR_END_PARAMS, "npc_target_speed_mps": [0.8, 5.0]},
        "vru_longitudinal_same_direction",
    ),
    "oncoming_lane_invasion": _template(
        "oncoming_lane_invasion",
        "opposite_direction",
        "Oncoming actor invades ego lane.",
        {"nhtsa": ["Opposite direction: 2 vehicles going straight and 1 vehicle encroaching"], "carla": ["Vehicle invading lane on bend"], "euro_ncap": ["Car-to-Car Front Head-On lane change"]},
        ["opposite-direction actor", "narrow road or obstacle", "potential lane invasion"],
        {"ego": COMMON_EGO, "oncoming": COMMON_VEHICLE},
        ["oncoming actor heading_relation opposite"],
        STEER_PARAMS,
        "oncoming_lane_invasion",
        execution_status="implemented",
    ),
    "head_on_straight": _template(
        "head_on_straight",
        "opposite_direction",
        "Ego and target approach head-on in the same lane.",
        {"nhtsa": ["Opposite direction: 2 vehicles going straight both in same lane"], "carla": ["ControlLoss"], "euro_ncap": ["Car-to-Car Front Head-On straight"]},
        ["opposite-direction actor aligned with ego lane", "same lane conflict"],
        {"ego": COMMON_EGO, "oncoming": COMMON_VEHICLE},
        ["oncoming actor heading_relation opposite", "abs(oncoming lateral_m) <= same_lane_threshold"],
        CROSSING_PARAMS,
        "head_on_straight",
    ),
    "curve_lane_invasion": _template(
        "curve_lane_invasion",
        "opposite_direction",
        "Oncoming actor invades ego lane on a curve.",
        {"nhtsa": ["Opposite direction: 2 vehicles negotiating a curve and 1 vehicle encroaching"], "carla": ["Vehicle invading lane on bend"], "euro_ncap": []},
        ["curved road", "opposite-direction actor", "lane boundary conflict"],
        {"ego": COMMON_EGO, "oncoming": COMMON_VEHICLE},
        ["road curvature is curve", "oncoming actor heading_relation opposite"],
        STEER_PARAMS,
        "curve_lane_invasion",
    ),
    "opposing_vehicle_passing_conflict": _template(
        "opposing_vehicle_passing_conflict",
        "opposite_direction",
        "Opposing vehicle passes another actor and conflicts with ego.",
        {"nhtsa": ["Opposite direction: involves 1 vehicle passing"], "carla": ["Vehicle invading lane"], "euro_ncap": []},
        ["opposite lane traffic", "passing or overtaking clue", "ego path conflict"],
        {"ego": COMMON_EGO, "oncoming": COMMON_VEHICLE},
        ["oncoming actor heading_relation opposite"],
        STEER_PARAMS,
        "opposing_vehicle_passing_conflict",
    ),
    "door_opening_obstacle": _template(
        "door_opening_obstacle",
        "obstacle_and_edge",
        "Parked vehicle door opens into ego lane.",
        {"nhtsa": ["Lane change: 1 vehicle going straight and another entering or leaving parking position"], "carla": ["Door obstacle"], "euro_ncap": []},
        ["parked vehicle at edge", "ego passes close to parking lane"],
        {"ego": COMMON_EGO, "parked_vehicle": COMMON_VEHICLE},
        ["parked_vehicle lane_side_relation in edge_lane_set"],
        {**STEER_PARAMS, "npc_target_speed_mps": [0.0, 0.5]},
        "door_opening_obstacle",
    ),
    "static_obstacle_in_lane": _template(
        "static_obstacle_in_lane",
        "obstacle_and_edge",
        "Static obstacle blocks ego lane.",
        {"nhtsa": ["Single vehicle: object in road"], "carla": ["Obstacle in lane"], "euro_ncap": []},
        ["cone, barrier, stopped vehicle, debris, or obstacle in ego lane"],
        {"ego": COMMON_EGO, "obstacle": ["car", "truck", "bus", "cone_group", "barrier_group"]},
        ["obstacle actor longitudinal_m > 0"],
        {**REAR_END_PARAMS, "npc_target_speed_mps": [0.0, 0.0]},
        "static_obstacle_in_lane",
    ),
    "slow_hazard_at_lane_edge": _template(
        "slow_hazard_at_lane_edge",
        "obstacle_and_edge",
        "Slow actor partially blocks ego lane at the edge.",
        {"nhtsa": ["Pedestrian: vehicle going straight and pedestrian walking along road"], "carla": ["Slow moving hazard at lane edge"], "euro_ncap": []},
        ["slow actor at lane edge", "partial lane blockage", "ego must brake or nudge"],
        {"ego": COMMON_EGO, "hazard": COMMON_VRU + COMMON_VEHICLE},
        ["hazard actor longitudinal_m > 0", "hazard actor near lane edge"],
        CROSSING_PARAMS,
        "slow_hazard_at_lane_edge",
    ),
    "construction_zone_narrowing": _template(
        "construction_zone_narrowing",
        "obstacle_and_edge",
        "Cones or barriers narrow the ego lane.",
        {"nhtsa": ["Single vehicle: object in road"], "carla": ["Obstacle in lane"], "euro_ncap": []},
        ["cones or barriers", "construction zone", "lane narrowing"],
        {"ego": COMMON_EGO, "obstacle": ["cone_group", "barrier_group"]},
        ["obstacle actor longitudinal_m > 0"],
        {**REAR_END_PARAMS, "npc_trigger_distance_m": [5.0, 18.0]},
        "construction_zone_narrowing",
    ),
    "debris_or_barrier_avoidance": _template(
        "debris_or_barrier_avoidance",
        "obstacle_and_edge",
        "Ego encounters debris or barrier in lane.",
        {"nhtsa": ["Single vehicle: object in road"], "carla": ["Obstacle avoidance without prior action"], "euro_ncap": []},
        ["debris or barrier visible", "ego lane blockage", "avoidance or braking needed"],
        {"ego": COMMON_EGO, "obstacle": ["cone_group", "barrier_group", "car"]},
        ["obstacle actor longitudinal_m > 0"],
        {**REAR_END_PARAMS, "npc_trigger_distance_m": [5.0, 18.0]},
        "debris_or_barrier_avoidance",
    ),
}


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    template = ACCIDENT_TEMPLATE_LIBRARY.get(str(template_id))
    return deepcopy(template) if template else None


def all_templates() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ACCIDENT_TEMPLATE_LIBRARY)


def allowed_template_ids() -> List[str]:
    return sorted(ACCIDENT_TEMPLATE_LIBRARY)


def executable_template_ids() -> List[str]:
    return sorted(
        template_id
        for template_id, template in ACCIDENT_TEMPLATE_LIBRARY.items()
        if template.get("execution_status") == "implemented"
    )


def template_summary_for_prompt() -> str:
    lines = []
    for template_id in allowed_template_ids():
        template = ACCIDENT_TEMPLATE_LIBRARY[template_id]
        cues = "; ".join(template.get("vlm_selection_cues", [])[:3])
        status = template.get("execution_status", "planned_not_implemented")
        lines.append(
            f"- {template_id} | family={template['family']} | status={status} | "
            f"description={template['description']} | cues={cues}"
        )
    return "\n".join(lines)
