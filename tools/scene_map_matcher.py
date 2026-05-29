import ast
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.utils import extract_text_section, read_file, write_to_file


OBJECT_INFO_PREFIX = "__AUTOSCENARIO_OBJECT_INFO__="
SCENE_SPECTATOR_MARKER = "# __AUTOSCENARIO_FOCUS_SPECTATOR__"
SCENE_MATCH_COMMENT_PREFIX = "# scene_match_status: "
VEHICLE_TYPES = {"bike", "car", "jeep", "motorcycle", "suv", "truck", "van"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _distance_2d(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def _angle_difference_deg(angle_a: float, angle_b: float) -> float:
    return abs(_normalize_angle_deg(angle_a - angle_b))


def _heading_to_unit_vector(yaw_deg: float) -> Tuple[float, float]:
    yaw_rad = math.radians(yaw_deg)
    return math.cos(yaw_rad), math.sin(yaw_rad)


def _project_to_local_frame(
    anchor_xy: Tuple[float, float],
    anchor_yaw_deg: float,
    target_xy: Tuple[float, float],
) -> Dict[str, float]:
    dx = target_xy[0] - anchor_xy[0]
    dy = target_xy[1] - anchor_xy[1]
    forward_x, forward_y = _heading_to_unit_vector(anchor_yaw_deg)
    lateral_x, lateral_y = -forward_y, forward_x
    return {
        "dx": dx,
        "dy": dy,
        "distance": math.hypot(dx, dy),
        "longitudinal": dx * forward_x + dy * forward_y,
        "lateral": dx * lateral_x + dy * lateral_y,
    }


def _parse_shape(shape_value: Optional[str]) -> List[Tuple[float, float]]:
    if not shape_value:
        return []

    points = []
    for pair in shape_value.split():
        if "," not in pair:
            continue
        x_str, y_str = pair.split(",", 1)
        points.append((_safe_float(x_str), _safe_float(y_str)))
    return points


def _nearest_segment_info(
    point_xy: Tuple[float, float], polyline: List[Tuple[float, float]]
) -> Dict[str, Optional[float]]:
    if len(polyline) < 2:
        return {"distance": None, "heading": None}

    px, py = point_xy
    best_distance = None
    best_heading = None
    for start, end in zip(polyline, polyline[1:]):
        x1, y1 = start
        x2, y2 = end
        vx, vy = x2 - x1, y2 - y1
        length_sq = vx * vx + vy * vy
        if length_sq == 0:
            continue

        t = ((px - x1) * vx + (py - y1) * vy) / length_sq
        t = max(0.0, min(1.0, t))
        cx = x1 + t * vx
        cy = y1 + t * vy
        distance = math.hypot(px - cx, py - cy)
        heading = math.degrees(math.atan2(vy, vx))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_heading = heading
    return {"distance": best_distance, "heading": best_heading}


def _cluster_headings(yaws: List[float], tolerance_deg: float = 30.0) -> List[List[float]]:
    clusters: List[List[float]] = []
    for yaw in yaws:
        for cluster in clusters:
            center = sum(cluster) / len(cluster)
            if _angle_difference_deg(yaw, center) <= tolerance_deg:
                cluster.append(yaw)
                break
        else:
            clusters.append([yaw])
    return clusters


@dataclass
class SceneEntity:
    name: str
    entity_type: str
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "location": list(self.location),
            "rotation": list(self.rotation),
            "role": self.role,
        }


class SceneMapMatcher:
    """Match a generated SUMO scene to a region in the current CARLA world."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        sample_step: float = 5.0,
        search_radius: float = 30.0,
        coarse_top_k: int = 20,
        large_map_spawn_threshold: int = 1000,
        large_map_candidate_limit: int = 300,
        large_map_step: float = 12.0,
        large_map_name_keywords: Optional[List[str]] = None,
        load_world_name: Optional[str] = None,
        min_refined_score: float = 0.45,
        max_average_snap_distance: float = 5.0,
        topology_weight: float = 0.70,
        side_context_weight: float = 0.20,
        auxiliary_weight: float = 0.10,
        blacklist_radius_m: float = 35.0,
        topology_cache_dir: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sample_step = sample_step
        self.search_radius = search_radius
        self.coarse_top_k = coarse_top_k
        self.large_map_spawn_threshold = large_map_spawn_threshold
        self.large_map_candidate_limit = large_map_candidate_limit
        self.large_map_step = large_map_step
        self.large_map_name_keywords = large_map_name_keywords or ["town12"]
        self.load_world_name = load_world_name
        self.min_refined_score = min_refined_score
        self.max_average_snap_distance = max_average_snap_distance
        self.topology_weight = topology_weight
        self.side_context_weight = side_context_weight
        self.auxiliary_weight = auxiliary_weight
        self.blacklist_radius_m = blacklist_radius_m
        self.topology_cache_dir = topology_cache_dir

    def analyze_scene_assets(self, scene_id: str, output_folder: str) -> str:
        paths = self._build_source_paths(scene_id, output_folder)
        report = self._build_base_report(scene_id, paths)
        report_path = os.path.join(output_folder, f"{scene_id}_scene_match.json")

        try:
            bundle = self._load_generated_scene_bundle(paths)
            scene_features = self._extract_scene_features(bundle)
            match_result = self._match_to_carla(scene_features)

            report["status"] = match_result.get("status", "invalid_input")
            report["world_name"] = match_result.get("world_name")
            report["scene_features"] = scene_features
            report["candidate_summary"] = match_result.get("candidate_summary", {})
            report["best_match"] = match_result.get("best_match")
            report["projected_layout"] = match_result.get("projected_layout", {})
            report["reason"] = match_result.get("reason")
        except Exception as exc:
            report["status"] = "invalid_input"
            report["reason"] = str(exc)

        write_to_file(report_path, json.dumps(report, indent=2, sort_keys=True))
        return report_path

    def analyze_topology_scene_assets(
        self,
        scene_id: str,
        output_folder: str,
        scene_understanding: Dict[str, Any],
        spawn_context: Optional[Dict[str, Any]] = None,
        blacklist_locations: Optional[List[Dict[str, float]]] = None,
    ) -> str:
        report_path = os.path.join(output_folder, f"{scene_id}_scene_match.json")
        candidates_path = os.path.join(output_folder, f"{scene_id}_map_match_candidates.json")
        signature_path = os.path.join(
            output_folder, f"{scene_id}_road_topology_signature.json"
        )
        report = self._build_base_report(
            scene_id,
            {
                "scene_understanding": os.path.join(
                    output_folder, f"{scene_id}_scene_understanding.json"
                ),
                "road_topology_signature": signature_path,
                "map_match_candidates": candidates_path,
            },
        )

        try:
            signature = self.build_road_topology_signature(scene_understanding)
            write_to_file(
                signature_path,
                json.dumps(signature, indent=2, sort_keys=True, ensure_ascii=False),
            )
            scene_features = {
                "description": "",
                "reasoning": "",
                "entities": [],
                "road_topology_signature": signature,
                "semantic_hints": {
                    "mentions_crosswalk": bool(signature.get("has_crosswalk")),
                    "mentions_pedestrian": False,
                    "mentions_parked_vehicle": bool(
                        signature.get("right_parking_presence")
                        or signature.get("left_parking_presence")
                    ),
                    "mentions_barrier": False,
                },
                "road_hints": {
                    "straight_road": signature.get("topology_type")
                    == "straight_two_way",
                    "has_crosswalk": bool(signature.get("has_crosswalk")),
                    "has_center_median": bool(signature.get("has_center_median")),
                    "near_junction": bool(signature.get("junction_visible")),
                },
                "layout_summary": {
                    "anchor_entity": None,
                    "anchor_heading": None,
                    "anchor_lane_count": signature.get("driving_lane_count"),
                    "anchor_near_junction": bool(signature.get("junction_visible")),
                    "target_branch_count": signature.get("target_branch_count", 1),
                    "relative_layout": {},
                },
            }
            report["scene_features"] = scene_features
            spawn_unavailable = (spawn_context or {}).get("status") == "unavailable"
            cache_available = bool(
                self.topology_cache_dir
                and os.path.isdir(self.topology_cache_dir)
                and __import__("glob").glob(
                    os.path.join(self.topology_cache_dir, "*.json")
                )
            )
            if spawn_unavailable and not cache_available:
                report["status"] = "unavailable"
                report["world_name"] = None
                report["reason"] = (spawn_context or {}).get("failure_reason")
                report["candidate_summary"] = {}
                report["best_match"] = None
                report["projected_layout"] = {}
            else:
                match_result = self._match_to_carla(
                    scene_features,
                    blacklist_locations=blacklist_locations,
                    topology_only=True,
                )
                report["status"] = match_result.get("status", "invalid_input")
                report["world_name"] = match_result.get("world_name")
                report["candidate_summary"] = match_result.get("candidate_summary", {})
                report["best_match"] = match_result.get("best_match")
                report["projected_layout"] = {}
                report["reason"] = match_result.get("reason")
                write_to_file(
                    candidates_path,
                    json.dumps(
                        match_result.get("candidate_debug", {}),
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                )
        except Exception as exc:
            report["status"] = "invalid_input"
            report["reason"] = str(exc)

        write_to_file(report_path, json.dumps(report, indent=2, sort_keys=True))
        return report_path

    def analyze_structured_scene_assets(
        self,
        scene_id: str,
        output_folder: str,
        scene_understanding: Dict[str, Any],
        refined_coordinates: Dict[str, Any],
        validation: Optional[Dict[str, Any]] = None,
        spawn_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        report_path = os.path.join(output_folder, f"{scene_id}_scene_match.json")
        report = self._build_base_report(
            scene_id,
            {
                "scene_understanding": os.path.join(
                    output_folder, f"{scene_id}_scene_understanding.json"
                ),
                "refined_coordinates": os.path.join(
                    output_folder, f"{scene_id}_coordinates_projected_refined.json"
                ),
                "relation_validation": os.path.join(
                    output_folder, f"{scene_id}_relation_validation.json"
                ),
            },
        )

        try:
            scene_features = self._extract_structured_scene_features(
                scene_understanding,
                refined_coordinates,
                validation=validation,
                spawn_context=spawn_context,
            )
            report["scene_features"] = scene_features

            spawn_unavailable = (spawn_context or {}).get("status") == "unavailable"
            cache_available = bool(
                self.topology_cache_dir
                and os.path.isdir(self.topology_cache_dir)
                and __import__("glob").glob(
                    os.path.join(self.topology_cache_dir, "*.json")
                )
            )
            if spawn_unavailable and not cache_available:
                report["status"] = "unavailable"
                report["world_name"] = None
                report["reason"] = (spawn_context or {}).get("failure_reason")
                report["candidate_summary"] = {}
                report["best_match"] = None
                report["projected_layout"] = {}
            else:
                match_result = self._match_to_carla(scene_features)
                report["status"] = match_result.get("status", "invalid_input")
                report["world_name"] = match_result.get("world_name")
                report["candidate_summary"] = match_result.get("candidate_summary", {})
                report["best_match"] = match_result.get("best_match")
                report["projected_layout"] = match_result.get("projected_layout", {})
                report["reason"] = match_result.get("reason")
        except Exception as exc:
            report["status"] = "invalid_input"
            report["reason"] = str(exc)

        write_to_file(report_path, json.dumps(report, indent=2, sort_keys=True))
        return report_path

    def apply_match_to_scene_script(
        self, scene_id: str, output_folder: str
    ) -> Optional[str]:
        report_path = os.path.join(output_folder, f"{scene_id}_scene_match.json")
        scene_final_path = os.path.join(output_folder, f"{scene_id}_scene_final.py")
        pre_match_path = os.path.join(output_folder, f"{scene_id}_scene_final_pre_match.py")
        matched_path = os.path.join(output_folder, f"{scene_id}_scene_final_matched.py")

        if not os.path.exists(report_path) or not os.path.exists(scene_final_path):
            return None

        original_code = read_file(scene_final_path)
        shutil.copyfile(scene_final_path, pre_match_path)

        report = json.loads(read_file(report_path))
        if report.get("status") == "matched":
            matched_code = self._rewrite_scene_script_with_match(original_code, report)
        else:
            matched_code = self._build_fallback_matched_script(
                original_code,
                report.get("status", "skipped"),
                report.get("reason", "scene matching did not produce a usable candidate"),
            )

        write_to_file(matched_path, matched_code)
        write_to_file(scene_final_path, matched_code)
        return matched_path

    @staticmethod
    def _build_source_paths(scene_id: str, output_folder: str) -> Dict[str, str]:
        return {
            "scene_obj": os.path.join(output_folder, f"{scene_id}_scene_obj.txt"),
            "scene_final_text": os.path.join(output_folder, f"{scene_id}_scene_final.txt"),
            "scene_final_py": os.path.join(output_folder, f"{scene_id}_scene_final.py"),
            "net_xml": os.path.join(output_folder, f"{scene_id}.net.xml"),
            "nod_xml": os.path.join(output_folder, f"{scene_id}.nod.xml"),
            "edg_xml": os.path.join(output_folder, f"{scene_id}.edg.xml"),
        }

    @staticmethod
    def _build_base_report(scene_id: str, paths: Dict[str, str]) -> Dict[str, Any]:
        return {
            "scene_id": scene_id,
            "source_files": paths,
            "status": "skipped",
            "world_name": None,
            "scene_features": {},
            "candidate_summary": {},
            "best_match": None,
            "projected_layout": {},
            "reason": None,
        }

    @classmethod
    def build_road_topology_signature(
        cls, scene_understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        road_network = scene_understanding.get("road_network", {}) or {}
        metadata = scene_understanding.get("metadata", {}) or {}
        decisive = metadata.get("decisive_map_matching_cues") or {}
        lane_groups = road_network.get("lane_groups") or []
        primary_lane_group = lane_groups[0] if lane_groups and isinstance(lane_groups[0], dict) else {}
        road_segments = road_network.get("road_segments") or []
        special_areas = road_network.get("special_road_areas") or []
        control_elements = road_network.get("control_elements") or []
        junctions = road_network.get("junctions") or []

        forward_lanes = int(primary_lane_group.get("forward_lane_count", 1) or 1)
        opposing_lanes = int(primary_lane_group.get("opposing_lane_count", 0) or 0)
        left_parking = int(primary_lane_group.get("left_parking_lane_count", 0) or 0) > 0
        right_parking = int(primary_lane_group.get("right_parking_lane_count", 0) or 0) > 0

        junction_visible = cls._explicit_bool(
            decisive,
            ("junction_visible", "near_junction", "intersection_visible"),
            default=None,
        )
        target_branch_count = 1
        junction_type = "none"
        if junction_visible is None:
            junction_visible = False
            for junction in junctions:
                parsed_visible, parsed_type, parsed_branch_count = cls._parse_junction_hint(junction)
                if parsed_visible:
                    junction_visible = True
                    junction_type = parsed_type
                    target_branch_count = parsed_branch_count
                    break
        else:
            for junction in junctions:
                _visible, parsed_type, parsed_branch_count = cls._parse_junction_hint(junction)
                if parsed_type != "none":
                    junction_type = parsed_type
                    target_branch_count = parsed_branch_count
                    break

        if not junction_visible:
            junction_type = "none"
            target_branch_count = 1

        geometry_text = " ".join(
            str(segment.get("geometry_type") or segment.get("curvature_hint") or "")
            for segment in road_segments
            if isinstance(segment, dict)
        ).lower()
        straight_road = cls._explicit_bool(
            decisive,
            ("simple_straight_street", "straight_road"),
            default=("straight" in geometry_text and "curve" not in geometry_text),
        )
        curved_road = "curve" in geometry_text or "curved" in geometry_text
        two_way = "two_way" in str(road_network.get("directionality") or "").lower() or opposing_lanes > 0

        topology_type = cls._topology_type_from_hints(
            junction_visible=junction_visible,
            junction_type=junction_type,
            target_branch_count=target_branch_count,
            straight_road=straight_road,
            curved_road=curved_road,
            two_way=two_way,
        )

        has_crosswalk = cls._has_area_or_control(
            special_areas,
            control_elements,
            ("crosswalk", "zebra", "pedestrian_crossing", "斑马"),
        ) or bool(decisive.get("near_crosswalk"))
        has_traffic_light = cls._has_area_or_control(
            special_areas,
            control_elements,
            ("traffic_light", "traffic_signal", "signal", "red_light", "红绿灯"),
        )
        has_traffic_sign = cls._has_area_or_control(
            special_areas,
            control_elements,
            ("traffic_sign", "sign", "direction_sign", "road_sign", "标志"),
        )
        has_center_median = cls._explicit_bool(
            decisive,
            ("raised_center_median_visible", "center_island_visible", "has_center_median"),
            default=False,
        )

        side_context = cls._side_context(scene_understanding)
        return {
            "topology_type": topology_type,
            "junction_type": junction_type,
            "junction_visible": bool(junction_visible),
            "target_branch_count": int(target_branch_count),
            "directionality": "two_way" if two_way else "one_way_or_unknown",
            "driving_lane_count": max(1, forward_lanes + opposing_lanes),
            "forward_lane_count": forward_lanes,
            "opposing_lane_count": opposing_lanes,
            "left_parking_presence": bool(
                left_parking or decisive.get("continuous_curbside_parking_left")
            ),
            "right_parking_presence": bool(
                right_parking or decisive.get("continuous_curbside_parking_right")
            ),
            "has_crosswalk": bool(has_crosswalk),
            "has_traffic_light": bool(has_traffic_light),
            "has_traffic_sign": bool(has_traffic_sign),
            "has_center_median": bool(has_center_median),
            "side_context": side_context,
        }

    @staticmethod
    def _explicit_bool(
        payload: Dict[str, Any],
        keys: Tuple[str, ...],
        default: Optional[bool],
    ) -> Optional[bool]:
        for key in keys:
            if key in payload and isinstance(payload.get(key), bool):
                return bool(payload[key])
        return default

    @staticmethod
    def _parse_junction_hint(junction: Any) -> Tuple[bool, str, int]:
        if not isinstance(junction, dict):
            return False, "none", 1
        raw_type = str(junction.get("type") or junction.get("junction_type") or "").lower()
        location = str(junction.get("location") or junction.get("position") or "").lower()
        complexity = str(junction.get("complexity") or "").lower()
        confidence_raw = junction.get("confidence")
        confidence_str = str(confidence_raw or "").lower()

        # Midblock features (crosswalk, pedestrian zone) are NOT junctions even
        # though "crosswalk" contains the substring "cross".  Check these first
        # before any intersection classification to avoid false positives.
        if any(kw in raw_type for kw in (
            "midblock", "crosswalk", "pedestrian_crossing", "pedestrian_zone",
            "not_junction", "not_a_junction",
        )):
            return False, "none", 1

        # Complexity field explicitly says "not a junction at ego position".
        if "not_a_junction" in complexity or "no_junction" in complexity:
            return False, "none", 1

        # Explicit non-junction / not-visible indicators in the type string.
        if (
            "none" in raw_type
            or "no_" in raw_type
            or "not_visible" in raw_type
            or "distant" in raw_type   # e.g. "distant_intersection"
            or "far_" in raw_type
            or "far" in location
            or "distant" in location
            or confidence_str == "low"
        ):
            return False, "none", 1

        # Numeric confidence: below 0.5 is not reliably visible.
        try:
            if float(confidence_raw) < 0.5:
                return False, "none", 1
        except (TypeError, ValueError):
            pass

        # Classify junction type using precise terms so that "crosswalk" can
        # never accidentally trigger the cross_intersection branch.
        if "t_junction" in raw_type or "t-junction" in raw_type or "tee" in raw_type:
            return True, "t_junction", 3
        if raw_type.startswith("t_") or raw_type == "t":
            return True, "t_junction", 3
        if (
            "cross_intersection" in raw_type
            or "four_way" in raw_type
            or "four-way" in raw_type
            or "fourway" in raw_type
            or "十字" in raw_type
        ):
            return True, "cross_intersection", 4
        if "roundabout" in raw_type or "rotary" in raw_type:
            return True, "roundabout", 4
        if "multi" in raw_type or "complex" in raw_type:
            return True, "multi_branch", 5
        # Generic fallback: "cross" alone (e.g. "crossroads"), "intersection",
        # "junction" — only reached after all false-positive guards above.
        if "cross" in raw_type or "intersection" in raw_type or "junction" in raw_type:
            return True, "intersection", 3
        if raw_type:
            return True, "intersection", 3
        return False, "none", 1

    @staticmethod
    def _topology_type_from_hints(
        junction_visible: bool,
        junction_type: str,
        target_branch_count: int,
        straight_road: bool,
        curved_road: bool,
        two_way: bool,
    ) -> str:
        if junction_visible:
            if junction_type in {"t_junction", "cross_intersection", "roundabout", "multi_branch"}:
                return junction_type
            if target_branch_count >= 5:
                return "multi_branch"
            if target_branch_count == 4:
                return "cross_intersection"
            return "t_junction"
        if curved_road:
            return "curve"
        if straight_road and two_way:
            return "straight_two_way"
        if straight_road:
            return "straight_road"
        return "unknown"

    @staticmethod
    def _has_area_or_control(
        special_areas: List[Any],
        control_elements: List[Any],
        keywords: Tuple[str, ...],
    ) -> bool:
        for item in list(special_areas or []) + list(control_elements or []):
            if not isinstance(item, dict):
                continue
            text = json.dumps(item, sort_keys=True, ensure_ascii=False).lower()
            if any(keyword.lower() in text for keyword in keywords):
                return True
        return False

    @staticmethod
    def _side_context(scene_understanding: Dict[str, Any]) -> Dict[str, Any]:
        road_network = scene_understanding.get("road_network", {}) or {}
        general_environment = scene_understanding.get("general_environment", {}) or {}
        text_all = json.dumps(scene_understanding, ensure_ascii=False).lower()
        text_left = json.dumps(
            general_environment.get("roadside_context_left") or [],
            ensure_ascii=False,
        ).lower()
        text_right = json.dumps(
            general_environment.get("roadside_context_right") or [],
            ensure_ascii=False,
        ).lower()
        if not text_left or text_left == "[]":
            text_left = text_all
        if not text_right or text_right == "[]":
            text_right = text_all
        special_areas = road_network.get("special_road_areas") or []
        left_parking = any(
            isinstance(area, dict)
            and "parking" in str(area.get("type") or "").lower()
            and "left" in json.dumps(area, ensure_ascii=False).lower()
            for area in special_areas
        )
        right_parking = any(
            isinstance(area, dict)
            and "parking" in str(area.get("type") or "").lower()
            and "right" in json.dumps(area, ensure_ascii=False).lower()
            for area in special_areas
        )
        return {
            "left_continuous_buildings": any(
                token in text_left for token in ("building", "shop", "storefront", "建筑", "商铺")
            ),
            "right_continuous_buildings": any(
                token in text_right for token in ("building", "shop", "storefront", "建筑", "商铺")
            ),
            "tree_lined": any(token in text_all for token in ("tree", "trees", "树")),
            "left_parking_strip": left_parking,
            "right_parking_strip": right_parking,
        }

    def _extract_structured_scene_features(
        self,
        scene_understanding: Dict[str, Any],
        refined_coordinates: Dict[str, Any],
        validation: Optional[Dict[str, Any]],
        spawn_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        entities = self._build_structured_entities(refined_coordinates)
        anchor = self._select_anchor(entities)
        if anchor is None:
            raise ValueError("Unable to choose an anchor entity for scene matching.")

        road_network = scene_understanding.get("road_network", {}) or {}
        combined_text = json.dumps(scene_understanding, sort_keys=True).lower()
        anchor_lane = (
            refined_coordinates.get("selected_anchor_lane")
            or self._first_topology_lane(spawn_context)
            or {}
        )
        lane_count = self._structured_lane_count(road_network)
        anchor_heading = self._structured_anchor_heading(anchor, anchor_lane)
        anchor_near_junction = self._structured_anchor_near_junction(
            road_network, spawn_context
        )
        target_branch_count = self._structured_target_branch_count(
            road_network, spawn_context, anchor_near_junction
        )

        anchor_xy = (anchor.location[0], anchor.location[1])
        relative_layout = {}
        for entity in entities:
            relative_layout[entity.name] = {
                **entity.to_dict(),
                "relative_to_anchor": _project_to_local_frame(
                    anchor_xy,
                    anchor.rotation[1],
                    (entity.location[0], entity.location[1]),
                ),
                "structured_match": {
                    "projected_lane": self._structured_entity_lane(
                        refined_coordinates, entity.name
                    )
                },
            }

        parked_vehicle_count = sum(1 for entity in entities if entity.role == "parked_vehicle")
        pedestrian_count = sum(1 for entity in entities if entity.role == "pedestrian")
        static_object_count = sum(1 for entity in entities if entity.role == "static_object")
        dynamic_vehicle_count = sum(
            1
            for entity in entities
            if entity.role in {"ego", "dynamic_vehicle", "parked_vehicle"}
        )
        road_hints = self._structured_road_hints(road_network, combined_text)

        return {
            "description": "",
            "reasoning": "",
            "entities": [entity.to_dict() for entity in entities],
            "structured_summary": {
                "lane_count": lane_count,
                "junction_count": len(road_network.get("junctions") or []),
                "validation_summary": (validation or {}).get("summary", {}),
                "spawn_context_status": (spawn_context or {}).get("status"),
            },
            "semantic_hints": {
                "mentions_crosswalk": road_hints["has_crosswalk"],
                "mentions_pedestrian": "pedestrian" in combined_text,
                "mentions_parked_vehicle": "parked" in combined_text,
                "mentions_barrier": "barrier" in combined_text,
            },
            "road_hints": road_hints,
            "layout_summary": {
                "anchor_entity": anchor.to_dict(),
                "anchor_heading": anchor_heading,
                "anchor_lane_count": lane_count,
                "anchor_nearest_lane": anchor_lane or None,
                "anchor_junction_distance": None,
                "anchor_near_junction": anchor_near_junction,
                "target_branch_count": target_branch_count,
                "nearest_junction": None,
                "dynamic_vehicle_count": dynamic_vehicle_count,
                "parked_vehicle_count": parked_vehicle_count,
                "pedestrian_count": pedestrian_count,
                "static_object_count": static_object_count,
                "relative_layout": relative_layout,
            },
        }

    def _build_structured_entities(
        self, refined_coordinates: Dict[str, Any]
    ) -> List[SceneEntity]:
        entities = []
        for entity in refined_coordinates.get("entities", []) or []:
            entity_id = str(entity.get("id") or entity.get("entity_id") or "")
            if not entity_id:
                continue
            category = str(entity.get("category") or "car")
            spawn_kind = str(entity.get("spawn_kind") or "vehicle")
            blueprint_name = str(entity.get("blueprint_name") or category)
            role = self._infer_structured_role(entity)
            entity_type = self._structured_entity_type(category, spawn_kind, blueprint_name)
            location = self._coerce_xyz_triplet(entity.get("location", {}))
            rotation_dict = entity.get("rotation", {}) or {}
            rotation = (
                _safe_float(rotation_dict.get("pitch")),
                _safe_float(rotation_dict.get("yaw")),
                _safe_float(rotation_dict.get("roll")),
            )
            entities.append(
                SceneEntity(
                    name=entity_id,
                    entity_type=entity_type,
                    location=location,
                    rotation=rotation,
                    role=role,
                )
            )
        return entities

    @staticmethod
    def _infer_structured_role(entity: Dict[str, Any]) -> str:
        category = str(entity.get("category") or "")
        priority = str(entity.get("priority") or "")
        spawn_kind = str(entity.get("spawn_kind") or "")
        if priority == "subject" and category not in {"pedestrian", "cone_group", "barrier_group"}:
            return "dynamic_vehicle"
        if category == "parked_vehicle":
            return "parked_vehicle"
        if spawn_kind == "pedestrian" or category == "pedestrian":
            return "pedestrian"
        if spawn_kind == "static" or category in {"cone_group", "barrier_group"}:
            return "static_object"
        if spawn_kind == "vehicle":
            return "dynamic_vehicle"
        return "other"

    @staticmethod
    def _structured_entity_type(
        category: str, spawn_kind: str, blueprint_name: str
    ) -> str:
        if spawn_kind == "pedestrian":
            return "pedestrian"
        if spawn_kind == "static":
            return str(blueprint_name or category)
        if category == "parked_vehicle":
            return "car" if blueprint_name in {"", "None"} else blueprint_name
        if category in {"motorcycle", "bicycle", "bike", "truck", "car"}:
            return "bike" if category == "bicycle" else category
        return blueprint_name or "car"

    @staticmethod
    def _structured_lane_count(road_network: Dict[str, Any]) -> Optional[int]:
        lane_groups = road_network.get("lane_groups") or []
        counts = []
        for lane_group in lane_groups:
            if not isinstance(lane_group, dict):
                continue
            counts.append(
                int(lane_group.get("forward_lane_count", 0) or 0)
                + int(lane_group.get("opposing_lane_count", 0) or 0)
                + int(lane_group.get("left_parking_lane_count", 0) or 0)
                + int(lane_group.get("right_parking_lane_count", 0) or 0)
            )
        return max(counts) if counts else None

    @staticmethod
    def _first_topology_lane(
        spawn_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        topology_sample = (spawn_context or {}).get("topology_sample") or []
        for item in topology_sample:
            if isinstance(item, dict):
                return item
        return None

    @staticmethod
    def _structured_anchor_heading(
        anchor: SceneEntity, anchor_lane: Dict[str, Any]
    ) -> float:
        start = anchor_lane.get("start") or {}
        end = anchor_lane.get("end") or {}
        if start and end:
            dx = _safe_float(end.get("x")) - _safe_float(start.get("x"))
            dy = _safe_float(end.get("y")) - _safe_float(start.get("y"))
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                return math.degrees(math.atan2(dy, dx))
        return anchor.rotation[1]

    @staticmethod
    def _structured_anchor_near_junction(
        road_network: Dict[str, Any], spawn_context: Optional[Dict[str, Any]]
    ) -> bool:
        for junction in road_network.get("junctions") or []:
            if not isinstance(junction, dict):
                continue
            confidence = str(junction.get("confidence") or "").lower()
            location = str(junction.get("location") or junction.get("position") or "").lower()
            if confidence == "low" or "far" in location or "distant" in location:
                continue
            return True
        return False

    @staticmethod
    def _structured_road_hints(
        road_network: Dict[str, Any], combined_text: str
    ) -> Dict[str, bool]:
        road_segments = road_network.get("road_segments") or []
        straight_road = any(
            "straight" in str(segment.get("geometry_type") or segment.get("curvature_hint") or "").lower()
            for segment in road_segments
            if isinstance(segment, dict)
        )
        special_areas = road_network.get("special_road_areas") or []
        has_crosswalk = (
            "crosswalk" in combined_text
            or "zebra" in combined_text
            or "斑马" in combined_text
            or any(
                "crosswalk" in str(area.get("type") or area.get("location") or "").lower()
                for area in special_areas
                if isinstance(area, dict)
            )
        )
        has_center_median = any(
            token in combined_text
            for token in (
                '"has_center_median": true',
                '"center_median_present": true',
                "raised median",
                "raised_median",
                "center island",
                "central island",
                "中央隔离带",
                "中央岛",
            )
        )
        near_junction = any(
            isinstance(junction, dict)
            and str(junction.get("confidence") or "").lower() != "low"
            and "far" not in str(junction.get("location") or "").lower()
            for junction in road_network.get("junctions") or []
        )
        return {
            "straight_road": straight_road,
            "has_crosswalk": has_crosswalk,
            "has_center_median": has_center_median,
            "near_junction": near_junction,
        }

    @staticmethod
    def _structured_target_branch_count(
        road_network: Dict[str, Any],
        spawn_context: Optional[Dict[str, Any]],
        anchor_near_junction: bool,
    ) -> int:
        if not anchor_near_junction:
            return 1
        junctions = road_network.get("junctions") or []
        if junctions:
            return max(2, len(junctions) + 1)
        road_ids = {
            item.get("road_id")
            for item in (spawn_context or {}).get("topology_sample") or []
            if isinstance(item, dict) and item.get("road_id") is not None
        }
        return max(2, len(road_ids))

    @staticmethod
    def _structured_entity_lane(
        refined_coordinates: Dict[str, Any], entity_name: str
    ) -> Optional[Dict[str, Any]]:
        for entity in refined_coordinates.get("entities", []) or []:
            if str(entity.get("id")) == str(entity_name):
                return entity.get("projected_lane")
        return None

    def _load_generated_scene_bundle(self, paths: Dict[str, str]) -> Dict[str, Any]:
        required_paths = ("scene_obj", "scene_final_text", "net_xml", "nod_xml", "edg_xml")
        missing = [key for key in required_paths if not os.path.exists(paths[key])]
        if missing:
            raise FileNotFoundError(
                "Missing scene matching inputs: " + ", ".join(sorted(missing))
            )

        description, reasoning = self._parse_scene_text(paths["scene_final_text"])
        agent_dict, object_dict = self._parse_scene_obj_output(paths["scene_obj"])
        entities = self._build_entities(agent_dict, object_dict)
        if not entities:
            raise ValueError("No entities were parsed from the generated scene object info.")

        return {
            "description": description,
            "reasoning": reasoning,
            "entities": entities,
            "sumo_topology": self._parse_sumo_topology(
                paths["net_xml"], paths["nod_xml"], paths["edg_xml"]
            ),
        }

    @staticmethod
    def _parse_scene_text(scene_final_text_path: str) -> Tuple[str, str]:
        text = read_file(scene_final_text_path)
        description = extract_text_section(
            text, r"## Description\s+(.*?)(?=\n## |\Z)"
        )
        reasoning = extract_text_section(text, r"## Reasoning\s+(.*?)(?=\n## |\Z)")
        return description or "", reasoning or ""

    def _parse_scene_obj_output(
        self, scene_obj_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        text = read_file(scene_obj_path)
        payload = self._extract_object_info_payload(text)
        if payload is not None:
            agent_dict = payload.get("agent_dict", {})
            object_dict = payload.get("object_dict", {})
            if not isinstance(agent_dict, dict) or not isinstance(object_dict, dict):
                raise ValueError("Structured object info payload is malformed.")
            return agent_dict, object_dict

        return self._parse_legacy_scene_obj_output(text)

    @staticmethod
    def _extract_object_info_payload(text: str) -> Optional[Dict[str, Any]]:
        for line in reversed(text.splitlines()):
            if not line.startswith(OBJECT_INFO_PREFIX):
                continue
            payload_text = line[len(OBJECT_INFO_PREFIX) :].strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed structured object info payload: {exc.msg}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError("Structured object info payload is not a dictionary.")
            return payload
        return None

    @staticmethod
    def _parse_legacy_scene_obj_output(
        text: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        agent_dict = None
        object_dict = {}
        unlabeled_dicts: List[Dict[str, Any]] = []
        pending_label = None

        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("Agent Dictionary:"):
                inline_value = stripped.split(":", 1)[1].strip()
                if inline_value:
                    agent_dict = ast.literal_eval(inline_value)
                    pending_label = None
                else:
                    pending_label = "agent"
                continue
            if stripped.startswith("Object Dictionary:"):
                inline_value = stripped.split(":", 1)[1].strip()
                if inline_value:
                    object_dict = ast.literal_eval(inline_value)
                    pending_label = None
                else:
                    pending_label = "object"
                continue

            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                continue

            if not isinstance(parsed, dict):
                continue
            if pending_label == "agent":
                agent_dict = parsed
                pending_label = None
                continue
            if pending_label == "object":
                object_dict = parsed
                pending_label = None
                continue
            unlabeled_dicts.append(parsed)

        if agent_dict is None and unlabeled_dicts:
            agent_dict = unlabeled_dicts[0]
        if not object_dict and len(unlabeled_dicts) >= 2:
            object_dict = unlabeled_dicts[1]
        if agent_dict is None:
            raise ValueError("Failed to parse legacy scene object dictionaries.")
        return agent_dict, object_dict

    def _build_entities(
        self, agent_dict: Dict[str, Any], object_dict: Dict[str, Any]
    ) -> List[SceneEntity]:
        entities: List[SceneEntity] = []
        for name, payload in agent_dict.items():
            entities.extend(self._expand_entity_payload(name, payload, source="agent"))
        for name, payload in object_dict.items():
            entities.extend(self._expand_entity_payload(name, payload, source="object"))
        return entities

    def _expand_entity_payload(
        self, name: str, payload: Any, source: str
    ) -> List[SceneEntity]:
        if isinstance(payload, list):
            entities: List[SceneEntity] = []
            for index, item in enumerate(payload, start=1):
                if isinstance(item, dict):
                    entities.append(
                        self._build_entity(f"{name}_{index}", item, source=source)
                    )
            return entities
        if isinstance(payload, dict):
            return [self._build_entity(name, payload, source=source)]
        return []

    def _build_entity(self, name: str, payload: Dict[str, Any], source: str) -> SceneEntity:
        entity_type = str(payload.get("type", "unknown"))
        role = self._infer_role(name, entity_type)
        if source == "object" and role == "other":
            role = "static_object"

        location_raw = payload.get("location", (0.0, 0.0, 0.0))
        rotation_raw = payload.get("rotation", (0.0, 0.0, 0.0))
        location = self._coerce_xyz_triplet(location_raw)
        rotation = self._coerce_xyz_triplet(rotation_raw)
        return SceneEntity(
            name=name,
            entity_type=entity_type,
            location=location,
            rotation=rotation,
            role=role,
        )

    @staticmethod
    def _coerce_xyz_triplet(raw_value: Any) -> Tuple[float, float, float]:
        if isinstance(raw_value, dict):
            return (
                _safe_float(raw_value.get("x")),
                _safe_float(raw_value.get("y")),
                _safe_float(raw_value.get("z")),
            )
        if isinstance(raw_value, (list, tuple)) and len(raw_value) >= 3:
            return (
                _safe_float(raw_value[0]),
                _safe_float(raw_value[1]),
                _safe_float(raw_value[2]),
            )
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _infer_role(name: str, entity_type: str) -> str:
        lowered = name.lower()
        normalized_type = entity_type.lower()
        if lowered in {"ego", "av", "host"}:
            return "ego"
        if normalized_type == "pedestrian":
            return "pedestrian"
        if lowered.startswith("pc") or "park" in lowered:
            return "parked_vehicle"
        if normalized_type in VEHICLE_TYPES:
            return "dynamic_vehicle"
        return "other"

    def _select_anchor(self, entities: List[SceneEntity]) -> Optional[SceneEntity]:
        if not entities:
            return None

        ranked_entities = []
        for index, entity in enumerate(entities):
            ranked_entities.append((self._anchor_priority(entity), index, entity))
        ranked_entities.sort(key=lambda item: (item[0], item[1]))
        return ranked_entities[0][2]

    @staticmethod
    def _anchor_priority(entity: SceneEntity) -> int:
        lowered_name = entity.name.lower()
        if entity.role == "ego":
            return 0

        if entity.role == "dynamic_vehicle":
            if any(
                keyword in lowered_name
                for keyword in ("ego", "av", "host", "vehicle", "agent", "car")
            ):
                return 1
            if entity.entity_type in {"car", "jeep", "suv", "truck", "van"}:
                return 2
            if entity.entity_type == "motorcycle":
                return 3
            return 4

        if entity.role == "parked_vehicle":
            return 5
        if entity.role == "pedestrian":
            return 6
        return 7

    def _parse_sumo_topology(
        self, net_xml_path: str, nod_xml_path: str, edg_xml_path: str
    ) -> Dict[str, Any]:
        net_root = ET.parse(net_xml_path).getroot()
        nod_root = ET.parse(nod_xml_path).getroot()
        edg_root = ET.parse(edg_xml_path).getroot()

        nodes = {
            node.get("id"): {
                "x": _safe_float(node.get("x")),
                "y": _safe_float(node.get("y")),
                "type": node.get("type", "unknown"),
            }
            for node in nod_root.findall("node")
        }

        edges: Dict[str, Dict[str, Any]] = {}
        for edge in net_root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            lanes = edge.findall("lane")
            edges[edge.get("id")] = {
                "edge_id": edge.get("id"),
                "from": edge.get("from"),
                "to": edge.get("to"),
                "lane_count": len(lanes),
                "length": max(
                    [_safe_float(lane.get("length")) for lane in lanes] or [0.0]
                ),
                "lane_shapes": [
                    {
                        "lane_id": lane.get("id"),
                        "shape": _parse_shape(lane.get("shape")),
                    }
                    for lane in lanes
                ],
            }

        edge_nodes = {}
        for edge in edg_root.findall("edge"):
            if edge.attrib.get("function") == "internal":
                continue
            edge_nodes[edge.get("id")] = {
                "from": edge.get("from"),
                "to": edge.get("to"),
            }
        for edge_id, payload in edges.items():
            if edge_id in edge_nodes:
                payload["from"] = edge_nodes[edge_id]["from"]
                payload["to"] = edge_nodes[edge_id]["to"]

        junctions = {}
        for junction in net_root.findall("junction"):
            if junction.attrib.get("type") == "internal":
                continue
            inc_lanes = junction.attrib.get("incLanes", "").split()
            incoming_edges = sorted(
                {lane.rsplit("_", 1)[0] for lane in inc_lanes if lane}
            )
            junctions[junction.get("id")] = {
                "junction_id": junction.get("id"),
                "type": junction.attrib.get("type", "unknown"),
                "x": _safe_float(junction.get("x")),
                "y": _safe_float(junction.get("y")),
                "incoming_edges": incoming_edges,
                "incoming_edge_count": len(incoming_edges),
            }

        return {
            "nodes": nodes,
            "edges": edges,
            "junctions": junctions,
            "edge_count": len(edges),
            "lane_count": sum(edge["lane_count"] for edge in edges.values()),
            "junction_count": len(junctions),
        }

    def _extract_scene_features(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        description = bundle["description"]
        reasoning = bundle["reasoning"]
        combined_text = f"{description}\n{reasoning}".lower()
        entities: List[SceneEntity] = bundle["entities"]
        sumo_topology = bundle["sumo_topology"]
        anchor = self._select_anchor(entities)
        if anchor is None:
            raise ValueError("Unable to choose an anchor entity for scene matching.")

        entity_matches = self._match_entities_to_sumo_edges(entities, sumo_topology)
        anchor_match = entity_matches.get(anchor.name)
        anchor_heading = (
            anchor_match.get("lane_heading")
            if anchor_match and anchor_match.get("lane_heading") is not None
            else anchor.rotation[1]
        )
        nearest_junction = self._nearest_junction(anchor.location, sumo_topology)
        anchor_junction_distance = (
            nearest_junction["distance"] if nearest_junction is not None else None
        )
        anchor_near_junction = (
            anchor_junction_distance is not None and anchor_junction_distance <= 30.0
        )
        target_branch_count = 1
        if nearest_junction is not None and anchor_near_junction:
            target_branch_count = max(1, nearest_junction["incoming_edge_count"])

        relative_layout = {}
        anchor_xy = (anchor.location[0], anchor.location[1])
        for entity in entities:
            relative_layout[entity.name] = {
                **entity.to_dict(),
                "relative_to_anchor": _project_to_local_frame(
                    anchor_xy, anchor.rotation[1], (entity.location[0], entity.location[1])
                ),
                "sumo_match": entity_matches.get(entity.name),
            }

        parked_vehicle_count = sum(1 for entity in entities if entity.role == "parked_vehicle")
        pedestrian_count = sum(1 for entity in entities if entity.role == "pedestrian")
        static_object_count = sum(1 for entity in entities if entity.role == "static_object")
        dynamic_vehicle_count = sum(
            1
            for entity in entities
            if entity.role in {"ego", "dynamic_vehicle", "parked_vehicle"}
        )

        return {
            "description": description,
            "reasoning": reasoning,
            "entities": [entity.to_dict() for entity in entities],
            "sumo_summary": {
                "edge_count": sumo_topology["edge_count"],
                "lane_count": sumo_topology["lane_count"],
                "junction_count": sumo_topology["junction_count"],
            },
            "semantic_hints": {
                "mentions_crosswalk": "crosswalk" in combined_text,
                "mentions_pedestrian": "pedestrian" in combined_text,
                "mentions_parked_vehicle": "parked" in combined_text,
                "mentions_barrier": "barrier" in combined_text,
            },
            "layout_summary": {
                "anchor_entity": anchor.to_dict(),
                "anchor_heading": anchor_heading,
                "anchor_lane_count": anchor_match.get("lane_count") if anchor_match else None,
                "anchor_nearest_lane": anchor_match,
                "anchor_junction_distance": anchor_junction_distance,
                "anchor_near_junction": anchor_near_junction,
                "target_branch_count": target_branch_count,
                "nearest_junction": nearest_junction,
                "dynamic_vehicle_count": dynamic_vehicle_count,
                "parked_vehicle_count": parked_vehicle_count,
                "pedestrian_count": pedestrian_count,
                "static_object_count": static_object_count,
                "relative_layout": relative_layout,
            },
        }

    def _match_entities_to_sumo_edges(
        self, entities: List[SceneEntity], sumo_topology: Dict[str, Any]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        matches = {}
        for entity in entities:
            point_xy = (entity.location[0], entity.location[1])
            best_match = None
            for edge in sumo_topology["edges"].values():
                for lane in edge["lane_shapes"]:
                    segment_info = _nearest_segment_info(point_xy, lane["shape"])
                    if segment_info["distance"] is None:
                        continue
                    candidate = {
                        "edge_id": edge["edge_id"],
                        "lane_id": lane["lane_id"],
                        "lane_count": edge["lane_count"],
                        "distance_to_lane": segment_info["distance"],
                        "lane_heading": segment_info["heading"],
                    }
                    if best_match is None or (
                        candidate["distance_to_lane"] < best_match["distance_to_lane"]
                    ):
                        best_match = candidate
            matches[entity.name] = best_match
        return matches

    @staticmethod
    def _nearest_junction(
        location: Tuple[float, float, float], sumo_topology: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        best_junction = None
        point_xy = (location[0], location[1])
        for junction in sumo_topology["junctions"].values():
            distance = _distance_2d(point_xy, (junction["x"], junction["y"]))
            candidate = {**junction, "distance": distance}
            if best_junction is None or distance < best_junction["distance"]:
                best_junction = candidate
        return best_junction

    def _cache_is_blacklisted(
        self,
        loc: Dict[str, float],
        blacklist_locations: Optional[List[Dict[str, float]]],
    ) -> bool:
        for item in blacklist_locations or []:
            if not isinstance(item, dict):
                continue
            dist = _distance_2d(
                (_safe_float(loc.get("x")), _safe_float(loc.get("y"))),
                (_safe_float(item.get("x")), _safe_float(item.get("y"))),
            )
            if dist <= _safe_float(item.get("radius"), self.blacklist_radius_m):
                return True
        return False

    def _match_from_cache(
        self,
        scene_features: Dict[str, Any],
        blacklist_locations: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Score candidates from pre-computed JSON caches across multiple maps.

        No CARLA load_world() is called. Returns the same structure as _match_to_carla.
        """
        import glob
        import json as _json

        cache_files = sorted(glob.glob(os.path.join(self.topology_cache_dir, "*.json")))
        if not cache_files:
            return {
                "status": "unavailable",
                "world_name": None,
                "reason": f"No topology cache files found in {self.topology_cache_dir}",
            }

        global_best: Optional[Dict[str, Any]] = None
        global_best_score = -1.0
        global_best_world: Optional[str] = None

        fallback_best: Optional[Dict[str, Any]] = None
        fallback_best_score = -1.0
        fallback_best_world: Optional[str] = None

        top_candidates_summary: List[Dict[str, Any]] = []

        # World-level blacklist: entries with a "world" key skip the entire cache file.
        blacklisted_worlds = {
            item["world"]
            for item in (blacklist_locations or [])
            if isinstance(item, dict) and item.get("world")
        }

        for cache_path in cache_files:
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = _json.load(f)
            except Exception:
                continue
            world_name = cache.get("world_name", os.path.basename(cache_path))
            if world_name in blacklisted_worlds:
                continue
            candidates = cache.get("candidates") or []

            map_best: Optional[Dict[str, Any]] = None
            map_best_score = -1.0

            for candidate in candidates:
                loc = candidate.get("location") or {}
                if self._cache_is_blacklisted(loc, blacklist_locations):
                    continue
                score, details = self._score_candidate_features(candidate, scene_features)
                details["blacklisted"] = False
                hard_reject = bool(details.get("hard_reject"))

                entry = {
                    "score": score,
                    "score_details": details,
                    "hard_reject": hard_reject,
                    "reject_reason": details.get("reject_reason"),
                    "search_strategy": "cache",
                    "location": loc,
                    "yaw": candidate.get("yaw", 0.0),
                    "candidate_lane": candidate.get("candidate_lane", {}),
                    **{k: candidate[k] for k in candidate
                       if k not in ("location", "yaw", "candidate_lane")},
                    "_world_name": world_name,
                }

                # Track global fallback (best score regardless of hard_reject)
                if score > fallback_best_score:
                    fallback_best_score = score
                    fallback_best = entry
                    fallback_best_world = world_name

                # Track per-map best (non-hard-reject preferred)
                if score > map_best_score:
                    map_best_score = score
                    map_best = entry

                # Track global best (non-hard-reject only)
                if not hard_reject and score > global_best_score:
                    global_best_score = score
                    global_best = entry
                    global_best_world = world_name

            if map_best:
                top_candidates_summary.append({
                    "world": world_name,
                    "score": map_best_score,
                    "hard_reject": map_best.get("hard_reject"),
                    "candidate_topology_type": map_best.get("candidate_topology_type"),
                })

        # If all candidates were hard-rejected, use the best-scoring one anyway
        if global_best is None and fallback_best is not None:
            global_best = fallback_best
            global_best_score = fallback_best_score
            global_best_world = fallback_best_world

        if global_best is None:
            return {
                "status": "no_candidates",
                "world_name": None,
                "candidate_summary": {"top_candidates": top_candidates_summary},
                "candidate_debug": {"top_candidates": top_candidates_summary,
                                    "blacklist_locations": blacklist_locations or []},
                "reason": "No candidates found across all cached maps.",
            }

        used_rejected = global_best.get("hard_reject", False)
        candidate_feature_keys = (
            "same_road_lane_count", "nearby_road_count", "nearby_lane_count",
            "heading_cluster_count", "estimated_junction_degree",
            "candidate_topology_type", "junction_waypoint_ratio", "is_junction",
        )
        return {
            "status": "matched",
            "world_name": global_best_world,
            "best_match": {
                "search_strategy": "cache",
                "spawn_point_index": None,
                "location": global_best["location"],
                "yaw": global_best["yaw"],
                "coarse_score": global_best_score,
                "refined_score": global_best_score,
                "score_details": global_best["score_details"],
                "candidate_features": {
                    k: global_best[k] for k in candidate_feature_keys if k in global_best
                },
                "candidate_lane": global_best["candidate_lane"],
                "layout_penalties": {},
                "used_rejected_candidate": used_rejected,
                "reject_reason": global_best.get("reject_reason") if used_rejected else None,
            },
            "projected_layout": {},
            "candidate_summary": {"top_candidates": top_candidates_summary},
            "candidate_debug": {"top_candidates": top_candidates_summary,
                                "blacklist_locations": blacklist_locations or []},
            "reason": (
                "No CARLA candidates satisfied topology hard constraints across all cached maps; "
                "continuing with the highest-scoring rejected candidate."
            ) if used_rejected else None,
        }

    def _match_to_carla(
        self,
        scene_features: Dict[str, Any],
        blacklist_locations: Optional[List[Dict[str, float]]] = None,
        topology_only: bool = False,
    ) -> Dict[str, Any]:
        if (
            self.topology_cache_dir
            and os.path.isdir(self.topology_cache_dir)
        ):
            import glob
            if glob.glob(os.path.join(self.topology_cache_dir, "*.json")):
                return self._match_from_cache(scene_features, blacklist_locations)

        try:
            import carla
        except ImportError as exc:
            return {
                "status": "unavailable",
                "world_name": None,
                "reason": f"CARLA Python API is not available: {exc}",
            }

        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(self.timeout)
            world = (
                client.load_world(self.load_world_name)
                if self.load_world_name
                else client.get_world()
            )
            world_map = world.get_map()
        except Exception as exc:
            return {
                "status": "unavailable",
                "world_name": None,
                "reason": f"Failed to connect to CARLA: {exc}",
            }

        world_name = world_map.name
        spawn_points = world_map.get_spawn_points()
        coarse_candidates = self._score_carla_candidates(
            world_map,
            spawn_points,
            scene_features,
            blacklist_locations=blacklist_locations,
        )
        candidate_debug = {
            "blacklist_locations": blacklist_locations or [],
            "top_candidates": [
                self._summarize_candidate(candidate)
                for candidate in coarse_candidates[: min(50, len(coarse_candidates))]
            ],
        }
        candidate_summary = {
            "coarse_candidate_count": len(coarse_candidates),
            "evaluated_candidate_count": 0,
            "rejected_candidate_count": 0,
            "top_candidates": [
                self._summarize_candidate(candidate)
                for candidate in coarse_candidates[: min(5, len(coarse_candidates))]
            ],
        }

        if not coarse_candidates:
            return {
                "status": "no_candidates",
                "world_name": world_name,
                "candidate_summary": candidate_summary,
                "candidate_debug": candidate_debug,
                "reason": "No CARLA candidates were scored in the current world.",
            }

        if topology_only:
            accepted_candidates = [
                candidate for candidate in coarse_candidates if not candidate.get("hard_reject")
            ]
            best_candidate = None
            reason = None
            if accepted_candidates:
                best_candidate = accepted_candidates[0]
            elif coarse_candidates:
                fallback_candidates = [
                    candidate
                    for candidate in coarse_candidates
                    if not (candidate.get("score_details") or {}).get("blacklisted")
                ] or coarse_candidates
                best_candidate = max(
                    fallback_candidates,
                    key=self._candidate_sort_key,
                )
                reason = (
                    "No CARLA candidates satisfied topology hard constraints; "
                    "continuing with the highest-scoring rejected candidate."
                )
            if best_candidate is None:
                return {
                    "status": "no_valid_candidates",
                    "world_name": world_name,
                    "candidate_summary": candidate_summary,
                    "candidate_debug": candidate_debug,
                    "reason": "No CARLA candidates satisfied topology hard constraints.",
                }
            best_match = self._build_topology_match_record(best_candidate)
            if best_candidate.get("hard_reject"):
                best_match["used_rejected_candidate"] = True
                best_match["reject_reason"] = best_candidate.get("reject_reason")
            return {
                "status": "matched",
                "world_name": world_name,
                "candidate_summary": candidate_summary,
                "candidate_debug": candidate_debug,
                "best_match": best_match,
                "projected_layout": {},
                "reason": reason,
            }

        refined_candidates = []
        for candidate in coarse_candidates[: self.coarse_top_k]:
            candidate_summary["evaluated_candidate_count"] += 1
            refined = self._refine_candidate_layout(
                carla, world_map, scene_features, candidate
            )
            if not refined["accepted"]:
                candidate_summary["rejected_candidate_count"] += 1
                continue
            refined_candidates.append(refined)

        if not refined_candidates:
            return {
                "status": "no_valid_candidates",
                "world_name": world_name,
                "candidate_summary": candidate_summary,
                "candidate_debug": candidate_debug,
                "reason": "No CARLA candidates survived the layout refinement step.",
            }

        best_candidate = max(
            refined_candidates, key=lambda payload: payload["refined_score"]
        )
        quality_failure = self._match_quality_failure(best_candidate, scene_features)
        if quality_failure:
            return {
                "status": "low_quality_match",
                "world_name": world_name,
                "candidate_summary": candidate_summary,
                "candidate_debug": candidate_debug,
                "best_match": best_candidate["match_record"],
                "projected_layout": best_candidate["projected_layout"],
                "reason": quality_failure,
            }
        return {
            "status": "matched",
            "world_name": world_name,
            "candidate_summary": candidate_summary,
            "candidate_debug": candidate_debug,
            "best_match": best_candidate["match_record"],
            "projected_layout": best_candidate["projected_layout"],
            "reason": None,
        }

    @staticmethod
    def _build_topology_match_record(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "search_strategy": candidate.get("search_strategy"),
            "spawn_point_index": candidate.get("spawn_point_index"),
            "location": candidate.get("location"),
            "yaw": candidate.get("yaw"),
            "coarse_score": candidate.get("score"),
            "refined_score": candidate.get("score"),
            "score_details": candidate.get("score_details", {}),
            "candidate_features": {
                "same_road_lane_count": candidate.get("same_road_lane_count"),
                "nearby_road_count": candidate.get("nearby_road_count"),
                "nearby_lane_count": candidate.get("nearby_lane_count"),
                "heading_cluster_count": candidate.get("heading_cluster_count"),
                "estimated_junction_degree": candidate.get("estimated_junction_degree"),
                "junction_waypoint_ratio": candidate.get("junction_waypoint_ratio"),
                "is_junction": candidate.get("is_junction"),
                "candidate_topology_type": candidate.get("candidate_topology_type"),
            },
            "candidate_lane": candidate.get("candidate_lane"),
            "layout_penalties": {},
        }

    def _match_quality_failure(
        self,
        refined_candidate: Dict[str, Any],
        scene_features: Dict[str, Any],
    ) -> Optional[str]:
        match_record = refined_candidate.get("match_record") or {}
        refined_score = _safe_float(match_record.get("refined_score"))
        if refined_score < self.min_refined_score:
            return (
                f"Best match refined_score {refined_score:.3f} is below "
                f"threshold {self.min_refined_score:.3f}."
            )
        penalties = match_record.get("layout_penalties") or {}
        average_snap = _safe_float(penalties.get("average_vehicle_snap_distance"))
        if average_snap > self.max_average_snap_distance:
            return (
                f"Best match average vehicle snap distance {average_snap:.2f}m "
                f"exceeds threshold {self.max_average_snap_distance:.2f}m."
            )
        road_hints = scene_features.get("road_hints") or {}
        candidate_features = match_record.get("candidate_features") or {}
        if road_hints.get("straight_road") and not road_hints.get("near_junction"):
            heading_clusters = int(candidate_features.get("heading_cluster_count") or 1)
            nearby_roads = int(candidate_features.get("nearby_road_count") or 1)
            if heading_clusters > 2 or nearby_roads > 2:
                return (
                    "Straight non-junction scene matched to a complex multi-direction "
                    f"region (heading_cluster_count={heading_clusters}, "
                    f"nearby_road_count={nearby_roads})."
                )
        return None

    def _score_carla_candidates(
        self,
        world_map: Any,
        spawn_points: List[Any],
        scene_features: Dict[str, Any],
        blacklist_locations: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        if self._is_large_map(world_map, spawn_points):
            return self._score_large_map_candidates(
                world_map,
                spawn_points,
                scene_features,
                blacklist_locations=blacklist_locations,
            )

        sampled_waypoints = world_map.generate_waypoints(self.sample_step)
        scored_candidates = []
        for waypoint in sampled_waypoints:
            candidate = self._extract_candidate_features(sampled_waypoints, waypoint)
            score, details = self._score_candidate_features(candidate, scene_features)
            if self._is_blacklisted(waypoint.transform.location, blacklist_locations):
                details["hard_reject"] = True
                details["blacklisted"] = True
                details["reject_reason"] = "Candidate is inside a blacklisted rematch region."
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "hard_reject": bool(details.get("hard_reject")),
                    "reject_reason": details.get("reject_reason"),
                    "search_strategy": "full_waypoints",
                    "location": {
                        "x": waypoint.transform.location.x,
                        "y": waypoint.transform.location.y,
                        "z": waypoint.transform.location.z,
                    },
                    "yaw": waypoint.transform.rotation.yaw,
                    "candidate_lane": self._waypoint_to_lane_dict(waypoint),
                    **candidate,
                }
            )

        scored_candidates.sort(
            key=self._candidate_sort_key,
            reverse=True,
        )
        return scored_candidates[: self._candidate_return_limit(scene_features)]

    def _is_large_map(self, world_map: Any, spawn_points: Optional[List[Any]] = None) -> bool:
        map_name = getattr(world_map, "name", "") or ""
        normalized_name = map_name.lower()
        if any(keyword.lower() in normalized_name for keyword in self.large_map_name_keywords):
            return True

        if spawn_points is None:
            try:
                spawn_points = world_map.get_spawn_points()
            except Exception:
                spawn_points = []
        return len(spawn_points) >= self.large_map_spawn_threshold

    def _score_large_map_candidates(
        self,
        world_map: Any,
        spawn_points: List[Any],
        scene_features: Dict[str, Any],
        blacklist_locations: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        if not spawn_points:
            return []

        scored_candidates = []
        stride = max(
            1,
            math.ceil(len(spawn_points) / max(1, self.large_map_candidate_limit)),
        )
        for spawn_index in range(0, len(spawn_points), stride):
            spawn_transform = spawn_points[spawn_index]
            waypoint = world_map.get_waypoint(
                spawn_transform.location, project_to_road=True
            )
            if waypoint is None:
                continue

            candidate = self._extract_local_candidate_features(world_map, waypoint)
            score, details = self._score_candidate_features(candidate, scene_features)
            if self._is_blacklisted(waypoint.transform.location, blacklist_locations):
                details["hard_reject"] = True
                details["blacklisted"] = True
                details["reject_reason"] = "Candidate is inside a blacklisted rematch region."
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "hard_reject": bool(details.get("hard_reject")),
                    "reject_reason": details.get("reject_reason"),
                    "search_strategy": "spawn_points",
                    "spawn_point_index": spawn_index,
                    "location": {
                        "x": waypoint.transform.location.x,
                        "y": waypoint.transform.location.y,
                        "z": waypoint.transform.location.z,
                    },
                    "yaw": waypoint.transform.rotation.yaw,
                    "candidate_lane": self._waypoint_to_lane_dict(waypoint),
                    **candidate,
                }
            )

        scored_candidates.sort(
            key=self._candidate_sort_key,
            reverse=True,
        )
        return scored_candidates[: self._candidate_return_limit(scene_features)]

    def _candidate_return_limit(self, scene_features: Dict[str, Any]) -> int:
        if scene_features.get("road_topology_signature"):
            return max(self.coarse_top_k, 120)
        return max(self.coarse_top_k, 20)

    @staticmethod
    def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
        details = candidate.get("score_details") or {}
        return (
            not candidate.get("hard_reject"),
            not details.get("blacklisted"),
            float(details.get("fallback_total_score", details.get("uncapped_total_score", 0.0)) or 0.0),
            float(candidate.get("score", 0.0) or 0.0),
        )

    def _is_blacklisted(
        self,
        location: Any,
        blacklist_locations: Optional[List[Dict[str, float]]],
    ) -> bool:
        for item in blacklist_locations or []:
            if not isinstance(item, dict):
                continue
            distance = _distance_2d(
                (float(location.x), float(location.y)),
                (_safe_float(item.get("x")), _safe_float(item.get("y"))),
            )
            radius = _safe_float(item.get("radius"), self.blacklist_radius_m)
            if distance <= radius:
                return True
        return False

    @staticmethod
    def _waypoint_to_lane_dict(waypoint: Any) -> Dict[str, Any]:
        try:
            next_waypoints = waypoint.next(20.0)
            end_wp = next_waypoints[0] if next_waypoints else waypoint
        except Exception:
            end_wp = waypoint
        start = waypoint.transform
        end = end_wp.transform
        return {
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
            "start": {
                "x": float(start.location.x),
                "y": float(start.location.y),
                "z": float(start.location.z),
                "yaw": float(start.rotation.yaw),
                "is_junction": bool(waypoint.is_junction),
            },
            "end": {
                "x": float(end.location.x),
                "y": float(end.location.y),
                "z": float(end.location.z),
                "yaw": float(end.rotation.yaw),
                "is_junction": bool(getattr(end_wp, "is_junction", False)),
            },
        }

    def _extract_candidate_features(
        self,
        sampled_waypoints: List[Any],
        center_waypoint: Any,
        crosswalk_locations: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        center = center_waypoint.transform.location
        nearby = [
            waypoint
            for waypoint in sampled_waypoints
            if waypoint.transform.location.distance(center) <= self.search_radius
        ]
        if not nearby:
            nearby = [center_waypoint]

        road_ids = sorted({waypoint.road_id for waypoint in nearby})
        lane_keys = sorted({(waypoint.road_id, waypoint.lane_id) for waypoint in nearby})
        headings = [waypoint.transform.rotation.yaw for waypoint in nearby]
        junction_ratio = sum(1 for waypoint in nearby if waypoint.is_junction) / len(nearby)
        heading_cluster_count = len(_cluster_headings(headings)) or 1
        estimated_junction_degree = max(len(road_ids), heading_cluster_count)
        candidate_topology_type = self._candidate_topology_type(
            junction_ratio=junction_ratio,
            is_junction=bool(center_waypoint.is_junction),
            heading_cluster_count=heading_cluster_count,
            nearby_road_count=len(road_ids),
            estimated_junction_degree=estimated_junction_degree,
        )
        left_parking, right_parking = self._detect_parking_sides(center_waypoint)
        has_crosswalk_nearby = self._detect_crosswalk_nearby(center, crosswalk_locations)
        return {
            "yaw": center_waypoint.transform.rotation.yaw,
            "is_junction": center_waypoint.is_junction,
            "junction_waypoint_ratio": junction_ratio,
            "nearby_road_count": len(road_ids),
            "nearby_lane_count": len(lane_keys),
            "same_road_lane_count": self._estimate_same_road_lane_count(
                nearby, center_waypoint.road_id
            ),
            "heading_cluster_count": heading_cluster_count,
            "estimated_junction_degree": estimated_junction_degree,
            "candidate_topology_type": candidate_topology_type,
            "left_parking_lane_present": left_parking,
            "right_parking_lane_present": right_parking,
            "has_crosswalk_nearby": has_crosswalk_nearby,
        }

    def _extract_local_candidate_features(
        self,
        world_map: Any,
        center_waypoint: Any,
        crosswalk_locations: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        nearby = self._collect_local_waypoints(
            center_waypoint, max(self.sample_step, self.large_map_step)
        )
        if not nearby:
            nearby = [center_waypoint]

        road_ids = sorted({waypoint.road_id for waypoint in nearby})
        lane_keys = sorted({(waypoint.road_id, waypoint.lane_id) for waypoint in nearby})
        headings = [waypoint.transform.rotation.yaw for waypoint in nearby]
        junction_ratio = sum(1 for waypoint in nearby if waypoint.is_junction) / len(nearby)
        heading_cluster_count = len(_cluster_headings(headings)) or 1
        estimated_junction_degree = max(len(road_ids), heading_cluster_count)
        candidate_topology_type = self._candidate_topology_type(
            junction_ratio=junction_ratio,
            is_junction=bool(center_waypoint.is_junction),
            heading_cluster_count=heading_cluster_count,
            nearby_road_count=len(road_ids),
            estimated_junction_degree=estimated_junction_degree,
        )
        center = center_waypoint.transform.location
        left_parking, right_parking = self._detect_parking_sides(center_waypoint)
        has_crosswalk_nearby = self._detect_crosswalk_nearby(center, crosswalk_locations)
        return {
            "yaw": center_waypoint.transform.rotation.yaw,
            "is_junction": center_waypoint.is_junction,
            "junction_waypoint_ratio": junction_ratio,
            "nearby_road_count": len(road_ids),
            "nearby_lane_count": len(lane_keys),
            "same_road_lane_count": self._estimate_same_road_lane_count(
                nearby, center_waypoint.road_id
            ),
            "heading_cluster_count": heading_cluster_count,
            "estimated_junction_degree": estimated_junction_degree,
            "candidate_topology_type": candidate_topology_type,
            "left_parking_lane_present": left_parking,
            "right_parking_lane_present": right_parking,
            "has_crosswalk_nearby": has_crosswalk_nearby,
        }

    @staticmethod
    def _detect_parking_sides(center_waypoint: Any) -> Tuple[bool, bool]:
        try:
            left_lane = center_waypoint.get_left_lane()
            left_parking = left_lane is not None and "Parking" in str(left_lane.lane_type)
        except Exception:
            left_parking = False
        try:
            right_lane = center_waypoint.get_right_lane()
            right_parking = right_lane is not None and "Parking" in str(right_lane.lane_type)
        except Exception:
            right_parking = False
        return left_parking, right_parking

    @staticmethod
    def _detect_crosswalk_nearby(
        center_location: Any,
        crosswalk_locations: Optional[List[Any]],
        radius: float = 30.0,
    ) -> bool:
        if not crosswalk_locations:
            return False
        try:
            return any(center_location.distance(cw) < radius for cw in crosswalk_locations)
        except Exception:
            return False

    @staticmethod
    def _candidate_topology_type(
        junction_ratio: float,
        is_junction: bool,
        heading_cluster_count: int,
        nearby_road_count: int,
        estimated_junction_degree: int,
    ) -> str:
        near_junction = is_junction or junction_ratio >= 0.15
        if near_junction:
            if estimated_junction_degree >= 5:
                return "multi_branch"
            if estimated_junction_degree >= 4 or heading_cluster_count >= 4:
                return "cross_intersection"
            return "t_junction"
        if heading_cluster_count <= 2 and nearby_road_count <= 2:
            return "straight_two_way"
        if heading_cluster_count <= 3:
            return "curve"
        return "multi_branch"

    def _collect_local_waypoints(self, center_waypoint: Any, step_size: float) -> List[Any]:
        max_samples = 256
        nearby: List[Any] = []
        visited = set()
        queue = deque([(center_waypoint, 0.0)])

        while queue and len(nearby) < max_samples:
            waypoint, traversed = queue.popleft()
            if waypoint is None:
                continue

            waypoint_s = round(getattr(waypoint, "s", 0.0), 1)
            key = (waypoint.road_id, waypoint.lane_id, waypoint_s)
            if key in visited:
                continue
            visited.add(key)
            nearby.append(waypoint)

            if traversed >= self.search_radius:
                continue

            neighbor_candidates = []
            try:
                neighbor_candidates.extend(waypoint.next(step_size))
            except Exception:
                pass
            try:
                neighbor_candidates.extend(waypoint.previous(step_size))
            except Exception:
                pass

            for lateral_waypoint in (waypoint.get_left_lane(), waypoint.get_right_lane()):
                if lateral_waypoint is not None:
                    neighbor_candidates.append(lateral_waypoint)

            for neighbor in neighbor_candidates:
                additional_distance = step_size
                if neighbor.lane_id != waypoint.lane_id:
                    additional_distance = max(2.0, step_size * 0.5)
                queue.append((neighbor, traversed + additional_distance))

        return nearby

    @staticmethod
    def _estimate_same_road_lane_count(
        nearby_waypoints: List[Any], road_id: int
    ) -> int:
        lane_ids = {
            waypoint.lane_id
            for waypoint in nearby_waypoints
            if waypoint.road_id == road_id
            and waypoint.lane_id != 0
            and SceneMapMatcher._is_driving_waypoint(waypoint)
        }
        return len(lane_ids) or 1

    @staticmethod
    def _is_driving_waypoint(waypoint: Any) -> bool:
        lane_type = getattr(waypoint, "lane_type", None)
        if lane_type is None:
            return True
        return "driving" in str(lane_type).lower()

    def _score_candidate_features(
        self, candidate: Dict[str, Any], scene_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        if scene_features.get("road_topology_signature"):
            return self._score_topology_candidate_features(candidate, scene_features)

        layout_summary = scene_features["layout_summary"]
        semantic_hints = scene_features["semantic_hints"]
        road_hints = scene_features.get("road_hints") or {}
        target_lane_count = layout_summary.get("anchor_lane_count")
        target_branch_count = layout_summary.get("target_branch_count", 1)
        target_heading = layout_summary.get("anchor_heading")
        expected_near_junction = layout_summary.get("anchor_near_junction", False)
        candidate_near_junction = (
            candidate["is_junction"] or candidate["junction_waypoint_ratio"] >= 0.2
        )

        if target_lane_count:
            lane_score = max(
                0.0,
                1.0
                - abs(candidate["same_road_lane_count"] - target_lane_count) / 6.0,
            )
        else:
            lane_score = 0.5

        branch_score = max(
            0.0,
            1.0
            - abs(candidate["estimated_junction_degree"] - target_branch_count) / 4.0,
        )
        proximity_score = 1.0 if candidate_near_junction == expected_near_junction else 0.4

        if target_heading is None:
            heading_score = 0.5
        else:
            heading_score = max(
                0.0,
                1.0 - _angle_difference_deg(candidate["yaw"], target_heading) / 90.0,
            )

        if road_hints.get("straight_road") and not road_hints.get("near_junction"):
            road_diversity = max(
                0.0,
                1.0
                - max(0, candidate["nearby_road_count"] - 1) / 3.0
                - max(0, candidate["heading_cluster_count"] - 1) / 3.0,
            )
        else:
            road_diversity = min(candidate["nearby_road_count"] / 4.0, 1.0)
        text_hint_score = self._build_text_hint_score(
            semantic_hints, candidate_near_junction, road_diversity
        )

        details = {
            "lane_count_match": lane_score,
            "junction_degree_match": branch_score,
            "junction_presence_or_proximity_match": proximity_score,
            "anchor_heading_match": heading_score,
            "road_diversity": road_diversity,
            "text_semantic_hints": text_hint_score,
        }
        total_score = (
            0.30 * lane_score
            + 0.25 * branch_score
            + 0.20 * proximity_score
            + 0.10 * heading_score
            + 0.10 * road_diversity
            + 0.05 * text_hint_score
        )
        return total_score, details

    def _score_topology_candidate_features(
        self, candidate: Dict[str, Any], scene_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        signature = scene_features.get("road_topology_signature") or {}
        target_topology = str(signature.get("topology_type") or "unknown")
        candidate_topology = str(candidate.get("candidate_topology_type") or "unknown")
        target_branch_count = int(signature.get("target_branch_count") or 1)
        candidate_branch_count = int(candidate.get("estimated_junction_degree") or 1)
        target_driving_lanes = int(signature.get("driving_lane_count") or 2)
        candidate_driving_lanes = int(candidate.get("same_road_lane_count") or 1)
        junction_ratio = float(candidate.get("junction_waypoint_ratio") or 0.0)
        heading_clusters = int(candidate.get("heading_cluster_count") or 1)
        nearby_roads = int(candidate.get("nearby_road_count") or 1)

        hard_reject = False
        reject_reason = None
        if target_topology == "straight_two_way":
            if heading_clusters > 2 or nearby_roads > 2 or junction_ratio > 0.12:
                hard_reject = True
                reject_reason = (
                    "Straight two-way target rejects complex or junction-like candidate "
                    f"(heading_cluster_count={heading_clusters}, nearby_road_count={nearby_roads}, "
                    f"junction_waypoint_ratio={junction_ratio:.2f})."
                )
            if candidate_driving_lanes > target_driving_lanes + 1:
                hard_reject = True
                reject_reason = (
                    "Straight two-way target rejects overly wide driving-lane candidate "
                    f"(target_driving_lane_count={target_driving_lanes}, "
                    f"candidate_driving_lane_count={candidate_driving_lanes})."
                )
        elif target_topology in {"t_junction", "cross_intersection", "multi_branch"}:
            if junction_ratio < 0.08 and not candidate.get("is_junction"):
                hard_reject = True
                reject_reason = "Junction target rejects non-junction candidate."

        topology_score = 0.0
        if candidate_topology == target_topology:
            topology_score = 1.0
        elif target_topology == "straight_two_way" and candidate_topology == "curve":
            topology_score = 0.45
        elif target_topology in {"t_junction", "cross_intersection"} and candidate_topology in {
            "t_junction",
            "cross_intersection",
        }:
            topology_score = max(
                0.0,
                1.0 - abs(candidate_branch_count - target_branch_count) / 3.0,
            )
        elif target_topology == "unknown":
            topology_score = 0.5
        else:
            topology_score = 0.15

        lane_delta = abs(candidate_driving_lanes - target_driving_lanes)
        lane_score = max(0.0, 1.0 - lane_delta / 2.0)
        branch_score = max(
            0.0,
            1.0 - abs(candidate_branch_count - target_branch_count) / 4.0,
        )
        topology_score = 0.60 * topology_score + 0.25 * branch_score + 0.15 * lane_score

        side_context_score = self._score_side_context(candidate, signature)
        auxiliary_score = self._score_auxiliary_context(candidate, signature)
        total_score = (
            self.topology_weight * topology_score
            + self.side_context_weight * side_context_score
            + self.auxiliary_weight * auxiliary_score
        )
        uncapped_total_score = total_score
        fallback_total_score = self._topology_fallback_score(
            uncapped_total_score=uncapped_total_score,
            target_topology=target_topology,
            candidate_topology=candidate_topology,
            target_driving_lanes=target_driving_lanes,
            candidate_driving_lanes=candidate_driving_lanes,
            heading_clusters=heading_clusters,
            nearby_roads=nearby_roads,
            junction_ratio=junction_ratio,
        )
        if hard_reject:
            total_score = min(total_score, 0.05)

        return total_score, {
            "uncapped_total_score": uncapped_total_score,
            "fallback_total_score": fallback_total_score,
            "topology_score": topology_score,
            "side_context_score": side_context_score,
            "auxiliary_element_score": auxiliary_score,
            "target_topology_type": target_topology,
            "candidate_topology_type": candidate_topology,
            "target_driving_lane_count": target_driving_lanes,
            "candidate_driving_lane_count": candidate_driving_lanes,
            "target_branch_count": target_branch_count,
            "candidate_branch_count": candidate_branch_count,
            "hard_reject": hard_reject,
            "blacklisted": False,
            "reject_reason": reject_reason,
        }

    @staticmethod
    def _topology_fallback_score(
        uncapped_total_score: float,
        target_topology: str,
        candidate_topology: str,
        target_driving_lanes: int,
        candidate_driving_lanes: int,
        heading_clusters: int,
        nearby_roads: int,
        junction_ratio: float,
    ) -> float:
        score = float(uncapped_total_score or 0.0)
        lane_delta = abs(candidate_driving_lanes - target_driving_lanes)
        if candidate_driving_lanes > target_driving_lanes:
            score *= max(0.10, 1.0 - 0.35 * lane_delta)
        elif lane_delta:
            score *= max(0.35, 1.0 - 0.20 * lane_delta)

        if target_topology == "straight_two_way":
            if candidate_topology != target_topology:
                score *= 0.80
            complexity_delta = max(0, heading_clusters - 2) + max(0, nearby_roads - 2)
            if junction_ratio > 0.12:
                complexity_delta += int(math.ceil(junction_ratio * 4.0))
            score *= max(0.45, 1.0 - 0.04 * complexity_delta)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_side_context(candidate: Dict[str, Any], signature: Dict[str, Any]) -> float:
        score = 0.5

        left_expected = bool(signature.get("left_parking_presence"))
        left_actual = candidate.get("left_parking_lane_present")
        if left_actual is not None:
            if left_expected and left_actual:
                score += 0.12

        right_expected = bool(signature.get("right_parking_presence"))
        right_actual = candidate.get("right_parking_lane_present")
        if right_actual is not None:
            if right_expected and right_actual:
                score += 0.12

        side_context = signature.get("side_context") or {}
        if side_context.get("tree_lined"):
            score += 0.05
        if side_context.get("left_continuous_buildings") or side_context.get(
            "right_continuous_buildings"
        ):
            score += 0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_auxiliary_context(candidate: Dict[str, Any], signature: Dict[str, Any]) -> float:
        score = 0.5
        target_topology = str(signature.get("topology_type") or "unknown")
        is_straight = target_topology in {"straight_two_way", "straight_road"}
        candidate_near_junction = bool(candidate.get("is_junction")) or float(
            candidate.get("junction_waypoint_ratio") or 0.0
        ) >= 0.15

        if signature.get("has_crosswalk"):
            candidate_has_cw = candidate.get("has_crosswalk_nearby")
            if candidate_has_cw is not None:
                # Use real candidate data: crosswalk near a midblock is good; near a junction is ok
                score += 0.15 if candidate_has_cw else -0.05
            elif is_straight:
                # midblock scene: prefer straight candidates (no junction) for crosswalk match
                score += 0.12 if not candidate_near_junction else 0.02
            else:
                # intersection scene: original logic — junction candidates more likely have crosswalks
                score += 0.15 if candidate_near_junction else 0.05

        if signature.get("has_traffic_light"):
            # traffic lights rarely appear on midblock straight segments
            if not is_straight:
                score += 0.15 if candidate_near_junction else 0.0

        if signature.get("has_traffic_sign"):
            score += 0.05
        if signature.get("has_center_median"):
            score += 0.05 if int(candidate.get("nearby_lane_count") or 1) >= 4 else -0.05
        return max(0.0, min(1.0, score))

    @staticmethod
    def _build_text_hint_score(
        semantic_hints: Dict[str, bool],
        candidate_near_junction: bool,
        road_diversity: float,
    ) -> float:
        sub_scores = []
        if semantic_hints.get("mentions_crosswalk") or semantic_hints.get("mentions_pedestrian"):
            sub_scores.append(1.0 if candidate_near_junction else 0.4)
        if semantic_hints.get("mentions_parked_vehicle"):
            sub_scores.append(0.8 if road_diversity >= 0.25 else 0.5)
        if semantic_hints.get("mentions_barrier"):
            sub_scores.append(0.8 if road_diversity >= 0.25 else 0.5)
        return sum(sub_scores) / len(sub_scores) if sub_scores else 0.5

    def _refine_candidate_layout(
        self,
        carla: Any,
        world_map: Any,
        scene_features: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        projected_layout = self._project_layout_to_candidate(
            carla, world_map, scene_features, candidate
        )
        anchor = scene_features["layout_summary"]["anchor_entity"]
        anchor_name = anchor["name"]

        vehicle_snap_distances = []
        over_limit_count = 0
        ego_snap_distance = None
        for entity_name, payload in projected_layout.items():
            if payload["entity_type"] not in VEHICLE_TYPES:
                continue
            snap_distance = _safe_float(payload.get("vehicle_snap_distance"))
            vehicle_snap_distances.append(snap_distance)
            if snap_distance > 8.0:
                over_limit_count += 1
            if entity_name == anchor_name:
                ego_snap_distance = snap_distance

        if ego_snap_distance is not None and ego_snap_distance > 8.0:
            return {
                "accepted": False,
                "reason": "Anchor vehicle snap distance exceeded 8 meters.",
            }

        if vehicle_snap_distances and over_limit_count > len(vehicle_snap_distances) / 2.0:
            return {
                "accepted": False,
                "reason": "More than half of vehicle snap distances exceeded 8 meters.",
            }

        average_snap = (
            sum(vehicle_snap_distances) / len(vehicle_snap_distances)
            if vehicle_snap_distances
            else 0.0
        )
        high_snap_penalty = 0.10 * sum(
            1 for distance in vehicle_snap_distances if distance > 6.0
        )
        average_snap_penalty = min(average_snap * 0.05, 0.25)
        refined_score = candidate["score"] - average_snap_penalty - high_snap_penalty
        match_record = {
            "search_strategy": candidate.get("search_strategy"),
            "location": candidate["location"],
            "yaw": candidate["yaw"],
            "anchor": {
                "projected": {
                    "x": anchor["location"][0],
                    "y": anchor["location"][1],
                    "z": anchor["location"][2],
                },
                "world": candidate["location"],
                "dx": candidate["location"]["x"] - anchor["location"][0],
                "dy": candidate["location"]["y"] - anchor["location"][1],
            },
            "coarse_score": candidate["score"],
            "refined_score": refined_score,
            "score_details": candidate.get("score_details", {}),
            "candidate_features": {
                "same_road_lane_count": candidate.get("same_road_lane_count"),
                "nearby_road_count": candidate.get("nearby_road_count"),
                "nearby_lane_count": candidate.get("nearby_lane_count"),
                "heading_cluster_count": candidate.get("heading_cluster_count"),
                "estimated_junction_degree": candidate.get("estimated_junction_degree"),
                "junction_waypoint_ratio": candidate.get("junction_waypoint_ratio"),
                "is_junction": candidate.get("is_junction"),
            },
            "layout_penalties": {
                "average_vehicle_snap_distance": average_snap,
                "average_vehicle_snap_penalty": average_snap_penalty,
                "high_snap_vehicle_count": sum(
                    1 for distance in vehicle_snap_distances if distance > 6.0
                ),
                "high_snap_penalty": high_snap_penalty,
                "ego_vehicle_snap_distance": ego_snap_distance,
            },
        }
        return {
            "accepted": True,
            "reason": None,
            "refined_score": refined_score,
            "match_record": match_record,
            "projected_layout": projected_layout,
        }

    def _project_layout_to_candidate(
        self,
        carla: Any,
        world_map: Any,
        scene_features: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        anchor = scene_features["layout_summary"]["anchor_entity"]
        relative_layout = scene_features["layout_summary"]["relative_layout"]
        entity_lookup = {
            entity["name"]: entity for entity in scene_features.get("entities", [])
        }

        anchor_yaw = _safe_float(anchor["rotation"][1])
        candidate_yaw = _safe_float(candidate["yaw"])
        yaw_delta_deg = candidate_yaw - anchor_yaw
        yaw_delta_rad = math.radians(yaw_delta_deg)
        cos_delta = math.cos(yaw_delta_rad)
        sin_delta = math.sin(yaw_delta_rad)

        projected_entities = {}
        for entity_name, payload in relative_layout.items():
            relative = payload["relative_to_anchor"]
            rotated_dx = relative["dx"] * cos_delta - relative["dy"] * sin_delta
            rotated_dy = relative["dx"] * sin_delta + relative["dy"] * cos_delta
            raw_location = {
                "x": candidate["location"]["x"] + rotated_dx,
                "y": candidate["location"]["y"] + rotated_dy,
                "z": _safe_float(payload["location"][2]),
            }

            entity_info = entity_lookup.get(entity_name, {})
            entity_rotation = payload.get("rotation") or entity_info.get("rotation") or [
                0.0,
                anchor_yaw,
                0.0,
            ]
            raw_rotation = {
                "pitch": _safe_float(entity_rotation[0]),
                "yaw": _normalize_angle_deg(_safe_float(entity_rotation[1]) + yaw_delta_deg),
                "roll": _safe_float(entity_rotation[2]),
            }

            projected_location = dict(raw_location)
            projected_rotation = dict(raw_rotation)
            snap_distance = None
            if payload["entity_type"] in VEHICLE_TYPES:
                waypoint = world_map.get_waypoint(
                    carla.Location(
                        x=raw_location["x"],
                        y=raw_location["y"],
                        z=raw_location["z"],
                    ),
                    project_to_road=True,
                )
                if waypoint is None:
                    snap_distance = 9999.0
                else:
                    snapped_location = waypoint.transform.location
                    projected_location = {
                        "x": snapped_location.x,
                        "y": snapped_location.y,
                        "z": snapped_location.z,
                    }
                    projected_rotation = {
                        "pitch": raw_rotation["pitch"],
                        "yaw": waypoint.transform.rotation.yaw,
                        "roll": raw_rotation["roll"],
                    }
                    snap_distance = _distance_2d(
                        (raw_location["x"], raw_location["y"]),
                        (projected_location["x"], projected_location["y"]),
                    )

            projected_entities[entity_name] = {
                "entity_type": payload["entity_type"],
                "role": payload.get("role") or entity_info.get("role", "other"),
                "raw_location": raw_location,
                "projected_location": projected_location,
                "raw_rotation": raw_rotation,
                "projected_rotation": projected_rotation,
                "vehicle_snap_distance": snap_distance,
            }
        return projected_entities

    @staticmethod
    def _summarize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "search_strategy": candidate.get("search_strategy"),
            "location": candidate.get("location"),
            "yaw": candidate.get("yaw"),
            "score": candidate.get("score"),
            "score_details": candidate.get("score_details"),
            "same_road_lane_count": candidate.get("same_road_lane_count"),
            "nearby_road_count": candidate.get("nearby_road_count"),
            "heading_cluster_count": candidate.get("heading_cluster_count"),
            "estimated_junction_degree": candidate.get("estimated_junction_degree"),
            "candidate_topology_type": candidate.get("candidate_topology_type"),
            "hard_reject": candidate.get("hard_reject"),
            "reject_reason": candidate.get("reject_reason"),
        }

    def _rewrite_scene_script_with_match(
        self, original_code: str, report: Dict[str, Any]
    ) -> str:
        start_index, end_index = self._find_spawn_block_bounds(original_code)
        spawn_block = self._build_spawn_block(report)

        if start_index is None:
            rewritten = original_code.rstrip()
            if rewritten:
                rewritten += "\n\n"
            rewritten += spawn_block
            return rewritten

        prefix = original_code[:start_index].rstrip()
        suffix = original_code[end_index:].lstrip("\n") if end_index is not None else ""
        parts = [prefix, spawn_block]
        if suffix:
            parts.append(suffix)
        return "\n\n".join(part for part in parts if part)

    def _build_fallback_matched_script(
        self, original_code: str, status: str, reason: str
    ) -> str:
        sanitized_reason = " ".join(str(reason).split())
        comment_line = f"{SCENE_MATCH_COMMENT_PREFIX}{status}; reason: {sanitized_reason}"
        if original_code.startswith(SCENE_MATCH_COMMENT_PREFIX):
            lines = original_code.splitlines()
            lines[0] = comment_line
            return "\n".join(lines)
        return f"{comment_line}\n{original_code}"

    def _find_spawn_block_bounds(self, code: str) -> Tuple[Optional[int], Optional[int]]:
        lines = code.splitlines(keepends=True)
        activation_seen = False
        start_line = None
        spectator_line = None
        time_sleep_line = None

        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped == SCENE_SPECTATOR_MARKER:
                spectator_line = index
                break
            if SCENE_SPECTATOR_MARKER in stripped and spectator_line is None:
                spectator_line = index
                break

        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# __AUTOSCENARIO_ACTIVATE_HELPERS__"):
                activation_seen = True
                continue
            if not activation_seen:
                continue
            if stripped.startswith("# Spawn into Carla") or stripped.startswith(
                "# The code to spawn"
            ):
                start_line = index
                break
            if (
                ("spawn_vehicle(" in stripped and not stripped.startswith("def "))
                or ("spawn_static_prop(" in stripped and not stripped.startswith("def "))
                or ("spawn_pedestrian(" in stripped and not stripped.startswith("def "))
            ):
                start_line = index
                break

        if start_line is None:
            return None, None

        if spectator_line is None:
            for index in range(start_line + 1, len(lines)):
                if "time.sleep(" in lines[index]:
                    time_sleep_line = index
                    break
        end_line = spectator_line if spectator_line is not None else time_sleep_line
        if end_line is None:
            end_line = len(lines)

        start_index = sum(len(line) for line in lines[:start_line])
        end_index = sum(len(line) for line in lines[:end_line])
        return start_index, end_index

    def _build_spawn_block(self, report: Dict[str, Any]) -> str:
        world_name = report.get("world_name", "unknown")
        best_match = report.get("best_match") or {}
        projected_layout = report.get("projected_layout") or {}
        ordered_entities = report.get("scene_features", {}).get("entities", [])
        entity_names = [
            entity["name"] for entity in ordered_entities if entity["name"] in projected_layout
        ]
        vehicle_names = [
            name
            for name in entity_names
            if projected_layout[name].get("entity_type") in VEHICLE_TYPES
        ]
        pedestrian_names = [
            name
            for name in entity_names
            if projected_layout[name].get("entity_type") == "pedestrian"
        ]
        static_names = [
            name
            for name in entity_names
            if name not in vehicle_names and name not in pedestrian_names
        ]
        color_cycle = [
            "255,0,0",
            "0,255,0",
            "0,0,255",
            "255,255,0",
            "255,165,0",
            "128,0,255",
        ]

        lines = [
            "# Spawn into Carla",
            "# Scene match status: matched",
            f"# Matched world: {world_name}",
            f"# Best candidate coarse score: {_safe_float(best_match.get('coarse_score')):.4f}",
            f"# Best candidate refined score: {_safe_float(best_match.get('refined_score')):.4f}",
            (
                "# Best candidate center: "
                f"({_safe_float((best_match.get('location') or {}).get('x')):.3f}, "
                f"{_safe_float((best_match.get('location') or {}).get('y')):.3f}, "
                f"{_safe_float((best_match.get('location') or {}).get('z')):.3f}), "
                f"yaw={_safe_float(best_match.get('yaw')):.3f}"
            ),
            "",
            "# The code to spawn vehicles",
        ]

        if not vehicle_names:
            lines.append("# No matched vehicles")
        for index, name in enumerate(vehicle_names, start=1):
            payload = projected_layout[name]
            color = color_cycle[(index - 1) % len(color_cycle)]
            lines.append(
                f"spawned_vehicle_{index} = spawn_vehicle("
                f"'{payload['entity_type']}', "
                f"{self._format_location_code(payload['projected_location'])}, "
                f"{self._format_rotation_code(payload['projected_rotation'])}, "
                f"'{color}'"
                f")  # matched from {name}"
            )

        lines.extend(["", "# The code to spawn static objects"])
        if not static_names:
            lines.append("# No matched static objects")
        for index, name in enumerate(static_names, start=1):
            payload = projected_layout[name]
            lines.append(
                f"spawned_static_{index} = spawn_static_prop("
                f"'{payload['entity_type']}', "
                f"{self._format_location_code(payload['projected_location'])}, "
                f"{self._format_rotation_code(payload['projected_rotation'])}"
                f")  # matched from {name}"
            )

        lines.extend(["", "# The code to spawn pedestrians"])
        if not pedestrian_names:
            lines.append("# No matched pedestrians")
        for index, name in enumerate(pedestrian_names, start=1):
            payload = projected_layout[name]
            lines.append(
                f"spawned_pedestrian_{index} = spawn_pedestrian("
                f"{self._format_location_code(payload['projected_location'])}, "
                f"{self._format_rotation_code(payload['projected_rotation'])}"
                f")  # matched from {name}"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_location_code(location: Dict[str, float]) -> str:
        return (
            f"carla.Location(x={_safe_float(location.get('x')):.6f}, "
            f"y={_safe_float(location.get('y')):.6f}, "
            f"z={_safe_float(location.get('z')):.6f})"
        )

    @staticmethod
    def _format_rotation_code(rotation: Dict[str, float]) -> str:
        return (
            f"carla.Rotation(pitch={_safe_float(rotation.get('pitch')):.6f}, "
            f"yaw={_safe_float(rotation.get('yaw')):.6f}, "
            f"roll={_safe_float(rotation.get('roll')):.6f})"
        )
