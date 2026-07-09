import ast
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.map_matcher_v2 import score_candidate_v2, summarize_entry
from tools.utils import extract_text_section, read_file, write_to_file


OBJECT_INFO_PREFIX = "__AUTOSCENARIO_OBJECT_INFO__="
SCENE_SPECTATOR_MARKER = "# __AUTOSCENARIO_FOCUS_SPECTATOR__"
SCENE_MATCH_COMMENT_PREFIX = "# scene_match_status: "
VEHICLE_TYPES = {"bike", "car", "jeep", "motorcycle", "suv", "truck", "van"}
CURVE_LOOKAHEAD_M = 20.0
CURVE_YAW_THRESHOLD_DEG = 8.0
SAME_DIRECTION_LANE_YAW_TOLERANCE_DEG = 35.0
JUNCTION_AHEAD_LOOKAHEAD_M = 80.0
JUNCTION_AHEAD_STEP_M = 5.0
# A junction-target scene must anchor on a REAL junction: the candidate is in
# the junction, or one lies within this distance ahead on its lane. Beyond it a
# "junction-like" classification (from nearby road/heading diversity) is treated
# as a false positive. Kept in sync with build_matched_structure_from_waypoint's
# junction lookahead so an accepted approach candidate yields a junction frame.
JUNCTION_TARGET_REACH_M = 40.0
# Traffic-light alignment gate. When the scene clearly shows a signalized
# junction ahead of ego, a junction-target candidate is multiplicatively
# rewarded/penalised by how close it actually sits to a real traffic light
# (from the candidate's cached environment_context.nearest_m.TrafficLight).
# A no-light "junction-like" point must not outscore a genuine signalized
# junction — critical for accident reconstruction where the signal is part of
# the scene. Distances in metres; factors multiply the candidate total_score.
SIGNAL_MATCH_NEAR_M = 30.0
SIGNAL_MATCH_MID_M = 60.0
SIGNAL_MATCH_NEAR_BOOST = 1.03
SIGNAL_MATCH_MID_FACTOR = 0.85
SIGNAL_MATCH_PENALTY_FACTOR = 0.5
# Branch-direction gate. Which side a junction forks is structural/causal for
# scenario reconstruction, so a wrong-side junction is penalised multiplicatively
# on the TOTAL score (not a soft topology nudge).
BRANCH_MATCH_EXTRA_FACTOR = 0.45
BRANCH_MATCH_MISSING_FACTOR = 0.35
BRANCH_MATCH_MIRROR_FACTOR = 0.15
BRANCH_MATCH_CANDIDATE_MISSING_FACTOR = 0.55
# Lane types that genuinely denote a center median (as opposed to road-edge
# features such as shoulder/curb/sidewalk, which can appear at the outer edge of
# a one-directional road and must not be mistaken for a center separator).
STRONG_MEDIAN_LANE_TOKENS = ("median", "restricted", "bidirectional")


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


_TOKEN_RE_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _token_in_text(token: str, text: str) -> bool:
    """Return True when *token* appears as a whole word in *text*.

    ASCII tokens use regex word-boundary matching to avoid false positives from
    substrings (e.g. "river" inside "driver", "sea" inside "sealed", "urban"
    inside "suburban").  CJK tokens fall back to plain substring search because
    Unicode word boundaries are not reliable with \\b.
    """
    if re.search(r"[一-鿿]", token):
        return token in text
    pat = _TOKEN_RE_CACHE.get(token)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
        _TOKEN_RE_CACHE[token] = pat
    return bool(pat.search(text))


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
        topology_weight: float = 0.80,
        side_context_weight: float = 0.10,
        auxiliary_weight: float = 0.10,
        blacklist_radius_m: float = 35.0,
        topology_cache_dir: Optional[str] = None,
        legacy_map_match: bool = False,
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
        self.legacy_map_match = legacy_map_match

    def analyze_scene_assets(self, scene_id: str, output_folder: str) -> str:
        paths = self._build_source_paths(scene_id, output_folder)
        report = self._build_base_report(scene_id, paths)
        report_path = os.path.join(output_folder, f"{scene_id}_match.json")

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
        image_path: Optional[str] = None,
    ) -> str:
        report_path = os.path.join(output_folder, f"{scene_id}_match.json")
        candidates_path = os.path.join(output_folder, f"{scene_id}_mm_candidates.json")
        signature_path = os.path.join(output_folder, f"{scene_id}_topo.json")
        report = self._build_base_report(
            scene_id,
            {
                "scene_understanding": os.path.join(
                    output_folder, f"{scene_id}_su.json"
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
                    "curved_road": signature.get("topology_type") == "curve",
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
        report_path = os.path.join(output_folder, f"{scene_id}_match.json")
        report = self._build_base_report(
            scene_id,
            {
                "scene_understanding": os.path.join(
                    output_folder, f"{scene_id}_su.json"
                ),
                "refined_coordinates": os.path.join(
                    output_folder, f"{scene_id}_coord_refined.json"
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
        report_path = os.path.join(output_folder, f"{scene_id}_match.json")
        scene_final_path = os.path.join(output_folder, f"{scene_id}_static.py")
        pre_match_path = os.path.join(output_folder, f"{scene_id}_static_pre_match.py")
        matched_path = os.path.join(output_folder, f"{scene_id}_static_matched.py")

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
            "scene_final_py": os.path.join(output_folder, f"{scene_id}_static.py"),
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
        ego_localization = (
            metadata.get("ego_localization")
            if isinstance(metadata.get("ego_localization"), dict)
            else {}
        )
        map_matching = (
            road_network.get("map_matching")
            if isinstance(road_network.get("map_matching"), dict)
            else {}
        )
        lane_groups = road_network.get("lane_groups") or []
        valid_lane_groups = [
            lane_group for lane_group in lane_groups if isinstance(lane_group, dict)
        ]
        primary_lane_group = valid_lane_groups[0] if valid_lane_groups else {}

        def _count_from_group(group: Dict[str, Any], key: str, default: int = 0) -> int:
            try:
                return max(0, int(group.get(key, default) or default))
            except (TypeError, ValueError):
                return max(0, int(default))

        lane_forward_lanes = _count_from_group(primary_lane_group, "forward_lane_count", 1)
        primary_opposing_lanes = _count_from_group(primary_lane_group, "opposing_lane_count", 0)
        extra_opposing_lanes = sum(
            _count_from_group(lane_group, "opposing_lane_count", 0)
            for lane_group in valid_lane_groups[1:]
            if _count_from_group(lane_group, "forward_lane_count", 0) == 0
        )
        lane_opposing_lanes = primary_opposing_lanes + extra_opposing_lanes
        lane_left_parking = any(
            _count_from_group(lane_group, "left_parking_lane_count", 0) > 0
            for lane_group in valid_lane_groups
        )
        lane_right_parking = any(
            _count_from_group(lane_group, "right_parking_lane_count", 0) > 0
            for lane_group in valid_lane_groups
        )

        forward_lanes = cls._optional_int(map_matching, "forward_lane_count")
        if forward_lanes is None:
            forward_lanes = lane_forward_lanes
        forward_lanes = max(1, int(forward_lanes or 1))

        opposing_lanes = cls._optional_int(map_matching, "opposing_lane_count")
        if opposing_lanes is None:
            opposing_lanes = lane_opposing_lanes
        opposing_lanes = max(0, int(opposing_lanes or 0))

        driving_lanes = cls._optional_int(map_matching, "driving_lane_count")
        if driving_lanes is None:
            driving_lanes = forward_lanes + opposing_lanes
        driving_lanes = max(1, int(driving_lanes or 1))

        def _slug(value: Any) -> str:
            return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")

        def _canonical_topology(value: Any) -> str:
            raw = _slug(value)
            if raw in {
                "straight_road",
                "straight_two_way",
                "curve",
                "t_junction",
                "cross_intersection",
                "multi_branch",
                "roundabout",
                "unknown",
            }:
                return raw
            if "roundabout" in raw or "rotary" in raw:
                return "roundabout"
            if raw in {"intersection", "signalized_intersection", "signalized_junction"}:
                return "cross_intersection"
            if "cross" in raw or "four_way" in raw or "4_way" in raw:
                return "cross_intersection"
            if "t_junction" in raw or "three_way" in raw or "3_way" in raw:
                return "t_junction"
            if "multi" in raw or "complex" in raw:
                return "multi_branch"
            if "curve" in raw or "bend" in raw:
                return "curve"
            if "straight" in raw:
                return "straight_two_way" if opposing_lanes > 0 else "straight_road"
            return ""

        def _branches_from_value(value: Any, junction_visible_hint: bool) -> Dict[str, bool]:
            if isinstance(value, dict):
                return {
                    "ahead": bool(value.get("ahead", junction_visible_hint)),
                    "left": bool(value.get("left", False)),
                    "right": bool(value.get("right", False)),
                    "known": bool(value.get("known", True)),
                }
            if isinstance(value, (list, tuple, set)):
                tokens = {_slug(item) for item in value}
                return {
                    "ahead": bool(
                        tokens
                        & {"ahead", "ahead_arm", "straight", "through", "oncoming", "oncoming_arm"}
                    ),
                    "left": bool(tokens & {"left", "left_arm", "left_branch"}),
                    "right": bool(tokens & {"right", "right_arm", "right_branch"}),
                    "known": bool(tokens),
                }
            return {"ahead": False, "left": False, "right": False, "known": False}

        topology_type = _canonical_topology(map_matching.get("topology_type"))
        has_map_matching = bool(map_matching)
        if not topology_type:
            if has_map_matching:
                topology_type = "unknown"
            else:
                road_segments = road_network.get("road_segments") or []
                geometry_text = " ".join(
                    str(segment.get("geometry_type") or segment.get("curvature_hint") or "")
                    for segment in road_segments
                    if isinstance(segment, dict)
                ).lower()
                if "curve" in geometry_text or "bend" in geometry_text:
                    topology_type = "curve"
                elif forward_lanes or opposing_lanes:
                    topology_type = "straight_two_way" if opposing_lanes > 0 else "straight_road"
                else:
                    topology_type = "unknown"

        junction_topologies = {"t_junction", "cross_intersection", "multi_branch", "roundabout"}
        junction_visible = cls._optional_bool(map_matching, "junction_visible")
        if junction_visible is None:
            junction_visible = topology_type in junction_topologies
        if not has_map_matching and not junction_visible:
            for junction in road_network.get("junctions") or []:
                parsed_visible, parsed_type, _parsed_branch_count = cls._parse_junction_hint(junction)
                if parsed_visible:
                    junction_visible = True
                    topology_type = _canonical_topology(parsed_type) or "t_junction"
                    break

        junction_type = str(map_matching.get("junction_type") or "").strip()
        if not junction_type:
            junction_type = topology_type if topology_type in junction_topologies else "none"
        if not junction_visible and topology_type not in junction_topologies:
            junction_type = "none"

        target_branch_count = cls._optional_int(map_matching, "target_branch_count")
        if target_branch_count is None:
            if topology_type == "cross_intersection":
                target_branch_count = 4
            elif topology_type == "t_junction":
                target_branch_count = 3
            elif topology_type == "multi_branch":
                target_branch_count = 5
            else:
                target_branch_count = 1
        target_branch_count = max(1, int(target_branch_count or 1))

        junction_branches = _branches_from_value(
            map_matching.get("junction_branches"),
            bool(junction_visible),
        )
        if not junction_branches.get("known") and not has_map_matching and junction_visible:
            junction_branches = cls._target_junction_branches(
                road_network.get("special_road_areas") or [],
                road_network.get("junctions") or [],
                {},
                bool(junction_visible),
                junction_type,
            )
        if junction_branches.get("known") and "target_branch_count" not in map_matching:
            visible_directions = int(junction_branches["ahead"]) + int(
                junction_branches["left"]
            ) + int(junction_branches["right"])
            if visible_directions > 0:
                target_branch_count = max(target_branch_count, visible_directions + 1)

        directionality_text = str(road_network.get("directionality") or "").lower()
        directionality = str(map_matching.get("directionality") or "").strip()
        if not directionality:
            directionality = (
                "two_way"
                if opposing_lanes > 0
                or "two_way" in directionality_text
                or "two-way" in directionality_text
                else "one_way_or_unknown"
            )

        has_crosswalk = cls._optional_bool(map_matching, "has_crosswalk")
        if has_crosswalk is None:
            has_crosswalk = False if has_map_matching else cls._has_area_or_control(
                road_network.get("special_road_areas") or [],
                road_network.get("control_elements") or [],
                ("crosswalk", "zebra", "pedestrian_crossing", "斑马"),
            ) or cls._visible_lane_marking(
                road_network.get("lane_markings") or {},
                ("zebra", "crosswalk", "pedestrian"),
            )

        has_traffic_light = cls._optional_bool(map_matching, "has_traffic_light")
        if has_traffic_light is None:
            has_traffic_light = False if has_map_matching else cls._has_area_or_control(
                road_network.get("special_road_areas") or [],
                road_network.get("control_elements") or [],
                ("traffic_light", "traffic_signal", "signal", "red_light", "红绿灯"),
            )

        has_traffic_sign = cls._optional_bool(map_matching, "has_traffic_sign")
        if has_traffic_sign is None:
            has_traffic_sign = False if has_map_matching else cls._has_area_or_control(
                road_network.get("special_road_areas") or [],
                road_network.get("control_elements") or [],
                ("traffic_sign", "sign", "direction_sign", "road_sign", "标志"),
            )

        has_center_median = cls._optional_bool(map_matching, "has_center_median")
        if has_center_median is None:
            has_center_median = False if has_map_matching else cls._has_positive_area_or_control(
                road_network.get("special_road_areas") or [],
                road_network.get("control_elements") or [],
                (
                    "center_median",
                    "central_median",
                    "planted_center_median",
                    "planted_median",
                    "center median",
                    "central median",
                    "planted median",
                    "median island",
                    "center island",
                    "central island",
                    "中央隔离",
                    "中央绿化",
                    "中央岛",
                    "分隔带",
                ),
            )

        curve_direction = str(map_matching.get("curve_direction") or "").lower()
        if curve_direction not in {"straight", "left", "right", "unknown"}:
            curve_direction = "straight" if topology_type in {"straight_road", "straight_two_way"} else "unknown"

        left_parking_presence = cls._optional_bool(map_matching, "left_parking_presence")
        if left_parking_presence is None:
            left_parking_presence = lane_left_parking
        right_parking_presence = cls._optional_bool(map_matching, "right_parking_presence")
        if right_parking_presence is None:
            right_parking_presence = lane_right_parking

        side_context = cls._side_context(scene_understanding)
        environment_context = (
            map_matching.get("environment_context")
            if isinstance(map_matching.get("environment_context"), dict)
            else cls._target_environment_context(scene_understanding, side_context)
        )

        signature = {
            "topology_type": topology_type,
            "junction_type": junction_type,
            "junction_visible": bool(junction_visible),
            "target_branch_count": int(target_branch_count),
            "junction_branches": junction_branches,
            "directionality": directionality,
            "driving_lane_count": driving_lanes,
            "forward_lane_count": forward_lanes,
            "opposing_lane_count": opposing_lanes,
            "left_parking_presence": bool(left_parking_presence),
            "right_parking_presence": bool(right_parking_presence),
            "has_crosswalk": bool(has_crosswalk),
            "has_traffic_light": bool(has_traffic_light),
            "has_traffic_sign": bool(has_traffic_sign),
            "has_center_median": bool(has_center_median),
            "curve_direction": curve_direction,
            "side_context": side_context,
            "environment_context": environment_context,
            "topology_signature_source": "structured_map_matching_v1"
            if has_map_matching
            else "structured_legacy_fallback_v1",
        }
        ego_lane = cls._optional_int(map_matching, "ego_lane_from_right")
        if ego_lane is None:
            ego_lane = cls._optional_int(ego_localization, "ego_lane_from_right")
        if ego_lane is not None:
            signature["ego_lane_from_right"] = max(0, ego_lane)

        ego_distance = cls._optional_float(map_matching, "ego_to_junction_distance_m")
        if ego_distance is None:
            ego_distance = cls._optional_float(ego_localization, "ego_to_junction_distance_m")
        if ego_distance is not None and ego_distance >= 0:
            signature["ego_to_junction_distance_m"] = ego_distance
        return signature

    @staticmethod
    def _optional_bool(payload: Dict[str, Any], key: str) -> Optional[bool]:
        if key not in payload:
            return None
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "present", "visible"}:
                return True
            if lowered in {"false", "no", "absent", "not_visible", "unknown"}:
                return False
        return None

    @staticmethod
    def _optional_int(payload: Dict[str, Any], key: str) -> Optional[int]:
        if key not in payload:
            return None
        try:
            return int(payload.get(key))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(payload: Dict[str, Any], key: str) -> Optional[float]:
        if key not in payload:
            return None
        try:
            return float(payload.get(key))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _apply_map_matching_overrides(
        cls,
        signature: Dict[str, Any],
        map_matching: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prefer explicit VLM map-matching fields over text/keyword inference."""
        merged = dict(signature)
        for key in ("topology_type", "junction_type", "directionality"):
            value = map_matching.get(key)
            if value not in (None, ""):
                merged[key] = str(value)

        for key in (
            "junction_visible",
            "has_crosswalk",
            "has_traffic_light",
            "has_traffic_sign",
            "has_center_median",
            "left_parking_presence",
            "right_parking_presence",
        ):
            value = cls._optional_bool(map_matching, key)
            if value is not None:
                merged[key] = value

        for key in (
            "target_branch_count",
            "driving_lane_count",
            "forward_lane_count",
            "opposing_lane_count",
            "ego_lane_from_right",
        ):
            value = cls._optional_int(map_matching, key)
            if value is not None:
                merged[key] = max(0, value)

        curve_direction = str(map_matching.get("curve_direction") or "").lower()
        if curve_direction in {"straight", "left", "right", "unknown"}:
            merged["curve_direction"] = curve_direction

        ego_distance = cls._optional_float(map_matching, "ego_to_junction_distance_m")
        if ego_distance is not None and ego_distance >= 0:
            merged["ego_to_junction_distance_m"] = ego_distance

        branches = map_matching.get("junction_branches")
        if isinstance(branches, (list, tuple, set)):
            branch_tokens = {
                str(item or "").strip().lower().replace("-", "_").replace(" ", "_")
                for item in branches
            }
            branches = {
                "ahead": bool(
                    branch_tokens
                    & {"ahead", "ahead_arm", "straight", "through", "oncoming", "oncoming_arm"}
                ),
                "left": bool(branch_tokens & {"left", "left_arm", "left_branch"}),
                "right": bool(branch_tokens & {"right", "right_arm", "right_branch"}),
                "known": bool(branch_tokens),
            }
        if isinstance(branches, dict):
            merged["junction_branches"] = {
                "ahead": bool(branches.get("ahead", merged.get("junction_visible", False))),
                "left": bool(branches.get("left", False)),
                "right": bool(branches.get("right", False)),
                "known": bool(branches.get("known", True)),
            }
            if "target_branch_count" not in map_matching:
                visible_directions = int(merged["junction_branches"]["ahead"]) + int(
                    merged["junction_branches"]["left"]
                ) + int(merged["junction_branches"]["right"])
                if visible_directions > 0:
                    merged["target_branch_count"] = visible_directions + 1

        if "driving_lane_count" not in map_matching:
            forward = int(merged.get("forward_lane_count") or 0)
            opposing = int(merged.get("opposing_lane_count") or 0)
            if forward or opposing:
                merged["driving_lane_count"] = max(1, forward + opposing)

        for key in ("side_context", "environment_context"):
            value = map_matching.get(key)
            if isinstance(value, dict):
                merged[key] = value
        return merged

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
        description = str(junction.get("description") or junction.get("evidence") or "").lower()
        junction_text = " ".join([raw_type, location, complexity, description])
        confidence_raw = junction.get("confidence")
        confidence_str = str(confidence_raw or "").lower()

        # Midblock features (crosswalk, pedestrian zone) are NOT junctions even
        # though "crosswalk" contains the substring "cross".  Check these first
        # before any intersection classification to avoid false positives.
        if any(kw in raw_type for kw in (
            "midblock", "crosswalk", "pedestrian_crossing", "pedestrian_zone",
            "not_junction", "not_a_junction",
            "ramp", "gore", "diverge", "merge_diverge", "weave",
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
        # Three-way / Y junction synonyms (a signalized junction can be a T).
        if any(
            tok in junction_text
            for tok in (
                "three_way", "three-way", "threeway", "three way",
                "3_way", "3-way", "3 way",
                "y_junction", "y-junction",
                "三岔", "丁字",
            )
        ):
            return True, "t_junction", 3
        if (
            "cross_intersection" in raw_type
            or "four_way" in raw_type
            or "four-way" in raw_type
            or "fourway" in raw_type
            or "十字" in raw_type
        ):
            return True, "cross_intersection", 4
        if "signalized_intersection" in raw_type and any(
            token in junction_text
            for token in (
                "major",
                "cross",
                "zebra",
                "stop line",
                "lane arrow",
                "overhead signal",
                "urban intersection",
                "十字",
            )
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
    def _target_junction_branches(
        special_areas: List[Any],
        junctions: List[Any],
        decisive: Dict[str, Any],
        junction_visible: bool,
        junction_type: str,
    ) -> Dict[str, Any]:
        """Infer which branches the junction has relative to the ego heading.

        Encodes the junction's *orientation* (e.g. a right-stem T-junction the
        ego can turn right but not left into) so matching can go beyond a raw
        branch count.  ``known`` reports whether any left/right evidence was
        found; when False the scorer treats branch direction as unconstrained.
        """
        if not junction_visible:
            return {"ahead": False, "left": False, "right": False, "known": False}

        branches = {"ahead": True, "left": False, "right": False}
        known = False

        # Negation phrases mean a side branch is explicitly absent (e.g. a
        # three-way junction where ego "cannot turn left"). Without these the
        # bare substring "left" in "no left turn" would wrongly open the branch.
        _neg_left = (
            "no left", "not left", "cannot turn left", "can't turn left",
            "no left turn", "without left", "禁止左", "不能左", "不可左",
            "无左", "没有左",
        )
        _neg_right = (
            "no right", "not right", "cannot turn right", "can't turn right",
            "no right turn", "without right", "禁止右", "不能右", "不可右",
            "无右", "没有右",
        )

        branch_tokens = (
            "side_road",
            "side street",
            "side_street",
            "driveway",
            "access",
            "branch",
            "cross",
            "turn",
            "approach",
            "intersection",
            "junction",
            "fork",
            "侧路",
            "支路",
            "岔",
            "路口",
        )

        def _scan(items: List[Any]) -> None:
            nonlocal known
            for item in items or []:
                if isinstance(item, dict):
                    text = json.dumps(item, ensure_ascii=False).lower()
                else:
                    text = str(item).lower()
                if not any(token in text for token in branch_tokens):
                    continue
                neg_right = any(p in text for p in _neg_right)
                neg_left = any(p in text for p in _neg_left)
                if ("right" in text or "右" in text) and not neg_right:
                    branches["right"] = True
                    known = True
                if ("left" in text or "左" in text) and not neg_left:
                    branches["left"] = True
                    known = True
                # An explicit "no left/right turn" is itself directional
                # evidence: the junction's orientation is known (that side is
                # absent), so do not fall back to the all-open cross default.
                if neg_right or neg_left:
                    known = True

        _scan(special_areas)
        _scan(junctions)

        for left_key in ("left_branch", "can_turn_left", "left_turn_available"):
            if left_key in decisive:
                branches["left"] = bool(decisive.get(left_key))
                known = True
        for right_key in ("right_branch", "can_turn_right", "right_turn_available"):
            if right_key in decisive:
                branches["right"] = bool(decisive.get(right_key))
                known = True

        # A signalized 4-way / cross intersection opens all directions — but only
        # as a fallback when no directional evidence was found above. Applying it
        # unconditionally would clobber a confirmed single-side (T) branch back
        # to a full cross.
        if not known and junction_type == "cross_intersection":
            branches["left"] = True
            branches["right"] = True
            known = True

        branches["known"] = known
        return branches

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
    def _item_presence_is_false(item: Dict[str, Any]) -> bool:
        if item.get("presence") is False or item.get("visible") is False:
            return True
        for key in ("presence", "visible", "present"):
            value = item.get(key)
            if isinstance(value, str) and value.strip().lower() in {
                "false",
                "no",
                "none",
                "absent",
                "not_visible",
            }:
                return True
        return False

    @staticmethod
    def _has_negative_area_or_control(
        special_areas: List[Any],
        control_elements: List[Any],
        keywords: Tuple[str, ...],
    ) -> bool:
        for item in list(special_areas or []) + list(control_elements or []):
            if not isinstance(item, dict):
                continue
            text = json.dumps(item, sort_keys=True, ensure_ascii=False).lower()
            if any(keyword.lower() in text for keyword in keywords) and (
                SceneMapMatcher._item_presence_is_false(item)
            ):
                return True
        return False

    @staticmethod
    def _has_positive_area_or_control(
        special_areas: List[Any],
        control_elements: List[Any],
        keywords: Tuple[str, ...],
    ) -> bool:
        negative_tokens = (
            "no center median",
            "no central median",
            "no median",
            "without center median",
            "without central median",
            "absence of center median",
            "undivided",
            "无中央隔离",
            "没有中央隔离",
        )
        for item in list(special_areas or []) + list(control_elements or []):
            if not isinstance(item, dict):
                continue
            if SceneMapMatcher._item_presence_is_false(item):
                continue
            text = json.dumps(item, sort_keys=True, ensure_ascii=False).lower()
            if any(token in text for token in negative_tokens):
                continue
            if any(keyword.lower() in text for keyword in keywords):
                return True
        return False

    @staticmethod
    def _visible_lane_marking(lane_markings: Dict[str, Any], keywords: Tuple[str, ...]) -> bool:
        visible_tokens = ("true", "present", "visible", "yes", "有", "可见")
        for key, value in (lane_markings or {}).items():
            key_text = str(key or "").lower()
            value_text = str(value or "").lower()
            if not any(keyword in key_text for keyword in keywords):
                continue
            if any(token in value_text for token in visible_tokens):
                return True
            if value is True:
                return True
        return False

    @staticmethod
    def _text_has_positive_tokens(
        text: str,
        positive_tokens: Tuple[str, ...],
        negative_tokens: Tuple[str, ...] = (),
    ) -> bool:
        lowered = str(text or "").lower()
        if any(token.lower() in lowered for token in negative_tokens):
            return False
        return any(token.lower() in lowered for token in positive_tokens)

    @staticmethod
    def _text_has_curbside_parked_vehicles(text: str, side: str) -> bool:
        lowered = str(text or "").lower()
        if not lowered:
            return False
        parking_tokens = (
            "parked car",
            "parked cars",
            "parked vehicle",
            "parked vehicles",
            "parking row",
            "curbside parking",
            "roadside parking",
            "路边停车",
            "停在路边",
            "停车车辆",
        )
        if not any(token in lowered for token in parking_tokens):
            return False
        global_negative_patterns = (
            r"(?:no|without)[^.，。,;]*(?:curbside parked vehicles|parked vehicles|parked cars)",
            r"(?:no|without)[^.，。,;]*(?:marked parking strip|parking strip|parking lane|parking lanes)",
            r"no extra[^.，。,;]*(?:curbside parking lanes|parking lanes)",
            r"(?:无|没有|未见)[^。；，.]*(?:路边停车|停车车辆|停车道|停车带)",
        )
        if any(re.search(pattern, lowered) for pattern in global_negative_patterns):
            return False
        if any(
            token in lowered
            for token in (
                "moving traffic",
                "stopped traffic",
                "traffic queue",
                "queued traffic",
                "排队车辆",
                "等待车辆",
            )
        ):
            return False
        side_tokens = {
            "left": (
                "left curb",
                "left curbside",
                "left road edge",
                "left edge",
                "left side",
                "left roadside",
                "左侧路边",
                "左路边",
                "左侧",
            ),
            "right": (
                "right curb",
                "right curbside",
                "right road edge",
                "right edge",
                "right side",
                "right roadside",
                "右侧路边",
                "右路边",
                "右侧",
            ),
        }
        side_negative_patterns = {
            "left": (
                r"left[^.，。,;]*parking[^.，。,;]*(?:not evident|absent|none)",
                r"(?:no|without)[^.，。,;]*left[^.，。,;]*parking",
                r"左侧[^。；，.]*(?:无|没有|未见)[^。；，.]*停车",
            ),
            "right": (
                r"right[^.，。,;]*parking[^.，。,;]*(?:not evident|absent|none)",
                r"(?:no|without)[^.，。,;]*right[^.，。,;]*parking",
                r"右侧[^。；，.]*(?:无|没有|未见)[^。；，.]*停车",
            ),
        }
        if any(re.search(pattern, lowered) for pattern in side_negative_patterns.get(side, ())):
            return False
        return any(token in lowered for token in side_tokens.get(side, ()))

    @staticmethod
    def _apply_straight_road_parking_symmetry(signature: Dict[str, Any]) -> None:
        topology = str(signature.get("topology_type") or "").lower()
        if topology not in {"straight_road", "straight_two_way"}:
            return
        if signature.get("junction_visible"):
            return
        if signature.get("directionality") != "two_way":
            return
        if bool(signature.get("has_center_median")):
            return
        left = bool(signature.get("left_parking_presence"))
        right = bool(signature.get("right_parking_presence"))
        if left == right:
            return
        sources = signature.get("_parking_presence_sources") or {}
        source_side = "left" if left else "right"
        source = sources.get(source_side) or {}
        can_infer_from_source = any(
            bool(source.get(key)) for key in ("lane_group", "decisive")
        )
        if not can_infer_from_source:
            return
        if left or right:
            signature["left_parking_presence"] = True
            signature["right_parking_presence"] = True
            side_context = signature.setdefault("side_context", {})
            side_context["left_parking_strip"] = True
            side_context["right_parking_strip"] = True
            signature["parking_symmetry_inferred"] = True

    @staticmethod
    def _side_context(scene_understanding: Dict[str, Any]) -> Dict[str, Any]:
        road_network = scene_understanding.get("road_network", {}) or {}
        general_environment = scene_understanding.get("general_environment", {}) or {}
        descriptive_payload = {
            "actor_layout": scene_understanding.get("actor_layout") or {},
            "general_environment": general_environment,
            "metadata": scene_understanding.get("metadata") or {},
        }
        text_all = json.dumps(descriptive_payload, ensure_ascii=False).lower()
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
        left_special_parking = any(
            isinstance(area, dict)
            and "parking" in str(area.get("type") or "").lower()
            and "left" in json.dumps(area, ensure_ascii=False).lower()
            for area in special_areas
        )
        right_special_parking = any(
            isinstance(area, dict)
            and "parking" in str(area.get("type") or "").lower()
            and "right" in json.dumps(area, ensure_ascii=False).lower()
            for area in special_areas
        )
        left_text_parking = SceneMapMatcher._text_has_curbside_parked_vehicles(
            text_left,
            "left",
        )
        right_text_parking = SceneMapMatcher._text_has_curbside_parked_vehicles(
            text_right,
            "right",
        )
        return {
            "left_continuous_buildings": any(
                _token_in_text(token, text_left)
                for token in ("building", "shop", "storefront", "建筑", "商铺")
            ),
            "right_continuous_buildings": any(
                _token_in_text(token, text_right)
                for token in ("building", "shop", "storefront", "建筑", "商铺")
            ),
            "tree_lined": any(_token_in_text(t, text_all) for t in ("tree", "trees", "树")),
            "left_parking_strip": left_special_parking,
            "right_parking_strip": right_special_parking,
            "left_parking_from_special_area": left_special_parking,
            "right_parking_from_special_area": right_special_parking,
            "left_parking_from_text": left_text_parking,
            "right_parking_from_text": right_text_parking,
        }

    @staticmethod
    def _target_environment_context(
        scene_understanding: Dict[str, Any],
        side_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        road_network = scene_understanding.get("road_network", {}) or {}
        general_environment = scene_understanding.get("general_environment", {}) or {}
        environment_payload = {
            "urban_density": general_environment.get("urban_density"),
            "roadside_context_left": general_environment.get("roadside_context_left"),
            "roadside_context_right": general_environment.get("roadside_context_right"),
            "non_spawnable_landmarks": general_environment.get("non_spawnable_landmarks"),
            "roadside_boundaries": road_network.get("roadside_boundaries"),
            "road_type": road_network.get("road_type"),
        }
        # SU emits snake_case enums (e.g. "suburban_commercial", "urban_arterial",
        # "business_signage").  Word-boundary token matching treats "_" as a word
        # char, so "urban"/"commercial" would never match inside those compounds.
        # Normalize underscores to spaces so whole-word tokens match.
        text = json.dumps(environment_payload, ensure_ascii=False).lower().replace("_", " ")
        side_context = side_context or {}

        urban_tokens = (
            "urban",
            "city",
            "commercial",
            "downtown",
            "storefront",
            "shop",
            "building",
            "sidewalk",
            "curb",
            "residential",
            "城市",
            "城区",
            "商业",
            "商铺",
            "建筑",
            "人行道",
            "路缘",
        )
        building_tokens = (
            "building",
            "storefront",
            "shop",
            "commercial",
            "residential",
            "建筑",
            "商铺",
            "店铺",
        )
        sidewalk_tokens = ("sidewalk", "pavement", "curb", "人行道", "路缘")
        natural_tokens = (
            "mountain",
            "rural",
            "forest",
            "grass",
            "rock",
            "rocky",
            "cliff",
            "hill",
            "terrain",
            "vegetation",
            "tree-covered",
            "山",
            "乡村",
            "草地",
            "岩石",
            "悬崖",
            "地形",
            "植被",
        )
        water_tokens = ("water", "river", "lake", "sea", "waterside", "coast", "水", "河", "湖", "海")

        expects_buildings = any(_token_in_text(token, text) for token in building_tokens) or bool(
            side_context.get("left_continuous_buildings")
            or side_context.get("right_continuous_buildings")
        )
        expects_sidewalks = any(_token_in_text(token, text) for token in sidewalk_tokens)
        expects_urban = (
            any(_token_in_text(token, text) for token in urban_tokens)
            or expects_buildings
            or expects_sidewalks
        )
        expects_water = any(_token_in_text(token, text) for token in water_tokens)
        expects_natural = any(_token_in_text(token, text) for token in natural_tokens) or expects_water
        return {
            "expects_urban": bool(expects_urban),
            "expects_buildings": bool(expects_buildings),
            "expects_sidewalks": bool(expects_sidewalks),
            "expects_natural": bool(expects_natural),
            "expects_water": bool(expects_water),
            "avoid_water": bool(expects_urban and not expects_water),
            "avoid_terrain_dominant": bool(expects_urban and not expects_natural),
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
        curve_tokens = ("curve", "curved", "bend", "bending", "弯", "弯道", "转弯")
        curved_road = any(
            any(
                token in str(segment.get("geometry_type") or segment.get("curvature_hint") or "").lower()
                for token in curve_tokens
            )
            for segment in road_segments
            if isinstance(segment, dict)
        ) or any(token in combined_text for token in curve_tokens)
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
            "curved_road": curved_road,
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

        best_rejected: Optional[Dict[str, Any]] = None
        best_rejected_score = -1.0
        best_rejected_world: Optional[str] = None

        top_candidates_summary: List[Dict[str, Any]] = []
        accepted_summary: List[Dict[str, Any]] = []
        rejected_summary: List[Dict[str, Any]] = []
        # Full per-map best entries, kept so an optional LLM re-ranker can choose
        # among a closed set without re-loading caches.
        map_best_entries: List[Dict[str, Any]] = []

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
            self._normalize_cached_parking_sides(candidates)

            map_best: Optional[Dict[str, Any]] = None
            map_best_score = -1.0
            map_best_accepted: Optional[Dict[str, Any]] = None
            map_best_accepted_score = -1.0
            map_best_rejected: Optional[Dict[str, Any]] = None
            map_best_rejected_score = -1.0

            for candidate in candidates:
                loc = candidate.get("location") or {}
                blacklisted = self._cache_is_blacklisted(loc, blacklist_locations)
                if (
                    scene_features.get("road_topology_signature")
                    and not self.legacy_map_match
                ):
                    score, details = score_candidate_v2(
                        scene_features.get("road_topology_signature") or {},
                        candidate,
                        blacklisted=blacklisted,
                    )
                else:
                    if blacklisted:
                        continue
                    score, details = self._score_candidate_features(candidate, scene_features)
                details["blacklisted"] = False
                if blacklisted:
                    details["blacklisted"] = True
                hard_reject = bool(details.get("hard_reject"))
                reject_reasons = details.get("reject_reasons") or []
                reject_reason = (
                    "; ".join(str(item) for item in reject_reasons)
                    if reject_reasons
                    else details.get("reject_reason")
                )

                entry = {
                    "score": score,
                    "score_details": details,
                    "hard_reject": hard_reject,
                    "reject_reason": reject_reason,
                    "search_strategy": "cache_v2"
                    if scene_features.get("road_topology_signature")
                    and not self.legacy_map_match
                    else "cache",
                    "location": loc,
                    "yaw": candidate.get("yaw", 0.0),
                    "candidate_lane": candidate.get("candidate_lane", {}),
                    **{k: candidate[k] for k in candidate
                       if k not in ("location", "yaw", "candidate_lane")},
                    "_world_name": world_name,
                }

                # Track per-map best (non-hard-reject preferred)
                if score > map_best_score:
                    map_best_score = score
                    map_best = entry
                if hard_reject:
                    if score > map_best_rejected_score:
                        map_best_rejected_score = score
                        map_best_rejected = entry
                elif score > map_best_accepted_score:
                    map_best_accepted_score = score
                    map_best_accepted = entry

                # Track global best (non-hard-reject only)
                if not hard_reject and score > global_best_score:
                    global_best_score = score
                    global_best = entry
                    global_best_world = world_name
                elif hard_reject and score > best_rejected_score:
                    best_rejected_score = score
                    best_rejected = entry
                    best_rejected_world = world_name

            if map_best:
                top_candidates_summary.append(summarize_entry(map_best))
                map_best_entries.append(map_best)
            if map_best_accepted:
                accepted_summary.append(summarize_entry(map_best_accepted))
            if map_best_rejected:
                rejected_summary.append(summarize_entry(map_best_rejected))

        if global_best is None:
            return {
                "status": "unmatched" if best_rejected is not None else "no_candidates",
                "world_name": None,
                "candidate_summary": {
                    "accepted": accepted_summary[:20],
                    "rejected": rejected_summary[:20],
                    "top_candidates": top_candidates_summary,
                },
                "candidate_debug": {
                    "accepted": accepted_summary[:50],
                    "rejected": rejected_summary[:50],
                    "top_candidates": top_candidates_summary,
                    "best_rejected_candidate": self._assemble_cache_best_match(best_rejected)
                    if best_rejected is not None
                    else None,
                    "blacklist_locations": blacklist_locations or [],
                },
                "best_match": None,
                "projected_layout": {},
                "reason": "No candidates satisfied v2 topology gates."
                if best_rejected is not None
                else "No candidates found across all cached maps.",
            }

        return {
            "status": "matched",
            "world_name": global_best_world,
            "best_match": self._assemble_cache_best_match(global_best),
            "projected_layout": {},
            "candidate_summary": {
                "accepted": accepted_summary[:20],
                "rejected": rejected_summary[:20],
                "top_candidates": top_candidates_summary,
            },
            "candidate_debug": {
                "accepted": accepted_summary[:50],
                "rejected": rejected_summary[:50],
                "top_candidates": top_candidates_summary,
                "best_rejected_candidate": self._assemble_cache_best_match(best_rejected)
                if best_rejected is not None
                else None,
                "blacklist_locations": blacklist_locations or [],
            },
            "reason": None,
        }

    # Candidate feature keys surfaced in best_match for downstream stages/debug.
    _CACHE_CANDIDATE_FEATURE_KEYS = (
        "same_road_lane_count", "nearby_road_count", "nearby_lane_count",
        "heading_cluster_count", "estimated_junction_degree",
        "candidate_topology_type", "junction_waypoint_ratio", "is_junction",
        "distance_to_junction_ahead", "distance_to_traffic_light_ahead",
        "environment_context", "has_center_median_candidate",
        "center_median_evidence",
        "same_direction_lane_count", "has_parallel_same_direction_lanes",
        "same_direction_lane_evidence",
        "physical_junction_arms", "lane_maneuver_dirs",
        "is_curve", "curve_yaw_delta_deg", "curve_abs_yaw_delta_deg",
        "curve_direction", "curve_score", "curve_sample_distance_m",
        "left_parking_lane_present", "right_parking_lane_present",
    )

    @classmethod
    def _assemble_cache_best_match(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Build a best_match dict from a cache candidate entry."""
        used_rejected = bool(entry.get("hard_reject"))
        score = float(entry.get("score") or 0.0)
        best_match = {
            "search_strategy": entry.get("search_strategy") or "cache",
            "spawn_point_index": None,
            "location": entry.get("location"),
            "yaw": entry.get("yaw"),
            "coarse_score": score,
            "refined_score": score,
            "score_details": entry.get("score_details"),
            "candidate_features": {
                k: entry[k] for k in cls._CACHE_CANDIDATE_FEATURE_KEYS if k in entry
            },
            "candidate_lane": entry.get("candidate_lane"),
            "layout_penalties": {},
            "used_rejected_candidate": used_rejected,
            "reject_reason": entry.get("reject_reason") if used_rejected else None,
            "world": entry.get("_world_name"),
        }
        details = entry.get("score_details") or {}
        candidate_kind = details.get("candidate_kind")
        if candidate_kind:
            best_match["candidate_features"]["candidate_kind"] = candidate_kind
        matched_structure = entry.get("matched_structure")
        if isinstance(matched_structure, dict) and not matched_structure.get("error"):
            best_match["matched_structure"] = matched_structure
        return best_match

    @classmethod
    def _normalize_cached_parking_sides(cls, candidates: List[Dict[str, Any]]) -> None:
        """Backfill road-side parking semantics for cached two-way road candidates.

        Older caches only recorded whether a parking lane was reachable by repeatedly
        following a waypoint's fixed left/right link. On two-way roads, the far curb's
        parking lane is usually on the opposing lane's own outer side, so a single
        candidate can miss it even though another candidate on the same road sees it.
        """
        road_signs: Dict[Any, set] = {}
        for candidate in candidates or []:
            lane = candidate.get("candidate_lane") or {}
            road_id = lane.get("road_id")
            lane_sign = cls._lane_id_sign(lane.get("lane_id"))
            if road_id is None or lane_sign == 0:
                continue
            road_signs.setdefault(road_id, set()).add(lane_sign)

        road_side_parking: Dict[Any, Dict[str, bool]] = {}
        for candidate in candidates or []:
            lane = candidate.get("candidate_lane") or {}
            road_id = lane.get("road_id")
            lane_sign = cls._lane_id_sign(lane.get("lane_id"))
            if road_id is None or lane_sign == 0 or road_signs.get(road_id) != {-1, 1}:
                continue
            sides = road_side_parking.setdefault(
                road_id,
                {"negative": False, "positive": False},
            )
            if bool(candidate.get("right_parking_lane_present")):
                sides["negative" if lane_sign < 0 else "positive"] = True
            if bool(candidate.get("left_parking_lane_present")):
                sides["positive" if lane_sign < 0 else "negative"] = True

        for candidate in candidates or []:
            lane = candidate.get("candidate_lane") or {}
            road_id = lane.get("road_id")
            lane_sign = cls._lane_id_sign(lane.get("lane_id"))
            sides = road_side_parking.get(road_id)
            if lane_sign == 0 or not sides:
                continue
            if lane_sign < 0:
                candidate["right_parking_lane_present"] = bool(
                    candidate.get("right_parking_lane_present") or sides["negative"]
                )
                candidate["left_parking_lane_present"] = bool(
                    candidate.get("left_parking_lane_present") or sides["positive"]
                )
            else:
                candidate["right_parking_lane_present"] = bool(
                    candidate.get("right_parking_lane_present") or sides["positive"]
                )
                candidate["left_parking_lane_present"] = bool(
                    candidate.get("left_parking_lane_present") or sides["negative"]
                )

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
            if accepted_candidates:
                best_candidate = accepted_candidates[0]
            if best_candidate is None:
                fallback_candidates = [
                    candidate
                    for candidate in coarse_candidates
                    if not (candidate.get("score_details") or {}).get("blacklisted")
                ] or coarse_candidates
                best_rejected = max(
                    fallback_candidates,
                    key=self._candidate_sort_key,
                ) if fallback_candidates else None
                if best_rejected is not None:
                    candidate_debug["best_rejected_candidate"] = (
                        self._build_topology_match_record(best_rejected)
                    )
                return {
                    "status": "unmatched",
                    "world_name": world_name,
                    "candidate_summary": candidate_summary,
                    "candidate_debug": candidate_debug,
                    "best_match": None,
                    "projected_layout": {},
                    "reason": "No CARLA candidates satisfied v2 topology gates.",
                }
            best_match = self._build_topology_match_record(best_candidate)
            best_match["matched_structure"] = self._build_matched_structure(
                world_map, best_candidate
            )
            return {
                "status": "matched",
                "world_name": world_name,
                "candidate_summary": candidate_summary,
                "candidate_debug": candidate_debug,
                "best_match": best_match,
                "projected_layout": {},
                "reason": None,
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
    def _build_matched_structure(
        world_map: Any, candidate: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Recover the candidate waypoint and emit its structural description.

        Additive: failures degrade to ``None`` (with an error note) so the rest
        of the match record is unaffected. Consumed by reference-frame-based
        actor placement (see tools/reference_frame.py).
        """
        try:
            import carla

            from tools import map_structure

            loc = candidate.get("location") or {}
            waypoint = world_map.get_waypoint(
                carla.Location(
                    x=float(loc.get("x", 0.0)),
                    y=float(loc.get("y", 0.0)),
                    z=float(loc.get("z", 0.0)),
                ),
                project_to_road=True,
            )
            if waypoint is None:
                return None
            return map_structure.build_matched_structure_from_waypoint(waypoint)
        except Exception as exc:  # pragma: no cover - defensive, CARLA runtime only
            return {"error": str(exc)}

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
                "has_center_median_candidate": candidate.get("has_center_median_candidate"),
                "center_median_evidence": candidate.get("center_median_evidence"),
                "same_direction_lane_count": candidate.get("same_direction_lane_count"),
                "has_parallel_same_direction_lanes": candidate.get(
                    "has_parallel_same_direction_lanes"
                ),
                "same_direction_lane_evidence": candidate.get(
                    "same_direction_lane_evidence"
                ),
                "is_curve": candidate.get("is_curve"),
                "curve_yaw_delta_deg": candidate.get("curve_yaw_delta_deg"),
                "curve_abs_yaw_delta_deg": candidate.get("curve_abs_yaw_delta_deg"),
                "curve_direction": candidate.get("curve_direction"),
                "curve_score": candidate.get("curve_score"),
                "curve_sample_distance_m": candidate.get("curve_sample_distance_m"),
                "left_parking_lane_present": candidate.get("left_parking_lane_present"),
                "right_parking_lane_present": candidate.get("right_parking_lane_present"),
                "candidate_kind": (candidate.get("score_details") or {}).get(
                    "candidate_kind"
                ),
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
            blacklisted = self._is_blacklisted(waypoint.transform.location, blacklist_locations)
            if scene_features.get("road_topology_signature") and not self.legacy_map_match:
                score, details = score_candidate_v2(
                    scene_features.get("road_topology_signature") or {},
                    candidate,
                    blacklisted=blacklisted,
                )
            else:
                score, details = self._score_candidate_features(candidate, scene_features)
                if blacklisted:
                    details["hard_reject"] = True
                    details["blacklisted"] = True
                    details["reject_reason"] = "Candidate is inside a blacklisted rematch region."
            reject_reasons = details.get("reject_reasons") or []
            reject_reason = (
                "; ".join(str(item) for item in reject_reasons)
                if reject_reasons
                else details.get("reject_reason")
            )
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "hard_reject": bool(details.get("hard_reject")),
                    "reject_reason": reject_reason,
                    "search_strategy": "full_waypoints_v2"
                    if scene_features.get("road_topology_signature")
                    and not self.legacy_map_match
                    else "full_waypoints",
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
            blacklisted = self._is_blacklisted(waypoint.transform.location, blacklist_locations)
            if scene_features.get("road_topology_signature") and not self.legacy_map_match:
                score, details = score_candidate_v2(
                    scene_features.get("road_topology_signature") or {},
                    candidate,
                    blacklisted=blacklisted,
                )
            else:
                score, details = self._score_candidate_features(candidate, scene_features)
                if blacklisted:
                    details["hard_reject"] = True
                    details["blacklisted"] = True
                    details["reject_reason"] = "Candidate is inside a blacklisted rematch region."
            reject_reasons = details.get("reject_reasons") or []
            reject_reason = (
                "; ".join(str(item) for item in reject_reasons)
                if reject_reasons
                else details.get("reject_reason")
            )
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "hard_reject": bool(details.get("hard_reject")),
                    "reject_reason": reject_reason,
                    "search_strategy": "spawn_points_v2"
                    if scene_features.get("road_topology_signature")
                    and not self.legacy_map_match
                    else "spawn_points",
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

    @staticmethod
    def _single_waypoint_at_distance(
        waypoint: Any, method_name: str, distance_m: float
    ) -> Optional[Any]:
        try:
            method = getattr(waypoint, method_name)
            waypoints = method(distance_m)
        except Exception:
            return None
        return waypoints[0] if waypoints else None

    @classmethod
    def _extract_curve_features(cls, waypoint: Any) -> Dict[str, Any]:
        previous_wp = cls._single_waypoint_at_distance(
            waypoint, "previous", CURVE_LOOKAHEAD_M
        )
        next_wp = cls._single_waypoint_at_distance(waypoint, "next", CURVE_LOOKAHEAD_M)
        if previous_wp is None or next_wp is None:
            return {
                "is_curve": False,
                "curve_yaw_delta_deg": 0.0,
                "curve_abs_yaw_delta_deg": 0.0,
                "curve_direction": "unknown",
                "curve_score": 0.0,
                "curve_sample_distance_m": CURVE_LOOKAHEAD_M,
            }

        previous_yaw = _safe_float(previous_wp.transform.rotation.yaw)
        next_yaw = _safe_float(next_wp.transform.rotation.yaw)
        yaw_delta = _normalize_angle_deg(next_yaw - previous_yaw)
        abs_delta = abs(yaw_delta)
        is_curve = abs_delta >= CURVE_YAW_THRESHOLD_DEG
        if not is_curve:
            direction = "straight"
        elif yaw_delta > 0.0:
            # CARLA yaw increases clockwise in its x-east/y-south frame.
            direction = "right"
        else:
            direction = "left"

        return {
            "is_curve": is_curve,
            "curve_yaw_delta_deg": round(yaw_delta, 3),
            "curve_abs_yaw_delta_deg": round(abs_delta, 3),
            "curve_direction": direction,
            "curve_score": round(min(1.0, abs_delta / 30.0), 3),
            "curve_sample_distance_m": CURVE_LOOKAHEAD_M,
        }

    @classmethod
    def _detect_junction_ahead(
        cls,
        center_waypoint: Any,
        max_distance: float = JUNCTION_AHEAD_LOOKAHEAD_M,
        step: float = JUNCTION_AHEAD_STEP_M,
    ) -> Optional[float]:
        """Walk forward along the lane and report the distance (m) to the first
        junction ahead.

        Returns the accumulated distance to the first junction waypoint found
        within *max_distance*, or ``None`` when no junction is reached (the lane
        ends, nothing is found, or CARLA raises).  The ego spawn point itself
        being inside a junction is reported separately via ``is_junction``; this
        only looks downstream of a non-junction spawn.
        """
        try:
            current = center_waypoint
            traversed = 0.0
            while traversed < max_distance:
                next_waypoints = current.next(step)
                if not next_waypoints:
                    return None
                traversed += step
                junction_next = next(
                    (wp for wp in next_waypoints if getattr(wp, "is_junction", False)),
                    None,
                )
                if junction_next is not None:
                    return round(traversed, 1)
                current = next_waypoints[0]
            return None
        except Exception:
            return None

    @staticmethod
    def _detect_traffic_light_ahead(
        center_waypoint: Any,
        max_distance: float = JUNCTION_AHEAD_LOOKAHEAD_M,
    ) -> Optional[float]:
        """Distance (m) to the nearest traffic-light landmark AHEAD on the lane.

        Uses CARLA's OpenDRIVE landmark query (signal type ``'1000001'`` =
        traffic light), which follows the lane forward. A light governing a
        junction BEHIND ego, or one on a crossing street, is therefore NOT
        counted -- unlike the omnidirectional ``environment_context.nearest_m``
        distance, which a behind/side light can satisfy (the dashcam then shows
        no signal ahead). Returns ``None`` when no traffic light is found ahead
        within ``max_distance`` (or CARLA raises / the build lacks the API).
        """
        try:
            landmarks = center_waypoint.get_landmarks_of_type(
                float(max_distance), "1000001", False
            )
        except Exception:
            return None
        nearest: Optional[float] = None
        for landmark in landmarks or []:
            dist = getattr(landmark, "distance", None)
            if isinstance(dist, (int, float)) and dist >= 0:
                if nearest is None or dist < nearest:
                    nearest = float(dist)
        return round(nearest, 1) if nearest is not None else None

    @staticmethod
    def _turn_direction(
        approach_yaw: float,
        exit_yaw: float,
        ahead_tol_deg: float = 35.0,
        uturn_tol_deg: float = 150.0,
    ) -> str:
        """Classify the maneuver from *approach_yaw* to *exit_yaw* as one of
        ``ahead`` / ``left`` / ``right`` / ``uturn``.

        Uses CARLA's left-handed convention where forward = (cos yaw, sin yaw)
        and +Y is to the vehicle's right, so a positive cross product
        (signed angle) means a right turn.
        """
        approach = math.radians(approach_yaw)
        exit_ = math.radians(exit_yaw)
        fx, fy = math.cos(approach), math.sin(approach)
        ex, ey = math.cos(exit_), math.sin(exit_)
        dot = fx * ex + fy * ey
        cross = fx * ey - fy * ex  # > 0 => right turn in CARLA's frame
        angle = math.degrees(math.atan2(cross, dot))  # signed; positive = right
        if abs(angle) <= ahead_tol_deg:
            return "ahead"
        if abs(angle) >= uturn_tol_deg:
            return "uturn"
        return "right" if angle > 0 else "left"

    def _classify_junction_branches(
        self,
        center_waypoint: Any,
        max_distance: float = JUNCTION_AHEAD_LOOKAHEAD_M,
        step: float = JUNCTION_AHEAD_STEP_M,
    ) -> Optional[Dict[str, Any]]:
        """Return which maneuvers (ahead/left/right/uturn) the ego lane can take
        through the first junction ahead, by following CARLA lane connectivity.

        Returns ``None`` when no junction is reachable within *max_distance*.
        The returned dict carries boolean ``ahead/left/right/uturn`` flags plus
        a ``branch_count`` of distinct maneuver directions — this captures the
        junction's orientation relative to the ego heading (e.g. a right-stem
        T-junction yields ``{ahead, right}`` and no ``left``).
        """
        try:
            approach = center_waypoint
            junction_entry = None
            if bool(getattr(center_waypoint, "is_junction", False)):
                junction_entry = center_waypoint
            else:
                current = center_waypoint
                traversed = 0.0
                while traversed < max_distance:
                    next_waypoints = current.next(step)
                    if not next_waypoints:
                        break
                    junction_next = next(
                        (wp for wp in next_waypoints if getattr(wp, "is_junction", False)),
                        None,
                    )
                    if junction_next is not None:
                        junction_entry = junction_next
                        approach = current
                        break
                    traversed += step
                    current = next_waypoints[0]
            if junction_entry is None:
                return None

            approach_yaw = float(approach.transform.rotation.yaw)
            exits: List[Any] = []
            visited: set = set()
            frontier = [(approach, False)]
            depth = 0
            while frontier and depth < 16:
                new_frontier = []
                for waypoint, in_junction in frontier:
                    for nxt in waypoint.next(step) or []:
                        key = (
                            round(float(nxt.transform.location.x), 1),
                            round(float(nxt.transform.location.y), 1),
                            int(nxt.road_id),
                            int(nxt.lane_id),
                        )
                        if key in visited:
                            continue
                        visited.add(key)
                        if bool(getattr(nxt, "is_junction", False)):
                            new_frontier.append((nxt, True))
                        elif in_junction:
                            exits.append(nxt)  # left the junction => an exit road
                        else:
                            new_frontier.append((nxt, False))  # still approaching
                frontier = new_frontier
                depth += 1

            dirs = {"ahead": False, "left": False, "right": False, "uturn": False}
            for exit_wp in exits:
                direction = self._turn_direction(
                    approach_yaw, float(exit_wp.transform.rotation.yaw)
                )
                dirs[direction] = True
            dirs["branch_count"] = sum(
                1 for key in ("ahead", "left", "right", "uturn") if dirs[key]
            )
            return dirs
        except Exception:
            return None

    @classmethod
    def _physical_junction_arms_for_waypoint(
        cls,
        center_waypoint: Any,
        max_distance: float = JUNCTION_TARGET_REACH_M,
    ) -> Optional[Dict[str, Any]]:
        """Return physical junction arms relative to the candidate's approach.

        Unlike ``_classify_junction_branches``, this uses the whole
        ``carla.Junction`` leg graph. It describes road topology, not which
        maneuvers the current lane can take.
        """
        try:
            from tools.map_structure import build_matched_structure_from_waypoint

            ego_yaw = cls._waypoint_yaw(center_waypoint)
            structure = build_matched_structure_from_waypoint(
                center_waypoint,
                ego_inbound_yaw=ego_yaw,
                junction_lookahead_m=float(max_distance),
            )
        except Exception:
            try:
                ego_yaw = cls._waypoint_yaw(center_waypoint)
                junction_wp = center_waypoint
                if not bool(getattr(junction_wp, "is_junction", False)):
                    current = center_waypoint
                    traversed = 0.0
                    junction_wp = None
                    while traversed < max_distance:
                        next_waypoints = current.next(JUNCTION_AHEAD_STEP_M)
                        if not next_waypoints:
                            break
                        junction_wp = next(
                            (wp for wp in next_waypoints if getattr(wp, "is_junction", False)),
                            None,
                        )
                        if junction_wp is not None:
                            break
                        current = next_waypoints[0]
                        traversed += JUNCTION_AHEAD_STEP_M
                if junction_wp is None or not bool(getattr(junction_wp, "is_junction", False)):
                    return None
                junction = junction_wp.get_junction()
                pairs = list(junction.get_waypoints(None) or [])
                dirs = {"ahead": False, "left": False, "right": False}
                for _entry_wp, exit_wp in pairs:
                    direction = cls._turn_direction(
                        ego_yaw,
                        float(exit_wp.transform.rotation.yaw),
                    )
                    if direction in dirs:
                        dirs[direction] = True
                arm_count = max(
                    1 + sum(1 for key in ("ahead", "left", "right") if dirs[key]),
                    len(pairs),
                )
                return {
                    **dirs,
                    "ego": True,
                    "known": bool(pairs),
                    "arm_count": arm_count,
                    "leg_count": arm_count,
                    "junction_id": getattr(junction, "id", None),
                    "source": "junction_waypoints_fallback",
                }
            except Exception:
                return None
        if not isinstance(structure, dict) or structure.get("kind") != "junction":
            return None

        legs = structure.get("legs") or []
        names = {str(leg.get("name") or "").lower() for leg in legs if isinstance(leg, dict)}
        leg_count = int(structure.get("leg_count") or len(legs) or 0)
        arms = {
            "ahead": "opposite" in names or "ahead" in names,
            "left": "left" in names,
            "right": "right" in names,
            "ego": "ego" in names,
            "known": bool(legs),
            "arm_count": leg_count,
            "leg_count": leg_count,
            "junction_id": structure.get("junction_id"),
            "source": "junction_legs",
        }
        return arms

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
        curve_features = self._extract_curve_features(center_waypoint)
        if (
            curve_features["is_curve"]
            and not bool(center_waypoint.is_junction)
            and junction_ratio < 0.15
            and candidate_topology_type == "straight_two_way"
        ):
            candidate_topology_type = "curve"
        left_parking, right_parking = self._detect_parking_sides(center_waypoint)
        has_crosswalk_nearby = self._detect_crosswalk_nearby(center, crosswalk_locations)
        has_center_median, center_median_evidence = self._detect_center_median_candidate(
            center_waypoint
        )
        has_highway_shoulder = self._detect_highway_shoulder(center_waypoint)
        same_direction_lane_features = self._estimate_same_direction_lane_features(
            center_waypoint
        )
        distance_to_junction_ahead = self._detect_junction_ahead(center_waypoint)
        distance_to_traffic_light_ahead = self._detect_traffic_light_ahead(center_waypoint)
        junction_branch_dirs = self._classify_junction_branches(center_waypoint)
        physical_junction_arms = self._physical_junction_arms_for_waypoint(
            center_waypoint
        )
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
            "has_center_median_candidate": has_center_median,
            "center_median_evidence": center_median_evidence,
            "has_highway_shoulder": has_highway_shoulder,
            "distance_to_junction_ahead": distance_to_junction_ahead,
            "distance_to_traffic_light_ahead": distance_to_traffic_light_ahead,
            "physical_junction_arms": physical_junction_arms,
            "lane_maneuver_dirs": junction_branch_dirs,
            "junction_branch_dirs": junction_branch_dirs,
            **same_direction_lane_features,
            **curve_features,
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
        curve_features = self._extract_curve_features(center_waypoint)
        if (
            curve_features["is_curve"]
            and not bool(center_waypoint.is_junction)
            and junction_ratio < 0.15
            and candidate_topology_type == "straight_two_way"
        ):
            candidate_topology_type = "curve"
        center = center_waypoint.transform.location
        left_parking, right_parking = self._detect_parking_sides(center_waypoint)
        has_crosswalk_nearby = self._detect_crosswalk_nearby(center, crosswalk_locations)
        has_center_median, center_median_evidence = self._detect_center_median_candidate(
            center_waypoint
        )
        has_highway_shoulder = self._detect_highway_shoulder(center_waypoint)
        same_direction_lane_features = self._estimate_same_direction_lane_features(
            center_waypoint
        )
        distance_to_junction_ahead = self._detect_junction_ahead(center_waypoint)
        distance_to_traffic_light_ahead = self._detect_traffic_light_ahead(center_waypoint)
        junction_branch_dirs = self._classify_junction_branches(center_waypoint)
        physical_junction_arms = self._physical_junction_arms_for_waypoint(
            center_waypoint
        )
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
            "has_center_median_candidate": has_center_median,
            "center_median_evidence": center_median_evidence,
            "has_highway_shoulder": has_highway_shoulder,
            "distance_to_junction_ahead": distance_to_junction_ahead,
            "distance_to_traffic_light_ahead": distance_to_traffic_light_ahead,
            "physical_junction_arms": physical_junction_arms,
            "lane_maneuver_dirs": junction_branch_dirs,
            "junction_branch_dirs": junction_branch_dirs,
            **same_direction_lane_features,
            **curve_features,
        }

    @classmethod
    def _detect_parking_sides(
        cls, center_waypoint: Any, max_lateral_steps: int = 4
    ) -> Tuple[bool, bool]:
        anchor_road_id = getattr(center_waypoint, "road_id", None)
        anchor_yaw = cls._waypoint_yaw(center_waypoint)

        def _same_road(lane: Any) -> bool:
            lane_road_id = getattr(lane, "road_id", anchor_road_id)
            return not (
                anchor_road_id is not None
                and lane_road_id is not None
                and lane_road_id != anchor_road_id
            )

        def _scan_from(lane: Any, side: str, steps: int) -> bool:
            for _step in range(steps):
                if lane is None:
                    break
                if not _same_road(lane):
                    break
                if "parking" in cls._lane_type_text(lane).lower():
                    return True
                lane = cls._get_lateral_lane(lane, side)
            return False

        def _scan(side: str) -> bool:
            lane = cls._get_lateral_lane(center_waypoint, side)
            opposite_side = "right" if side == "left" else "left"
            for step in range(max_lateral_steps):
                if lane is None:
                    break
                if not _same_road(lane):
                    break
                if "parking" in cls._lane_type_text(lane).lower():
                    return True
                if cls._is_opposing_driving_waypoint(lane, anchor_yaw, center_waypoint):
                    return _scan_from(
                        cls._get_lateral_lane(lane, opposite_side),
                        opposite_side,
                        max_lateral_steps - step - 1,
                    )
                lane = cls._get_lateral_lane(lane, side)
            return False

        return _scan("left"), _scan("right")

    @classmethod
    def _is_opposing_driving_waypoint(
        cls,
        lane: Any,
        anchor_yaw: Optional[float],
        anchor_waypoint: Any,
    ) -> bool:
        if not cls._is_driving_waypoint(lane):
            return False
        lane_yaw = cls._waypoint_yaw(lane)
        if lane_yaw is not None and anchor_yaw is not None:
            return (
                _angle_difference_deg(lane_yaw, anchor_yaw)
                >= 180.0 - SAME_DIRECTION_LANE_YAW_TOLERANCE_DEG
            )
        try:
            lane_id = int(getattr(lane, "lane_id"))
            anchor_lane_id = int(getattr(anchor_waypoint, "lane_id"))
            return lane_id * anchor_lane_id < 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _waypoint_yaw(waypoint: Any) -> Optional[float]:
        try:
            return float(waypoint.transform.rotation.yaw)
        except Exception:
            return None

    @staticmethod
    def _detect_highway_shoulder(center_waypoint: Any) -> bool:
        """Return True if an adjacent lane is Stop or Shoulder type (highway indicator)."""
        highway_types = ("stop", "shoulder")
        try:
            for get_lane in (center_waypoint.get_left_lane, center_waypoint.get_right_lane):
                lane = get_lane()
                if lane is not None:
                    lane_type = str(getattr(lane, "lane_type", "") or "").lower()
                    if any(t in lane_type for t in highway_types):
                        return True
        except Exception:
            pass
        return False

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

    @classmethod
    def _detect_center_median_candidate(
        cls,
        center_waypoint: Any,
        max_lateral_steps: int = 8,
    ) -> Tuple[bool, Dict[str, Any]]:
        center_side = cls._center_side_for_lane_id(getattr(center_waypoint, "lane_id", 0))
        sides = [center_side] if center_side else ["left", "right"]
        best_evidence: Dict[str, Any] = {}

        for side in sides:
            lane = cls._get_lateral_lane(center_waypoint, side)
            driving_before_separator = 0
            separator_types: List[str] = []
            inspected: List[Dict[str, Any]] = []
            saw_separator = False

            for step in range(1, max_lateral_steps + 1):
                if lane is None:
                    break

                lane_type_text = cls._lane_type_text(lane)
                lane_info = {
                    "step": step,
                    "road_id": getattr(lane, "road_id", None),
                    "lane_id": getattr(lane, "lane_id", None),
                    "lane_type": lane_type_text,
                }
                inspected.append(lane_info)

                if cls._is_driving_waypoint(lane):
                    if saw_separator:
                        return True, {
                            "side": side,
                            "method": "lateral_lane_chain",
                            "opposite_driving_lane_found": True,
                            "driving_lanes_before_separator": driving_before_separator,
                            "separator_lane_types": separator_types,
                            "inspected_lanes": inspected,
                        }
                    driving_before_separator += 1
                    lane = cls._get_lateral_lane(lane, side)
                    continue

                if cls._is_median_like_lane_type(lane_type_text):
                    saw_separator = True
                    separator_types.append(lane_type_text)
                    lane = cls._get_lateral_lane(lane, side)
                    continue

                if "parking" in lane_type_text.lower():
                    break

                lane = cls._get_lateral_lane(lane, side)

            if (
                saw_separator
                and driving_before_separator >= 1
                and cls._separator_types_indicate_true_median(separator_types)
            ):
                # No opposing driving lane was found, so only an explicit median
                # lane type (not a bare shoulder/curb) can confirm a center median.
                best_evidence = {
                    "side": side,
                    "method": "lateral_lane_chain",
                    "opposite_driving_lane_found": False,
                    "driving_lanes_before_separator": driving_before_separator,
                    "separator_lane_types": separator_types,
                    "inspected_lanes": inspected,
                }
                break

        if best_evidence:
            return True, best_evidence

        return False, {
            "method": "lateral_lane_chain",
            "inspected_center_side": center_side,
        }

    @staticmethod
    def _center_side_for_lane_id(lane_id: Any) -> Optional[str]:
        lane_sign = SceneMapMatcher._lane_id_sign(lane_id)
        if lane_sign < 0:
            return "left"
        if lane_sign > 0:
            return "right"
        return None

    @staticmethod
    def _lane_id_sign(lane_id: Any) -> int:
        try:
            numeric_lane_id = int(lane_id)
        except (TypeError, ValueError):
            return 0
        if numeric_lane_id < 0:
            return -1
        if numeric_lane_id > 0:
            return 1
        return 0

    @staticmethod
    def _get_lateral_lane(waypoint: Any, side: str) -> Any:
        try:
            if side == "left":
                return waypoint.get_left_lane()
            if side == "right":
                return waypoint.get_right_lane()
        except Exception:
            return None
        return None

    @staticmethod
    def _lane_type_text(waypoint: Any) -> str:
        return str(getattr(waypoint, "lane_type", "") or "")

    @staticmethod
    def _is_median_like_lane_type(lane_type_text: str) -> bool:
        lowered = str(lane_type_text or "").lower()
        if not lowered or "driving" in lowered or "parking" in lowered:
            return False
        return any(
            token in lowered
            for token in (
                "median",
                "sidewalk",
                "shoulder",
                "border",
                "restricted",
                "bidirectional",
                "curb",
            )
        )

    @staticmethod
    def _separator_types_indicate_true_median(separator_types: List[Any]) -> bool:
        """True when the separator contains an explicit median lane type.

        Shoulder/curb/sidewalk/border alone are road-edge features and do not, on
        their own, prove a center median.
        """
        return any(
            any(token in str(lane_type or "").lower() for token in STRONG_MEDIAN_LANE_TOKENS)
            for lane_type in (separator_types or [])
        )

    @classmethod
    def _candidate_has_true_center_median(cls, candidate: Dict[str, Any]) -> Optional[bool]:
        """Interpret center-median evidence strictly at scoring time.

        A separator reached at the road edge (shoulder/curb only, with no opposing
        driving lane beyond it) is *not* a center median.  Returns:

        * ``True`` when the candidate genuinely has a center median, or evidence is
          absent and the legacy flag must be trusted for backward compatibility;
        * ``False`` when the candidate has no median, or the evidence proves the
          "median" was only a road-edge separator;
        * ``None`` when the candidate carries no median information at all.
        """
        flagged = candidate.get("has_center_median_candidate")
        if flagged is not True:
            return flagged
        evidence = candidate.get("center_median_evidence")
        if not evidence:
            # Older caches / minimal candidates without evidence: trust the flag.
            return True
        if evidence.get("opposite_driving_lane_found"):
            return True
        if cls._separator_types_indicate_true_median(evidence.get("separator_lane_types")):
            return True
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
            if is_junction:
                # Anchor is inside a junction: road_id count is a reliable degree estimate.
                degree = estimated_junction_degree
            else:
                # Anchor is on an approach road: nearby road_ids bleed in from the junction
                # via waypoint traversal, so heading diversity is the only reliable signal.
                degree = heading_cluster_count
            if degree >= 5:
                return "multi_branch"
            if degree >= 4:
                return "cross_intersection"
            if degree >= 3:
                return "t_junction"
            # heading_cluster_count <= 2 with is_junction=False → approach road, treat as straight.
            return "straight_two_way"
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

    @classmethod
    def _estimate_same_direction_lane_features(
        cls,
        center_waypoint: Any,
        max_lateral_steps: int = 4,
    ) -> Dict[str, Any]:
        anchor_yaw = float(getattr(center_waypoint.transform.rotation, "yaw", 0.0))
        anchor_road_id = getattr(center_waypoint, "road_id", None)
        anchor_lane_id = getattr(center_waypoint, "lane_id", None)
        lane_count = 1 if cls._is_driving_waypoint(center_waypoint) else 0
        inspected: List[Dict[str, Any]] = [
            {
                "side": "anchor",
                "step": 0,
                "road_id": anchor_road_id,
                "lane_id": anchor_lane_id,
                "lane_type": cls._lane_type_text(center_waypoint),
                "yaw_delta_deg": 0.0,
                "accepted": bool(lane_count),
            }
        ]

        for side in ("ahead", "left", "right"):
            lane = cls._get_lateral_lane(center_waypoint, side)
            for step in range(1, max_lateral_steps + 1):
                if lane is None:
                    break
                lane_yaw = float(getattr(lane.transform.rotation, "yaw", anchor_yaw))
                yaw_delta = _angle_difference_deg(lane_yaw, anchor_yaw)
                same_road = getattr(lane, "road_id", None) == anchor_road_id
                driving = cls._is_driving_waypoint(lane)
                accepted = (
                    driving
                    and same_road
                    and yaw_delta <= SAME_DIRECTION_LANE_YAW_TOLERANCE_DEG
                )
                entry = {
                    "side": side,
                    "step": step,
                    "road_id": getattr(lane, "road_id", None),
                    "lane_id": getattr(lane, "lane_id", None),
                    "lane_type": cls._lane_type_text(lane),
                    "yaw_delta_deg": round(yaw_delta, 3),
                    "accepted": accepted,
                }
                # Bake the sibling lane's start geometry so cache-only ego
                # lateral alignment can hop to it without a live CARLA world.
                # Only accepted same-direction driving lanes carry geometry
                # (the anchor's own start is the candidate_lane start); this
                # keeps the size overhead confined to genuine multi-lane roads.
                if accepted:
                    start_geom = cls._lane_start_geometry(lane)
                    if start_geom is not None:
                        entry["start"] = start_geom
                inspected.append(entry)
                if not accepted:
                    break
                lane_count += 1
                lane = cls._get_lateral_lane(lane, side)

        return {
            "same_direction_lane_count": lane_count or 1,
            "has_parallel_same_direction_lanes": lane_count >= 2,
            "same_direction_lane_evidence": {
                "method": "lateral_lane_chain_same_road_yaw",
                "yaw_tolerance_deg": SAME_DIRECTION_LANE_YAW_TOLERANCE_DEG,
                "inspected_lanes": inspected,
            },
        }

    @staticmethod
    def _lane_start_geometry(waypoint: Any) -> Optional[Dict[str, float]]:
        """Compact {x, y, z, yaw} of a lane waypoint for cache-only lateral hops."""
        try:
            loc = waypoint.transform.location
            rot = waypoint.transform.rotation
            return {
                "x": round(float(loc.x), 3),
                "y": round(float(loc.y), 3),
                "z": round(float(loc.z), 3),
                "yaw": round(float(rot.yaw), 3),
            }
        except Exception:
            return None

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

        candidate_is_curve = bool(candidate.get("is_curve"))
        candidate_curve_score = float(candidate.get("curve_score") or 0.0)
        if road_hints.get("curved_road") and not road_hints.get("near_junction"):
            road_diversity = 0.65 + 0.35 * candidate_curve_score if candidate_is_curve else 0.20
        elif road_hints.get("straight_road") and not road_hints.get("near_junction"):
            road_diversity = max(
                0.0,
                1.0
                - max(0, candidate["nearby_road_count"] - 1) / 3.0
                - max(0, candidate["heading_cluster_count"] - 1) / 3.0,
            )
            if candidate_is_curve:
                road_diversity = min(road_diversity, 0.35)
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
        target_forward_lanes = int(signature.get("forward_lane_count") or target_driving_lanes)
        target_opposing_lanes = int(signature.get("opposing_lane_count") or 0)
        junction_ratio = float(candidate.get("junction_waypoint_ratio") or 0.0)
        heading_clusters = int(candidate.get("heading_cluster_count") or 1)
        nearby_roads = int(candidate.get("nearby_road_count") or 1)
        candidate_is_curve = bool(candidate.get("is_curve"))
        candidate_curve_score = float(candidate.get("curve_score") or 0.0)
        candidate_same_direction_lane_count = candidate.get("same_direction_lane_count")
        has_same_direction_field = candidate_same_direction_lane_count is not None

        has_highway_shoulder = bool(candidate.get("has_highway_shoulder"))

        hard_reject = False
        reject_reason = None
        if has_highway_shoulder and target_driving_lanes <= 2:
            hard_reject = True
            reject_reason = (
                "Non-highway scene (driving_lane_count≤2) rejects highway-style candidate "
                "with adjacent Stop/Shoulder lane."
            )
        straight_targets = {"straight_two_way", "straight_road"}
        candidate_straight_like = candidate_topology in straight_targets
        if target_topology in straight_targets:
            if heading_clusters > 2 or nearby_roads > 2 or junction_ratio > 0.12:
                hard_reject = True
                reject_reason = (
                    "Straight-road target rejects complex or junction-like candidate "
                    f"(heading_cluster_count={heading_clusters}, nearby_road_count={nearby_roads}, "
                    f"junction_waypoint_ratio={junction_ratio:.2f})."
                )
            if not hard_reject and candidate_driving_lanes > target_driving_lanes + 1:
                hard_reject = True
                reject_reason = (
                    "Straight-road target rejects overly wide driving-lane candidate "
                    f"(target_driving_lane_count={target_driving_lanes}, "
                    f"candidate_driving_lane_count={candidate_driving_lanes})."
                )
            if not hard_reject and candidate_is_curve and candidate_curve_score >= 0.45:
                hard_reject = True
                reject_reason = (
                    "Straight-road target rejects explicit curved-road candidate "
                    f"(curve_score={candidate_curve_score:.2f})."
                )
        elif target_topology in {"t_junction", "cross_intersection", "multi_branch"}:
            if junction_ratio < 0.08 and not candidate.get("is_junction"):
                hard_reject = True
                reject_reason = "Junction target rejects non-junction candidate."
            elif not candidate.get("is_junction") and heading_clusters < 3:
                hard_reject = True
                reject_reason = (
                    "Junction target rejects approach-road candidate "
                    f"(is_junction=False, heading_cluster_count={heading_clusters})."
                )

        topology_score = 0.0
        if target_topology == "curve":
            if candidate_is_curve:
                topology_score = 0.75 + 0.25 * candidate_curve_score
            elif candidate_topology == "curve":
                # Backward-compatible path for caches generated before explicit curve fields.
                topology_score = 0.80
            else:
                topology_score = 0.10
        elif candidate_topology == target_topology:
            topology_score = 1.0
        elif target_topology in straight_targets and candidate_straight_like:
            # ``straight_road`` is the compact one-way/unknown-direction contract
            # emitted by map_matching; cache candidates historically call the
            # same open-road geometry ``straight_two_way``.
            topology_score = 1.0
        elif target_topology in straight_targets and candidate_topology == "curve":
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
        expects_parallel_same_direction_lanes = (
            target_forward_lanes >= 2 and target_opposing_lanes == 0
        )
        if expects_parallel_same_direction_lanes:
            if has_same_direction_field:
                same_direction_lane_delta = max(
                    0,
                    target_forward_lanes - int(candidate_same_direction_lane_count or 1),
                )
                directional_lane_score = max(
                    0.0,
                    1.0 - 0.45 * same_direction_lane_delta,
                )
            else:
                directional_lane_score = 0.75
        else:
            directional_lane_score = lane_score
        branch_score = max(
            0.0,
            1.0 - abs(candidate_branch_count - target_branch_count) / 4.0,
        )
        if expects_parallel_same_direction_lanes:
            topology_score = (
                0.52 * topology_score
                + 0.20 * branch_score
                + 0.13 * lane_score
                + 0.15 * directional_lane_score
            )
        else:
            topology_score = 0.60 * topology_score + 0.25 * branch_score + 0.15 * lane_score

        # Branch-orientation alignment: which side the junction forks (right turn
        # yes / left turn no). Computed here; applied as a strong multiplicative
        # gate on the TOTAL score below (not a soft topology nudge) so a
        # wrong-side / mirror junction cannot win on cosmetic context.
        branch_dir_score, branch_dir_detail = self._score_branch_directions(
            candidate, signature
        )

        # Junction-anchor gate: a junction-target scene must match a candidate
        # that is actually at (or directly approaching) a real junction. Points
        # merely *classified* junction-like from nearby road/heading diversity
        # (is_junction False, no junction ahead) are false positives that
        # otherwise outscore genuine junctions -- penalise them hard so a real
        # junction (or an approach lane leading into one) wins.
        junction_anchor_quality = 1.0
        if target_topology in {"t_junction", "cross_intersection", "multi_branch"}:
            distance_ahead_raw = candidate.get("distance_to_junction_ahead")
            if candidate.get("is_junction"):
                junction_anchor_quality = 1.0
            elif (
                isinstance(distance_ahead_raw, (int, float))
                and 0.0 <= float(distance_ahead_raw) <= JUNCTION_TARGET_REACH_M
            ):
                junction_anchor_quality = 0.9
            else:
                junction_anchor_quality = 0.30
            topology_score *= junction_anchor_quality

        side_context_score = self._score_side_context(candidate, signature)
        environment_context_score = self._score_environment_context(candidate, signature)
        auxiliary_score = self._score_auxiliary_context(candidate, signature)
        total_score = (
            self.topology_weight * topology_score
            + self.side_context_weight * side_context_score
            + self.auxiliary_weight * auxiliary_score
        )
        median_alignment_adjustment = "neutral"
        if signature.get("has_center_median"):
            candidate_has_median = self._candidate_has_true_center_median(candidate)
            if candidate_has_median is True:
                total_score = min(1.0, total_score + 0.04)
                median_alignment_adjustment = "boost"
            elif candidate_has_median is False:
                total_score *= 0.72
                median_alignment_adjustment = "strong_penalty"

        # Traffic-light alignment gate: when the scene shows a signalized
        # junction, require a junction-target candidate to actually sit at a
        # real traffic light. A no-light point (or one only "junction-like" from
        # heading diversity) is strongly penalised so a genuine signalized
        # junction wins. Applied multiplicatively, mirroring the median gate.
        signal_alignment_adjustment = "neutral"
        candidate_traffic_light_distance_m = None
        if signature.get("has_traffic_light") and target_topology in {
            "t_junction",
            "cross_intersection",
            "multi_branch",
        }:
            candidate_traffic_light_distance_m = self._candidate_traffic_light_distance(
                candidate
            )
            tl_dist = candidate_traffic_light_distance_m
            if tl_dist is not None and tl_dist <= SIGNAL_MATCH_NEAR_M:
                total_score = min(1.0, total_score * SIGNAL_MATCH_NEAR_BOOST)
                signal_alignment_adjustment = "boost"
            elif tl_dist is not None and tl_dist <= SIGNAL_MATCH_MID_M:
                total_score *= SIGNAL_MATCH_MID_FACTOR
                signal_alignment_adjustment = "mid"
            else:
                # No light within reach (or no cached distance): the scene's
                # signal is unaccounted for -> strong penalty.
                total_score *= SIGNAL_MATCH_PENALTY_FACTOR
                signal_alignment_adjustment = "strong_penalty"

        # Branch-direction gate: which side the junction forks is a structural,
        # causal feature for scenario reconstruction. Use the classified
        # severity, not just a soft side-count score, so a right-stem target
        # cannot be won by a left-stem mirror junction with good cosmetic cues.
        branch_alignment_factor = self._branch_alignment_factor(
            branch_dir_score, branch_dir_detail
        )
        total_score *= branch_alignment_factor
        if (
            branch_dir_detail.get("severity") == "mirror_branch_direction"
            and target_topology in {"t_junction", "cross_intersection", "multi_branch"}
        ):
            hard_reject = True
            reject_reason = "Branch direction mirror mismatch."

        distance_to_junction_ahead = candidate.get("distance_to_junction_ahead")
        expects_clear_road_ahead = (
            not signature.get("junction_visible")
            and target_branch_count <= 1
            and target_topology in {"straight_two_way", "straight_road", "curve"}
        )
        junction_ahead_penalty = 1.0
        if (
            expects_clear_road_ahead
            and isinstance(distance_to_junction_ahead, (int, float))
            and distance_to_junction_ahead >= 0
        ):
            distance_ahead = float(distance_to_junction_ahead)
            if distance_ahead < 20.0:
                junction_ahead_penalty = 0.35
            elif distance_ahead < 40.0:
                junction_ahead_penalty = 0.60
            elif distance_ahead < 60.0:
                junction_ahead_penalty = 0.80
            else:  # within JUNCTION_AHEAD_LOOKAHEAD_M but comfortably ahead
                junction_ahead_penalty = 0.92
            total_score *= junction_ahead_penalty

        curve_direction_factor, curve_direction_detail = self._score_curve_direction_alignment(
            candidate, signature, target_topology
        )
        total_score *= curve_direction_factor

        junction_distance_factor, junction_distance_detail = (
            self._score_ego_junction_distance_alignment(
                candidate, signature, target_topology
            )
        )
        total_score *= junction_distance_factor

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
            "environment_context_score": environment_context_score,
            "auxiliary_element_score": auxiliary_score,
            "target_topology_type": target_topology,
            "candidate_topology_type": candidate_topology,
            "candidate_is_curve": candidate_is_curve,
            "candidate_curve_score": candidate_curve_score,
            "candidate_curve_direction": candidate.get("curve_direction"),
            "candidate_curve_yaw_delta_deg": candidate.get("curve_yaw_delta_deg"),
            "target_curve_direction": signature.get("curve_direction"),
            "curve_direction_factor": curve_direction_factor,
            "curve_direction_detail": curve_direction_detail,
            "target_driving_lane_count": target_driving_lanes,
            "candidate_driving_lane_count": candidate_driving_lanes,
            "target_forward_lane_count": target_forward_lanes,
            "target_opposing_lane_count": target_opposing_lanes,
            "target_ego_lane_from_right": signature.get("ego_lane_from_right"),
            "candidate_same_direction_lane_count": candidate_same_direction_lane_count,
            "directional_lane_score": directional_lane_score,
            "expects_parallel_same_direction_lanes": expects_parallel_same_direction_lanes,
            "target_branch_count": target_branch_count,
            "candidate_branch_count": candidate_branch_count,
            "junction_anchor_quality": junction_anchor_quality,
            "branch_direction_score": branch_dir_score,
            "branch_alignment_factor": branch_alignment_factor,
            "branch_direction_detail": branch_dir_detail,
            "target_has_center_median": bool(signature.get("has_center_median")),
            "candidate_has_center_median": candidate.get("has_center_median_candidate"),
            "median_alignment_adjustment": median_alignment_adjustment,
            "target_has_traffic_light": bool(signature.get("has_traffic_light")),
            "candidate_traffic_light_distance_m": candidate_traffic_light_distance_m,
            "signal_alignment_adjustment": signal_alignment_adjustment,
            "target_ego_to_junction_distance_m": signature.get("ego_to_junction_distance_m"),
            "junction_distance_factor": junction_distance_factor,
            "junction_distance_detail": junction_distance_detail,
            "distance_to_junction_ahead": distance_to_junction_ahead,
            "junction_ahead_penalty": junction_ahead_penalty,
            "hard_reject": hard_reject,
            "blacklisted": False,
            "reject_reason": reject_reason,
        }

    @staticmethod
    def _score_curve_direction_alignment(
        candidate: Dict[str, Any],
        signature: Dict[str, Any],
        target_topology: str,
    ) -> Tuple[float, Dict[str, Any]]:
        target_direction = str(signature.get("curve_direction") or "unknown").lower()
        if target_topology != "curve" or target_direction not in {"left", "right"}:
            return 1.0, {"status": "not_applicable"}
        candidate_direction = str(candidate.get("curve_direction") or "unknown").lower()
        if candidate_direction == target_direction:
            return 1.05, {
                "status": "matched",
                "target": target_direction,
                "candidate": candidate_direction,
            }
        if candidate_direction in {"left", "right"}:
            return 0.65, {
                "status": "mismatch",
                "target": target_direction,
                "candidate": candidate_direction,
            }
        return 0.85, {
            "status": "candidate_unknown",
            "target": target_direction,
            "candidate": candidate_direction,
        }

    @staticmethod
    def _score_ego_junction_distance_alignment(
        candidate: Dict[str, Any],
        signature: Dict[str, Any],
        target_topology: str,
    ) -> Tuple[float, Dict[str, Any]]:
        if target_topology not in {"t_junction", "cross_intersection", "multi_branch"}:
            return 1.0, {"status": "not_applicable"}
        target_distance = signature.get("ego_to_junction_distance_m")
        candidate_distance = candidate.get("distance_to_junction_ahead")
        if not isinstance(target_distance, (int, float)):
            return 1.0, {"status": "target_missing"}
        if not isinstance(candidate_distance, (int, float)) or candidate_distance < 0:
            return 0.90, {
                "status": "candidate_missing",
                "target_m": float(target_distance),
            }
        error_m = abs(float(candidate_distance) - float(target_distance))
        if error_m <= 10.0:
            factor = 1.05
            status = "matched"
        elif error_m <= 25.0:
            factor = 0.90
            status = "near"
        else:
            factor = 0.75
            status = "far"
        return factor, {
            "status": status,
            "target_m": float(target_distance),
            "candidate_m": float(candidate_distance),
            "error_m": round(error_m, 3),
        }

    @staticmethod
    def _candidate_traffic_light_distance(candidate: Dict[str, Any]) -> Optional[float]:
        """Distance (m) to the traffic light governing ego's forward approach.

        Prefers the DIRECTIONAL ``distance_to_traffic_light_ahead`` (a light
        found by following the lane forward), so a light behind ego or on a
        crossing street does NOT satisfy the gate. When that field is present
        but None, there is genuinely no light ahead -> treated as "no signal".
        Falls back to the omnidirectional ``environment_context.nearest_m``
        distance only for caches built before the directional field existed.
        """
        if "distance_to_traffic_light_ahead" in candidate:
            ahead = candidate.get("distance_to_traffic_light_ahead")
            return float(ahead) if isinstance(ahead, (int, float)) else None
        env_ctx = candidate.get("environment_context") or {}
        nearest = env_ctx.get("nearest_m") or {}
        value = nearest.get("TrafficLight")
        if isinstance(value, (int, float)):
            return float(value)
        return None

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

        if target_topology in {"straight_two_way", "straight_road"}:
            if candidate_topology not in {"straight_two_way", "straight_road"}:
                score *= 0.80
            complexity_delta = max(0, heading_clusters - 2) + max(0, nearby_roads - 2)
            if junction_ratio > 0.12:
                complexity_delta += int(math.ceil(junction_ratio * 4.0))
            score *= max(0.45, 1.0 - 0.04 * complexity_delta)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _branch_alignment_factor(
        branch_dir_score: float, branch_dir_detail: Optional[Dict[str, Any]] = None
    ) -> float:
        """Map a branch-direction sub-score to a TOTAL-score multiplier.

        The detail severity is authoritative when available. ``branch_dir_score``
        is kept for older callers/tests and for defensive fallback.
        """
        detail = branch_dir_detail or {}
        severity = detail.get("severity")
        if severity == "mirror_branch_direction":
            return BRANCH_MATCH_MIRROR_FACTOR
        if severity == "missing_required_branch":
            return BRANCH_MATCH_MISSING_FACTOR
        if severity == "extra_forbidden_branch":
            return BRANCH_MATCH_EXTRA_FACTOR
        if severity == "candidate_missing":
            return BRANCH_MATCH_CANDIDATE_MISSING_FACTOR
        if severity in {"match", "target_unknown"}:
            return 1.0
        score = float(branch_dir_score)
        if score >= 0.99:
            return 1.0
        if score >= 0.4:
            return BRANCH_MATCH_EXTRA_FACTOR
        if score >= 0.2:
            return BRANCH_MATCH_MISSING_FACTOR
        return BRANCH_MATCH_MIRROR_FACTOR

    @staticmethod
    def _score_branch_directions(
        candidate: Dict[str, Any], signature: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Score how well physical candidate junction arms match the target.

        Neutral (1.0) when the target branch sides are unknown. When the target
        is known but an older cache lacks ``physical_junction_arms``, return a
        non-rejecting penalty. The older ``junction_branch_dirs`` field is
        lane-level maneuver reachability and is intentionally not used here.
        """
        target = signature.get("junction_branches") or {}
        if not target.get("known"):
            return 1.0, {
                "status": "target_unknown",
                "severity": "target_unknown",
                "errors": [],
                "factor": 1.0,
            }
        candidate_dirs = candidate.get("physical_junction_arms")
        if isinstance(candidate_dirs, dict) and candidate_dirs.get("known") is False:
            candidate_dirs = None
        if not isinstance(candidate_dirs, dict):
            return BRANCH_MATCH_CANDIDATE_MISSING_FACTOR, {
                "status": "candidate_missing",
                "severity": "candidate_missing",
                "errors": ["physical_junction_arms_missing"],
                "factor": BRANCH_MATCH_CANDIDATE_MISSING_FACTOR,
                "legacy_lane_maneuver_dirs": candidate.get("junction_branch_dirs"),
            }

        missing_required = []
        extra_forbidden = []
        for side in ("ahead", "left", "right"):
            target_has = bool(target.get(side))
            candidate_has = bool(candidate_dirs.get(side))
            if target_has and not candidate_has:
                missing_required.append(side)
            elif candidate_has and not target_has:
                extra_forbidden.append(side)

        errors = [
            f"missing_required_{side}_branch" for side in missing_required
        ] + [
            f"extra_forbidden_{side}_branch" for side in extra_forbidden
        ]
        if missing_required and extra_forbidden:
            severity = "mirror_branch_direction"
            factor = BRANCH_MATCH_MIRROR_FACTOR
        elif missing_required:
            severity = "missing_required_branch"
            factor = BRANCH_MATCH_MISSING_FACTOR
        elif extra_forbidden:
            severity = "extra_forbidden_branch"
            factor = BRANCH_MATCH_EXTRA_FACTOR
        else:
            severity = "match"
            factor = 1.0
        return factor, {
            "status": "compared",
            "target": {k: target.get(k) for k in ("ahead", "left", "right")},
            "candidate": {
                k: candidate_dirs.get(k)
                for k in ("ahead", "left", "right", "uturn", "branch_count")
            },
            "errors": errors,
            "severity": severity,
            "factor": factor,
            "score": round(factor, 3),
        }

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
        score += SceneMapMatcher._score_environment_context(candidate, signature)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_environment_context(candidate: Dict[str, Any], signature: Dict[str, Any]) -> float:
        target_env = signature.get("environment_context") or {}
        candidate_env = candidate.get("environment_context") or {}
        if not target_env or not candidate_env:
            return 0.0

        counts = candidate_env.get("counts") or {}
        urban_score = float(candidate_env.get("urban_score") or 0.0)
        natural_score = float(candidate_env.get("natural_score") or 0.0)
        environment_class = str(candidate_env.get("environment_class") or "unknown")
        water_nearby = bool(candidate_env.get("water_nearby"))
        buildings_nearby = bool(candidate_env.get("buildings_nearby"))
        sidewalks_nearby = bool(candidate_env.get("sidewalks_nearby"))
        traffic_control_nearby = bool(candidate_env.get("traffic_control_nearby"))
        terrain_count = int(counts.get("Terrain") or 0)
        vegetation_count = int(counts.get("Vegetation") or 0)
        building_count = int(counts.get("Buildings") or 0)

        delta = 0.0
        if target_env.get("expects_urban"):
            delta += 0.10 * urban_score
            if target_env.get("expects_buildings"):
                delta += 0.06 if buildings_nearby else -0.05
            if target_env.get("expects_sidewalks"):
                delta += 0.05 if sidewalks_nearby else -0.03
            if traffic_control_nearby:
                delta += 0.03
            if natural_score > urban_score:
                delta -= min(0.12, 0.08 * (natural_score - urban_score + 0.25))
            if target_env.get("avoid_water") and water_nearby:
                delta -= 0.12
            if target_env.get("avoid_terrain_dominant") and terrain_count + vegetation_count >= 5:
                delta -= 0.06
            if environment_class == "natural_like":
                delta -= 0.06

        if target_env.get("expects_natural"):
            delta += 0.08 * natural_score
            if environment_class == "natural_like":
                delta += 0.04
            if not target_env.get("expects_urban") and building_count >= 8:
                delta -= 0.04
            if target_env.get("expects_water") and water_nearby:
                delta += 0.05

        return max(-0.20, min(0.20, delta))

    @staticmethod
    def _environment_debug_summary(
        environment_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(environment_context, dict) or not environment_context:
            return {}
        return {
            "environment_class": environment_context.get("environment_class"),
            "urban_score": environment_context.get("urban_score"),
            "natural_score": environment_context.get("natural_score"),
            "water_nearby": environment_context.get("water_nearby"),
            "buildings_nearby": environment_context.get("buildings_nearby"),
            "sidewalks_nearby": environment_context.get("sidewalks_nearby"),
            "traffic_control_nearby": environment_context.get("traffic_control_nearby"),
        }

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
        elif candidate.get("has_crosswalk_nearby"):
            # Scene has no crosswalk: penalize matching onto a crosswalk location.
            score -= 0.20

        if signature.get("has_traffic_light"):
            env_ctx = candidate.get("environment_context") or {}
            nearest_tl = (env_ctx.get("nearest_m") or {}).get("TrafficLight")
            if nearest_tl is not None:
                if nearest_tl <= 30:
                    score += 0.25
                elif nearest_tl <= 60:
                    score += 0.12
                elif nearest_tl <= 100:
                    pass  # neutral
                else:
                    score -= 0.20
            # No cached TrafficLight distance means no signal was found within
            # the cache radius: the scene's signal is unaccounted for, so do NOT
            # reward mere junction proximity here (that wrongly treats a no-light
            # junction as plausibly signalized). The signal-alignment gate in
            # _score_topology_candidate_features handles the strong penalty.
        else:
            # Scene has no traffic light: penalize matching next to one.
            env_ctx = candidate.get("environment_context") or {}
            nearest_tl = (env_ctx.get("nearest_m") or {}).get("TrafficLight")
            if nearest_tl is not None:
                if nearest_tl <= 30:
                    score -= 0.20
                elif nearest_tl <= 60:
                    score -= 0.10

        if signature.get("has_traffic_sign"):
            score += 0.05
        if signature.get("has_center_median"):
            candidate_has_median = SceneMapMatcher._candidate_has_true_center_median(candidate)
            if candidate_has_median is True:
                score += 0.30
            elif candidate_has_median is False:
                score -= 0.30
            else:
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
                "candidate_topology_type": candidate.get("candidate_topology_type"),
                "has_center_median_candidate": candidate.get("has_center_median_candidate"),
                "center_median_evidence": candidate.get("center_median_evidence"),
                "same_direction_lane_count": candidate.get("same_direction_lane_count"),
                "has_parallel_same_direction_lanes": candidate.get(
                    "has_parallel_same_direction_lanes"
                ),
                "same_direction_lane_evidence": candidate.get(
                    "same_direction_lane_evidence"
                ),
                "is_curve": candidate.get("is_curve"),
                "curve_yaw_delta_deg": candidate.get("curve_yaw_delta_deg"),
                "curve_abs_yaw_delta_deg": candidate.get("curve_abs_yaw_delta_deg"),
                "curve_direction": candidate.get("curve_direction"),
                "curve_score": candidate.get("curve_score"),
                "curve_sample_distance_m": candidate.get("curve_sample_distance_m"),
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
