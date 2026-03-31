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

    def _match_to_carla(self, scene_features: Dict[str, Any]) -> Dict[str, Any]:
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
            world_map, spawn_points, scene_features
        )
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
                "reason": "No CARLA candidates were scored in the current world.",
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
                "reason": "No CARLA candidates survived the layout refinement step.",
            }

        best_candidate = max(
            refined_candidates, key=lambda payload: payload["refined_score"]
        )
        return {
            "status": "matched",
            "world_name": world_name,
            "candidate_summary": candidate_summary,
            "best_match": best_candidate["match_record"],
            "projected_layout": best_candidate["projected_layout"],
            "reason": None,
        }

    def _score_carla_candidates(
        self,
        world_map: Any,
        spawn_points: List[Any],
        scene_features: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if self._is_large_map(world_map, spawn_points):
            return self._score_large_map_candidates(
                world_map, spawn_points, scene_features
            )

        sampled_waypoints = world_map.generate_waypoints(self.sample_step)
        scored_candidates = []
        for waypoint in sampled_waypoints:
            candidate = self._extract_candidate_features(sampled_waypoints, waypoint)
            score, details = self._score_candidate_features(candidate, scene_features)
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "search_strategy": "full_waypoints",
                    "location": {
                        "x": waypoint.transform.location.x,
                        "y": waypoint.transform.location.y,
                        "z": waypoint.transform.location.z,
                    },
                    "yaw": waypoint.transform.rotation.yaw,
                    **candidate,
                }
            )

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        return scored_candidates[: max(self.coarse_top_k, 20)]

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
            scored_candidates.append(
                {
                    "score": score,
                    "score_details": details,
                    "search_strategy": "spawn_points",
                    "spawn_point_index": spawn_index,
                    "location": {
                        "x": waypoint.transform.location.x,
                        "y": waypoint.transform.location.y,
                        "z": waypoint.transform.location.z,
                    },
                    "yaw": waypoint.transform.rotation.yaw,
                    **candidate,
                }
            )

        scored_candidates.sort(key=lambda item: item["score"], reverse=True)
        return scored_candidates[: max(self.coarse_top_k, 20)]

    def _extract_candidate_features(
        self, sampled_waypoints: List[Any], center_waypoint: Any
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
            "estimated_junction_degree": max(len(road_ids), heading_cluster_count),
        }

    def _extract_local_candidate_features(
        self, world_map: Any, center_waypoint: Any
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
            "estimated_junction_degree": max(len(road_ids), heading_cluster_count),
        }

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
            waypoint.lane_id for waypoint in nearby_waypoints if waypoint.road_id == road_id
        }
        return len(lane_ids) or 1

    def _score_candidate_features(
        self, candidate: Dict[str, Any], scene_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        layout_summary = scene_features["layout_summary"]
        semantic_hints = scene_features["semantic_hints"]
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
        anchor_name = scene_features["layout_summary"]["anchor_entity"]["name"]

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
