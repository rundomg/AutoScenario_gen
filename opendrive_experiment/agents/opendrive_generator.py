import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

from agents.net_generator import NetGenerator
from agents.task_agent import OPENAI_TIMEOUT
from tools.utils import extract_text_section, read_file, write_to_file


class OpendriveGenerator(NetGenerator):
    """Generate OpenDRIVE maps either via SUMO transition output or DSL rendering."""

    DSL_SYSTEM_PROMPT = """
    You generate a minimal road DSL that can be deterministically rendered to OpenDRIVE.
    The road description comes from a single traffic image and should stay conservative.

    Return this exact format:
    ## Description
    Briefly restate only supported road facts.
    ## Reasoning
    Explain only the minimum assumptions required to make the road valid.
    ## OpenDRIVE DSL
    ```json
    {
      "roads": [],
      "junctions": [],
      "connections": [],
      "metadata": {
        "assumptions": []
      }
    }
    ```

    DSL requirements:
    - `roads[]` is required and must not be empty.
    - Every road must include `id`, `length_m`, `plan_view`, `lane_sections`, `successor`, and `predecessor`.
    - `plan_view[]` items must include `s`, `x`, `y`, `hdg`, `length`, and `geometry`.
    - `geometry` may be `line` or `arc`. `arc` additionally requires `curvature`.
    - `lane_sections[]` items must include `s`, `left`, and `right`.
    - `left` and `right` must be arrays of lane descriptors with `type` and `width`.
    - Lane `type` may include `driving`, `parking`, and `shoulder`; preserve explicit parking-lane counts from the input instead of collapsing them into assumptions.
    - If the input road network includes `left_parking_lane_count` or `right_parking_lane_count`, emit explicit `parking` lanes in the matching lane section.
    - Add a conservative outer `shoulder` buffer beyond the outermost traffic/parking lane so roadside objects are not forced off the mesh.
    - Order each side's lanes from the road center outward; for example `driving` then `parking` on the same side.
    - Keep coordinates simple and non-negative when possible.
    - If details are unclear, choose the simplest valid structure and note it in `metadata.assumptions`.
    """

    def __init__(
        self,
        save_dir,
        generation_stage: str = "stage1",
        netconvert_path: str = "netconvert",
    ) -> None:
        super().__init__(save_dir)
        self.generation_stage = generation_stage.lower()
        self.netconvert_path = netconvert_path

    def refine_request(self, user_request, add_info=None):
        if self.generation_stage == "stage2":
            validation_error = (add_info or {}).get("validation_error")
            request = f"{self.DSL_SYSTEM_PROMPT}\nRoad description:\n{user_request}"
            if validation_error:
                request += (
                    "\n\nPrevious DSL output failed validation. Fix the issue below and regenerate.\n"
                    f"{validation_error}"
                )
            return request
        return super().refine_request(user_request, add_info)

    def call_agent(self, user_request, scenario_id, add_info=None):
        if self.generation_stage == "stage2":
            return self._call_stage2(user_request, scenario_id, add_info or {})
        return self._call_stage1(user_request, scenario_id, add_info or {})

    def _call_stage1(self, user_request, scenario_id, add_info):
        output_fn = add_info.get(
            "output_fn", os.path.join(self.save_dir, f"{scenario_id}_net.txt")
        )
        success_bool, generation_cnt = super().call_agent(
            user_request, scenario_id, {"output_fn": output_fn}
        )
        if not success_bool:
            raise RuntimeError("Stage1 OpenDRIVE generation did not produce a valid SUMO net.")

        net_path = os.path.join(self.save_dir, f"{scenario_id}.net.xml")
        xodr_path = os.path.join(self.save_dir, f"{scenario_id}.xodr")
        self.convert_sumo_net_to_opendrive(net_path, xodr_path)

        artifact = {
            "stage": "stage1",
            "attempts": generation_cnt,
            "xodr_path": xodr_path,
            "debug_net_path": net_path,
            "node_path": os.path.join(self.save_dir, f"{scenario_id}.nod.xml"),
            "edge_path": os.path.join(self.save_dir, f"{scenario_id}.edg.xml"),
            "net_prompt_path": output_fn,
            "summary": self._build_stage1_summary(user_request, net_path),
        }
        self._write_context_file(scenario_id, artifact)
        return artifact

    def _call_stage2(self, user_request, scenario_id, add_info):
        output_fn = add_info.get(
            "output_fn", os.path.join(self.save_dir, f"{scenario_id}_xodr.txt")
        )
        request_valid_result = True
        generation_cnt = 0
        validation_error = None

        while request_valid_result:
            request_info = {
                "output_fn": output_fn,
                "request_timeout": max(OPENAI_TIMEOUT, 180),
                "request_retries": 2,
                "request_label": "OpenDRIVE DSL generation",
            }
            if validation_error:
                request_info["validation_error"] = validation_error

            self.send_request(user_request, add_info=request_info)
            artifact, validation_error = self.extract_stage2_artifacts(
                scenario_id=scenario_id,
                output_fn=output_fn,
            )
            request_valid_result = artifact is None
            generation_cnt += 1

            if request_valid_result and generation_cnt >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "OpenDRIVE DSL generation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {validation_error}"
                )
            if request_valid_result and generation_cnt > 1:
                print(
                    "----------------------------regenerating-------------------generation_cnt--",
                    generation_cnt,
                )

        artifact["attempts"] = generation_cnt
        self._write_context_file(scenario_id, artifact)
        return artifact

    def extract_stage2_artifacts(
        self, scenario_id: str, output_fn: str
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        response = read_file(output_fn)
        decision_content = extract_text_section(
            response, r"## OpenDRIVE DSL\s+(.*?)(?=\s+##|$)"
        )
        if decision_content is None:
            return None, "Missing ## OpenDRIVE DSL section."

        dsl_text = extract_text_section(decision_content, r"```json\s+(.*?)\s+```")
        if dsl_text is None:
            return None, "Missing JSON fenced block in ## OpenDRIVE DSL."

        try:
            dsl = json.loads(dsl_text)
        except json.JSONDecodeError as exc:
            return None, f"OpenDRIVE DSL JSON is invalid: {exc.msg}."

        dsl = self._sanitize_stage2_dsl(dsl)

        validation_error = self.validate_road_dsl(dsl)
        if not validation_error:
            validation_error = self.validate_road_connectivity(dsl)
        if not validation_error:
            validation_error = self.validate_road_geometry_continuity(dsl)
        if validation_error:
            return None, validation_error

        dsl = self._ensure_outer_shoulder_buffers(dsl)

        dsl_path = os.path.join(self.save_dir, f"{scenario_id}.road_dsl.json")
        xodr_path = os.path.join(self.save_dir, f"{scenario_id}.xodr")
        debug_net_path = os.path.join(self.save_dir, f"{scenario_id}_debug.net.xml")

        write_to_file(dsl_path, json.dumps(dsl, indent=2, sort_keys=True))
        self.render_road_dsl_to_xodr(dsl, xodr_path)
        validation_error = self.validate_xodr_output(xodr_path, debug_net_path)
        if validation_error:
            return None, validation_error

        return (
            {
                "stage": "stage2",
                "xodr_path": xodr_path,
                "debug_net_path": debug_net_path,
                "dsl_path": dsl_path,
                "summary": self.summarize_road_dsl(dsl),
                "metadata": dsl.get("metadata", {}),
            },
            None,
        )

    @staticmethod
    def _normalize_road_link_payload(payload: Any) -> Optional[Dict[str, str]]:
        if not isinstance(payload, dict):
            return None

        element_type = payload.get("elementType", payload.get("type"))
        element_id = payload.get("elementId", payload.get("id"))
        contact_point = str(payload.get("contactPoint", "start"))

        element_type_text = str(element_type).strip().lower() if element_type is not None else ""
        element_id_text = str(element_id).strip() if element_id is not None else ""

        if element_type_text in {"", "none", "null"}:
            return None
        if element_id_text in {"", "none", "null", "-1"}:
            return None
        if element_type_text not in {"road", "junction"}:
            element_type_text = "road"

        return {
            "elementType": element_type_text,
            "elementId": element_id_text,
            "contactPoint": contact_point,
        }

    @classmethod
    def _sanitize_stage2_dsl(cls, dsl: Dict[str, Any]) -> Dict[str, Any]:
        roads = dsl.get("roads")
        if not isinstance(roads, list):
            return dsl

        for road in roads:
            if not isinstance(road, dict):
                continue
            for link_name in ("predecessor", "successor"):
                road[link_name] = cls._normalize_road_link_payload(road.get(link_name))

        junctions = dsl.get("junctions")
        if not isinstance(junctions, list):
            dsl["junctions"] = []
            junctions = dsl["junctions"]

        referenced_junction_ids = set()
        for road in roads:
            if not isinstance(road, dict):
                continue
            for link_name in ("predecessor", "successor"):
                payload = road.get(link_name)
                if isinstance(payload, dict) and payload.get("elementType") == "junction":
                    referenced_junction_ids.add(str(payload.get("elementId")))

        valid_junction_ids = set()
        for junction in junctions:
            if not isinstance(junction, dict):
                continue
            junction_connections = junction.get("connections")
            if not isinstance(junction_connections, list):
                junction_connections = []
            junction["connections"] = [
                connection for connection in junction_connections if isinstance(connection, dict)
            ]
            if junction["connections"]:
                valid_junction_ids.add(str(junction.get("id", "")))

        if referenced_junction_ids:
            for road in roads:
                if not isinstance(road, dict):
                    continue
                for link_name in ("predecessor", "successor"):
                    payload = road.get(link_name)
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("elementType") != "junction":
                        continue
                    if str(payload.get("elementId")) not in valid_junction_ids:
                        road[link_name] = None

            dsl["junctions"] = [
                junction
                for junction in junctions
                if isinstance(junction, dict)
                and str(junction.get("id", "")) in valid_junction_ids
            ]

        connections = dsl.get("connections")
        if not isinstance(connections, list):
            dsl["connections"] = []

        return dsl

    @staticmethod
    def _ensure_outer_shoulder_buffers(
        dsl: Dict[str, Any],
        shoulder_width: float = 3.0,
    ) -> Dict[str, Any]:
        roads = dsl.get("roads")
        if not isinstance(roads, list):
            return dsl

        assumptions = dsl.setdefault("metadata", {}).setdefault("assumptions", [])
        assumption_text = (
            "Added outer shoulder buffers beyond the outermost lanes to provide conservative roadside placement surface."
        )
        added_or_widened_shoulder = False

        for road in roads:
            lane_sections = road.get("lane_sections")
            if not isinstance(lane_sections, list):
                continue
            for lane_section in lane_sections:
                if not isinstance(lane_section, dict):
                    continue
                for side_name in ("left", "right"):
                    lanes = lane_section.get(side_name)
                    if not isinstance(lanes, list) or not lanes:
                        continue
                    outermost = lanes[-1]
                    if not isinstance(outermost, dict):
                        continue
                    if str(outermost.get("type") or "").strip().lower() == "shoulder":
                        existing_width = float(outermost.get("width", 0.0) or 0.0)
                        if existing_width < shoulder_width:
                            outermost["width"] = shoulder_width
                            added_or_widened_shoulder = True
                        continue
                    lanes.append(
                        {
                            "type": "shoulder",
                            "width": shoulder_width,
                            "road_mark_type": "solid",
                            "road_mark_color": "white",
                            "lane_change": "none",
                        }
                    )
                    added_or_widened_shoulder = True

        if added_or_widened_shoulder and assumption_text not in assumptions:
            assumptions.append(assumption_text)
        return dsl

    def convert_sumo_net_to_opendrive(self, net_path: str, xodr_path: str):
        result = subprocess.run(
            [
                self.netconvert_path,
                "--sumo-net-file",
                net_path,
                "--opendrive-output",
                xodr_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Failed to convert SUMO net to OpenDRIVE: {stderr}")

    def validate_xodr_output(self, xodr_path: str, debug_net_path: str) -> Optional[str]:
        result = subprocess.run(
            [
                self.netconvert_path,
                "--opendrive-files",
                xodr_path,
                "--output-file",
                debug_net_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return None
        stderr = (result.stderr or result.stdout).strip()
        if len(stderr) > 1200:
            stderr = f"{stderr[:1200]}..."
        return f"Rendered OpenDRIVE failed netconvert validation: {stderr}"

    @staticmethod
    def validate_road_dsl(dsl: Dict[str, Any]) -> Optional[str]:
        if not isinstance(dsl, dict):
            return "OpenDRIVE DSL root must be a dictionary."
        roads = dsl.get("roads")
        if not isinstance(roads, list) or not roads:
            return "`roads` must be a non-empty array."
        if not isinstance(dsl.get("junctions", []), list):
            return "`junctions` must be an array."
        if not isinstance(dsl.get("connections", []), list):
            return "`connections` must be an array."

        metadata = dsl.get("metadata")
        if not isinstance(metadata, dict):
            return "`metadata` must be a dictionary."
        if not isinstance(metadata.get("assumptions", []), list):
            return "`metadata.assumptions` must be an array."

        for road in roads:
            if not isinstance(road, dict):
                return "Each road must be a dictionary."
            for key in (
                "id",
                "length_m",
                "plan_view",
                "lane_sections",
                "successor",
                "predecessor",
            ):
                if key not in road:
                    return f"Road entry is missing required key: {key}"
            if not isinstance(road["plan_view"], list) or not road["plan_view"]:
                return f"Road {road['id']} must contain at least one plan_view segment."
            if not isinstance(road["lane_sections"], list) or not road["lane_sections"]:
                return f"Road {road['id']} must contain at least one lane section."

            for segment in road["plan_view"]:
                if not isinstance(segment, dict):
                    return f"Road {road['id']} has a non-dictionary plan_view segment."
                for key in ("s", "x", "y", "hdg", "length", "geometry"):
                    if key not in segment:
                        return (
                            f"Road {road['id']} plan_view segment is missing required key: {key}"
                        )
                if segment["geometry"] == "arc" and "curvature" not in segment:
                    return f"Road {road['id']} arc segment is missing curvature."

            for lane_section in road["lane_sections"]:
                if not isinstance(lane_section, dict):
                    return f"Road {road['id']} has a non-dictionary lane section."
                for key in ("s", "left", "right"):
                    if key not in lane_section:
                        return (
                            f"Road {road['id']} lane section is missing required key: {key}"
                        )
                for side_name in ("left", "right"):
                    if not isinstance(lane_section[side_name], list):
                        return (
                            f"Road {road['id']} lane section `{side_name}` must be an array."
                        )
                    for lane in lane_section[side_name]:
                        if not isinstance(lane, dict):
                            return f"Road {road['id']} has a non-dictionary lane entry."
                        for key in ("type", "width"):
                            if key not in lane:
                                return (
                                    f"Road {road['id']} lane entry is missing required key: {key}"
                                )
        return None

    @staticmethod
    def validate_road_connectivity(dsl: Dict[str, Any]) -> Optional[str]:
        """Check graph-level road connectivity: duplicate IDs, reference integrity, bidirectionality.

        Runs after validate_road_dsl(), before rendering to XML.
        Returns an error string if invalid, else None.
        """
        roads = dsl.get("roads", [])
        junctions = dsl.get("junctions", [])

        road_ids = [str(r["id"]) for r in roads]
        junction_ids = [str(j["id"]) for j in junctions]
        road_id_set = set(road_ids)
        junction_id_set = set(junction_ids)

        errors = []

        # P2 – duplicate IDs
        seen: set = set()
        dupes = []
        for rid in road_ids:
            if rid in seen:
                dupes.append(rid)
            seen.add(rid)
        if dupes:
            errors.append(f"Duplicate road IDs detected: {dupes}.")

        seen = set()
        dupes = []
        for jid in junction_ids:
            if jid in seen:
                dupes.append(jid)
            seen.add(jid)
        if dupes:
            errors.append(f"Duplicate junction IDs detected: {dupes}.")

        road_by_id: Dict[str, Any] = {str(r["id"]): r for r in roads}

        # Build which road IDs each junction references in its connections
        junction_referenced_roads: Dict[str, set] = {}
        for junc in junctions:
            jid = str(junc["id"])
            refs: set = set()
            for conn in junc.get("connections", []):
                inc = str(conn.get("incomingRoad", ""))
                con = str(conn.get("connectingRoad", ""))
                # P0 – junction connection references must exist as roads
                if inc and inc not in road_id_set:
                    errors.append(
                        f"Junction '{jid}' connection references non-existent incomingRoad '{inc}'."
                    )
                if con and con not in road_id_set:
                    errors.append(
                        f"Junction '{jid}' connection references non-existent connectingRoad '{con}'."
                    )
                refs.add(inc)
                refs.add(con)
            junction_referenced_roads[jid] = refs

        # P0 + P1 – road predecessor/successor reference integrity and bidirectionality
        for road in roads:
            rid = str(road["id"])
            for link_name in ("predecessor", "successor"):
                link = road.get(link_name)
                if not link:
                    continue
                elem_type = str(link.get("elementType", "road"))
                elem_id = str(link.get("elementId", ""))

                if elem_type == "road":
                    # P0: referenced road must exist
                    if elem_id not in road_id_set:
                        errors.append(
                            f"Road '{rid}' {link_name} references non-existent road '{elem_id}'."
                        )
                        continue
                    # P1: if A.successor → road B, then B.predecessor must point back to A
                    reverse = "predecessor" if link_name == "successor" else "successor"
                    other = road_by_id[elem_id]
                    other_link = other.get(reverse)
                    if other_link is None:
                        errors.append(
                            f"Road '{rid}' {link_name} → road '{elem_id}', "
                            f"but road '{elem_id}' has no {reverse} pointing back to '{rid}'."
                        )
                    elif str(other_link.get("elementId", "")) != rid:
                        back = str(other_link.get("elementId", ""))
                        errors.append(
                            f"Road '{rid}' {link_name} → road '{elem_id}', "
                            f"but road '{elem_id}'.{reverse} points to '{back}' not '{rid}'."
                        )

                elif elem_type == "junction":
                    # P0: referenced junction must exist
                    if elem_id not in junction_id_set:
                        errors.append(
                            f"Road '{rid}' {link_name} references non-existent junction '{elem_id}'."
                        )
                        continue
                    # P1: road referencing a junction → junction must mention that road
                    if rid not in junction_referenced_roads.get(elem_id, set()):
                        errors.append(
                            f"Road '{rid}' {link_name} → junction '{elem_id}', "
                            f"but junction '{elem_id}' has no connection referencing road '{rid}'."
                        )

        return "\n".join(errors) if errors else None

    @staticmethod
    def _plan_view_endpoint(plan_view: list) -> tuple[float, float]:
        """Return the (x, y) endpoint of the last geometry segment in a plan_view."""
        last = plan_view[-1]
        x, y = float(last["x"]), float(last["y"])
        hdg = float(last["hdg"])
        length = float(last["length"])
        if last.get("geometry") == "arc":
            curvature = float(last.get("curvature", 0.0))
            if abs(curvature) > 1e-9:
                r = 1.0 / curvature
                cx = x - math.sin(hdg) * r
                cy = y + math.cos(hdg) * r
                angle_start = math.atan2(y - cy, x - cx)
                angle_end = angle_start + length * curvature
                abs_r = abs(r)
                return cx + abs_r * math.cos(angle_end), cy + abs_r * math.sin(angle_end)
        return x + math.cos(hdg) * length, y + math.sin(hdg) * length

    @staticmethod
    def validate_road_geometry_continuity(
        dsl: Dict[str, Any], tolerance_m: float = 3.0
    ) -> Optional[str]:
        """Check that directly connected roads are geometrically contiguous.

        For every road A whose successor is another road B (not a junction),
        the computed endpoint of A's plan_view must be within *tolerance_m* of
        B's plan_view start point.  A gap larger than this means the roads are
        logically linked but physically disconnected, which will produce a
        broken mesh in CARLA.

        Runs after validate_road_connectivity(), so reference integrity is
        already guaranteed.
        """
        roads = dsl.get("roads", [])
        road_by_id: Dict[str, Any] = {str(r["id"]): r for r in roads}
        errors = []

        for road in roads:
            rid = str(road["id"])
            successor = road.get("successor")
            if not successor or str(successor.get("elementType", "")) != "road":
                continue
            target_id = str(successor.get("elementId", ""))
            if target_id not in road_by_id:
                continue  # already caught by connectivity check

            plan_a = road.get("plan_view", [])
            plan_b = road_by_id[target_id].get("plan_view", [])
            if not plan_a or not plan_b:
                continue

            end_x, end_y = OpendriveGenerator._plan_view_endpoint(plan_a)
            start_x = float(plan_b[0]["x"])
            start_y = float(plan_b[0]["y"])
            dist = math.hypot(end_x - start_x, end_y - start_y)

            if dist > tolerance_m:
                errors.append(
                    f"Road '{rid}' endpoint ({end_x:.1f}, {end_y:.1f}) is {dist:.1f} m "
                    f"from road '{target_id}' start ({start_x:.1f}, {start_y:.1f}); "
                    f"max allowed gap is {tolerance_m} m. "
                    f"Adjust plan_view coordinates so the roads physically connect."
                )

        return "\n".join(errors) if errors else None

    # PROJ string for a local Cartesian plane centred at the WGS84 origin.
    # CARLA requires a non-empty geoReference to suppress the georeference
    # warning; the exact projection does not matter for local simulations.
    _GEO_REFERENCE = (
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m "
        "+geoidgrids=egm96_15.gtx +vunits=m +no_defs"
    )

    def render_road_dsl_to_xodr(self, dsl: Dict[str, Any], xodr_path: str):
        root = ET.Element("OpenDRIVE")
        header_bounds = self._estimate_header_bounds(dsl)
        header = ET.SubElement(
            root,
            "header",
            {
                "revMajor": "1",
                "revMinor": "4",
                "name": "AutoScenarioOpenDRIVE",
                "version": "1.00",
                "date": "2026-04-12",
                "north": f"{header_bounds['north']:.3f}",
                "south": f"{header_bounds['south']:.3f}",
                "east": f"{header_bounds['east']:.3f}",
                "west": f"{header_bounds['west']:.3f}",
                "vendor": "AutoScenario",
            },
        )
        geo_ref = ET.SubElement(header, "geoReference")
        geo_ref.text = self._GEO_REFERENCE

        for road in dsl["roads"]:
            road_element = ET.SubElement(
                root,
                "road",
                {
                    "name": str(road.get("name", road["id"])),
                    "length": f"{float(road['length_m']):.3f}",
                    "id": str(road["id"]),
                    "junction": str(road.get("junction", "-1")),
                },
            )
            link_element = ET.SubElement(road_element, "link")
            self._append_road_link(link_element, "predecessor", road.get("predecessor"))
            self._append_road_link(link_element, "successor", road.get("successor"))

            type_element = ET.SubElement(
                road_element,
                "type",
                {"s": "0.0", "type": str(road.get("road_type", "town"))},
            )
            ET.SubElement(
                type_element,
                "speed",
                {
                    "max": f"{float(road.get('speed_mps', 13.9)):.3f}",
                    "unit": "m/s",
                },
            )

            plan_view_element = ET.SubElement(road_element, "planView")
            for segment in road["plan_view"]:
                geometry = ET.SubElement(
                    plan_view_element,
                    "geometry",
                    {
                        "s": f"{float(segment['s']):.3f}",
                        "x": f"{float(segment['x']):.3f}",
                        "y": f"{float(segment['y']):.3f}",
                        "hdg": f"{float(segment['hdg']):.6f}",
                        "length": f"{float(segment['length']):.3f}",
                    },
                )
                if segment["geometry"] == "arc":
                    ET.SubElement(
                        geometry,
                        "arc",
                        {"curvature": f"{float(segment['curvature']):.8f}"},
                    )
                else:
                    ET.SubElement(geometry, "line")

            lanes_element = ET.SubElement(road_element, "lanes")
            for lane_section in road["lane_sections"]:
                section_element = ET.SubElement(
                    lanes_element,
                    "laneSection",
                    {"s": f"{float(lane_section['s']):.3f}"},
                )
                center = ET.SubElement(section_element, "center")
                center_lane = ET.SubElement(
                    center,
                    "lane",
                    {"id": "0", "type": "none", "level": "false"},
                )
                self._append_road_mark(
                    center_lane,
                    {
                        "type": lane_section.get("center_road_mark_type", "solid"),
                        "color": lane_section.get("center_road_mark_color", "yellow"),
                        "width": lane_section.get("center_road_mark_width", 0.15),
                        "lane_change": lane_section.get("center_lane_change", "none"),
                        "line_length": lane_section.get("center_road_mark_line_length", 3.0),
                        "space_length": lane_section.get("center_road_mark_space_length", 0.0),
                        "pattern_name": lane_section.get("center_road_mark_pattern_name"),
                    },
                    default_color="yellow",
                    default_lane_change="none",
                )

                left_element = ET.SubElement(section_element, "left")
                for lane_id, lane in enumerate(lane_section["left"], start=1):
                    self._append_lane(left_element, lane_id, lane)

                right_element = ET.SubElement(section_element, "right")
                for lane_id, lane in enumerate(lane_section["right"], start=1):
                    self._append_lane(right_element, -lane_id, lane)

        for junction in dsl.get("junctions", []):
            junction_element = ET.SubElement(
                root,
                "junction",
                {
                    "id": str(junction["id"]),
                    "name": str(junction.get("name", junction["id"])),
                },
            )
            for connection in junction.get("connections", []):
                ET.SubElement(
                    junction_element,
                    "connection",
                    {
                        "id": str(connection["id"]),
                        "incomingRoad": str(connection["incomingRoad"]),
                        "connectingRoad": str(connection["connectingRoad"]),
                        "contactPoint": str(connection.get("contactPoint", "start")),
                    },
                )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xodr_path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _append_road_link(link_element, tag_name: str, payload: Any):
        if not payload:
            return
        ET.SubElement(
            link_element,
            tag_name,
            {
                "elementType": str(payload.get("elementType", "road")),
                "elementId": str(payload.get("elementId", "-1")),
                "contactPoint": str(payload.get("contactPoint", "start")),
            },
        )

    @staticmethod
    def _append_lane(parent, lane_id: int, lane: Dict[str, Any]):
        lane_element = ET.SubElement(
            parent,
            "lane",
            {
                "id": str(lane_id),
                "type": str(lane.get("type", "driving")),
                "level": "false",
            },
        )
        ET.SubElement(
            lane_element,
            "width",
            {
                "sOffset": "0.0",
                "a": f"{float(lane.get('width', 3.5)):.3f}",
                "b": "0.0",
                "c": "0.0",
                "d": "0.0",
            },
        )
        OpendriveGenerator._append_road_mark(
            lane_element,
            {
                "type": lane.get("road_mark_type", "broken"),
                "color": lane.get("road_mark_color", "white"),
                "width": lane.get("road_mark_width", 0.15),
                "lane_change": lane.get("lane_change", "both"),
                "line_length": lane.get("road_mark_line_length", 3.0),
                "space_length": lane.get("road_mark_space_length", 9.0),
                "pattern_name": lane.get("road_mark_pattern_name"),
            },
            default_color="white",
            default_lane_change="both",
        )

    @staticmethod
    def _append_road_mark(
        lane_element,
        road_mark: Dict[str, Any],
        *,
        default_color: str,
        default_lane_change: str,
    ):
        road_mark_type = str(road_mark.get("type", "solid")).strip().lower() or "solid"
        road_mark_color = str(road_mark.get("color", default_color)).strip().lower() or default_color
        road_mark_width = float(road_mark.get("width", 0.15))
        road_mark_element = ET.SubElement(
            lane_element,
            "roadMark",
            {
                "sOffset": "0.0",
                "type": road_mark_type,
                "weight": str(road_mark.get("weight", "standard")),
                "color": road_mark_color,
                "material": str(road_mark.get("material", "standard")),
                "width": f"{road_mark_width:.3f}",
                "laneChange": str(road_mark.get("lane_change", default_lane_change)),
            },
        )
        if road_mark_type == "broken":
            pattern_name = road_mark.get("pattern_name") or f"broken_{road_mark_color}"
            type_element = ET.SubElement(
                road_mark_element,
                "type",
                {
                    "name": str(pattern_name),
                    "width": f"{road_mark_width:.3f}",
                },
            )
            ET.SubElement(
                type_element,
                "line",
                {
                    "length": f"{float(road_mark.get('line_length', 3.0)):.3f}",
                    "space": f"{float(road_mark.get('space_length', 9.0)):.3f}",
                    "tOffset": "0.0",
                    "width": f"{road_mark_width:.3f}",
                    "sOffset": "0.0",
                },
            )

    def summarize_road_dsl(self, dsl: Dict[str, Any]) -> str:
        lines = [
            f"roads={len(dsl.get('roads', []))}, junctions={len(dsl.get('junctions', []))}, "
            f"connections={len(dsl.get('connections', []))}",
        ]
        for road in dsl.get("roads", [])[:6]:
            first_section = road["lane_sections"][0]
            lines.append(
                "- "
                f"road_id={road['id']}, length_m={float(road['length_m']):.2f}, "
                f"segments={len(road['plan_view'])}, "
                f"left_lanes={len(first_section['left'])}, right_lanes={len(first_section['right'])}"
            )
        assumptions = dsl.get("metadata", {}).get("assumptions", [])
        if assumptions:
            lines.append("assumptions=" + "; ".join(str(item) for item in assumptions[:6]))
        return "\n".join(lines)

    def _build_stage1_summary(self, road_description: str, net_path: str) -> str:
        lines = ["Stage1 SUMO->OpenDRIVE conversion.", road_description.strip()]
        if os.path.exists(net_path):
            try:
                root = ET.parse(net_path).getroot()
                edge_count = len(
                    [
                        edge
                        for edge in root.findall("edge")
                        if edge.attrib.get("function") != "internal"
                    ]
                )
                lines.append(f"non_internal_edges={edge_count}")
            except ET.ParseError:
                lines.append("net_summary=unavailable")
        return "\n".join(line for line in lines if line)

    def _write_context_file(self, scenario_id: str, artifact: Dict[str, Any]):
        context_path = os.path.join(self.save_dir, f"{scenario_id}_xodr_context.json")
        write_to_file(context_path, json.dumps(artifact, indent=2, sort_keys=True))

    @staticmethod
    def _estimate_header_bounds(dsl: Dict[str, Any]) -> Dict[str, float]:
        xs = []
        ys = []
        for road in dsl.get("roads", []):
            for segment in road.get("plan_view", []):
                start_x = float(segment["x"])
                start_y = float(segment["y"])
                hdg = float(segment["hdg"])
                length = float(segment["length"])
                end_x = start_x + math.cos(hdg) * length
                end_y = start_y + math.sin(hdg) * length
                xs.extend([start_x, end_x])
                ys.extend([start_y, end_y])
        if not xs or not ys:
            return {"north": 1.0, "south": 0.0, "east": 1.0, "west": 0.0}
        return {
            "north": max(ys),
            "south": min(ys),
            "east": max(xs),
            "west": min(xs),
        }
