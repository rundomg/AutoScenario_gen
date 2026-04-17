import os
import re
import requests
import math
import sys
import subprocess
import numpy as np

sys.path.insert(0, "../")
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from tools.utils import (
    check_process_finish,
    strip_out_xml_md,
    read_file,
)


class NetGenerator(TaskAgent):
    def __init__(self, save_dir) -> None:

        super().__init__()
        SYSTEM_PROMPT = """
        You generate a minimal SUMO road network from a road description that comes from a single traffic image.
        Your goal is faithful reconstruction, not creative scenario design.

        Core rules:
        1. Use only road facts supported by the input description.
        2. Do not invent road length, lane width, turn angles, traffic density, or vehicle counts unless they are explicitly provided or strictly required to create a minimal valid network.
        3. When a detail is uncertain, choose the simplest valid network that preserves the visible road structure and state the uncertainty in ## Reasoning.
        4. Keep the generated network short and minimal. Prefer a straight segment over extra branches unless a branch or intersection is clearly described.
        5. Do not add traffic lights, extra lanes, or extra roads unless the input clearly requires them.
        6. Do not mention or estimate number of vehicles in ## Decision.

        Your answer must strictly follow this format:
        ## Description
        Briefly restate only the visible road facts from the input.
        ## Reasoning
        Explain only the minimum necessary assumptions used to make the network valid.
        ## Decision
        Summarize the chosen network structure and clearly note any remaining uncertainty.
        ## SUMO Files Specification
        **Nodes (nodes.xml)**
        ```xml
        ...
        ```
        **Edges (edges.xml)**
        ```xml
        ...
        ```
        """

        task_constraints = """
        SUMO constraints:
        - Don't generate Tram Lines.
        - Ensure the in and out edges are well defined.
        - Avoid duplicate edge ids.
        - Keep coordinates non-negative.
        - Use the edge shape property only when the road visibly curves.
        - If a crosswalk is visible but cannot be represented directly in nodes/edges, mention it in ## Description or ## Reasoning instead of fabricating unsupported geometry.
        """
        self.pre_prompt = SYSTEM_PROMPT + task_constraints
        self.save_dir = save_dir

    def call_agent(self, user_request, scenario_id, add_info=None):
        request_valid_result = True
        generation_cnt = 0
        output_fn = add_info["output_fn"]
        while request_valid_result:
            self.send_request(user_request, add_info=add_info)
            # save data: node, edge and combined net file
            result = self.extract_decision_data(scenario_id, output_fn)
            request_valid_result = not result
            generation_cnt += 1
            if request_valid_result and generation_cnt >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Road network generation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts."
                )
            if generation_cnt > 1:
                print(
                    "----------------------------regenerating-------------------generation_cnt--",
                    generation_cnt,
                )

        return result, generation_cnt

    def refine_request(self, user_request, add_info=None):
        final_request = (
            self.pre_prompt + f"\nThe generation request is \n{user_request}"
        )
        if add_info is not None:
            if "example" in add_info:
                final_request += add_info["example"]
        return final_request

    def extract_decision_data(self, scenario_id, output_fn):
        response = read_file(output_fn)
        pattern = "##\s+SUMO\s+Files\s+Specification(.*?)edges(.*?)edges(.*?)```" ""
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return False
        sucess_bool = self.save_node_edge_files(
            self.save_dir, scenario_id, match.group()
        )
        if sucess_bool:
            print(
                "saved node and edge file into", self.save_dir, "with key", scenario_id
            )
        return sucess_bool

    def generate_sumo_road_net(self, node_fn, edge_fn, net_fn):
        try:
            # assuming one direction lanes
            results = subprocess.run(
                [
                    "netconvert",
                    f"--node-files={node_fn}",
                    f"--edge-files={edge_fn}",
                    "--default.spreadtype",
                    "center",
                    "--no-turnarounds",
                    f"--output-file={net_fn}",
                ],
                capture_output=True,
            )
            success_bool = check_process_finish(
                results, "use netconvert to convert net file!"
            )
            return success_bool

        except subprocess.CalledProcessError as e:

            return False

    def save_node_edge_files(self, save_dir, scenario_id, response):
        pattern = "Nodes\s+\(nodes.xml\)(.*?)```(.*?)```"
        # dump node file
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result = match.group()
            if len(result.split("**")) < 2:
                return False
            core_result = result.split("**")[1]
            node_fn = os.path.join(save_dir, f"{scenario_id}.nod.xml")
            with open(node_fn, "w") as file:
                file.write(strip_out_xml_md(core_result))
        else:
            print("failed to match node result")

        pattern = "Edges\s+\(edges.xml\)(.*?)```(.*?)```"

        # dump edge file
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result = match.group()
            core_result = result.split("**")[1]
            edge_fn = os.path.join(save_dir, f"{scenario_id}.edg.xml")
            with open(edge_fn, "w") as file:
                file.write(strip_out_xml_md(core_result))
        else:

            return False

        success_bool = self.generate_sumo_road_net(
            node_fn, edge_fn, os.path.join(save_dir, f"{scenario_id}.net.xml")
        )
        return success_bool

    def generate_random_route(self, data_dir, scenario_id, number_vehicles):
        net_fn = os.path.join(data_dir, f"{scenario_id}.net.xml")
        route_fn = os.path.join(data_dir, f"{scenario_id}.rou.xml")
        subprocess.run(
            [
                "python",
                "/usr/share/sumo/tools/randomTrips.py",
                "-n",
                net_fn,
                "-r",
                route_fn,
                "-e",
                "50",
                "-p",
                "1",
                "-l",
                "--trip-number",
                number_vehicles,
                "--allow-fringe",
            ]
        )

    def randomtrips(self, data_dir, scenario_id, number_vehicles):
        net_fn = os.path.join(data_dir, f"{scenario_id}.net.xml")
        route_fn = os.path.join(data_dir, f"{scenario_id}.rou.xml")
        subprocess.run(
            [
                "python",
                "/usr/share/sumo/tools/randomTrips.py",
                "-n",
                net_fn,
                "-r",
                route_fn,
                "-b",
                "0",
                "-e",
                "100",
                "-p",
                str(100 / number_vehicles),
            ]
        )

    def prepare_net_based_on_gps(self, data_dir, scenario_id, gps_info, radius_m=50):
        create_osm_file_from_point(
            lat=gps_info["lat"],
            lon=gps_info["lon"],
            radius_m=gps_info.get("radius_m", radius_m),
            data_dir=data_dir,
            scenario_id=scenario_id,
        )
        convert_osm_to_sumo_network(
            osm_path=os.path.join(data_dir, f"{scenario_id}.osm"), output_dir=data_dir
        )


def convert_osm_to_sumo_network(osm_path, output_dir, netconvert_path="netconvert"):
    """
    Convert an OSM file to SUMO .nod.xml and .edg.xml, and then generate .net.xml.
    """
    if not os.path.exists(osm_path):
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(osm_path))[0]
    nod_path = os.path.join(output_dir, f"{base}.nod.xml")
    edg_path = os.path.join(output_dir, f"{base}.edg.xml")
    net_path = os.path.join(output_dir, f"{base}.net.xml")

    # Step 1: Convert .osm to .nod.xml and .edg.xml
    extract_cmd = [
        netconvert_path,
        "--osm-files",
        osm_path,
        "--plain-output-prefix",
        os.path.join(output_dir, base),
        "--no-internal-links",
    ]
    print("Extracting .nod.xml and .edg.xml from .osm...")
    subprocess.run(extract_cmd, check=True)

    # Step 2: Use .nod.xml and .edg.xml to generate .net.xml
    netconvert_cmd = [
        netconvert_path,
        "--node-files",
        nod_path,
        "--edge-files",
        edg_path,
        "--output-file",
        net_path,
    ]

    print("Converting .nod.xml and .edg.xml to .net.xml...")
    subprocess.run(netconvert_cmd, check=True)
    print(f"Generated files in {output_dir}:")
    print(f"   Nodes: {nod_path}")
    print(f"   Edges: {edg_path}")
    print(f"   Network: {net_path}")


def create_osm_file_from_point(lat, lon, radius_m, data_dir, scenario_id):
    # Earth radius in meters
    R = 6378137

    # Convert meters to degrees
    delta_lat = (radius_m / R) * (180 / math.pi)
    delta_lon = (radius_m / (R * math.cos(math.pi * lat / 180))) * (180 / math.pi)

    # Bounding box: [south, west, north, east]
    bbox = [lat - delta_lat, lon - delta_lon, lat + delta_lat, lon + delta_lon]

    # Overpass API
    url = "http://overpass-api.de/api/interpreter"
    os.makedirs(data_dir, exist_ok=True)

    query = f"""
    [out:xml];
    (
      way({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})["highway"];
    );
    (._;>;);
    out body;
    """

    response = requests.get(url, params={"data": query})
    if response.status_code == 200:
        out_osm_fn = os.path.join(data_dir, f"{scenario_id}.osm")
        with open(out_osm_fn, "w", encoding="utf-8") as file:
            file.write(response.text)
            print(f"OSM data saved to {out_osm_fn}")
    else:
        print(f"Error fetching data: {response.status_code}")
