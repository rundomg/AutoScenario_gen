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
        """Figure out how many lanes, how many vehicles and what is the generated road type the user wants to generate. """
        SYSTEM_PROMPT = "Now you act as a professional scenario initializer, who can generate realistic vehicle positions \
        and road structure in complex urban driving scenarios according to user's generation request. You'll receive an scenario description. \
        First figure out how many lanes, how many vehicles and what is the generated road type the user wants to generate. \
            Then generate SUMO node and edge files starting with <?xml version>. Your answer should follow this format:\n## Description \nYour description of the request.\
        #                   \n## Reasoning \nreasoning based on the lane information and the user's request.\
        #                   \n## Decision \nnumber of scenarios \nnumber of lanes, \nnumber of vehicles.\
        #                   \n## SUMO Files Specification \n**Nodes (nodes.xml)** \n**Edges (edges.xml)** .\
        #                   \nMake sure your answer follow this given format strictly."

        crossing_instruction = """If there are any crossing: 1. Determine Crossing Locations: Crossings are typically placed at the ends of each approach to the intersection, aligning with sidewalks or pedestrian paths.
            2. Create Nodes for Crossings: Define nodes at the start and end points of each crossing.
            3. Connect Crossing Nodes with Edges: Create edges between crossing nodes to represent pedestrian and bicycle paths.
            4. Do not have negative number in the net."""

        task_constraints = "Don't generate Tram Lines. If it is an intersection, then all roads are bidirectional. Determine the number of lanes in each direction and ensure they match the description(the total number of lanes or the number of lanes in the specified direction) by setting the numLanes attribute in the edge definition. Ensure the in and out edges are well defined. If there are more than one lane in one direction, add traffic light."
        road_constraints = "The generated road network should be very detailed and shorter than 200m. Avoid using duplicate edge ids."
        additional_hints = "Use the edge shape property to define curves according to the description. Ensure the intermedia points inside shape create natural curve without sharpe edges while maintaining reasonable start and end locations. For exmaple: 581.45,148.50 578.00,142.23 575.40,139.59 571.84,137.13 568.36,135.48 564.31,134.51 562.23,134.16"
        final_request = ""
        final_request += SYSTEM_PROMPT
        final_request += task_constraints
        final_request += road_constraints
        final_request += additional_hints
        self.pre_prompt = final_request
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
            if generation_cnt > 1:
                print(
                    "----------------------------regenerating-------------------generation_cnt--",
                    generation_cnt,
                )
            generation_cnt += 1

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
