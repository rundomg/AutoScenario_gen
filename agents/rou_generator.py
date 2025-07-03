import os
import re
import subprocess
from typing import Optional
import xml.etree.ElementTree as ET
from agents.task_agent import TaskAgent
from xml.dom import minidom
from tools.utils import extract_text_section, read_file


def check_process_finish(results, message):
    stderr_lines = results.stderr.decode("utf-8").splitlines()
    errors = [line for line in stderr_lines if "warning" not in line.lower()]
    if errors:
        print("Failed to ", errors)
        return False
    print("Finished ", message)
    return True


class RouteGenerator(TaskAgent):
    def __init__(self, mode="default") -> None:
        super().__init__()
        # Number of vehicles and (x, y, heading, speed) for each background vehicle for 10 steps and (x, y, heading, speed) for autonomous vehicle being tested. (SHOULD BE exactly same and no other words!)
        if mode == "default":
            SYSTEM_PROMPT = """
            You are GPT-4V(ision), a large multi-modal model trained by OpenAI. Now you act as a mature autonomous driving tester, who can understand user's testing request and design the correspondinng testing scenarios. 
            Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.
            Make sure as many BVs near AV as possible and trips are long (start edge and end edges are far apart).

            Your answer should follow this format:
            ## Description
            Your description of the user request.
            ## Reasoning
            Reasoning based on user request, identify the testing goal, constraints, and optimal testing scenarios. Ensure there are background vehicles (BVs) around the autonomous vehicle (AV) in both time and space to create challenging scenarios for the AV.
            Find reachable paths based on edge connections(start and end nodes)
            ## Decision
            Generate (start edge, end edge, depart time) for each background vehicle and (start edge, end edge, depart time) for autonomous vehicle being tested
            Follow this format exactly!
            **BV_1**: [e0, e1, 5.1]
            **AV**: [e0, e2, 6]  
            **BV_2**: [e0, e5, 6.5]  
            **BV_3**: [e0, e3, 12]  
            """
        else:
            SYSTEM_PROMPT = """
            You are GPT-4V(ision), a large multi-modal model trained by OpenAI. Now you act as a mature autonomous driving tester, who can understand user's testing request and generate the correspondinng testing scenarios. 
            Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.

            Your answer should follow this format:
            ## Description
            Your description of the user request.
            ## Reasoning
            Reasoning based on user request, identify the constraints, and generate testing scenarios. Find the start and end node for all vehicles as described in the description. Determine the sequence of vehicles based on the description and set the departure times accordingly. Consider the front and rear details to accurately infer the order. Suggest one vehicle as AV being tested. 
            Ensure there are reachable paths based on edge connections(start and end nodes)
            ## Decision
            Generate (start edge, end edge, depart time) for each vehicle.
            Follow this format exactly!
            **Vehicle_1**: [e0, e1, 5.1]
            **Vehicle_2**: [e0, e2, 6]  
            **Vehicle_3**: [e0, e5, 6.5]  
            """
        self.pre_prompt = SYSTEM_PROMPT
        self.vehicle_data = None

    def refine_request(self, user_request, added_info=None):
        # There are
        final_request = self.pre_prompt + f"\nUser request is : {user_request}"
        return final_request

    def generate_trip_file(self, agents, filename="my_trips.trips.xml"):

        root = ET.Element("routes")
        # Define trips
        for agent in agents:
            trip = ET.SubElement(
                root,
                "trip",
                {
                    "id": agent["agent_id"],
                    "depart": agent["departtime"],
                    "from": agent["start_edge"],
                    "to": agent["end_edge"],
                },
            )

        # tree = ET.ElementTree(root)
        xml_str = ET.tostring(root, "utf-8")
        parsed_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
        with open(filename, "w") as f:
            f.write(parsed_str)

    def convert_trip_to_route(
        self, net_file, trip_file, route_file, partial_success=False
    ):
        results = subprocess.run(
            [
                "duarouter",
                "--net-file",
                net_file,
                "--verbose",
                "--routing-algorithm",
                "dijkstra",
                "--route-length",
                "true",
                "--keep-all-routes",
                "--repair",
                "--route-files",
                trip_file,
                "-o",
                route_file,
            ],
            capture_output=True,
        )
        if partial_success:
            success_bool = self.check_route_paritial_correct(
                results,
                "Use duarouter to generate important vehicles routes (AV and at least one BV)!",
            )
        else:
            success_bool = check_process_finish(
                results, "Use duarouter to generate route!"
            )
        return success_bool, results

    def check_route_paritial_correct(self, results, message):
        def extract_trip_name(text):
            # Define the regex pattern to match the trip name
            pattern = r"trip '([^']*)'"

            # Search the log message using the pattern
            match = re.search(pattern, text)

            if match:
                # Extract and return the trip name
                return match.group(1)
            else:
                return None

        def extract_vehicle_id(text):
            # Define the regex pattern to match the vehicle ID
            pattern = r"vehicle '([^']*)'"

            # Search the log message using the pattern
            match = re.search(pattern, text)

            if match:
                # Extract and return the vehicle ID
                return match.group(1)
            else:
                return None

        stderr_lines = results.stderr.decode("utf-8").splitlines()
        errors = [line for line in stderr_lines if "warning" not in line.lower()]
        print("errors", errors)
        if not errors:
            return True

        vehicle_name_list = []
        for error_msg in errors:
            vehicle_name = extract_trip_name(error_msg)
            vehicle_id = extract_vehicle_id(error_msg)
            if vehicle_name == "AV":
                return False
            if vehicle_id == "AV":
                return False
            if vehicle_name not in vehicle_name_list and (vehicle_name is not None):
                vehicle_name_list.append(vehicle_name)
            if vehicle_id not in vehicle_name_list and (vehicle_id is not None):
                vehicle_name_list.append(vehicle_id)
        # there are at least one correct vehicle:
        print(
            "=================vehicle_name_list================",
            vehicle_name_list,
            "len(self.vehicle_data)",
            len(self.vehicle_data),
        )

        # At least one BV in the scene
        if (len(vehicle_name_list) < len(self.vehicle_data) - 1) and (
            len(self.vehicle_data) - len(vehicle_name_list) > 1
        ):
            return True
        return False

    def extract_decision_data(self, scenario_id=None, save_folder=None):
        file_path = os.path.join(save_folder, f"{scenario_id}_tripdata_raw.txt")
        text = read_file(file_path)
        decision_content = extract_text_section(text, r"## Decision(.*)")
        if decision_content is None:
            return False

        decision_text = decision_content.group(1)
        pattern = re.compile(r"\*\*([\w\s_]+)\*\*: \[(.*?)\]")
        matches = pattern.findall(decision_text)
        if not matches:
            print("No vehicle data matches found.")
            return False
        vehicle_data = []
        for agent_id, trip_info in matches:
            try:
                start_edge, end_edge, departtime = map(str.strip, trip_info.split(","))
            except ValueError:
                print(f"Invalid trip format for agent {agent_id}: {trip_info}")
                return False, None

            entry = {
                "agent_id": agent_id,
                "start_edge": start_edge,
                "end_edge": end_edge,
                "departtime": departtime,
            }
            vehicle_data.append(entry)

        self.vehicle_data = vehicle_data
        save_path = os.path.join(save_folder, f"{scenario_id}_trip_data.txt")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for v in vehicle_data:
                    line = f"{v['agent_id']}: [{v['start_edge']}, {v['end_edge']}, {v['departtime']}]\n"
                    f.write(line)
        except Exception as e:
            print(f"Failed to write result to {save_path}: {e}")
            return False, None

        return True, vehicle_data

    def generate_route_from_trips(
        self, agents, scenario_id, save_dir, partial_success=False
    ):
        net_fn = os.path.join(save_dir, f"{scenario_id}.net.xml")
        trip_fn = os.path.join(save_dir, f"{scenario_id}_llm.trips.xml")
        route_fn = os.path.join(save_dir, f"{scenario_id}_llm.rou.xml")
        self.generate_trip_file(agents, filename=trip_fn)
        success_bool, results = self.convert_trip_to_route(
            net_fn, trip_fn, route_fn, partial_success
        )
        return success_bool, results

    def call_agent(
        self, user_request: str, scenario_id: str, output_folder: str
    ) -> int:
        """Generates agent routes and extracts the decision data."""
        attempt_count = 0
        success = False
        output_file = os.path.join(output_folder, f"{scenario_id}_tripdata_raw.txt")

        # Read in the scene description
        while not success:
            self.send_request(user_request, {"output_fn": output_file})
            success, agents_dict = self.extract_decision_data(
                scenario_id, output_folder
            )
            if success:
                success, results = self.generate_route_from_trips(
                    agents_dict, scenario_id, scenario_id, output_folder
                )
            attempt_count += 1
            if attempt_count > 1:
                print(f"Regenerating vehicles Routes... Attempt {attempt_count}")
        return attempt_count


if __name__ == "__main__":

    file_folder = os.path.join(os.getcwd(), "auto_result", "module_test")
    rougen = RouteGenerator(mode="described")

    scene_id = f"request_interpreter_0000_split"
    subfolder_path = os.path.join(file_folder, scene_id)
    net_xml_path = os.path.join(subfolder_path, f"{scene_id}.net.xml")
    txt_xml_path = os.path.join(subfolder_path, f"{scene_id}_net.txt")

    rougen.call_agent("Generate dense traffic in intersection")
