"Thanks GPT for all its major contribution:) Glory belongs to it. qiujing 2024.6.4."

import re
import os
import numpy as np
from agents.task_agent import TaskAgent


class TextInterpreter(TaskAgent):

    def __init__(self) -> None:
        super().__init__()
        SYSTEM_PROMPT = """
        You are GPT-4V(ision), a large multi-modal model trained by OpenAI. Now you act as a mature autonomous driving tester, who can understand the crash report content and design the correspondinng testing scenarios. 
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.

        Your answer should follow this format:
        ## Description
        Provide a clear and concise description of the crash report.
        ## Reasoning
        Explain the reasoning behind the crash scenario. Outline the testing objectives and identify the most relevant testing scenarios.
        ## Decision
        Should start with answer directly no other words.
        Provide a comprehensive account of the crash event, including details about the road structure, the vehicles involved, static objects, and their respective roles in the incident in the `## Decision` section.
        """

        FORMAT_PROMT = """For road structure, Try to describe from following perspectives: 1. General Layout. Overall Structure: Grid, radial, organic, or a mix. Extent and Coverage: Geographic area covered, including urban, suburban, and rural areas. Key Features: Major roads, highways, and interchanges.
        2. Road Types and Configurations Road Types: Highways, arterial roads, local streets, alleys, and service roads. 3. Number of lanes for each road.
        For exmaple, one answer is: Road Description: At the heart of this network, Central Square Intersection itself is a four-way junction where the city's two major arterial roads, Grand Avenue and Main Street, converge.
        On the northern approach, Grand Avenue features four lanes that split into dedicated left-turn, straight, and right-turn lanes as it nears the intersection. This road is flanked by parallel service lanes that provide access to local businesses and residential areas, ensuring that through traffic remains unobstructed. To the south, Grand Avenue continues with similar lane configurations, but also includes a bus lane that integrates seamlessly with the city's public transport system.Main Street, running east to west, mirrors this complexity. The western approach of Main Street includes three lanes for straight-going traffic, complemented by additional lanes for left and right turns. The eastern stretch of Main Street also features multiple lanes, with a dedicated tram line running down the center, offering another layer of public transportation connectivity.
        Adjacent to the intersection are several critical connectors. On the northwest corner, Elm Street branches off from Grand Avenue, acting as a key feeder road for the residential neighborhood it serves. This street features traffic calming measures such as speed bumps and narrow lanes to maintain safe speeds. On the southeast side, Oak Road provides a direct route to the nearby commercial district, with wide lanes to accommodate delivery trucks and heavy traffic"""

        self.pre_prompt = SYSTEM_PROMPT + FORMAT_PROMT

    def call_agent(self, user_request, input_dict):
        answer_not_right = True
        generation_cnt = 0
        output_fn = input_dict["output_fn"]

        while answer_not_right:
            self.send_request(user_request, input_dict)
            result, answer_not_right = self.extract_decision_data(output_fn)
            generation_cnt += 1
            if answer_not_right and generation_cnt >= self.MAX_REGENERATE_ATTEMPTS:
                raise RuntimeError(
                    "Scene interpretation failed after "
                    f"{self.MAX_REGENERATE_ATTEMPTS} attempts: {result}"
                )
            if generation_cnt > 1:
                print(
                    "Regenerating the scene interpretation! Generation Round:",
                    generation_cnt,
                )
        return result

    def refine_request(self, user_request, add_info=None):

        assert os.path.exists(add_info["text_path"])
        if user_request is not None:
            generation_request = self.pre_prompt + f"\nUser request is : {user_request}"
        else:
            generation_request = self.pre_prompt

        file_path = add_info["text_path"]
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        final_request = (
            generation_request
            + f"\n Crash report is :\n {text}"
            + f"\nUser request is : {user_request}"
        )
        return final_request

    def extract_decision_data(self, file_path):
        """Return extracted structured answer and Need to regenerate or not."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        decision_text = re.search(r"## Decision\n(.+)", text, re.S)
        if decision_text:
            decision_text = decision_text.group(1)
        else:
            return "No decision section found.", True
        return decision_text, False
