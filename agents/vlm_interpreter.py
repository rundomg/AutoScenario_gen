import os, re, sys

sys.path.insert(0, "../")
import cv2
import base64
import numpy as np
from tools.utils import read_file, extract_text_section
from agents.task_agent import TaskAgent


class VLMInterpreter(TaskAgent):
    REQUIRED_SECTION_HEADERS = (
        "Road Net Description",
        "Road Users Description",
        "Static Objects Description",
        "Vehicles' Locations and Behaviors",
        "Scenario Description",
    )
    LEGACY_HEADERS = ("## Description", "## Reasoning", "## Decision")

    def __init__(self):
        super().__init__()

        SYSTEM_PROMPT = """
        You are an assistant for faithfully describing a traffic image from the ego vehicle perspective.
        Your task is to describe only what is directly visible in the image so the scene can later be reconstructed.
        Do not invent, enrich, imagine, extrapolate, or plan any scenario beyond the image evidence.
        Do not output JSON or dictionary format.
        Do not output reasoning, chain-of-thought, test goals, route plans, or ego driving suggestions.
        Do not use the old sections `## Description`, `## Reasoning`, or `## Decision`.
        If something is not clearly visible, partially occluded, too small, or uncertain, explicitly say `not visible`, `unclear`, or `occluded`.
        Do not provide numeric values such as lane width, road length, distance, speed, or angle unless they are directly and reliably visible in the image itself.
        Describe each visible object once to avoid repetition.
        Your answer must strictly follow this exact section format:
        ## Road Net Description:
        ## Road Users Description:
        ## Static Objects Description:
        ## Vehicles' Locations and Behaviors:
        ## Scenario Description:
        """

        section_guidance = """
        Section rules:
        1. `## Road Net Description:` only describe visible road geometry and road-state evidence, such as lane markings, intersections, crosswalks, curbs, medians, shoulders, sidewalks, road surface condition, and visible traffic organization.
        2. `## Road Users Description:` only describe dynamic road users that can affect the ego car, including cars, trucks, buses, motorcycles, cyclists, and pedestrians. For each object, include a visible description and why it matters to the ego vehicle. If none are clearly visible, say so.
        3. `## Static Objects Description:` describe visible static scene elements relevant to driving, such as traffic signs, traffic lights and their visible state, cones, barriers, parked objects, debris, bins, storefront fixtures, or other fixed roadside objects. For each object, include a visible description and why it matters to the ego vehicle. If none are clearly visible, say so.
        4. `## Vehicles' Locations and Behaviors:` summarize only visible vehicles from the ego perspective, including relative lane position, orientation, whether they appear stopped or moving if visually evident, and any uncertainty. Do not infer future trajectories.
        5. `## Scenario Description:` provide a short factual summary of the visible traffic scene and the immediate constraints it presents to the ego vehicle. Keep it grounded in visible evidence only.
        """

        output_rules = """
        Additional output rules:
        - Keep the five section headers exactly as written.
        - Do not add extra headers.
        - Do not leave sections blank; if evidence is missing, explicitly state that it is not visible or unclear.
        - Do not repeat the same object across multiple bullet points unless needed for a brief cross-reference.
        - Prefer relative descriptions such as left, right, ahead, near the curb, in the oncoming lane, or on the sidewalk when these are visually supported.
        """

        self.pre_prompt = SYSTEM_PROMPT + section_guidance + output_rules

    def ImageEncode(self, image):
        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return img_base64

    def refine_request(self, user_request, add_info=None):
        assert "image_path" in add_info
        image_path = add_info["image_path"]
        assert os.path.exists(image_path)
        image = cv2.imread(image_path)

        if user_request is not None:
            generation_request = self.pre_prompt + f"\nUser request is : {user_request}"
        else:
            generation_request = self.pre_prompt

        image_base64 = self.ImageEncode(image)
        final_request = [
            {"type": "text", "text": generation_request},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
        return final_request

    def call_agent(self, user_request, added_info):
        answer_not_right = True
        generation_cnt = 0
        output_fn = added_info["output_fn"]

        while answer_not_right:
            self.send_request(user_request, added_info)
            result, answer_not_right = self.extract_decision_data(output_fn)
            generation_cnt += 1
            if generation_cnt > 1:
                print(
                    "Regenerating the scene interpretation! Generation Round:",
                    generation_cnt,
                )
        return result

    def validate_output_structure(self, text):
        for header in self.LEGACY_HEADERS:
            if re.search(rf"(?m)^{re.escape(header)}\b", text):
                return f"Legacy section header found: {header}"

        pattern = re.compile(
            r"(?m)^##\s*(Road Net Description|Road Users Description|Static Objects Description|Vehicles' Locations and Behaviors|Scenario Description)\s*:\s*$"
        )
        matches = list(pattern.finditer(text))
        if len(matches) != len(self.REQUIRED_SECTION_HEADERS):
            return "Missing required section headers or unexpected header format."

        found_headers = tuple(match.group(1) for match in matches)
        if found_headers != self.REQUIRED_SECTION_HEADERS:
            return "Section headers are missing, duplicated, or out of order."

        for index, match in enumerate(matches):
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.end() : next_start].strip()
            if not content:
                return f"Section `{match.group(1)}` is empty."

        return None

    def extract_decision_data(self, file_path):
        """Return extracted structured answer and determine if regeneration is needed."""
        text = read_file(file_path)
        validation_error = self.validate_output_structure(text)
        if validation_error is not None:
            return validation_error, True

        road_net_description = extract_text_section(
            text,
            r"##\s*Road Net Description\s*:\s*(.*?)\s*##\s*Road Users Description\s*:",
        )
        scenario_description = extract_text_section(
            text, r"##\s*Scenario Description\s*:\s*(.*)"
        )
        if not road_net_description:
            return "Road Net Description is empty.", True
        if not scenario_description:
            return "Scenario Description is empty.", True

        return text, False
