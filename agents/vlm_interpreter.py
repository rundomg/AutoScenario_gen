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

        # SYSTEM_PROMPT = """
        # You are an assistant for faithfully describing a traffic image from the ego vehicle perspective.
        # Your task is to describe only what is directly visible in the image so the scene can later be reconstructed.
        # Do not invent, enrich, imagine, extrapolate, or plan any scenario beyond the image evidence.
        # If something is not clearly visible, partially occluded, too small, or uncertain, explicitly say `not visible`, `unclear`, or `occluded`.
        # Do not provide numeric values such as lane width, road length, distance, speed, or angle unless they are directly and reliably visible in the image itself.
        # Describe each visible object once to avoid repetition.
        # Your answer must strictly follow this exact section format:
        # ## Road Net Description:
        # ## Road Users Description:
        # ## Static Objects Description:
        # ## Vehicles' Locations and Behaviors:
        # ## Scenario Description:
        # """

        # section_guidance = """
        # Section rules:
        # 1. `## Road Net Description:` only describe visible road geometry and road-state evidence, such as lane markings, intersections, crosswalks, curbs, medians, shoulders, sidewalks, road surface condition, and visible traffic organization.
        # 2. `## Road Users Description:` only describe dynamic road users that can affect the ego car, including cars, trucks, buses, motorcycles, cyclists, and pedestrians. For each object, include a visible description and why it matters to the ego vehicle. If none are clearly visible, say so.
        # 3. `## Static Objects Description:` describe visible static scene elements relevant to driving, such as traffic signs, traffic lights and their visible state, cones, barriers, parked objects, debris, bins, storefront fixtures, or other fixed roadside objects. For each object, include a visible description and why it matters to the ego vehicle. If none are clearly visible, say so.
        # 4. `## Vehicles' Locations and Behaviors:` summarize only visible vehicles from the ego perspective, including relative lane position, orientation, whether they appear stopped or moving if visually evident, and any uncertainty. Do not infer future trajectories.
        # 5. `## Scenario Description:` provide a short factual summary of the visible traffic scene and the immediate constraints it presents to the ego vehicle. Keep it grounded in visible evidence only.
        # """

        # output_rules = """
        # Additional output rules:
        # - Keep the five section headers exactly as written.
        # - Do not add extra headers.
        # - Do not leave sections blank; if evidence is missing, explicitly state that it is not visible or unclear.
        # - Do not repeat the same object across multiple bullet points unless needed for a brief cross-reference.
        # - Prefer relative descriptions such as left, right, ahead, near the curb, in the oncoming lane, or on the sidewalk when these are visually supported.
        # """

        # self.pre_prompt = SYSTEM_PROMPT + section_guidance + output_rules
        SYSTEM_PROMPT = """
        You are an assistant for faithfully describing a traffic image from the ego vehicle perspective for later scene reconstruction in a simulator.

        Your task is to describe only what is directly visible in the image.
        Do not describe the scene as a driving challenge, test case, or planning problem.

        If something is not clearly visible, partially occluded, too small, or genuinely uncertain, explicitly say `not visible`, `unclear`, or `occluded`.
        Use these uncertainty words only when the image evidence is genuinely insufficient. Otherwise, state the strongest directly supported visible description.

        Do not provide numeric values such as lane width, road length, distance, speed, or angle unless they are directly and reliably visible in the image itself.
        Describe each visible object once and refer to it consistently to avoid repetition.

        Prioritize the nearest and most restrictive visible objects for the ego vehicle, especially those occupying or narrowing the immediate drivable space.
        Preserve relative spatial relationships faithfully, including left/right ordering, near/far ordering, lane-side occupancy, curb proximity, and whether objects appear aligned, grouped, or distributed along the roadside.

        Your answer must strictly follow this exact section format:
        ## Road Net Description:
        ## Road Users Description:
        ## Static Objects Description:
        ## Vehicles' Locations and Behaviors:
        ## Scenario Description:
        """

        section_guidance = """
        Section rules:
        1. `## Road Net Description:`
        Only describe visible road geometry and road-state evidence, such as road direction, lane organization, lane markings, intersections, crosswalks, curbs, medians, shoulders, sidewalks, road surface condition, and visible traffic organization.
        Focus on the road structure that affects scene layout and object placement.

        2. `## Road Users Description:`
        Describe visible dynamic road users that could affect or constrain the ego vehicle, including cars, trucks, buses, motorcycles, cyclists, and pedestrians.
        For each object, describe:
        - what it is,
        - where it is relative to the ego vehicle,
        - its visible relevance based only on proximity, lane occupancy, roadside obstruction, or closeness to the drivable area.
        Prioritize near-field and more influential objects before distant background road users.
        If none are clearly visible, explicitly say so.

        3. `## Static Objects Description:`
        Describe visible static scene elements relevant to driving and reconstruction, such as traffic signs, traffic lights and their visible state, cones, barriers, parked vehicles, debris, bins, storefront fixtures, poles, trees, benches, and other fixed roadside objects.
        Prioritize static elements that affect traffic control, usable road space, road boundaries, or object placement over purely decorative background details.
        For each object, describe its visible form, location, and visible relevance to road space or scene structure.
        If none are clearly visible, explicitly say so.

        4. `## Vehicles' Locations and Behaviors:`
        Summarize only visible vehicles from the ego perspective.
        For each visible vehicle, describe its relative position, orientation, lane-side relation, and whether it appears parked, stopped, or moving only if visually supported.
        Do not infer future trajectory or intent.
        If motion state cannot be judged from the image, say so briefly.

        5. `## Scenario Description:`
        Provide a short reconstruction-oriented summary of the visible traffic scene.
        Focus on the dominant road structure, the nearest influential road users, the main roadside constraints, and the most important visible control elements.
        Keep it factual and grounded in visible evidence only.
        """

        output_rules = """
        Additional output rules:
        - Keep the five section headers exactly as written.
        - Do not add extra headers.
        - Do not leave sections blank; if evidence is missing, explicitly state that it is not visible or unclear.
        - Do not repeat the same object in multiple long descriptions; keep cross-references brief.
        - Prefer relative descriptions such as left, right, ahead, ahead-left, ahead-right, near the curb, near the sidewalk, near the center of the road, or along the roadside when visually supported.
        - Preserve layout-oriented information over narrative fluency.
        - Do not turn the description into a story; keep it precise, visual, and reconstruction-oriented.
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
