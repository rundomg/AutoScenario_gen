import os, sys

sys.path.insert(0, "../")
import cv2
import base64
import numpy as np
from tools.utils import read_file, extract_text_section
from agents.task_agent import TaskAgent


class VLMInterpreter(TaskAgent):
    def __init__(self):
        super().__init__()

        SYSTEM_PROMPT = """
        You are an assistant for generating autonomous vehicle testing scenario. You should generate a detailed description of the road network, the behaviors of the vehicles and the scenario based on the given image data.
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format.
        Your answer should restrictly follow this format:
        ## Description
        Your description of the user request.
        ## Reasoning
        reasoning based on user request, what is the testing goal and what are the best testing scenarios. Try to create complex road network with varying road types, road sturcture and connections.  The generated road network should be *very detailed and shorter than 100m*. Provide detailed description in "road geometry" part in the `## Decision` section.
        ## Decision
        This part should be as detailed as possible. Your output should contain as much concrete data as possible and include the number of the lanes, the width of the lanes, the number of vehicles and so on.
        Important: If you can, plan a route that will reach each point in the network, and describe the road in the first perspective of the vehicle.
        Important: If there is a fork in the road, you must point out the angle of the road. And if there is a intersection, you must point out the angle of the intersection.(For example, there is a four-way intersection, you can describe that, the main road points to the north, and the angle between the second road and the main road(the first road) is about 30 degrees, and the angle between the third road and the second road is about 150 degrees, and the third road and the main road is in a straight line and so on.)
        You can also describe angle information like that: "The main road, Pine Street, runs north-south. On the east side, Maple Avenue is located 30 degrees north northeast of the Pine Street, and on the west side, Oak Street is located 45 degrees north northwest of the Pine Street."
        You can identify and analyze the geometric structure of roads by referring to surrounding buildings and trees, as well as cars parked on the roadside and cars walking on the road. You need to give me the position of the surrounding objects relative to the vehicle.  You need to give me the starting position of other vehicles on the lane relative to this vehicle, or the absolute starting position of other vehicles on the lane.
        """

        example_prompt = """
        I give you a example to help you generate better description for road geometry in ##Decision. You should conduct the description like this example.
        ----example begin----
        The road network primarily consists of an intersection where four roads converge. Two of the roads (Road 1 and Road 2) run in the north-south direction and form the main route on which the current vehicle is traveling. Both Road 1 and Road 2 have three lanes for northbound traffic and three lanes for southbound traffic. Road 1 extends northward from the intersection, while Road 2 extends southward from the intersection. Road 3 intersects with Road 1 at a 90-degree angle and extends eastward from the intersection. It has two lanes for eastbound traffic and two lanes for westbound traffic. Road 4 intersects with Road 2 at a 60-degree angle and extends southwestward (i.e., 60 degrees south of east relative to Road 2). It has two lanes for southwestbound traffic and two lanes for traffic in the opposite direction.
        ----example end----
        """

        additional_hints = """
        Use realistic road network descriptions. Typical intersection types include:
        - Crossroad
        - T-intersection
        - Y-intersection
        - Ramp merges
        - Deformed intersections

        Be precise with geometric details, such as:
        - Lane widths (e.g., 3.6 meters standard, turn lanes may be wider).
        - Intersection angles (e.g., 90 degrees, 45 degrees).
        - Road lengths (under 100 meters).
        """

        accuracy_prompt = """
        If there is a fork in the road, you need to point out the angle of the road.
        If there is a intersection, you need to point out the angle of the intersection.(For example, there is a four-way intersection, you can describe that, the main road points to the north, and the angle between the second road and the main road(the first road) is about 30 degrees, and the angle between the third road and the second road is about 150 degrees, and the third road and the main road is in a straight line and so on.)
        You can identify and analyze the geometric structure of roads by referring to surrounding buildings and trees, as well as cars parked on the roadside and cars walking on the road.
        """

        self.pre_prompt = (
            SYSTEM_PROMPT + example_prompt + additional_hints + accuracy_prompt
        )

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

    def extract_decision_data(self, file_path):
        """Return extracted structured answer and determine if regeneration is needed."""
        text = read_file(file_path)
        decision_text = extract_text_section(text, r"## Decision\n(.+)")
        if decision_text is None:
            return "No decision section found.", True

        return decision_text, False
