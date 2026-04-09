"Thanks GPT for all its major contribution:) Glory belongs to it. qiujing 2024.6.4."

import os
from agents.task_agent import TaskAgent
from agents.vlm_interpreter import VLMInterpreter
from agents.video_interpreter import VideoInterpreter
from agents.command_interpreter import CommandInterpreter
from agents.text_interpretor import TextInterpreter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API parameters from environment variables
OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 2000))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", 30))
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT")


class UniInterpreter(TaskAgent):

    def __init__(self, input_type="image") -> None:
        super().__init__()
        self.input_type = input_type

        SYSTEM_PROMPT = """
        You are GPT-4V(ision), a large multi-modal model trained by OpenAI. 
        You will now process multimodal inputs, which may include images, text, or other data formats. Your task is to convert these inputs into a standardized scenario description. This description should include the following elements:

        1. Roads: Identify and describe the roads, including their layout, lanes, direction, and any other relevant road features.
        2. Agent Location and Behavior: Define the position and behavior of any agents (e.g., vehicles, pedestrians, etc.) involved in the scenario. This includes their current location, movement, and intended actions.
        3. Static Objects: Identify any static objects within the environment (e.g., buildings, traffic signs, obstacles, etc.) and describe their position and characteristics.
        These standardized scenario descriptions will be used to reconstruct the testing scenario for simulation.
        
        Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the answers in the given format."""
        self.pre_prompt = SYSTEM_PROMPT

        if input_type == "image":
            self.interpreter = VLMInterpreter()
        elif input_type == "video":
            self.interpreter = VideoInterpreter()
        elif input_type == "crash_report":
            # crash report
            self.interpreter = TextInterpreter()
        else:
            # text
            self.interpreter = CommandInterpreter()

    def call_agent(self, user_request, input_info, use_system_prompt=True):
        # Use the system pre_prompt or not
        if use_system_prompt and self.input_type != "image":
            self.interpreter.pre_prompt += self.pre_prompt
        return self.interpreter.call_agent(user_request, input_info)

    def refine_request(self, user_request, add_info=None):
        """Read in original description from XX.txt and structure them into XXX_split.txt"""
        raw_description_fn = add_info["output_fn"].replace("_split.txt", ".txt")
        final_request = self.prepare_structurer_prompt(raw_description_fn)
        return final_request

    def prepare_structurer_prompt(self, raw_description_fn):
        decision = self.interpreter.extract_decision_data(raw_description_fn)
        request = """
        Before this I give you a text about decision.
        The text about decision I give you contains information about the road net, road users and static objects within the scenario..
        But the content dosen't match the format I need, so I need you to help me split the content into three parts.
        You need to unleash your imagination and creativity to generate a more detailed description of the road net, road users and static objects within the scenario. 
        The scenario involves hazardous situations for autonomous vehicles.
        """
        format = """
        The output shoud be in the following format:
        ## Road Net Description:
        The description of the road net.
        ## Road Users Description:
        The description of the road users, including their relative positions, movements in the scenario.
        ## Static Objects Description:
        The description of objects in the scene, including traffic cones, fences, traffic signs, etc.
        ## Vehicles' Locations and Behaviors
        The description of the agent vehicle location and behavior, and the surrounding vehicles' locations and behaviors.
        ## Scenario Description:
        The description of the whole scenario.
        """
        task_prompt = str(decision) + request + format
        return task_prompt

    def structure_output(self, raw_description_fn):
        """Generate structured output and save into xxx_split.txt"""
        if self.input_type == "image":
            return raw_description_fn
        add_info = {"output_fn": raw_description_fn.replace(".txt", "_split.txt")}
        self.send_request("", add_info)
        return

    def prepare_structurer_prompt_chinese(self, raw_description_fn):
        decision = self.interpreter.extract_full_data(raw_description_fn)
        request = """
        Translate the  English text into Chinese. 
        """
        task_prompt = str(decision) + request
        return task_prompt
