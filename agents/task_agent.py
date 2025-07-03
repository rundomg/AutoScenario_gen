import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API parameters from environment variables
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 2000))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", 30))
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT")


class TaskAgent:
    def __init__(self):
        self.post_header = {"Authorization": f"Bearer {OPENAI_KEY}"}

    def refine_request(self, user_request=None, add_info=None):
        pass

    def send_request(self, user_request, add_info=None):
        final_request = self.refine_request(user_request, add_info)
        params = {
            "messages": [{"role": "user", "content": final_request}],
            "model": OPENAI_MODEL,
            "timeout": OPENAI_TIMEOUT,
            "max_tokens": OPENAI_MAX_TOKENS,
        }
        response = requests.post(
            OPENAI_URL,
            headers=self.post_header,
            json=params,
            stream=False,
        )
        res = response.json()

        res_content = res["choices"][0]["message"]["content"]
        # print("res", res)

        output_fn = None if "output_fn" not in add_info else add_info["output_fn"]
        if output_fn is not None:
            with open(output_fn, "w", encoding="utf-8") as file:
                file.write(res_content)

        return res_content
