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
        """
        Send a request to the OpenAI API and handle the response.
        
        Args:
            user_request (str): The user's request
            add_info (dict, optional): Additional information including output file path
            
        Returns:
            str: The response content from the API
            
        Raises:
            Exception: If the API request fails or response is invalid
        """
        try:
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
            response.raise_for_status()  # Raise an exception for bad status codes
            
            res = response.json()
            
            # Check if response has the expected structure
            if "choices" not in res or not res["choices"]:
                raise Exception("Invalid response format: missing choices")
                
            res_content = res["choices"][0]["message"]["content"]
            
            # Save output to file if specified
            if add_info and "output_fn" in add_info:
                output_fn = add_info["output_fn"]
                with open(output_fn, "w", encoding="utf-8") as file:
                    file.write(res_content)
            
            return res_content
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid response format: missing key {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
