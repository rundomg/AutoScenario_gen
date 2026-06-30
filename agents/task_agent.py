import os
import time

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API parameters from environment variables
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", 2000))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", 30))
OPENAI_CONNECT_TIMEOUT = int(os.getenv("OPENAI_CONNECT_TIMEOUT", 10))
OPENAI_REQUEST_RETRIES = int(os.getenv("OPENAI_REQUEST_RETRIES", 2))
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT")


class APIResponseError(Exception):
    """Raised for API responses that cannot be parsed as chat-completions JSON."""


class TaskAgent:
    def __init__(self):
        self.post_header = {
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        }
        self.MAX_REGENERATE_ATTEMPTS = 3

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
            connect_timeout, read_timeout = self._resolve_request_timeout(add_info)
            max_attempts = self._resolve_request_attempts(add_info)
            request_label = self._resolve_request_label(add_info)

            params = {
                "messages": [{"role": "user", "content": final_request}],
                "model": OPENAI_MODEL,
                "max_tokens": self._resolve_max_tokens(add_info),
            }

            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.post(
                        OPENAI_URL,
                        headers=self.post_header,
                        json=params,
                        stream=False,
                        timeout=(connect_timeout, read_timeout),
                    )
                    response.raise_for_status()

                    try:
                        res = response.json()
                    except ValueError as e:
                        raise APIResponseError(
                            self._format_response_error(
                                response,
                                "API returned a non-JSON response",
                            )
                        ) from e

                    try:
                        res_content = self._extract_response_content(res)
                    except Exception as e:
                        raise APIResponseError(
                            self._format_response_error(
                                response,
                                f"API response JSON has invalid chat format: {e}",
                            )
                        ) from e

                    if add_info and "output_fn" in add_info:
                        output_fn = add_info["output_fn"]
                        with open(output_fn, "w", encoding="utf-8") as file:
                            file.write(res_content)

                    return res_content
                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    APIResponseError,
                ) as e:
                    last_error = e
                    if attempt >= max_attempts:
                        break
                    wait_seconds = min(2** (attempt - 1), 8)
                    print(
                        f"{request_label} request attempt {attempt}/{max_attempts} "
                        f"failed: {e}. Retrying in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)
                except requests.exceptions.HTTPError as e:
                    raise Exception(self._format_http_error(e)) from e

            raise Exception(
                f"{request_label} request failed after {max_attempts} attempts: "
                f"{str(last_error)}"
            ) from last_error
        except KeyError as e:
            raise Exception(f"Invalid response format: missing key {str(e)}")
        except Exception as e:
            message = str(e)
            if (
                "request failed after" in message
                or message.startswith("API request failed")
                or message.startswith("Invalid response format")
            ):
                raise
            raise Exception(f"Unexpected error: {message}")

    @staticmethod
    def _resolve_request_timeout(add_info):
        if add_info and "request_timeout" in add_info:
            timeout = add_info["request_timeout"]
            if isinstance(timeout, (list, tuple)) and len(timeout) == 2:
                return float(timeout[0]), float(timeout[1])
            return OPENAI_CONNECT_TIMEOUT, float(timeout)
        return OPENAI_CONNECT_TIMEOUT, float(OPENAI_TIMEOUT)

    @staticmethod
    def _resolve_request_attempts(add_info):
        retry_count = (
            int(add_info["request_retries"])
            if add_info and "request_retries" in add_info
            else OPENAI_REQUEST_RETRIES
        )
        return max(1, retry_count + 1)

    @staticmethod
    def _resolve_request_label(add_info):
        if add_info and add_info.get("request_label"):
            return str(add_info["request_label"])
        return "API"

    @staticmethod
    def _resolve_max_tokens(add_info):
        if add_info and "request_max_tokens" in add_info:
            return int(add_info["request_max_tokens"])
        return OPENAI_MAX_TOKENS

    @staticmethod
    def _format_http_error(error):
        response = error.response
        if response is None:
            return f"API request failed: {str(error)}"

        body_preview = (response.text or "").strip()
        if len(body_preview) > 400:
            body_preview = f"{body_preview[:400]}..."
        if body_preview:
            return (
                f"API request failed with status {response.status_code}: "
                f"{body_preview}"
            )
        return f"API request failed with status {response.status_code}: {str(error)}"

    @staticmethod
    def _format_response_error(response, reason):
        status_code = getattr(response, "status_code", "unknown")
        headers = getattr(response, "headers", {}) or {}
        content_type = ""
        if hasattr(headers, "get"):
            content_type = headers.get("Content-Type") or headers.get("content-type") or ""

        body_preview = (getattr(response, "text", "") or "").strip()
        if len(body_preview) > 400:
            body_preview = f"{body_preview[:400]}..."
        if not body_preview:
            body_preview = "<empty>"

        return (
            f"{reason}; status={status_code}; content_type={content_type or 'unknown'}; "
            f"body_preview={body_preview}"
        )

    @staticmethod
    def _extract_response_content(res):
        if "choices" not in res or not res["choices"]:
            raise Exception("Invalid response format: missing choices")

        res_content = res["choices"][0]["message"]["content"]
        if not isinstance(res_content, str) or not res_content.strip():
            raise Exception("Empty model response content.")

        return res_content
