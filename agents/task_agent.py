import os
import threading
import time
from email.utils import parsedate_to_datetime

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
OPENAI_MIN_READ_TIMEOUT = float(os.getenv("OPENAI_MIN_READ_TIMEOUT", 0))
OPENAI_REQUEST_RETRIES = int(os.getenv("OPENAI_REQUEST_RETRIES", 2))
OPENAI_REQUEST_MIN_INTERVAL = float(os.getenv("OPENAI_REQUEST_MIN_INTERVAL", 0))
OPENAI_RETRY_MAX_WAIT = float(os.getenv("OPENAI_RETRY_MAX_WAIT", 30))
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT")

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC = 0.0


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
                    self._wait_for_rate_limit()
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
                    if not self._is_retryable_http_error(e):
                        raise Exception(self._format_http_error(e)) from e
                    last_error = Exception(self._format_http_error(e))
                    if attempt >= max_attempts:
                        break
                    wait_seconds = self._retry_wait_seconds(attempt, e)
                    print(
                        f"{request_label} request attempt {attempt}/{max_attempts} "
                        f"failed: {last_error}. Retrying in {wait_seconds:g}s..."
                    )
                    time.sleep(wait_seconds)

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
                return float(timeout[0]), max(float(timeout[1]), OPENAI_MIN_READ_TIMEOUT)
            return OPENAI_CONNECT_TIMEOUT, max(float(timeout), OPENAI_MIN_READ_TIMEOUT)
        return OPENAI_CONNECT_TIMEOUT, max(float(OPENAI_TIMEOUT), OPENAI_MIN_READ_TIMEOUT)

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
    def _wait_for_rate_limit():
        if OPENAI_REQUEST_MIN_INTERVAL <= 0:
            return
        global _LAST_REQUEST_MONOTONIC
        with _REQUEST_LOCK:
            now = time.monotonic()
            wait_seconds = OPENAI_REQUEST_MIN_INTERVAL - (now - _LAST_REQUEST_MONOTONIC)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.monotonic()
            _LAST_REQUEST_MONOTONIC = now

    @staticmethod
    def _is_retryable_http_error(error):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code == 429 or status_code in {500, 502, 503, 504}

    @staticmethod
    def _retry_wait_seconds(attempt, error=None):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = ""
        if hasattr(headers, "get"):
            retry_after = headers.get("Retry-After") or headers.get("retry-after") or ""
        parsed_retry_after = TaskAgent._parse_retry_after_seconds(retry_after)
        if parsed_retry_after is not None:
            return min(max(0.0, parsed_retry_after), OPENAI_RETRY_MAX_WAIT)
        return min(float(2 ** (attempt - 1)), OPENAI_RETRY_MAX_WAIT)

    @staticmethod
    def _parse_retry_after_seconds(value):
        if value is None or value == "":
            return None
        text = str(value).strip()
        try:
            return float(text)
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.astimezone()
        return max(0.0, retry_at.timestamp() - time.time())

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
