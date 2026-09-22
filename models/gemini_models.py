"""Google Gemini (AI Studio) clients used by the agents.

Two wrappers, matching the pattern the other providers follow:

* :class:`GeminiJSONModel` - forces ``application/json`` output, used by the
  planner, selector, reviewer and router, which all return structured JSON.
* :class:`GeminiModel`     - free-form text, used by the reporter.

Both hit the REST endpoint directly (no extra SDK dependency) and always return a
``HumanMessage`` so the rest of the graph can treat every provider the same way.
"""

import os
import json
import requests
from langchain_core.messages.human import HumanMessage

from utils.helper_functions import load_config

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
REQUEST_TIMEOUT = 120
DEFAULT_MODEL = "gemini-2.0-flash"


def _extract_text(response_json):
    """Pull the text out of a generateContent response, with a useful error if there is none."""
    if "error" in response_json:
        message = response_json["error"].get("message", "unknown error")
        raise ValueError(f"Gemini API error: {message}")

    candidates = response_json.get("candidates")
    if not candidates:
        feedback = response_json.get("promptFeedback", {})
        blocked = feedback.get("blockReason")
        if blocked:
            raise ValueError(f"Gemini blocked the prompt: {blocked}")
        raise ValueError("No candidates in Gemini response")

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        reason = candidate.get("finishReason", "unknown")
        raise ValueError(f"Gemini returned no text (finishReason: {reason})")

    return text


def _strip_code_fence(text):
    """Gemini occasionally wraps JSON in ```json fences even when asked not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[: -3]
    return stripped.strip()


class _BaseGeminiModel:
    def __init__(self, temperature=0, model=None):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)

        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.headers = {'Content-Type': 'application/json'}
        self.model_endpoint = f"{API_BASE}/{self.model}:generateContent?key={self.api_key}"

    def _generation_config(self, json_mode):
        config = {"temperature": self.temperature}
        if json_mode:
            config["response_mime_type"] = "application/json"

        # 2.5 Flash reasons before answering by default, which burns the output budget
        # on a task where the agents only need the answer. Turn it off where supported.
        if "2.5-flash" in self.model:
            config["thinkingConfig"] = {"thinkingBudget": 0}

        return config

    def _post(self, prompt_text, json_mode):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it in the sidebar or in Streamlit secrets.")

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": self._generation_config(json_mode),
        }

        response = requests.post(
            self.model_endpoint,
            headers=self.headers,
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )
        print("REQUEST RESPONSE", response.status_code)

        try:
            response_json = response.json()
        except json.JSONDecodeError:
            raise ValueError(f"Gemini returned a non-JSON response ({response.status_code}): {response.text[:300]}")

        return _extract_text(response_json)


class GeminiJSONModel(_BaseGeminiModel):
    def invoke(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]

        prompt_text = (
            f"system:{system}. Your output must be JSON formatted. Just return the specified "
            f"JSON format, do not prepend your response with anything.\n\nuser:{user}"
        )

        try:
            text = self._post(prompt_text, json_mode=True)
            parsed = json.loads(_strip_code_fence(text))
            return HumanMessage(content=json.dumps(parsed))
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            return HumanMessage(content=json.dumps({"error": error_message}))


class GeminiModel(_BaseGeminiModel):
    def invoke(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]

        prompt_text = f"system:{system}\n\nuser:{user}"

        try:
            text = self._post(prompt_text, json_mode=False)
            return HumanMessage(content=text)
        except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
            error_message = f"Error in invoking model! {str(e)}"
            print("ERROR", error_message)
            return HumanMessage(content=json.dumps({"error": error_message}))
