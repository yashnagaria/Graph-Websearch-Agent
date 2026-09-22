"""Google Gemini (AI Studio) clients.

Two wrappers, matching the pattern the other providers follow:

* :class:`GeminiJSONModel` - forces ``application/json`` output, used by the
  planner, selector, reviewer and router, which all return structured JSON.
* :class:`GeminiModel`     - free-form text, used by the reporter.

Both hit the REST endpoint directly (no extra SDK dependency) and always return a
``HumanMessage`` so the rest of the graph can treat every provider the same way.
"""

import json
import os

from langchain_core.messages.human import HumanMessage

from models._common import as_error, parse_json_payload, post_json
from utils.helper_functions import load_config

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.6-flash"


def _extract_text(response_json):
    """Pull the text out of a generateContent response, with a useful error if there is none."""
    candidates = response_json.get("candidates")
    if not candidates:
        blocked = response_json.get("promptFeedback", {}).get("blockReason")
        if blocked:
            raise ValueError(f"Gemini blocked the prompt: {blocked}")
        raise ValueError("No candidates in Gemini response")

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        raise ValueError(
            f"Gemini returned no text (finishReason: {candidate.get('finishReason', 'unknown')})"
        )

    return text


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

    def _request(self, json_mode, prompt_text):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Add it in the sidebar or in Streamlit secrets.")

        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": self._generation_config(json_mode),
        }

        data = post_json(self.model_endpoint, self.headers, payload, "Gemini")
        return _extract_text(data)


class GeminiJSONModel(_BaseGeminiModel):
    def invoke(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]
        prompt_text = (
            f"system:{system}. Your output must be JSON formatted. Just return the specified "
            f"JSON format, do not prepend your response with anything.\n\nuser:{user}"
        )

        try:
            text = self._request(True, prompt_text)
            return HumanMessage(content=json.dumps(parse_json_payload(text)))
        except Exception as e:
            return as_error(e)


class GeminiModel(_BaseGeminiModel):
    def invoke(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]
        prompt_text = f"system:{system}\n\nuser:{user}"

        try:
            return HumanMessage(content=self._request(False, prompt_text))
        except Exception as e:
            return as_error(e)
