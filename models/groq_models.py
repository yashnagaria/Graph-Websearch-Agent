"""Groq clients.

Groq speaks the OpenAI chat-completions shape. Note that Groq retires model ids
fairly aggressively - ``llama3-70b-8192`` and ``llama3-8b-8192`` are both gone - and
returns HTTP 400 with an explicit "has been decommissioned" message when you use one.
That message is now surfaced to the user rather than being swallowed.
"""

import json
import os

from langchain_core.messages.human import HumanMessage

from models._common import (
    as_error,
    openai_choice_text,
    parse_json_payload,
    post_json,
)
from utils.helper_functions import load_config

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class _BaseGroqModel:
    def __init__(self, temperature=0, model=None):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)

        self.api_key = os.environ.get("GROQ_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        self.model_endpoint = API_URL

    def _request(self, messages, json_mode):
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it in the sidebar or in Streamlit secrets.")

        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"system:{system}\n\n user:{user}"}],
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = post_json(self.model_endpoint, self.headers, payload, "Groq")
        return openai_choice_text(data, "Groq")


class GroqJSONModel(_BaseGroqModel):
    def invoke(self, messages):
        try:
            text = self._request(messages, json_mode=True)
            return HumanMessage(content=json.dumps(parse_json_payload(text)))
        except Exception as e:
            return as_error(e)


class GroqModel(_BaseGroqModel):
    def invoke(self, messages):
        try:
            return HumanMessage(content=self._request(messages, json_mode=False))
        except Exception as e:
            return as_error(e)
