"""Anthropic Claude clients (the Messages API)."""

import json
import os

from langchain_core.messages.human import HumanMessage

from models._common import as_error, parse_json_payload, post_json
from utils.helper_functions import load_config

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 4096


class _BaseClaudModel:
    def __init__(self, temperature=0, model=None):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        load_config(config_path)

        self.api_key = os.environ.get("CLAUD_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01',
        }
        self.model_endpoint = API_URL

    def _request(self, messages, json_mode):
        if not self.api_key:
            raise ValueError("CLAUD_API_KEY is not set. Add it in the sidebar or in Streamlit secrets.")

        system = messages[0]["content"]
        user = messages[1]["content"]

        if json_mode:
            system = (
                f"{system}. Your output must be json formatted. Just return the specified "
                f"json format, do not prepend your response with anything."
            )

        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": MAX_TOKENS,
            "temperature": self.temperature,
        }

        data = post_json(self.model_endpoint, self.headers, payload, "Claude")

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
        if not text:
            raise ValueError(f"Claude returned no text (stop_reason: {data.get('stop_reason', 'unknown')})")
        return text


class ClaudJSONModel(_BaseClaudModel):
    def invoke(self, messages):
        try:
            text = self._request(messages, json_mode=True)
            return HumanMessage(content=json.dumps(parse_json_payload(text)))
        except Exception as e:
            return as_error(e)


class ClaudModel(_BaseClaudModel):
    def invoke(self, messages):
        try:
            return HumanMessage(content=self._request(messages, json_mode=False))
        except Exception as e:
            return as_error(e)
