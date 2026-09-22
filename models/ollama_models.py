"""Ollama clients - a local model server, so no API key is involved.

The endpoint defaults to the standard local daemon but can be pointed elsewhere with
``OLLAMA_BASE_URL``, which is what lets a hosted app talk to a tunnelled local Ollama.
Timeouts are generous because local models on CPU are slow to first token.
"""

import json
import os

from langchain_core.messages.human import HumanMessage

from models._common import as_error, parse_json_payload, post_json

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3:instruct"
OLLAMA_TIMEOUT = 600


def _base_url():
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class _BaseOllamaModel:
    def __init__(self, temperature=0, model=DEFAULT_MODEL, model_endpoint=None):
        self.temperature = temperature
        self.model = model or DEFAULT_MODEL
        self.headers = {"Content-Type": "application/json"}
        base = (model_endpoint or _base_url()).rstrip("/")
        self.model_endpoint = base if base.endswith("/api/generate") else f"{base}/api/generate"

    def _request(self, messages, json_mode):
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        data = post_json(
            self.model_endpoint, self.headers, payload, "Ollama", timeout=OLLAMA_TIMEOUT
        )

        text = (data.get("response") or "").strip()
        if not text:
            raise ValueError("Ollama returned an empty response")
        return text


class OllamaJSONModel(_BaseOllamaModel):
    def invoke(self, messages):
        try:
            text = self._request(messages, json_mode=True)
            return HumanMessage(content=json.dumps(parse_json_payload(text)))
        except Exception as e:
            return as_error(e)


class OllamaModel(_BaseOllamaModel):
    def invoke(self, messages):
        try:
            return HumanMessage(content=self._request(messages, json_mode=False))
        except Exception as e:
            return as_error(e)
