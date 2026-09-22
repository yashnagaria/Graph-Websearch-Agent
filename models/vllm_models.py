"""vLLM clients - a self-hosted OpenAI-compatible server.

``model_endpoint`` is the server root (e.g. ``https://my-host/``); the chat-completions
path is appended here. MistralAI builds do not take a separate system turn, so the two
messages are merged for that prefix.
"""

import json

from langchain_core.messages.human import HumanMessage

from models._common import (
    as_error,
    openai_choice_text,
    parse_json_payload,
    post_json,
)

VLLM_TIMEOUT = 300


def _endpoint(model_endpoint):
    if not model_endpoint:
        raise ValueError("No vLLM endpoint was provided. Set it in the sidebar.")
    base = model_endpoint.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/v1/chat/completions"


class _BaseVllmModel:
    def __init__(self, temperature=0, model=None, model_endpoint=None, guided_json=None, stop=None):
        self.headers = {"Content-Type": "application/json"}
        self.raw_endpoint = model_endpoint
        self.temperature = temperature
        self.model = model
        self.guided_json = guided_json
        self.stop = stop

    def _messages_payload(self, system, user, json_mode):
        prefix = (self.model or "").split("/")[0]

        # MistralAI builds reject a separate system turn.
        if prefix == "mistralai":
            messages = [{"role": "user", "content": f"system:{system}\n\n user:{user}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stop": self.stop,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            if self.guided_json:
                payload["guided_json"] = self.guided_json
        return payload

    def _request(self, messages, json_mode):
        payload = self._messages_payload(messages[0]["content"], messages[1]["content"], json_mode)
        data = post_json(
            _endpoint(self.raw_endpoint), self.headers, payload, "vLLM", timeout=VLLM_TIMEOUT
        )
        return openai_choice_text(data, "vLLM")


class VllmJSONModel(_BaseVllmModel):
    def invoke(self, messages):
        try:
            text = self._request(messages, json_mode=True)
            return HumanMessage(content=json.dumps(parse_json_payload(text)))
        except Exception as e:
            return as_error(e)


class VllmModel(_BaseVllmModel):
    def invoke(self, messages):
        try:
            return HumanMessage(content=self._request(messages, json_mode=False))
        except Exception as e:
            return as_error(e)
