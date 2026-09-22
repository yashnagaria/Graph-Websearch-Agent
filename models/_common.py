"""Shared HTTP plumbing for the REST-based model clients.

Every provider client in this package owes the graph one guarantee: **it never raises**.
An agent that throws takes the whole workflow down with it and discards every step
completed so far, so a failed call has to come back as an ``{"error": ...}`` payload
that the reviewer and router can see and route around.

This module holds the parts that would otherwise be copy-pasted into each client:
retrying transient failures, surfacing the provider's own error text (which is what
tells you a model id was decommissioned), and parsing JSON that arrived wrapped in a
code fence.
"""

import json
import time

import requests
from langchain_core.messages.human import HumanMessage

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = (2, 5, 12)
DEFAULT_TIMEOUT = 120


def api_error_message(data):
    """Pull the provider's own error text out of a response body, if there is one.

    Providers disagree on the shape: OpenAI/Groq/Gemini nest it under
    ``{"error": {"message": ...}}``, Ollama returns a bare ``{"error": "..."}`` string.
    """
    if not isinstance(data, dict):
        return None

    error = data.get("error")
    if isinstance(error, dict):
        return error.get("message") or json.dumps(error)
    if isinstance(error, str):
        return error
    if isinstance(data.get("detail"), str):
        return data["detail"]
    return None


def post_json(url, headers, payload, provider, timeout=DEFAULT_TIMEOUT):
    """POST a JSON payload, retrying transient failures, and return the parsed body.

    Raises ``ValueError`` with a message the user can act on - notably the provider's
    own wording for a retired model, which is far more useful than "no choices in
    response".
    """
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=timeout
            )
        except requests.RequestException as network_error:
            last_error = network_error
            print(f"{provider}: request failed ({network_error}); attempt {attempt + 1}/{MAX_ATTEMPTS}")
            continue

        print(f"{provider}: HTTP {response.status_code}")

        if response.status_code in RETRY_STATUSES:
            last_error = ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
            print(f"{provider}: transient {response.status_code}; attempt {attempt + 1}/{MAX_ATTEMPTS}")
            continue

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise ValueError(
                f"{provider} returned a non-JSON response "
                f"(HTTP {response.status_code}): {response.text[:300]}"
            )

        message = api_error_message(data)
        if message:
            raise ValueError(f"{provider} API error: {message}")
        if not response.ok:
            raise ValueError(f"{provider} returned HTTP {response.status_code}: {response.text[:300]}")

        return data

    raise ValueError(f"{provider} is unavailable after {MAX_ATTEMPTS} attempts. Last error: {last_error}")


def openai_choice_text(data, provider):
    """Read the text out of an OpenAI-shaped response (Groq, vLLM, OpenAI-compatible)."""
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"{provider} returned no choices")

    message = choices[0].get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        reason = choices[0].get("finish_reason", "unknown")
        raise ValueError(f"{provider} returned empty content (finish_reason: {reason})")

    return text


def strip_code_fence(text):
    """Models often wrap JSON in ```json fences even when told not to."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    if "\n" in stripped:
        stripped = stripped.split("\n", 1)[1]
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def parse_json_payload(text):
    """Parse a model's JSON answer, tolerating fences and single-element arrays."""
    parsed = json.loads(strip_code_fence(text))
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), {})
    return parsed


def as_error(exc):
    """The uniform failure return value: an error payload, never an exception."""
    text = f"Error in invoking model! {exc}"
    print("ERROR", text)
    return HumanMessage(content=json.dumps({"error": text}))
