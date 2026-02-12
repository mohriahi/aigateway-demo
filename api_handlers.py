"""
API format handlers for different LLM provider protocols.

Each handler class defines three static methods:
  - build_url(url, model)          -> str
  - build_payload(model, query)    -> dict
  - parse_response(response_json)  -> str | None

The get_handler() function resolves an API_FORMAT name to its handler class.
"""

import re


class OpenAIHandler:
    """Handler for OpenAI-compatible APIs (OpenAI, Mistral, Anthropic via gateway, etc.)."""

    @staticmethod
    def build_url(url, model):
        return url

    @staticmethod
    def build_payload(model, query):
        return {
            "model": model,
            "messages": [
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

    @staticmethod
    def parse_response(response_json):
        if "choices" in response_json and response_json["choices"]:
            return response_json["choices"][0]["message"]["content"]
        return None


class GeminiHandler:
    """Handler for Google Gemini API format."""

    @staticmethod
    def build_url(url, model):
        if "{model}" in url:
            return url.replace("{model}", model)
        if "/models/" in url:
            return re.sub(r'/models/[^/:]+', f'/models/{model}', url)
        return url

    @staticmethod
    def build_payload(model, query):
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": query}],
                }
            ]
        }

    @staticmethod
    def parse_response(response_json):
        if "candidates" in response_json and response_json["candidates"]:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        return None


# ---- Registry ----

_HANDLERS = {
    "openai": OpenAIHandler,
    "gemini": GeminiHandler,
}


def get_handler(api_format):
    """
    Return the handler class for the given API format name.

    Args:
        api_format: Format string from config.yaml API_FORMAT field.
                    Defaults to "openai" if None or empty.

    Returns:
        Handler class with build_url, build_payload, parse_response methods.

    Raises:
        ValueError: If the api_format is not recognized.
    """
    fmt = (api_format or "openai").lower()
    if fmt not in _HANDLERS:
        raise ValueError(
            f"Unknown API_FORMAT '{api_format}'. "
            f"Available formats: {', '.join(_HANDLERS.keys())}"
        )
    return _HANDLERS[fmt]
