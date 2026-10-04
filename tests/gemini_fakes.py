"""A fake Gemini client for tests: returns scripted responses, records every call."""

import json
from types import SimpleNamespace

from google.genai import errors


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []        # (model, config) per call

    def generate_content(self, model, contents, config):
        self.calls.append((model, config))
        self.contents = getattr(self, "contents", []) + [contents]
        if not self.responses:
            raise errors.APIError(503, {"error": {"message": "no more scripted responses"}})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def text(data) -> SimpleNamespace:
    return SimpleNamespace(text=data if isinstance(data, str) else json.dumps(data), function_calls=None)


def call(name: str, signature: bytes | None = None, **args) -> SimpleNamespace:
    """A function-call response. Gemini 3 also attaches a thought signature to the call part."""
    part = SimpleNamespace(thought_signature=signature)
    return SimpleNamespace(text=None, function_calls=[SimpleNamespace(name=name, args=args)],
                           candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": f"fake {code}"}})
