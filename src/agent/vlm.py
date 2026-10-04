"""Gemini as a vision-language model: the `ask_vlm` tool's backend.

Called only for the few moments the agent can't settle from the timeline.
It sends 2-4 downscaled JPEG frames plus a question and gets back structured
JSON: {state, confidence, answer, reason}.

Built to never break a run:
- Disk cache keyed by hash(frames + question + model). Reruns, ablations and
  tests cost zero API calls and give the same answers.
- Minimum delay between calls, retries with exponential backoff (2, 4, 8,
  16 s) on rate limits / server errors, then the fallback model.
- The answer must be valid JSON with a known state and a 0-1 confidence;
  anything else counts as a failure.
- If everything fails: {state: UNKNOWN, confidence: 0, reason: vlm_unavailable}.
  The agent then treats the question as unanswered.

The same `make_client` works for an AI Studio API key or Vertex AI; only the
way the client is created differs (decision D27). Keys come from the
environment and are never logged.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

import cv2

from src.config import get_env
from src.states import State
from src.video_io import read_frame_at, resize_to_width

log = logging.getLogger(__name__)

RETRY_CODES = {429, 500, 502, 503, 504}
VLM_STATES = [s.value for s in State]

PROMPT = """These {n} frames come from a fixed camera in the bedroom of an elderly person, \
taken at {times} seconds into the video.{marked}

Question: {question}

Possible states of the patient:
- LYING_IN_BED: lying on the bed
- SITTING_ON_BED: sitting on the bed, including on its edge
- SITTING_OUTSIDE_BED: sitting on a chair, stool, couch, etc.
- STANDING: upright, not walking
- WALKING: upright and moving
- LYING_OUTSIDE_BED: lying on the floor or anywhere that is not the bed
- OUT_OF_BED: the patient is not in the picture
- UNKNOWN: you can't tell

Answer in JSON. "state" is the patient's state (if it changes across the frames, \
the state in the last frame). "confidence" is 0 to 1. "answer" answers the question \
in one sentence. "reason" says what you see that supports it. If you can't tell, \
use UNKNOWN with low confidence."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "state": {"type": "STRING", "enum": VLM_STATES},
        "confidence": {"type": "NUMBER"},
        "answer": {"type": "STRING"},
        "reason": {"type": "STRING"},
    },
    "required": ["state", "confidence", "answer", "reason"],
}

UNAVAILABLE = {"state": "UNKNOWN", "confidence": 0.0, "answer": "", "reason": "vlm_unavailable"}


def make_client(provider: str):
    """Gemini client for 'aistudio' or 'vertex'; None for 'none'."""
    if provider == "none":
        return None
    from google import genai
    if provider == "aistudio":
        return genai.Client(api_key=get_env("GEMINI_API_KEY"))
    if provider == "vertex":
        return genai.Client(vertexai=True, project=get_env("GOOGLE_CLOUD_PROJECT"),
                            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    raise ValueError(f"Unknown vlm provider: {provider}")


def validate(data) -> dict | None:
    """The VLM's JSON, cleaned up, or None if it doesn't match what we asked for."""
    if not isinstance(data, dict) or data.get("state") not in VLM_STATES:
        return None
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    return {"state": data["state"], "confidence": round(min(max(conf, 0.0), 1.0), 2),
            "answer": str(data.get("answer", "")), "reason": str(data.get("reason", ""))}


class GeminiCaller:
    """Shared plumbing for VLM and LLM-policy calls: min delay, retries, fallback model."""

    def __init__(self, client, cfg: dict):
        self.client = client
        self.vc = cfg["vlm"]
        self.last_call = 0.0
        self.stats = {"api_calls": 0, "cache_hits": 0, "failures": 0, "fallback_used": 0}

    def _wait(self) -> None:
        gap = time.monotonic() - self.last_call
        if gap < self.vc["min_delay_sec"]:
            time.sleep(self.vc["min_delay_sec"] - gap)
        self.last_call = time.monotonic()

    def call(self, contents, config, parse: Callable):
        """Try the main model, then the fallback. `parse(response)` returns a value or None (invalid).
        Returns (value, model_used) or (None, None) if everything failed."""
        from google.genai import errors
        for model, retries in ((self.vc["model"], self.vc["max_retries"]),
                               (self.vc["fallback_model"], self.vc["fallback_retries"])):
            for attempt in range(retries + 1):
                self._wait()
                self.stats["api_calls"] += 1
                try:
                    value = parse(self.client.models.generate_content(model=model, contents=contents, config=config))
                    if value is not None:
                        if model != self.vc["model"]:
                            self.stats["fallback_used"] += 1
                        return value, model
                    log.warning("%s returned an invalid answer (attempt %d)", model, attempt + 1)
                except errors.APIError as e:
                    if e.code not in RETRY_CODES:
                        log.warning("%s failed with %s, trying the next model", model, e.code)
                        break
                    log.warning("%s: %s, retrying", model, e.code)
                except Exception as e:     # network problems etc.
                    log.warning("%s call failed: %s", model, type(e).__name__)
                if attempt < retries:
                    time.sleep(self.vc["backoff_base_sec"] * 2 ** attempt)
        self.stats["failures"] += 1
        return None, None


class VLM:
    """Callable used by the ask_vlm tool: vlm(times, question) -> dict."""

    def __init__(self, client, cfg: dict, video_path: str,
                 patient_box_at: Callable[[float], list[float] | None] | None = None):
        self.caller = GeminiCaller(client, cfg)
        self.vc = cfg["vlm"]
        self.video_path = video_path
        self.patient_box_at = patient_box_at
        self.cache_dir = Path(self.vc["cache_dir"])

    @property
    def stats(self) -> dict:
        return self.caller.stats

    def _jpeg(self, t: float) -> bytes | None:
        frame = read_frame_at(self.video_path, t)
        if frame is None:
            return None
        box = self.patient_box_at(t) if (self.patient_box_at and self.vc["mark_patient"]) else None
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), max(2, frame.shape[1] // 300))
        small, _ = resize_to_width(frame, self.vc["image_max_width"])
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.vc["jpeg_quality"]])
        return buf.tobytes() if ok else None

    def __call__(self, times: list[float], question: str) -> dict:
        from google.genai import types
        images = [(t, j) for t in times if (j := self._jpeg(t)) is not None]
        if not images:
            return dict(UNAVAILABLE, reason="no frames could be read")
        marked = any(self.patient_box_at and self.patient_box_at(t) for t, _ in images) and self.vc["mark_patient"]
        prompt = PROMPT.format(n=len(images), times=", ".join(f"{t:.1f}" for t, _ in images), question=question,
                               marked=" The patient is marked with a green box." if marked else "")

        key = hashlib.sha256((self.vc["model"] + prompt).encode() + b"".join(j for _, j in images)).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            self.caller.stats["cache_hits"] += 1
            return json.loads(cached.read_text(encoding="utf-8"))

        contents = [types.Part.from_bytes(data=j, mime_type="image/jpeg") for _, j in images] + [prompt]
        config = types.GenerateContentConfig(
            temperature=self.vc["temperature"], response_mime_type="application/json", response_schema=RESPONSE_SCHEMA,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))

        def parse(resp):
            try:
                return validate(json.loads(resp.text))
            except (json.JSONDecodeError, TypeError):
                return None

        answer, model = self.caller.call(contents, config, parse)
        if answer is None:
            return dict(UNAVAILABLE)
        answer["model"] = model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(answer, indent=2), encoding="utf-8")
        return answer
