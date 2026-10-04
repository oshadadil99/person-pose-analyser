import cv2
import numpy as np
import pytest

from src.agent.vlm import UNAVAILABLE, VLM, make_client, validate
from src.config import load_config
from tests.gemini_fakes import FakeClient, api_error, text

GOOD = {"state": "SITTING_ON_BED", "confidence": 0.9, "answer": "sitting on the bed edge", "reason": "legs down"}


@pytest.fixture
def cfg(tmp_path):
    c = load_config("configs/default.yaml")
    c["vlm"].update(cache_dir=str(tmp_path / "cache"), min_delay_sec=0, backoff_base_sec=0)
    return c


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(50):
        w.write(np.full((240, 320, 3), i * 5 % 255, np.uint8))
    w.release()
    return str(path)


def test_validate():
    assert validate(GOOD)["state"] == "SITTING_ON_BED"
    assert validate({**GOOD, "confidence": 7})["confidence"] == 1.0          # clamped
    assert validate({**GOOD, "state": "DANCING"}) is None
    assert validate({**GOOD, "confidence": "high"}) is None
    assert validate("not a dict") is None


def test_answer_is_parsed_and_cached(cfg, video):
    client = FakeClient([text(GOOD)])
    vlm = VLM(client, cfg, video)
    a = vlm([1.0, 2.0], "What is the person doing?")
    assert a["state"] == "SITTING_ON_BED" and a["model"] == cfg["vlm"]["model"]
    again = VLM(FakeClient([]), cfg, video)([1.0, 2.0], "What is the person doing?")
    assert again["state"] == "SITTING_ON_BED"                                # from disk, no API call
    assert len(client.models.calls) == 1


def test_rate_limit_is_retried(cfg, video):
    client = FakeClient([api_error(429), api_error(503), text(GOOD)])
    vlm = VLM(client, cfg, video)
    assert vlm([1.0], "q")["state"] == "SITTING_ON_BED"
    assert vlm.stats["api_calls"] == 3


def test_fallback_model_after_repeated_failures(cfg, video):
    retries = cfg["vlm"]["max_retries"]
    client = FakeClient([api_error(503)] * (retries + 1) + [text(GOOD)])
    vlm = VLM(client, cfg, video)
    a = vlm([1.0], "q")
    assert a["model"] == cfg["vlm"]["fallback_model"]
    assert vlm.stats["fallback_used"] == 1


def test_invalid_json_counts_as_failure(cfg, video):
    client = FakeClient([text("this is not json"), text({"state": "FLYING"}), text(GOOD)])
    assert VLM(client, cfg, video)([1.0], "q")["state"] == "SITTING_ON_BED"


def test_non_retryable_error_goes_straight_to_fallback(cfg, video):
    client = FakeClient([api_error(400), text(GOOD)])
    a = VLM(client, cfg, video)([1.0], "q")
    assert a["model"] == cfg["vlm"]["fallback_model"]
    assert [m for m, _ in client.models.calls] == [cfg["vlm"]["model"], cfg["vlm"]["fallback_model"]]


def test_total_failure_never_crashes_and_is_not_cached(cfg, video):
    vlm = VLM(FakeClient([]), cfg, video)                                    # every call fails
    a = vlm([1.0], "q")
    assert a == UNAVAILABLE and vlm.stats["failures"] == 1
    later = VLM(FakeClient([text(GOOD)]), cfg, video)([1.0], "q")
    assert later["state"] == "SITTING_ON_BED"                                # failure wasn't cached


def test_unreadable_frames(cfg, video):
    a = VLM(FakeClient([text(GOOD)]), cfg, video)([999.0], "q")
    assert a["state"] == "UNKNOWN"


def test_make_client(monkeypatch):
    assert make_client("none") is None
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_client("aistudio")
