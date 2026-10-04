import pytest

from src.agent.investigator import Step, investigate
from src.agent.policy_llm import LLMPolicy, sanitize
from src.agent.tools import AgentContext
from src.config import load_config
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.states import State
from src.triggers import Trigger
from tests.gemini_fakes import FakeClient, call

SB, ST, W, LO = State.SITTING_ON_BED, State.STANDING, State.WALKING, State.LYING_OUTSIDE_BED


@pytest.fixture
def cfg(tmp_path):
    c = load_config("configs/default.yaml")
    c["vlm"].update(cache_dir=str(tmp_path / "vlm"), llm_cache_dir=str(tmp_path / "llm"),
                    min_delay_sec=0, backoff_base_sec=0, max_retries=0, fallback_retries=0)
    return c


def ctx_for(cfg):
    segs = [Segment(0, 20, SB, 0.8), Segment(20, 23, ST, 0.8), Segment(23, 45, W, 0.8)]
    feats, states = [], []
    for i in range(225):
        t = i / 5
        st = next(s.state for s in segs if s.start_sec <= t < s.end_sec)
        feats.append(FrameFeatures(i, t, True, 1, kp_conf_mean=0.8, bed_edge_dist=-1.0, hip_speed=0.0))
        states.append(FrameState(t, st, 0.8, ""))
    return AgentContext(segs, states, feats, [], [1], cfg)


def exit_trigger():
    return Trigger("bed_event_check", 20, 26, {"event": "bed_exit", "start_sec": 20, "confidence": 0.2,
                                               "previous_state": "sitting_on_bed", "current_state": "walking"})


def fall_trigger():
    return Trigger("possible_fall", 30, 60, {"duration_sec": 30})


def test_gemini_drives_the_loop(cfg):
    client = FakeClient([
        call("look_back", t=20, seconds=10, thought="was the patient in bed?"),
        call("look_forward", t=20, seconds=15, thought="did they move away?"),
        call("conclude", outcome="confirmed", state="NONE", confidence=0.9, reason="in bed, then walked off",
             thought="clear"),
    ])
    tr = investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(client, cfg))
    assert tr.policy == "llm"
    assert [s.tool for s in tr.steps] == ["look_back", "look_forward"]
    assert tr.steps[0].thought == "was the patient in bed?"
    assert tr.conclusion.outcome == "confirmed" and tr.conclusion.confidence == 0.9


def test_thought_signature_is_sent_back(cfg):
    # Gemini 3 rejects a replayed function call that lost its thought signature
    client = FakeClient([
        call("look_back", signature=b"sig-1", t=20, seconds=10, thought="x"),
        call("conclude", outcome="confirmed", state="NONE", confidence=0.9, reason="r", thought="y"),
    ])
    investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(client, cfg))
    second_request = client.models.contents[1]
    replayed_call = second_request[1].parts[0]
    assert replayed_call.function_call.name == "look_back"
    assert replayed_call.thought_signature == b"sig-1"


def test_answers_are_cached(cfg):
    script = [call("look_back", t=20, seconds=10, thought="x"),
              call("conclude", outcome="confirmed", state="NONE", confidence=0.9, reason="r", thought="y")]
    investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(FakeClient(script), cfg))
    replay = FakeClient([])                                   # would fail if it had to call the API
    tr = investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(replay, cfg))
    assert tr.conclusion.outcome == "confirmed" and replay.models.calls == []


def test_after_the_budget_only_conclude_is_allowed(cfg):
    script = [call("look_back", t=20, seconds=10, thought="x")] * 4 + \
             [call("conclude", outcome="unresolved", state="NONE", confidence=0.3, reason="r", thought="y")]
    client = FakeClient(script)
    tr = investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(client, cfg))
    assert len(tr.steps) == 4
    last_config = client.models.calls[-1][1]
    assert last_config.tool_config.function_calling_config.allowed_function_names == ["conclude"]


def test_gemini_failure_falls_back_to_rules(cfg):
    tr = investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(FakeClient([]), cfg))
    assert tr.policy == "rules (llm failed)"
    assert tr.conclusion.outcome == "confirmed"               # the rule policy handles it


def test_bad_tool_arguments_become_an_error_result(cfg):
    client = FakeClient([call("look_back", when=20, thought="typo in args"),
                         call("conclude", outcome="unresolved", state="NONE", confidence=0.2, reason="r", thought="y")])
    tr = investigate(exit_trigger(), ctx_for(cfg), "t1", LLMPolicy(client, cfg))
    assert "error" in tr.steps[0].result


# ---- sanitize: the model can't override safety

def step(tool, **result):
    return Step("", tool, {}, dict(result, summary=""))


def test_fall_cannot_be_dismissed_without_evidence(cfg):
    args = dict(outcome="rejected", state="LYING_IN_BED", confidence=0.9, reason="probably fine")
    c = sanitize(fall_trigger(), args, [step("check_bed_overlap", bed_edge_dist=-2.0)], cfg)
    assert c.outcome == "unresolved" and c.state is None


def test_fall_dismissed_with_bed_evidence(cfg):
    args = dict(outcome="rejected", state="NONE", confidence=0.8, reason="hips on the bed edge")
    c = sanitize(fall_trigger(), args, [step("check_bed_overlap", bed_edge_dist=-0.1)], cfg)
    assert c.outcome == "rejected" and c.state == State.LYING_IN_BED and c.apply_range == (30, 60)

    vlm_says_bed = [step("ask_vlm", state="LYING_IN_BED", confidence=0.9)]
    assert sanitize(fall_trigger(), args, vlm_says_bed, cfg).outcome == "rejected"


def test_outcome_must_fit_the_trigger(cfg):
    args = dict(outcome="resolved", state="WALKING", confidence=0.9, reason="r")
    assert sanitize(exit_trigger(), args, [], cfg).outcome == "unresolved"
    people = Trigger("multiple_people", 0, 5, {"max_people": 2})
    assert sanitize(people, dict(outcome="confirmed", state="NONE", confidence=1, reason="r"), [], cfg).outcome == "noted"


def test_resolved_needs_a_valid_state(cfg):
    unknown = Trigger("unknown_stretch", 10, 20, {"duration_sec": 10})
    ok = sanitize(unknown, dict(outcome="resolved", state="STANDING", confidence=0.7, reason="r"), [], cfg)
    assert ok.state == State.STANDING and ok.apply_range == (10, 20)
    bad = sanitize(unknown, dict(outcome="resolved", state="NONE", confidence=0.7, reason="r"), [], cfg)
    assert bad.outcome == "unresolved"
