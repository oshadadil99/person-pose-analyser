import pytest

from src.agent.investigator import apply_to_events, apply_to_segments, investigate
from src.agent.tools import AgentContext, look_back, look_forward
from src.config import load_config
from src.events import Event
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.states import State
from src.triggers import Trigger

L, SB, SO, ST, W, LO, U = (State.LYING_IN_BED, State.SITTING_ON_BED, State.SITTING_OUTSIDE_BED,
                           State.STANDING, State.WALKING, State.LYING_OUTSIDE_BED, State.UNKNOWN)
FPS = 5


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def make_ctx(cfg, spec, edge_dist=-1.0, visible=True, vlm=None):
    """Timeline from (state, seconds) pairs, with per-frame features that match it."""
    segments, t = [], 0.0
    for state, dur in spec:
        segments.append(Segment(t, t + dur, state, 0.0 if state == U else 0.8))
        t += dur
    feats, states = [], []
    for i in range(int(t * FPS)):
        ts = i / FPS
        st = next(s.state for s in segments if s.start_sec <= ts < s.end_sec)
        feats.append(FrameFeatures(i, ts, visible, 1, torso_angle_deg=85.0 if st in (L, LO) else 5.0,
                                   kp_conf_mean=0.8, bed_edge_dist=edge_dist, bed_overlap=0.0,
                                   hip_in_bed=edge_dist >= 0, hip_speed=0.0))
        states.append(FrameState(ts, st, 0.8, ""))
    return AgentContext(segments, states, feats, [], [1], cfg, vlm)


def fake_vlm(state, conf=0.85):
    calls = []

    def vlm(times, question):
        calls.append((times, question))
        return {"state": state, "confidence": conf, "reason": f"looks like {state.lower()}"}
    vlm.calls = calls
    return vlm


def exit_trigger(start=20.0, conf=0.2):
    return Trigger("bed_event_check", start, start + 6, {"event": "bed_exit", "start_sec": start, "confidence": conf,
                                                         "previous_state": "sitting_on_bed", "current_state": "walking"})


# ---- tools

def test_look_back_and_forward(cfg):
    ctx = make_ctx(cfg, [(L, 10), (SB, 10), (ST, 3), (W, 20)])
    back = look_back(ctx, 20, 10)
    assert back["in_bed_sec"] == 10 and back["out_of_bed_sec"] == 0
    fwd = look_forward(ctx, 20, 10)
    assert fwd["out_of_bed_sec"] == 10 and "walking" in fwd["summary"]


# ---- bed events

def test_clear_exit_is_confirmed_with_two_calls(cfg):
    ctx = make_ctx(cfg, [(L, 10), (SB, 10), (ST, 3), (W, 20)])
    tr = investigate(exit_trigger(), ctx, "t1")
    assert tr.conclusion.outcome == "confirmed"
    assert [s.tool for s in tr.steps] == ["look_back", "look_forward"]
    assert tr.conclusion.confidence > 0.9


def test_ambiguous_exit_asks_the_vlm(cfg):
    spec = [(SB, 20), (ST, 2), (U, 13), (W, 5)]          # mostly unknown after standing up
    offline = investigate(exit_trigger(), make_ctx(cfg, spec), "t1")
    assert [s.tool for s in offline.steps] == ["look_back", "look_forward", "ask_vlm"]
    assert offline.conclusion.outcome == "unresolved"

    vlm = fake_vlm("WALKING")
    online = investigate(exit_trigger(), make_ctx(cfg, spec, vlm=vlm), "t2")
    assert online.conclusion.outcome == "confirmed" and len(vlm.calls) == 1


def test_exit_that_goes_straight_back_is_rejected(cfg):
    ctx = make_ctx(cfg, [(SB, 20), (ST, 2), (SB, 18)])
    tr = investigate(exit_trigger(), ctx, "t1")
    assert tr.conclusion.outcome == "rejected"


def test_apply_to_events(cfg):
    ctx = make_ctx(cfg, [(L, 10), (SB, 10), (ST, 3), (W, 20)])
    tr = investigate(exit_trigger(), ctx, "trace_0001")
    ev = [Event("bed_exit", 20.0, 26.0, SB, W, 0.2)]
    out = apply_to_events(ev, [tr], 5.0)
    assert out[0].agent_trace_id == "trace_0001" and out[0].confidence > 0.9

    rejected = investigate(exit_trigger(), make_ctx(cfg, [(SB, 20), (ST, 2), (SB, 18)]), "trace_0002")
    assert apply_to_events([Event("bed_exit", 20.0, 26.0, SB, W, 0.2)], [rejected], 5.0) == []


# ---- possible fall

def fall_trigger():
    return Trigger("possible_fall", 30.0, 60.0, {"duration_sec": 30.0})


def test_lying_at_the_bed_edge_is_not_a_fall(cfg):
    ctx = make_ctx(cfg, [(W, 30), (LO, 30)], edge_dist=-0.1)     # just past the polygon edge
    tr = investigate(fall_trigger(), ctx, "t1")
    assert tr.conclusion.outcome == "rejected" and tr.conclusion.state == L
    fixed = apply_to_segments(ctx.segments, [tr])
    assert [s.state for s in fixed] == [W, L]


def test_unconfirmed_fall_is_kept(cfg):
    ctx = make_ctx(cfg, [(W, 30), (LO, 30)], edge_dist=-2.0)     # well away from the bed
    tr = investigate(fall_trigger(), ctx, "t1")
    assert len(tr.steps) == cfg["agent"]["max_tool_calls"]       # used the whole budget
    assert tr.conclusion.outcome == "unresolved" and tr.conclusion.state is None
    assert [s.state for s in apply_to_segments(ctx.segments, [tr])] == [W, LO]   # alert stays


def test_vlm_confirms_fall(cfg):
    ctx = make_ctx(cfg, [(W, 30), (LO, 30)], edge_dist=-2.0, vlm=fake_vlm("LYING_OUTSIDE_BED"))
    assert investigate(fall_trigger(), ctx, "t1").conclusion.outcome == "confirmed"


def test_vlm_says_on_the_bed(cfg):
    ctx = make_ctx(cfg, [(W, 30), (LO, 30)], edge_dist=-2.0, vlm=fake_vlm("LYING_IN_BED"))
    tr = investigate(fall_trigger(), ctx, "t1")
    assert tr.conclusion.outcome == "rejected" and tr.conclusion.state == L


def test_unsure_vlm_answer_is_ignored(cfg):
    ctx = make_ctx(cfg, [(W, 30), (LO, 30)], edge_dist=-2.0, vlm=fake_vlm("LYING_IN_BED", conf=0.3))
    assert investigate(fall_trigger(), ctx, "t1").conclusion.outcome == "unresolved"


# ---- unknown stretches

def test_short_unknown_between_same_states_is_filled(cfg):
    ctx = make_ctx(cfg, [(ST, 20), (U, 8), (ST, 20)])
    tr = investigate(Trigger("unknown_stretch", 20, 28, {"duration_sec": 8}), ctx, "t1")
    assert tr.conclusion.outcome == "resolved" and tr.conclusion.state == ST
    assert [s.state for s in apply_to_segments(ctx.segments, [tr])] == [ST]


def test_unknown_in_bed_on_both_sides_is_lying(cfg):
    ctx = make_ctx(cfg, [(L, 30), (U, 120), (L, 30)], visible=False)
    tr = investigate(Trigger("unknown_stretch", 30, 150, {"duration_sec": 120}), ctx, "t1")
    assert tr.conclusion.state == L


def test_unknown_between_different_states_stays_unknown_offline(cfg):
    ctx = make_ctx(cfg, [(W, 20), (U, 40), (SO, 20)], visible=False)
    tr = investigate(Trigger("unknown_stretch", 20, 60, {"duration_sec": 40}), ctx, "t1")
    assert tr.conclusion.outcome == "unresolved"
    assert tr.steps[-1].tool == "ask_vlm"


# ---- loop limits and trace format

def test_tool_budget_is_enforced(cfg):
    cfg["agent"]["max_tool_calls"] = 1
    ctx = make_ctx(cfg, [(L, 10), (SB, 10), (ST, 3), (W, 20)])
    tr = investigate(exit_trigger(), ctx, "t1")
    assert len(tr.steps) == 1 and tr.conclusion.outcome == "unresolved"
    assert "budget" in tr.conclusion.reason


def test_trace_reads_like_the_assignment_example(cfg):
    ctx = make_ctx(cfg, [(L, 10), (SB, 10), (ST, 3), (W, 20)])
    text = investigate(exit_trigger(), ctx, "trace_0001").to_text()
    for word in ("Observation:", "Thought:", "Action:", "Finding:", "Conclusion:"):
        assert word in text
    assert "look_back(t=20.0, seconds=10.0)" in text
