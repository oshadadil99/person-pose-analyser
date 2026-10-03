import pytest

from src.config import load_config
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import build_timeline
from src.states import State

FPS = 5
L, SB, SO, ST, W, U = (State.LYING_IN_BED, State.SITTING_ON_BED, State.SITTING_OUTSIDE_BED,
                       State.STANDING, State.WALKING, State.UNKNOWN)


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def make(spec, conf=0.8):
    """spec: list of (state, n_frames) or (state, n_frames, visible). Returns (states, feats)."""
    states, feats, i = [], [], 0
    for item in spec:
        state, n = item[0], item[1]
        visible = item[2] if len(item) > 2 else True
        for _ in range(n):
            t = i / FPS
            c = 0.0 if state == U else conf
            states.append(FrameState(t, state, c, "test"))
            feats.append(FrameFeatures(i, t, visible=visible, num_persons=int(visible),
                                       hip_in_bed=state in (L, SB) if visible else None))
            i += 1
    return states, feats


def run(cfg, spec, conf=0.8):
    states, feats = make(spec, conf)
    end = len(states) / FPS
    return build_timeline(states, feats, cfg, 0.0, end)


def seg_states(segments):
    return [s.state for s in segments]


def test_single_frame_flicker_is_removed(cfg):
    _, segs = run(cfg, [(L, 20), (SB, 1), (L, 20), (ST, 1), (L, 20)])
    assert seg_states(segs) == [L]


def test_short_change_below_min_dwell_is_ignored(cfg):
    # sitting up for 1.2 s (< 2 s dwell for sitting_on_bed) is not a new segment
    _, segs = run(cfg, [(L, 30), (SB, 6), (L, 30)])
    assert seg_states(segs) == [L]


def test_real_change_is_back_dated(cfg):
    # 10 s lying, then 10 s sitting: the boundary must be at 10.0 s,
    # not at 12.0 s when the 2 s dwell was satisfied
    _, segs = run(cfg, [(L, 50), (SB, 50)])
    assert seg_states(segs) == [L, SB]
    assert segs[1].start_sec == pytest.approx(10.0, abs=0.4)


def test_segments_tile_the_whole_range(cfg):
    _, segs = run(cfg, [(L, 40), (SB, 25), (ST, 10), (W, 30), (SO, 40), (W, 20), (SB, 15), (L, 40)])
    assert segs[0].start_sec == 0.0
    assert segs[-1].end_sec == pytest.approx(220 / FPS)
    for a, b in zip(segs, segs[1:]):
        assert a.end_sec == b.start_sec          # no gaps, no overlaps
    assert sum(s.duration_sec for s in segs) == pytest.approx(220 / FPS)


def test_full_exit_and_return_sequence(cfg):
    _, segs = run(cfg, [(L, 50), (SB, 20), (ST, 10), (W, 30), (SO, 50), (W, 20), (SB, 20), (L, 50)])
    assert seg_states(segs) == [L, SB, ST, W, SO, W, SB, L]


def test_blanket_hold_keeps_lying(cfg):
    # 10 s of nothing visible while lying in bed (pulled the duvet over): still lying
    _, segs = run(cfg, [(L, 30), (U, 50, False), (L, 30)])
    assert seg_states(segs) == [L]


def test_blanket_hold_expires(cfg):
    # gone for 60 s with no exit seen: after the 30 s hold it becomes UNKNOWN, not OUT_OF_BED
    _, segs = run(cfg, [(L, 30), (U, 300, False)])
    assert seg_states(segs) == [L, U]
    assert segs[1].start_sec == pytest.approx(6.0 + 30.0, abs=0.5)


def test_disappearing_after_walking_is_out_of_bed(cfg):
    _, segs = run(cfg, [(L, 30), (SB, 20), (ST, 10), (W, 30), (U, 100, False)])
    assert seg_states(segs)[-1] == State.OUT_OF_BED


def test_unreadable_but_visible_is_not_out_of_bed(cfg):
    # pose unclear (visible=True) after walking: we can see someone, just not their posture
    _, segs = run(cfg, [(W, 30), (U, 50, True)])
    assert seg_states(segs) == [W, U]


def test_low_confidence_frames_dont_outvote_confident_ones(cfg):
    states, feats = make([(SB, 40)])
    for i in range(0, 40, 3):    # every third frame a shaky STANDING guess
        states[i] = FrameState(states[i].t_sec, ST, 0.15, "shaky")
    final, segs = build_timeline(states, feats, cfg, 0.0, 8.0)
    assert seg_states(segs) == [SB]


def test_weak_guesses_beat_unknown_frames(cfg):
    # legs-hidden walking: low-confidence guesses mixed with a few unreadable frames
    states, feats = make([(W, 30)], conf=0.12)
    for i in range(1, 30, 4):
        states[i] = FrameState(states[i].t_sec, U, 0.0, "pose unclear")
    _, segs = build_timeline(states, feats, cfg, 0.0, 6.0)
    assert seg_states(segs) == [W]


def test_lone_guess_does_not_break_long_unknown(cfg):
    states, feats = make([(U, 40)])
    states[20] = FrameState(states[20].t_sec, W, 0.12, "stray")
    _, segs = build_timeline(states, feats, cfg, 0.0, 8.0)
    assert seg_states(segs) == [U]


def test_empty_input(cfg):
    final, segs = build_timeline([], [], cfg, 0.0, 10.0)
    assert final == [] and seg_states(segs) == [U] and segs[0].duration_sec == 10.0
