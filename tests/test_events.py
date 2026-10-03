import pytest

from src.config import load_config
from src.events import bed_status_segments, detect_events, out_of_bed_periods
from src.state_machine import Segment
from src.states import State

L, SB, SO, ST, W, OOB, LO, U = (State.LYING_IN_BED, State.SITTING_ON_BED, State.SITTING_OUTSIDE_BED,
                               State.STANDING, State.WALKING, State.OUT_OF_BED,
                               State.LYING_OUTSIDE_BED, State.UNKNOWN)


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def segs(*spec, conf=0.8):
    """spec: (state, duration_sec), ... -> back-to-back segments starting at 0."""
    out, t = [], 0.0
    for state, dur in spec:
        out.append(Segment(t, t + dur, state, 0.0 if state == U else conf))
        t += dur
    return out


def kinds(events):
    return [e.event for e in events]


# ---- bed exit ----

def test_full_bed_exit(cfg):
    events, _ = detect_events(segs((L, 60), (SB, 20), (ST, 5), (W, 35)), cfg)
    assert kinds(events) == ["bed_exit"]
    e = events[0]
    assert e.start_sec == 80                 # stood up
    assert e.confirmed_sec == 85 + 3         # walking away for exit_confirm_sec
    assert e.previous_state == SB and e.current_state == W
    assert e.confidence == pytest.approx(0.8)


def test_sitting_up_is_not_an_exit(cfg):
    events, _ = detect_events(segs((L, 60), (SB, 20), (L, 60)), cfg)
    assert events == []


def test_long_edge_sitting_is_not_an_exit(cfg):
    events, _ = detect_events(segs((L, 60), (SB, 300), (L, 60)), cfg)
    assert events == []


def test_brief_stand_then_sit_back_is_not_an_exit(cfg):
    events, _ = detect_events(segs((L, 60), (SB, 10), (ST, 5), (SB, 20), (L, 60)), cfg)
    assert events == []


def test_stand_and_two_steps_then_sit_back_is_not_an_exit(cfg):
    # walked for 1.5 s (< exit_confirm_sec) and sat straight back down
    events, _ = detect_events(segs((SB, 20), (ST, 2), (W, 1.5), (SB, 30)), cfg)
    assert events == []


def test_standing_by_the_bed_for_a_long_time_counts_as_up(cfg):
    # 15 s standing (> brief_stand_max_sec) = up; sitting back for 30 s = returned
    events, _ = detect_events(segs((SB, 20), (ST, 15), (SB, 30)), cfg)
    assert kinds(events) == ["bed_exit", "return_to_bed"]
    assert events[0].confirmed_sec == 20 + 10    # brief_stand_max_sec


def test_exit_to_chair(cfg):
    events, _ = detect_events(segs((L, 30), (SB, 10), (ST, 3), (SO, 60)), cfg)
    assert kinds(events) == ["bed_exit"]
    assert events[0].current_state == SO


def test_exit_across_unknown_has_lower_confidence(cfg):
    events, _ = detect_events(segs((SB, 20), (U, 7), (W, 30)), cfg)
    assert kinds(events) == ["bed_exit"]
    assert events[0].start_sec == 27
    assert events[0].confidence == pytest.approx(0.8 * cfg["events"]["unknown_gap_conf_factor"])
    assert "unknown" in events[0].note


def test_video_ending_before_confirmation_has_no_exit(cfg):
    events, _ = detect_events(segs((SB, 20), (ST, 2), (W, 1)), cfg)
    assert events == []


# ---- return to bed ----

def test_exit_and_return(cfg):
    events, _ = detect_events(segs((L, 60), (SB, 10), (ST, 5), (W, 60), (SB, 5), (L, 100)), cfg)
    assert kinds(events) == ["bed_exit", "return_to_bed"]
    r = events[1]
    assert r.start_sec == 135                # sat on the bed
    assert r.confirmed_sec == 140            # lay down (before the 10 s settle time)
    assert r.previous_state == W and r.current_state == L


def test_return_confirmed_by_whichever_comes_first(cfg):
    # sat for 20 s before lying down: the 10 s of sitting already confirms the return
    events, _ = detect_events(segs((W, 30), (SB, 20), (L, 60)), cfg)
    assert kinds(events) == ["return_to_bed"]
    assert events[0].confirmed_sec == 40 and events[0].current_state == SB


def test_brief_sit_does_not_enable_a_new_exit(cfg):
    # real exit, then sits on the bed 4 s (not a return) and walks off again:
    # still only the one exit, the second walk-off is not a new exit
    events, _ = detect_events(segs((L, 30), (ST, 3), (W, 30), (SB, 4), (W, 30)), cfg)
    assert kinds(events) == ["bed_exit"]


def test_return_by_sitting_long_enough(cfg):
    events, _ = detect_events(segs((W, 30), (SB, 40)), cfg)
    assert kinds(events) == ["return_to_bed"]
    assert events[0].confirmed_sec == 30 + cfg["events"]["return_settle_sec"]


def test_sitting_on_bed_briefly_then_leaving_is_not_a_return(cfg):
    events, _ = detect_events(segs((W, 30), (SB, 4), (W, 30)), cfg)
    assert events == []


def test_two_exits_and_two_returns(cfg):
    spec = [(L, 60), (SB, 10), (ST, 3), (W, 40), (SB, 15), (L, 120),
            (SB, 10), (ST, 3), (W, 20), (SO, 60), (W, 10), (SB, 5), (L, 60)]
    events, _ = detect_events(segs(*spec), cfg)
    assert kinds(events) == ["bed_exit", "return_to_bed", "bed_exit", "return_to_bed"]


# ---- OUT_OF_BED only after a confirmed exit ----

def test_out_of_view_after_exit_stays_out_of_bed(cfg):
    _, fixed = detect_events(segs((L, 30), (SB, 10), (ST, 3), (W, 10), (OOB, 120)), cfg)
    assert fixed[-1].state == OOB


def test_out_of_bed_without_confirmed_exit_becomes_unknown(cfg):
    # video starts with the person walking, then they leave: never seen leaving the BED
    _, fixed = detect_events(segs((W, 20), (OOB, 60)), cfg)
    assert [s.state for s in fixed] == [W, U]


def test_unknown_neighbours_merge_after_fix(cfg):
    _, fixed = detect_events(segs((W, 20), (U, 10), (OOB, 60)), cfg)
    assert [s.state for s in fixed] == [W, U]
    assert fixed[-1].start_sec == 20 and fixed[-1].end_sec == 90


# ---- bed status layer ----

def test_bed_status_timeline(cfg):
    s = segs((L, 60), (SB, 20), (ST, 5), (W, 35), (SO, 40), (SB, 10), (L, 30))
    assert bed_status_segments(s) == [(0, 80, "IN_BED"), (80, 160, "OUT"), (160, 200, "IN_BED")]


def test_out_of_bed_periods(cfg):
    s = segs((L, 60), (ST, 5), (W, 35), (SB, 10), (L, 30), (W, 20))
    periods = out_of_bed_periods(s)
    assert [p["duration_sec"] for p in periods] == [40, 20]
    assert periods[0]["start"] == "01:00"
