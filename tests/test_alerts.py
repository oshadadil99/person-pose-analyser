import pytest

from src.alerts import decision_timeline, evaluate_alerts, label_events, overall_level
from src.config import load_config
from src.events import Event
from src.features import FrameFeatures
from src.state_machine import Segment
from src.states import State

L, SB, SO, ST, W, OOB, LO, U = (State.LYING_IN_BED, State.SITTING_ON_BED, State.SITTING_OUTSIDE_BED,
                               State.STANDING, State.WALKING, State.OUT_OF_BED,
                               State.LYING_OUTSIDE_BED, State.UNKNOWN)


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def segs(*spec):
    out, t = [], 0.0
    for state, dur in spec:
        out.append(Segment(t, t + dur, state, 0.8))
        t += dur
    return out


def feats_for(end_sec, fps=5, two_people=None):
    """One feature row per frame; two_people = (t0, t1) range where a caregiver is in view."""
    rows = []
    for i in range(int(end_sec * fps)):
        t = i / fps
        n = 2 if two_people and two_people[0] <= t < two_people[1] else 1
        rows.append(FrameFeatures(i, t, visible=True, num_persons=n))
    return rows


def exit_event(start, confirmed):
    return Event("bed_exit", start, confirmed, SB, W, 0.8)


def return_event(start, confirmed):
    return Event("return_to_bed", start, confirmed, W, L, 0.8)


def rules(decisions):
    return [(d.level, d.rule) for d in decisions]


def test_normal_night_has_no_decisions(cfg):
    s = segs((L, 600), (SB, 60), (L, 600))
    d = evaluate_alerts(s, [], feats_for(1260), cfg)
    assert d == [] and overall_level(d) == "NORMAL"
    assert decision_timeline(d, 0, 1260) == [(0, 1260, "NORMAL")]


def test_prolonged_sitting_on_bed(cfg):
    d = evaluate_alerts(segs((L, 100), (SB, 200), (L, 100)), [], feats_for(400), cfg)
    assert rules(d) == [("MONITOR", "prolonged_sitting_on_bed")]
    assert d[0].start_sec == 100 + 180 and d[0].end_sec == 300


def test_short_sitting_is_normal(cfg):
    assert evaluate_alerts(segs((L, 100), (SB, 100), (L, 100)), [], feats_for(300), cfg) == []


def test_prolonged_unknown(cfg):
    assert rules(evaluate_alerts(segs((L, 100), (U, 90)), [], feats_for(190), cfg)) == [("MONITOR", "prolonged_unknown")]
    assert evaluate_alerts(segs((L, 100), (U, 30), (L, 100)), [], feats_for(230), cfg) == []


def test_possible_fall(cfg):
    d = evaluate_alerts(segs((W, 30), (LO, 40)), [], feats_for(70), cfg)
    assert rules(d) == [("ALERT", "possible_fall")]
    assert d[0].start_sec == 30 + 20
    # lying outside the bed briefly (e.g. picked something up, misread pose) is not an alert
    assert evaluate_alerts(segs((W, 30), (LO, 10), (W, 30)), [], feats_for(70), cfg) == []


def test_bed_exit_is_monitor_until_return(cfg):
    s = segs((L, 100), (ST, 5), (W, 95), (SB, 10), (L, 100))
    events = [exit_event(100, 108), return_event(200, 210)]
    d = evaluate_alerts(s, events, feats_for(310), cfg)
    assert rules(d) == [("MONITOR", "bed_exit")]
    assert d[0].start_sec == 108 and d[0].end_sec == 200


def test_bed_exit_monitor_can_be_switched_off(cfg):
    cfg["alerts"]["monitor_every_bed_exit"] = False
    s = segs((L, 100), (ST, 5), (W, 95), (SB, 10), (L, 100))
    assert evaluate_alerts(s, [exit_event(100, 108), return_event(200, 210)], feats_for(310), cfg) == []


def test_long_absence_escalates(cfg):
    # exit at 100 s, never comes back, video ends at 1500 s (1400 s away)
    s = segs((L, 100), (ST, 5), (W, 20), (OOB, 1375))
    d = evaluate_alerts(s, [exit_event(100, 108)], feats_for(1500), cfg)
    assert rules(d) == [("MONITOR", "bed_exit"), ("MONITOR", "out_of_bed_long"), ("ALERT", "prolonged_absence")]
    assert d[1].start_sec == 100 + 600 and d[2].start_sec == 100 + 1200
    tl = decision_timeline(d, 0, 1500)
    assert tl == [(0, 108, "NORMAL"), (108, 1300, "MONITOR"), (1300, 1500, "ALERT")]


def test_caregiver_lowers_absence_rules(cfg):
    s = segs((L, 100), (ST, 5), (W, 20), (OOB, 1375))
    feats = feats_for(1500, two_people=(150, 200))      # caregiver in view for 50 s
    d = evaluate_alerts(s, [exit_event(100, 108)], feats, cfg)
    # absence ALERT -> MONITOR, out_of_bed_long MONITOR -> NORMAL (dropped); the exit itself stays MONITOR
    assert rules(d) == [("MONITOR", "bed_exit"), ("MONITOR", "prolonged_absence")]
    assert "caregiver" in d[1].reason


def test_caregiver_never_lowers_possible_fall(cfg):
    feats = feats_for(70, two_people=(0, 70))
    d = evaluate_alerts(segs((W, 30), (LO, 40)), [], feats, cfg)
    assert rules(d) == [("ALERT", "possible_fall")]


def test_brief_caregiver_glimpse_does_not_count(cfg):
    s = segs((L, 100), (ST, 5), (W, 20), (OOB, 1375))
    feats = feats_for(1500, two_people=(150, 152))      # 2 s, below caregiver_min_sec
    d = evaluate_alerts(s, [exit_event(100, 108)], feats, cfg)
    assert ("ALERT", "prolonged_absence") in rules(d)


def test_timeline_takes_highest_active_level(cfg):
    s = segs((W, 30), (LO, 40), (U, 100))
    d = evaluate_alerts(s, [], feats_for(170), cfg)
    tl = decision_timeline(d, 0, 170)
    assert tl == [(0, 50, "NORMAL"), (50, 70, "ALERT"), (70, 130, "NORMAL"), (130, 170, "MONITOR")]


def test_events_get_a_decision(cfg):
    events = [exit_event(100, 108), return_event(200, 210)]
    label_events(events, cfg)
    assert [e.decision for e in events] == ["MONITOR", "NORMAL"]
