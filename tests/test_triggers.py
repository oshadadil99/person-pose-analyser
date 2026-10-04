import pytest

from src.config import load_config
from src.events import Event
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.states import State
from src.triggers import find_triggers

L, SB, ST, W, LO, U = (State.LYING_IN_BED, State.SITTING_ON_BED, State.STANDING, State.WALKING,
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


def frames(end, fps=5, people=None):
    feats, raw = [], []
    for i in range(int(end * fps)):
        t = i / fps
        n = 2 if people and people[0] <= t < people[1] else 1
        feats.append(FrameFeatures(i, t, True, n))
        raw.append(FrameState(t, L, 0.8, ""))
    return feats, raw


def kinds(trigs):
    return [t.kind for t in trigs]


def test_clear_night_has_no_triggers(cfg):
    feats, raw = frames(100)
    assert find_triggers(segs((L, 100)), [], raw, feats, cfg) == []


def test_low_confidence_event_triggers_a_check(cfg):
    feats, raw = frames(60)
    low = Event("bed_exit", 20, 26, SB, W, 0.17)
    high = Event("return_to_bed", 40, 45, W, L, 0.9)
    trigs = find_triggers(segs((SB, 20), (W, 20), (L, 20)), [low, high], raw, feats, cfg)
    assert kinds(trigs) == ["bed_event_check"]
    assert trigs[0].evidence["start_sec"] == 20


def test_lying_outside_bed_and_long_unknown(cfg):
    feats, raw = frames(100)
    trigs = find_triggers(segs((W, 20), (LO, 30), (U, 8), (W, 10), (U, 3), (W, 29)), [], raw, feats, cfg)
    assert kinds(trigs) == ["possible_fall", "unknown_stretch"]     # the 3 s unknown is ignored


def test_second_person_for_a_while(cfg):
    feats, raw = frames(60, people=(10, 20))
    assert kinds(find_triggers(segs((L, 60)), [], raw, feats, cfg)) == ["multiple_people"]
    feats, raw = frames(60, people=(10, 11))                       # 1 s glimpse
    assert find_triggers(segs((L, 60)), [], raw, feats, cfg) == []


def test_flicker_zone(cfg):
    feats, raw = frames(30)
    for i in range(50, 60):                                        # 10 s..12 s alternates
        raw[i] = FrameState(raw[i].t_sec, ST if i % 2 else SB, 0.3, "")
    assert kinds(find_triggers(segs((SB, 30)), [], raw, feats, cfg)) == ["flicker"]


def test_flicker_inside_another_trigger_is_dropped(cfg):
    feats, raw = frames(30)
    for i in range(50, 60):
        raw[i] = FrameState(raw[i].t_sec, ST if i % 2 else SB, 0.3, "")
    s = segs((SB, 8), (U, 8), (SB, 14))                            # unknown 8-16 s covers the flicker
    assert kinds(find_triggers(s, [], raw, feats, cfg)) == ["unknown_stretch"]


def test_cap_keeps_the_most_important(cfg):
    cfg["agent"]["max_investigations"] = 1
    feats, raw = frames(100)
    trigs = find_triggers(segs((U, 10), (W, 20), (LO, 30), (W, 40)), [], raw, feats, cfg)
    assert kinds(trigs) == ["possible_fall"]
