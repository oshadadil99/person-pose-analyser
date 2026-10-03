import pytest

from src.state_machine import Segment
from src.states import State
from src.summary import build_summary, format_clock, format_duration, timeline_lines

# The example timeline from the assignment (15 minutes)
EXAMPLE = [
    Segment(0, 272, State.LYING_IN_BED, 0.9),
    Segment(272, 308, State.SITTING_ON_BED, 0.8),
    Segment(308, 320, State.STANDING, 0.7),
    Segment(320, 461, State.WALKING, 0.8),
    Segment(461, 555, State.SITTING_OUTSIDE_BED, 0.8),
    Segment(555, 582, State.WALKING, 0.8),
    Segment(582, 601, State.SITTING_ON_BED, 0.8),
    Segment(601, 900, State.LYING_IN_BED, 0.9),
]


@pytest.mark.parametrize("sec, text", [
    (702, "11m 42s"), (45, "45s"), (1200, "20m 00s"), (3725, "1h 02m 05s"), (0, "0s"), (59.6, "1m 00s"),
])
def test_format_duration(sec, text):
    assert format_duration(sec) == text


def test_format_clock():
    assert format_clock(272) == "04:32"
    assert format_clock(308, with_hours=True) == "00:05:08"
    assert format_clock(3725) == "01:02:05"


def test_durations_sum_to_observation_time():
    s = build_summary(EXAMPLE, 0, 900)
    assert sum(s["activity_duration_sec"].values()) == pytest.approx(s["observation_duration_sec"])
    assert s["total_in_bed_sec"] + s["total_out_of_bed_sec"] == pytest.approx(900)


def test_in_bed_and_out_of_bed():
    s = build_summary(EXAMPLE, 0, 900)
    assert s["total_in_bed_sec"] == pytest.approx(272 + 36 + 19 + 299)
    assert s["activity_duration_sec"]["walking"] == pytest.approx(141 + 27)
    assert s["final_state"] == "lying_in_bed"


def test_longest_out_of_bed_joins_consecutive_out_states():
    # standing + walking + chair + walking = 308 .. 582 in one go
    assert build_summary(EXAMPLE, 0, 900)["longest_out_of_bed_period_sec"] == pytest.approx(582 - 308)


def test_unknown_counts_as_out_of_bed_and_is_reported():
    segs = [Segment(0, 100, State.LYING_IN_BED, 0.9), Segment(100, 130, State.UNKNOWN, 0.0),
            Segment(130, 200, State.LYING_IN_BED, 0.9)]
    s = build_summary(segs, 0, 200)
    assert s["unknown_sec"] == 30
    assert s["total_out_of_bed_sec"] == 30


def test_human_readable_block():
    hr = build_summary(EXAMPLE, 0, 900, bed_exit_count=1)["human_readable"]
    assert hr["total_observation_time"] == "15m 00s"
    assert hr["activity_summary"]["lying_in_bed"] == "9m 31s"
    assert hr["bed_summary"]["bed_exit_count"] == 1
    assert "out_of_bed" not in hr["activity_summary"]   # zero durations are left out


def test_timeline_lines_match_assignment_format():
    lines = timeline_lines(EXAMPLE)
    assert lines[0] == "00:00 – 04:32  LYING_IN_BED"
    assert lines[3] == "05:20 – 07:41  WALKING"
