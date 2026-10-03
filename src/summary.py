"""Durations, bed summary and timeline text from the segments.

Conventions (also stated in the README):
- time in bed      = LYING_IN_BED + SITTING_ON_BED
- time out of bed  = everything else, INCLUDING unknown, so in + out = total.
  Unknown time is also reported on its own so nothing is hidden (decision D12).
- longest out-of-bed period = longest unbroken stretch of segments that are not
  in-bed states (walking -> chair -> walking counts as one period).
Times are kept as float seconds and only formatted here, at output.
"""

from src.state_machine import Segment
from src.states import IN_BED_STATES, State


def format_duration(sec: float) -> str:
    """702 -> '11m 42s', 45 -> '45s', 3725 -> '1h 02m 05s' (assignment style)."""
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_clock(sec: float, with_hours: bool = False) -> str:
    """Timestamp as MM:SS, or HH:MM:SS for long videos / event output."""
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if with_hours or h else f"{m:02d}:{s:02d}"


def state_durations(segments: list[Segment]) -> dict[str, float]:
    durations = {st.value.lower(): 0.0 for st in State}
    for seg in segments:
        durations[seg.state.value.lower()] += seg.duration_sec
    return durations


def longest_out_of_bed(segments: list[Segment]) -> float:
    longest = run = 0.0
    for seg in segments:
        run = 0.0 if seg.state in IN_BED_STATES else run + seg.duration_sec
        longest = max(longest, run)
    return longest


def build_summary(segments: list[Segment], start_sec: float, end_sec: float,
                  bed_exit_count: int = 0, bed_return_count: int = 0) -> dict:
    total = end_sec - start_sec
    dur = state_durations(segments)
    in_bed = sum(dur[s.value.lower()] for s in IN_BED_STATES)

    summary = {
        "analysed_range_sec": [round(start_sec, 1), round(end_sec, 1)],
        "observation_duration_sec": round(total, 1),
        "activity_duration_sec": {k: round(v, 1) for k, v in dur.items()},
        "bed_exit_count": bed_exit_count,
        "bed_return_count": bed_return_count,
        "total_in_bed_sec": round(in_bed, 1),
        "total_out_of_bed_sec": round(total - in_bed, 1),
        "unknown_sec": round(dur["unknown"], 1),
        "longest_out_of_bed_period_sec": round(longest_out_of_bed(segments), 1),
        "final_state": segments[-1].state.value.lower() if segments else "unknown",
    }
    # Same numbers in the human-readable format from the assignment.
    summary["human_readable"] = {
        "total_observation_time": format_duration(total),
        "activity_summary": {k: format_duration(v) for k, v in dur.items() if v > 0},
        "bed_summary": {
            "time_in_bed": format_duration(in_bed),
            "time_out_of_bed": format_duration(total - in_bed),
            "bed_exit_count": bed_exit_count,
        },
    }
    return summary


def timeline_lines(segments: list[Segment],
                   decision_tl: list[tuple[float, float, str]] | None = None) -> list[str]:
    """'00:00 – 04:32  LYING_IN_BED', one line per segment.

    With a decision timeline, each line also shows NORMAL / MONITOR / ALERT, and a
    segment is split where the decision changes inside it (e.g. walking that
    becomes MONITOR once the bed exit is confirmed)."""
    long_video = bool(segments) and segments[-1].end_sec >= 3600
    rows = [(s.start_sec, s.end_sec, s.state.value, None) for s in segments]
    if decision_tl:
        rows = []
        for s in segments:
            for a, b, level in decision_tl:
                lo, hi = max(s.start_sec, a), min(s.end_sec, b)
                if hi > lo:
                    rows.append((lo, hi, s.state.value, level))
    return [f"{format_clock(a, long_video)} – {format_clock(b, long_video)}  {state:20s}{'  ' + lvl if lvl else ''}".rstrip()
            for a, b, state, lvl in rows]
