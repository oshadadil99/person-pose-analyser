"""Bed-exit and return-to-bed events, detected from the timeline segments.

First every state is mapped to a simple bed status:
    IN_BED   lying in bed, sitting on bed
    OUT      standing, walking, sitting outside, lying outside, out of view
    UNKNOWN  unknown
UNKNOWN segments don't change the status: an event can be detected across
them, but with lower confidence because we didn't see the whole sequence.

BED_EXIT (in bed -> up -> moving away), for each OUT stretch that follows IN_BED:
  start_time     = first moment out of bed (standing up)
  confirmed_time = "moving away" (walking, chair, lying outside, out of view)
                   has lasted exit_confirm_sec; or, if the person only stands
                   there, after brief_stand_max_sec of being up.
  If they're back in bed before confirmed_time it is NOT an exit: that covers
  "stand briefly and sit back down". Sitting up or edge sitting never leave
  IN_BED, so they can't start an exit at all.

RETURN_TO_BED (out -> sits on bed -> lies down), for each IN_BED stretch after OUT:
  start_time     = sits (or lies) on the bed
  confirmed_time = lies down, or has stayed on the bed for return_settle_sec.
  Sitting on the bed for a few seconds and leaving again is not a return.

OUT_OF_BED as a state means "not visible after a CONFIRMED bed exit"
(assignment definition). The state machine marks it from the last known
state; here any OUT_OF_BED that isn't after a confirmed exit becomes UNKNOWN.
"""

from dataclasses import dataclass

from src.state_machine import Segment, merge_segments
from src.states import IN_BED_STATES, State
from src.summary import format_clock

AWAY_STATES = {State.WALKING, State.SITTING_OUTSIDE_BED, State.LYING_OUTSIDE_BED, State.OUT_OF_BED}


def bed_status(state: State) -> str:
    if state in IN_BED_STATES:
        return "IN_BED"
    if state == State.UNKNOWN:
        return "UNKNOWN"
    return "OUT"


@dataclass
class Event:
    event: str                    # "bed_exit" or "return_to_bed"
    start_sec: float
    confirmed_sec: float
    previous_state: State
    current_state: State
    confidence: float
    note: str = ""
    decision: str | None = None   # filled by the alert engine (Phase 7)
    agent_trace_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "start_time": format_clock(self.start_sec, with_hours=True),
            "confirmed_time": format_clock(self.confirmed_sec, with_hours=True),
            "start_sec": round(self.start_sec, 2),
            "confirmed_sec": round(self.confirmed_sec, 2),
            "previous_state": self.previous_state.value.lower(),
            "current_state": self.current_state.value.lower(),
            "confidence": self.confidence,
            "decision": self.decision,
            "agent_trace_id": self.agent_trace_id,
            "note": self.note,
        }


def _state_at(segments: list[Segment], t: float) -> State:
    for s in segments:
        if s.start_sec <= t < s.end_sec:
            return s.state
    return segments[-1].state


def _mean_conf(segments: list[Segment], t0: float, t1: float) -> float:
    """Duration-weighted mean confidence of the known segments overlapping [t0, t1]."""
    total = weight = 0.0
    for s in segments:
        overlap = min(s.end_sec, t1) - max(s.start_sec, t0)
        if overlap > 0 and s.state != State.UNKNOWN:
            total += s.confidence * overlap
            weight += overlap
    return total / weight if weight else 0.0


def _stretches(segments: list[Segment]) -> list[tuple[str, int, int]]:
    """Group consecutive segments by bed status, skipping UNKNOWN.

    Returns (status, first segment index, last segment index). UNKNOWN segments
    between two stretches of the same status are absorbed into it."""
    out: list[list] = []
    for i, s in enumerate(segments):
        st = bed_status(s.state)
        if st == "UNKNOWN":
            continue
        if out and out[-1][0] == st:
            out[-1][2] = i
        else:
            out.append([st, i, i])
    return [tuple(x) for x in out]


def _unknown_between(segments: list[Segment], t0: float, t1: float) -> bool:
    return any(s.state == State.UNKNOWN and s.start_sec < t1 and s.end_sec > t0 for s in segments)


def detect_events(segments: list[Segment], cfg: dict) -> tuple[list[Event], list[Segment]]:
    """Returns (events in time order, segments with unconfirmed OUT_OF_BED turned into UNKNOWN)."""
    ec = cfg["events"]
    events: list[Event] = []
    exit_periods: list[tuple[float, float]] = []   # time ranges after a confirmed exit, before a return
    stretches = _stretches(segments)
    end_of_video = segments[-1].end_sec if segments else 0.0

    # Where the patient officially is. It only changes when an event is confirmed,
    # so a 4 s sit on the bed doesn't make the next walk-off a "bed exit".
    # At the start of the video it's simply wherever we first see them.
    where = stretches[0][0] if stretches else None
    exited_at = None
    for k, (status, i0, i1) in enumerate(stretches):
        stretch_start = segments[i0].start_sec
        stretch_end = segments[stretches[k + 1][1]].start_sec if k + 1 < len(stretches) else end_of_video
        prev = stretches[k - 1] if k > 0 else None
        prev_end = segments[prev[2]].end_sec if prev else 0.0

        if status == "OUT" and prev is not None and where == "IN_BED":
            away = next((segments[j].start_sec for j in range(i0, i1 + 1) if segments[j].state in AWAY_STATES), None)
            if away is not None:
                confirm = away + ec["exit_confirm_sec"]          # moved away and kept going
            else:
                confirm = stretch_start + ec["brief_stand_max_sec"]   # only stood there, but for a while
            # Did we have to look across UNKNOWN to see this sequence?
            gap = _unknown_between(segments, prev_end, confirm)
            factor = ec["unknown_gap_conf_factor"] if gap else 1.0
            if confirm <= stretch_end:
                events.append(Event("bed_exit", stretch_start, confirm,
                                    previous_state=segments[prev[2]].state,
                                    current_state=_state_at(segments, confirm - 1e-6),
                                    confidence=round(_mean_conf(segments, stretch_start, confirm) * factor, 2),
                                    note="crossed an unknown stretch" if gap else ""))
                exited_at, where = stretch_start, "OUT"

        elif status == "IN_BED" and prev is not None and where == "OUT":
            lying = next((segments[j].start_sec for j in range(i0, i1 + 1)
                          if segments[j].state == State.LYING_IN_BED), None)
            settle = stretch_start + ec["return_settle_sec"]
            confirm = min(lying, settle) if lying is not None else settle
            gap = _unknown_between(segments, prev_end, confirm)
            factor = ec["unknown_gap_conf_factor"] if gap else 1.0
            if confirm <= stretch_end:
                events.append(Event("return_to_bed", stretch_start, confirm,
                                    previous_state=segments[prev[2]].state,
                                    current_state=_state_at(segments, confirm + 1e-6),
                                    confidence=round(_mean_conf(segments, stretch_start, confirm) * factor, 2),
                                    note="crossed an unknown stretch" if gap else ""))
                if exited_at is not None:
                    exit_periods.append((exited_at, stretch_start))
                exited_at, where = None, "IN_BED"

    if exited_at is not None:
        exit_periods.append((exited_at, end_of_video))
    return events, _fix_out_of_bed(segments, exit_periods)


def _fix_out_of_bed(segments: list[Segment], exit_periods: list[tuple[float, float]]) -> list[Segment]:
    """OUT_OF_BED only counts after a confirmed exit; otherwise it is UNKNOWN. Re-merge neighbours."""
    fixed = []
    for s in segments:
        state = s.state
        if state == State.OUT_OF_BED and not any(a <= s.start_sec < b for a, b in exit_periods):
            state = State.UNKNOWN
        fixed.append(Segment(s.start_sec, s.end_sec, state, s.confidence))
    return merge_segments(fixed)


def bed_status_segments(segments: list[Segment]) -> list[tuple[float, float, str]]:
    """The coarse timeline: (start, end, IN_BED / OUT / UNKNOWN), neighbours merged."""
    out: list[list] = []
    for s in segments:
        st = bed_status(s.state)
        if out and out[-1][2] == st:
            out[-1][1] = s.end_sec
        else:
            out.append([s.start_sec, s.end_sec, st])
    return [tuple(x) for x in out]


def out_of_bed_periods(segments: list[Segment]) -> list[dict]:
    """Unbroken stretches not in bed (UNKNOWN included, same convention as the summary)."""
    periods, start = [], None
    for s in segments:
        if s.state in IN_BED_STATES:
            if start is not None:
                periods.append((start, s.start_sec))
                start = None
        elif start is None:
            start = s.start_sec
    if start is not None and segments:
        periods.append((start, segments[-1].end_sec))
    return [{"start": format_clock(a), "end": format_clock(b), "duration_sec": round(b - a, 1)} for a, b in periods]
