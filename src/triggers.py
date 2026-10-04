"""Decides which moments the investigator agent should look at.

Most of a video is clear and never wakes the agent. A trigger fires on:
  possible_fall     any LYING_OUTSIDE_BED segment (bed or floor?)
  bed_event_check   a bed exit / return with confidence below agent.min_confidence
  unknown_stretch   UNKNOWN lasting at least triggers.unknown_min_sec
  multiple_people   2+ people in view for triggers.multi_person_min_sec (caregiver?)
  flicker           the raw per-frame state changed 4+ times in 5 s (rules couldn't decide)

If there are more triggers than agent.max_investigations, the most important
kinds are kept (fall first, flicker last). A flicker zone that overlaps a more
important trigger is dropped: that investigation already covers the moment.
"""

from dataclasses import dataclass, field

from src.events import Event
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.states import State
from src.summary import format_clock

PRIORITY = {"possible_fall": 0, "bed_event_check": 1, "unknown_stretch": 2, "multiple_people": 3, "flicker": 4}


@dataclass
class Trigger:
    kind: str
    t_start: float
    t_end: float
    evidence: dict = field(default_factory=dict)

    def describe(self) -> str:
        ev = self.evidence
        when = f"{format_clock(self.t_start)}-{format_clock(self.t_end)}"
        if self.kind == "possible_fall":
            return f"Person lying outside the bed region at {when} ({ev['duration_sec']:.0f}s)."
        if self.kind == "bed_event_check":
            return (f"{ev['event']} detected at {format_clock(self.t_start)} "
                    f"({ev['previous_state']} -> {ev['current_state']}) but confidence is only {ev['confidence']}.")
        if self.kind == "unknown_stretch":
            return f"Activity unknown for {ev['duration_sec']:.0f}s at {when}."
        if self.kind == "multiple_people":
            return f"{ev['max_people']} people in view at {when}."
        return f"State changed {ev['changes']} times in {ev['window_sec']:.0f}s at {when}."


def _people_runs(feats: list[FrameFeatures], min_sec: float) -> list[tuple[float, float, int]]:
    runs, start, peak = [], None, 0
    for i, f in enumerate(feats + [None]):
        many = f is not None and f.num_persons >= 2
        if many and start is None:
            start, peak = f.t_sec, f.num_persons
        elif many:
            peak = max(peak, f.num_persons)
        elif start is not None:
            end = feats[i - 1].t_sec
            if end - start >= min_sec:
                runs.append((start, end, peak))
            start = None
    return runs


def _flicker_zones(raw: list[FrameState], window: float, min_changes: int) -> list[tuple[float, float, int]]:
    zones: list[list] = []
    for i in range(len(raw)):
        j, changes = i, 0
        while j + 1 < len(raw) and raw[j + 1].t_sec - raw[i].t_sec <= window:
            changes += raw[j + 1].state != raw[j].state
            j += 1
        if changes >= min_changes:
            a, b = raw[i].t_sec, raw[j].t_sec
            if zones and a <= zones[-1][1]:
                zones[-1][1] = max(zones[-1][1], b)
                zones[-1][2] = max(zones[-1][2], changes)
            else:
                zones.append([a, b, changes])
    return [tuple(z) for z in zones]


def find_triggers(segments: list[Segment], events: list[Event], raw: list[FrameState],
                  feats: list[FrameFeatures], cfg: dict) -> list[Trigger]:
    tc, ac = cfg["triggers"], cfg["agent"]
    out: list[Trigger] = []

    for s in segments:
        if s.state == State.LYING_OUTSIDE_BED:
            out.append(Trigger("possible_fall", s.start_sec, s.end_sec, {"duration_sec": s.duration_sec}))
        elif s.state == State.UNKNOWN and s.duration_sec >= tc["unknown_min_sec"]:
            out.append(Trigger("unknown_stretch", s.start_sec, s.end_sec, {"duration_sec": s.duration_sec}))

    for e in events:
        if e.confidence < ac["min_confidence"]:
            out.append(Trigger("bed_event_check", e.start_sec, e.confirmed_sec, {
                "event": e.event, "start_sec": e.start_sec, "confidence": e.confidence,
                "previous_state": e.previous_state.value.lower(), "current_state": e.current_state.value.lower()}))

    for a, b, peak in _people_runs(feats, tc["multi_person_min_sec"]):
        out.append(Trigger("multiple_people", a, b, {"max_people": peak}))

    for a, b, n in _flicker_zones(raw, tc["flicker_window_sec"], tc["flicker_min_changes"]):
        if not any(t.t_start < b and a < t.t_end for t in out):
            out.append(Trigger("flicker", a, b, {"changes": n, "window_sec": tc["flicker_window_sec"]}))

    out.sort(key=lambda t: (PRIORITY[t.kind], t.t_start))
    return sorted(out[:ac["max_investigations"]], key=lambda t: t.t_start)
