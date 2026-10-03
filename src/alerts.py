"""NORMAL / MONITOR / ALERT decisions from the timeline and the bed events.

Deterministic rules with thresholds from config, so a decision is always
explainable ("rule X fired because Y lasted Z seconds") and testable. The LLM
never makes this call; at most it improves the states the rules look at.
Each rule's reasoning is in docs/alert_rules.md.

Every rule produces a Decision WINDOW: it becomes active at the moment the
condition is met (e.g. 180 s into a sitting segment) and stays active until
the condition ends. The level at any moment is the highest active window,
NORMAL if none. That gives both a list of decisions with reasons and a
decision timeline.

  MONITOR  prolonged_sitting_on_bed   SITTING_ON_BED > edge_sitting_max_sec
  MONITOR  prolonged_unknown          UNKNOWN > unknown_max_sec
  MONITOR  bed_exit                   every confirmed exit, until the return
  MONITOR  out_of_bed_long            out of bed (after an exit) > out_of_bed_monitor_sec
  ALERT    possible_fall              LYING_OUTSIDE_BED > floor_lying_alert_sec
  ALERT    prolonged_absence          out of bed (after an exit) > absence_alert_sec

A caregiver in the room (2+ people for caregiver_min_sec) lowers the two
absence rules by one level: someone is already with the patient. It never
touches possible_fall: a caregiver being there doesn't make a fall less serious.
"""

from dataclasses import dataclass

from src.events import Event
from src.features import FrameFeatures
from src.state_machine import Segment
from src.states import State
from src.summary import format_clock

LEVELS = ["NORMAL", "MONITOR", "ALERT"]


@dataclass
class Decision:
    level: str          # MONITOR or ALERT
    rule: str
    start_sec: float    # when the rule fired
    end_sec: float      # when its condition ended
    reason: str

    def to_dict(self) -> dict:
        return {"level": self.level, "rule": self.rule,
                "time": format_clock(self.start_sec, with_hours=True),
                "until": format_clock(self.end_sec, with_hours=True),
                "start_sec": round(self.start_sec, 2), "end_sec": round(self.end_sec, 2),
                "reason": self.reason}


def _lower(level: str) -> str:
    return LEVELS[max(0, LEVELS.index(level) - 1)]


def caregiver_seconds(feats: list[FrameFeatures], t0: float, t1: float, fps: float) -> float:
    """How long 2+ people were in view between t0 and t1."""
    return sum(1 for f in feats if t0 <= f.t_sec < t1 and f.num_persons >= 2) / fps


def away_periods(events: list[Event], end_sec: float) -> list[tuple[Event, float]]:
    """(bed exit, end of the time away) for every confirmed exit. Ends at the next return or the video end."""
    out = []
    for i, e in enumerate(events):
        if e.event != "bed_exit":
            continue
        ret = next((r for r in events[i + 1:] if r.event == "return_to_bed"), None)
        out.append((e, ret.start_sec if ret else end_sec))
    return out


def evaluate_alerts(segments: list[Segment], events: list[Event], feats: list[FrameFeatures],
                    cfg: dict) -> list[Decision]:
    ac = cfg["alerts"]
    fps = cfg["sampling"]["fps"]
    end_sec = segments[-1].end_sec if segments else 0.0
    decisions: list[Decision] = []

    # Rules on single segments
    for s in segments:
        if s.state == State.SITTING_ON_BED and s.duration_sec > ac["edge_sitting_max_sec"]:
            decisions.append(Decision("MONITOR", "prolonged_sitting_on_bed", s.start_sec + ac["edge_sitting_max_sec"],
                                      s.end_sec, f"sitting on the bed for {s.duration_sec:.0f}s "
                                                 f"(limit {ac['edge_sitting_max_sec']}s): may be about to get up unassisted"))
        if s.state == State.UNKNOWN and s.duration_sec > ac["unknown_max_sec"]:
            decisions.append(Decision("MONITOR", "prolonged_unknown", s.start_sec + ac["unknown_max_sec"],
                                      s.end_sec, f"activity unknown for {s.duration_sec:.0f}s "
                                                 f"(limit {ac['unknown_max_sec']}s): can't see what is happening"))
        if s.state == State.LYING_OUTSIDE_BED and s.duration_sec > ac["floor_lying_alert_sec"]:
            decisions.append(Decision("ALERT", "possible_fall", s.start_sec + ac["floor_lying_alert_sec"],
                                      s.end_sec, f"lying outside the bed for {s.duration_sec:.0f}s "
                                                 f"(limit {ac['floor_lying_alert_sec']}s): possible fall"))

    # Rules on time away from bed after a confirmed exit
    for exit_ev, away_end in away_periods(events, end_sec):
        away = away_end - exit_ev.start_sec
        carer = caregiver_seconds(feats, exit_ev.start_sec, away_end, fps) >= ac["caregiver_min_sec"]
        rules = []
        if ac["monitor_every_bed_exit"]:
            rules.append(("MONITOR", "bed_exit", exit_ev.confirmed_sec, "confirmed bed exit: fall-risk moment", False))
        if away > ac["out_of_bed_monitor_sec"]:
            rules.append(("MONITOR", "out_of_bed_long", exit_ev.start_sec + ac["out_of_bed_monitor_sec"],
                          f"out of bed for {away:.0f}s (limit {ac['out_of_bed_monitor_sec']}s)", True))
        if away > ac["absence_alert_sec"]:
            rules.append(("ALERT", "prolonged_absence", exit_ev.start_sec + ac["absence_alert_sec"],
                          f"away from bed for {away:.0f}s (limit {ac['absence_alert_sec']}s)", True))
        for level, rule, t, reason, absence_rule in rules:
            if absence_rule and carer and ac["caregiver_downgrades_absence"]:
                level, reason = _lower(level), reason + "; lowered: caregiver present"
            if level != "NORMAL":
                decisions.append(Decision(level, rule, t, away_end, reason))

    return sorted(decisions, key=lambda d: d.start_sec)


def decision_timeline(decisions: list[Decision], start_sec: float, end_sec: float) -> list[tuple[float, float, str]]:
    """Level over time: the highest active decision, NORMAL when none. Neighbours merged."""
    cuts = sorted({start_sec, end_sec} | {d.start_sec for d in decisions} | {d.end_sec for d in decisions})
    cuts = [c for c in cuts if start_sec <= c <= end_sec]
    out: list[list] = []
    for a, b in zip(cuts, cuts[1:]):
        active = [d.level for d in decisions if d.start_sec <= a < d.end_sec]
        level = max(active, key=LEVELS.index) if active else "NORMAL"
        if out and out[-1][2] == level:
            out[-1][1] = b
        else:
            out.append([a, b, level])
    return [tuple(x) for x in out]


def overall_level(decisions: list[Decision]) -> str:
    return max((d.level for d in decisions), key=LEVELS.index, default="NORMAL")


def label_events(events: list[Event], cfg: dict) -> None:
    """Fill the `decision` field of each event (the assignment's event format has one)."""
    for e in events:
        if e.event == "bed_exit":
            e.decision = "MONITOR" if cfg["alerts"]["monitor_every_bed_exit"] else "NORMAL"
        else:
            e.decision = "NORMAL"
