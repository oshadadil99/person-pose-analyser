"""Deterministic agent policy: decides the next tool call, or concludes.

This is the offline controller (no LLM). For each trigger type it follows a
short investigation plan, but it looks at each finding before choosing the
next step, and stops as soon as the evidence is clear. The VLM is only asked
when the cheap tools leave the question open.

  possible_fall    check_bed_overlap -> (in bed? reject) -> get_pose_summary
                   -> look_back -> ask_vlm -> conclude
  bed_event_check  look_back -> look_forward -> (clear? conclude) -> ask_vlm -> conclude
  unknown_stretch  get_pose_summary -> look_back -> look_forward
                   -> (same state both sides? fill) -> ask_vlm -> conclude
  multiple_people  count_people -> look_forward -> note (no state change)
  flicker          look_forward -> get_pose_summary -> note (no state change)

A conclusion can change the timeline (fill an UNKNOWN stretch, turn a false
"lying outside the bed" into lying in bed), raise or lower a bed event's
confidence, or just record what was found. "unresolved" changes nothing:
the state stays as it was and the alert rules treat it as usual (UNKNOWN
gets MONITOR, a possible fall keeps its ALERT). The agent can only remove a
fall alert when it has positive evidence the person is on the bed.
"""

from dataclasses import dataclass, field

from src.states import IN_BED_STATES, State
from src.triggers import Trigger

OUT_STATES = {State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED, State.LYING_OUTSIDE_BED, State.OUT_OF_BED}
AWAY_STATES = {State.WALKING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED, State.LYING_OUTSIDE_BED}


@dataclass
class Conclusion:
    outcome: str                         # confirmed / rejected / resolved / unresolved / noted
    reason: str
    confidence: float = 0.0
    state: State | None = None           # state to write over apply_range, if any
    apply_range: tuple[float, float] | None = None


@dataclass
class Action:
    thought: str
    tool: str | None = None              # None means "conclude"
    args: dict = field(default_factory=dict)
    conclusion: Conclusion | None = None


def _results(steps) -> dict:
    """Latest result per tool name, for easy lookup."""
    return {s.tool: s.result for s in steps}


def _states_in(window: dict) -> list[State]:
    return [State(p["state"].upper()) for p in window["states"]]


def _vlm_state(res: dict, cfg: dict) -> State | None:
    """The VLM's answer, or None if it's unavailable / unsure."""
    try:
        state = State(str(res.get("state", "UNKNOWN")).upper())
    except ValueError:
        return None
    if state == State.UNKNOWN or res.get("confidence", 0) < cfg["agent"]["min_confidence"]:
        return None
    return state


class RulePolicy:
    name = "rules"

    def next_action(self, trigger: Trigger, steps: list, cfg: dict) -> Action:
        return next_action(trigger, steps, cfg)


def next_action(trigger: Trigger, steps: list, cfg: dict) -> Action:
    plan = {"possible_fall": _possible_fall, "bed_event_check": _bed_event, "unknown_stretch": _unknown,
            "multiple_people": _people, "flicker": _flicker}[trigger.kind]
    return plan(trigger, steps, cfg)


def _frames(trigger: Trigger, cfg: dict) -> list[float]:
    """Evenly spaced times inside the trigger window for the VLM."""
    n = cfg["agent"]["vlm_frames"]
    a, b = trigger.t_start, max(trigger.t_end, trigger.t_start + 1.0)
    return [round(a + (b - a) * (i + 0.5) / n, 2) for i in range(n)]


# ---------------------------------------------------------------- possible fall

def _possible_fall(trig: Trigger, steps: list, cfg: dict) -> Action:
    a, b = trig.t_start, trig.t_end
    mid = (a + b) / 2
    r = _results(steps)
    if not steps:
        return Action("Body is horizontal. First check: is this on the bed or on the floor?",
                      "check_bed_overlap", {"t": mid})
    if len(steps) == 1:
        bed = r["check_bed_overlap"]
        margin = cfg["patient"]["bed_margin"]
        dist = bed.get("bed_edge_dist")
        if bed.get("visible") and dist is not None and (dist >= -margin or
                                                         (bed.get("bed_overlap") or 0) >= cfg["frame_rules"]["in_bed_overlap_ratio"]):
            return Action("Hips are on or right at the bed edge, so this is lying in bed, not on the floor.",
                          conclusion=Conclusion("rejected", f"bed region + body position: {bed['summary']}",
                                                0.7, State.LYING_IN_BED, (a, b)))
        return Action("Hips are clearly outside the bed. Check the posture over the whole stretch.",
                      "get_pose_summary", {"t": a, "seconds": b - a})
    if len(steps) == 2:
        return Action("What was the person doing just before lying down?", "look_back",
                      {"t": a, "seconds": cfg["agent"]["look_back_sec"]})
    if len(steps) == 3:
        return Action("Geometry says floor. Ask the VLM to look at the frames before trusting it.", "ask_vlm",
                      {"times": _frames(trig, cfg), "question": "Is the person lying on the floor, or on the bed?"})
    vlm = _vlm_state(r["ask_vlm"], cfg)
    before = r["look_back"]["summary"]
    if vlm == State.LYING_IN_BED:
        return Action("The VLM sees the person on the bed.", conclusion=Conclusion(
            "rejected", f"VLM: {r['ask_vlm'].get('reason')}", r["ask_vlm"]["confidence"], State.LYING_IN_BED, (a, b)))
    if vlm == State.LYING_OUTSIDE_BED:
        return Action("The VLM confirms the person is on the floor.", conclusion=Conclusion(
            "confirmed", f"possible fall confirmed by VLM; before: {before}", r["ask_vlm"]["confidence"]))
    return Action("No visual confirmation. Keep it as a possible fall: missing a fall is worse than a false alarm.",
                  conclusion=Conclusion("unresolved", f"geometry only: {r['check_bed_overlap']['summary']}; "
                                                      f"{r['get_pose_summary']['summary']}; before: {before}", 0.5))


# ---------------------------------------------------------------- bed exit / return

def _bed_event(trig: Trigger, steps: list, cfg: dict) -> Action:
    ac = cfg["agent"]
    ev = trig.evidence
    is_exit = ev["event"] == "bed_exit"
    t = ev["start_sec"]
    r = _results(steps)
    if not steps:
        return Action(f"Current frames are not enough to be sure this is a real {ev['event']}. "
                      "Check what happened before.", "look_back", {"t": t, "seconds": ac["look_back_sec"]})
    if len(steps) == 1:
        return Action("Now check what happens after.", "look_forward", {"t": t, "seconds": ac["look_forward_sec"]})

    back, fwd = r["look_back"], r["look_forward"]
    known = fwd["in_bed_sec"] + fwd["out_of_bed_sec"]
    if is_exit:
        before_ok = back["in_bed_sec"] >= ac["min_in_bed_before_sec"]
        share = fwd["out_of_bed_sec"] / known if known else 0.0
        moving_away = any(s in AWAY_STATES for s in _states_in(fwd))
        clear_yes = before_ok and share >= ac["out_share_confirm"] and moving_away
        clear_no = known > 0 and share < 1 - ac["out_share_confirm"]
    else:
        before_ok = back["out_of_bed_sec"] > 0
        share = fwd["in_bed_sec"] / known if known else 0.0
        clear_yes = before_ok and share >= ac["out_share_confirm"]
        clear_no = known > 0 and share < 1 - ac["out_share_confirm"]
    covered = known / max(fwd["to"] - fwd["from"], 1e-6)
    name = "BED_EXIT" if is_exit else "RETURN_TO_BED"

    if len(steps) == 2:
        if clear_yes:
            conf = round(0.5 + 0.45 * share * covered, 2)
            done = "in bed before, out of bed and moving away after" if is_exit else "out of bed before, back in bed after"
            return Action(f"{done.capitalize()}: the sequence is clear.",
                          conclusion=Conclusion("confirmed", f"{name} confirmed: before {back['summary']}; "
                                                             f"after {fwd['summary']}", conf))
        if clear_no:
            return Action(f"After: {fwd['summary']}. The person is mostly back where they started.",
                          conclusion=Conclusion("rejected", f"not a {name}: after {fwd['summary']}", 0.6))
        return Action("Still ambiguous from the timeline. Ask the VLM.", "ask_vlm", {
            "times": [round(t + d, 2) for d in (0.0, 3.0, 8.0)],
            "question": "Is the person getting out of bed and moving away, or staying in / returning to bed?"})

    vlm = _vlm_state(r["ask_vlm"], cfg)
    if vlm is not None and (vlm in OUT_STATES) == is_exit:
        return Action("The VLM agrees with the event.", conclusion=Conclusion(
            "confirmed", f"{name} confirmed by VLM: {r['ask_vlm'].get('reason')}", r["ask_vlm"]["confidence"]))
    if vlm is not None:
        return Action("The VLM contradicts the event.", conclusion=Conclusion(
            "rejected", f"VLM says {vlm.value.lower()}: {r['ask_vlm'].get('reason')}", r["ask_vlm"]["confidence"]))
    return Action("Can't settle it. Keep the event with low confidence.",
                  conclusion=Conclusion("unresolved", f"ambiguous: before {back['summary']}; after {fwd['summary']}",
                                        trig.evidence["confidence"]))


# ---------------------------------------------------------------- unknown stretch

def _unknown(trig: Trigger, steps: list, cfg: dict) -> Action:
    ac = cfg["agent"]
    a, b = trig.t_start, trig.t_end
    r = _results(steps)
    if not steps:
        return Action("Why is this unknown: is the patient out of view, or just hard to read?",
                      "get_pose_summary", {"t": a, "seconds": b - a})
    if len(steps) == 1:
        return Action("Check the state right before.", "look_back", {"t": a, "seconds": ac["look_back_sec"]})
    if len(steps) == 2:
        return Action("Check the state right after.", "look_forward", {"t": b, "seconds": ac["look_back_sec"]})

    before = [s for s in _states_in(r["look_back"]) if s != State.UNKNOWN]
    after = [s for s in _states_in(r["look_forward"]) if s != State.UNKNOWN]
    last, first = (before[-1] if before else None), (after[0] if after else None)
    visible = r["get_pose_summary"].get("visible_share", 0.0)
    dur = b - a

    if len(steps) == 3:
        if last is not None and last == first and visible >= 0.5 and dur <= ac["max_fill_sec"]:
            return Action(f"{last.value.lower()} before and after, person visible but hard to read, short gap.",
                          conclusion=Conclusion("resolved", f"same state before and after "
                                                            f"({last.value.lower()}), {dur:.0f}s gap", 0.5, last, (a, b)))
        if last in IN_BED_STATES and first in IN_BED_STATES and dur <= ac["max_fill_in_bed_sec"]:
            return Action("In bed before and after, nobody seen getting up: still in bed, probably covered.",
                          conclusion=Conclusion("resolved", f"in bed before ({last.value.lower()}) and after "
                                                            f"({first.value.lower()}), {dur:.0f}s not readable",
                                                0.45, State.LYING_IN_BED, (a, b)))
        return Action("Context doesn't settle it. Ask the VLM.", "ask_vlm",
                      {"times": _frames(trig, cfg), "question": "What is the person doing?"})

    vlm = _vlm_state(r["ask_vlm"], cfg)
    if vlm is not None:
        return Action("The VLM can see what's happening.", conclusion=Conclusion(
            "resolved", f"VLM: {r['ask_vlm'].get('reason')}", r["ask_vlm"]["confidence"], vlm, (a, b)))
    return Action("Can't tell. Leave it UNKNOWN.", conclusion=Conclusion(
        "unresolved", f"before: {r['look_back']['summary']}; after: {r['look_forward']['summary']}; "
                      f"{r['get_pose_summary']['summary']}"))


# ---------------------------------------------------------------- multiple people / flicker

def _people(trig: Trigger, steps: list, cfg: dict) -> Action:
    a, b = trig.t_start, trig.t_end
    r = _results(steps)
    if not steps:
        return Action("More than one person. Who is here, and is the patient one of them?",
                      "count_people", {"t": a, "seconds": b - a})
    if len(steps) == 1:
        return Action("What was the patient doing meanwhile?", "look_forward", {"t": a, "seconds": b - a})
    return Action("Second person present: treat as a caregiver/visitor.", conclusion=Conclusion(
        "noted", f"{r['count_people']['summary']}; patient: {r['look_forward']['summary']}", 0.8))


def _flicker(trig: Trigger, steps: list, cfg: dict) -> Action:
    a, b = trig.t_start, trig.t_end
    r = _results(steps)
    if not steps:
        return Action("The per-frame rules kept changing their mind. What did smoothing settle on?",
                      "look_forward", {"t": a, "seconds": b - a})
    if len(steps) == 1:
        return Action("Check the pose quality in this stretch.", "get_pose_summary", {"t": a, "seconds": b - a})
    return Action("Smoothing already chose a state; nothing better to go on without the VLM.", conclusion=Conclusion(
        "noted", f"smoothed result: {r['look_forward']['summary']}; {r['get_pose_summary']['summary']}", 0.5))
