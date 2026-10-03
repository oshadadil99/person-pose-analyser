"""Turns noisy per-frame states into a clean timeline of segments.

Four steps, in this order:

1. Blanket hold. If the patient was lying in bed and then becomes unreadable
   (pose unclear while the box is still on the bed, or not detected at all),
   keep LYING_IN_BED for up to `occlusion_hold_sec`. A person under a duvet
   hasn't gone anywhere; without this every blanket pull would be UNKNOWN.

2. Smoothing. Each frame takes a confidence-weighted vote over a short window
   centred on it (we analyse a recorded video, so looking a little ahead is
   fine). One- or two-frame blips disappear.

3. Hysteresis (minimum dwell). A new state must last `min_dwell_sec[state]`
   before the switch is accepted, so rolling over in bed doesn't create fake
   segments. When it is accepted, the change is BACK-DATED to the first frame
   of the new state; otherwise every segment would start late by the dwell
   time and all durations would be biased.

4. OUT_OF_BED. A patient who disappears after being out of bed (standing,
   walking, sitting in a chair) has left the camera view: OUT_OF_BED.
   Disappearing from the bed stays UNKNOWN, as the assignment requires.
   Phase 6 tightens this to "after a confirmed bed exit".

Every sampled frame covers [its time, next frame's time), and the last one
runs to the end of the analysed range. Segments therefore tile the whole
range with no gaps, so state durations always add up to the video length.
"""

from collections import defaultdict
from dataclasses import dataclass

from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.states import State

OUT_OF_BED_SOURCES = {State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED}


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    state: State
    confidence: float   # mean smoothed confidence of its frames

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def blanket_hold(states: list[FrameState], feats: list[FrameFeatures], cfg: dict) -> list[FrameState]:
    sm = cfg["state_machine"]
    overlap_needed = cfg["frame_rules"]["in_bed_overlap_ratio"]
    out, last_lying_t = [], None
    for st, f in zip(states, feats):
        if st.state == State.LYING_IN_BED:
            last_lying_t = st.t_sec
        elif st.state == State.UNKNOWN and last_lying_t is not None \
                and st.t_sec - last_lying_t <= sm["occlusion_hold_sec"]:
            box_on_bed = f.hip_in_bed or (f.bed_overlap or 0.0) >= overlap_needed
            if not f.visible or box_on_bed:
                st = FrameState(st.t_sec, State.LYING_IN_BED, sm["occlusion_hold_conf"],
                                f"held: lying {st.t_sec - last_lying_t:.1f}s ago, {st.reason}")
        else:
            last_lying_t = None   # any other real state ends the hold
        out.append(st)
    return out


def smooth(states: list[FrameState], cfg: dict) -> list[FrameState]:
    sm = cfg["state_machine"]
    half = sm["vote_window_sec"] / 2
    out = []
    for i, st in enumerate(states):
        votes: dict[State, float] = defaultdict(float)
        confs: dict[State, list[float]] = defaultdict(list)
        j = i
        while j > 0 and st.t_sec - states[j - 1].t_sec <= half:
            j -= 1
        while j < len(states) and states[j].t_sec - st.t_sec <= half:
            s = states[j]
            votes[s.state] += sm["unknown_vote_weight"] if s.state == State.UNKNOWN else s.confidence
            confs[s.state].append(s.confidence)
            j += 1
        # Highest vote wins; on a tie keep the frame's own state.
        winner = max(votes, key=lambda k: (votes[k], k == st.state))
        c = sum(confs[winner]) / len(confs[winner])
        reason = st.reason if winner == st.state else f"smoothed from {st.state.value}"
        out.append(FrameState(st.t_sec, winner, round(c, 2), reason))
    return out


def frame_ends(states: list[FrameState], end_sec: float) -> list[float]:
    """Each frame lasts until the next one; the last until the end of the range."""
    return [states[i + 1].t_sec for i in range(len(states) - 1)] + [end_sec]


def apply_min_dwell(states: list[FrameState], cfg: dict, end_sec: float) -> list[FrameState]:
    dwell = cfg["state_machine"]["min_dwell_sec"]
    ends = frame_ends(states, end_sec)
    labels: list[State | None] = [None] * len(states)
    current = states[0].state
    labels[0] = current
    pending = None   # index where a possible new state started

    for i in range(1, len(states)):
        s = states[i].state
        if s == current:
            if pending is not None:              # the new state didn't last: undo it
                labels[pending:i] = [current] * (i - pending)
                pending = None
            labels[i] = current
            continue
        if pending is None or s != states[pending].state:
            if pending is not None:              # a third state interrupted the candidate
                labels[pending:i] = [current] * (i - pending)
            pending = i
        if ends[i] - states[pending].t_sec >= dwell[s.value.lower()]:
            labels[pending:i + 1] = [s] * (i + 1 - pending)   # accept, back-dated to `pending`
            current, pending = s, None
    if pending is not None:                      # unconfirmed candidate at the very end
        labels[pending:] = [current] * (len(states) - pending)

    return [st if lab == st.state else FrameState(st.t_sec, lab, st.confidence, f"held {lab.value} (min dwell)")
            for st, lab in zip(states, labels)]


def mark_out_of_bed(states: list[FrameState], feats: list[FrameFeatures]) -> list[FrameState]:
    """Judge each UNKNOWN stretch as a whole, so a stray half-visible frame (a hand
    at the edge of the image) doesn't split it into slivers."""
    out = list(states)
    last_known = None
    i = 0
    while i < len(out):
        if out[i].state != State.UNKNOWN:
            last_known = out[i].state
            i += 1
            continue
        j = i
        while j < len(out) and out[j].state == State.UNKNOWN:
            j += 1
        invisible = sum(1 for f in feats[i:j] if not f.visible)
        if last_known in OUT_OF_BED_SOURCES and invisible * 2 >= j - i:
            for k in range(i, j):
                out[k] = FrameState(out[k].t_sec, State.OUT_OF_BED, out[k].confidence,
                                    f"not visible after {last_known.value}")
        i = j
    return out


def to_segments(states: list[FrameState], start_sec: float, end_sec: float) -> list[Segment]:
    if not states:
        return [Segment(start_sec, end_sec, State.UNKNOWN, 0.0)]
    segments: list[Segment] = []
    run_start, confs = start_sec, [states[0].confidence]
    for i in range(1, len(states)):
        if states[i].state != states[i - 1].state:
            segments.append(Segment(run_start, states[i].t_sec, states[i - 1].state,
                                    round(sum(confs) / len(confs), 2)))
            run_start, confs = states[i].t_sec, []
        confs.append(states[i].confidence)
    segments.append(Segment(run_start, end_sec, states[-1].state, round(sum(confs) / len(confs), 2)))
    return segments


def build_timeline(states: list[FrameState], feats: list[FrameFeatures], cfg: dict,
                   start_sec: float, end_sec: float) -> tuple[list[FrameState], list[Segment]]:
    """Run all four steps. Returns (final per-frame states, segments)."""
    if not states:
        return [], to_segments([], start_sec, end_sec)
    s = blanket_hold(states, feats, cfg)
    s = smooth(s, cfg)
    s = apply_min_dwell(s, cfg, end_sec)
    s = mark_out_of_bed(s, feats)
    return s, to_segments(s, start_sec, end_sec)
