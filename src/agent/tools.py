"""The investigator agent's tools.

Each tool is a plain function that answers one question about the already
processed video (timeline, per-frame measurements, detections) and returns a
small JSON-friendly dict with a one-line `summary`. They are cheap, so the
agent can call several per investigation. Only `ask_vlm` looks at actual
pixels; it's the expensive one and is used last.

    look_back(t, seconds)        what happened in [t - seconds, t]
    look_forward(t, seconds)     what happens in [t, t + seconds] (fine: recorded video)
    check_bed_overlap(t)         where the hips are relative to the bed
    get_pose_summary(t, seconds) posture numbers at t, or averaged over [t, t + seconds]
    count_people(t, seconds)     how many people are in view, and is the patient one of them
    ask_vlm(times, question)     ask Gemini about 2-4 frames (Phase 9; offline it says "unavailable")
"""

from dataclasses import dataclass
from typing import Callable

from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.perception import FramePerception
from src.state_machine import Segment
from src.states import IN_BED_STATES, State


@dataclass
class AgentContext:
    segments: list[Segment]
    states: list[FrameState]       # final per-frame states
    feats: list[FrameFeatures]
    frames: list[FramePerception]
    patient_ids: list[int]
    cfg: dict
    vlm: Callable[[list[float], str], dict] | None = None   # None = offline


def _window(ctx: AgentContext, t0: float, t1: float) -> dict:
    parts = []
    for s in ctx.segments:
        lo, hi = max(s.start_sec, t0), min(s.end_sec, t1)
        if hi - lo >= 0.05:   # ignore slivers from rounding at the window edge
            parts.append({"state": s.state.value.lower(), "from": round(lo, 1), "to": round(hi, 1),
                          "sec": round(hi - lo, 1)})
    in_bed = sum(p["sec"] for p in parts if State(p["state"].upper()) in IN_BED_STATES)
    unknown = sum(p["sec"] for p in parts if p["state"] == "unknown")
    out = sum(p["sec"] for p in parts) - in_bed - unknown
    fs = [f for f in ctx.feats if t0 <= f.t_sec < t1]
    speeds = [f.hip_speed for f in fs if f.hip_speed is not None]
    return {
        "from": round(t0, 1), "to": round(t1, 1), "states": parts,
        "in_bed_sec": round(in_bed, 1), "out_of_bed_sec": round(out, 1), "unknown_sec": round(unknown, 1),
        "visible_share": round(sum(f.visible for f in fs) / len(fs), 2) if fs else 0.0,
        "mean_speed": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "summary": ", then ".join(f"{p['state']} {p['sec']:.1f}s" for p in parts) or "nothing recorded",
    }


def look_back(ctx: AgentContext, t: float, seconds: float) -> dict:
    return _window(ctx, max(0.0, t - seconds), t)


def look_forward(ctx: AgentContext, t: float, seconds: float) -> dict:
    return _window(ctx, t, t + seconds)


def _nearest(ctx: AgentContext, t: float) -> int:
    return min(range(len(ctx.feats)), key=lambda i: abs(ctx.feats[i].t_sec - t))


def check_bed_overlap(ctx: AgentContext, t: float) -> dict:
    f = ctx.feats[_nearest(ctx, t)]
    if not f.visible:
        return {"visible": False, "summary": "patient not visible at this moment"}
    if f.bed_edge_dist is None:
        where = "no bed region known"
    elif f.bed_edge_dist >= 0:
        where = f"hips inside the bed, {f.bed_edge_dist:.2f} box-heights from the edge"
    else:
        where = f"hips outside the bed, {-f.bed_edge_dist:.2f} box-heights from the edge"
    if f.on_seat:
        where += f", on a {f.seat_label}"
    return {"visible": True, "hip_in_bed": f.hip_in_bed, "bed_edge_dist": f.bed_edge_dist,
            "bed_overlap": f.bed_overlap, "on_seat": f.on_seat, "box_truncated": f.box_truncated,
            "summary": where}


def get_pose_summary(ctx: AgentContext, t: float, seconds: float = 0.0) -> dict:
    if seconds <= 0:
        i = _nearest(ctx, t)
        f, st = ctx.feats[i], ctx.states[i]
        if not f.visible:
            return {"visible": False, "summary": "patient not visible"}
        torso = f"{f.torso_angle_deg:.0f}deg" if f.torso_angle_deg is not None else "hidden"
        thigh = f"{f.thigh_angle_deg:.0f}deg" if f.thigh_angle_deg is not None else "hidden"
        return {"visible": True, "torso_angle_deg": f.torso_angle_deg, "thigh_angle_deg": f.thigh_angle_deg,
                "hip_speed": f.hip_speed, "kp_conf": f.kp_conf_mean, "state": st.state.value.lower(),
                "summary": f"torso {torso}, thigh {thigh}, keypoint confidence {f.kp_conf_mean:.2f}"}
    fs = [f for f in ctx.feats if t <= f.t_sec < t + seconds]
    vis = [f for f in fs if f.visible]
    if not vis:
        return {"visible_share": 0.0, "summary": f"patient not visible at all in these {seconds:.0f}s"}
    torsos = [f.torso_angle_deg for f in vis if f.torso_angle_deg is not None]
    kps = [f.kp_conf_mean for f in vis if f.kp_conf_mean is not None]
    share = len(vis) / len(fs)
    torso_mean = sum(torsos) / len(torsos) if torsos else None
    return {"visible_share": round(share, 2),
            "mean_torso_angle_deg": round(torso_mean, 1) if torso_mean is not None else None,
            "mean_kp_conf": round(sum(kps) / len(kps), 2) if kps else None,
            "legs_out_of_frame_share": round(sum(bool(f.box_truncated) for f in vis) / len(vis), 2),
            "summary": f"visible {share:.0%} of the time, mean torso "
                       f"{'%.0fdeg' % torso_mean if torso_mean is not None else 'unknown'}, "
                       f"keypoint confidence {sum(kps) / len(kps) if kps else 0:.2f}"}


def count_people(ctx: AgentContext, t: float, seconds: float = 2.0) -> dict:
    frames = [f for f in ctx.frames if t <= f.t_sec < t + max(seconds, 0.2)]
    if not frames:
        return {"max_people": 0, "summary": "no frames in this range"}
    ids = set(ctx.patient_ids)
    others = sorted({p.track_id for f in frames for p in f.persons if p.track_id not in ids and p.track_id >= 0})
    peak = max(len(f.persons) for f in frames)
    patient_seen = sum(any(p.track_id in ids for p in f.persons) for f in frames) / len(frames)
    return {"max_people": peak, "patient_visible_share": round(patient_seen, 2), "other_track_ids": others,
            "summary": f"up to {peak} people; patient visible {patient_seen:.0%} of the time; "
                       f"other track ids {others or 'none'}"}


def ask_vlm(ctx: AgentContext, times: list[float], question: str) -> dict:
    if ctx.vlm is None:
        return {"state": "UNKNOWN", "confidence": 0.0, "reason": "vlm_unavailable",
                "summary": "no VLM available (offline run)"}
    answer = ctx.vlm(times, question)
    answer.setdefault("summary", f"VLM: {answer.get('state')} ({answer.get('confidence')}): {answer.get('reason')}")
    return answer


TOOLS: dict[str, Callable[..., dict]] = {
    "look_back": look_back,
    "look_forward": look_forward,
    "check_bed_overlap": check_bed_overlap,
    "get_pose_summary": get_pose_summary,
    "count_people": count_people,
    "ask_vlm": ask_vlm,
}
