"""Per-frame state guess from the geometric features.

A short, ordered chain of rules. Each frame is judged on its own here; the
state machine (next stage) smooths the result over time.

  1. Not visible                                -> UNKNOWN
  2. Pose unreadable (low keypoint confidence)
       wide box mostly on the bed (blanket)     -> LYING_IN_BED, low confidence
       otherwise                                -> UNKNOWN
  3. Body horizontal                            -> LYING_IN_BED / LYING_OUTSIDE_BED
  4. Upright, legs not measurable
       moving fast                              -> WALKING
       legs out of frame (box at image bottom)  -> UNKNOWN
       hips on bed / seat                       -> SITTING_..., low confidence
       otherwise                                -> STANDING, low confidence
  5. Upright, thigh angle measured
       clearly bent  (>= 55 deg)                -> SITTING_ON_BED / SITTING_OUTSIDE_BED
       clearly straight (<= 20 deg)             -> WALKING if moving, else STANDING
       unsure band in between                   -> WALKING if moving; sitting if hips are
                                                   on the bed or a seat; else STANDING.
                                                   Low confidence either way.

Why the unsure band: a seated person facing the camera has thighs pointing at
the lens, which look steep in 2D (~30 deg on test1, vs ~90 deg side-on).
Context (something to sit on under the hips) is better evidence there than
the angle. See docs/decisions.md D18.

Confidence = keypoint confidence x how clearly the deciding measurement is
past its threshold (0.5 at the threshold, 1.0 when `margin_scale_deg` past it).
Hips right at the bed edge lower it further. Low confidence is what later
wakes up the agent.
"""

from dataclasses import dataclass

from src.features import FrameFeatures
from src.states import State


@dataclass
class FrameState:
    t_sec: float
    state: State
    confidence: float
    reason: str


def _margin(value: float, threshold: float, scale: float) -> float:
    """0 at the threshold, 1 when `scale` or more past it."""
    return min(1.0, abs(value - threshold) / scale)


def _in_bed(f: FrameFeatures, fr: dict) -> bool:
    if f.hip_in_bed is not None:
        return f.hip_in_bed
    return (f.bed_overlap or 0.0) >= fr["in_bed_overlap_ratio"]


def classify_frame(f: FrameFeatures, cfg: dict) -> FrameState:
    fr = cfg["frame_rules"]
    scale = fr["margin_scale_deg"]
    t = f.t_sec

    # 1. Nothing to look at
    if not f.visible:
        return FrameState(t, State.UNKNOWN, 0.0, "patient not visible")

    in_bed = _in_bed(f, fr)
    where = "in bed" if in_bed else "outside bed"

    # 2. Pose unreadable
    if f.kp_conf_mean is None or f.kp_conf_mean < fr["min_kp_conf_mean"]:
        if in_bed and f.bbox_aspect is not None and f.bbox_aspect > fr["lying_bbox_aspect"]:
            return FrameState(t, State.LYING_IN_BED, fr["low_kp_lying_conf"],
                              f"pose unclear (kp {f.kp_conf_mean:.2f}), wide box on bed")
        return FrameState(t, State.UNKNOWN, 0.0, f"pose unclear (kp {f.kp_conf_mean:.2f})")

    kp = min(1.0, f.kp_conf_mean)
    near_edge = f.bed_edge_dist is not None and abs(f.bed_edge_dist) < cfg["triggers"]["near_edge_ratio"]
    edge = fr["edge_conf_factor"] if near_edge else 1.0
    unsure = fr["unsure_conf_factor"]

    def conf(margin: float, location_matters: bool = True) -> float:
        c = kp * (0.5 + 0.5 * margin)
        return round(c * (edge if location_matters else 1.0), 2)

    # 3. Horizontal body. Torso angle decides if we have it, else the box shape.
    if f.torso_angle_deg is not None:
        horizontal = f.torso_angle_deg > fr["lying_torso_angle_deg"]
        posture_margin = _margin(f.torso_angle_deg, fr["lying_torso_angle_deg"], scale)
        posture = f"torso {f.torso_angle_deg:.0f}deg"
    else:
        horizontal = f.bbox_aspect > fr["lying_bbox_aspect"]
        posture_margin = 0.0   # box shape alone is weak evidence: half confidence at most
        posture = f"no torso, box {f.bbox_aspect:.2f}"

    if horizontal:
        state = State.LYING_IN_BED if in_bed else State.LYING_OUTSIDE_BED
        return FrameState(t, state, conf(posture_margin), f"{posture} (lying), {where}")

    # Upright from here on.
    moving = f.hip_speed is not None and f.hip_speed > fr["walking_speed"]
    supported = in_bed or bool(f.on_seat)   # something to sit on under the hips
    speed_margin = _margin(f.hip_speed, fr["walking_speed"], fr["walking_speed"]) if moving else 0.0
    speed = f"speed {f.hip_speed:.2f}" if f.hip_speed is not None else "no speed yet"

    def sitting(c: float, why: str) -> FrameState:
        if in_bed:
            return FrameState(t, State.SITTING_ON_BED, c, f"{why}, in bed")
        seat = f", on {f.seat_label}" if f.on_seat else ""
        return FrameState(t, State.SITTING_OUTSIDE_BED, c, f"{why}, outside bed{seat}")

    # 4. Legs not measurable
    if f.thigh_angle_deg is None:
        if moving:
            return FrameState(t, State.WALKING, round(conf(speed_margin, False) * unsure, 2), f"legs hidden, {speed}")
        if f.box_truncated:
            return FrameState(t, State.UNKNOWN, 0.0, "legs out of frame, can't tell sitting from standing")
        if supported:
            return sitting(round(conf(0.0) * unsure, 2), "upright, legs hidden")
        return FrameState(t, State.STANDING, round(conf(0.0, False) * unsure, 2), f"upright, legs hidden, {speed}")

    # 5. Thigh angle measured
    th = f.thigh_angle_deg
    legs = f"thigh {th:.0f}deg"
    if th >= fr["sitting_thigh_angle_deg"]:
        return sitting(conf(_margin(th, fr["sitting_thigh_angle_deg"], scale)), f"{legs} (bent)")

    if th <= fr["standing_thigh_max_deg"]:
        if moving:
            return FrameState(t, State.WALKING, conf(speed_margin, False), f"{legs} (straight), {speed}")
        return FrameState(t, State.STANDING, conf(_margin(th, fr["standing_thigh_max_deg"], scale), False),
                          f"{legs} (straight), {speed}")

    # Unsure band
    if moving:
        return FrameState(t, State.WALKING, round(conf(speed_margin, False) * unsure, 2), f"{legs} (unsure), {speed}")
    if supported:
        return sitting(round(conf(0.0) * unsure, 2), f"{legs} (unsure), hips supported")
    return FrameState(t, State.STANDING, round(conf(0.0, False) * unsure, 2), f"{legs} (unsure), {speed}")


def classify_frames(features: list[FrameFeatures], cfg: dict) -> list[FrameState]:
    return [classify_frame(f, cfg) for f in features]
