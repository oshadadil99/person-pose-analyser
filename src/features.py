"""Per-frame geometric features of the patient.

Turns the patient's keypoints, box, and the scene (bed + seats) into a few
numbers the frame rules can reason about: is the body upright or flat, are the
legs bent, where are the hips, how fast are they moving.

Angles are measured against the vertical in the image: 0 deg = pointing
straight up/down, 90 deg = horizontal. This is a 2D measurement, so a body
pointing at the camera looks more upright than it is; the rules also use the
box shape and bed position to compensate.

Anything that can't be measured (hidden keypoints, no bed found) is None,
never 0, so a missing hip can't be mistaken for a hip at the image corner.
"""

import math
from collections import deque
from dataclasses import dataclass

from src.geometry import box_height, box_overlap_ratio, signed_distance
from src.patient import reference_point
from src.perception import KP, FramePerception, Person
from src.scene import Scene

# Keypoints that matter for posture. Face and arms are ignored on purpose:
# they move a lot and say little about lying/sitting/standing.
POSTURE_KPS = [KP[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                                "left_knee", "right_knee", "left_ankle", "right_ankle")]


@dataclass
class FrameFeatures:
    frame_idx: int
    t_sec: float
    visible: bool
    num_persons: int
    torso_angle_deg: float | None = None
    thigh_angle_deg: float | None = None
    bbox_aspect: float | None = None
    kp_conf_mean: float | None = None
    hip_in_bed: bool | None = None
    bed_overlap: float | None = None
    bed_edge_dist: float | None = None    # in box heights, + inside the bed, - outside
    on_seat: bool | None = None
    seat_label: str | None = None
    hip_speed: float | None = None       # box heights per second
    ref_source: str | None = None        # "hips", "one_hip" or "box_centre"
    box_h: float | None = None
    box_truncated: bool | None = None    # box reaches the bottom of the frame: legs out of view


def angle_from_vertical(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Angle of the line p1 -> p2 against the vertical, 0 to 90 degrees."""
    dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
    return math.degrees(math.atan2(dx, dy))


def _midpoint(person: Person, names: list[str], kp_min_conf: float) -> tuple[float, float] | None:
    pts = [person.keypoints[KP[n]] for n in names]
    good = [p for p in pts if p[2] >= kp_min_conf]
    if not good:
        return None
    return sum(p[0] for p in good) / len(good), sum(p[1] for p in good) / len(good)


def torso_angle(person: Person, kp_min_conf: float) -> float | None:
    shoulders = _midpoint(person, ["left_shoulder", "right_shoulder"], kp_min_conf)
    hips = _midpoint(person, ["left_hip", "right_hip"], kp_min_conf)
    if shoulders is None or hips is None:
        return None
    return angle_from_vertical(shoulders, hips)


def thigh_angle(person: Person, kp_min_conf: float) -> float | None:
    """Mean hip -> knee angle over the legs that are visible."""
    angles = []
    for side in ("left", "right"):
        hip, knee = person.keypoints[KP[f"{side}_hip"]], person.keypoints[KP[f"{side}_knee"]]
        if hip[2] >= kp_min_conf and knee[2] >= kp_min_conf:
            angles.append(angle_from_vertical((hip[0], hip[1]), (knee[0], knee[1])))
    return sum(angles) / len(angles) if angles else None


def extract_features(frames: list[FramePerception], patient: list[Person | None],
                     scene: Scene | None, cfg: dict, frame_height: int | None = None) -> list[FrameFeatures]:
    fc = cfg["features"]
    if frame_height is None and scene is not None:
        frame_height = scene.frame_height
    kp_min = fc["kp_min_conf"]
    bed = scene.bed.polygon if scene and scene.bed else None
    seats = scene.seats if scene else []

    out = []
    recent: deque = deque()   # (t, point, is_box_centre) for the speed window
    for fr, p in zip(frames, patient):
        f = FrameFeatures(fr.frame_idx, fr.t_sec, visible=p is not None, num_persons=len(fr.persons))
        if p is None:
            recent.clear()   # speed across a gap in visibility would be meaningless
            out.append(f)
            continue

        x1, y1, x2, y2 = p.box
        h = max(box_height(p.box), 1.0)
        f.box_h = h
        f.bbox_aspect = (x2 - x1) / h
        if frame_height:
            f.box_truncated = y2 >= frame_height * (1 - fc["truncated_margin"])
        f.torso_angle_deg = torso_angle(p, kp_min)
        f.thigh_angle_deg = thigh_angle(p, kp_min)
        f.kp_conf_mean = sum(p.keypoints[i][2] for i in POSTURE_KPS) / len(POSTURE_KPS)

        x, y, src = reference_point(p, kp_min)
        f.ref_source = src
        if bed:
            f.bed_edge_dist = signed_distance((x, y), bed) / h
            f.hip_in_bed = f.bed_edge_dist >= -fc["in_bed_margin"]
            f.bed_overlap = box_overlap_ratio(p.box, bed)
        if scene is not None:
            f.on_seat = False
            for s in seats:
                if signed_distance((x, y), s.polygon) / h >= -fc["seat_margin"]:
                    f.on_seat, f.seat_label = True, s.label
                    break

        # Speed: distance from the oldest point still inside the window. Only compare
        # like with like, since jumping between hips and box centre isn't movement.
        is_box = src == "box_centre"
        while recent and fr.t_sec - recent[0][0] > fc["speed_window_sec"]:
            recent.popleft()
        same = [r for r in recent if r[2] == is_box]
        if same and fr.t_sec > same[0][0]:
            t0, (x0, y0), _ = same[0]
            f.hip_speed = math.dist((x, y), (x0, y0)) / (fr.t_sec - t0) / h
        recent.append((fr.t_sec, (x, y), is_box))
        out.append(f)
    return out
