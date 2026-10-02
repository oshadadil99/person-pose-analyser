"""Decides which tracked person is the patient.

ByteTrack gives every person a track ID, but doesn't know who is who. The
patient is the track that spends the most frames in or near the bed. Anyone
else (e.g. a caregiver) is still counted in `num_persons`, but not analysed.

ByteTrack also hands out a new ID when it loses someone for too long (heavy
occlusion, leaving the camera view). So starting from the best track we build
a chain, joining a later (or earlier) track to the patient when either:
  - it starts within `relink_max_gap_sec` and close to where the patient was
    last seen (a short tracking break), or
  - it is the only person in view when it appears (the patient coming back
    into the room; `relink_lone_returner`).
Tracks that overlap in time with the chain can't be the patient: one person
can't be in two places.

Known limit: a caregiver walking in alone while the patient is out of view
would be taken for the patient. Fixing that needs appearance re-identification.
"""

import logging
import math
from dataclasses import dataclass

from src.geometry import box_height, signed_distance
from src.perception import KP, FramePerception, Person

log = logging.getLogger(__name__)


def reference_point(person: Person, kp_min_conf: float) -> tuple[float, float, str]:
    """Where the person "is": hip midpoint if visible, else box centre.

    Hips are what actually rest on the bed or a chair. Under a blanket they are
    often missing, so the box centre is the fallback (see decisions D10).
    """
    hips = [person.keypoints[KP["left_hip"]], person.keypoints[KP["right_hip"]]]
    good = [h for h in hips if h[2] >= kp_min_conf]
    if good:
        return (sum(h[0] for h in good) / len(good), sum(h[1] for h in good) / len(good),
                "hips" if len(good) == 2 else "one_hip")
    x1, y1, x2, y2 = person.box
    return ((x1 + x2) / 2, (y1 + y2) / 2, "box_centre")


@dataclass
class TrackSummary:
    track_id: int
    first_t: float
    last_t: float
    first_pt: tuple[float, float]
    last_pt: tuple[float, float]
    first_h: float
    last_h: float
    n_frames: int = 0
    near_bed_frames: int = 0


def summarize_tracks(frames: list[FramePerception], bed_polygon: list | None, cfg: dict) -> dict[int, TrackSummary]:
    kp_min_conf = cfg["features"]["kp_min_conf"]
    margin = cfg["patient"]["bed_margin"]
    tracks: dict[int, TrackSummary] = {}
    for fr in frames:
        for p in fr.persons:
            if p.track_id < 0:      # not yet confirmed by the tracker
                continue
            x, y, _ = reference_point(p, kp_min_conf)
            h = box_height(p.box)
            s = tracks.get(p.track_id)
            if s is None:
                s = tracks[p.track_id] = TrackSummary(p.track_id, fr.t_sec, fr.t_sec, (x, y), (x, y), h, h)
            s.last_t, s.last_pt, s.last_h = fr.t_sec, (x, y), h
            s.n_frames += 1
            # "near" allows a margin outside the polygon so sitting on the edge counts
            if bed_polygon and signed_distance((x, y), bed_polygon) >= -margin * h:
                s.near_bed_frames += 1
    return tracks


def _active_at(tracks: dict[int, TrackSummary], t: float, exclude: int) -> bool:
    return any(s.first_t <= t <= s.last_t for tid, s in tracks.items() if tid != exclude)


def select_patient_ids(tracks: dict[int, TrackSummary], cfg: dict) -> list[int]:
    """Return the chain of track IDs that belong to the patient, in time order."""
    if not tracks:
        return []
    pc = cfg["patient"]
    best = max(tracks.values(), key=lambda s: (s.near_bed_frames, s.n_frames))
    chain = [best]

    def links(prev_pt, prev_h, gap, cand_pt, cand_t, cand_id) -> bool:
        dist = math.dist(prev_pt, cand_pt) / max(prev_h, 1.0)
        if gap <= pc["relink_max_gap_sec"] and dist <= pc["relink_max_dist"]:
            return True
        return pc["relink_lone_returner"] and not _active_at(tracks, cand_t, cand_id)

    # Forward: tracks starting after the chain ends. Take the earliest one that links.
    while True:
        end = chain[-1]
        later = sorted((s for s in tracks.values() if s.first_t > end.last_t), key=lambda s: s.first_t)
        nxt = next((s for s in later
                    if links(end.last_pt, end.last_h, s.first_t - end.last_t, s.first_pt, s.first_t, s.track_id)),
                   None)
        if nxt is None:
            break
        chain.append(nxt)

    # Backward: tracks ending before the chain starts. Take the latest one that links.
    while True:
        start = chain[0]
        earlier = sorted((s for s in tracks.values() if s.last_t < start.first_t), key=lambda s: -s.last_t)
        prv = next((s for s in earlier
                    if links(start.first_pt, start.first_h, start.first_t - s.last_t, s.last_pt, s.last_t, s.track_id)),
                   None)
        if prv is None:
            break
        chain.insert(0, prv)

    ids = [s.track_id for s in chain]
    log.info("Patient track IDs: %s (best=%d, %d of %d frames near bed)",
             ids, best.track_id, best.near_bed_frames, best.n_frames)
    return ids


def patient_per_frame(frames: list[FramePerception], patient_ids: list[int]) -> list[Person | None]:
    """The patient's detection in each frame, or None if not visible."""
    ids = set(patient_ids)
    out = []
    for fr in frames:
        mine = [p for p in fr.persons if p.track_id in ids]
        out.append(max(mine, key=lambda p: p.conf) if mine else None)
    return out
