"""Detects when the (supposedly fixed) camera was moved.

Everything about the room, the bed and seat outlines, is detected once and
assumed to stay put. If the phone is nudged, the bed outline no longer lines
up with the bed and "hips in bed" goes wrong. That happened in test2 at 8:10.

Every 2 s a small grey frame is compared with a reference frame using phase
correlation, which measures how far the whole picture has shifted (a person
moving is a small part of the image; the room dominates). A shift above
`move_px` that lasts `confirm_checks` checks starts a new camera segment, and
the shifted frame becomes the new reference. Very dark frames are skipped
(nothing to compare). Segments shorter than `min_segment_sec` are merged into
the next one, so a camera that wobbles for a few seconds doesn't create many
tiny segments.

The result is a list of (start, end) time ranges; the scene is detected again
for each one (see scene.SceneTimeline).
"""

import logging
import math

import cv2
import numpy as np

from src.video_io import get_video_info, sample_frames

log = logging.getLogger(__name__)


def _prep(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return g * cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)


def frame_shift(ref: np.ndarray, gray: np.ndarray, scale: float) -> float:
    """How far (in original pixels) the picture moved between two prepared frames."""
    (dx, dy), _ = cv2.phaseCorrelate(ref, _prep(gray))
    return math.hypot(dx, dy) * scale


def find_camera_segments(video_path: str, cfg: dict) -> list[tuple[float, float]]:
    cc = cfg["camera"]
    info = get_video_info(video_path)
    if not cc["detect_moves"]:
        return [(0.0, info.duration_sec)]

    scale = info.width / cc["work_width"]   # small-frame pixels -> original pixels
    starts = [0.0]
    ref, pending = None, []
    for _, t, frame, _ in sample_frames(video_path, cc["check_fps"], cc["work_width"]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < cc["dark_level"]:
            continue
        if ref is None:
            ref = _prep(gray)
            continue
        if frame_shift(ref, gray, scale) > cc["move_px"]:
            pending.append((t, gray))
            if len(pending) >= cc["confirm_checks"]:
                starts.append(pending[0][0])
                ref, pending = _prep(pending[-1][1]), []
        else:
            pending = []

    # Merge camera positions that didn't last: keep a start only if the next start is far enough away.
    kept = [0.0]
    for s in starts[1:]:
        if s - kept[-1] < cc["min_segment_sec"]:
            kept[-1] = s if len(kept) > 1 else kept[-1]
        else:
            kept.append(s)
    segments = [(a, b) for a, b in zip(kept, kept[1:] + [info.duration_sec])]
    if len(segments) > 1:
        log.info("Camera moved: %d positions, starting at %s s",
                 len(segments), [round(a, 1) for a, _ in segments])
    return segments
