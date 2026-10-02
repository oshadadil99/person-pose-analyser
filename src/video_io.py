"""Reading a video at a fixed sample rate.

Bed activity changes over seconds, so we don't need every frame of a 25/30 fps
video. We keep roughly `sampling.fps` frames per second and skip the rest.

Timestamps always come from the real frame index (frame_idx / native_fps), not
from a counter, so durations stay correct even if the native fps isn't a
multiple of the sample rate.

Skipped frames are read with grab(), which advances the stream without decoding
the image. That is much faster than read() for frames we throw away.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    native_fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.native_fps


def get_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = VideoInfo(
        path=str(path),
        native_fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    if info.native_fps <= 0:
        raise ValueError(f"Video reports an invalid fps ({info.native_fps}): {path}")
    return info


def resize_to_width(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Shrink the frame so it is at most max_width wide. Returns (frame, scale)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    return cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA), scale


def sample_frames(path: str | Path, fps: float, max_width: int
                  ) -> Iterator[tuple[int, float, np.ndarray, float]]:
    """Yield (frame_idx, t_sec, frame, scale) at roughly `fps` frames per second.

    `scale` is how much the frame was shrunk, so callers can map pixel
    coordinates back to the original resolution.
    """
    info = get_video_info(path)
    step = info.native_fps / fps   # e.g. 30 fps video, 5 fps sampling -> every 6th frame
    next_pick = 0.0

    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    try:
        while True:
            if frame_idx >= next_pick:
                ok, frame = cap.read()
                if not ok:
                    break
                small, scale = resize_to_width(frame, max_width)
                yield frame_idx, frame_idx / info.native_fps, small, scale
                next_pick += step
            else:
                if not cap.grab():
                    break
            frame_idx += 1
    finally:
        cap.release()


def read_frame_at(path: str | Path, t_sec: float) -> np.ndarray | None:
    """Read one full-resolution frame at a given time (used later for VLM frames)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
