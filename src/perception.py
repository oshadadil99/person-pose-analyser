"""Person detection, pose and tracking with YOLO11-pose + ByteTrack.

YOLO11-pose gives, for each person, a box and 17 COCO keypoints (x, y, conf)
in a single forward pass. ByteTrack (built into Ultralytics) links detections
across frames so the same person keeps the same track ID. That matters later
for telling the patient apart from a caregiver.

Running YOLO is the slow part of the pipeline, so the results are saved to a
JSON-lines file (perception.jsonl). Every later stage reads that file, which
means thresholds can be re-tuned in seconds without re-running the model.

All coordinates are stored in ORIGINAL video pixels, even though inference
runs on a downscaled frame. The bed polygon is drawn on the original frame,
so both must be in the same coordinate system.

Primary-person (patient) selection needs the bed region, so it is done later
on top of these raw tracks, not here.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.video_io import get_video_info, sample_frames

log = logging.getLogger(__name__)

# COCO keypoint order used by YOLO pose models.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Pairs of keypoints joined when drawing a skeleton.
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # shoulders and arms
    (5, 11), (6, 12), (11, 12),                    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),        # legs
    (0, 5), (0, 6),                                # head to shoulders
]


@dataclass
class Person:
    track_id: int                 # -1 if the tracker hasn't assigned an ID yet
    box: list[float]              # x1, y1, x2, y2 in original pixels
    conf: float                   # detection confidence
    keypoints: list[list[float]]  # 17 x [x, y, conf] in original pixels


@dataclass
class FramePerception:
    frame_idx: int
    t_sec: float
    persons: list[Person] = field(default_factory=list)


class PoseTracker:
    """Thin wrapper so the rest of the code never touches Ultralytics objects."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # imported here so tests that don't need YOLO stay fast
        p = cfg["perception"]
        self.model = YOLO(p["model"])
        self.args = dict(conf=p["det_conf"], iou=p["iou"], imgsz=p["imgsz"],
                         device=p["device"], tracker=p["tracker"], verbose=False)

    def process(self, frame: np.ndarray, scale: float) -> list[Person]:
        # persist=True keeps the tracker state between calls, i.e. across frames.
        result = self.model.track(frame, persist=True, **self.args)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy() / scale
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id
        ids = ids.int().cpu().numpy() if ids is not None else [-1] * len(boxes)
        kps = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
        kps[:, :, :2] /= scale

        return [
            Person(track_id=int(ids[i]),
                   box=[round(float(v), 1) for v in boxes[i]],
                   conf=round(float(confs[i]), 3),
                   keypoints=[[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                              for x, y, c in kps[i]])
            for i in range(len(boxes))
        ]


def run_perception(video_path: str | Path, cfg: dict) -> tuple[dict, list[FramePerception]]:
    """Run YOLO + tracking over the sampled frames. Returns (meta, frames)."""
    info = get_video_info(video_path)
    s = cfg["sampling"]
    tracker = PoseTracker(cfg)

    meta = {"video": str(video_path), "native_fps": info.native_fps,
            "width": info.width, "height": info.height,
            "duration_sec": info.duration_sec, "sample_fps": s["fps"],
            "model": cfg["perception"]["model"]}

    frames = []
    expected = math.ceil(info.frame_count / (info.native_fps / s["fps"]))
    for frame_idx, t_sec, frame, scale in tqdm(sample_frames(video_path, s["fps"], s["max_width"]),
                                               total=expected, desc="perception", unit="frame"):
        frames.append(FramePerception(frame_idx, round(t_sec, 3), tracker.process(frame, scale)))

    log.info("Perception done: %d frames, %d with at least one person",
             len(frames), sum(1 for f in frames if f.persons))
    return meta, frames


def save_perception(path: str | Path, meta: dict, frames: list[FramePerception]) -> None:
    """First line is the metadata, then one JSON object per sampled frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for fr in frames:
            f.write(json.dumps({
                "frame_idx": fr.frame_idx, "t": fr.t_sec,
                "persons": [{"id": p.track_id, "box": p.box, "conf": p.conf, "kps": p.keypoints}
                            for p in fr.persons],
            }) + "\n")


def load_perception(path: str | Path) -> tuple[dict, list[FramePerception]]:
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        frames = []
        for line in f:
            d = json.loads(line)
            persons = [Person(p["id"], p["box"], p["conf"], p["kps"]) for p in d["persons"]]
            frames.append(FramePerception(d["frame_idx"], d["t"], persons))
    return meta, frames


def draw_person(img: np.ndarray, person: Person, color: tuple[int, int, int],
                kp_min_conf: float, label: str | None = None) -> None:
    """Draw box, skeleton and an optional label onto img (in place)."""
    x1, y1, x2, y2 = map(int, person.box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    pts = person.keypoints
    for a, b in SKELETON:
        if pts[a][2] >= kp_min_conf and pts[b][2] >= kp_min_conf:
            cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 2)
    for x, y, c in pts:
        if c >= kp_min_conf:
            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
    if label:
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
