"""Finds the bed and the seats (chair, couch, ...) automatically.

The pose model only knows people, so furniture comes from a second model,
YOLO11s-seg, trained on the 80 COCO classes (which include bed and chair).
It gives an outline mask per object, which matters for the bed: seen from an
angle, a bed's box also covers floor next to it, a mask doesn't.

The camera is fixed and furniture doesn't move, so we only run it on
`scene.num_frames` frames spread across the video, then merge:
  1. Group detections from different frames into objects (box IoU).
  2. Drop seats seen in too few frames (false detections). The bed only needs
     a couple of sightings, because the person lying on it often hides it.
  3. Per object, keep pixels present in at least `mask_vote` of its masks,
     take the convex hull (fills notches cut by a person or blanket) and
     simplify it to a polygon.
See docs/decisions.md (D7) for why this replaced a hand-drawn bed polygon.
"""

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.geometry import box_iou, polygon_bbox
from src.video_io import get_video_info, read_frame_at, resize_to_width

log = logging.getLogger(__name__)


@dataclass
class SceneObject:
    role: str              # "bed" or "seat"
    label: str             # COCO class name, e.g. "chair"
    polygon: list[list[float]]
    conf: float            # mean detection confidence
    seen_ratio: float      # fraction of sampled frames it was detected in


@dataclass
class Scene:
    frame_width: int
    frame_height: int
    bed: SceneObject | None
    seats: list[SceneObject] = field(default_factory=list)
    source: str = "auto"   # "auto" or "manual"


@dataclass
class Detection:
    role: str
    label: str
    conf: float
    box: list[float]       # in working (resized) pixels
    mask: np.ndarray       # bool, working size


@dataclass
class _Cluster:
    role: str
    boxes: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    labels: Counter = field(default_factory=Counter)
    confs: list = field(default_factory=list)

    @property
    def box(self) -> list[float]:
        return np.mean(self.boxes, axis=0).tolist()

    def add(self, d: Detection) -> None:
        self.boxes.append(d.box)
        self.masks.append(d.mask)
        self.labels[d.label] += 1
        self.confs.append(d.conf)


def segment_frames(video_path: str | Path, cfg: dict) -> tuple[list[list[Detection]], float, tuple[int, int]]:
    """Run the segmentation model on evenly spaced frames.

    Returns (detections per frame, scale from original to working size, working (h, w)).
    """
    from ultralytics import YOLO
    sc = cfg["scene"]
    info = get_video_info(video_path)
    model = YOLO(sc["model"])

    roles = {name: "bed" for name in sc["bed_classes"]} | {name: "seat" for name in sc["seat_classes"]}
    class_ids = [i for i, name in model.names.items() if name in roles]

    # Middle of n equal slices, so we never pick the very first/last frame (often black).
    n = sc["num_frames"]
    times = [info.duration_sec * (i + 0.5) / n for i in range(n)]

    all_dets, scale, work_shape = [], 1.0, (0, 0)
    for t in times:
        frame = read_frame_at(video_path, t)
        if frame is None:
            continue
        small, scale = resize_to_width(frame, cfg["sampling"]["max_width"])
        work_shape = small.shape[:2]
        r = model.predict(small, conf=sc["det_conf"], imgsz=sc["imgsz"], classes=class_ids,
                          device=cfg["perception"]["device"], verbose=False)[0]
        dets = []
        if r.masks is not None:
            for i, poly in enumerate(r.masks.xy):   # outline in working-frame pixels
                if len(poly) < 3:
                    continue
                mask = np.zeros(work_shape, np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                label = model.names[int(r.boxes.cls[i])]
                dets.append(Detection(roles[label], label, float(r.boxes.conf[i]),
                                      r.boxes.xyxy[i].tolist(), mask.astype(bool)))
        all_dets.append(dets)
    return all_dets, scale, work_shape


def merge_detections(frames: list[list[Detection]], cfg: dict) -> tuple[_Cluster | None, list[_Cluster]]:
    """Group per-frame detections into objects. Returns (bed cluster, seat clusters)."""
    sc = cfg["scene"]
    clusters: list[_Cluster] = []
    for dets in frames:
        # An object appears at most once per frame, so two chairs side by side in
        # the same frame must never end up in the same cluster.
        used: set[int] = set()
        for d in sorted(dets, key=lambda d: -d.conf):
            free = [c for c in clusters if c.role == d.role and id(c) not in used]
            best = max(free, key=lambda c: box_iou(c.box, d.box), default=None)
            if best is None or box_iou(best.box, d.box) < sc["match_iou"]:
                best = _Cluster(d.role)
                clusters.append(best)
            best.add(d)
            used.add(id(best))

    n = max(len(frames), 1)
    beds = [c for c in clusters if c.role == "bed" and len(c.masks) >= sc["bed_min_frames"]]
    # One bed per room: the one seen most often (ties broken by size).
    bed = max(beds, key=lambda c: (len(c.masks), np.mean([m.sum() for m in c.masks])), default=None)

    seats = [c for c in clusters if c.role == "seat" and len(c.masks) / n >= sc["seat_min_presence"]]
    if bed is not None:
        seats = [s for s in seats if box_iou(s.box, bed.box) < sc["seat_bed_max_iou"]]
    return bed, seats


def cluster_to_polygon(c: _Cluster, cfg: dict, scale: float) -> list[list[float]] | None:
    """Vote the masks, take the convex hull, simplify, and scale back to original pixels."""
    votes = np.mean(np.stack(c.masks), axis=0)
    mask = (votes >= cfg["scene"]["mask_vote"]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    eps = cfg["scene"]["polygon_epsilon"] * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2) / scale
    return [[round(float(x), 1), round(float(y), 1)] for x, y in approx]


def _to_object(c: _Cluster, cfg: dict, scale: float, n_frames: int) -> SceneObject | None:
    poly = cluster_to_polygon(c, cfg, scale)
    if poly is None:
        return None
    return SceneObject(c.role, c.labels.most_common(1)[0][0], poly,
                       round(float(np.mean(c.confs)), 3), round(len(c.masks) / max(n_frames, 1), 2))


def detect_scene(video_path: str | Path, cfg: dict) -> Scene:
    info = get_video_info(video_path)
    frames, scale, _ = segment_frames(video_path, cfg)
    bed_c, seat_cs = merge_detections(frames, cfg)

    bed = _to_object(bed_c, cfg, scale, len(frames)) if bed_c else None
    seats = [o for o in (_to_object(c, cfg, scale, len(frames)) for c in seat_cs) if o]
    if bed is None:
        log.warning("No bed found. Check the preview, or draw it with tools/draw_bed_polygon.py")
    log.info("Scene: bed=%s, seats=%s", bed.label if bed else None, [s.label for s in seats])
    return Scene(info.width, info.height, bed, seats)


def save_scene(path: str | Path, scene: Scene) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(scene), indent=2), encoding="utf-8")


def load_scene(path: str | Path) -> Scene:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    bed = SceneObject(**d["bed"]) if d.get("bed") else None
    seats = [SceneObject(**s) for s in d.get("seats", [])]
    return Scene(d["frame_width"], d["frame_height"], bed, seats, d.get("source", "auto"))


def draw_scene(img: np.ndarray, scene: Scene) -> np.ndarray:
    """Return a copy of img with the bed (blue) and seats (orange) drawn on it."""
    out = img.copy()
    objects = ([scene.bed] if scene.bed else []) + scene.seats
    for o in objects:
        color = (255, 120, 0) if o.role == "bed" else (0, 165, 255)
        pts = np.asarray(o.polygon, np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [pts], True, color, 3)
        x1, y1, _, _ = polygon_bbox(o.polygon)
        cv2.putText(out, f"{o.label} {o.seen_ratio:.0%}", (int(x1) + 4, int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
