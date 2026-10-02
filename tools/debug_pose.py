"""Run YOLO-pose + tracking on a video and write a video with skeletons drawn.

Used to eyeball perception quality before building anything on top of it:
are people detected, are keypoints sensible, do track IDs stay stable?

    python tools/debug_pose.py --video data/videos/room1.mp4 --out outputs/room1
    python tools/debug_pose.py --video data/videos/room1.mp4 --out outputs/room1 --reuse
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "src" imports work when run as a script

from src.config import load_config
from src.perception import draw_person, load_perception, run_perception, save_perception
from src.video_io import resize_to_width, sample_frames

# One colour per track ID so ID switches are easy to spot.
COLORS = [(0, 200, 0), (0, 140, 255), (255, 80, 80), (200, 0, 200), (0, 220, 220), (255, 200, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reuse", action="store_true", help="load perception.jsonl instead of re-running YOLO")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = Path(args.out)
    cache = out / "perception.jsonl"

    if args.reuse and cache.exists():
        meta, frames = load_perception(cache)
    else:
        meta, frames = run_perception(args.video, cfg)
        save_perception(cache, meta, frames)

    # Draw on full-resolution frames (that's what the coordinates refer to), then shrink for the file.
    by_idx = {f.frame_idx: f for f in frames}
    kp_min_conf = cfg["features"]["kp_min_conf"]
    max_w = cfg["sampling"]["max_width"]
    writer = None
    cap = cv2.VideoCapture(args.video)
    frame_idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fp = by_idx.get(frame_idx)
        frame_idx += 1
        if fp is None:
            continue
        for p in fp.persons:
            draw_person(img, p, COLORS[p.track_id % len(COLORS)], kp_min_conf,
                        label=f"id {p.track_id} {p.conf:.2f}")
        cv2.putText(img, f"t={fp.t_sec:7.2f}s  persons={len(fp.persons)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        img, _ = resize_to_width(img, max_w)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out / "debug_pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     meta["sample_fps"], (w, h))
        writer.write(img)
    cap.release()
    if writer:
        writer.release()

    # Quick numbers to judge perception at a glance.
    n = len(frames)
    with_person = sum(1 for f in frames if f.persons)
    count_hist = Counter(len(f.persons) for f in frames)
    track_frames = Counter(p.track_id for f in frames for p in f.persons)
    print(f"\nSampled frames:        {n} ({meta['duration_sec']:.1f}s at {meta['sample_fps']} fps)")
    print(f"Frames with a person:  {with_person} ({100 * with_person / max(n, 1):.0f}%)")
    print(f"Persons per frame:     {dict(sorted(count_hist.items()))}")
    print(f"Track IDs (frames):    {dict(track_frames.most_common())}")
    print(f"Wrote {cache} and {out / 'debug_pose.mp4'}")


if __name__ == "__main__":
    main()
