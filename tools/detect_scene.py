"""Detect the bed and seats in a video and write scene.json + a preview image.

If perception.jsonl already exists in the output folder (from debug_pose.py),
also pick the patient track and print a per-track table.

    python tools/detect_scene.py --video data/videos/room1.mp4 --out outputs/room1
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.patient import select_patient_ids, summarize_tracks
from src.perception import load_perception
from src.scene import detect_scene, draw_scene, save_scene
from src.video_io import get_video_info, read_frame_at, resize_to_width


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = Path(args.out)

    scene = detect_scene(args.video, cfg)
    save_scene(out / "scene.json", scene)

    frame = read_frame_at(args.video, get_video_info(args.video).duration_sec / 2)
    preview, _ = resize_to_width(draw_scene(frame, scene), cfg["sampling"]["max_width"])
    cv2.imwrite(str(out / "scene_preview.jpg"), preview)

    print(f"\nBed:   {scene.bed.label + f' (seen {scene.bed.seen_ratio:.0%})' if scene.bed else 'NOT FOUND'}")
    print(f"Seats: {[f'{s.label} ({s.seen_ratio:.0%})' for s in scene.seats] or 'none'}")
    print(f"Wrote {out / 'scene.json'} and {out / 'scene_preview.jpg'}")

    perception = out / "perception.jsonl"
    if perception.exists():
        _, frames = load_perception(perception)
        tracks = summarize_tracks(frames, scene.bed.polygon if scene.bed else None, cfg)
        ids = select_patient_ids(tracks, cfg)
        print("\nTrack   first_t   last_t  frames  near_bed  patient")
        for s in sorted(tracks.values(), key=lambda s: s.first_t):
            print(f"{s.track_id:5d} {s.first_t:9.1f} {s.last_t:8.1f} {s.n_frames:7d} {s.near_bed_frames:9d}"
                  f"  {'yes' if s.track_id in ids else ''}")


if __name__ == "__main__":
    main()
