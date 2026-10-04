"""Show the per-frame state guesses on the video, to check and tune the rules.

Needs perception.jsonl (debug_pose.py) and scene.json (detect_scene.py) in the
output folder. Writes frame_states.csv (every feature + state per frame) and
states_debug.mp4, and prints how the raw (unsmoothed) states change over time.

    python tools/debug_states.py --video data/videos/room1.mp4 --out outputs/room1
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import extract_features
from src.frame_rules import classify_frames
from src.patient import patient_per_frame, select_patient_ids, summarize_tracks
from src.perception import draw_person, load_perception
from src.scene import draw_scene, load_scenes
from src.video_io import resize_to_width


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-video", action="store_true", help="only write the CSV and print the summary")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(args.config)
    out = Path(args.out)
    _, frames = load_perception(out / "perception.jsonl")
    scenes = load_scenes(out / "scene.json")

    def bed_at(t):
        sc = scenes.at(t)
        return sc.bed.polygon if sc and sc.bed else None

    tracks = summarize_tracks(frames, bed_at, cfg)
    patient = patient_per_frame(frames, select_patient_ids(tracks, cfg))
    feats = extract_features(frames, patient, scenes, cfg)
    states = classify_frames(feats, cfg)

    with open(out / "frame_states.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(asdict(feats[0]).keys())
        w.writerow(cols + ["state", "confidence", "reason"])
        for ft, st in zip(feats, states):
            w.writerow([fmt(v) for v in asdict(ft).values()] + [st.state.value, st.confidence, st.reason])

    # Raw runs: consecutive frames with the same state. Lots of tiny runs = flicker.
    print("\nRaw per-frame states (before smoothing):")
    run_start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i].state != states[run_start].state:
            run = states[run_start:i]
            mean_c = sum(s.confidence for s in run) / len(run)
            print(f"  {run[0].t_sec:6.1f} - {run[-1].t_sec:6.1f}s  {run[0].state.value:20s} "
                  f"{len(run):3d} frames  conf {mean_c:.2f}")
            run_start = i
    counts = Counter(s.state.value for s in states)
    print("\nFrames per state:", dict(counts.most_common()))
    print(f"Wrote {out / 'frame_states.csv'}")

    if args.no_video:
        return
    by_idx = {fr.frame_idx: (fr, p, st) for fr, p, st in zip(frames, patient, states)}
    kp_min = cfg["features"]["kp_min_conf"]
    cap = cv2.VideoCapture(args.video)
    writer, idx = None, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        item = by_idx.get(idx)
        idx += 1
        if item is None:
            continue
        fr, p, st = item
        img = draw_scene(img, scenes.at(fr.t_sec))
        for other in fr.persons:
            if other is not p:
                draw_person(img, other, (128, 128, 128), kp_min)
        if p is not None:
            draw_person(img, p, (0, 255, 0), kp_min)
        cv2.rectangle(img, (0, 0), (img.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(img, f"{fr.t_sec:6.1f}s  {st.state.value}  {st.confidence:.2f}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, st.reason[:90], (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img, _ = resize_to_width(img, cfg["sampling"]["max_width"])
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out / "states_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     cfg["sampling"]["fps"], (w, h))
        writer.write(img)
    cap.release()
    if writer:
        writer.release()
    print(f"Wrote {out / 'states_debug.mp4'}")


if __name__ == "__main__":
    main()
