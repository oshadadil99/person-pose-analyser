"""Runs the full pipeline on one video.

    python -m src.main --video data/videos/room1.mp4 --out outputs/room1
    python -m src.main --video data/videos/room1.mp4 --out outputs/room1 --end 48

Slow steps are cached in the output folder and reused on the next run:
perception.jsonl (YOLO pose) and scene.json (bed + seats). Use --rerun to
recompute them, e.g. after changing the YOLO model. Everything after that
takes seconds, so thresholds can be tuned and re-run quickly.
"""

import argparse
import logging
from pathlib import Path

import cv2

from src.alerts import decision_timeline, evaluate_alerts, label_events, overall_level
from src.config import load_config
from src.events import detect_events, out_of_bed_periods
from src.features import extract_features
from src.frame_rules import classify_frames
from src.patient import patient_per_frame, select_patient_ids, summarize_tracks
from src.perception import load_perception, run_perception, save_perception
from src.report import write_outputs
from src.scene import detect_scene, draw_scene, load_scene, save_scene
from src.state_machine import build_timeline
from src.summary import build_summary, format_clock, format_duration, timeline_lines
from src.video_io import get_video_info, read_frame_at, resize_to_width

log = logging.getLogger("main")


def get_perception(video: str, out: Path, cfg: dict, rerun: bool):
    cache = out / "perception.jsonl"
    if cache.exists() and not rerun:
        meta, frames = load_perception(cache)
        if Path(meta["video"]).name == Path(video).name:
            log.info("Reusing %s", cache)
            return meta, frames
        log.info("Cached perception is for another video, re-running")
    meta, frames = run_perception(video, cfg)
    save_perception(cache, meta, frames)
    return meta, frames


def get_scene(video: str, out: Path, cfg: dict, rerun: bool, override: str | None):
    if override:
        return load_scene(override)
    cache = out / "scene.json"
    if cache.exists() and not rerun:
        log.info("Reusing %s", cache)
        return load_scene(cache)
    scene = detect_scene(video, cfg)
    save_scene(cache, scene)
    frame = read_frame_at(video, get_video_info(video).duration_sec / 2)
    if frame is not None:
        preview, _ = resize_to_width(draw_scene(frame, scene), cfg["sampling"]["max_width"])
        cv2.imwrite(str(out / "scene_preview.jpg"), preview)
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description="Bed-activity analysis of one video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scene", help="use this scene.json (e.g. hand-drawn bed) instead of auto-detection")
    ap.add_argument("--start", type=float, default=0.0, help="analyse from this second")
    ap.add_argument("--end", type=float, help="analyse up to this second (default: end of video)")
    ap.add_argument("--rerun", action="store_true", help="recompute perception and scene even if cached")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta, frames = get_perception(args.video, out, cfg, args.rerun)
    scene = get_scene(args.video, out, cfg, args.rerun, args.scene)

    start = max(0.0, args.start)
    end = min(args.end, meta["duration_sec"]) if args.end else meta["duration_sec"]
    frames = [f for f in frames if start <= f.t_sec < end]
    if not frames:
        raise SystemExit(f"No frames between {start}s and {end}s")

    tracks = summarize_tracks(frames, scene.bed.polygon if scene.bed else None, cfg)
    patient = patient_per_frame(frames, select_patient_ids(tracks, cfg))
    feats = extract_features(frames, patient, scene, cfg, frame_height=meta["height"])
    raw = classify_frames(feats, cfg)
    final, segments = build_timeline(raw, feats, cfg, start, end)
    events, segments = detect_events(segments, cfg)
    summary = build_summary(segments, start, end,
                            bed_exit_count=sum(e.event == "bed_exit" for e in events),
                            bed_return_count=sum(e.event == "return_to_bed" for e in events))
    summary["out_of_bed_periods"] = out_of_bed_periods(segments)
    decisions = evaluate_alerts(segments, events, feats, cfg)
    label_events(events, cfg)
    decision_tl = decision_timeline(decisions, start, end)
    summary["overall_decision"] = overall_level(decisions)
    write_outputs(out, segments, summary, events, decisions, decision_tl, feats, raw, final)

    print("\nTimeline (state, decision)")
    print("\n".join("  " + line for line in timeline_lines(segments, decision_tl)))
    print("\nBed events")
    for e in events or []:
        d = e.to_dict()
        print(f"  {d['event']:14s} start {d['start_time']}  confirmed {d['confirmed_time']}  "
              f"{d['previous_state']} -> {d['current_state']}  conf {d['confidence']}")
    if not events:
        print("  none")
    print(f"\nOverall decision: {summary['overall_decision']}")
    print("Decision over time")
    for a, b, level in decision_tl:
        why = "; ".join(f"{d.rule}: {d.reason}" for d in decisions if d.start_sec <= a < d.end_sec)
        print(f"  {format_clock(a)} – {format_clock(b)}  {level:8s}{why or 'no rule fired'}")
    print("\nTime per state")
    for state, sec in summary["activity_duration_sec"].items():
        if sec > 0:
            print(f"  {state:22s} {format_duration(sec):>8s}")
    hr = summary["human_readable"]["bed_summary"]
    print(f"\nIn bed {hr['time_in_bed']}, out of bed {hr['time_out_of_bed']} "
          f"(of which unknown {format_duration(summary['unknown_sec'])})")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
