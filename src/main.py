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

from src.agent.investigator import apply_to_events, apply_to_segments, run_agent
from src.agent.policy_llm import LLMPolicy
from src.agent.policy_rules import RulePolicy
from src.agent.tools import AgentContext
from src.agent.vlm import VLM, make_client
from src.alerts import LEVELS, decision_timeline, evaluate_alerts, label_events, overall_level
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
from src.triggers import find_triggers
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
    ap.add_argument("--no-agent", action="store_true", help="skip the investigator agent (for comparison)")
    ap.add_argument("--vlm", choices=["none", "aistudio", "vertex"], help="Gemini provider (default: config)")
    ap.add_argument("--policy", choices=["rules", "llm"], help="agent policy (default: config)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
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
    patient_ids = select_patient_ids(tracks, cfg)
    patient = patient_per_frame(frames, patient_ids)
    feats = extract_features(frames, patient, scene, cfg, frame_height=meta["height"])
    raw = classify_frames(feats, cfg)
    final, segments = build_timeline(raw, feats, cfg, start, end)
    events, segments = detect_events(segments, cfg)

    # Investigator agent: look into the ambiguous moments, correct the timeline,
    # then re-detect events on the corrected timeline.
    traces, vlm, policy = [], None, None
    if cfg["agent"]["enabled"] and not args.no_agent:
        vlm, policy = setup_gemini(args, cfg, frames, patient)
        triggers = find_triggers(segments, events, raw, feats, cfg)
        ctx = AgentContext(segments, final, feats, frames, patient_ids, cfg, vlm=vlm)
        traces = run_agent(triggers, ctx, policy)
        segments = apply_to_segments(segments, traces)
        events, segments = detect_events(segments, cfg)
        events = apply_to_events(events, traces, cfg["evaluation"]["event_tolerance_sec"])

    summary = build_summary(segments, start, end,
                            bed_exit_count=sum(e.event == "bed_exit" for e in events),
                            bed_return_count=sum(e.event == "return_to_bed" for e in events))
    summary["out_of_bed_periods"] = out_of_bed_periods(segments)
    decisions = evaluate_alerts(segments, events, feats, cfg)
    label_events(events, cfg)
    decision_tl = decision_timeline(decisions, start, end)
    summary["overall_decision"] = overall_level(decisions)
    for tr in traces:   # the decision in force during each investigated moment
        levels = [lvl for a, b, lvl in decision_tl if a <= tr.trigger.t_end and b > tr.trigger.t_start]
        tr.decision = max(levels, key=LEVELS.index) if levels else "NORMAL"
    summary["agent"] = {"enabled": not args.no_agent and cfg["agent"]["enabled"],
                        "vlm_provider": cfg["vlm"]["provider"] if vlm else "none",
                        "policy": policy.name if policy else "rules",
                        "investigations": len(traces), "tool_calls": sum(len(t.steps) for t in traces),
                        "outcomes": {o: sum(t.conclusion.outcome == o for t in traces)
                                     for o in sorted({t.conclusion.outcome for t in traces})},
                        "vlm_stats": vlm.stats if vlm else None,
                        "llm_policy_stats": policy.caller.stats if isinstance(policy, LLMPolicy) else None}
    write_outputs(out, segments, summary, events, decisions, decision_tl, traces, feats, raw, final)
    print_report(segments, events, decisions, decision_tl, traces, summary, out)


def setup_gemini(args, cfg: dict, frames, patient):
    """Create the VLM and pick the agent policy. Falls back to offline with a warning."""
    if args.vlm:
        cfg["vlm"]["provider"] = args.vlm
    if args.policy:
        cfg["agent"]["policy"] = args.policy
    provider = cfg["vlm"]["provider"]
    try:
        client = make_client(provider)
    except RuntimeError as e:     # missing key / project in .env
        log.warning("Gemini (%s) not available: %s Running offline.", provider, e)
        client = None
    if client is None:
        if cfg["agent"]["policy"] == "llm":
            log.warning("LLM policy needs Gemini; using the rule policy")
        return None, RulePolicy()

    boxes = {round(f.t_sec, 1): p.box for f, p in zip(frames, patient) if p is not None}

    def patient_box_at(t: float):
        near = [k for k in boxes if abs(k - t) <= 0.3]
        return boxes[min(near, key=lambda k: abs(k - t))] if near else None

    vlm = VLM(client, cfg, args.video, patient_box_at)
    policy = LLMPolicy(client, cfg) if cfg["agent"]["policy"] == "llm" else RulePolicy()
    log.info("Gemini via %s (%s), agent policy: %s", provider, cfg["vlm"]["model"], policy.name)
    return vlm, policy


def print_report(segments, events, decisions, decision_tl, traces, summary, out) -> None:
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
    print(f"\nAgent investigations: {len(traces)}")
    for tr in traces:
        c = tr.conclusion
        print(f"  {tr.trace_id} {tr.trigger.kind:16s} {format_clock(tr.trigger.t_start)}  "
              f"{c.outcome}{' -> ' + c.state.value if c.state else ''} ({len(tr.steps)} tool calls)")
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
