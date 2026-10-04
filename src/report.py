"""Writes the run outputs to the output folder.

- timeline.txt      human-readable timeline
- bed_status.txt    the coarse version: IN_BED / OUT / UNKNOWN
- summary.json      durations and bed summary (seconds + "11m 42s" style)
- events.json       bed exits and returns
- decisions.json    NORMAL / MONITOR / ALERT: overall, each fired rule with its reason, and over time
- agent_traces.json / .txt  every agent investigation: observation, thoughts, tool calls, findings, conclusion
- segments.json     the segments with confidence, for evaluation and later stages
- frame_states.csv  every sampled frame: features, raw state, final state
- annotated.mp4     optional (--annotate): skeleton, bed/seats, state and decision per frame
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.alerts import Decision, overall_level
from src.events import Event, bed_status_segments
from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.summary import format_clock, timeline_lines


def write_outputs(out_dir: str | Path, segments: list[Segment], summary: dict, events: list[Event],
                  decisions: list[Decision], decision_tl: list[tuple[float, float, str]], traces: list,
                  feats: list[FrameFeatures], raw: list[FrameState], final: list[FrameState]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "agent_traces.json").write_text(json.dumps([t.to_dict() for t in traces], indent=2), encoding="utf-8")
    (out / "agent_traces.txt").write_text("\n\n".join(t.to_text() for t in traces) + "\n", encoding="utf-8")

    (out / "decisions.json").write_text(json.dumps({
        "overall_decision": overall_level(decisions),
        "decisions": [d.to_dict() for d in decisions],
        "timeline": [{"start": format_clock(a), "end": format_clock(b), "level": lvl} for a, b, lvl in decision_tl],
    }, indent=2), encoding="utf-8")

    (out / "timeline.txt").write_text("\n".join(timeline_lines(segments, decision_tl)) + "\n", encoding="utf-8")
    (out / "bed_status.txt").write_text("".join(
        f"{format_clock(a)} – {format_clock(b)}  {st}\n" for a, b, st in bed_status_segments(segments)),
        encoding="utf-8")
    (out / "events.json").write_text(json.dumps([e.to_dict() for e in events], indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "segments.json").write_text(json.dumps(
        [{"start_sec": round(s.start_sec, 2), "end_sec": round(s.end_sec, 2),
          "state": s.state.value, "confidence": s.confidence} for s in segments], indent=2), encoding="utf-8")

    if not feats:
        return
    _write_frame_csv(out, feats, raw, final)


LEVEL_COLOURS = {"NORMAL": (80, 200, 80), "MONITOR": (0, 200, 255), "ALERT": (40, 40, 255)}


def write_annotated_video(video_path: str, out_path: str | Path, frames: list, patient: list, scenes,
                          segments: list[Segment], decision_tl: list[tuple[float, float, str]], cfg: dict) -> None:
    """annotated.mp4: bed/seat outlines, the patient's skeleton (others grey), state and decision.

    Written at the sampling rate, so it plays faster than real time."""
    import cv2
    from src.perception import draw_person
    from src.scene import draw_scene
    from src.video_io import resize_to_width

    def at(items, t):
        return next((x for x in items if x[0] <= t < x[1]), items[-1])

    by_idx = {f.frame_idx: (f, p) for f, p in zip(frames, patient)}
    kp_min = cfg["features"]["kp_min_conf"]
    cap = cv2.VideoCapture(str(video_path))
    writer, idx = None, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        item = by_idx.get(idx)
        idx += 1
        if item is None:
            continue
        fr, p = item
        img = draw_scene(img, scenes.at(fr.t_sec))
        for person in fr.persons:
            mine = p is not None and person.track_id == p.track_id
            draw_person(img, person, (0, 255, 0) if mine else (150, 150, 150), kp_min)
        seg = next((s for s in segments if s.start_sec <= fr.t_sec < s.end_sec), segments[-1])
        level = at(decision_tl, fr.t_sec)[2]
        img, _ = resize_to_width(img, cfg["sampling"]["max_width"])
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, f"{format_clock(fr.t_sec)}  {seg.state.value}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, level, (img.shape[1] - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LEVEL_COLOURS[level], 2)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg["sampling"]["fps"], (w, h))
        writer.write(img)
    cap.release()
    if writer:
        writer.release()


def _write_frame_csv(out: Path, feats, raw, final) -> None:
    with open(out / "frame_states.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(asdict(feats[0]).keys())
        w.writerow(cols + ["raw_state", "raw_conf", "raw_reason", "state", "confidence", "reason"])
        for ft, r, s in zip(feats, raw, final):
            values = [round(v, 3) if isinstance(v, float) else v for v in asdict(ft).values()]
            w.writerow(values + [r.state.value, r.confidence, r.reason, s.state.value, s.confidence, s.reason])
