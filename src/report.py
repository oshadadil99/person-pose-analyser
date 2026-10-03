"""Writes the run outputs to the output folder.

- timeline.txt      human-readable timeline
- bed_status.txt    the coarse version: IN_BED / OUT / UNKNOWN
- summary.json      durations and bed summary (seconds + "11m 42s" style)
- events.json       bed exits and returns
- decisions.json    NORMAL / MONITOR / ALERT: overall, each fired rule with its reason, and over time
- segments.json     the segments with confidence, for evaluation and later stages
- frame_states.csv  every sampled frame: features, raw state, final state
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
                  decisions: list[Decision], decision_tl: list[tuple[float, float, str]],
                  feats: list[FrameFeatures], raw: list[FrameState], final: list[FrameState]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

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
    with open(out / "frame_states.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(asdict(feats[0]).keys())
        w.writerow(cols + ["raw_state", "raw_conf", "raw_reason", "state", "confidence", "reason"])
        for ft, r, s in zip(feats, raw, final):
            values = [round(v, 3) if isinstance(v, float) else v for v in asdict(ft).values()]
            w.writerow(values + [r.state.value, r.confidence, r.reason, s.state.value, s.confidence, s.reason])
