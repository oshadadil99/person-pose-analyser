"""Writes the run outputs to the output folder.

- timeline.txt      human-readable timeline
- summary.json      durations and bed summary (seconds + "11m 42s" style)
- segments.json     the segments with confidence, for evaluation and later stages
- frame_states.csv  every sampled frame: features, raw state, final state
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src.features import FrameFeatures
from src.frame_rules import FrameState
from src.state_machine import Segment
from src.summary import timeline_lines


def write_outputs(out_dir: str | Path, segments: list[Segment], summary: dict,
                  feats: list[FrameFeatures], raw: list[FrameState], final: list[FrameState]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "timeline.txt").write_text("\n".join(timeline_lines(segments)) + "\n", encoding="utf-8")
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
