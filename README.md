# person-pose-analyser

Watches a fixed-camera bedroom video of an elderly person and works out what
they are doing over time, when they leave or return to bed, how long they spend
in each state, and whether the situation is NORMAL, needs MONITORing, or needs
an ALERT.

YOLO11-pose looks at every sampled frame, a state machine turns the per-frame
guesses into a timeline, and an agent investigates only the ambiguous moments
(looking back and forward in time, and asking Gemini when geometry isn't
enough). Alert decisions are made by plain rules, not by the LLM.

> Work in progress.

## Setup

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# GPU: install the CUDA build of torch first (pick the right one at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

Gemini is only needed if you want the VLM step. Copy `.env.example` to `.env`
and fill it in. For Vertex AI, point `GOOGLE_APPLICATION_CREDENTIALS` at a
service-account key with the Vertex AI User role. Running with `--vlm none`
needs no credentials at all.

Run the tests:

```bash
python -m pytest
```

## Usage

Put a video in `data/videos/` and run:

```bash
python -m src.main --video data/videos/room1.mp4 --out outputs/room1
```

The first run detects people (YOLO11-pose) and furniture (YOLO11-seg) and
caches both in the output folder; later runs reuse them and take seconds.
Useful options:

- `--start 10 --end 48` analyse only part of the video
- `--rerun` recompute the cached detections
- `--scene path/to/scene.json` use a hand-drawn bed instead of the detected one
  (draw it with `python tools/draw_bed_polygon.py --video ... --scene ...`)

Outputs in the output folder:

| File | Content |
|---|---|
| `timeline.txt` | `00:00 – 00:11  LYING_IN_BED`, one line per segment |
| `bed_status.txt` | the coarse version: `IN_BED` / `OUT` / `UNKNOWN` |
| `events.json` | bed exits and returns with start/confirmed time, states, confidence, decision |
| `decisions.json` | NORMAL / MONITOR / ALERT: overall, each fired rule with reason, decision over time |
| `summary.json` | time per state, time in/out of bed, exit/return counts, out-of-bed periods, final state |
| `segments.json` | segments with confidence |
| `frame_states.csv` | every sampled frame: measurements, raw state, final state |
| `scene_preview.jpg` | detected bed and seats |

Debug tools: `tools/debug_pose.py` (skeleton video), `tools/detect_scene.py`
(bed/seat detection and patient track table), `tools/debug_states.py`
(per-frame states drawn on the video).

**Time conventions.** Time in bed = lying in bed + sitting on bed. Everything
else counts as out of bed, including UNKNOWN, so in + out always equals the
analysed duration. Unknown time is also reported separately (`unknown_sec`).

## Design

TODO

## Alert rules

Decisions come from fixed rules, not the LLM, so every alert is explainable
and testable. Full reasoning per rule is in [docs/alert_rules.md](docs/alert_rules.md).

| Level | When |
|---|---|
| NORMAL | lying, sitting, standing or walking, none of the below |
| MONITOR | sitting on the bed > 3 min; activity unknown > 1 min; any confirmed bed exit (until back in bed); out of bed > 10 min |
| ALERT | lying outside the bed > 20 s (possible fall); away from bed > 20 min |

A caregiver in view lowers the absence rules by one level, but never the
possible-fall alert. The output `decisions.json` lists each fired rule with
its reason, the overall decision, and the decision over time.

## Results

TODO

## Limitations and what I'd do with more time

TODO

## Licence note

YOLO11 (Ultralytics) is AGPL-3.0.
