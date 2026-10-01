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

TODO

## Design

TODO

## Alert rules

TODO

## Results

TODO

## Limitations and what I'd do with more time

TODO

## Licence note

YOLO11 (Ultralytics) is AGPL-3.0.
