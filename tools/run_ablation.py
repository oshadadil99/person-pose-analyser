"""Run the pipeline in several configurations on one video and compare them.

What does each layer add? Same video, same YOLO results (copied from the base
output folder, so YOLO runs once), four variants:

    no_agent         timeline + events + alerts only
    agent_rules      + investigator agent, rule policy, no Gemini
    agent_rules_vlm  + rule policy that may call Gemini vision when stuck
    agent_llm        + Gemini chooses the tools (LLM policy) and can use vision

    python tools/run_ablation.py --video "data/videos/test 2.mp4" --base outputs/test2 \
        --gt data/ground_truth/test2_segments.csv --gt-events data/ground_truth/test2_events.csv \
        --vlm vertex --out docs/evaluation.md

Gemini answers are cached, so a rerun costs no API calls.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    "no_agent": ["--no-agent", "--vlm", "none"],
    "agent_rules": ["--vlm", "none", "--policy", "rules"],
    "agent_rules_vlm": ["--vlm", "{vlm}", "--policy", "rules"],
    "agent_llm": ["--vlm", "{vlm}", "--policy", "llm"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--base", required=True, help="output folder that already has perception.jsonl and scene.json")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-events", required=True)
    ap.add_argument("--vlm", default="vertex", choices=["vertex", "aistudio"])
    ap.add_argument("--out", default="docs/evaluation.md")
    ap.add_argument("--only", nargs="*", help="run just these variants")
    args = ap.parse_args()

    base = Path(args.base)
    runs = []
    for name, flags in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        out = base.parent / f"{base.name}_{name}"
        out.mkdir(parents=True, exist_ok=True)
        for f in ("perception.jsonl", "scene.json"):
            shutil.copy(base / f, out / f)
        cmd = [sys.executable, "-m", "src.main", "--video", args.video, "--out", str(out)] + \
              [x.format(vlm=args.vlm) for x in flags]
        print(f"== {name}: {' '.join(cmd[3:])}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        runs += ["--run", f"{name}={out}"]

    subprocess.run([sys.executable, "-m", "src.evaluate", *runs, "--gt", args.gt, "--gt-events", args.gt_events,
                    "--out", args.out], check=True)


if __name__ == "__main__":
    main()
