"""Scores pipeline outputs against hand-made labels.

    python -m src.evaluate --pred outputs/test2 \
        --gt data/ground_truth/test2_segments.csv --gt-events data/ground_truth/test2_events.csv \
        --out docs/evaluation.md
    # several runs side by side (ablation):
    python -m src.evaluate --run no_agent=outputs/test2_noagent --run rules=outputs/test2_rules ... \
        --gt ... --gt-events ... --out docs/evaluation.md

What is measured:
- States: compared once per second (`evaluation.resolution_sec`) at the
  middle of each second. Accuracy, per-state precision/recall/F1, confusion
  matrix (saved as PNG). Seconds labelled IGNORE are skipped.
- Bed events: a predicted event matches a labelled one of the same kind if it
  starts within `evaluation.event_tolerance_sec`; each label can be matched
  once (closest first). Precision, recall, false and missed events.
  Predicted events inside IGNORE ranges are not scored.
- Durations: seconds per state over the scored seconds, labels vs prediction,
  and the absolute error.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.states import IN_BED_STATES, State
from src.summary import format_clock

IGNORE = "IGNORE"
SIMILAR_PAIRS = [("SITTING_ON_BED", "LYING_IN_BED"), ("STANDING", "WALKING"),
                 ("SITTING_ON_BED", "STANDING"), ("SITTING_OUTSIDE_BED", "STANDING"),
                 ("OUT_OF_BED", "UNKNOWN"), ("LYING_OUTSIDE_BED", "LYING_IN_BED")]


# ---------------------------------------------------------------- loading

def load_gt_segments(path: str | Path) -> list[tuple[float, float, str]]:
    with open(path, encoding="utf-8") as f:
        return [(float(r["start_sec"]), float(r["end_sec"]), r["state"].strip().upper()) for r in csv.DictReader(f)]


def load_gt_events(path: str | Path) -> list[tuple[str, float]]:
    with open(path, encoding="utf-8") as f:
        return [(r["event"].strip().lower(), float(r["time_sec"])) for r in csv.DictReader(f)]


def load_run(pred_dir: str | Path) -> dict:
    d = Path(pred_dir)
    segments = [(s["start_sec"], s["end_sec"], s["state"]) for s in json.loads((d / "segments.json").read_text())]
    events = [(e["event"], e["start_sec"]) for e in json.loads((d / "events.json").read_text())]
    summary = json.loads((d / "summary.json").read_text())
    return {"segments": segments, "events": events, "agent": summary.get("agent", {})}


# ---------------------------------------------------------------- states

def state_at(segments: list[tuple[float, float, str]], t: float) -> str | None:
    for a, b, s in segments:
        if a <= t < b:
            return s
    return None


def per_second(gt, pred, resolution: float = 1.0) -> tuple[list[str], list[str]]:
    """Labels and predictions sampled at the middle of every scored second."""
    end = max(b for _, b, _ in gt)
    y_true, y_pred = [], []
    for t in np.arange(resolution / 2, end, resolution):
        g = state_at(gt, t)
        if g is None or g == IGNORE:
            continue
        y_true.append(g)
        y_pred.append(state_at(pred, t) or State.UNKNOWN.value)
    return y_true, y_pred


def state_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix
    labels = [s.value for s in State if s.value in set(y_true) | set(y_pred)]
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    in_bed = lambda s: "IN_BED" if State(s) in IN_BED_STATES else "OUT"   # noqa: E731
    bed_acc = float(np.mean([in_bed(a) == in_bed(b) for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    confusions = []
    for a, b in SIMILAR_PAIRS:
        if a in labels and b in labels:
            i, j = labels.index(a), labels.index(b)
            confusions.append((a, b, int(cm[i, j]), int(cm[j, i])))
    return {"labels": labels, "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0,
            "macro_f1": report["macro avg"]["f1-score"], "per_class": {l: report[l] for l in labels},
            "confusion": cm.tolist(), "bed_status_accuracy": bed_acc, "similar_confusions": confusions,
            "scored_seconds": len(y_true)}


def save_confusion_png(metrics: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(metrics["confusion"])
    labels = [l.replace("_", " ").lower() for l in metrics["labels"]]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("labelled (ground truth)")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- events

def match_events(gt: list[tuple[str, float]], pred: list[tuple[str, float]], tolerance: float,
                 ignore: list[tuple[float, float]]) -> dict:
    pred = [(k, t) for k, t in pred if not any(a <= t < b for a, b in ignore)]
    out = {}
    for kind in ("bed_exit", "return_to_bed"):
        g = [t for k, t in gt if k == kind]
        p = [t for k, t in pred if k == kind]
        pairs = sorted(((abs(pt - gt_t), i, j) for i, gt_t in enumerate(g) for j, pt in enumerate(p)
                        if abs(pt - gt_t) <= tolerance))
        used_g, used_p, matched = set(), set(), []
        for dt, i, j in pairs:
            if i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
                matched.append((g[i], p[j]))
        tp = len(matched)
        out[kind] = {
            "labelled": len(g), "predicted": len(p), "matched": tp,
            "precision": tp / len(p) if p else None, "recall": tp / len(g) if g else None,
            "false": [p[j] for j in range(len(p)) if j not in used_p],
            "missed": [g[i] for i in range(len(g)) if i not in used_g],
            "pairs": sorted(matched),
        }
    return out


# ---------------------------------------------------------------- durations

def durations(y_true: list[str], y_pred: list[str], resolution: float) -> list[tuple[str, float, float]]:
    states = [s.value for s in State if s.value in set(y_true) | set(y_pred)]
    return [(s, y_true.count(s) * resolution, y_pred.count(s) * resolution) for s in states]


def evaluate_run(run: dict, gt, gt_events, cfg) -> dict:
    ec = cfg["evaluation"]
    y_true, y_pred = per_second(gt, run["segments"], ec["resolution_sec"])
    ignore = [(a, b) for a, b, s in gt if s == IGNORE]
    return {"states": state_metrics(y_true, y_pred),
            "events": match_events(gt_events, run["events"], ec["event_tolerance_sec"], ignore),
            "durations": durations(y_true, y_pred, ec["resolution_sec"]),
            "agent": run["agent"]}


# ---------------------------------------------------------------- report

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _mmss(sec: float) -> str:
    return format_clock(sec)


def ablation_table(results: dict[str, dict]) -> list[str]:
    lines = ["| Run | State accuracy | Macro F1 | In/out of bed accuracy | Exit precision | Exit recall "
             "| False exits | Return precision | Return recall | Agent tool calls | Gemini calls |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s, ex, rt, ag = r["states"], r["events"]["bed_exit"], r["events"]["return_to_bed"], r["agent"] or {}
        gem = sum((ag.get(k) or {}).get("api_calls", 0) + (ag.get(k) or {}).get("cache_hits", 0)
                  for k in ("vlm_stats", "llm_policy_stats"))
        lines.append(f"| {name} | {_pct(s['accuracy'])} | {s['macro_f1']:.2f} | {_pct(s['bed_status_accuracy'])} "
                     f"| {_pct(ex['precision'])} | {_pct(ex['recall'])} | {len(ex['false'])} "
                     f"| {_pct(rt['precision'])} | {_pct(rt['recall'])} | {ag.get('tool_calls', 0)} | {gem} |")
    return lines


def detail_section(name: str, r: dict, png_rel: str) -> list[str]:
    s = r["states"]
    out = [f"### Run `{name}`", "",
           f"Scored {s['scored_seconds']} s. State accuracy **{_pct(s['accuracy'])}**, macro F1 "
           f"**{s['macro_f1']:.2f}**, in/out of bed accuracy **{_pct(s['bed_status_accuracy'])}**.", "",
           "| State | Precision | Recall | F1 | Labelled seconds |", "|---|---|---|---|---|"]
    for label, m in s["per_class"].items():
        out.append(f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1-score']:.2f} | {int(m['support'])} |")
    out += ["", f"![Confusion matrix]({png_rel})", "", "Confusion between similar states (seconds):", "",
            "| A | B | labelled A, predicted B | labelled B, predicted A |", "|---|---|---|---|"]
    out += [f"| {a} | {b} | {ab} | {ba} |" for a, b, ab, ba in s["similar_confusions"] if ab or ba]

    out += ["", "Bed events (match within tolerance):", "",
            "| Event | Labelled | Predicted | Matched | Precision | Recall | Timing errors | False | Missed |",
            "|---|---|---|---|---|---|---|---|---|"]
    for kind, e in r["events"].items():
        timing = ", ".join(f"{p - g:+.0f}s" for g, p in e["pairs"]) or "-"
        out.append(f"| {kind} | {e['labelled']} | {e['predicted']} | {e['matched']} | {_pct(e['precision'])} "
                   f"| {_pct(e['recall'])} | {timing} | {', '.join(_mmss(t) for t in e['false']) or '-'} "
                   f"| {', '.join(_mmss(t) for t in e['missed']) or '-'} |")

    out += ["", "Duration per state over the scored seconds:", "",
            "| State | Labelled | Predicted | Error |", "|---|---|---|---|"]
    for state, g, p in r["durations"]:
        out.append(f"| {state} | {_mmss(g)} | {_mmss(p)} | {abs(p - g):.0f} s |")
    return out + [""]


def write_report(results: dict[str, dict], out_path: Path, gt_path: str, detail_runs: list[str]) -> None:
    lines = ["# Evaluation results", "",
             f"Labels: `{gt_path}`. Generated by `src/evaluate.py`.", ""]
    if len(results) > 1:
        lines += ["## Comparison", ""] + ablation_table(results) + [""]
    lines += ["## Details", ""]
    for name in detail_runs:
        png = out_path.parent / "images" / f"confusion_{name}.png"
        save_confusion_png(results[name]["states"], png, f"States per second: {name}")
        lines += detail_section(name, results[name], f"images/{png.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score pipeline outputs against labels")
    ap.add_argument("--pred", help="one output folder")
    ap.add_argument("--run", action="append", default=[], help="name=output_folder, repeatable")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-events", required=True)
    ap.add_argument("--out", required=True, help="markdown report path")
    ap.add_argument("--detail", action="append", help="runs to show in detail (default: all)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs = dict(r.split("=", 1) for r in args.run)
    if args.pred:
        runs[Path(args.pred).name] = args.pred
    gt, gt_events = load_gt_segments(args.gt), load_gt_events(args.gt_events)
    results = {name: evaluate_run(load_run(d), gt, gt_events, cfg) for name, d in runs.items()}

    out = Path(args.out)
    write_report(results, out, args.gt, args.detail or list(results))
    (out.with_suffix(".json")).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("\n".join(ablation_table(results)))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
