import pytest

from src.config import load_config
from src.evaluate import durations, evaluate_run, match_events, per_second, state_metrics


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


GT = [(0, 5, "IGNORE"), (5, 65, "LYING_IN_BED"), (65, 85, "SITTING_ON_BED"), (85, 95, "STANDING"),
      (95, 125, "WALKING")]


def test_per_second_skips_ignore():
    y_true, y_pred = per_second(GT, [(0, 125, "LYING_IN_BED")])
    assert len(y_true) == 120                     # 125 s minus 5 s ignored
    assert "IGNORE" not in y_true


def test_perfect_prediction(cfg):
    run = {"segments": [(a, b, s) for a, b, s in GT if s != "IGNORE"], "events": [], "agent": {}}
    r = evaluate_run(run, GT, [], cfg)
    assert r["states"]["accuracy"] == 1.0
    assert all(abs(g - p) < 1e-9 for _, g, p in r["durations"])


def test_accuracy_and_bed_status(cfg):
    # wrong: sitting 65-85 (as lying, then walking) and standing 85-95 (as walking) = 30 of 120 s
    pred = [(0, 75, "LYING_IN_BED"), (75, 125, "WALKING")]
    y_true, y_pred = per_second(GT, pred)
    m = state_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(90 / 120)
    # sitting 75-85 predicted walking is wrong in/out of bed too; standing predicted walking is fine
    assert m["bed_status_accuracy"] == pytest.approx(110 / 120)


def test_unlabelled_time_in_prediction_counts_as_unknown():
    y_true, y_pred = per_second(GT, [(0, 50, "LYING_IN_BED")])
    assert y_pred.count("UNKNOWN") == 75


def test_event_matching_with_tolerance():
    gt = [("bed_exit", 100), ("bed_exit", 300), ("return_to_bed", 200)]
    pred = [("bed_exit", 103), ("bed_exit", 150), ("return_to_bed", 210)]
    r = match_events(gt, pred, tolerance=5, ignore=[])
    assert r["bed_exit"]["matched"] == 1 and r["bed_exit"]["false"] == [150] and r["bed_exit"]["missed"] == [300]
    assert r["bed_exit"]["precision"] == 0.5 and r["bed_exit"]["recall"] == 0.5
    assert r["return_to_bed"]["matched"] == 0          # 10 s off is outside the tolerance


def test_each_label_matches_once_closest_first():
    r = match_events([("bed_exit", 100)], [("bed_exit", 97), ("bed_exit", 101)], tolerance=5, ignore=[])
    assert r["bed_exit"]["pairs"] == [(100, 101)] and r["bed_exit"]["false"] == [97]


def test_events_in_ignored_time_are_not_scored():
    r = match_events([], [("return_to_bed", 3)], tolerance=5, ignore=[(0, 7)])
    assert r["return_to_bed"]["predicted"] == 0


def test_durations():
    y_true = ["LYING_IN_BED"] * 3 + ["WALKING"]
    y_pred = ["LYING_IN_BED"] * 2 + ["WALKING"] * 2
    d = {s: (g, p) for s, g, p in durations(y_true, y_pred, 1.0)}
    assert d["LYING_IN_BED"] == (3, 2) and d["WALKING"] == (1, 2)
