import numpy as np
import pytest

from src.config import load_config
from src.scene import (Detection, Scene, SceneObject, cluster_to_polygon, load_scene,
                       merge_detections, save_scene)

H, W = 300, 400


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def det(role, label, box, conf=0.8):
    mask = np.zeros((H, W), bool)
    x1, y1, x2, y2 = map(int, box)
    mask[y1:y2, x1:x2] = True
    return Detection(role, label, conf, list(map(float, box)), mask)


BED_BOX = [50, 100, 250, 250]
CHAIR_BOX = [300, 150, 360, 260]


def test_merge_groups_same_object_across_frames(cfg):
    frames = [[det("bed", "bed", BED_BOX), det("seat", "chair", CHAIR_BOX)] for _ in range(10)]
    bed, seats = merge_detections(frames, cfg)
    assert bed is not None and len(bed.masks) == 10
    assert len(seats) == 1 and len(seats[0].masks) == 10


def test_two_overlapping_chairs_stay_separate(cfg):
    # side-by-side chairs overlapping more than match_iou, in every frame
    a, b = [300, 150, 360, 260], [320, 150, 380, 260]
    frames = [[det("seat", "chair", a), det("seat", "chair", b, conf=0.7)] for _ in range(10)]
    _, seats = merge_detections(frames, cfg)
    assert len(seats) == 2
    assert all(len(s.masks) == 10 for s in seats)


def test_rare_seat_is_dropped_but_rare_bed_is_kept(cfg):
    # bed seen in only 2 of 10 frames (person on it), chair seen once (false detection)
    frames = [[] for _ in range(10)]
    frames[0] = [det("bed", "bed", BED_BOX), det("seat", "chair", CHAIR_BOX)]
    frames[5] = [det("bed", "bed", BED_BOX)]
    bed, seats = merge_detections(frames, cfg)
    assert bed is not None
    assert seats == []


def test_couch_on_top_of_bed_is_dropped(cfg):
    frames = [[det("bed", "bed", BED_BOX), det("seat", "couch", [55, 105, 245, 245])] for _ in range(10)]
    bed, seats = merge_detections(frames, cfg)
    assert bed is not None
    assert seats == []


def test_most_seen_bed_wins(cfg):
    other = [300, 20, 390, 80]
    frames = [[det("bed", "bed", BED_BOX)] for _ in range(8)] + [[det("bed", "bed", other)] for _ in range(3)]
    bed, _ = merge_detections(frames, cfg)
    assert len(bed.masks) == 8


def test_polygon_scaled_back_to_original_pixels(cfg):
    frames = [[det("bed", "bed", BED_BOX)] for _ in range(5)]
    bed, _ = merge_detections(frames, cfg)
    poly = np.array(cluster_to_polygon(bed, cfg, scale=0.5))
    # working-size box 50..250 x 100..250 becomes 100..500 x 200..500 at scale 0.5
    assert poly[:, 0].min() == pytest.approx(100, abs=4)
    assert poly[:, 0].max() == pytest.approx(500, abs=4)
    assert poly[:, 1].min() == pytest.approx(200, abs=4)
    assert poly[:, 1].max() == pytest.approx(500, abs=4)


def test_save_and_load(tmp_path):
    scene = Scene(1920, 1080,
                  SceneObject("bed", "bed", [[1, 2], [3, 4], [5, 6]], 0.9, 0.5),
                  [SceneObject("seat", "chair", [[7, 8], [9, 10], [11, 12]], 0.7, 0.8)])
    save_scene(tmp_path / "scene.json", scene)
    assert load_scene(tmp_path / "scene.json") == scene
