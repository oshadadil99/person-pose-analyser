import pytest

from src.config import load_config
from src.features import angle_from_vertical, extract_features, thigh_angle, torso_angle
from src.perception import FramePerception
from src.scene import Scene, SceneObject
from tests.pose_helpers import LYING, SITTING_SIDE, STANDING, make_person

BED = SceneObject("bed", "bed", [[50, 150], [400, 150], [400, 260], [50, 260]], 0.9, 1.0)
CHAIR = SceneObject("seat", "chair", [[500, 120], [600, 120], [600, 250], [500, 250]], 0.8, 1.0)


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def scene(bed=BED, seats=(CHAIR,)):
    return Scene(1000, 500, bed, list(seats))


def test_angle_from_vertical():
    assert angle_from_vertical((0, 0), (0, 10)) == pytest.approx(0)
    assert angle_from_vertical((0, 0), (10, 0)) == pytest.approx(90)
    assert angle_from_vertical((0, 0), (10, 10)) == pytest.approx(45)
    assert angle_from_vertical((0, 10), (0, 0)) == pytest.approx(0)   # direction doesn't matter


def test_torso_and_thigh_angles():
    assert torso_angle(make_person(**STANDING), 0.3) == pytest.approx(0)
    assert thigh_angle(make_person(**STANDING), 0.3) == pytest.approx(0)
    assert thigh_angle(make_person(**SITTING_SIDE), 0.3) == pytest.approx(90)
    assert torso_angle(make_person(**LYING), 0.3) > 80


def test_missing_keypoints_give_none_not_zero():
    p = make_person(shoulder=None, hip=(200, 170), knee=None)
    assert torso_angle(p, 0.3) is None
    assert thigh_angle(p, 0.3) is None


def test_not_visible_frame(cfg):
    frames = [FramePerception(0, 0.0, [])]
    f = extract_features(frames, [None], scene(), cfg)[0]
    assert not f.visible and f.num_persons == 0
    assert f.torso_angle_deg is None and f.hip_in_bed is None


def test_bed_and_seat_location(cfg):
    in_bed = make_person(**LYING)                                        # hips at (170, 205), inside the bed
    on_chair = make_person(shoulder=(550, 100), hip=(550, 170), knee=(610, 170))
    frames = [FramePerception(0, 0.0, [in_bed]), FramePerception(1, 0.2, [on_chair])]
    f1, f2 = extract_features(frames, [in_bed, on_chair], scene(), cfg)
    assert f1.hip_in_bed and f1.bed_edge_dist > 0 and not f1.on_seat
    assert not f2.hip_in_bed and f2.on_seat and f2.seat_label == "chair"


def test_no_bed_means_unknown_location(cfg):
    p = make_person(**STANDING)
    f = extract_features([FramePerception(0, 0.0, [p])], [p], None, cfg)[0]
    assert f.hip_in_bed is None and f.on_seat is None


def test_hip_speed_in_box_heights_per_second(cfg):
    # hips move 50 px per 0.2 s frame; box is 240 px tall -> 250 px/s = ~1.04 heights/s
    people = [make_person(shoulder=(600 + 50 * i, 100), hip=(600 + 50 * i, 170),
                          knee=(600 + 50 * i, 240), ankle=(600 + 50 * i, 300)) for i in range(8)]
    frames = [FramePerception(i, i * 0.2, [p]) for i, p in enumerate(people)]
    feats = extract_features(frames, people, scene(), cfg)
    assert feats[0].hip_speed is None              # nothing to compare with yet
    assert feats[6].hip_speed == pytest.approx(250 / 240, rel=0.01)


def test_speed_resets_after_patient_disappears(cfg):
    p1 = make_person(**STANDING)
    p2 = make_person(shoulder=(800, 100), hip=(800, 170), knee=(800, 240), ankle=(800, 300))
    frames = [FramePerception(0, 0.0, [p1]), FramePerception(1, 0.2, []), FramePerception(2, 0.4, [p2])]
    feats = extract_features(frames, [p1, None, p2], scene(), cfg)
    assert feats[2].hip_speed is None              # no fake "jump" across the gap


def test_box_truncated_at_frame_bottom(cfg):
    cut = make_person(**STANDING, box=[150, 60, 250, 499])
    whole = make_person(**STANDING)
    frames = [FramePerception(0, 0.0, [cut]), FramePerception(1, 0.2, [whole])]
    f1, f2 = extract_features(frames, [cut, whole], scene(), cfg)
    assert f1.box_truncated and not f2.box_truncated
