import pytest

from src.config import load_config
from src.features import FrameFeatures
from src.frame_rules import classify_frame
from src.states import State


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def feats(**kw):
    """A clear, visible, upright person standing far from the bed, then overridden by kw."""
    base = dict(frame_idx=0, t_sec=0.0, visible=True, num_persons=1, torso_angle_deg=5.0,
                thigh_angle_deg=5.0, bbox_aspect=0.4, kp_conf_mean=0.9, hip_in_bed=False,
                bed_overlap=0.0, bed_edge_dist=-2.0, on_seat=False, hip_speed=0.0,
                ref_source="hips", box_h=200.0, box_truncated=False)
    base.update(kw)
    return FrameFeatures(**base)


def state(cfg, **kw):
    return classify_frame(feats(**kw), cfg).state


def test_not_visible_is_unknown(cfg):
    r = classify_frame(feats(visible=False), cfg)
    assert r.state == State.UNKNOWN and r.confidence == 0.0


def test_unreadable_pose(cfg):
    assert state(cfg, kp_conf_mean=0.1) == State.UNKNOWN
    # blanket: pose unreadable but a wide box lying on the bed
    r = classify_frame(feats(kp_conf_mean=0.1, bbox_aspect=2.0, hip_in_bed=True, bed_edge_dist=0.5), cfg)
    assert r.state == State.LYING_IN_BED and r.confidence == cfg["frame_rules"]["low_kp_lying_conf"]


def test_lying_in_and_outside_bed(cfg):
    assert state(cfg, torso_angle_deg=85, hip_in_bed=True, bed_edge_dist=0.5) == State.LYING_IN_BED
    assert state(cfg, torso_angle_deg=85) == State.LYING_OUTSIDE_BED


def test_lying_from_box_shape_when_torso_hidden(cfg):
    assert state(cfg, torso_angle_deg=None, bbox_aspect=2.0, hip_in_bed=True, bed_edge_dist=0.5) == State.LYING_IN_BED


def test_hips_hidden_uses_bed_overlap(cfg):
    assert state(cfg, torso_angle_deg=85, hip_in_bed=None, bed_overlap=0.8) == State.LYING_IN_BED
    assert state(cfg, torso_angle_deg=85, hip_in_bed=None, bed_overlap=0.1) == State.LYING_OUTSIDE_BED


def test_clearly_bent_legs_are_sitting(cfg):
    assert state(cfg, thigh_angle_deg=85, hip_in_bed=True, bed_edge_dist=0.3) == State.SITTING_ON_BED
    assert state(cfg, thigh_angle_deg=85) == State.SITTING_OUTSIDE_BED
    r = classify_frame(feats(thigh_angle_deg=85, on_seat=True, seat_label="chair"), cfg)
    assert r.state == State.SITTING_OUTSIDE_BED and "chair" in r.reason


def test_unsure_thigh_uses_context(cfg):
    # facing the camera: thigh looks ~30 deg
    assert state(cfg, thigh_angle_deg=30, hip_in_bed=True, bed_edge_dist=0.3) == State.SITTING_ON_BED
    assert state(cfg, thigh_angle_deg=30, on_seat=True, seat_label="chair") == State.SITTING_OUTSIDE_BED
    assert state(cfg, thigh_angle_deg=30) == State.STANDING
    assert state(cfg, thigh_angle_deg=30, hip_speed=0.8) == State.WALKING


def test_unsure_band_has_low_confidence(cfg):
    unsure = classify_frame(feats(thigh_angle_deg=30, hip_in_bed=True, bed_edge_dist=0.3), cfg)
    clear = classify_frame(feats(thigh_angle_deg=85, hip_in_bed=True, bed_edge_dist=0.3), cfg)
    assert unsure.confidence < 0.5 < clear.confidence


def test_standing_and_walking(cfg):
    assert state(cfg, hip_speed=0.05) == State.STANDING
    assert state(cfg, hip_speed=0.6) == State.WALKING
    assert state(cfg, hip_speed=None) == State.STANDING    # first frame, no speed yet


def test_legs_hidden(cfg):
    # legs cut off by the bottom of the image, nothing to sit on: standing (or walking if moving)
    assert state(cfg, thigh_angle_deg=None, box_truncated=True) == State.STANDING
    assert state(cfg, thigh_angle_deg=None, box_truncated=True, hip_speed=0.8) == State.WALKING
    # ...even if the hips overlap the bed in 2D: close to the camera means in front of the bed
    assert state(cfg, thigh_angle_deg=None, box_truncated=True, hip_in_bed=True, bed_edge_dist=0.2) == State.STANDING
    # over a seat it could be sitting or standing in front of it: can't tell
    assert state(cfg, thigh_angle_deg=None, box_truncated=True, on_seat=True, seat_label="chair") == State.UNKNOWN
    # legs under a blanket, sitting up in bed
    assert state(cfg, thigh_angle_deg=None, hip_in_bed=True, bed_edge_dist=0.4) == State.SITTING_ON_BED
    assert state(cfg, thigh_angle_deg=None) == State.STANDING


def test_confidence_grows_with_margin(cfg):
    borderline = classify_frame(feats(torso_angle_deg=62, hip_in_bed=True, bed_edge_dist=0.5), cfg)
    clear = classify_frame(feats(torso_angle_deg=90, hip_in_bed=True, bed_edge_dist=0.5), cfg)
    assert borderline.state == clear.state == State.LYING_IN_BED
    assert borderline.confidence < clear.confidence


def test_bed_edge_lowers_confidence(cfg):
    middle = classify_frame(feats(torso_angle_deg=85, hip_in_bed=True, bed_edge_dist=0.5), cfg)
    edge = classify_frame(feats(torso_angle_deg=85, hip_in_bed=True, bed_edge_dist=0.05), cfg)
    assert edge.confidence < middle.confidence
