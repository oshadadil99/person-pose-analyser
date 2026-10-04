import cv2
import numpy as np
import pytest

from src.camera import find_camera_segments
from src.config import load_config
from src.scene import Scene, SceneObject, SceneTimeline, load_scene, load_scenes, save_scene, save_scenes


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def room(shift_x=0):
    """A textured fake room, shifted sideways by shift_x pixels."""
    rng = np.random.default_rng(0)
    img = (rng.random((360, 640)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.roll(img, shift_x, axis=1)


def make_video(path, parts, fps=10):
    """parts: list of (seconds, shift_x)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 360))
    for seconds, shift in parts:
        frame = room(shift)
        for _ in range(int(seconds * fps)):
            w.write(frame)
    w.release()
    return str(path)


def test_still_camera_is_one_segment(cfg, tmp_path):
    segs = find_camera_segments(make_video(tmp_path / "v.avi", [(60, 0)]), cfg)
    assert len(segs) == 1


def test_camera_move_starts_a_new_segment(cfg, tmp_path):
    segs = find_camera_segments(make_video(tmp_path / "v.avi", [(40, 0), (40, 45)]), cfg)
    assert len(segs) == 2
    assert segs[1][0] == pytest.approx(40, abs=2.5)


def test_small_wobble_is_not_a_move(cfg, tmp_path):
    segs = find_camera_segments(make_video(tmp_path / "v.avi", [(40, 0), (40, 5)]), cfg)
    assert len(segs) == 1


def test_detection_can_be_switched_off(cfg, tmp_path):
    cfg["camera"]["detect_moves"] = False
    assert len(find_camera_segments(make_video(tmp_path / "v.avi", [(40, 0), (40, 45)]), cfg)) == 1


def scene(x):
    return Scene(640, 360, SceneObject("bed", "bed", [[x, 100], [x + 200, 100], [x + 200, 250]], 0.9, 1.0), [])


def test_scene_timeline_lookup_and_file_round_trip(tmp_path):
    tl = SceneTimeline([(0.0, 40.0, scene(10)), (40.0, 80.0, scene(55))])
    assert tl.at(10).bed.polygon[0][0] == 10
    assert tl.at(50).bed.polygon[0][0] == 55
    assert tl.at(999).bed.polygon[0][0] == 55          # past the end: last position
    save_scenes(tmp_path / "scene.json", tl)
    assert load_scenes(tmp_path / "scene.json") == tl
    assert load_scene(tmp_path / "scene.json") == tl.at(0)   # tools that need one scene get the first


def test_old_single_scene_files_still_load(tmp_path):
    save_scene(tmp_path / "scene.json", scene(10))
    tl = load_scenes(tmp_path / "scene.json")
    assert len(tl.segments) == 1 and tl.at(123).bed.polygon[0][0] == 10
