import numpy as np

from src.perception import FramePerception, Person, draw_person, load_perception, save_perception


def fake_person(track_id=1):
    kps = [[100.0 + i, 200.0 + i, 0.9] for i in range(17)]
    return Person(track_id=track_id, box=[50.0, 60.0, 150.0, 400.0], conf=0.88, keypoints=kps)


def test_save_and_load_roundtrip(tmp_path):
    meta = {"video": "x.mp4", "native_fps": 30.0, "sample_fps": 5, "duration_sec": 0.4}
    frames = [
        FramePerception(0, 0.0, [fake_person(1), fake_person(2)]),
        FramePerception(6, 0.2, []),                # no one detected
        FramePerception(12, 0.4, [fake_person(-1)]),  # tracker hasn't assigned an ID
    ]
    path = tmp_path / "perception.jsonl"
    save_perception(path, meta, frames)
    meta2, frames2 = load_perception(path)

    assert meta2 == meta
    assert frames2 == frames


def test_draw_person_runs():
    img = np.zeros((480, 640, 3), np.uint8)
    draw_person(img, fake_person(), (0, 255, 0), kp_min_conf=0.3, label="id 1")
    assert img.any()
