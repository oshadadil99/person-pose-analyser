import cv2
import numpy as np
import pytest

from src.video_io import get_video_info, resize_to_width, sample_frames


def make_video(path, fps, n_frames, size=(320, 240)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n_frames):
        writer.write(np.full((h, w, 3), i % 255, np.uint8))
    writer.release()
    return path


def test_video_info(tmp_path):
    info = get_video_info(make_video(tmp_path / "v.avi", 30, 90))
    assert info.frame_count == 90
    assert info.duration_sec == pytest.approx(3.0)
    assert (info.width, info.height) == (320, 240)


def test_sampling_30fps_to_5fps(tmp_path):
    path = make_video(tmp_path / "v.avi", 30, 90)
    out = list(sample_frames(path, fps=5, max_width=1000))
    assert [f[0] for f in out] == list(range(0, 90, 6))
    assert [f[1] for f in out] == pytest.approx([i * 0.2 for i in range(15)])


def test_sampling_non_integer_step(tmp_path):
    # 24 / 5 = 4.8 frames per sample: picks must never drift more than one frame
    path = make_video(tmp_path / "v.avi", 24, 240)
    out = list(sample_frames(path, fps=5, max_width=1000))
    assert len(out) == 50
    for k, (_, t, _, _) in enumerate(out):
        assert abs(t - k * 0.2) < 1 / 24 + 1e-9


def test_resize_only_shrinks():
    big = np.zeros((1080, 1920, 3), np.uint8)
    small, scale = resize_to_width(big, 960)
    assert small.shape[:2] == (540, 960)
    assert scale == pytest.approx(0.5)

    tiny = np.zeros((240, 320, 3), np.uint8)
    same, scale = resize_to_width(tiny, 960)
    assert same.shape == tiny.shape and scale == 1.0


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "nope.mp4")
