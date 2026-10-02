import pytest

from src.config import load_config
from src.patient import patient_per_frame, reference_point, select_patient_ids, summarize_tracks
from src.perception import FramePerception, Person

BED = [[100, 100], [400, 100], [400, 300], [100, 300]]


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


def person(track_id, cx, cy, hips_visible=True, h=200):
    """A person whose hips are at (cx, cy) and whose box is h pixels tall."""
    kps = [[cx, cy - 80, 0.9]] * 17
    hip_conf = 0.9 if hips_visible else 0.05
    kps = [list(k) for k in kps]
    kps[11] = [cx - 10, cy, hip_conf]
    kps[12] = [cx + 10, cy, hip_conf]
    return Person(track_id, [cx - 40, cy - h / 2, cx + 40, cy + h / 2], 0.9, kps)


def frames_from(spec, fps=5):
    """spec: list of lists of persons, one list per frame."""
    return [FramePerception(i, i / fps, ps) for i, ps in enumerate(spec)]


def test_reference_point_uses_hips_then_box_centre(cfg):
    p = person(1, 200, 150)
    assert reference_point(p, 0.3) == (200, 150, "hips")
    hidden = person(1, 200, 150, hips_visible=False)
    x, y, src = reference_point(hidden, 0.3)
    assert src == "box_centre"


def test_patient_is_the_track_near_the_bed(cfg):
    # id 1 lies in bed the whole time, id 2 (caregiver) stands far away
    frames = frames_from([[person(1, 250, 200), person(2, 700, 200)] for _ in range(50)])
    tracks = summarize_tracks(frames, BED, cfg)
    assert select_patient_ids(tracks, cfg) == [1]


def test_short_tracking_break_is_relinked(cfg):
    # patient is id 1, lost for 1 s, comes back as id 7 at almost the same spot;
    # a caregiver (id 2) is in the room the whole time so the lone-returner rule can't apply
    spec = [[person(1, 250, 200), person(2, 700, 200)] for _ in range(50)]
    spec += [[person(2, 700, 200)] for _ in range(5)]
    spec += [[person(7, 260, 205), person(2, 700, 200)] for _ in range(50)]
    tracks = summarize_tracks(frames_from(spec), BED, cfg)
    assert select_patient_ids(tracks, cfg) == [1, 7]


def test_far_new_track_with_others_present_is_not_relinked(cfg):
    spec = [[person(1, 250, 200), person(2, 700, 200)] for _ in range(50)]
    spec += [[person(9, 900, 600), person(2, 700, 200)] for _ in range(20)]
    tracks = summarize_tracks(frames_from(spec), BED, cfg)
    assert select_patient_ids(tracks, cfg) == [1]


def test_lone_returner_after_leaving_view(cfg):
    # patient in bed (id 1), walks out, room empty for 60 s, comes back alone as id 5 at the door
    spec = [[person(1, 250, 200)] for _ in range(100)]
    spec += [[] for _ in range(300)]
    spec += [[person(5, 900, 600)] for _ in range(50)]
    tracks = summarize_tracks(frames_from(spec), BED, cfg)
    assert select_patient_ids(tracks, cfg) == [1, 5]

    cfg["patient"]["relink_lone_returner"] = False
    assert select_patient_ids(tracks, cfg) == [1]


def test_backward_relink(cfg):
    # an early short track (id 3) ends just before the main in-bed track (id 4) starts, same place
    spec = [[person(3, 250, 200), person(2, 700, 200)] for _ in range(10)]
    spec += [[person(4, 255, 200), person(2, 700, 200)] for _ in range(80)]
    tracks = summarize_tracks(frames_from(spec), BED, cfg)
    assert select_patient_ids(tracks, cfg) == [3, 4]


def test_patient_per_frame(cfg):
    spec = [[person(1, 250, 200), person(2, 700, 200)], [person(2, 700, 200)], []]
    out = patient_per_frame(frames_from(spec), [1])
    assert out[0].track_id == 1
    assert out[1] is None and out[2] is None


def test_no_bed_falls_back_to_longest_track(cfg):
    spec = [[person(1, 250, 200)] for _ in range(10)] + [[person(2, 700, 200)] for _ in range(30)]
    tracks = summarize_tracks(frames_from(spec), None, cfg)
    assert select_patient_ids(tracks, cfg)[-1] == 2
