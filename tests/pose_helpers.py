"""Builds fake YOLO people from a few joint positions, for tests."""

from src.perception import KP, Person


def make_person(shoulder, hip, knee=None, ankle=None, box=None, conf=0.9, track_id=1):
    """Both sides get the same point; None means "not detected" (confidence 0)."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]

    def put(name, pt):
        if pt is not None:
            for side in ("left", "right"):
                kps[KP[f"{side}_{name}"]] = [float(pt[0]), float(pt[1]), conf]

    put("shoulder", shoulder)
    put("hip", hip)
    put("knee", knee)
    put("ankle", ankle)
    if box is None:
        pts = [p for p in (shoulder, hip, knee, ankle) if p is not None]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = [min(xs) - 20, min(ys) - 30, max(xs) + 20, max(ys) + 10]
    return Person(track_id, [float(v) for v in box], 0.9, kps)


# Body poses, all about 200 px tall
STANDING = dict(shoulder=(200, 100), hip=(200, 170), knee=(200, 240), ankle=(200, 300))
SITTING_SIDE = dict(shoulder=(200, 100), hip=(200, 170), knee=(260, 170), ankle=(260, 240))
SITTING_FACING_CAMERA = dict(shoulder=(200, 100), hip=(200, 170), knee=(215, 196), ankle=(215, 260))
LYING = dict(shoulder=(100, 200), hip=(170, 205), knee=(240, 205), ankle=(300, 205))
