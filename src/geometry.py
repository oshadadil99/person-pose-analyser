"""Small geometry helpers for polygons (bed, seats) and boxes (people).

Polygons are lists of [x, y] points in original video pixels. OpenCV already
has point-in-polygon and distance-to-edge, so no extra geometry library is
needed. Box/polygon overlap is done by painting the polygon into a small mask
the size of the box and counting the pixels.
"""

import math

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = list[list[float]]
Box = list[float]  # x1, y1, x2, y2


def _as_contour(poly: Polygon) -> np.ndarray:
    return np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(pt: Point, poly: Polygon) -> bool:
    """True if the point is inside or on the edge."""
    return cv2.pointPolygonTest(_as_contour(poly), (float(pt[0]), float(pt[1])), False) >= 0


def signed_distance(pt: Point, poly: Polygon) -> float:
    """Distance in pixels to the polygon edge: positive inside, negative outside."""
    return float(cv2.pointPolygonTest(_as_contour(poly), (float(pt[0]), float(pt[1])), True))


def box_overlap_ratio(box: Box, poly: Polygon) -> float:
    """Fraction of the box area that lies inside the polygon (0 to 1)."""
    x1, y1 = math.floor(box[0]), math.floor(box[1])
    w, h = math.ceil(box[2]) - x1, math.ceil(box[3]) - y1
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    shifted = (np.asarray(poly, dtype=np.float32) - [x1, y1]).round().astype(np.int32)
    cv2.fillPoly(mask, [shifted], 1)
    return float(mask.mean())


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_bbox(poly: Polygon) -> Box:
    arr = np.asarray(poly, dtype=np.float32)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def box_height(box: Box) -> float:
    return box[3] - box[1]


def long_axis_angle(poly: Polygon) -> float:
    """Angle of the polygon's long axis against the vertical, 0-90 degrees.

    Uses the minimum-area rotated rectangle around the polygon. For a bed seen
    in perspective this is the direction a person lying in it would point."""
    rect = cv2.minAreaRect(np.asarray(poly, dtype=np.float32))
    corners = cv2.boxPoints(rect)
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    a, b = max(edges, key=lambda e: float(np.linalg.norm(e[1] - e[0])))
    return float(math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1]))))
