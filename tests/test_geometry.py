import pytest

from src.geometry import box_iou, box_overlap_ratio, long_axis_angle, point_in_polygon, signed_distance

# A 100 x 100 square "bed" from (100, 100) to (200, 200)
BED = [[100, 100], [200, 100], [200, 200], [100, 200]]


def test_point_in_polygon():
    assert point_in_polygon((150, 150), BED)
    assert point_in_polygon((100, 150), BED)          # on the edge counts as inside
    assert not point_in_polygon((50, 150), BED)


def test_signed_distance():
    assert signed_distance((150, 150), BED) == pytest.approx(50)    # centre, 50 px from every edge
    assert signed_distance((110, 150), BED) == pytest.approx(10)
    assert signed_distance((80, 150), BED) == pytest.approx(-20)    # outside is negative


def test_box_overlap_ratio():
    assert box_overlap_ratio([120, 120, 180, 180], BED) == pytest.approx(1.0)
    assert box_overlap_ratio([0, 0, 50, 50], BED) == pytest.approx(0.0)
    assert box_overlap_ratio([50, 100, 150, 200], BED) == pytest.approx(0.5, abs=0.02)


def test_box_overlap_degenerate_box():
    assert box_overlap_ratio([10, 10, 10, 50], BED) == 0.0


def test_long_axis_angle():
    wide = [[0, 0], [300, 0], [300, 100], [0, 100]]
    tall = [[0, 0], [100, 0], [100, 300], [0, 300]]
    assert long_axis_angle(wide) == pytest.approx(90, abs=1)
    assert long_axis_angle(tall) == pytest.approx(0, abs=1)
    diagonal = [[0, 0], [20, -20], [220, 180], [200, 200]]    # long side at 45 deg
    assert long_axis_angle(diagonal) == pytest.approx(45, abs=2)


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
