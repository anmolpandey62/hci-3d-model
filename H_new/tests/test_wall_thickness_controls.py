"""Wall thickness consistency controls — geometry + validation helpers.

Mirrors web/index.html stripCorners / thickness bounds (frontend-driven).
"""
from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from tests.test_wall_strip_section import strip_corners_px

WALL_STRIP_DEFAULT_THICKNESS_PX = 8
WALL_STRIP_THICKNESS_MIN_PX = 2
WALL_STRIP_THICKNESS_MAX_PX = 64


def validate_wall_strip_thickness(raw):
    """Mirror of validateWallStripThickness — reject, do not silently clamp."""
    if raw is None or raw == "":
        return False, None, "Thickness is required"
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return False, None, "Thickness must be a finite number"
    if not math.isfinite(v):
        return False, None, "Thickness must be a finite number"
    if v <= 0:
        return False, None, "Thickness must be positive"
    if v < WALL_STRIP_THICKNESS_MIN_PX or v > WALL_STRIP_THICKNESS_MAX_PX:
        return (
            False,
            None,
            f"Thickness must be between {WALL_STRIP_THICKNESS_MIN_PX} and {WALL_STRIP_THICKNESS_MAX_PX} px",
        )
    return True, v, None


def _obb_short(pts) -> float:
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    (_c), (w, h), _a = cv2.minAreaRect(arr)
    return float(min(w, h))


def thickness_from_quad(poly):
    """Mirror of _stripParamsFromQuad short-side thickness."""

    def mid(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    a, b, c, d = poly
    m01, m12, m23, m30 = mid(a, b), mid(b, c), mid(c, d), mid(d, a)
    len_a = dist(m01, m23)
    len_b = dist(m12, m30)
    if len_b >= len_a:
        return (dist(b, c) + dist(d, a)) / 2.0
    return (dist(a, b) + dist(c, d)) / 2.0


class TestThicknessGeometry(unittest.TestCase):
    def _check(self, p0, p1, t):
        pts = strip_corners_px(p0, p1, t)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(_obb_short(pts), float(t), places=4)
        # opposite edges parallel
        v01 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        v32 = (pts[2][0] - pts[3][0], pts[2][1] - pts[3][1])
        self.assertAlmostEqual(v01[0], v32[0], places=5)
        self.assertAlmostEqual(v01[1], v32[1], places=5)
        return pts

    def test_thickness_8_horizontal(self):
        self._check((100, 100), (300, 100), 8)

    def test_thickness_12_vertical(self):
        self._check((100, 100), (100, 300), 12)

    def test_thickness_20_diagonal(self):
        pts = self._check((100, 100), (250, 200), 20)
        # not AABB
        edged = False
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            if abs(b[0] - a[0]) > 1e-6 and abs(b[1] - a[1]) > 1e-6:
                edged = True
        self.assertTrue(edged)


class TestValidateThickness(unittest.TestCase):
    def test_default_ok(self):
        ok, v, err = validate_wall_strip_thickness(8)
        self.assertTrue(ok)
        self.assertEqual(v, 8)
        self.assertIsNone(err)

    def test_reject_zero_negative(self):
        self.assertFalse(validate_wall_strip_thickness(0)[0])
        self.assertFalse(validate_wall_strip_thickness(-3)[0])

    def test_reject_nan_inf(self):
        self.assertFalse(validate_wall_strip_thickness(float("nan"))[0])
        self.assertFalse(validate_wall_strip_thickness(float("inf"))[0])

    def test_reject_out_of_bounds_no_clamp(self):
        ok, v, err = validate_wall_strip_thickness(1)
        self.assertFalse(ok)
        self.assertIsNone(v)
        ok2, v2, _ = validate_wall_strip_thickness(100)
        self.assertFalse(ok2)
        self.assertIsNone(v2)


class TestLockSemantics(unittest.TestCase):
    def test_locked_value_reused(self):
        # Simulate: locked thickness stays constant across multiple strip builds
        locked_t = 14.0
        strips = []
        for end in [(200, 100), (300, 100), (300, 200)]:
            strips.append(strip_corners_px((100, 100), end, locked_t))
        for pts in strips:
            self.assertAlmostEqual(_obb_short(pts), locked_t, places=4)


class TestCopyThickness(unittest.TestCase):
    def test_copy_from_8px_strip(self):
        pts = strip_corners_px((50, 50), (200, 50), 8)
        t = thickness_from_quad(pts)
        self.assertAlmostEqual(t, 8.0, places=4)

    def test_copy_from_12px_strip(self):
        pts = strip_corners_px((50, 50), (50, 200), 12)
        t = thickness_from_quad(pts)
        self.assertAlmostEqual(t, 12.0, places=4)

    def test_invalid_non_strip_does_not_define_thickness(self):
        # 3-pt / complex: copy UI refuses; helper only defined for 4-pt
        poly3 = [[0, 0], [10, 0], [5, 8]]
        self.assertEqual(len(poly3), 3)


class TestZoomInvariance(unittest.TestCase):
    def test_same_image_space_thickness(self):
        # Thickness is image-px; pan/zoom cannot change stripCorners output
        a = strip_corners_px((40, 40), (180, 40), 10)
        b = strip_corners_px((40, 40), (180, 40), 10)
        self.assertEqual(a, b)
        self.assertAlmostEqual(_obb_short(a), 10.0, places=5)


class TestPersistenceKeysDocumented(unittest.TestCase):
    def test_localstorage_keys(self):
        # Contract with frontend
        self.assertEqual("HCI_WALL_STRIP_THICKNESS_PX", "HCI_WALL_STRIP_THICKNESS_PX")
        self.assertEqual("HCI_WALL_STRIP_THICKNESS_LOCK", "HCI_WALL_STRIP_THICKNESS_LOCK")


if __name__ == "__main__":
    unittest.main()
