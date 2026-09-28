"""Phase 2 — wall strip /api/section points path (unit tests, no server required for geometry)."""
from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from logic.yolo_inference import contour_to_yolo_seg


def strip_corners_px(p0, p1, thickness_px: float):
    """Mirror of frontend stripCorners() for verification."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    nx, ny = -dy / length, dx / length
    half = thickness_px / 2.0
    return [
        [x0 + nx * half, y0 + ny * half],
        [x1 + nx * half, y1 + ny * half],
        [x1 - nx * half, y1 - ny * half],
        [x0 - nx * half, y0 - ny * half],
    ]


def _obb_short(pts) -> float:
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    (_c), (w, h), _a = cv2.minAreaRect(arr)
    return float(min(w, h))


class TestStripGeometry(unittest.TestCase):
    def test_horizontal(self):
        pts = strip_corners_px((100, 100), (300, 100), 8)
        self.assertIsNotNone(pts)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(_obb_short(pts), 8.0, places=5)

    def test_vertical(self):
        pts = strip_corners_px((100, 100), (100, 300), 8)
        self.assertIsNotNone(pts)
        self.assertAlmostEqual(_obb_short(pts), 8.0, places=5)

    def test_diagonal_not_aabb(self):
        pts = strip_corners_px((100, 100), (250, 200), 8)
        self.assertIsNotNone(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # AABB would have corners only at min/max combos; oriented strip has angled edges
        # Check opposite edges are parallel (vectors equal)
        v01 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        v32 = (pts[2][0] - pts[3][0], pts[2][1] - pts[3][1])
        self.assertAlmostEqual(v01[0], v32[0], places=5)
        self.assertAlmostEqual(v01[1], v32[1], places=5)
        self.assertAlmostEqual(_obb_short(pts), 8.0, places=4)
        # Not an axis-aligned box: at least one edge has both dx and dy nonzero
        edged = False
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            if abs(b[0] - a[0]) > 1e-6 and abs(b[1] - a[1]) > 1e-6:
                edged = True
        self.assertTrue(edged)

    def test_zero_length(self):
        self.assertIsNone(strip_corners_px((100, 100), (100, 100), 8))


class TestSectionContourHelpers(unittest.TestCase):
    def test_points_preserved_not_aabb(self):
        from web.server import _contour_from_points, _contour_from_bbox

        pts = strip_corners_px((100, 100), (250, 200), 8)
        cnt, err = _contour_from_points(pts, 800, 600)
        self.assertIsNone(err)
        out = cnt.reshape(-1, 2).tolist()
        self.assertEqual(len(out), 4)
        # Must not equal axis-aligned bbox corners of the same points
        x, y, w, h = cv2.boundingRect(cnt)
        aabb = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        self.assertNotEqual(out, aabb)

    def test_bbox_legacy(self):
        from web.server import _contour_from_bbox

        cnt = _contour_from_bbox([10, 20, 40, 30], 800, 600)
        out = cnt.reshape(-1, 2).tolist()
        self.assertEqual(out, [[10, 20], [50, 20], [50, 50], [10, 50]])

    def test_points_zero_area_rejected(self):
        from web.server import _contour_from_points

        cnt, err = _contour_from_points([[0, 0], [0, 0], [0, 0]], 100, 100)
        self.assertIsNone(cnt)
        self.assertIn("area", err.lower())

    def test_yolo_seg_four_points(self):
        pts = strip_corners_px((100, 100), (300, 100), 8)
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        line = contour_to_yolo_seg(cnt, 800, 600, 3)
        parts = line.split()
        self.assertEqual(parts[0], "3")
        # 1 class + 4 points * 2 coords
        self.assertEqual(len(parts), 1 + 8)
        coords = [float(x) for x in parts[1:]]
        self.assertTrue(all(0.0 <= c <= 1.0 for c in coords))


if __name__ == "__main__":
    unittest.main()
