"""CVAT-style Wall UX — non-geometry regression checks (static + geometry reuse)."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from tests.test_wall_strip_section import strip_corners_px


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


class TestWallUxNoBrowserConfirm(unittest.TestCase):
    def test_confirm_draw_strip_has_no_confirm_dialog(self):
        # Extract _confirmDrawStrip body
        m = re.search(
            r"async function _confirmDrawStrip\(\) \{([\s\S]*?)\nasync function _confirmDrawRect",
            INDEX,
        )
        self.assertIsNotNone(m, "_confirmDrawStrip not found")
        body = m.group(1)
        # Ignore comments
        body_code = re.sub(r"//.*?$", "", body, flags=re.M)
        self.assertNotIn("confirm(", body_code)
        self.assertIn("points", body_code)
        self.assertIn("stripCorners", body_code)
        self.assertIn("/api/section", body_code)


class TestWallToolLabels(unittest.TestCase):
    def test_draw_wall_idle_label(self):
        self.assertIn("_drawToolIdleLabel", INDEX)
        self.assertIn("Draw Wall", INDEX)
        self.assertIn("Click/drag along the wall", INDEX)

    def test_start_end_preview_labels(self):
        self.assertIn("● START", INDEX)
        self.assertIn("● END", INDEX)
        self.assertIn("● CONTINUE", INDEX)


class TestGeometryPipelineUntouched(unittest.TestCase):
    def test_strip_corners_formula_present(self):
        self.assertIn("function stripCorners(p0, p1, thicknessPx)", INDEX)
        # Still used by preview + confirm
        self.assertGreaterEqual(INDEX.count("stripCorners("), 3)

    def test_snap_constants_unchanged(self):
        self.assertIn("WALL_STRIP_HV_TOL_DEG = 9", INDEX)
        self.assertIn("WALL_STRIP_EP_TOL_CAP_PX = 12", INDEX)

    def test_wrap_to_image_present(self):
        self.assertIn("function _wrapToImage", INDEX)
        self.assertIn("function _pzGet", INDEX)

    def test_door_rect_confirm_still_uses_bbox(self):
        m = re.search(
            r"async function _confirmDrawRect\(\) \{([\s\S]*?)\n// ═+",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertIn("bbox:", body)
        self.assertIn("confirm(", body)  # Door/Room still confirm


class TestEscShortcut(unittest.TestCase):
    def test_escape_handler_present(self):
        self.assertIn("Escape", INDEX)
        self.assertIn("_cancelIncompleteDraw", INDEX)


class TestStripStillFourPoints(unittest.TestCase):
    def test_preview_geometry_still_four(self):
        pts = strip_corners_px((10, 10), (110, 10), 8)
        self.assertEqual(len(pts), 4)


if __name__ == "__main__":
    unittest.main()
