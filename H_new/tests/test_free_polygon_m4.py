"""M4 free N-click polygon — static UI wiring + YOLO class IDs via /api/section points."""
from __future__ import annotations

import base64
import re
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from config.classes import CLASS_IDS
from logic.yolo_inference import contour_to_yolo_seg


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


class TestFreePolyUiStatic(unittest.TestCase):
    def test_free_poly_symbols_present(self):
        for sym in (
            "FREE_POLY_CLASSES",
            "_isFreePolyDraw",
            "_freePolyPoints",
            "_addFreePolyPointAtWrap",
            "_confirmFreePolygon",
            "_renderFreePoly",
            "_popFreePolyPoint",
        ):
            self.assertIn(sym, INDEX, f"missing {sym}")

    def test_free_poly_classes_list(self):
        self.assertIn("'Door'", INDEX)
        self.assertIn("'Window'", INDEX)
        self.assertIn("'Furniture'", INDEX)
        self.assertIn("'ElectricAppliance'", INDEX)
        # FREE_POLY_CLASSES freeze list should include exactly these four for M4
        m = re.search(
            r"const FREE_POLY_CLASSES = Object\.freeze\(\[([^\]]+)\]\)",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        for name in ("Door", "Window", "Furniture", "ElectricAppliance"):
            self.assertIn(f"'{name}'", body)
        self.assertNotIn("'Wall'", body)
        self.assertNotIn("'Room'", body)

    def test_confirm_posts_points_to_section(self):
        m = re.search(
            r"async function _confirmFreePolygon\(\) \{([\s\S]*?)\n/\*\* Idle draw-mode:",
            INDEX,
        )
        self.assertIsNotNone(m, "_confirmFreePolygon not found")
        body = m.group(1)
        self.assertIn("/api/section", body)
        self.assertIn("points", body)
        self.assertNotIn("bbox:", body)

    def test_wall_strip_path_untouched_markers(self):
        self.assertIn("function stripCorners(", INDEX)
        self.assertIn("function _confirmDrawStrip(", INDEX)
        self.assertIn("function snapHV(", INDEX)
        self.assertIn("_isWallStripDraw", INDEX)
        # Wall create still uses stripCorners → points (regression guard)
        m = re.search(
            r"async function _confirmDrawStrip\(\) \{([\s\S]*?)\nasync function _confirmDrawRect",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = re.sub(r"//.*?$", "", m.group(1), flags=re.M)
        self.assertNotIn("confirm(", body)
        self.assertIn("stripCorners", body)
        self.assertIn("/api/section", body)


class TestFreePolyYoloClassIds(unittest.TestCase):
    def test_taxonomy_ids(self):
        self.assertEqual(CLASS_IDS["Furniture"], 11)
        self.assertEqual(CLASS_IDS["ElectricAppliance"], 14)
        self.assertEqual(CLASS_IDS["Door"], 2)
        self.assertEqual(CLASS_IDS["Window"], 1)

    def _mock_and_section(self, label: str, points: list):
        import os

        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"
        import web.server as srv

        basename = "m4_free_poly"
        tmp = Path(tempfile.mkdtemp())
        (tmp / "labels" / "train").mkdir(parents=True)
        lbl = tmp / "labels" / "train" / f"{basename}.txt"
        lbl.write_text("")
        old_ds = srv.DATASET_DIR
        srv.DATASET_DIR = tmp
        img_w, img_h = 400, 300
        img = np.zeros((img_h, img_w, 3), np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        b64 = base64.b64encode(buf.tobytes()).decode()
        srv._analysis[basename] = {
            "labelled": {},
            "label_lines": [],
            "n_labels": 0,
            "img_h": img_h,
            "img_w": img_w,
            "img_b64": b64,
            "marked_path": None,
            "from_disk": False,
        }
        try:
            out = srv.add_section(
                {
                    "basename": basename,
                    "label": label,
                    "points": points,
                }
            )
            if hasattr(out, "body"):
                # JSONResponse on error
                self.fail(f"add_section failed: {out.body}")
            self.assertTrue(out.get("ok"))
            text = lbl.read_text().strip()
            self.assertTrue(text, "label file empty")
            line = text.splitlines()[0]
            cid = int(line.split()[0])
            parts = line.split()
            coords = list(map(float, parts[1:]))
            self.assertGreaterEqual(len(coords), 6)
            return cid, line, out
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_furniture_points_yolo_id_11(self):
        pts = [[50, 50], [150, 50], [150, 120], [50, 120]]
        cid, line, _ = self._mock_and_section("Furniture", pts)
        self.assertEqual(cid, 11)
        self.assertTrue(line.startswith("11 "))

    def test_electric_appliance_points_yolo_id_14(self):
        pts = [[10, 10], [80, 10], [80, 60]]
        cid, line, _ = self._mock_and_section("ElectricAppliance", pts)
        self.assertEqual(cid, 14)
        self.assertTrue(line.startswith("14 "))

    def test_door_window_points_yolo_ids(self):
        cid_d, _, _ = self._mock_and_section(
            "Door", [[20, 20], [60, 20], [60, 90], [20, 90]]
        )
        cid_w, _, _ = self._mock_and_section(
            "Window", [[100, 100], [180, 100], [180, 140]]
        )
        self.assertEqual(cid_d, CLASS_IDS["Door"])
        self.assertEqual(cid_w, CLASS_IDS["Window"])

    def test_contour_to_yolo_preserves_vertex_count(self):
        pts = np.array([[10, 10], [40, 10], [40, 40], [25, 50], [10, 40]], np.int32)
        cnt = pts.reshape(-1, 1, 2)
        line = contour_to_yolo_seg(cnt, 200, 200, CLASS_IDS["Furniture"])
        coords = line.split()[1:]
        self.assertEqual(len(coords), 10)  # 5 vertices * 2


if __name__ == "__main__":
    unittest.main()
