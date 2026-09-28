"""M5a — geometry-preserving free-polygon edit (move + vertex) for FREE_POLY_CLASSES."""
from __future__ import annotations

import base64
import os
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


class TestFreePolyEditModeStatic(unittest.TestCase):
    def test_edit_mode_has_freepoly_branch(self):
        self.assertIn("freepoly", INDEX)
        self.assertIn("_startFreePolyVertexEdit", INDEX)
        # Door/free-poly must not fall through to aabb when N>=3
        m = re.search(
            r"function _editModeForLabel\(cls, poly\) \{([\s\S]*?)\nfunction _dist2",
            INDEX,
        )
        self.assertIsNotNone(m, "_editModeForLabel not found")
        body = m.group(1)
        self.assertIn("FREE_POLY_CLASSES", body)
        self.assertIn("'freepoly'", body)
        # Wall still first
        self.assertLess(body.find("Wall"), body.find("FREE_POLY_CLASSES"))

    def test_apply_resize_includes_vertex_mode(self):
        m = re.search(
            r"async function _applyResize\(\) \{([\s\S]*?)\n// Hide edit overlay when image changes",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertIn("vertex", body)
        self.assertIn("points", body)
        self.assertIn("/api/resize_label", body)

    def test_freepoly_overlay_adds_vertex_handles(self):
        self.assertIn("mode === 'freepoly'", INDEX)
        self.assertIn("_startFreePolyVertexEdit", INDEX)


class TestFreePolyEditApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"

    def _mock(self, label: str, points: list, basename: str = "m5a_door"):
        import web.server as srv

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
        cnt = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        srv._analysis[basename] = {
            "labelled": {label: [cnt]},
            "label_lines": [],
            "n_labels": 1,
            "img_h": img_h,
            "img_w": img_w,
            "img_b64": b64,
            "marked_path": None,
            "from_disk": False,
        }
        return srv, old_ds, tmp, lbl, basename

    def test_door_edit_mode_not_aabb_static(self):
        # Static proof Door with poly resolves to freepoly in source
        self.assertRegex(
            INDEX,
            r"FREE_POLY_CLASSES\.indexOf\(cls\)\s*>=\s*0",
        )
        self.assertIn("'Door'", INDEX)

    def test_door_points_move_preserves_n(self):
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts)
        try:
            moved = [[p[0] + 10, p[1] + 5] for p in pts]
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "points": moved,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"resize_label failed: {out.body}")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("n_vertices"), 4)
            self.assertEqual(len(out.get("points")), 4)
            line = lbl.read_text().strip().splitlines()[0]
            self.assertTrue(line.startswith("2 "))
            coords = line.split()[1:]
            self.assertEqual(len(coords), 8)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_door_vertex_edit_preserves_others(self):
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts)
        try:
            edited = [list(p) for p in pts]
            edited[1] = [130, 50]
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "points": edited,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"resize_label failed: {out.body}")
            self.assertEqual(out["n_vertices"], 4)
            got = out["points"]
            self.assertEqual(got[0], [40, 40])
            self.assertEqual(got[1], [130, 50])
            self.assertEqual(got[2], [120, 160])
            self.assertEqual(got[3], [40, 160])
            line = lbl.read_text().strip().splitlines()[0]
            self.assertEqual(int(line.split()[0]), CLASS_IDS["Door"])
            self.assertEqual(len(line.split()[1:]), 8)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_door_yolo_class_id_2(self):
        pts = np.array([[10, 10], [50, 10], [50, 80], [10, 80]], np.int32)
        cnt = pts.reshape(-1, 1, 2)
        line = contour_to_yolo_seg(cnt, 200, 200, CLASS_IDS["Door"])
        self.assertTrue(line.startswith("2 "))
        self.assertEqual(len(line.split()[1:]), 8)

    def test_door_bbox_rejected_without_force_aabb(self):
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts)
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "bbox": [10, 10, 100, 100],
                }
            )
            self.assertTrue(hasattr(out, "status_code"))
            self.assertEqual(out.status_code, 400)
            # Original label file still empty or unchanged contour in memory
            cnt = srv._analysis[basename]["labelled"]["Door"][0]
            self.assertEqual(cnt.reshape(-1, 2).shape[0], 4)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_furniture_bbox_also_guarded(self):
        pts = [[10, 10], [60, 10], [60, 50]]
        srv, old_ds, tmp, lbl, basename = self._mock(
            "Furniture", pts, basename="m5a_furn"
        )
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Furniture",
                    "idx": 1,
                    "bbox": [1, 1, 80, 80],
                }
            )
            self.assertEqual(out.status_code, 400)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_room_aabb_still_allowed(self):
        """Room remains AABB-editable (not in FREE_POLY)."""
        pts = [[10, 10], [90, 10], [90, 90], [10, 90]]
        srv, old_ds, tmp, lbl, basename = self._mock("Room", pts, basename="m5a_room")
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Room",
                    "idx": 1,
                    "bbox": [20, 20, 50, 50],
                }
            )
            if hasattr(out, "body"):
                self.fail(f"Room bbox resize failed: {out.body}")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("n_vertices"), 4)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)


if __name__ == "__main__":
    unittest.main()
