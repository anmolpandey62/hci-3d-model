"""M5b — free-polygon vertex insert + delete (topology) for FREE_POLY_CLASSES."""
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


def _insert_midpoint(pts: list, edge_i: int) -> list:
    """Mirror FE: insert midpoint after edge_i (between P[i] and P[(i+1)%N])."""
    n = len(pts)
    a, b = pts[edge_i], pts[(edge_i + 1) % n]
    mid = [round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2)]
    out = [list(p) for p in pts]
    out.insert(edge_i + 1, mid)
    return out


def _delete_vertex(pts: list, vi: int) -> list:
    out = [list(p) for p in pts]
    out.pop(vi)
    return out


class TestM5bStaticWiring(unittest.TestCase):
    def test_insert_delete_helpers_present(self):
        self.assertIn("_insertFreePolyEdgeVertex", INDEX)
        self.assertIn("_deleteSelectedFreePolyVertex", INDEX)
        self.assertIn("_commitFreePolyPoints", INDEX)
        self.assertIn("edge-insert", INDEX)
        self.assertIn("selectedVertex", INDEX)

    def test_delete_key_prefers_selected_vertex(self):
        # selectedVertex branch must appear before confirmRemove in keydown handler
        m = re.search(
            r"Delete.*?Backspace[\s\S]*?_activeEdit[\s\S]*?confirmRemove",
            INDEX,
        )
        self.assertIsNotNone(m)
        chunk = m.group(0)
        self.assertIn("_deleteSelectedFreePolyVertex", chunk)
        self.assertLess(
            chunk.find("_deleteSelectedFreePolyVertex"),
            chunk.find("confirmRemove"),
        )

    def test_min_three_guard_in_delete(self):
        m = re.search(
            r"function _deleteSelectedFreePolyVertex\(\) \{([\s\S]*?)\n\}",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertIn("length <= 3", body)
        self.assertIn("at least 3", body)

    def test_insert_uses_splice_not_append(self):
        m = re.search(
            r"function _insertFreePolyEdgeVertex\([\s\S]*?\n\}",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(0)
        self.assertIn("splice(edgeIndex + 1, 0", body)
        self.assertNotIn(".push([nx, ny])", body)

    def test_commit_uses_points_not_bbox(self):
        m = re.search(
            r"async function _commitFreePolyPoints\(newPoly\) \{([\s\S]*?)\n\}",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertIn("points: newPoly", body)
        self.assertIn("/api/resize_label", body)
        self.assertNotIn("bbox:", body)

    def test_wall_still_first_in_edit_mode(self):
        m = re.search(
            r"function _editModeForLabel\(cls, poly\) \{([\s\S]*?)\nfunction _dist2",
            INDEX,
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        self.assertLess(body.find("Wall"), body.find("FREE_POLY_CLASSES"))


class TestM5bOrderingHelpers(unittest.TestCase):
    def test_insert_between_edge_not_append(self):
        pts = [[0, 0], [100, 0], [100, 80], [0, 80]]
        # edge 1 = P1→P2 → insert between index 1 and 2
        out = _insert_midpoint(pts, 1)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0], [0, 0])
        self.assertEqual(out[1], [100, 0])
        self.assertEqual(out[2], [100, 40])
        self.assertEqual(out[3], [100, 80])
        self.assertEqual(out[4], [0, 80])

    def test_delete_preserves_order(self):
        pts = [[0, 0], [100, 0], [100, 80], [0, 80]]
        out = _delete_vertex(pts, 2)
        self.assertEqual(out, [[0, 0], [100, 0], [0, 80]])


class TestM5bApiTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"

    def _mock(self, label: str, points: list, basename: str = "m5b_door"):
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

    def test_door_insert_preserves_order_and_yolo(self):
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        inserted = _insert_midpoint(pts, 1)  # between P1 and P2
        self.assertEqual(len(inserted), 5)
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts)
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "points": inserted,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"insert resize failed: {out.body}")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("n_vertices"), 5)
            got = out["points"]
            self.assertEqual(got[0], [40, 40])
            self.assertEqual(got[1], [120, 40])
            self.assertEqual(got[2], [120, 100])
            self.assertEqual(got[3], [120, 160])
            self.assertEqual(got[4], [40, 160])
            line = lbl.read_text().strip().splitlines()[0]
            self.assertEqual(int(line.split()[0]), CLASS_IDS["Door"])
            self.assertEqual(len(line.split()[1:]), 10)  # 2*(N+1)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_door_delete_preserves_order_and_yolo(self):
        pts = [[40, 40], [120, 40], [120, 100], [120, 160], [40, 160]]
        deleted = _delete_vertex(pts, 2)
        self.assertEqual(len(deleted), 4)
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts, basename="m5b_del")
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "points": deleted,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"delete resize failed: {out.body}")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("n_vertices"), 4)
            self.assertEqual(out["points"], [[40, 40], [120, 40], [120, 160], [40, 160]])
            line = lbl.read_text().strip().splitlines()[0]
            self.assertEqual(int(line.split()[0]), CLASS_IDS["Door"])
            self.assertEqual(len(line.split()[1:]), 8)  # 2*(N-1)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_triangle_still_valid_via_points(self):
        """N=3 is valid; FE rejects delete — API still accepts 3-pt polygon."""
        pts = [[20, 20], [80, 20], [50, 90]]
        srv, old_ds, tmp, lbl, basename = self._mock("Door", pts, basename="m5b_tri")
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Door",
                    "idx": 1,
                    "points": pts,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"triangle resize failed: {out.body}")
            self.assertTrue(out.get("ok"))
            self.assertEqual(out.get("n_vertices"), 3)
            line = lbl.read_text().strip().splitlines()[0]
            self.assertEqual(int(line.split()[0]), 2)
            self.assertEqual(len(line.split()[1:]), 6)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_window_insert_class_id(self):
        pts = [[10, 10], [60, 10], [60, 40], [10, 40]]
        inserted = _insert_midpoint(pts, 0)
        srv, old_ds, tmp, lbl, basename = self._mock(
            "Window", pts, basename="m5b_win"
        )
        try:
            out = srv.resize_label(
                {
                    "basename": basename,
                    "cls_name": "Window",
                    "idx": 1,
                    "points": inserted,
                }
            )
            if hasattr(out, "body"):
                self.fail(f"window insert failed: {out.body}")
            self.assertEqual(out.get("n_vertices"), 5)
            line = lbl.read_text().strip().splitlines()[0]
            self.assertEqual(int(line.split()[0]), CLASS_IDS["Window"])
            self.assertEqual(len(line.split()[1:]), 10)
        finally:
            srv.DATASET_DIR = old_ds
            srv._analysis.pop(basename, None)

    def test_yolo_coord_count_matches_n(self):
        pts = np.array([[10, 10], [50, 10], [50, 40], [30, 50], [10, 40]], np.int32)
        line = contour_to_yolo_seg(pts.reshape(-1, 1, 2), 200, 200, CLASS_IDS["Door"])
        self.assertTrue(line.startswith("2 "))
        self.assertEqual(len(line.split()[1:]), 10)


if __name__ == "__main__":
    unittest.main()
