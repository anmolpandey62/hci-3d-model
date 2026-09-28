"""Phase 3 — geometry-preserving Wall polygon edit via /api/resize_label."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from logic.yolo_inference import contour_to_yolo_seg
from tests.test_wall_strip_section import strip_corners_px


def _n_vertices(cnt) -> int:
    return int(cnt.reshape(-1, 2).shape[0])


def _mock_analysis(basename: str, labelled: dict, img_w=800, img_h=600):
    import web.server as srv

    tmp = Path(tempfile.mkdtemp())
    (tmp / "labels" / "train").mkdir(parents=True)
    (tmp / "labels" / "train" / f"{basename}.txt").write_text("")
    old_ds = srv.DATASET_DIR
    srv.DATASET_DIR = tmp

    img = np.zeros((img_h, img_w, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    import base64

    b64 = base64.b64encode(buf.tobytes()).decode()
    srv._analysis[basename] = {
        "labelled": labelled,
        "label_lines": [],
        "n_labels": 0,
        "img_h": img_h,
        "img_w": img_w,
        "img_b64": b64,
        "marked_path": None,
        "from_disk": False,
    }
    return srv, old_ds, tmp


class TestWallStripEdit(unittest.TestCase):
    def test_strip_edit_preserves_4_points(self):
        pts = strip_corners_px((100, 100), (300, 100), 8)
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        before = _n_vertices(cnt)
        self.assertEqual(before, 4)

        srv, old_ds, _tmp = _mock_analysis("p3strip", {"Wall": [cnt]})
        try:
            # Lengthen strip along same axis (still 8px thick)
            new_pts = strip_corners_px((100, 100), (400, 100), 8)
            r = srv.resize_label(
                {
                    "basename": "p3strip",
                    "cls_name": "Wall",
                    "idx": 1,
                    "points": new_pts,
                }
            )
            self.assertTrue(r.get("ok"), r)
            after_cnt = srv._analysis["p3strip"]["labelled"]["Wall"][0]
            after = _n_vertices(after_cnt)
            self.assertEqual(after, 4, f"before={before} after={after}")
            self.assertEqual(r.get("n_vertices"), 4)

            # YOLO rebuild + reload simulation
            lines = srv._analysis["p3strip"]["label_lines"]
            wall_lines = [ln for ln in lines if ln.startswith("3 ")]
            self.assertTrue(wall_lines)
            parts = wall_lines[-1].split()
            self.assertEqual(len(parts), 9)  # cid + 4*2

            # decode like _load_existing_labels
            img_w, img_h = 800, 600
            coords = list(map(float, parts[1:]))
            reloaded = [
                [int(coords[k] * img_w), int(coords[k + 1] * img_h)]
                for k in range(0, len(coords) - 1, 2)
            ]
            self.assertEqual(len(reloaded), 4)
            print(f"EVIDENCE strip: before={before} after={after} reload={len(reloaded)}")
        finally:
            srv.DATASET_DIR = old_ds


class TestMultiVertexWallEdit(unittest.TestCase):
    def test_move_preserves_vertex_count(self):
        # Octagon-ish wall polygon
        pts = [
            [50, 50],
            [100, 40],
            [150, 50],
            [160, 100],
            [150, 150],
            [100, 160],
            [50, 150],
            [40, 100],
        ]
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        before = _n_vertices(cnt)
        self.assertEqual(before, 8)

        srv, old_ds, _tmp = _mock_analysis("p3oct", {"Wall": [cnt]})
        try:
            moved = [[x + 20, y + 10] for x, y in pts]
            r = srv.resize_label(
                {
                    "basename": "p3oct",
                    "cls_name": "Wall",
                    "idx": 1,
                    "points": moved,
                }
            )
            self.assertTrue(r.get("ok"), r)
            after_cnt = srv._analysis["p3oct"]["labelled"]["Wall"][0]
            after = _n_vertices(after_cnt)
            self.assertEqual(after, 8)
            self.assertEqual(r.get("n_vertices"), 8)

            line = [ln for ln in srv._analysis["p3oct"]["label_lines"] if ln.startswith("3 ")][-1]
            parts = line.split()
            # cid + 8*2
            self.assertEqual(len(parts), 1 + 16)
            coords = list(map(float, parts[1:]))
            reloaded = [
                [int(coords[k] * 800), int(coords[k + 1] * 600)]
                for k in range(0, len(coords) - 1, 2)
            ]
            self.assertEqual(len(reloaded), 8)
            print(f"EVIDENCE multi: before={before} after={after} reload={len(reloaded)}")
        finally:
            srv.DATASET_DIR = old_ds


class TestYoloRoundTrip(unittest.TestCase):
    def test_rebuild_preserves_geometry(self):
        pts = strip_corners_px((80, 90), (280, 190), 8)
        cnt = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32).reshape(
            -1, 1, 2
        )
        line = contour_to_yolo_seg(cnt, 800, 600, 3)
        parts = line.split()
        self.assertEqual(parts[0], "3")
        self.assertEqual(len(parts), 9)
        coords = list(map(float, parts[1:]))
        decoded = [
            [int(coords[k] * 800), int(coords[k + 1] * 600)]
            for k in range(0, len(coords) - 1, 2)
        ]
        stored = cnt.reshape(-1, 2).tolist()
        for a, b in zip(stored, decoded):
            self.assertLessEqual(abs(a[0] - b[0]), 1)
            self.assertLessEqual(abs(a[1] - b[1]), 1)


class TestLegacyDoorRoom(unittest.TestCase):
    def test_door_bbox_resize_rejected_without_force_aabb(self):
        """M5a: Door is a free-polygon class — bbox-only resize must not silently destroy N-gons."""
        cnt = np.array(
            [[10, 10], [50, 10], [50, 60], [10, 60]], dtype=np.int32
        ).reshape(-1, 1, 2)
        srv, old_ds, _tmp = _mock_analysis("p3door", {"Door": [cnt]})
        try:
            r = srv.resize_label(
                {
                    "basename": "p3door",
                    "cls_name": "Door",
                    "idx": 1,
                    "bbox": [20, 20, 40, 50],
                }
            )
            self.assertEqual(r.status_code, 400)
            # Contour unchanged
            out = srv._analysis["p3door"]["labelled"]["Door"][0].reshape(-1, 2).tolist()
            self.assertEqual(out, [[10, 10], [50, 10], [50, 60], [10, 60]])
        finally:
            srv.DATASET_DIR = old_ds

    def test_door_force_aabb_allowed(self):
        cnt = np.array(
            [[10, 10], [50, 10], [50, 60], [10, 60]], dtype=np.int32
        ).reshape(-1, 1, 2)
        srv, old_ds, _tmp = _mock_analysis("p3door2", {"Door": [cnt]})
        try:
            r = srv.resize_label(
                {
                    "basename": "p3door2",
                    "cls_name": "Door",
                    "idx": 1,
                    "bbox": [20, 20, 40, 50],
                    "force_aabb": True,
                }
            )
            self.assertTrue(r.get("ok"), r)
            out = srv._analysis["p3door2"]["labelled"]["Door"][0].reshape(-1, 2).tolist()
            self.assertEqual(out, [[20, 20], [60, 20], [60, 70], [20, 70]])
        finally:
            srv.DATASET_DIR = old_ds

    def test_room_bbox_resize(self):
        cnt = np.array(
            [[0, 0], [100, 0], [100, 80], [0, 80]], dtype=np.int32
        ).reshape(-1, 1, 2)
        srv, old_ds, _tmp = _mock_analysis("p3room", {"Room": [cnt]})
        try:
            r = srv.resize_label(
                {
                    "basename": "p3room",
                    "cls_name": "Room",
                    "idx": 1,
                    "bbox": [5, 5, 90, 70],
                }
            )
            self.assertTrue(r.get("ok"), r)
            out = srv._analysis["p3room"]["labelled"]["Room"][0].reshape(-1, 2).tolist()
            self.assertEqual(out, [[5, 5], [95, 5], [95, 75], [5, 75]])
        finally:
            srv.DATASET_DIR = old_ds


class TestDangerousWallBbox(unittest.TestCase):
    def test_wall_bbox_only_rejected(self):
        pts = strip_corners_px((100, 100), (250, 200), 8)
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        before = _n_vertices(cnt)
        srv, old_ds, _tmp = _mock_analysis("p3danger", {"Wall": [cnt]})
        try:
            from fastapi.responses import JSONResponse

            r = srv.resize_label(
                {
                    "basename": "p3danger",
                    "cls_name": "Wall",
                    "idx": 1,
                    "bbox": [10, 10, 200, 100],
                }
            )
            # FastAPI JSONResponse on error
            if isinstance(r, JSONResponse):
                self.assertEqual(r.status_code, 400)
            else:
                self.fail(f"Expected rejection, got {r}")

            after = _n_vertices(srv._analysis["p3danger"]["labelled"]["Wall"][0])
            self.assertEqual(after, before)
            print(f"EVIDENCE reject bbox: before={before} after={after} (unchanged)")
        finally:
            srv.DATASET_DIR = old_ds

    def test_wall_force_aabb_allowed(self):
        pts = strip_corners_px((100, 100), (250, 200), 8)
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        srv, old_ds, _tmp = _mock_analysis("p3force", {"Wall": [cnt]})
        try:
            r = srv.resize_label(
                {
                    "basename": "p3force",
                    "cls_name": "Wall",
                    "idx": 1,
                    "bbox": [10, 10, 200, 100],
                    "force_aabb": True,
                }
            )
            self.assertTrue(r.get("ok"), r)
            out = srv._analysis["p3force"]["labelled"]["Wall"][0].reshape(-1, 2).tolist()
            self.assertEqual(out, [[10, 10], [210, 10], [210, 110], [10, 110]])
        finally:
            srv.DATASET_DIR = old_ds

    def test_self_intersecting_rejected(self):
        # Bow-tie
        bad = [[0, 0], [100, 100], [100, 0], [0, 100]]
        cnt = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32).reshape(
            -1, 1, 2
        )
        srv, old_ds, _tmp = _mock_analysis("p3bow", {"Wall": [cnt]})
        try:
            from fastapi.responses import JSONResponse

            r = srv.resize_label(
                {
                    "basename": "p3bow",
                    "cls_name": "Wall",
                    "idx": 1,
                    "points": bad,
                }
            )
            self.assertIsInstance(r, JSONResponse)
            self.assertEqual(r.status_code, 400)
        finally:
            srv.DATASET_DIR = old_ds


class TestLabelDetailsNoSimplify(unittest.TestCase):
    def test_full_vertices_returned(self):
        pts = [
            [10, 10],
            [30, 5],
            [50, 10],
            [55, 30],
            [50, 50],
            [30, 55],
            [10, 50],
            [5, 30],
        ]
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        srv, old_ds, _tmp = _mock_analysis("p3det", {"Wall": [cnt]})
        try:
            r = srv.get_label_details("p3det")
            det = r["details"]["Wall"][0]
            self.assertEqual(det["n_vertices"], 8)
            self.assertEqual(len(det["poly"]), 8)
        finally:
            srv.DATASET_DIR = old_ds


if __name__ == "__main__":
    unittest.main()
