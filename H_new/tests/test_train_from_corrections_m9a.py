"""M9a — HITL correction session tracking for train_from_corrections 'corrected' scope."""
from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class TestM9aCorrectionSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"

    def setUp(self):
        import web.server as srv

        self.srv = srv
        self._old_ds = srv.DATASET_DIR
        self._tmp = Path(tempfile.mkdtemp())
        (self._tmp / "labels" / "train").mkdir(parents=True)
        srv.DATASET_DIR = self._tmp
        srv._corrected_basenames.clear()

    def tearDown(self):
        for key in list(self.srv._analysis.keys()):
            if key.startswith("m9a_"):
                self.srv._analysis.pop(key, None)
        self.srv._corrected_basenames.clear()
        self.srv.DATASET_DIR = self._old_ds

    def _seed(self, basename: str, label: str, points: list):
        srv = self.srv
        img_w, img_h = 400, 300
        img = np.zeros((img_h, img_w, 3), np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        self.assertTrue(ok)
        b64 = base64.b64encode(buf.tobytes()).decode()
        cnt = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        lbl = self._tmp / "labels" / "train" / f"{basename}.txt"
        lbl.write_text("")
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
        return lbl

    def test_resize_points_registers_corrected(self):
        basename = "m9a_door_edit"
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        self._seed(basename, "Door", pts)
        self.assertNotIn(basename, self.srv._corrected_basenames)

        moved = [[p[0] + 5, p[1]] for p in pts]
        out = self.srv.resize_label(
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
        self.assertIn(basename, self.srv._corrected_basenames)

        listing = self.srv.get_corrected_files()
        self.assertIn(basename, listing["corrected"])

    def test_section_points_registers_corrected(self):
        basename = "m9a_win_create"
        # Seed empty labelled analysis so section can append
        self._seed(basename, "Door", [[10, 10], [30, 10], [30, 30], [10, 30]])
        self.srv._corrected_basenames.clear()

        pts = [[50, 50], [100, 50], [100, 80], [50, 80]]
        out = self.srv.add_section(
            {
                "basename": basename,
                "label": "Window",
                "points": pts,
            }
        )
        if hasattr(out, "body"):
            self.fail(f"add_section failed: {out.body}")
        self.assertTrue(out.get("ok"))
        self.assertIn(basename, self.srv._corrected_basenames)
        self.assertIn(basename, self.srv.get_corrected_files()["corrected"])

    def test_failed_resize_does_not_register(self):
        basename = "m9a_door_fail"
        pts = [[40, 40], [120, 40], [120, 160], [40, 160]]
        self._seed(basename, "Door", pts)
        self.srv._corrected_basenames.clear()

        # Self-intersecting quad (bow-tie) — resize rejects with reject_self_intersect
        bad = [[40, 40], [120, 160], [120, 40], [40, 160]]
        out = self.srv.resize_label(
            {
                "basename": basename,
                "cls_name": "Door",
                "idx": 1,
                "points": bad,
            }
        )
        self.assertTrue(hasattr(out, "status_code"))
        self.assertEqual(out.status_code, 400)
        self.assertNotIn(basename, self.srv._corrected_basenames)

    def test_failed_section_does_not_register(self):
        basename = "m9a_sec_fail"
        self._seed(basename, "Door", [[10, 10], [20, 10], [20, 20]])
        self.srv._corrected_basenames.clear()

        out = self.srv.add_section(
            {
                "basename": basename,
                "label": "Door",
                "points": [[1, 1], [2, 2]],  # < 3 points
            }
        )
        self.assertTrue(hasattr(out, "status_code"))
        self.assertEqual(out.status_code, 400)
        self.assertNotIn(basename, self.srv._corrected_basenames)

    def test_api_correct_still_registers(self):
        basename = "m9a_relabel"
        pts = [[10, 10], [40, 10], [40, 40], [10, 40]]
        self._seed(basename, "Door", pts)
        self.srv._corrected_basenames.clear()

        out = self.srv.correct_label(
            {
                "basename": basename,
                "action": "relabel",
                "cls_name": "Door",
                "idx": 1,
                "new_cls": "Window",
            }
        )
        if hasattr(out, "body"):
            self.fail(f"correct_label failed: {out.body}")
        self.assertTrue(out.get("ok"))
        self.assertIn(basename, self.srv._corrected_basenames)

    def test_save_corrections_still_registers(self):
        basename = "m9a_save"
        pts = [[10, 10], [40, 10], [40, 40], [10, 40]]
        self._seed(basename, "Door", pts)
        # Simulate existing label_lines as save_corrections writes those
        self.srv._analysis[basename]["label_lines"] = ["2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"]
        self.srv._corrected_basenames.clear()

        out = self.srv.save_corrections({"basename": basename})
        if hasattr(out, "body"):
            self.fail(f"save_corrections failed: {out.body}")
        self.assertTrue(out.get("ok"))
        self.assertIn(basename, self.srv._corrected_basenames)

    def test_corrected_scope_selection_for_train(self):
        """Contract: train_from_corrections corrected scope uses _corrected_basenames."""
        basename = "m9a_scope"
        pts = [[20, 20], [80, 20], [80, 90], [20, 90]]
        lbl = self._seed(basename, "Door", pts)
        lbl.write_text("2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

        self.srv._corrected_basenames.clear()
        out = self.srv.resize_label(
            {
                "basename": basename,
                "cls_name": "Door",
                "idx": 1,
                "points": [[25, 20], [80, 20], [80, 90], [20, 90]],
            }
        )
        self.assertTrue(out.get("ok"))

        # Mirror selection logic used by train_from_corrections (no GPU train)
        lbl_dir = self.srv.DATASET_DIR / "labels" / "train"
        selected = [
            b
            for b in self.srv._corrected_basenames
            if (lbl_dir / (b + ".txt")).exists()
        ]
        self.assertEqual(selected, [basename])

    def test_furniture_edit_uses_same_registration(self):
        basename = "m9a_furn"
        pts = [[10, 10], [60, 10], [60, 50], [10, 50]]
        self._seed(basename, "Furniture", pts)
        self.srv._corrected_basenames.clear()
        out = self.srv.resize_label(
            {
                "basename": basename,
                "cls_name": "Furniture",
                "idx": 1,
                "points": [[12, 10], [60, 10], [60, 50], [10, 50]],
            }
        )
        self.assertTrue(out.get("ok"))
        self.assertIn(basename, self.srv._corrected_basenames)


if __name__ == "__main__":
    unittest.main()
