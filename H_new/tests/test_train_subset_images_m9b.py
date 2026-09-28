"""M9b — corrected training subset image resolution (images/train + images_raw)."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestFinetuneImageResolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"

    def test_prefers_images_train(self):
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "train"
            raw_dir = root / "images_raw"
            img_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            train_img = img_dir / "door_test.jpg"
            raw_img = raw_dir / "door_test.jpg"
            train_img.write_bytes(b"train")
            raw_img.write_bytes(b"raw")
            got = srv._resolve_finetune_image("door_test", img_dir, raw_dir)
            self.assertEqual(got, train_img)

    def test_falls_back_to_images_raw(self):
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "train"
            raw_dir = root / "images_raw"
            img_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            raw_img = raw_dir / "door_test.jpg"
            raw_img.write_bytes(b"raw")
            got = srv._resolve_finetune_image("door_test", img_dir, raw_dir)
            self.assertEqual(got, raw_img)

    def test_missing_image_returns_none(self):
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "train"
            raw_dir = root / "images_raw"
            img_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            got = srv._resolve_finetune_image("door_test", img_dir, raw_dir)
            self.assertIsNone(got)

    def test_subset_builder_copies_from_raw(self):
        """Simulate subset copy loop using the same helper as _finetune_worker."""
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "train"
            raw_dir = root / "images_raw"
            lbl_dir = root / "labels" / "train"
            img_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            (raw_dir / "door_test.jpg").write_bytes(b"JPEGDATA")
            (lbl_dir / "door_test.txt").write_text("2 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

            tmp = Path(tempfile.mkdtemp(prefix="ft_subset_test_"))
            try:
                tmp_imgs = tmp / "images" / "train"
                tmp_lbls = tmp / "labels" / "train"
                tmp_imgs.mkdir(parents=True)
                tmp_lbls.mkdir(parents=True)
                lf = lbl_dir / "door_test.txt"
                src_img = srv._resolve_finetune_image(lf.stem, img_dir, raw_dir)
                self.assertIsNotNone(src_img)
                shutil.copy2(str(lf), str(tmp_lbls / lf.name))
                shutil.copy2(str(src_img), str(tmp_imgs / src_img.name))
                self.assertTrue((tmp_imgs / "door_test.jpg").is_file())
                self.assertTrue((tmp_lbls / "door_test.txt").is_file())
                # Original raw untouched
                self.assertEqual((raw_dir / "door_test.jpg").read_bytes(), b"JPEGDATA")
            finally:
                shutil.rmtree(str(tmp), ignore_errors=True)

    def test_subset_missing_image_is_loud_failure_path(self):
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_dir = root / "images" / "train"
            raw_dir = root / "images_raw"
            img_dir.mkdir(parents=True)
            raw_dir.mkdir(parents=True)
            self.assertIsNone(srv._resolve_finetune_image("ghost", img_dir, raw_dir))


if __name__ == "__main__":
    unittest.main()
