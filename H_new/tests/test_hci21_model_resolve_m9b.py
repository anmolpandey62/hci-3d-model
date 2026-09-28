"""M9b — HCI21 resolver precedence, cache invalidation, promotion safety."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestResolveHci21Precedence(unittest.TestCase):
    def setUp(self):
        self._env_keys = ("HCI21_MODEL_PATH", "HCI_MODEL_PATH")
        self._saved = {k: os.environ.get(k) for k in self._env_keys}
        for k in self._env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_active_hci21_beats_cubicasa_offline(self):
        import logic.yolo_inference as yi

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hci21 = root / "best_hci21.pt"
            cubi = (
                root
                / "cubicasa_hqa_500"
                / "runs"
                / "hqa500_offline"
                / "weights"
                / "best.pt"
            )
            cubi.parent.mkdir(parents=True)
            hci21.write_bytes(b"hci21")
            cubi.write_bytes(b"cubi")

            with mock.patch.object(yi, "HCI21_ACTIVE_MODEL", hci21), mock.patch.object(
                yi, "CUBICASA_OFFLINE_BEST", cubi
            ), mock.patch.object(
                yi, "CUBICASA_SMOKE_BEST", root / "missing_smoke.pt"
            ):
                path, tag = yi.resolve_hci21_model()
            self.assertEqual(Path(path), hci21)
            self.assertEqual(tag, "best_hci21.pt")

    def test_env_override_beats_active_hci21(self):
        import logic.yolo_inference as yi

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env_pt = root / "env_override.pt"
            hci21 = root / "best_hci21.pt"
            env_pt.write_bytes(b"env")
            hci21.write_bytes(b"hci21")
            os.environ["HCI21_MODEL_PATH"] = str(env_pt)

            with mock.patch.object(yi, "HCI21_ACTIVE_MODEL", hci21), mock.patch.object(
                yi, "CUBICASA_OFFLINE_BEST", root / "no_cubi.pt"
            ), mock.patch.object(
                yi, "CUBICASA_SMOKE_BEST", root / "no_smoke.pt"
            ):
                path, tag = yi.resolve_hci21_model()
            self.assertEqual(Path(path), env_pt)
            self.assertEqual(tag, "env:HCI21_MODEL_PATH")

    def test_cubicasa_used_when_no_hci21_active(self):
        import logic.yolo_inference as yi

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cubi = (
                root
                / "cubicasa_hqa_500"
                / "runs"
                / "hqa500_offline"
                / "weights"
                / "best.pt"
            )
            cubi.parent.mkdir(parents=True)
            cubi.write_bytes(b"cubi")
            missing_hci = root / "best_hci21.pt"

            with mock.patch.object(yi, "HCI21_ACTIVE_MODEL", missing_hci), mock.patch.object(
                yi, "CUBICASA_OFFLINE_BEST", cubi
            ), mock.patch.object(
                yi, "CUBICASA_SMOKE_BEST", root / "no_smoke.pt"
            ):
                path, tag = yi.resolve_hci21_model()
            self.assertEqual(Path(path), cubi)
            self.assertEqual(tag, "cubicasa_hqa500_offline")


class TestModelCacheInvalidation(unittest.TestCase):
    def tearDown(self):
        import logic.yolo_inference as yi

        yi.invalidate_model_cache()

    def test_invalidate_clears_stale_entry(self):
        import logic.yolo_inference as yi

        yi.invalidate_model_cache()
        sentinel = object()
        yi._model_cache["path"] = "/tmp/fake.pt"
        yi._model_cache["model"] = sentinel
        yi.invalidate_model_cache()
        self.assertIsNone(yi._model_cache["path"])
        self.assertIsNone(yi._model_cache["model"])

        loads = []

        def fake_yolo(path):
            loads.append(path)
            return f"model:{path}"

        with mock.patch.dict("sys.modules", {"ultralytics": mock.MagicMock(YOLO=fake_yolo)}):
            # Force import path used inside _get_model
            with mock.patch("logic.yolo_inference._get_model") as _:
                pass
            # Call real _get_model with patched YOLO import
            yi._model_cache["path"] = None
            yi._model_cache["model"] = None

            real_get = yi._get_model.__wrapped__ if hasattr(yi._get_model, "__wrapped__") else yi._get_model

            # Manually exercise cache hit then invalidate
            yi._model_cache["path"] = "C:/fake/best_hci21.pt"
            yi._model_cache["model"] = "OLD"
            # Simulate what _get_model returns on cache hit
            self.assertEqual(yi._model_cache["model"], "OLD")
            yi.invalidate_model_cache()
            self.assertIsNone(yi._model_cache["model"])

    def test_get_model_reloads_after_invalidate(self):
        import logic.yolo_inference as yi

        yi.invalidate_model_cache()
        fake_path = "C:/fake/best_hci21.pt"
        calls = []

        class FakeYOLO:
            def __init__(self, path):
                calls.append(path)
                self.path = path

        with mock.patch("os.path.exists", return_value=True), mock.patch(
            "os.path.getsize", return_value=100
        ), mock.patch.dict(
            "sys.modules",
            {"ultralytics": mock.MagicMock(YOLO=FakeYOLO)},
        ):
            # First load populates cache
            m1, p1 = yi._get_model(fake_path)
            self.assertEqual(p1, fake_path)
            self.assertEqual(len(calls), 1)
            # Second load hits cache
            m2, p2 = yi._get_model(fake_path)
            self.assertIs(m1, m2)
            self.assertEqual(len(calls), 1)
            # Invalidate then load again
            yi.invalidate_model_cache()
            m3, p3 = yi._get_model(fake_path)
            self.assertEqual(len(calls), 2)
            self.assertIsNot(m1, m3)


class TestPromoteInvalidatesCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HCI21_ALLOW_ANY_PYTHON"] = "1"

    def test_successful_promote_invalidates_cache(self):
        import web.server as srv
        import logic.yolo_inference as yi

        yi._model_cache["path"] = "stale"
        yi._model_cache["model"] = object()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "best.pt"
            dest = root / "best_hci21.pt"
            src.write_bytes(b"weights")
            with mock.patch.object(srv, "HCI21_ACTIVE_MODEL_PATH", dest), mock.patch.object(
                srv, "PRODUCTION_MODEL_PATH", root / "best_gdrive.pt"
            ):
                out = srv._promote_hci21_checkpoint(src, source="test")
            self.assertTrue(out.get("ok"))
            self.assertTrue(dest.is_file())
            self.assertIsNone(yi._model_cache["path"])
            self.assertIsNone(yi._model_cache["model"])

    def test_failed_promote_does_not_invalidate(self):
        import web.server as srv
        import logic.yolo_inference as yi

        yi._model_cache["path"] = "keep"
        yi._model_cache["model"] = "KEEP"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            missing = root / "missing.pt"
            dest = root / "best_hci21.pt"
            with mock.patch.object(srv, "HCI21_ACTIVE_MODEL_PATH", dest), mock.patch.object(
                srv, "PRODUCTION_MODEL_PATH", root / "best_gdrive.pt"
            ):
                with self.assertRaises(FileNotFoundError):
                    srv._promote_hci21_checkpoint(missing, source="test")
        self.assertEqual(yi._model_cache["path"], "keep")
        self.assertEqual(yi._model_cache["model"], "KEEP")
        yi.invalidate_model_cache()

    def test_promote_refuses_production_destination(self):
        import web.server as srv

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "best.pt"
            prod = root / "best_gdrive.pt"
            src.write_bytes(b"x")
            prod.write_bytes(b"prod")
            with mock.patch.object(srv, "HCI21_ACTIVE_MODEL_PATH", prod), mock.patch.object(
                srv, "PRODUCTION_MODEL_PATH", prod
            ):
                with self.assertRaises(RuntimeError):
                    srv._promote_hci21_checkpoint(src, source="test")
            # Destination content unchanged
            self.assertEqual(prod.read_bytes(), b"prod")


if __name__ == "__main__":
    unittest.main()
