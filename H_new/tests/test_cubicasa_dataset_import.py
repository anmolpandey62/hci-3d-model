"""Tests for Cubicasa → H dataset importer (no Cubicasa package imports)."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from config.classes import CLASS_IDS, CLASS_NAMES
from logic.cubicasa_dataset_import import (
    ELECTRIC_APPLIANCE_ID,
    FURNITURE_ID,
    CubicasaImportError,
    import_cubicasa_dataset,
    parse_yolo_seg_line,
    resolve_training_yaml,
    validate_source_dataset,
    write_hci_dataset_yaml,
)


def _hci_yaml_text(path_value: str = ".") -> str:
    lines = [
        f"path: {path_value}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    return "\n".join(lines) + "\n"


def _write_png(path: Path) -> None:
    # Minimal valid 1x1 PNG
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415408d763f8ffff3f0005fe02fea75a87a90000000049454e44ae426082"
    )
    path.write_bytes(png)


def _build_mini_source(root: Path, *, leak: bool = False, bad_label: bool = False, bad_ids: bool = False) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # train samples
    _write_png(root / "images" / "train" / "a.png")
    _write_png(root / "images" / "train" / "b.png")
    (root / "labels" / "train" / "a.txt").write_text(
        "11 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        "14 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n",
        encoding="utf-8",
    )
    (root / "labels" / "train" / "b.txt").write_text(
        "3 0.0 0.0 1.0 0.0 1.0 0.1 0.0 0.1\n",
        encoding="utf-8",
    )

    # val samples
    _write_png(root / "images" / "val" / "c.png")
    val_label = "11 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n"
    if bad_label:
        val_label = "11 0.1 0.2 0.3\n"  # odd coords / <3 points
    (root / "labels" / "val" / "c.txt").write_text(val_label, encoding="utf-8")

    if leak:
        _write_png(root / "images" / "val" / "a.png")
        (root / "labels" / "val" / "a.txt").write_text(
            "14 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
            encoding="utf-8",
        )

    yaml_text = _hci_yaml_text(root.as_posix())
    if bad_ids:
        yaml_text = yaml_text.replace("  11: Furniture", "  11: Sofa")
    (root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    return root


class TestClassTaxonomy(unittest.TestCase):
    def test_furniture_and_ea_ids(self):
        self.assertEqual(CLASS_IDS["Furniture"], 11)
        self.assertEqual(CLASS_IDS["ElectricAppliance"], 14)
        self.assertEqual(FURNITURE_ID, 11)
        self.assertEqual(ELECTRIC_APPLIANCE_ID, 14)
        self.assertEqual(len(CLASS_NAMES), 17)

    def test_no_renumbering_constants(self):
        self.assertEqual(CLASS_NAMES[11], "Furniture")
        self.assertEqual(CLASS_NAMES[14], "ElectricAppliance")


class TestYoloSegParse(unittest.TestCase):
    def test_valid_polygon(self):
        cid, coords = parse_yolo_seg_line("11 0.1 0.1 0.2 0.1 0.2 0.2")
        self.assertEqual(cid, 11)
        self.assertEqual(len(coords), 6)

    def test_blank(self):
        self.assertIsNone(parse_yolo_seg_line("  "))

    def test_invalid_odd_coords(self):
        with self.assertRaises(CubicasaImportError):
            parse_yolo_seg_line("11 0.1 0.2 0.3")

    def test_invalid_class(self):
        with self.assertRaises(CubicasaImportError):
            parse_yolo_seg_line("99 0.1 0.1 0.2 0.1 0.2 0.2")


class TestValidateAndImport(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hci_cubi_imp_"))
        self.source = self.tmp / "source_run"
        self.imports = self.tmp / "imports"
        _build_mini_source(self.source)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_source_discovery_and_structure(self):
        report = validate_source_dataset(self.source)
        self.assertTrue(report.ok, msg=[i.message for i in report.errors()])
        self.assertEqual(report.train.paired, 2)
        self.assertEqual(report.val.paired, 1)

    def test_class_map_furniture_ea(self):
        report = validate_source_dataset(self.source)
        self.assertTrue(report.class_map_ok)
        self.assertEqual(report.source_yaml_names[11], "Furniture")
        self.assertEqual(report.source_yaml_names[14], "ElectricAppliance")
        self.assertGreaterEqual(report.furniture_instances, 1)
        self.assertGreaterEqual(report.electric_appliance_instances, 1)

    def test_rejects_class_renumber_mismatch(self):
        bad = self.tmp / "bad_ids"
        _build_mini_source(bad, bad_ids=True)
        report = validate_source_dataset(bad)
        self.assertFalse(report.ok)
        codes = {i.code for i in report.errors()}
        self.assertTrue("class_name_mismatch" in codes or "furniture_id" in codes)

    def test_rejects_bad_label(self):
        bad = self.tmp / "bad_lbl"
        _build_mini_source(bad, bad_label=True)
        report = validate_source_dataset(bad)
        self.assertFalse(report.ok)
        self.assertTrue(any("bad_label" in i.code for i in report.errors()))

    def test_rejects_train_val_leakage(self):
        leak = self.tmp / "leak"
        _build_mini_source(leak, leak=True)
        report = validate_source_dataset(leak)
        self.assertFalse(report.ok)
        self.assertTrue(any(i.code == "train_val_leakage" for i in report.errors()))

    def test_import_preserves_polygons_and_splits(self):
        src_a = (self.source / "labels" / "train" / "a.txt").read_text(encoding="utf-8")
        result = import_cubicasa_dataset(
            self.source,
            version="mini_v1",
            imports_base=self.imports,
        )
        self.assertTrue(result.ok)
        dest = Path(result.dest_root)
        dst_a = (dest / "labels" / "train" / "a.txt").read_text(encoding="utf-8")
        self.assertEqual(src_a, dst_a)
        self.assertTrue((dest / "images" / "train" / "a.png").is_file())
        self.assertTrue((dest / "images" / "val" / "c.png").is_file())
        self.assertFalse((dest / "images" / "train" / "c.png").exists())
        self.assertTrue((dest / "manifest.json").is_file())
        man = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(man["furniture_class_id"], 11)
        self.assertEqual(man["electric_appliance_class_id"], 14)
        self.assertEqual(man["nc"], 17)
        self.assertEqual(man["counts"]["train_images"], 2)
        self.assertEqual(man["counts"]["val_images"], 1)

        # dataset.yaml 17-class
        ytxt = (dest / "dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("nc: 17", ytxt)
        self.assertIn("11: Furniture", ytxt)
        self.assertIn("14: ElectricAppliance", ytxt)
        self.assertIn("train: images/train", ytxt)
        self.assertIn("val: images/val", ytxt)
        self.assertNotIn("val: images/train", ytxt)

    def test_refuse_overwrite(self):
        import_cubicasa_dataset(self.source, version="mini_v1", imports_base=self.imports)
        with self.assertRaises(CubicasaImportError):
            import_cubicasa_dataset(self.source, version="mini_v1", imports_base=self.imports)

    def test_source_unchanged_after_import(self):
        before = {}
        for p in self.source.rglob("*"):
            if p.is_file():
                before[str(p.relative_to(self.source))] = (
                    p.stat().st_mtime_ns,
                    hashlib.sha256(p.read_bytes()).hexdigest(),
                )
        import_cubicasa_dataset(self.source, version="mini_v2", imports_base=self.imports)
        for rel, (mt, hx) in before.items():
            p = self.source / rel
            self.assertEqual(p.stat().st_mtime_ns, mt)
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), hx)

    def test_resolve_training_yaml(self):
        import_cubicasa_dataset(self.source, version="mini_v3", imports_base=self.imports)
        y = resolve_training_yaml("mini_v3", self.imports)
        self.assertTrue(y.is_file())
        self.assertEqual(y.name, "dataset.yaml")

    def test_write_hci_dataset_yaml_nc17(self):
        d = self.tmp / "yaml_only"
        d.mkdir()
        p = write_hci_dataset_yaml(d)
        text = p.read_text(encoding="utf-8")
        self.assertIn("nc: 17", text)
        for i, name in enumerate(CLASS_NAMES):
            self.assertIn(f"  {i}: {name}", text)

    def test_gdrive_not_touched(self):
        """Importer must not write into a sibling gdrive_dataset folder."""
        gdrive = self.tmp / "gdrive_dataset"
        gdrive.mkdir()
        sentinel = gdrive / "KEEP.txt"
        sentinel.write_text("untouched", encoding="utf-8")
        mt = sentinel.stat().st_mtime_ns
        import_cubicasa_dataset(self.source, version="mini_v4", imports_base=self.imports)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
        self.assertEqual(sentinel.stat().st_mtime_ns, mt)
        self.assertFalse(any(gdrive.rglob("cubicasa*")))

    def test_no_cubicasa_package_dependency(self):
        """Runtime module must not import Cubicasa converter code."""
        import logic.cubicasa_dataset_import as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from cubicasa_converter", src)
        self.assertNotIn("import cubicasa_converter", src)
        self.assertNotIn("sys.path", src)  # must not inject Cubicasa into path


class TestRealFreezeSmoke(unittest.TestCase):
    """Optional smoke against the audited freeze (validate-only; no full copy)."""

    FREEZE = Path(r"C:\cubicasa_converter\output\runs\20260807_024316")

    @unittest.skipUnless(FREEZE.is_dir(), "Cubicasa freeze not present on this machine")
    def test_real_freeze_taxonomy_and_counts(self):
        report = validate_source_dataset(self.FREEZE)
        self.assertTrue(report.ok, msg=[i.message for i in report.errors()][:10])
        self.assertEqual(report.source_yaml_names[11], "Furniture")
        self.assertEqual(report.source_yaml_names[14], "ElectricAppliance")
        self.assertEqual(report.train.paired, 4188)
        self.assertEqual(report.val.paired, 400)
        self.assertEqual(report.furniture_instances, 73964)
        self.assertEqual(report.electric_appliance_instances, 20546)


if __name__ == "__main__":
    unittest.main()
