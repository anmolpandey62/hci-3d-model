#!/usr/bin/env python3
"""
H-side Cubicasa YOLO dataset importer / adapter.

Consumes a *frozen* Cubicasa converter output directory (images + labels +
dataset.yaml) and materializes an H-managed training artifact.

Hard rules:
- Never import Cubicasa Python modules or hard-code runtime dependency on
  C:\\cubicasa_converter.
- Never renumber class IDs; Furniture=11, ElectricAppliance=14 must match
  config.classes.CLASS_IDS exactly.
- Never overwrite gdrive_dataset or existing import versions silently.
- Preserve train/val separation (no leakage).
- Preserve YOLO-seg polygon coordinates byte-for-byte on each label line
  (whitespace-normalized only via exact line copy).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config.classes import CLASS_IDS, CLASS_NAMES, ID_TO_CLASS

# ── Paths ─────────────────────────────────────────────────────────────────────
# Mirror server.py: PROJECT_ROOT is parent of the H application tree.
HCI_ROOT = Path(__file__).resolve().parent.parent  # .../H
PROJECT_ROOT = HCI_ROOT.parent  # typically C:\

DEFAULT_IMPORTS_ROOT = Path(
    os.environ.get(
        "HCI_CUBICASA_IMPORTS_ROOT",
        str(PROJECT_ROOT / "hci_datasets" / "cubicasa_imports"),
    )
)

FURNITURE_ID = 11
ELECTRIC_APPLIANCE_ID = 14
FURNITURE_NAME = "Furniture"
ELECTRIC_APPLIANCE_NAME = "ElectricAppliance"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_YOLO_SEG_LINE = re.compile(
    r"^(\d+)(?:\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))+$"
)


class CubicasaImportError(Exception):
    """Fatal validation / import failure (safe abort; no partial promote)."""


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    path: str | None = None


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    paired: int = 0
    class_instances: dict[str, int] = field(default_factory=dict)
    stems: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    ok: bool
    source_root: str
    issues: list[ValidationIssue] = field(default_factory=list)
    train: SplitStats = field(default_factory=SplitStats)
    val: SplitStats = field(default_factory=SplitStats)
    furniture_instances: int = 0
    electric_appliance_instances: int = 0
    source_yaml_names: dict[int, str] = field(default_factory=dict)
    class_map_ok: bool = False

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_root": self.source_root,
            "issues": [asdict(i) for i in self.issues],
            "train": {
                "images": self.train.images,
                "labels": self.train.labels,
                "paired": self.train.paired,
                "class_instances": self.train.class_instances,
            },
            "val": {
                "images": self.val.images,
                "labels": self.val.labels,
                "paired": self.val.paired,
                "class_instances": self.val.class_instances,
            },
            "furniture_instances": self.furniture_instances,
            "electric_appliance_instances": self.electric_appliance_instances,
            "source_yaml_names": {str(k): v for k, v in self.source_yaml_names.items()},
            "class_map_ok": self.class_map_ok,
        }


@dataclass
class ImportResult:
    ok: bool
    version: str
    dest_root: str
    dataset_yaml: str
    manifest_path: str
    validation: ValidationReport
    skipped_copy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "dest_root": self.dest_root,
            "dataset_yaml": self.dataset_yaml,
            "manifest_path": self.manifest_path,
            "skipped_copy": self.skipped_copy,
            "validation": self.validation.to_dict(),
        }


def imports_root(override: Path | str | None = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(DEFAULT_IMPORTS_ROOT)


def imported_dataset_dir(version: str, root: Path | str | None = None) -> Path:
    return imports_root(root) / version


def imported_dataset_yaml(version: str, root: Path | str | None = None) -> Path:
    return imported_dataset_dir(version, root) / "dataset.yaml"


def list_imported_versions(root: Path | str | None = None) -> list[dict[str, Any]]:
    base = imports_root(root)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        yaml_path = child / "dataset.yaml"
        manifest_path = child / "manifest.json"
        entry: dict[str, Any] = {
            "version": child.name,
            "path": str(child),
            "dataset_yaml": str(yaml_path) if yaml_path.is_file() else None,
            "has_manifest": manifest_path.is_file(),
        }
        if manifest_path.is_file():
            try:
                entry["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                entry["manifest"] = None
        out.append(entry)
    return out


def _issue(issues: list[ValidationIssue], severity: str, code: str, message: str, path: str | None = None) -> None:
    issues.append(ValidationIssue(severity=severity, code=code, message=message, path=path))


def _parse_dataset_yaml(yaml_path: Path) -> tuple[dict[str, Any], dict[int, str]]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CubicasaImportError(f"dataset.yaml is not a mapping: {yaml_path}")
    names_raw = raw.get("names")
    names: dict[int, str] = {}
    if isinstance(names_raw, dict):
        for k, v in names_raw.items():
            names[int(k)] = str(v)
    elif isinstance(names_raw, list):
        for i, v in enumerate(names_raw):
            names[i] = str(v)
    else:
        raise CubicasaImportError(f"dataset.yaml missing usable names: {yaml_path}")
    return raw, names


def _assert_class_taxonomy(names: dict[int, str], issues: list[ValidationIssue]) -> bool:
    """Require exact HCI 17-class identity. STOP on any discrepancy (no remapping)."""
    ok = True
    if CLASS_IDS.get(FURNITURE_NAME) != FURNITURE_ID:
        _issue(
            issues,
            "error",
            "hci_furniture_id",
            f"H CLASS_IDS[{FURNITURE_NAME!r}]={CLASS_IDS.get(FURNITURE_NAME)!r}, expected {FURNITURE_ID}",
        )
        ok = False
    if CLASS_IDS.get(ELECTRIC_APPLIANCE_NAME) != ELECTRIC_APPLIANCE_ID:
        _issue(
            issues,
            "error",
            "hci_ea_id",
            f"H CLASS_IDS[{ELECTRIC_APPLIANCE_NAME!r}]={CLASS_IDS.get(ELECTRIC_APPLIANCE_NAME)!r}, "
            f"expected {ELECTRIC_APPLIANCE_ID}",
        )
        ok = False

    expected = {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}
    if set(names.keys()) != set(expected.keys()):
        _issue(
            issues,
            "error",
            "class_id_set_mismatch",
            f"Source class ID set {sorted(names.keys())} != HCI {sorted(expected.keys())}. "
            "Refusing remapping.",
        )
        ok = False
    for cid, expected_name in expected.items():
        got = names.get(cid)
        if got != expected_name:
            _issue(
                issues,
                "error",
                "class_name_mismatch",
                f"ID {cid}: source={got!r} HCI={expected_name!r}. Refusing remapping.",
            )
            ok = False

    if names.get(FURNITURE_ID) != FURNITURE_NAME:
        _issue(
            issues,
            "error",
            "furniture_id",
            f"Furniture must be ID {FURNITURE_ID}, got {names.get(FURNITURE_ID)!r}",
        )
        ok = False
    if names.get(ELECTRIC_APPLIANCE_ID) != ELECTRIC_APPLIANCE_NAME:
        _issue(
            issues,
            "error",
            "electric_appliance_id",
            f"ElectricAppliance must be ID {ELECTRIC_APPLIANCE_ID}, "
            f"got {names.get(ELECTRIC_APPLIANCE_ID)!r}",
        )
        ok = False

    nc = len(CLASS_NAMES)
    if len(names) != nc:
        _issue(issues, "error", "nc_mismatch", f"Expected nc={nc}, got {len(names)} names")
        ok = False
    return ok


def parse_yolo_seg_line(line: str) -> tuple[int, list[float]] | None:
    """
    Parse one YOLO-seg line. Returns (class_id, coords) or None if blank.
    Raises CubicasaImportError on invalid syntax (caller may catch per-line).
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if not _YOLO_SEG_LINE.match(s):
        raise CubicasaImportError(f"invalid YOLO-seg syntax: {s[:120]!r}")
    parts = s.split()
    cid = int(parts[0])
    if cid < 0 or cid >= len(CLASS_NAMES):
        raise CubicasaImportError(f"class id out of range 0..{len(CLASS_NAMES)-1}: {cid}")
    coords = [float(x) for x in parts[1:]]
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise CubicasaImportError(
            f"polygon needs >=3 points (even coord count >=6), got {len(coords)} floats"
        )
    for c in coords:
        if not (c == c) or c in (float("inf"), float("-inf")):  # NaN / inf
            raise CubicasaImportError("non-finite coordinate in polygon")
    return cid, coords


def _list_images(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if stem in out:
            raise CubicasaImportError(f"duplicate image stem in {folder}: {stem}")
        out[stem] = p
    return out


def _list_labels(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        stem = p.stem
        if stem in out:
            raise CubicasaImportError(f"duplicate label stem in {folder}: {stem}")
        out[stem] = p
    return out


def _scan_split(
    images_dir: Path,
    labels_dir: Path,
    split_name: str,
    issues: list[ValidationIssue],
    *,
    max_label_errors: int = 50,
) -> SplitStats:
    stats = SplitStats()
    try:
        images = _list_images(images_dir)
        labels = _list_labels(labels_dir)
    except CubicasaImportError as e:
        _issue(issues, "error", f"{split_name}_listing", str(e), path=str(images_dir))
        return stats

    stats.images = len(images)
    stats.labels = len(labels)
    stats.stems = sorted(set(images) | set(labels))
    class_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}

    missing_lbl = sorted(set(images) - set(labels))
    missing_img = sorted(set(labels) - set(images))
    for stem in missing_lbl[:20]:
        _issue(
            issues,
            "error",
            f"{split_name}_missing_label",
            f"image without label: {stem}",
            path=str(images[stem]),
        )
    if len(missing_lbl) > 20:
        _issue(
            issues,
            "error",
            f"{split_name}_missing_label_more",
            f"... and {len(missing_lbl) - 20} more images without labels",
        )
    for stem in missing_img[:20]:
        _issue(
            issues,
            "error",
            f"{split_name}_missing_image",
            f"label without image: {stem}",
            path=str(labels[stem]),
        )
    if len(missing_img) > 20:
        _issue(
            issues,
            "error",
            f"{split_name}_missing_image_more",
            f"... and {len(missing_img) - 20} more labels without images",
        )

    paired = sorted(set(images) & set(labels))
    stats.paired = len(paired)
    err_count = 0
    for stem in paired:
        lp = labels[stem]
        try:
            text = lp.read_text(encoding="utf-8")
        except Exception as e:
            _issue(issues, "error", f"{split_name}_label_read", str(e), path=str(lp))
            err_count += 1
            if err_count >= max_label_errors:
                break
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            try:
                parsed = parse_yolo_seg_line(line)
            except CubicasaImportError as e:
                _issue(
                    issues,
                    "error",
                    f"{split_name}_bad_label",
                    f"{lp.name}:{line_no}: {e}",
                    path=str(lp),
                )
                err_count += 1
                if err_count >= max_label_errors:
                    _issue(
                        issues,
                        "error",
                        f"{split_name}_label_error_cap",
                        f"stopped after {max_label_errors} label errors in {split_name}",
                    )
                    stats.class_instances = class_counts
                    return stats
                continue
            if parsed is None:
                continue
            cid, _coords = parsed
            class_counts[ID_TO_CLASS[cid]] += 1
    stats.class_instances = class_counts
    return stats


def validate_source_dataset(source_root: Path | str) -> ValidationReport:
    """Full structural + taxonomy + pairing + YOLO-seg validation (read-only)."""
    root = Path(source_root)
    issues: list[ValidationIssue] = []
    report = ValidationReport(ok=False, source_root=str(root.resolve()) if root.exists() else str(root))

    if not root.is_dir():
        _issue(issues, "error", "source_missing", f"source root does not exist: {root}")
        report.issues = issues
        return report

    yaml_path = root / "dataset.yaml"
    if not yaml_path.is_file():
        _issue(issues, "error", "yaml_missing", "dataset.yaml not found", path=str(yaml_path))
        report.issues = issues
        return report

    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    for d in required:
        if not d.is_dir():
            _issue(issues, "error", "structure", f"missing directory: {d.relative_to(root)}", path=str(d))

    try:
        _raw, names = _parse_dataset_yaml(yaml_path)
    except CubicasaImportError as e:
        _issue(issues, "error", "yaml_parse", str(e), path=str(yaml_path))
        report.issues = issues
        return report

    report.source_yaml_names = names
    report.class_map_ok = _assert_class_taxonomy(names, issues)

    if any(i.severity == "error" and i.code == "structure" for i in issues):
        report.issues = issues
        report.ok = False
        return report

    report.train = _scan_split(root / "images" / "train", root / "labels" / "train", "train", issues)
    report.val = _scan_split(root / "images" / "val", root / "labels" / "val", "val", issues)

    train_set = set(report.train.stems)
    val_set = set(report.val.stems)
    # Prefer image stems for leakage (paired stems include unpaired noise)
    try:
        train_img = set(_list_images(root / "images" / "train"))
        val_img = set(_list_images(root / "images" / "val"))
    except CubicasaImportError as e:
        _issue(issues, "error", "leakage_listing", str(e))
        train_img, val_img = train_set, val_set

    overlap = sorted(train_img & val_img)
    if overlap:
        _issue(
            issues,
            "error",
            "train_val_leakage",
            f"{len(overlap)} stem(s) appear in both train and val "
            f"(e.g. {overlap[:5]})",
        )

    report.furniture_instances = (
        report.train.class_instances.get(FURNITURE_NAME, 0)
        + report.val.class_instances.get(FURNITURE_NAME, 0)
    )
    report.electric_appliance_instances = (
        report.train.class_instances.get(ELECTRIC_APPLIANCE_NAME, 0)
        + report.val.class_instances.get(ELECTRIC_APPLIANCE_NAME, 0)
    )

    if report.furniture_instances == 0:
        _issue(issues, "warning", "no_furniture", "No Furniture (11) instances found")
    if report.electric_appliance_instances == 0:
        _issue(
            issues,
            "warning",
            "no_electric_appliance",
            "No ElectricAppliance (14) instances found",
        )

    report.issues = issues
    report.ok = len(report.errors()) == 0
    return report


def write_hci_dataset_yaml(dest_root: Path) -> Path:
    """Write Ultralytics dataset.yaml with frozen HCI 17-class names (no remapping)."""
    dest_root = Path(dest_root)
    lines = [
        f"path: {dest_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    out = dest_root / "dataset.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _file_sha256(path: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _copy_split(
    src_images: Path,
    src_labels: Path,
    dst_images: Path,
    dst_labels: Path,
    stems: list[str],
    image_map: dict[str, Path],
    label_map: dict[str, Path],
) -> tuple[int, int]:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    n_img = n_lbl = 0
    for stem in stems:
        if stem not in image_map or stem not in label_map:
            continue
        src_i = image_map[stem]
        src_l = label_map[stem]
        dst_i = dst_images / src_i.name
        dst_l = dst_labels / (stem + ".txt")
        if dst_i.exists() or dst_l.exists():
            raise CubicasaImportError(f"destination already has file for stem {stem}")
        shutil.copy2(src_i, dst_i)
        # Exact label text preservation (no coordinate rewrite)
        text = src_l.read_text(encoding="utf-8")
        dst_l.write_text(text, encoding="utf-8")
        if dst_l.read_text(encoding="utf-8") != text:
            raise CubicasaImportError(f"label bytes changed after copy: {stem}")
        n_img += 1
        n_lbl += 1
    return n_img, n_lbl


def derive_version_id(source_root: Path, explicit: str | None = None) -> str:
    if explicit:
        v = explicit.strip()
        if not v or any(c in v for c in '\\/:*?"<>|'):
            raise CubicasaImportError(f"invalid version id: {explicit!r}")
        return v
    name = source_root.resolve().name
    if not name:
        raise CubicasaImportError("cannot derive version id from source path")
    return name


def import_cubicasa_dataset(
    source_root: Path | str,
    *,
    version: str | None = None,
    dest_root: Path | str | None = None,
    imports_base: Path | str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> ImportResult:
    """
    Validate + copy a Cubicasa freeze into an H-managed versioned dataset.

    force=False (default): refuse if destination version already exists.
    """
    src = Path(source_root)
    report = validate_source_dataset(src)
    if not report.ok:
        raise CubicasaImportError(
            "source validation failed:\n"
            + "\n".join(f"  [{i.code}] {i.message}" for i in report.errors()[:30])
        )

    ver = derive_version_id(src, version)
    if dest_root is not None:
        dest = Path(dest_root)
    else:
        dest = imported_dataset_dir(ver, imports_base)

    if dest.exists() and any(dest.iterdir()) and not force:
        raise CubicasaImportError(
            f"import version already exists (refusing overwrite): {dest}. "
            "Pass a new --version or force=True only after explicit review."
        )

    yaml_out = dest / "dataset.yaml"
    manifest_path = dest / "manifest.json"

    if dry_run:
        return ImportResult(
            ok=True,
            version=ver,
            dest_root=str(dest),
            dataset_yaml=str(yaml_out),
            manifest_path=str(manifest_path),
            validation=report,
            skipped_copy=True,
        )

    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    train_images = _list_images(src / "images" / "train")
    train_labels = _list_labels(src / "labels" / "train")
    val_images = _list_images(src / "images" / "val")
    val_labels = _list_labels(src / "labels" / "val")

    paired_train = sorted(set(train_images) & set(train_labels))
    paired_val = sorted(set(val_images) & set(val_labels))

    _copy_split(
        src / "images" / "train",
        src / "labels" / "train",
        dest / "images" / "train",
        dest / "labels" / "train",
        paired_train,
        train_images,
        train_labels,
    )
    _copy_split(
        src / "images" / "val",
        src / "labels" / "val",
        dest / "images" / "val",
        dest / "labels" / "val",
        paired_val,
        val_images,
        val_labels,
    )

    write_hci_dataset_yaml(dest)

    # Spot-check polygon preservation on a few files
    for stem in paired_train[:3] + paired_val[:3]:
        split = "train" if stem in train_labels else "val"
        src_l = (src / "labels" / split / f"{stem}.txt")
        dst_l = dest / "labels" / split / f"{stem}.txt"
        if src_l.read_text(encoding="utf-8") != dst_l.read_text(encoding="utf-8"):
            raise CubicasaImportError(f"polygon preservation failed for {stem}")

    src_yaml_hash = _file_sha256(src / "dataset.yaml")
    manifest = {
        "format_version": 1,
        "source": "cubicasa_converter",
        "source_kind": "frozen_yolo_seg_dataset",
        "source_path_at_import": str(src.resolve()),
        "source_run_id": src.resolve().name,
        "import_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hci_taxonomy": {name: CLASS_IDS[name] for name in CLASS_NAMES},
        "furniture_class_id": FURNITURE_ID,
        "electric_appliance_class_id": ELECTRIC_APPLIANCE_ID,
        "nc": len(CLASS_NAMES),
        "source_dataset_yaml_sha256": src_yaml_hash,
        "counts": {
            "train_images": len(paired_train),
            "val_images": len(paired_val),
            "train_labels": len(paired_train),
            "val_labels": len(paired_val),
            "furniture_instances": report.furniture_instances,
            "electric_appliance_instances": report.electric_appliance_instances,
            "train_class_instances": report.train.class_instances,
            "val_class_instances": report.val.class_instances,
        },
        "validation": report.to_dict(),
        "notes": [
            "Source Cubicasa files were not modified.",
            "Class IDs were not remapped.",
            "Train/val splits preserved; no train=val leakage introduced.",
            "Runtime H must use this imported artifact, not C:\\cubicasa_converter.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ImportResult(
        ok=True,
        version=ver,
        dest_root=str(dest.resolve()),
        dataset_yaml=str((dest / "dataset.yaml").resolve()),
        manifest_path=str(manifest_path.resolve()),
        validation=report,
        skipped_copy=False,
    )


def resolve_training_yaml(version: str, root: Path | str | None = None) -> Path:
    """Resolve an imported dataset.yaml for training (H-managed only)."""
    path = imported_dataset_yaml(version, root)
    if not path.is_file():
        raise CubicasaImportError(
            f"imported dataset not found for version={version!r}: {path}. "
            "Run scripts/import_cubicasa_dataset.py first."
        )
    # Re-validate taxonomy on the H artifact
    _raw, names = _parse_dataset_yaml(path)
    issues: list[ValidationIssue] = []
    if not _assert_class_taxonomy(names, issues):
        raise CubicasaImportError(
            "imported dataset.yaml taxonomy invalid:\n"
            + "\n".join(i.message for i in issues)
        )
    return path
