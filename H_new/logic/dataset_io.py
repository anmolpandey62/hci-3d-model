#!/usr/bin/env python3
"""
Reusable dataset filesystem helpers for HCI_2.1.

Phase 1 extraction: same paths and rules previously embedded in web/server.py.
No behavioral change — Training UI endpoints call these helpers unchanged.
Future Production Orchestrator can reuse the same functions.
"""
from __future__ import annotations

import os
from pathlib import Path

# Canonical image extensions (must match prior server.py IMG_EXTS).
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".svg"}

_DATASET_SUBDIRS = (
    ("images_raw",),
    ("images", "train"),
    ("labels", "train"),
    ("metadata",),
    ("marked",),
    ("output",),
    ("runs",),
)


def ensure_dataset_dirs(dataset_dir: Path | str) -> Path:
    """Create the standard gdrive_dataset tree if missing. Returns dataset_dir Path."""
    root = Path(dataset_dir)
    for parts in _DATASET_SUBDIRS:
        (root.joinpath(*parts)).mkdir(parents=True, exist_ok=True)
    return root


def is_allowed_image_name(filename: str | None) -> bool:
    """True if filename has an allowed image suffix (case-insensitive)."""
    if not filename:
        return False
    return Path(filename).suffix.lower() in IMG_EXTS


def images_raw_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "images_raw"


def images_train_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "images" / "train"


def labels_train_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "labels" / "train"


def metadata_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "metadata"


def marked_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "marked"


def ifc_output_dir(dataset_dir: Path | str) -> Path:
    return Path(dataset_dir) / "output"


def label_train_path(dataset_dir: Path | str, basename: str) -> Path:
    """Path to YOLO-seg label file: labels/train/{basename}.txt"""
    return labels_train_dir(dataset_dir) / f"{basename}.txt"


def safe_fs_basename(basename: str) -> str:
    """
    Windows cannot create directories/files whose names end with spaces.
    Goldens like '1 BHK HOUSE ' must keep trailing space for label lookup,
    but output/_debug paths must be stripped.
    """
    s = str(basename or "").strip()
    return s if s else "unnamed"


def ifc_file_path(dataset_dir: Path | str, basename: str) -> Path:
    """Path to generated IFC: output/{safe_basename}.ifc"""
    return ifc_output_dir(dataset_dir) / f"{safe_fs_basename(basename)}.ifc"


def ifc_debug_dir(dataset_dir: Path | str, basename: str) -> Path:
    """Path to IFC debug sidecars: output/_debug/{safe_basename}/"""
    return ifc_output_dir(dataset_dir) / "_debug" / safe_fs_basename(basename)


def save_raw_image_bytes(
    dataset_dir: Path | str,
    filename: str | None,
    data: bytes,
) -> Path | None:
    """
    Save image bytes under images_raw/{filename}.

    Returns destination Path if saved; None if extension not allowed
    (same skip rule as former upload_images loop).
    """
    if not is_allowed_image_name(filename):
        return None
    raw = images_raw_dir(dataset_dir)
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / filename
    dest.write_bytes(data)
    return dest


def list_raw_image_names(dataset_dir: Path | str) -> list[str]:
    """Sorted list of filenames in images_raw with allowed extensions."""
    raw = images_raw_dir(dataset_dir)
    if not raw.is_dir():
        return []
    return sorted(f for f in os.listdir(raw) if Path(f).suffix.lower() in IMG_EXTS)


def resolve_image_for_basename(
    dataset_dir: Path | str,
    basename: str,
    img_exts: set[str] | None = None,
) -> Path | None:
    """
    Prefer images/train, then images_raw.
    Tries each extension lower and UPPER (same as prior _resolve_image_for_basename).
    """
    exts = list(img_exts if img_exts is not None else IMG_EXTS)
    root = Path(dataset_dir)
    for folder in (root / "images" / "train", root / "images_raw"):
        if not folder.is_dir():
            continue
        for ext in exts:
            for candidate in (folder / (basename + ext), folder / (basename + ext.upper())):
                if candidate.exists():
                    return candidate
    return None
