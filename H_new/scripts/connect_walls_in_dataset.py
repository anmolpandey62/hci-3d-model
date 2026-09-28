#!/usr/bin/env python3
"""
Make the Wall class of a YOLO-seg dataset "connected", so that a model trained on it
learns to predict walls that overlap at L / T junctions.

  python scripts/connect_walls_in_dataset.py --src D:\\hci_datasets\\cubicasa_imports\\v1 \\
                                             --dst D:\\hci_datasets\\cubicasa_imports\\v1_connected
  python scripts/connect_walls_in_dataset.py --src ... --dst ... --preview 8     # also write before/after PNGs

Source dataset is never modified.  Only class-3 (Wall) lines are rewritten; every other
class line is copied unchanged and in the same order.  Then train on --dst (for example with
/api/train_imported_dataset or `yolo segment train data=<dst>/dataset.yaml`).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.classes import CLASS_IDS  # noqa: E402
from logic.wall_connect import connect_yolo_label_lines  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WALL_ID = CLASS_IDS["Wall"]


def _image_size(path: Path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        img = cv2.imread(str(path))
        return (img.shape[1], img.shape[0]) if img is not None else None


def _parse_wall_polys(lines, W, H):
    out = []
    for ln in lines:
        p = ln.split()
        if len(p) >= 7 and int(float(p[0])) == WALL_ID:
            c = list(map(float, p[1:]))
            out.append(np.array([[c[i] * W, c[i + 1] * H] for i in range(0, len(c) - 1, 2)], np.int32))
    return out


def _preview(img_path: Path, before, after, W, H, out_png: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    a, b = img.copy(), img.copy()
    for poly in _parse_wall_polys(before, W, H):
        cv2.polylines(a, [poly], True, (0, 0, 255), 2)
    for poly in _parse_wall_polys(after, W, H):
        cv2.polylines(b, [poly], True, (0, 200, 0), 2)
    cv2.imwrite(str(out_png), np.hstack([a, b]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--preview", type=int, default=0, help="write N before/after PNGs to <dst>/_connect_preview")
    ap.add_argument("--dry-run", action="store_true", help="only print the report")
    a = ap.parse_args()

    if not a.src.is_dir():
        print(f"src not found: {a.src}")
        return 2
    if a.dst.exists() and any(a.dst.iterdir()) and not a.dry_run:
        print(f"dst already exists and is not empty: {a.dst}")
        return 2

    totals = Counter()
    previews = 0
    for split in ("train", "val", "test"):
        lbl_dir, img_dir = a.src / "labels" / split, a.src / "images" / split
        if not lbl_dir.is_dir():
            continue
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS} if img_dir.is_dir() else {}
        for lbl in sorted(lbl_dir.glob("*.txt")):
            lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            img_path = images.get(lbl.stem)
            size = _image_size(img_path) if img_path else None
            totals["labels"] += 1
            if size is None:
                totals["no_image_copy_unchanged"] += 1
                new = lines
                rep = {}
            else:
                W, H = size
                new, rep = connect_yolo_label_lines(lines, W, H, WALL_ID)
                for k in ("walls", "straight_runs", "skipped_non_rect", "junctions_L", "junctions_T", "junctions_X", "walls_extended"):
                    totals[k] += rep.get(k, 0)
                if a.preview and previews < a.preview and rep.get("walls_extended"):
                    if not a.dry_run:
                        _preview(img_path, lines, new, W, H, a.dst / "_connect_preview" / f"{split}_{lbl.stem}.png")
                    previews += 1
            if not a.dry_run:
                (a.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
                (a.dst / "labels" / split / lbl.name).write_text("\n".join(new) + "\n", encoding="utf-8")
                if img_path:
                    (a.dst / "images" / split).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, a.dst / "images" / split / img_path.name)

    if not a.dry_run:
        y = a.src / "dataset.yaml"
        if y.is_file():
            txt = y.read_text(encoding="utf-8").splitlines()
            txt = [f"path: {a.dst}" if ln.startswith("path:") else ln for ln in txt]
            (a.dst / "dataset.yaml").write_text("\n".join(txt) + "\n", encoding="utf-8")
    walls = max(totals["walls"], 1)
    print("label files      :", totals["labels"])
    print("wall polygons    :", totals["walls"])
    print("  straight runs  :", totals["straight_runs"])
    print("  skipped (L/blob):", totals["skipped_non_rect"], f"({100 * totals['skipped_non_rect'] / walls:.1f}%)")
    print("junctions        : L =", totals["junctions_L"], " T =", totals["junctions_T"], " X =", totals["junctions_X"])
    print("walls extended   :", totals["walls_extended"])
    if a.dry_run:
        print("(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
