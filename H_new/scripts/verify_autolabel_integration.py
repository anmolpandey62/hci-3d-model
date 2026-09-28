#!/usr/bin/env python3
"""Run one-image autolabel integration smoke test (no server required)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HCI_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HCI_DIR.parent
sys.path.insert(0, str(HCI_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from auto_label import draw_labelled_image, generate_labels  # noqa: E402
from config.classes import CLASS_IDS  # noqa: E402
from logic.yolo_inference import find_model_path  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "gdrive_dataset"
PILOT_NAME = "244a80fe000e5b8728c17211b2b7525d.jpg"
PILOT_STEM = Path(PILOT_NAME).stem


def main() -> int:
    report: dict = {}

    model_path = find_model_path()
    report["model_path"] = model_path
    report["model_exists"] = bool(model_path and os.path.exists(model_path))
    if not report["model_exists"]:
        print(json.dumps(report, indent=2))
        return 1

    img_path = DATASET_DIR / "images_raw" / PILOT_NAME
    if not img_path.exists():
        print(f"ERROR: missing {img_path}", file=sys.stderr)
        return 1

    label_lines, img, labelled = generate_labels(str(img_path), None)
    report["label_line_count"] = len(label_lines)
    report["walls"] = len(labelled.get("Wall", []))
    report["doors"] = len(labelled.get("Door", []))
    report["windows"] = len(labelled.get("Window", []))
    report["skip_reason"] = labelled.get("_skip_reason", "")

    img_out = DATASET_DIR / "images" / "train"
    lbl_out = DATASET_DIR / "labels" / "train"
    mark_out = DATASET_DIR / "marked"
    for d in (img_out, lbl_out, mark_out):
        d.mkdir(parents=True, exist_ok=True)

    if not label_lines:
        print(json.dumps(report, indent=2))
        return 2

    shutil.copy2(str(img_path), str(img_out / PILOT_NAME))
    (lbl_out / f"{PILOT_STEM}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    marked_path = mark_out / f"{PILOT_STEM}_labelled.jpg"
    draw_labelled_image(img, labelled, str(marked_path))

    report["labels_file"] = str(lbl_out / f"{PILOT_STEM}.txt")
    report["marked_file"] = str(marked_path)
    report["labels_file_exists"] = (lbl_out / f"{PILOT_STEM}.txt").exists()
    report["marked_file_exists"] = marked_path.exists()
    report["marked_file_bytes"] = marked_path.stat().st_size if marked_path.exists() else 0

    # Rebuild contours like _load_existing_labels
    h, w = img.shape[:2]
    rebuilt: dict = {}
    for line in label_lines:
        parts = line.split()
        cid = int(parts[0])
        cls = {v: k for k, v in CLASS_IDS.items()}.get(cid, f"cls{cid}")
        coords = list(map(float, parts[1:]))
        pts = []
        for k in range(0, len(coords) - 1, 2):
            pts.append([int(coords[k] * w), int(coords[k + 1] * h)])
        if len(pts) >= 3:
            import numpy as np

            cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            rebuilt.setdefault(cls, []).append(cnt)

    report["correct_labels_ready"] = bool(rebuilt)
    report["rebuilt_class_counts"] = {k: len(v) for k, v in rebuilt.items()}

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
