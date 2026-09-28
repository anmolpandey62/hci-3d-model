#!/usr/bin/env python3
"""End-to-end server pipeline test (autolabel worker + correct labels load)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HCI_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HCI_DIR.parent
sys.path.insert(0, str(HCI_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

PILOT_NAME = "244a80fe000e5b8728c17211b2b7525d.jpg"
PILOT_STEM = Path(PILOT_NAME).stem
DATASET_DIR = PROJECT_ROOT / "gdrive_dataset"


def main() -> int:
    report: dict = {"steps": []}

    def step(name: str, ok: bool, detail: str = ""):
        report["steps"].append({"step": name, "ok": ok, "detail": detail})

    # Clean prior outputs for idempotent test
    for sub in ("images/train", "labels/train", "marked"):
        d = DATASET_DIR / sub
        if d.is_dir():
            for f in d.glob(f"{PILOT_STEM}*"):
                f.unlink()

    # Reset server module state
    import importlib
    import web.server as srv

    importlib.reload(srv)
    srv._analysis.clear()
    srv._log_queue.clear()

    step("model_path", bool(__import__("logic.yolo_inference", fromlist=["find_model_path"]).find_model_path()), "")

    srv._autolabel_worker([PILOT_NAME], "local")

    lbl = DATASET_DIR / "labels" / "train" / f"{PILOT_STEM}.txt"
    marked = DATASET_DIR / "marked" / f"{PILOT_STEM}_labelled.jpg"
    img_train = DATASET_DIR / "images" / "train" / PILOT_NAME

    step("labels_train", lbl.exists(), str(lbl))
    step("marked_image", marked.exists(), str(marked))
    step("images_train", img_train.exists(), str(img_train))

    srv._load_existing_labels()
    in_analysis = PILOT_STEM in srv._analysis
    step("analysis_populated", in_analysis, f"keys={len(srv._analysis)}")

    listed = srv._list_labelled_images()
    step("correct_labels_list", PILOT_STEM in listed, str(listed))

    if in_analysis:
        info = srv._analysis[PILOT_STEM]
        report["walls"] = len(info.get("labelled", {}).get("Wall", []))
        report["doors"] = len(info.get("labelled", {}).get("Door", []))
        report["windows"] = len(info.get("labelled", {}).get("Window", []))
        report["n_labels"] = info.get("n_labels", 0)

        # Save corrections smoke test
        body = {"basename": PILOT_STEM, "action": "remove", "cls_name": "Door", "idx": 1}
        resp = srv.correct_label(body)
        ok = isinstance(resp, dict) and resp.get("ok")
        step("save_corrections_remove", ok, str(resp)[:200])

        save_resp = srv.save_corrections({"basename": PILOT_STEM})
        step("save_corrections", isinstance(save_resp, dict) and save_resp.get("ok"), str(save_resp)[:200])

    report["log_tail"] = srv._log_queue[-8:]
    report["all_ok"] = all(s["ok"] for s in report["steps"])
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
