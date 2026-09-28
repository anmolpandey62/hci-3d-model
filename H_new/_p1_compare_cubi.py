#!/usr/bin/env python3
"""Compare P1 instrumented geometry vs pre-existing Cubi debug JSON (pre-P1 artifact)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(r"C:\H")
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (
    CLASS_DOOR,
    CLASS_WALL,
    CLASS_WINDOW,
    associate_openings,
    build_wall_graph,
    yolo_polys,
)
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp

DATASET = Path(r"C:\gdrive_dataset")
STEM = "cubi_hqa_9333_20260811_140939_043808"
OLD_GRAPH = (
    DATASET
    / "output"
    / "_debug"
    / STEM
    / f"{STEM}_wall_graph_m.json"
)
OLD_MAP = (
    DATASET
    / "output"
    / "_debug"
    / STEM
    / f"{STEM}_opening_wall_map.json"
)


def round_poly(poly, nd=6):
    return [[round(p[0], nd), round(p[1], nd)] for p in poly]


def main():
    img = cv2.imread(str(DATASET / "images" / "train" / f"{STEM}.png"))
    H, W = img.shape[:2]
    lbl = DATASET / "labels" / "train" / f"{STEM}.txt"
    mpp, _ = resolve_mpp(DATASET, STEM, DEFAULT_MPP)
    walls = yolo_polys(lbl, CLASS_WALL, W, H)
    doors = yolo_polys(lbl, CLASS_DOOR, W, H)
    wins = yolo_polys(lbl, CLASS_WINDOW, W, H)
    warnings: list[str] = []
    g = build_wall_graph(walls, W, H, float(mpp), warnings)
    o = associate_openings(g, doors, wins, H, float(mpp), warnings)

    old_g = json.loads(OLD_GRAPH.read_text(encoding="utf-8"))
    old_o = json.loads(OLD_MAP.read_text(encoding="utf-8"))

    diffs = []
    if len(g["segments"]) != len(old_g["segments"]):
        diffs.append(f"segment count {len(g['segments'])} vs {len(old_g['segments'])}")
    if len(g["nodes"]) != len(old_g["nodes"]):
        diffs.append(f"node count {len(g['nodes'])} vs {len(old_g['nodes'])}")

    new_by_id = {s["id"]: s for s in g["segments"]}
    old_by_id = {s["id"]: s for s in old_g["segments"]}
    for sid in sorted(set(new_by_id) | set(old_by_id)):
        if sid not in new_by_id:
            diffs.append(f"missing new {sid}")
            continue
        if sid not in old_by_id:
            diffs.append(f"extra new {sid}")
            continue
        a, b = new_by_id[sid], old_by_id[sid]
        if round(float(a["thickness_m"]), 6) != round(float(b["thickness_m"]), 6):
            diffs.append(f"{sid} thickness {a['thickness_m']} vs {b['thickness_m']}")
        if round(float(a["length_m"]), 6) != round(float(b["length_m"]), 6):
            diffs.append(f"{sid} length {a['length_m']} vs {b['length_m']}")
        if round_poly(a["polyline_m"]) != round_poly(b["polyline_m"]):
            diffs.append(f"{sid} polyline differs")
        if a.get("start_node_id") != b.get("start_node_id") or a.get("end_node_id") != b.get("end_node_id"):
            diffs.append(
                f"{sid} nodes {a.get('start_node_id')}-{a.get('end_node_id')} vs "
                f"{b.get('start_node_id')}-{b.get('end_node_id')}"
            )

    new_hosts = {
        (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in o["mappings"]
    }
    old_hosts = {
        (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in old_o["mappings"]
    }
    if new_hosts != old_hosts:
        diffs.append(f"hosts new-old={sorted(new_hosts - old_hosts)} old-new={sorted(old_hosts - new_hosts)}")
    if sorted(o["unmapped"]) != sorted(old_o.get("unmapped", [])):
        diffs.append(f"unmapped {o['unmapped']} vs {old_o.get('unmapped')}")

    # public segment keys should not gain geometry fields beyond pre-P1
    sample_new = set(g["segments"][0].keys())
    sample_old = set(old_g["segments"][0].keys())
    extra = sample_new - sample_old
    missing = sample_old - sample_new

    report = {
        "match": len(diffs) == 0,
        "diff_count": len(diffs),
        "diffs": diffs[:50],
        "extra_segment_keys": sorted(extra),
        "missing_segment_keys": sorted(missing),
        "new_has_diagnostics": "diagnostics" in g,
        "old_has_diagnostics": "diagnostics" in old_g,
    }
    print(json.dumps(report, indent=2))
    out = ROOT / "_p1_diagnostics" / "p1_cubi_before_after.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
