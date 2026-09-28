#!/usr/bin/env python3
"""P1 in-memory golden replay — no IFC writes."""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\H")
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    CLASS_DOOR,
    CLASS_WALL,
    CLASS_WINDOW,
    associate_openings,
    build_wall_graph,
    stable_geometry_fingerprint,
    yolo_polys,
)
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp  # noqa: E402

DATASET = Path(r"C:\gdrive_dataset")
LABEL_DIR = DATASET / "labels" / "train"
IMAGE_DIR = DATASET / "images" / "train"

GOLDEN = [
    "cubi_hqa_9333_20260811_140939_043808",
    "3 BHK ",
    "2 BHK ",
    "1 BHK ",
    "1 BHK HOUSE ",
    "4 BHK ",
    "1 BHK GROUND FLOOR ",
]


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG"):
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def summarize(stem: str) -> dict:
    lbl = LABEL_DIR / f"{stem}.txt"
    img_path = find_image(stem)
    if not lbl.is_file():
        raise FileNotFoundError(f"label missing: {lbl}")
    if img_path is None:
        raise FileNotFoundError(f"image missing for {stem!r}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"cannot read {img_path}")
    H, W = img.shape[:2]
    mpp, mpp_source = resolve_mpp(DATASET, stem, DEFAULT_MPP)

    walls = yolo_polys(lbl, CLASS_WALL, W, H)
    doors = yolo_polys(lbl, CLASS_DOOR, W, H)
    wins = yolo_polys(lbl, CLASS_WINDOW, W, H)

    # Run 1
    w1: list[str] = []
    g1 = build_wall_graph(walls, W, H, float(mpp), w1)
    o1 = associate_openings(g1, doors, wins, H, float(mpp), w1)
    fp1 = stable_geometry_fingerprint(g1, o1)

    # Run 2 (determinism)
    w2: list[str] = []
    g2 = build_wall_graph(walls, W, H, float(mpp), w2)
    o2 = associate_openings(g2, doors, wins, H, float(mpp), w2)
    fp2 = stable_geometry_fingerprint(g2, o2)

    det_ok = fp1 == fp2
    det_diff = None
    if not det_ok:
        # locate first differing field
        s1 = {s["id"]: s for s in g1["segments"]}
        s2 = {s["id"]: s for s in g2["segments"]}
        diffs = []
        for sid in sorted(set(s1) | set(s2)):
            if sid not in s1:
                diffs.append(f"seg {sid} missing in run1")
            elif sid not in s2:
                diffs.append(f"seg {sid} missing in run2")
            else:
                a, b = s1[sid], s2[sid]
                for k in ("polyline_m", "thickness_m", "length_m", "start_node_id", "end_node_id"):
                    if a.get(k) != b.get(k):
                        diffs.append(f"seg {sid}.{k}: {a.get(k)!r} vs {b.get(k)!r}")
        m1 = {(m["opening_id"], m["host_wall_id"], round(m["t"], 6)) for m in o1["mappings"]}
        m2 = {(m["opening_id"], m["host_wall_id"], round(m["t"], 6)) for m in o2["mappings"]}
        if m1 != m2:
            diffs.append(f"mappings run1={sorted(m1)} run2={sorted(m2)}")
        det_diff = diffs[:20]

    tdiag = g1["diagnostics"]["thickness"]
    topo = g1["diagnostics"]["topology"]
    odiag = o1["diagnostics"]["openings"]

    raws = [d["raw_m"] for d in tdiag]
    finals = [d["thickness_m"] for d in tdiag]
    fb = [d for d in tdiag if d["fallback_occurred"]]
    fb_reasons: dict[str, int] = {}
    for d in fb:
        fb_reasons[d["fallback_reason"] or "OTHER"] = fb_reasons.get(d["fallback_reason"] or "OTHER", 0) + 1

    door_maps = [d for d in odiag if d["class"] == "door"]
    win_maps = [d for d in odiag if d["class"] == "window"]
    door_ok = sum(1 for d in door_maps if d["accepted"])
    win_ok = sum(1 for d in win_maps if d["accepted"])

    endpoint_host = 0
    interior_host = 0
    unmapped_reasons: dict[str, int] = {}
    for d in odiag:
        if not d["accepted"]:
            r = d.get("rejection_reason") or "OTHER"
            unmapped_reasons[r] = unmapped_reasons.get(r, 0) + 1
        else:
            pc = d.get("projection_class") or ""
            if pc.startswith("endpoint"):
                endpoint_host += 1
            elif pc == "interior":
                interior_host += 1

    geo_identity = {
        "segment_ids": sorted(s["id"] for s in g1["segments"]),
        "thicknesses": {s["id"]: round(float(s["thickness_m"]), 6) for s in g1["segments"]},
        "lengths": {s["id"]: round(float(s["length_m"]), 6) for s in g1["segments"]},
        "polylines": {
            s["id"]: [[round(p[0], 6), round(p[1], 6)] for p in s["polyline_m"]] for s in g1["segments"]
        },
        "hosts": sorted(
            (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in o1["mappings"]
        ),
        "unmapped": sorted(o1["unmapped"]),
    }

    return {
        "stem": stem,
        "image": str(img_path),
        "W": W,
        "H": H,
        "mpp": mpp,
        "mpp_source": mpp_source,
        "walls": len(walls),
        "doors": len(doors),
        "windows": len(wins),
        "segments": len(g1["segments"]),
        "nodes": len(g1["nodes"]),
        "fallback_count": len(fb),
        "fallback_pct": round(100.0 * len(fb) / len(tdiag), 2) if tdiag else 0.0,
        "fallback_reasons": fb_reasons,
        "raw_thickness": {
            "median": median(raws),
            "min": min(raws) if raws else None,
            "max": max(raws) if raws else None,
        },
        "final_thickness": {
            "median": median(finals),
            "min": min(finals) if finals else None,
            "max": max(finals) if finals else None,
        },
        "zero_length_count": len(topo["zero_length_segments"]),
        "duplicate_group_count": len(topo["duplicate_coincident_groups"]),
        "snap_group_count": len(topo["snap_groups"]),
        "t_junction_count": len(topo["t_junction_candidates"]),
        "x_junction_count": len(topo["x_junction_candidates"]),
        "collinear_overlap_count": len(topo["collinear_overlap_candidates"]),
        "door_map_pct": round(100.0 * door_ok / len(door_maps), 2) if door_maps else None,
        "window_map_pct": round(100.0 * win_ok / len(win_maps), 2) if win_maps else None,
        "endpoint_host_count": endpoint_host,
        "interior_host_count": interior_host,
        "unmapped_count": o1["unmapped_count"],
        "unmapped_ids": o1["unmapped"],
        "unmapped_reasons": unmapped_reasons,
        "deterministic": det_ok,
        "determinism_diff": det_diff,
        "fingerprint": fp1,
        "geo_identity": geo_identity,
        "diagnostics": {
            "thickness": tdiag,
            "topology": topo,
            "openings": odiag,
        },
        "mappings": o1["mappings"],
    }


def main():
    out_dir = ROOT / "_p1_artifacts"
    out_dir.mkdir(exist_ok=True)
    rows = []
    full = {}
    for stem in GOLDEN:
        try:
            r = summarize(stem)
            rows.append({k: v for k, v in r.items() if k not in ("diagnostics", "geo_identity", "mappings")})
            full[stem] = r
            print(f"OK  {stem!r}: walls={r['walls']} segs={r['segments']} fb={r['fallback_count']} "
                  f"unmap={r['unmapped_count']} det={r['deterministic']}")
        except Exception as e:
            print(f"FAIL {stem!r}: {e}")
            traceback.print_exc()
            rows.append({"stem": stem, "error": str(e)})

    (out_dir / "p1_baseline_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Full dumps are large; keep per-plan slim+examples
    for stem, r in full.items():
        slim = {
            "summary": {k: v for k, v in r.items() if k not in ("diagnostics",)},
            "examples": {},
        }
        odiag = r["diagnostics"]["openings"]
        # Cubi doors on w2 / t~1
        if "cubi" in stem:
            slim["examples"]["doors_w2_or_t1"] = [
                d for d in odiag
                if d.get("accepted") and (
                    d.get("host_wall_id") == "w2"
                    or (d.get("t") is not None and (d["t"] <= 1e-6 or d["t"] >= 1.0 - 1e-6))
                )
            ]
            slim["examples"]["all_door_mappings"] = [d for d in odiag if d["class"] == "door"]
        if stem.strip() in ("1 BHK", "1 BHK HOUSE"):
            slim["examples"]["unmapped"] = [d for d in odiag if not d["accepted"]]
            slim["examples"]["all_openings"] = odiag
        if stem.strip() == "3 BHK":
            slim["examples"]["fallback_walls"] = [
                d for d in r["diagnostics"]["thickness"] if d["fallback_occurred"]
            ][:40]
            slim["examples"]["fallback_reason_counts"] = r["fallback_reasons"]
        if stem.strip() == "2 BHK":
            slim["examples"]["zero_length"] = r["diagnostics"]["topology"]["zero_length_segments"]
            slim["examples"]["duplicate_groups"] = r["diagnostics"]["topology"]["duplicate_coincident_groups"]
            slim["examples"]["snap_groups_gt1"] = [
                g for g in r["diagnostics"]["topology"]["snap_groups"] if g["size"] > 1
            ][:30]
            slim["examples"]["t_junctions"] = r["diagnostics"]["topology"]["t_junction_candidates"][:20]
        safe = stem.strip().replace(" ", "_") or "empty"
        (out_dir / f"p1_{safe}.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    # Also dump full diagnostics for the five primary goldens
    for stem in GOLDEN[:5]:
        if stem in full:
            safe = stem.strip().replace(" ", "_") or "empty"
            (out_dir / f"p1_full_{safe}.json").write_text(
                json.dumps(full[stem]["diagnostics"], indent=2), encoding="utf-8"
            )

    print("\nWrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
