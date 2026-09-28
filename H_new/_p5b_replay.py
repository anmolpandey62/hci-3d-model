#!/usr/bin/env python3
"""P5B golden replay — compare P5A (flag off) vs P5B (flag on)."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
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
    opening_d_max,
    stable_geometry_fingerprint,
    yolo_polys,
)
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp

DATASET = Path(r"C:\gdrive_dataset")
LABEL_DIR = DATASET / "labels" / "train"
IMAGE_DIR = DATASET / "images" / "train"
P5A = ROOT / "_p5a_artifacts" / "p5a_report.json"
P2 = ROOT / "_p2_artifacts" / "p2_report.json"
P3 = ROOT / "_p3_artifacts" / "p3_report.json"

GOLDEN = [
    "cubi_hqa_9333_20260811_140939_043808",
    "3 BHK ",
    "2 BHK ",
    "1 BHK ",
    "1 BHK HOUSE ",
    "4 BHK ",
    "1 BHK GROUND FLOOR ",
]


def find_image(stem: str):
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG"):
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run(stem: str, enable_p5b: bool):
    if enable_p5b:
        os.environ["HCI_WALL_P5B"] = "1"
    else:
        os.environ["HCI_WALL_P5B"] = "0"

    lbl = LABEL_DIR / f"{stem}.txt"
    img = cv2.imread(str(find_image(stem)))
    H, W = img.shape[:2]
    mpp, _ = resolve_mpp(DATASET, stem, DEFAULT_MPP)
    walls = yolo_polys(lbl, CLASS_WALL, W, H)
    doors = yolo_polys(lbl, CLASS_DOOR, W, H)
    wins = yolo_polys(lbl, CLASS_WINDOW, W, H)

    w1: list = []
    g1 = build_wall_graph(walls, W, H, float(mpp), w1)
    o1 = associate_openings(g1, doors, wins, H, float(mpp), w1)
    fp1 = stable_geometry_fingerprint(g1, o1)

    w2: list = []
    g2 = build_wall_graph(walls, W, H, float(mpp), w2)
    o2 = associate_openings(g2, doors, wins, H, float(mpp), w2)
    fp2 = stable_geometry_fingerprint(g2, o2)

    topo = g1["diagnostics"]["topology"]
    tdiag = g1["diagnostics"]["thickness"]
    odiag = o1["diagnostics"]["openings"]
    p5b = g1["diagnostics"].get("p5b") or {}
    lengths = [float(s["length_m"]) for s in g1["segments"]]

    return {
        "stem": stem,
        "p5b_enabled": enable_p5b,
        "walls_gt": len(walls),
        "segs": len(g1["segments"]),
        "nodes": len(g1["nodes"]),
        "zero_after": topo.get("zero_length_after", 0),
        "dup_after": topo.get("duplicate_groups_after", 0),
        "p5a_merges": topo.get("p5a_merge_count", 0),
        "p5b_activations": p5b.get("activations", 0),
        "p5b_accepted_paths": p5b.get("accepted_paths", 0),
        "p5b_rejected_paths": p5b.get("rejected_paths", 0),
        "p5b_replaced": p5b.get("segments_replaced", 0),
        "p5b_split": p5b.get("segments_split", 0),
        "p5b_retained_obb": p5b.get("segments_retained_obb", 0),
        "avg_len": (sum(lengths) / len(lengths)) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "thickness_median": median([d["thickness_m"] for d in tdiag]),
        "global_fallback": sum(1 for d in tdiag if d.get("global_fallback_used")),
        "doors": len(doors),
        "windows": len(wins),
        "door_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "door"),
        "win_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "window"),
        "endpoint_hosts": sum(
            1 for d in odiag if d.get("accepted") and str(d.get("projection_class", "")).startswith("endpoint")
        ),
        "interior_hosts": sum(1 for d in odiag if d.get("accepted") and d.get("projection_class") == "interior"),
        "unmapped": sorted(o1["unmapped"]),
        "unmapped_reasons": dict(Counter(d.get("rejection_reason") for d in odiag if not d.get("accepted"))),
        "door_maps": [
            (
                m["opening_id"],
                m["host_wall_id"],
                round(float(m["t"]), 4),
                m.get("projection_class"),
                round(float(m["distance_to_wall"]), 4),
            )
            for m in o1["mappings"]
            if m["opening_type"] == "door"
        ],
        "win_maps": [
            (
                m["opening_id"],
                m["host_wall_id"],
                round(float(m["t"]), 4),
                m.get("projection_class"),
                round(float(m["distance_to_wall"]), 4),
            )
            for m in o1["mappings"]
            if m["opening_type"] == "window"
        ],
        "hosts": sorted(
            (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in o1["mappings"]
        ),
        "deterministic": fp1 == fp2,
        "d_max_check": opening_d_max(0.2),
        "odiag": odiag,
        "p5b": p5b,
    }


def main():
    out = ROOT / "_p5b_artifacts"
    out.mkdir(exist_ok=True)
    p5a = {r["stem"]: r for r in (json.loads(P5A.read_text(encoding="utf-8")).get("results") or [])} if P5A.is_file() else {}
    p2 = {r["stem"]: r for r in (json.loads(P2.read_text(encoding="utf-8")).get("results") or [])} if P2.is_file() else {}
    p3 = {r["stem"]: r for r in (json.loads(P3.read_text(encoding="utf-8")).get("results") or [])} if P3.is_file() else {}

    results_on = []
    results_off = []
    failures = []

    # Sanity: d_max unchanged
    if abs(opening_d_max(0.2) - max(0.55, 1.25 * 0.2 + 0.10)) > 1e-9:
        failures.append("P4 d_max changed")

    for stem in GOLDEN:
        off = run(stem, enable_p5b=False)
        on = run(stem, enable_p5b=True)
        results_off.append(off)
        results_on.append(on)
        print(
            f"OK {stem!r}: OFF segs={off['segs']} doors={off['door_mapped']}/{off['doors']} | "
            f"ON segs={on['segs']} act={on['p5b_activations']} doors={on['door_mapped']}/{on['doors']} "
            f"int={on['interior_hosts']} end={on['endpoint_hosts']} "
            f"reasons={on['unmapped_reasons']} det={on['deterministic']}"
        )

        if not on["deterministic"]:
            failures.append(f"{stem}: non-deterministic with P5B ON")
        if on["zero_after"] != 0:
            failures.append(f"{stem}: zero-length={on['zero_after']}")

        p2r = p2.get(stem)
        if p2r:
            p2med = (p2r.get("thickness") or {}).get("median") or p2r.get("thickness_median")
            if p2med is not None and abs((on["thickness_median"] or 0) - p2med) > 2e-3:
                failures.append(f"{stem}: thickness median regress")

        p3r = p3.get(stem)
        if p3r and on["zero_after"] != p3r.get("zero_after", 0):
            failures.append(f"{stem}: P3 zero-length regress")

        if stem.strip() in ("1 BHK", "1 BHK HOUSE"):
            if on["door_mapped"] or on["win_mapped"]:
                failures.append(f"{stem}: HARD FAIL became mapped under P5B")
            for d in on["odiag"]:
                if not d["accepted"] and d.get("rejection_reason") != "TOO_FAR":
                    if (d.get("nearest_distance_m") or 0) > 0.8:
                        failures.append(f"{stem}: {d['opening_id']} reason {d.get('rejection_reason')}")

        if stem.strip() == "3 BHK":
            p5ar = p5a.get(stem) or off
            # Preserve prior interior door mappings for c2_1 and c2_4 on w41
            p5_hosts = {h[0]: h[1] for h in on["hosts"]}
            for oid, hid in (("c2_1", "w41"), ("c2_4", "w41")):
                if oid not in p5_hosts:
                    failures.append(f"{stem}: lost mapping {oid}")
                elif p5_hosts[oid] != hid:
                    failures.append(f"{stem}: {oid} host {hid}->{p5_hosts[oid]}")
            if on["door_mapped"] < int(p5ar.get("door_mapped") or 5):
                failures.append(
                    f"{stem}: door_mapped {on['door_mapped']} < P5A {p5ar.get('door_mapped')}"
                )

        if "cubi" in stem:
            doors = on["door_maps"]
            if len(doors) == 3 and all(
                d[1] == "w2" and abs(d[2] - 1.0) < 1e-3 and str(d[3]).startswith("endpoint") for d in doors
            ):
                failures.append(f"{stem}: revived false endpoint w2/t=1 hosts")

    report = {
        "results_p5b_on": [{k: v for k, v in r.items() if k not in ("odiag", "p5b")} for r in results_on],
        "results_p5b_off": [{k: v for k, v in r.items() if k not in ("odiag", "p5b")} for r in results_off],
        "p5b_details": {r["stem"]: r["p5b"] for r in results_on},
        "failures": failures,
        "all_gates_passed": len(failures) == 0,
        "recommendation_default_on": len(failures) == 0,
    }
    (out / "p5b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results_on:
        if "cubi" in r["stem"] or r["stem"].strip() in ("1 BHK", "1 BHK HOUSE", "3 BHK"):
            safe = r["stem"].strip().replace(" ", "_") or "x"
            (out / f"p5b_detail_{safe}.json").write_text(json.dumps(r["odiag"], indent=2), encoding="utf-8")

    print("\nFAILURES:", failures or "NONE")
    print("Wrote", out / "p5b_report.json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
