#!/usr/bin/env python3
"""P3 golden replay — topology + mandated opening/P2 regression gates (read-only associate_openings)."""
from __future__ import annotations

import json
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
    stable_geometry_fingerprint,
    yolo_polys,
)
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp

DATASET = Path(r"C:\gdrive_dataset")
LABEL_DIR = DATASET / "labels" / "train"
IMAGE_DIR = DATASET / "images" / "train"
P2_REPORT = ROOT / "_p2_artifacts" / "p2_report.json"

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


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_plan(stem: str):
    lbl = LABEL_DIR / f"{stem}.txt"
    img_path = find_image(stem)
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    mpp, mpp_source = resolve_mpp(DATASET, stem, DEFAULT_MPP)
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
    finals = [d["thickness_m"] for d in tdiag]
    methods = Counter(d.get("thickness_method") for d in tdiag)
    hosts = sorted(
        (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in o1["mappings"]
    )
    return {
        "stem": stem,
        "W": W,
        "H": H,
        "mpp": mpp,
        "walls_gt": len(walls),
        "segs": len(g1["segments"]),
        "nodes": len(g1["nodes"]),
        "zero_after": topo.get("zero_length_after", 0),
        "zero_before": topo.get("zero_length_before", 0),
        "dup_before": topo.get("duplicate_groups_before", 0),
        "dup_after": topo.get("duplicate_groups_after", 0),
        "culled": topo.get("culled_count", 0),
        "duplicates_removed": topo.get("duplicates_removed", 0),
        "collinear_merges": len(topo.get("collinear_merges") or []),
        "T_snaps": len(topo.get("T_snaps") or []),
        "L": len(topo.get("L_junctions") or []),
        "X": len(topo.get("X_intersections") or []),
        "near_miss": len(topo.get("near_misses") or []),
        "snap_px": topo.get("snap_px"),
        "degree_distribution": topo.get("degree_distribution"),
        "thickness_median": median(finals),
        "thickness_min": min(finals) if finals else None,
        "thickness_max": max(finals) if finals else None,
        "global_fallback": sum(1 for d in tdiag if d.get("global_fallback_used")),
        "methods": dict(methods),
        "door_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "door"),
        "win_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "window"),
        "doors_n": len(doors),
        "wins_n": len(wins),
        "hosts": hosts,
        "unmapped": sorted(o1["unmapped"]),
        "deterministic": fp1 == fp2,
        "topo": topo,
    }


def main():
    out_dir = ROOT / "_p3_artifacts"
    out_dir.mkdir(exist_ok=True)

    p2 = {}
    if P2_REPORT.is_file():
        rep = json.loads(P2_REPORT.read_text(encoding="utf-8"))
        for r in rep.get("results") or []:
            p2[r["stem"]] = r
        for c in rep.get("p1_comparison") or []:
            # also has medians
            pass

    # P2 opening hosts from cubi debug (pre-P3 baseline for association)
    cubi_dbg = (
        DATASET
        / "output"
        / "_debug"
        / "cubi_hqa_9333_20260811_140939_043808"
        / "cubi_hqa_9333_20260811_140939_043808_opening_wall_map.json"
    )
    p2_cubi_hosts = None
    if cubi_dbg.is_file():
        om = json.loads(cubi_dbg.read_text(encoding="utf-8"))
        p2_cubi_hosts = sorted(
            (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in om["mappings"]
        )

    # P2 baseline topology counts from p2 report / known P1
    p2_topo = {
        "cubi_hqa_9333_20260811_140939_043808": {"walls": 34, "segs": 34, "nodes": 24, "zero": 0, "dup": 6},
        "3 BHK ": {"walls": 128, "segs": 128, "nodes": 17, "zero": 32, "dup": 16},
        "2 BHK ": {"walls": 25, "segs": 25, "nodes": 15, "zero": 5, "dup": 7},
        "1 BHK ": {"walls": 6, "segs": 6, "nodes": 9, "zero": 0, "dup": 1},
        "1 BHK HOUSE ": {"walls": 4, "segs": 4, "nodes": 8, "zero": 0, "dup": 0},
        "4 BHK ": {"walls": 5, "segs": 5, "nodes": 7, "zero": 0, "dup": 1},
        "1 BHK GROUND FLOOR ": {"walls": 3, "segs": 3, "nodes": 6, "zero": 0, "dup": 0},
    }

    results = []
    failures = []
    for stem in GOLDEN:
        r = run_plan(stem)
        results.append(r)
        print(
            f"OK {stem!r}: segs={r['segs']} nodes={r['nodes']} zero={r['zero_after']} "
            f"snap={r['snap_px']:.2f} det={r['deterministic']} unmapped={r['unmapped']}"
        )
        if r["zero_after"] != 0:
            failures.append(f"{stem}: zero-length after={r['zero_after']}")
        if not r["deterministic"]:
            failures.append(f"{stem}: non-deterministic")
        if r["global_fallback"] != 0 and stem.strip() in ("3 BHK", "2 BHK"):
            failures.append(f"{stem}: thickness global fallback {r['global_fallback']}")

        # Opening hard gates
        if "cubi" in stem:
            if r["door_mapped"] != 3 or r["win_mapped"] != 9:
                failures.append(f"{stem}: map counts door={r['door_mapped']} win={r['win_mapped']}")
            doors = [h for h in r["hosts"] if h[0].startswith("c2_")]
            for oid, host, t in doors:
                if host != "w2" or abs(t - 1.0) > 1e-6:
                    failures.append(f"{stem}: door {oid} -> {host}/{t} expected w2/1.0")
            if p2_cubi_hosts is not None and r["hosts"] != p2_cubi_hosts:
                failures.append(f"{stem}: hosts/t differ from P2 debug map")
            # thickness median ~0.285 ±15%
            med = r["thickness_median"]
            if med is None or not (0.285 * 0.85 <= med <= 0.285 * 1.15):
                failures.append(f"{stem}: thickness median {med} outside ±15% of 0.285")
        if stem == "1 BHK ":
            if r["door_mapped"] != 0 or r["win_mapped"] != 0 or len(r["unmapped"]) != 4:
                failures.append(f"{stem}: opening map changed {r['door_mapped']}/{r['win_mapped']}/{r['unmapped']}")
        if stem == "1 BHK HOUSE ":
            if r["door_mapped"] != 0 or r["win_mapped"] != 0 or len(r["unmapped"]) != 2:
                failures.append(f"{stem}: opening map changed")

        # Compare hosts to P2 in-memory if available in p2 report
        p2r = p2.get(stem)
        if p2r and "hosts" in p2r:
            def _norm(hosts, t_eps=5e-3):
                out = []
                for h in hosts:
                    oid, hid, t = h[0], h[1], float(h[2])
                    out.append((oid, hid, round(t / t_eps) * t_eps))
                return sorted(out)

            if _norm(r["hosts"]) != _norm(p2r["hosts"]):
                # host id must match exactly; t within epsilon
                p2_hosts = {(h[0], h[1]) for h in p2r["hosts"]}
                r_hosts = {(h[0], h[1]) for h in r["hosts"]}
                if p2_hosts != r_hosts:
                    failures.append(f"{stem}: hosts differ from P2 replay")
                else:
                    # t-only drift
                    for (oid, hid, t3) in r["hosts"]:
                        t2 = next(float(h[2]) for h in p2r["hosts"] if h[0] == oid)
                        if abs(t3 - t2) > 5e-3:
                            failures.append(f"{stem}: t drift {oid} {t2} vs {t3}")

    report = {
        "results": [{k: v for k, v in r.items() if k != "topo"} for r in results],
        "topology_full": {r["stem"]: r["topo"] for r in results},
        "p2_topo_baseline": p2_topo,
        "failures": failures,
        "all_gates_passed": len(failures) == 0,
    }
    (out_dir / "p3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFAILURES:", failures or "NONE")
    print("Wrote", out_dir / "p3_report.json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
