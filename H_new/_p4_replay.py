#!/usr/bin/env python3
"""P4 golden replay — opening association + P2/P3 regression gates."""
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
P3_REPORT = ROOT / "_p3_artifacts" / "p3_report.json"

GOLDEN = [
    "cubi_hqa_9333_20260811_140939_043808",
    "3 BHK ",
    "2 BHK ",
    "1 BHK ",
    "1 BHK HOUSE ",
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

    tdiag = g1["diagnostics"]["thickness"]
    topo = g1["diagnostics"]["topology"]
    odiag = o1["diagnostics"]["openings"]

    endpoint_hosts = sum(
        1
        for d in odiag
        if d.get("accepted") and str(d.get("projection_class", "")).startswith("endpoint")
    )
    interior_hosts = sum(1 for d in odiag if d.get("accepted") and d.get("projection_class") == "interior")
    reasons = Counter(d.get("rejection_reason") for d in odiag if not d.get("accepted"))

    door_maps = [
        (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6), m.get("projection_class"))
        for m in o1["mappings"]
        if m["opening_type"] == "door"
    ]
    win_maps = [
        (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6), m.get("projection_class"))
        for m in o1["mappings"]
        if m["opening_type"] == "window"
    ]

    return {
        "stem": stem,
        "walls": len(walls),
        "doors": len(doors),
        "windows": len(wins),
        "segs": len(g1["segments"]),
        "nodes": len(g1["nodes"]),
        "zero_after": topo.get("zero_length_after", 0),
        "thickness_median": median([d["thickness_m"] for d in tdiag]),
        "global_fallback": sum(1 for d in tdiag if d.get("global_fallback_used")),
        "door_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "door"),
        "win_mapped": sum(1 for m in o1["mappings"] if m["opening_type"] == "window"),
        "unmapped": sorted(o1["unmapped"]),
        "unmapped_reasons": dict(reasons),
        "endpoint_hosts": endpoint_hosts,
        "interior_hosts": interior_hosts,
        "door_maps": door_maps,
        "win_maps": win_maps,
        "hosts": sorted(
            (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in o1["mappings"]
        ),
        "deterministic": fp1 == fp2,
        "odiag": odiag,
    }


def main():
    out_dir = ROOT / "_p4_artifacts"
    out_dir.mkdir(exist_ok=True)

    p2 = {}
    if P2_REPORT.is_file():
        for r in json.loads(P2_REPORT.read_text(encoding="utf-8")).get("results") or []:
            p2[r["stem"]] = r
    p3 = {}
    if P3_REPORT.is_file():
        for r in json.loads(P3_REPORT.read_text(encoding="utf-8")).get("results") or []:
            p3[r["stem"]] = r

    results = []
    failures = []
    for stem in GOLDEN:
        r = run_plan(stem)
        results.append(r)
        print(
            f"OK {stem!r}: doors={r['door_mapped']}/{r['doors']} wins={r['win_mapped']}/{r['windows']} "
            f"endpoint={r['endpoint_hosts']} interior={r['interior_hosts']} "
            f"unmap={r['unmapped']} reasons={r['unmapped_reasons']} det={r['deterministic']}"
        )
        if not r["deterministic"]:
            failures.append(f"{stem}: non-deterministic")

        # P3 zero-length regression
        if r["zero_after"] != 0:
            failures.append(f"{stem}: zero-length={r['zero_after']}")
        p3r = p3.get(stem)
        if p3r and r["zero_after"] != p3r.get("zero_after", 0):
            failures.append(f"{stem}: zero-length changed vs P3")

        # P2 thickness regression
        p2r = p2.get(stem)
        if p2r:
            p2med = p2r.get("thickness", {}).get("median") or p2r.get("thickness_median")
            if p2med is not None and r["thickness_median"] is not None:
                if abs(r["thickness_median"] - p2med) > 1e-3:
                    failures.append(f"{stem}: thickness median drift {p2med} -> {r['thickness_median']}")
            if r["global_fallback"] != 0 and stem.strip() in ("3 BHK", "2 BHK"):
                failures.append(f"{stem}: thickness fallback regress")

        # 1 BHK / HOUSE must stay unmapped via TOO_FAR (not loosened threshold)
        if stem.strip() in ("1 BHK", "1 BHK HOUSE"):
            if r["door_mapped"] != 0 or r["win_mapped"] != 0:
                failures.append(f"{stem}: became mapped (distance loosened?)")
            for d in r["odiag"]:
                if not d["accepted"] and d.get("rejection_reason") not in ("TOO_FAR", "NO_WALLS"):
                    # allow only TOO_FAR primarily
                    if d.get("nearest_distance_m") is not None and d["nearest_distance_m"] > 0.8:
                        if d.get("rejection_reason") != "TOO_FAR":
                            failures.append(
                                f"{stem}: {d['opening_id']} reason={d.get('rejection_reason')} expected TOO_FAR"
                            )

        # Cubi: not all 3 doors on same coincident short wall at t=1.0 endpoint
        if "cubi" in stem:
            doors = [m for m in r["door_maps"]]
            if len(doors) == 3:
                hosts = {d[1] for d in doors}
                ts = {d[2] for d in doors}
                projs = {d[3] for d in doors}
                if hosts == {"w2"} and ts == {1.0} and projs == {"endpoint_end"}:
                    failures.append(f"{stem}: all 3 doors still endpoint-hosted on w2/t=1.0")
            # If mapped, prefer not 100% endpoint on identical host
            if r["door_mapped"] + r["win_mapped"] > 0:
                total = r["door_mapped"] + r["win_mapped"]
                if r["endpoint_hosts"] == total and len({h[1] for h in r["hosts"]}) == 1:
                    # check if that single host is endpoint for all
                    if all(
                        d.get("projection_class", "").startswith("endpoint")
                        for d in r["odiag"]
                        if d.get("accepted")
                    ):
                        failures.append(f"{stem}: 100% endpoint-hosted on a single wall")

    report = {
        "results": [{k: v for k, v in r.items() if k != "odiag"} for r in results],
        "failures": failures,
        "all_gates_passed": len(failures) == 0,
    }
    (out_dir / "p4_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # detailed cubi/1bhk dumps
    for r in results:
        if "cubi" in r["stem"] or r["stem"].strip() in ("1 BHK", "1 BHK HOUSE"):
            safe = r["stem"].strip().replace(" ", "_") or "x"
            (out_dir / f"p4_detail_{safe}.json").write_text(
                json.dumps(r["odiag"], indent=2), encoding="utf-8"
            )
    print("\nFAILURES:", failures or "NONE")
    print("Wrote", out_dir / "p4_report.json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
