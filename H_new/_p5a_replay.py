#!/usr/bin/env python3
"""P5A golden replay — wall-run reconstruction + existing P4 association."""
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
P2 = ROOT / "_p2_artifacts" / "p2_report.json"
P3 = ROOT / "_p3_artifacts" / "p3_report.json"
P4 = ROOT / "_p4_artifacts" / "p4_report.json"

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


def run(stem: str):
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
    lengths = [float(s["length_m"]) for s in g1["segments"]]

    return {
        "stem": stem,
        "walls_gt": len(walls),
        "segs": len(g1["segments"]),
        "nodes": len(g1["nodes"]),
        "zero_after": topo.get("zero_length_after", 0),
        "dup_after": topo.get("duplicate_groups_after", 0),
        "p5a_merges": topo.get("p5a_merge_count", 0),
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
        "odiag": odiag,
    }


def main():
    out = ROOT / "_p5a_artifacts"
    out.mkdir(exist_ok=True)
    p2 = {r["stem"]: r for r in (json.loads(P2.read_text(encoding="utf-8")).get("results") or [])} if P2.is_file() else {}
    p3 = {r["stem"]: r for r in (json.loads(P3.read_text(encoding="utf-8")).get("results") or [])} if P3.is_file() else {}
    p4 = {r["stem"]: r for r in (json.loads(P4.read_text(encoding="utf-8")).get("results") or [])} if P4.is_file() else {}

    results = []
    failures = []
    for stem in GOLDEN:
        r = run(stem)
        results.append(r)
        print(
            f"OK {stem!r}: segs={r['segs']} merges={r['p5a_merges']} "
            f"doors={r['door_mapped']}/{r['doors']} wins={r['win_mapped']}/{r['windows']} "
            f"int={r['interior_hosts']} end={r['endpoint_hosts']} "
            f"reasons={r['unmapped_reasons']} det={r['deterministic']}"
        )
        if not r["deterministic"]:
            failures.append(f"{stem}: non-deterministic")
        if r["zero_after"] != 0:
            failures.append(f"{stem}: zero-length={r['zero_after']}")

        p2r, p3r, p4r = p2.get(stem), p3.get(stem), p4.get(stem)
        if p2r:
            p2med = (p2r.get("thickness") or {}).get("median") or p2r.get("thickness_median")
            if p2med is not None and abs((r["thickness_median"] or 0) - p2med) > 2e-3:
                failures.append(f"{stem}: thickness median regress")
        if p3r and r["zero_after"] != p3r.get("zero_after", 0):
            failures.append(f"{stem}: P3 zero-length regress")

        if stem.strip() in ("1 BHK", "1 BHK HOUSE"):
            if r["door_mapped"] or r["win_mapped"]:
                failures.append(f"{stem}: became mapped")
            for d in r["odiag"]:
                if not d["accepted"] and d.get("rejection_reason") != "TOO_FAR":
                    if (d.get("nearest_distance_m") or 0) > 0.8:
                        failures.append(f"{stem}: {d['opening_id']} reason {d.get('rejection_reason')}")

        if "cubi" in stem:
            # Prefer not to revive the false all-doors-on-w2@t=1 pattern
            doors = r["door_maps"]
            if len(doors) == 3 and all(
                d[1] == "w2" and abs(d[2] - 1.0) < 1e-3 and str(d[3]).startswith("endpoint") for d in doors
            ):
                failures.append(f"{stem}: still all doors endpoint on w2/t=1")
            # Endpoint-only mapped hosts are only a hard fail if doors also false-host at t=0/1
            # on the short coincident run; unmapped ENDPOINT_ONLY for doors is acceptable when
            # no continuous wall run exists through the opening (P5B territory).

        if stem.strip() == "3 BHK" and p4r:
            p4_hosts = {h[0]: (h[1], h[2]) for h in (p4r.get("hosts") or [])}
            p5_hosts = {h[0]: h for h in r["hosts"]}
            for oid, (hid, t) in p4_hosts.items():
                if oid not in p5_hosts:
                    failures.append(f"{stem}: lost P4 interior mapping {oid}->{hid}")
                elif oid in ("c2_1", "c2_4") and p5_hosts[oid][1] != hid:
                    failures.append(f"{stem}: {oid} host changed {hid}->{p5_hosts[oid][1]}")

    report = {
        "results": [{k: v for k, v in r.items() if k != "odiag"} for r in results],
        "failures": failures,
        "all_gates_passed": len(failures) == 0,
    }
    (out / "p5a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        if "cubi" in r["stem"] or r["stem"].strip() in ("1 BHK", "1 BHK HOUSE", "3 BHK"):
            safe = r["stem"].strip().replace(" ", "_") or "x"
            (out / f"p5a_detail_{safe}.json").write_text(json.dumps(r["odiag"], indent=2), encoding="utf-8")
    print("\nFAILURES:", failures or "NONE")
    print("Wrote", out / "p5a_report.json")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
