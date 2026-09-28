#!/usr/bin/env python3
"""P2 golden replay: thickness before/after + topology/opening invariance."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

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
P1_BASE = ROOT / "_p1_artifacts" / "p1_baseline_table.json"

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


def topo_fingerprint(graph):
    return {
        "walls": graph["gt_wall_count"],
        "segs": len(graph["segments"]),
        "nodes": len(graph["nodes"]),
        "zero_len": sum(1 for s in graph["segments"] if float(s["length_m"]) < 1e-4),
        "polylines": {
            s["id"]: [[round(p[0], 6), round(p[1], 6)] for p in s["polyline_m"]]
            for s in sorted(graph["segments"], key=lambda x: x["id"])
        },
        "node_xy": {
            n["id"]: (round(n["x_px"], 4), round(n["y_px"], 4))
            for n in sorted(graph["nodes"], key=lambda x: x["id"])
        },
    }


def opening_fingerprint(omap):
    return {
        "hosts": sorted(
            (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in omap["mappings"]
        ),
        "unmapped": sorted(omap["unmapped"]),
        "door_mapped": sum(1 for m in omap["mappings"] if m["opening_type"] == "door"),
        "win_mapped": sum(1 for m in omap["mappings"] if m["opening_type"] == "window"),
    }


def run_plan(stem: str, scale_confidence: str = "low"):
    lbl = LABEL_DIR / f"{stem}.txt"
    img_path = find_image(stem)
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    mpp, mpp_source = resolve_mpp(DATASET, stem, DEFAULT_MPP)
    walls = yolo_polys(lbl, CLASS_WALL, W, H)
    doors = yolo_polys(lbl, CLASS_DOOR, W, H)
    wins = yolo_polys(lbl, CLASS_WINDOW, W, H)

    w1: list = []
    g1 = build_wall_graph(walls, W, H, float(mpp), w1, scale_confidence=scale_confidence)
    o1 = associate_openings(g1, doors, wins, H, float(mpp), w1)
    fp1 = stable_geometry_fingerprint(g1, o1)

    w2: list = []
    g2 = build_wall_graph(walls, W, H, float(mpp), w2, scale_confidence=scale_confidence)
    o2 = associate_openings(g2, doors, wins, H, float(mpp), w2)
    fp2 = stable_geometry_fingerprint(g2, o2)

    tdiag = g1["diagnostics"]["thickness"]
    methods = Counter(d["thickness_method"] for d in tdiag)
    finals = [d["thickness_m"] for d in tdiag]
    global_fb = sum(1 for d in tdiag if d.get("global_fallback_used"))
    plan_fb = sum(1 for d in tdiag if d.get("plan_median_used"))
    obb_n = methods.get("obb", 0)
    perp_n = methods.get("perpendicular_median", 0)
    clipped_n = sum(1 for d in tdiag if d.get("clipped"))
    invalid_samples = sum(int(d.get("rejected_sample_count") or 0) for d in tdiag)
    sample_meds = [
        d["median_sample_width_px"] for d in tdiag if d.get("median_sample_width_px") is not None
    ]

    return {
        "stem": stem,
        "W": W,
        "H": H,
        "mpp": mpp,
        "mpp_source": mpp_source,
        "doors": len(doors),
        "windows": len(wins),
        "deterministic": fp1 == fp2,
        "fingerprint": fp1,
        "topo": topo_fingerprint(g1),
        "openings": opening_fingerprint(o1),
        "thickness": {
            "median": median(finals),
            "min": min(finals) if finals else None,
            "max": max(finals) if finals else None,
            "global_fallback_count": global_fb,
            "global_fallback_pct": round(100.0 * global_fb / len(tdiag), 2) if tdiag else 0,
            "plan_median_count": plan_fb,
            "plan_median_pct": round(100.0 * plan_fb / len(tdiag), 2) if tdiag else 0,
            "obb_count": obb_n,
            "obb_pct": round(100.0 * obb_n / len(tdiag), 2) if tdiag else 0,
            "perp_count": perp_n,
            "perp_pct": round(100.0 * perp_n / len(tdiag), 2) if tdiag else 0,
            "method_distribution": dict(methods),
            "clipped_count": clipped_n,
            "invalid_sample_count": invalid_samples,
            "sample_width_median_px": median(sample_meds),
            "sample_width_min_px": min(sample_meds) if sample_meds else None,
            "sample_width_max_px": max(sample_meds) if sample_meds else None,
        },
        "door_map_pct": (
            round(100.0 * opening_fingerprint(o1)["door_mapped"] / len(doors), 2) if doors else None
        ),
        "window_map_pct": (
            round(100.0 * opening_fingerprint(o1)["win_mapped"] / len(wins), 2) if wins else None
        ),
        "hosts": opening_fingerprint(o1)["hosts"],
        "unmapped": opening_fingerprint(o1)["unmapped"],
    }


def main():
    out_dir = ROOT / "_p2_artifacts"
    out_dir.mkdir(exist_ok=True)

    p1_rows = {}
    if P1_BASE.is_file():
        for r in json.loads(P1_BASE.read_text(encoding="utf-8")):
            p1_rows[r["stem"]] = r

    expected_openings = {
        "cubi_hqa_9333_20260811_140939_043808": {
            "door_mapped": 3,
            "win_mapped": 9,
            "unmapped": [],
            "require_door_hosts_w2_t1": True,
        },
        "1 BHK ": {"door_mapped": 0, "win_mapped": 0, "unmapped_count": 4},
        "1 BHK HOUSE ": {"door_mapped": 0, "win_mapped": 0, "unmapped_count": 2},
    }

    results = []
    gate_failures = []
    for stem in GOLDEN:
        r = run_plan(stem)
        results.append(r)
        print(
            f"OK {stem!r}: med={r['thickness']['median']:.4f} "
            f"fb_global={r['thickness']['global_fallback_count']} "
            f"methods={r['thickness']['method_distribution']} det={r['deterministic']}"
        )
        if not r["deterministic"]:
            gate_failures.append(f"{stem}: non-deterministic")

        exp = expected_openings.get(stem)
        if exp:
            if "door_mapped" in exp and r["openings"]["door_mapped"] != exp["door_mapped"]:
                gate_failures.append(
                    f"{stem}: door_mapped {r['openings']['door_mapped']} != {exp['door_mapped']}"
                )
            if "win_mapped" in exp and r["openings"]["win_mapped"] != exp["win_mapped"]:
                gate_failures.append(
                    f"{stem}: win_mapped {r['openings']['win_mapped']} != {exp['win_mapped']}"
                )
            if "unmapped_count" in exp and len(r["unmapped"]) != exp["unmapped_count"]:
                gate_failures.append(
                    f"{stem}: unmapped {len(r['unmapped'])} != {exp['unmapped_count']}"
                )
            if exp.get("require_door_hosts_w2_t1"):
                doors = [h for h in r["hosts"] if h[0].startswith("c2_")]
                for oid, host, t in doors:
                    if host != "w2" or abs(t - 1.0) > 1e-6:
                        gate_failures.append(
                            f"{stem}: door {oid} host/t = {host}/{t} (expected w2/1.0)"
                        )

        p1 = p1_rows.get(stem)
        if p1:
            if r["topo"]["walls"] != p1["walls"]:
                gate_failures.append(f"{stem}: wall count changed")
            if r["topo"]["segs"] != p1["segments"]:
                gate_failures.append(f"{stem}: segment count changed")
            if r["topo"]["nodes"] != p1["nodes"]:
                gate_failures.append(f"{stem}: node count changed")
            if r["topo"]["zero_len"] != p1["zero_length_count"]:
                gate_failures.append(f"{stem}: zero-length count changed")

    cubi = next(r for r in results if "cubi" in r["stem"])
    cubi_med = cubi["thickness"]["median"]
    if cubi_med is None or not (0.29 * 0.85 <= cubi_med <= 0.29 * 1.15):
        gate_failures.append(f"Cubi median {cubi_med} outside ±15% of 0.29")

    for stem in ("cubi_hqa_9333_20260811_140939_043808",):
        dbg = DATASET / "output" / "_debug" / stem / f"{stem}_opening_wall_map.json"
        if dbg.is_file():
            om = json.loads(dbg.read_text(encoding="utf-8"))
            old_hosts = sorted(
                (m["opening_id"], m["host_wall_id"], round(float(m["t"]), 6)) for m in om["mappings"]
            )
            new = next(r for r in results if r["stem"] == stem)
            if old_hosts != new["hosts"]:
                gate_failures.append(f"{stem}: opening hosts/t changed vs pre-P1 debug")

    report = {
        "results": [{k: v for k, v in r.items() if k != "fingerprint"} for r in results],
        "gate_failures": gate_failures,
        "all_gates_passed": len(gate_failures) == 0,
        "p1_comparison": [],
    }
    for r in results:
        p1 = p1_rows.get(r["stem"], {})
        report["p1_comparison"].append(
            {
                "stem": r["stem"],
                "old_fallback": p1.get("fallback_count"),
                "old_fallback_pct": p1.get("fallback_pct"),
                "new_global_fallback": r["thickness"]["global_fallback_count"],
                "new_global_fallback_pct": r["thickness"]["global_fallback_pct"],
                "old_median": (p1.get("final_thickness") or {}).get("median"),
                "new_median": r["thickness"]["median"],
                "method_distribution": r["thickness"]["method_distribution"],
                "old_walls": p1.get("walls"),
                "new_walls": r["topo"]["walls"],
                "old_segs": p1.get("segments"),
                "new_segs": r["topo"]["segs"],
                "old_nodes": p1.get("nodes"),
                "new_nodes": r["topo"]["nodes"],
                "old_zero_length": p1.get("zero_length_count"),
                "new_zero_length": r["topo"]["zero_len"],
                "old_door_map_pct": p1.get("door_map_pct"),
                "new_door_map_pct": r["door_map_pct"],
                "old_window_map_pct": p1.get("window_map_pct"),
                "new_window_map_pct": r["window_map_pct"],
                "thickness_detail": r["thickness"],
            }
        )

    (out_dir / "p2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nGATE FAILURES:", gate_failures or "NONE")
    print("Wrote", out_dir / "p2_report.json")
    return 0 if not gate_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
