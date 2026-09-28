#!/usr/bin/env python3
"""
READ-ONLY P5B → IFC geometry diagnostic.

Does NOT modify production modules. Writes artifacts under _p5b_ifc_diagnostic/.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\H")
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    CLASS_DOOR,
    CLASS_WALL,
    CLASS_WINDOW,
    THICK_MAX,
    THICK_MIN,
    WALL_H,
    associate_openings,
    build_wall_graph,
    write_ifc4,
    yolo_polys,
)
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp  # noqa: E402

DATASET = Path(r"C:\gdrive_dataset")
OUT = ROOT / "_p5b_ifc_diagnostic"
GOLDEN_CUBI = "cubi_hqa_9333_20260811_140939_043808"
OTHERS = ["3 BHK ", "2 BHK ", "1 BHK ", "1 BHK HOUSE "]


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG"):
        p = DATASET / "images" / "train" / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def orient_deg(p0, p1) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def seg_dump(s: dict, H: int, mpp: float) -> dict:
    poly_px = s.get("polyline_px") or []
    poly_m = s.get("polyline_m") or []
    p0m, p1m = poly_m[0], poly_m[-1]
    length_m_calc = math.hypot(p1m[0] - p0m[0], p1m[1] - p0m[1])
    # Expected m from px
    expected_m = None
    px_m_ok = None
    if len(poly_px) >= 2:
        expected_m = [
            [poly_px[0][0] * mpp, (H - poly_px[0][1]) * mpp],
            [poly_px[-1][0] * mpp, (H - poly_px[-1][1]) * mpp],
        ]
        px_m_ok = (
            abs(expected_m[0][0] - p0m[0]) < 1e-6
            and abs(expected_m[0][1] - p0m[1]) < 1e-6
            and abs(expected_m[1][0] - p1m[0]) < 1e-6
            and abs(expected_m[1][1] - p1m[1]) < 1e-6
        )
    return {
        "id": s["id"],
        "source_polygon_id": s.get("source_polygon_id"),
        "source_polygon_ids": s.get("source_polygon_ids"),
        "polyline_px": poly_px,
        "polyline_m": poly_m,
        "expected_polyline_m_from_px": expected_m,
        "px_to_m_consistent": px_m_ok,
        "thickness_px": s.get("thickness_px"),
        "thickness_m": s.get("thickness_m"),
        "thickness_m_writer_clipped": float(np.clip(float(s["thickness_m"]), THICK_MIN, THICK_MAX)),
        "length_px": s.get("length_px"),
        "length_m": s.get("length_m"),
        "length_m_from_endpoints": length_m_calc,
        "length_agree": abs(float(s.get("length_m") or 0) - length_m_calc) < 1e-4,
        "orientation_deg": orient_deg(p0m, p1m),
        "n_polyline_points": len(poly_m),
        "merged_from": s.get("merged_from"),
        "centerline_method": s.get("centerline_method"),
        "classifier": s.get("classifier"),
        "confidence": s.get("confidence"),
        "finite": all(math.isfinite(float(v)) for p in poly_m for v in p)
        and math.isfinite(float(s.get("thickness_m") or 0)),
    }


def writer_expected_world(s: dict) -> dict:
    """Replicate write_ifc4 placement math (no IFC)."""
    x0, y0 = s["polyline_m"][0]
    x1, y1 = s["polyline_m"][-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    thick = float(np.clip(s["thickness_m"], THICK_MIN, THICK_MAX))
    c, sn = math.cos(angle), math.sin(angle)
    # Local corners of profile extruded: (0,-h)..(L,h) then rotate+translate
    half = thick / 2.0
    local = [(0.0, -half), (length, -half), (length, half), (0.0, half)]
    world_xy = []
    for lx, ly in local:
        wx = x0 + lx * c - ly * sn
        wy = y0 + lx * sn + ly * c
        world_xy.append([wx, wy])
    end0 = [x0, y0]
    end1 = [x0 + length * c, y0 + length * sn]
    return {
        "segment_id": s["id"],
        "placement_origin": [float(x0), float(y0), 0.0],
        "ref_direction": [float(c), float(sn), 0.0],
        "angle_rad": angle,
        "angle_deg": math.degrees(angle),
        "length": length,
        "thick_clipped": thick,
        "world_centerline_start": end0,
        "world_centerline_end": end1,
        "world_footprint_corners_xy": world_xy,
        "world_bbox_xy": [
            min(p[0] for p in world_xy),
            min(p[1] for p in world_xy),
            max(p[0] for p in world_xy),
            max(p[1] for p in world_xy),
        ],
        "z": [0.0, WALL_H],
        "uses_polyline_only_endpoints": True,
        "n_ignored_midpoints": max(0, len(s["polyline_m"]) - 2),
    }


def parse_ifc_walls(ifc_path: Path) -> list[dict]:
    import ifcopenshell
    import ifcopenshell.util.placement

    model = ifcopenshell.open(str(ifc_path))
    walls = []
    for wall in model.by_type("IfcWall"):
        name = wall.Name or ""
        # Matrix 4x4: local → world
        try:
            m = ifcopenshell.util.placement.get_local_placement(wall.ObjectPlacement)
            mat = np.array(m, dtype=float).reshape(4, 4)
        except Exception as e:
            walls.append({"name": name, "error": str(e)})
            continue

        # Get extrusion length/width/height from representation if possible
        length = thick = height = None
        solid_pos_is_world_pl = False
        try:
            rep = wall.Representation
            for shape in rep.Representations:
                for item in shape.Items:
                    if item.is_a("IfcExtrudedAreaSolid"):
                        height = float(item.Depth)
                        # Profile polyline
                        curve = item.SweptArea.OuterCurve
                        if curve.is_a("IfcPolyline"):
                            pts = [tuple(p.Coordinates) for p in curve.Points]
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            length = max(xs) - min(xs)
                            thick = max(ys) - min(ys)
                        # Position of solid
                        if item.Position and item.Position.Location:
                            loc = tuple(item.Position.Location.Coordinates)
                            # identity at origin is OK; shared world_pl also at origin
                            solid_pos_is_world_pl = abs(loc[0]) < 1e-12 and abs(loc[1]) < 1e-12
        except Exception as e:
            pass

        # Transform local centerline endpoints (0,0,0) and (L,0,0)
        def xform(x, y, z):
            v = mat @ np.array([x, y, z, 1.0])
            return [float(v[0]), float(v[1]), float(v[2])]

        L = float(length or 0)
        half = float(thick or 0) / 2.0
        corners = [
            xform(0, -half, 0),
            xform(L, -half, 0),
            xform(L, half, 0),
            xform(0, half, 0),
            xform(0, -half, WALL_H if height is None else height),
            xform(L, -half, WALL_H if height is None else height),
            xform(L, half, WALL_H if height is None else height),
            xform(0, half, WALL_H if height is None else height),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        zs = [c[2] for c in corners]
        walls.append(
            {
                "ifc_name": name,
                "global_id": wall.GlobalId,
                "matrix_4x4": mat.tolist(),
                "origin": xform(0, 0, 0),
                "local_x_axis_end": xform(1, 0, 0),
                "world_centerline_start": xform(0, 0, 0),
                "world_centerline_end": xform(L, 0, 0),
                "profile_length": length,
                "profile_thickness": thick,
                "extrusion_height": height,
                "solid_position_at_origin": solid_pos_is_world_pl,
                "world_min_x": min(xs),
                "world_max_x": max(xs),
                "world_min_y": min(ys),
                "world_max_y": max(ys),
                "world_min_z": min(zs),
                "world_max_z": max(zs),
                "corners": corners,
            }
        )
    return walls


def gap_matrix(segments: list[dict], use_ifc_ends: dict | None = None) -> list[dict]:
    """Endpoint-to-endpoint min gaps between distinct segments."""
    ends = []
    for s in segments:
        if use_ifc_ends and s["id"] in use_ifc_ends:
            a = use_ifc_ends[s["id"]]["start"]
            b = use_ifc_ends[s["id"]]["end"]
        else:
            a = s["polyline_m"][0]
            b = s["polyline_m"][-1]
        ends.append((s["id"], a, b))
    gaps = []
    for i, (ida, a0, a1) in enumerate(ends):
        for idb, b0, b1 in ends[i + 1 :]:
            d = min(
                math.hypot(a0[0] - b0[0], a0[1] - b0[1]),
                math.hypot(a0[0] - b1[0], a0[1] - b1[1]),
                math.hypot(a1[0] - b0[0], a1[1] - b0[1]),
                math.hypot(a1[0] - b1[0], a1[1] - b1[1]),
            )
            gaps.append({"a": ida, "b": idb, "min_endpoint_gap_m": d})
    gaps.sort(key=lambda g: g["min_endpoint_gap_m"])
    return gaps


def run_stem(stem: str, write_ifc: bool) -> dict:
    os.environ["HCI_WALL_P5B"] = "1"
    lbl = DATASET / "labels" / "train" / f"{stem}.txt"
    img_path = find_image(stem)
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    mpp, mpp_source = resolve_mpp(DATASET, stem, DEFAULT_MPP)
    walls = yolo_polys(lbl, CLASS_WALL, W, H)
    doors = yolo_polys(lbl, CLASS_DOOR, W, H)
    wins = yolo_polys(lbl, CLASS_WINDOW, W, H)

    warn: list = []
    graph = build_wall_graph(walls, W, H, float(mpp), warn, scale_confidence="low")
    omap = associate_openings(graph, doors, wins, H, float(mpp), warn)

    segs = graph["segments"]
    dumps = [seg_dump(s, H, float(mpp)) for s in segs]
    writer_preds = [writer_expected_world(s) for s in segs]

    xs = [p[0] for s in segs for p in s["polyline_m"]]
    ys = [p[1] for s in segs for p in s["polyline_m"]]
    bbox = {
        "min_x": min(xs) if xs else None,
        "max_x": max(xs) if xs else None,
        "min_y": min(ys) if ys else None,
        "max_y": max(ys) if ys else None,
    }

    topo = graph["diagnostics"]["topology"]
    p5b = graph["diagnostics"].get("p5b") or {}

    # Opening audit
    opening_audit = []
    by_id = {s["id"]: s for s in segs}
    for m in omap.get("mappings") or []:
        host = by_id.get(m["host_wall_id"])
        opening_audit.append(
            {
                "opening_id": m["opening_id"],
                "type": m["opening_type"],
                "host": m["host_wall_id"],
                "t": m["t"],
                "host_exists_in_final_graph": host is not None,
                "host_polyline_m": host["polyline_m"] if host else None,
                "host_length_m": host["length_m"] if host else None,
                "distance_to_wall": m.get("distance_to_wall"),
                "projection_class": m.get("projection_class"),
            }
        )
    for uid in omap.get("unmapped") or []:
        opening_audit.append({"opening_id": uid, "type": "?", "host": None, "unmapped": True})

    graph_gaps = gap_matrix(segs)

    result = {
        "stem": stem,
        "W": W,
        "H": H,
        "mpp": float(mpp),
        "mpp_source": mpp_source,
        "scale_confidence": "low",
        "gt_walls": len(walls),
        "segment_count": len(segs),
        "node_count": len(graph["nodes"]),
        "zero_length_after": topo.get("zero_length_after"),
        "duplicate_groups_after": topo.get("duplicate_groups_after"),
        "p5a_merge_count": topo.get("p5a_merge_count"),
        "p5b": {
            "enabled": p5b.get("enabled"),
            "activations": p5b.get("activations"),
            "replaced": p5b.get("segments_replaced"),
            "split": p5b.get("segments_split"),
            "retained_obb": p5b.get("segments_retained_obb"),
        },
        "bbox_m": bbox,
        "segments": dumps,
        "writer_predicted_world": writer_preds,
        "graph_endpoint_gaps": graph_gaps[:30],
        "opening_audit": opening_audit,
        "mapping_success_rate": omap.get("mapping_success_rate"),
        "mapped": omap.get("successfully_mapped"),
        "unmapped_count": omap.get("unmapped_count"),
        "writer_contract": {
            "uses": ["segments[].id", "segments[].polyline_m[0]", "segments[].polyline_m[-1]", "segments[].thickness_m"],
            "ignores_midpoints": True,
            "assumes_straight_2pt": True,
            "placement": "IfcLocalPlacement at polyline[0], RefDirection along segment",
            "profile": "rectangle length x thickness centered on local X axis",
            "extrusion": f"Z up, depth={WALL_H}",
            "thickness_clip": [THICK_MIN, THICK_MAX],
            "y_convention_in_graph": "py_to_m = (H - y_px) * mpp  (image Y inverted once)",
        },
    }

    if write_ifc:
        ifc_path = OUT / f"{stem.strip().replace(' ', '_')}_diag.ifc"
        write_ifc4(graph, omap, ifc_path, stem.strip())
        ifc_walls = parse_ifc_walls(ifc_path)
        result["ifc_path"] = str(ifc_path)
        result["ifc_walls"] = ifc_walls

        # Map segment → IFC
        mapping = []
        ifc_by_name = {w.get("ifc_name"): w for w in ifc_walls if "ifc_name" in w}
        for pred, dump in zip(writer_preds, dumps):
            iw = ifc_by_name.get(dump["id"])
            disc = {}
            if iw and "world_centerline_start" in iw:
                ps, pe = pred["world_centerline_start"], pred["world_centerline_end"]
                is_, ie = iw["world_centerline_start"], iw["world_centerline_end"]
                # Allow reversed direction
                d_fwd = math.hypot(ps[0] - is_[0], ps[1] - is_[1]) + math.hypot(pe[0] - ie[0], pe[1] - ie[1])
                d_rev = math.hypot(ps[0] - ie[0], ps[1] - ie[1]) + math.hypot(pe[0] - is_[0], pe[1] - is_[1])
                d_end = min(d_fwd, d_rev)
                disc = {
                    "centerline_endpoint_error_m": d_end,
                    "thickness_graph": dump["thickness_m"],
                    "thickness_writer_clipped": pred["thick_clipped"],
                    "thickness_ifc_profile": iw.get("profile_thickness"),
                    "length_graph": dump["length_m_from_endpoints"],
                    "length_ifc": iw.get("profile_length"),
                    "graph_start": dump["polyline_m"][0],
                    "graph_end": dump["polyline_m"][-1],
                    "ifc_start": is_[:2],
                    "ifc_end": ie[:2],
                    "pred_start": ps,
                    "pred_end": pe,
                }
                if d_end < 0.02 and abs((iw.get("profile_length") or 0) - pred["length"]) < 0.02:
                    classif = "MATCH"
                elif d_end < 0.02:
                    classif = "PLACEMENT_OK_PROFILE_DIFF"
                else:
                    classif = "IFC_TRANSFORM_ERROR"
                disc["class"] = classif
            mapping.append(
                {
                    "segment_id": dump["id"],
                    "ifc_wall_name": dump["id"],
                    "graph": {
                        "start": dump["polyline_m"][0],
                        "end": dump["polyline_m"][-1],
                        "thickness_m": dump["thickness_m"],
                        "length_m": dump["length_m"],
                    },
                    "writer_predicted": pred,
                    "ifc_parsed": iw,
                    "discrepancy": disc,
                }
            )
        result["graph_to_ifc"] = mapping

        # IFC gaps
        ifc_ends = {}
        for m in mapping:
            iw = m.get("ifc_parsed") or {}
            if "world_centerline_start" in iw:
                ifc_ends[m["segment_id"]] = {
                    "start": iw["world_centerline_start"][:2],
                    "end": iw["world_centerline_end"][:2],
                }
        ifc_gaps = gap_matrix(segs, use_ifc_ends=ifc_ends)
        # Classify continuity
        continuity = []
        gmap = {(g["a"], g["b"]): g["min_endpoint_gap_m"] for g in graph_gaps}
        gmap.update({(g["b"], g["a"]): g["min_endpoint_gap_m"] for g in graph_gaps})
        for g in ifc_gaps[:40]:
            gg = gmap.get((g["a"], g["b"]), gmap.get((g["b"], g["a"])))
            ig = g["min_endpoint_gap_m"]
            if gg is None:
                cls = "UNKNOWN"
            elif gg > 0.3 and abs(ig - gg) < 0.05:
                cls = "GRAPH_ALREADY_GAPPED"
            elif gg <= 0.15 and ig > gg + 0.2:
                cls = "IFC_INTRODUCED_GAP"
            elif abs(ig - gg) < 0.05:
                cls = "GRAPH_ALREADY_GAPPED" if gg > 0.05 else "CONTINUOUS_BOTH"
            else:
                cls = "UNKNOWN"
            continuity.append({**g, "graph_gap_m": gg, "ifc_gap_m": ig, "class": cls})
        result["continuity_audit"] = continuity

        # Transform audit samples (3 walls)
        transform_samples = []
        for m in mapping[:5]:
            d = m["discrepancy"]
            dump = next(x for x in dumps if x["id"] == m["segment_id"])
            # Check Y inversion: image y increases down; plan y = (H-y)*mpp
            transform_samples.append(
                {
                    "id": m["segment_id"],
                    "px": dump["polyline_px"],
                    "m": dump["polyline_m"],
                    "formula": "x_m=x_px*mpp; y_m=(H-y_px)*mpp",
                    "px_to_m_ok": dump["px_to_m_consistent"],
                    "ifc_vs_pred_endpoint_err": d.get("centerline_endpoint_error_m"),
                    "xy_swap_test_err": _xy_swap_error(dump, m.get("ifc_parsed")),
                    "double_invert_y_test_err": _double_invert_error(dump, H, mpp, m.get("ifc_parsed")),
                }
            )
        result["transform_samples"] = transform_samples

    return result


def _xy_swap_error(dump, iw) -> float | None:
    if not iw or "world_centerline_start" not in iw:
        return None
    # If IFC had swapped x/y of graph endpoints
    g0, g1 = dump["polyline_m"][0], dump["polyline_m"][-1]
    swap0, swap1 = [g0[1], g0[0]], [g1[1], g1[0]]
    is_, ie = iw["world_centerline_start"][:2], iw["world_centerline_end"][:2]
    d1 = math.hypot(swap0[0] - is_[0], swap0[1] - is_[1]) + math.hypot(swap1[0] - ie[0], swap1[1] - ie[1])
    d2 = math.hypot(swap0[0] - ie[0], swap0[1] - ie[1]) + math.hypot(swap1[0] - is_[0], swap1[1] - is_[1])
    return min(d1, d2)


def _double_invert_error(dump, H, mpp, iw) -> float | None:
    if not iw or "world_centerline_start" not in iw or not dump.get("polyline_px"):
        return None
    # Wrong: y_m = y_px * mpp (no invert)
    px = dump["polyline_px"]
    wrong = [
        [px[0][0] * mpp, px[0][1] * mpp],
        [px[-1][0] * mpp, px[-1][1] * mpp],
    ]
    is_, ie = iw["world_centerline_start"][:2], iw["world_centerline_end"][:2]
    d1 = math.hypot(wrong[0][0] - is_[0], wrong[0][1] - is_[1]) + math.hypot(wrong[1][0] - ie[0], wrong[1][1] - ie[1])
    d2 = math.hypot(wrong[0][0] - ie[0], wrong[0][1] - ie[1]) + math.hypot(wrong[1][0] - is_[0], wrong[1][1] - is_[1])
    return min(d1, d2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HCI_WALL_P5B"] = "1"

    cubi = run_stem(GOLDEN_CUBI, write_ifc=True)
    (OUT / "cubi_final_graph.json").write_text(
        json.dumps({k: cubi[k] for k in cubi if k not in ("ifc_walls", "graph_to_ifc", "continuity_audit", "transform_samples", "writer_predicted_world")}, indent=2),
        encoding="utf-8",
    )
    # fuller dumps
    (OUT / "cubi_final_graph_full.json").write_text(json.dumps(cubi, indent=2), encoding="utf-8")
    (OUT / "cubi_graph_to_ifc.json").write_text(
        json.dumps(
            {
                "ifc_path": cubi.get("ifc_path"),
                "writer_contract": cubi["writer_contract"],
                "mapping": cubi.get("graph_to_ifc"),
                "continuity_audit": cubi.get("continuity_audit"),
                "transform_samples": cubi.get("transform_samples"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    others = {}
    for stem in OTHERS:
        try:
            others[stem] = run_stem(stem, write_ifc=False)
        except Exception as e:
            others[stem] = {"error": str(e)}
    (OUT / "other_goldens_graph_summary.json").write_text(
        json.dumps(
            {
                k: {
                    "segment_count": v.get("segment_count"),
                    "bbox_m": v.get("bbox_m"),
                    "mapped": v.get("mapped"),
                    "unmapped_count": v.get("unmapped_count"),
                    "segments_brief": [
                        {
                            "id": s["id"],
                            "start": s["polyline_m"][0],
                            "end": s["polyline_m"][-1],
                            "L": s["length_m_from_endpoints"],
                            "thick": s["thickness_m"],
                            "thick_clipped": s["thickness_m_writer_clipped"],
                            "px_m_ok": s["px_to_m_consistent"],
                        }
                        for s in (v.get("segments") or [])
                    ],
                    "top_gaps": (v.get("graph_endpoint_gaps") or [])[:10],
                }
                for k, v in others.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Console verdict helpers
    print("=== CUBI ===")
    print("segs", cubi["segment_count"], "bbox", cubi["bbox_m"])
    print("px_m inconsistencies", sum(1 for s in cubi["segments"] if s["px_to_m_consistent"] is False))
    print("length disagree", sum(1 for s in cubi["segments"] if not s["length_agree"]))
    classes = {}
    for m in cubi.get("graph_to_ifc") or []:
        c = (m.get("discrepancy") or {}).get("class", "?")
        classes[c] = classes.get(c, 0) + 1
    print("graph_to_ifc classes", classes)
    cont = {}
    for c in cubi.get("continuity_audit") or []:
        cont[c["class"]] = cont.get(c["class"], 0) + 1
    print("continuity", cont)
    for s in cubi["segments"]:
        print(
            f"  {s['id']}: {s['polyline_m'][0]} -> {s['polyline_m'][-1]} "
            f"L={s['length_m_from_endpoints']:.3f} t={s['thickness_m']:.3f} "
            f"clip={s['thickness_m_writer_clipped']:.3f} pts={s['n_polyline_points']}"
        )
    for m in (cubi.get("graph_to_ifc") or [])[:5]:
        d = m["discrepancy"]
        print(
            f"  MAP {m['segment_id']}: err={d.get('centerline_endpoint_error_m')} "
            f"class={d.get('class')} graphL={d.get('length_graph')} ifcL={d.get('length_ifc')}"
        )
    print("Wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
