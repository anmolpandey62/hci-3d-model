#!/usr/bin/env python3
"""
P5B — Conservative complex/fragmented wall centerline recovery via skeletonization.

Fallback only: STRIP walls keep OBB/PCA. Activates for COMPLEX / BRANCHED / FRAGMENTED
(and STRIP_NOISY) masks. Never bridges empty gaps between disconnected mask components.
Never uses opening geometry.
"""
from __future__ import annotations

import math
import os
from typing import Any

import cv2
import numpy as np

# --- Feature flag -------------------------------------------------------------


def p5b_enabled(default: bool = False) -> bool:
    """HCI_WALL_P5B=0|1. Default OFF until golden gates pass."""
    v = os.environ.get("HCI_WALL_P5B", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return bool(default)


P5B_CONFIDENCE_MIN = 0.45
P5B_MIN_PATH_PX = 8.0
P5B_MIN_WIDTH_SAMPLES = 3
P5B_SPUR_WIDTH_FRAC = 0.5
P5B_MASK_SUPPORT_MIN = 0.85
P5B_WIDTH_IQR_FRAC_MAX = 0.55


# --- Classifier ---------------------------------------------------------------


def classify_wall_mask(
    pts_px: list[list[float]],
    *,
    length_px: float | None = None,
    thickness_px: float | None = None,
    skeleton_junctions: int | None = None,
) -> dict[str, Any]:
    """
    Conservative wall-mask classifier.
    Categories: INVALID, SHORT_STRIP, STRIP, STRIP_NOISY, FRAGMENTED, COMPLEX, BRANCHED
    """
    if not pts_px or len(pts_px) < 3:
        return {
            "classifier": "INVALID",
            "aspect": 0.0,
            "n_verts": 0,
            "area_px": 0.0,
            "reasons": ["too_few_points"],
        }

    arr = np.asarray(pts_px, dtype=np.float32)
    area = float(abs(cv2.contourArea(arr.reshape(-1, 1, 2))))
    peri = float(cv2.arcLength(arr.reshape(-1, 1, 2), True))
    n_verts = int(len(pts_px))
    rect = cv2.minAreaRect(arr.reshape(-1, 1, 2))
    (w, h) = rect[1]
    long_s, short_s = (float(max(w, h)), float(min(w, h)))
    aspect = long_s / max(short_s, 1e-6)
    if length_px is None:
        length_px = long_s
    if thickness_px is None:
        thickness_px = max(short_s, 1.0)

    compactness = (4.0 * math.pi * area / (peri * peri)) if peri > 1e-6 else 0.0
    vert_density = n_verts / max(peri, 1.0)
    reasons: list[str] = []

    if area < 4.0 or long_s < 3.0:
        return {
            "classifier": "INVALID",
            "aspect": aspect,
            "n_verts": n_verts,
            "area_px": area,
            "length_px": float(length_px),
            "thickness_px": float(thickness_px),
            "compactness": compactness,
            "vert_density": vert_density,
            "skeleton_junctions": skeleton_junctions,
            "reasons": ["tiny_area"],
        }

    jn = int(skeleton_junctions or 0)

    # High-aspect strips: never use skeleton-junction heuristics (morph thinning
    # creates spurious deg-3 pixels on clean rectangles).
    if aspect >= 4.0:
        if vert_density > 0.12 or n_verts >= 28:
            cat = "STRIP_NOISY"
            reasons.append("high_aspect_noisy_contour")
        else:
            cat = "STRIP"
            reasons.append("clean_high_aspect")
    elif float(length_px) < 2.0 * float(thickness_px) and aspect < 3.0:
        cat = "SHORT_STRIP"
        reasons.append("short_relative_to_thickness")
    elif jn >= 2 and aspect < 3.0:
        cat = "BRANCHED"
        reasons.append(f"junctions={jn}")
    elif jn >= 1 and aspect < 2.5 and n_verts >= 12:
        cat = "COMPLEX"
        reasons.append(f"junctions={jn}")
    elif aspect < 2.5 and n_verts >= 16:
        cat = "COMPLEX"
        reasons.append("low_aspect_high_verts")
    elif aspect >= 2.5 and (vert_density > 0.12 or n_verts >= 24):
        cat = "FRAGMENTED"
        reasons.append("medium_aspect_fragmented_contour")
    else:
        cat = "STRIP" if aspect >= 2.5 else "SHORT_STRIP"
        reasons.append("default_strip_like")

    return {
        "classifier": cat,
        "aspect": float(aspect),
        "n_verts": n_verts,
        "area_px": area,
        "length_px": float(length_px),
        "thickness_px": float(thickness_px),
        "compactness": float(compactness),
        "vert_density": float(vert_density),
        "skeleton_junctions": jn,
        "reasons": reasons,
    }


def classifier_authorizes_p5b(classifier: str) -> bool:
    # Minority path: only clear complex/branched masks. Noisy strips keep OBB/P5A.
    return classifier in ("COMPLEX", "BRANCHED")


# --- Raster / skeleton --------------------------------------------------------


def rasterize_polygon(
    pts_px: list[list[float]],
    pad: int = 2,
) -> tuple[np.ndarray, int, int]:
    xs = [float(p[0]) for p in pts_px]
    ys = [float(p[1]) for p in pts_px]
    minx = int(math.floor(min(xs))) - pad
    maxx = int(math.ceil(max(xs))) + pad
    miny = int(math.floor(min(ys))) - pad
    maxy = int(math.ceil(max(ys))) + pad
    w = max(1, maxx - minx + 1)
    h = max(1, maxy - miny + 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    local = np.array([[p[0] - minx, p[1] - miny] for p in pts_px], dtype=np.int32)
    cv2.fillPoly(mask, [local], 1)
    return mask, minx, miny


def optional_cleanup_mask(mask: np.ndarray, thickness_px: float) -> tuple[np.ndarray, dict]:
    t = max(1.0, float(thickness_px))
    k = 3 if t >= 6.0 else 1
    info = {"morph_kernel": k, "applied": False}
    if k <= 1:
        return mask.copy(), info
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
    info["applied"] = True
    return (out > 0).astype(np.uint8), info


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(img) == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = img.copy()
    for _ in range(10000):
        eroded = cv2.erode(work, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, temp)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def skeleton_degree_map(skel: np.ndarray) -> np.ndarray:
    h, w = skel.shape
    deg = np.zeros((h, w), dtype=np.int16)
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        c = sum(1 for ny, nx in _neighbors8(y, x, h, w) if skel[ny, nx])
        deg[y, x] = c
    return deg


def count_skeleton_junctions(skel: np.ndarray) -> int:
    deg = skeleton_degree_map(skel)
    return int(np.sum((skel > 0) & (deg >= 3)))


def prune_skeleton_spurs(
    skel: np.ndarray,
    max_spur_px: float,
) -> tuple[np.ndarray, list[dict]]:
    out = skel.copy()
    records: list[dict] = []
    h, w = out.shape
    max_spur = max(1, int(math.floor(max_spur_px)))
    changed = True
    guard = 0
    while changed and guard < 5000:
        guard += 1
        changed = False
        deg = skeleton_degree_map(out)
        ys, xs = np.where((out > 0) & (deg == 1))
        for y0, x0 in zip(ys.tolist(), xs.tolist()):
            if out[y0, x0] == 0:
                continue
            path = [(y0, x0)]
            prev = None
            y, x = y0, x0
            while True:
                nbrs = [
                    (ny, nx)
                    for ny, nx in _neighbors8(y, x, h, w)
                    if out[ny, nx] and (prev is None or (ny, nx) != prev)
                ]
                if len(nbrs) != 1:
                    break
                prev = (y, x)
                y, x = nbrs[0]
                path.append((y, x))
                if len(path) > max_spur + 5:
                    break
            last = path[-1]
            last_deg = sum(1 for ny, nx in _neighbors8(last[0], last[1], h, w) if out[ny, nx])
            if len(path) <= max_spur and last_deg >= 3:
                cut = path[:-1]
                for py, px in cut:
                    out[py, px] = 0
                records.append(
                    {
                        "spur_pruned": True,
                        "spur_length_px": float(len(cut)),
                        "local_width_px": float(max_spur_px / max(P5B_SPUR_WIDTH_FRAC, 1e-6)),
                    }
                )
                changed = True
    return out, records


def build_skeleton_graph(skel: np.ndarray) -> tuple[dict, list]:
    h, w = skel.shape
    deg = skeleton_degree_map(skel)
    special: dict[tuple[int, int], str] = {}
    for y, x in zip(*np.where(skel > 0)):
        d = int(deg[y, x])
        if d <= 1:
            special[(int(y), int(x))] = "endpoint"
        elif d >= 3:
            special[(int(y), int(x))] = "junction"

    nodes = {i: {"id": i, "yx": yx, "kind": kind} for i, (yx, kind) in enumerate(sorted(special.items()))}
    yx_to_id = {n["yx"]: n["id"] for n in nodes.values()}
    edges: list[dict] = []
    seen: set[tuple] = set()

    def walk(start: tuple[int, int], nxt: tuple[int, int]):
        path = [start, nxt]
        prev, cur = start, nxt
        while cur not in special:
            nbrs = [
                (ny, nx)
                for ny, nx in _neighbors8(cur[0], cur[1], h, w)
                if skel[ny, nx] and (ny, nx) != prev
            ]
            if len(nbrs) != 1:
                break
            prev, cur = cur, nbrs[0]
            path.append(cur)
            if len(path) > h * w:
                break
        return path

    for yx in special:
        y, x = yx
        for ny, nx in _neighbors8(y, x, h, w):
            if not skel[ny, nx]:
                continue
            path = walk(yx, (ny, nx))
            end = path[-1]
            if end not in special:
                continue
            a, b = yx, end
            key = tuple(sorted((a, b))) + (len(path),)
            if key in seen:
                continue
            seen.add(key)
            if a not in yx_to_id or b not in yx_to_id:
                continue
            if a == b and len(path) < 3:
                continue
            edges.append(
                {
                    "u": yx_to_id[a],
                    "v": yx_to_id[b],
                    "pixels": path,
                    "length_px": float(max(0, len(path) - 1)),
                }
            )

    best: dict[tuple[int, int], dict] = {}
    for e in edges:
        key = tuple(sorted((e["u"], e["v"])))
        if key not in best or e["length_px"] > best[key]["length_px"]:
            best[key] = e
    edges = sorted(best.values(), key=lambda e: (-e["length_px"], e["u"], e["v"]))
    return nodes, edges


def extract_principal_paths(nodes: dict, edges: list[dict]) -> list[dict]:
    paths: list[dict] = []
    for i, e in enumerate(edges):
        u, v = nodes[e["u"]], nodes[e["v"]]
        paths.append(
            {
                "path_id": f"p{i}",
                "pixels": e["pixels"],
                "length_px": e["length_px"],
                "u_kind": u["kind"],
                "v_kind": v["kind"],
                "junction_count": int(u["kind"] == "junction") + int(v["kind"] == "junction"),
            }
        )
    paths.sort(key=lambda p: (-p["length_px"], p["path_id"]))
    for i, p in enumerate(paths):
        p["path_id"] = f"p{i}"
    return paths


def path_pixels_to_xy(pixels: list[tuple[int, int]], ox: int, oy: int) -> list[list[float]]:
    return [[float(x + ox) + 0.5, float(y + oy) + 0.5] for (y, x) in pixels]


def refine_path_pca(xy: list[list[float]]) -> tuple[list[list[float]], dict]:
    if len(xy) < 2:
        return xy, {"method": "identity", "straightness": 0.0}
    pts = np.asarray(xy, dtype=np.float64)
    chord = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    plen = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    straightness = chord / max(plen, 1e-9)
    mean = pts.mean(axis=0)
    centered = pts - mean
    if len(pts) >= 3 and straightness >= 0.92:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        t = centered @ axis
        t0, t1 = float(t.min()), float(t.max())
        p0 = (mean + axis * t0).tolist()
        p1 = (mean + axis * t1).tolist()
        return [p0, p1], {"method": "pca", "straightness": float(straightness)}
    if straightness >= 0.85:
        return [pts[0].tolist(), pts[-1].tolist()], {"method": "endpoints", "straightness": float(straightness)}
    n_keep = min(8, len(pts))
    idxs = np.linspace(0, len(pts) - 1, n_keep).astype(int)
    return [pts[i].tolist() for i in idxs], {"method": "polyline", "straightness": float(straightness)}


def validate_centerline_on_mask(
    centerline_xy: list[list[float]],
    mask: np.ndarray,
    ox: int,
    oy: int,
    pts_px: list[list[float]] | None = None,
) -> dict[str, Any]:
    if len(centerline_xy) < 2:
        return {
            "ok": False,
            "reason": "short_centerline",
            "mask_support_ratio": 0.0,
            "width_samples": [],
            "width_median_px": None,
            "width_iqr_px": None,
            "n_valid": 0,
        }

    h, w = mask.shape
    ax, ay = centerline_xy[0]
    bx, by = centerline_xy[-1]
    lx, ly = bx - ax, by - ay
    seg_len = math.hypot(lx, ly)
    if seg_len < 1e-6:
        return {
            "ok": False,
            "reason": "zero_length",
            "mask_support_ratio": 0.0,
            "width_samples": [],
            "width_median_px": None,
            "width_iqr_px": None,
            "n_valid": 0,
        }
    tx, ty = lx / seg_len, ly / seg_len
    nx, ny = -ty, tx
    contour = None
    if pts_px and len(pts_px) >= 3:
        contour = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)

    def inside_mask(x: float, y: float) -> bool:
        if contour is not None:
            return float(cv2.pointPolygonTest(contour, (float(x), float(y)), False)) >= 0.0
        mx, my = int(round(x - ox)), int(round(y - oy))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy, xx = my + dy, mx + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    return True
        return False

    def ray_half(x: float, y: float, dx: float, dy: float) -> float | None:
        max_r, step = 80.0, 0.5
        r, prev = step, 0.0
        while r <= max_r:
            px, py = x + dx * r, y + dy * r
            if contour is not None:
                if float(cv2.pointPolygonTest(contour, (float(px), float(py)), False)) < 0.0:
                    return float(0.5 * (prev + r))
            else:
                mx, my = int(round(px - ox)), int(round(py - oy))
                if mx < 0 or my < 0 or mx >= w or my >= h or mask[my, mx] == 0:
                    return float(0.5 * (prev + r))
            prev = r
            r += step
        return None

    def ray_width(x: float, y: float) -> float | None:
        d0 = ray_half(x, y, nx, ny)
        d1 = ray_half(x, y, -nx, -ny)
        if d0 is None or d1 is None:
            return None
        return float(d0 + d1)

    n_samp = max(5, min(21, int(seg_len / 10.0)))
    ts = [0.1 + 0.8 * i / (n_samp - 1) for i in range(n_samp)] if n_samp > 1 else [0.5]
    inside = 0
    widths: list[float] = []
    for t in ts:
        sx, sy = ax + t * lx, ay + t * ly
        if inside_mask(sx, sy):
            inside += 1
            ww = ray_width(sx, sy)
            if ww is not None and ww > 0.5:
                widths.append(ww)

    support = inside / max(len(ts), 1)
    if not widths:
        return {
            "ok": False,
            "reason": "no_width_samples",
            "mask_support_ratio": float(support),
            "width_samples": [],
            "width_median_px": None,
            "width_iqr_px": None,
            "n_valid": 0,
        }
    w_arr = np.asarray(widths, dtype=np.float64)
    med = float(np.median(w_arr))
    q1, q3 = np.percentile(w_arr, [25, 75])
    iqr = float(q3 - q1)
    ok = (
        support >= P5B_MASK_SUPPORT_MIN
        and len(widths) >= P5B_MIN_WIDTH_SAMPLES
        and (iqr / max(med, 1e-6)) <= P5B_WIDTH_IQR_FRAC_MAX
    )
    reason = None
    if support < P5B_MASK_SUPPORT_MIN:
        reason = "low_mask_support"
    elif len(widths) < P5B_MIN_WIDTH_SAMPLES:
        reason = "insufficient_width_samples"
    elif (iqr / max(med, 1e-6)) > P5B_WIDTH_IQR_FRAC_MAX:
        reason = "unstable_width"
    return {
        "ok": bool(ok),
        "reason": reason,
        "mask_support_ratio": float(support),
        "width_samples": [float(x) for x in widths],
        "width_median_px": med,
        "width_iqr_px": iqr,
        "n_valid": len(widths),
    }


def compute_p5b_confidence(val: dict, refine: dict, path_len: float, classifier: str) -> float:
    support = float(val.get("mask_support_ratio") or 0)
    n_w = int(val.get("n_valid") or 0)
    med = float(val.get("width_median_px") or 0)
    iqr = float(val.get("width_iqr_px") or 0)
    straight = float(refine.get("straightness") or 0)
    width_stab = 1.0 - min(1.0, (iqr / max(med, 1e-6)))
    len_score = min(1.0, path_len / 40.0)
    class_bonus = 0.05 if classifier in ("COMPLEX", "BRANCHED", "FRAGMENTED") else 0.0
    conf = (
        0.35 * support
        + 0.25 * width_stab
        + 0.15 * min(1.0, n_w / 8.0)
        + 0.15 * straight
        + 0.10 * len_score
        + class_bonus
    )
    return float(max(0.0, min(1.0, conf)))


def recover_paths_from_mask(
    mask: np.ndarray,
    ox: int,
    oy: int,
    *,
    thickness_px: float,
    pts_px: list[list[float]] | None,
    classifier: str,
    component_id: str,
) -> tuple[list[dict], dict]:
    diag: dict[str, Any] = {
        "component_id": component_id,
        "classifier": classifier,
        "spur_records": [],
        "rejected": [],
        "accepted": [],
    }
    cleaned, morph_info = optional_cleanup_mask(mask, thickness_px)
    diag["morph"] = morph_info
    skel = morphological_skeleton(cleaned)
    n_cc, lab, stats, _ = cv2.connectedComponentsWithStats((skel > 0).astype(np.uint8), 8)
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            skel[lab == i] = 0

    spur_tol = max(2.0, P5B_SPUR_WIDTH_FRAC * float(thickness_px))
    skel, spur_recs = prune_skeleton_spurs(skel, spur_tol)
    diag["spur_records"] = spur_recs
    diag["junctions_after_prune"] = count_skeleton_junctions(skel)

    nodes, edges = build_skeleton_graph(skel)
    paths = extract_principal_paths(nodes, edges)
    results: list[dict] = []

    for p in paths:
        if p["length_px"] < P5B_MIN_PATH_PX:
            diag["rejected"].append({"path_id": p["path_id"], "reason": "tiny_path", "length_px": p["length_px"]})
            continue
        xy = path_pixels_to_xy(p["pixels"], ox, oy)
        refined, refine_info = refine_path_pca(xy)
        val = validate_centerline_on_mask(refined, cleaned, ox, oy, pts_px=pts_px)
        conf = compute_p5b_confidence(val, refine_info, p["length_px"], classifier)
        rec = {
            "path_id": p["path_id"],
            "polyline_px": refined,
            "path_length_px": float(p["length_px"]),
            "junction_count": int(p["junction_count"]),
            "width_samples": val.get("width_samples") or [],
            "width_median_px": val.get("width_median_px"),
            "width_iqr_px": val.get("width_iqr_px"),
            "mask_support_ratio": val.get("mask_support_ratio"),
            "classifier": classifier,
            "skeleton_component_id": component_id,
            "centerline_method": "skeleton_pca",
            "confidence": conf,
            "refine": refine_info,
            "u_kind": p["u_kind"],
            "v_kind": p["v_kind"],
        }
        if not val.get("ok") or conf < P5B_CONFIDENCE_MIN:
            rec["rejected_reason"] = val.get("reason") or "low_confidence"
            diag["rejected"].append(rec)
            continue
        diag["accepted"].append(rec)
        results.append(rec)

    results.sort(key=lambda r: (-float(r["confidence"]), -float(r["path_length_px"]), r["path_id"]))
    return results, diag


def segments_substantially_coincident_px(
    a0: list[float],
    a1: list[float],
    b0: list[float],
    b1: list[float],
    *,
    ang_tol_deg: float = 10.0,
    lat_tol_px: float = 8.0,
    overlap_min: float = 0.5,
) -> bool:
    def ang(p0, p1):
        return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    def adiff(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    if adiff(ang(a0, a1), ang(b0, b1)) > ang_tol_deg:
        return False
    la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    if la < 1e-6:
        return False
    ux, uy = (a1[0] - a0[0]) / la, (a1[1] - a0[1]) / la
    nx, ny = -uy, ux
    d_lat = 0.5 * (
        abs((b0[0] - a0[0]) * nx + (b0[1] - a0[1]) * ny)
        + abs((b1[0] - a0[0]) * nx + (b1[1] - a0[1]) * ny)
    )
    if d_lat > lat_tol_px:
        return False

    def proj(p):
        return (p[0] - a0[0]) * ux + (p[1] - a0[1]) * uy

    a_lo, a_hi = 0.0, la
    tb0, tb1 = proj(b0), proj(b1)
    b_lo, b_hi = min(tb0, tb1), max(tb0, tb1)
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    return inter / max(min(la, b_hi - b_lo), 1e-6) >= overlap_min


def apply_p5b_to_segments(
    segs: list[dict],
    W: int,
    H: int,
    mpp: float,
) -> tuple[list[dict], dict]:
    """
    Fallback centerline recovery on classified complex/fragmented walls.
    May split one wall into multiple path segments (L/T/X).
    Keeps existing geometry when P5B confidence is low.
    """
    diagnostics: dict[str, Any] = {
        "p5b_version": 1,
        "enabled": True,
        "activations": 0,
        "strip_skipped": 0,
        "accepted_paths": 0,
        "rejected_paths": 0,
        "segments_replaced": 0,
        "segments_split": 0,
        "segments_retained_obb": 0,
        "duplicate_skipped": 0,
        "per_wall": [],
    }
    if not segs:
        return segs, diagnostics

    out: list[dict] = []

    for s in segs:
        pts = s.get("points_px") or []
        length_px = float(s.get("length_px_pre_snap") or s.get("length_px") or 0)
        thick_px = float(s.get("thickness_px") or 2.0)

        if len(pts) >= 3:
            mask, ox, oy = rasterize_polygon(pts, pad=2)
            skel_probe = morphological_skeleton(mask) if mask.any() else mask
            jn = count_skeleton_junctions(skel_probe) if mask.any() else 0
        else:
            mask, ox, oy = np.zeros((1, 1), np.uint8), 0, 0
            jn = 0

        clf = classify_wall_mask(pts, length_px=length_px, thickness_px=thick_px, skeleton_junctions=jn)
        s["p5b_classifier"] = clf["classifier"]
        wall_diag: dict[str, Any] = {"segment_id": s["id"], "classifier": clf, "action": "keep_obb"}

        if not classifier_authorizes_p5b(clf["classifier"]):
            diagnostics["strip_skipped"] += 1
            diagnostics["segments_retained_obb"] += 1
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        diagnostics["activations"] += 1
        paths, skel_diag = recover_paths_from_mask(
            mask,
            ox,
            oy,
            thickness_px=thick_px,
            pts_px=pts,
            classifier=clf["classifier"],
            component_id=f"seg_{s['id']}",
        )
        wall_diag["skeleton"] = {
            "accepted": len(skel_diag.get("accepted") or []),
            "rejected": len(skel_diag.get("rejected") or []),
            "junctions": skel_diag.get("junctions_after_prune"),
        }
        diagnostics["rejected_paths"] += len(skel_diag.get("rejected") or [])

        if not paths:
            diagnostics["segments_retained_obb"] += 1
            wall_diag["action"] = "p5b_rejected_keep_obb"
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        kept: list[dict] = []
        for p in paths:
            dup = False
            for q in kept:
                if segments_substantially_coincident_px(
                    p["polyline_px"][0],
                    p["polyline_px"][-1],
                    q["polyline_px"][0],
                    q["polyline_px"][-1],
                    lat_tol_px=max(4.0, 0.45 * thick_px),
                ):
                    dup = True
                    diagnostics["duplicate_skipped"] += 1
                    break
            if not dup:
                kept.append(p)

        best = kept[0]
        existing_len = length_px
        new_len = math.hypot(
            best["polyline_px"][-1][0] - best["polyline_px"][0][0],
            best["polyline_px"][-1][1] - best["polyline_px"][0][1],
        )
        obb0 = s["centerline_px_pre_snap"][0]
        obb1 = s["centerline_px_pre_snap"][-1]
        same_as_obb = segments_substantially_coincident_px(
            best["polyline_px"][0],
            best["polyline_px"][-1],
            obb0,
            obb1,
            lat_tol_px=max(4.0, 0.5 * thick_px),
            overlap_min=0.7,
        )
        if same_as_obb and new_len <= existing_len * 1.05 and len(kept) == 1:
            diagnostics["segments_retained_obb"] += 1
            diagnostics["duplicate_skipped"] += 1
            wall_diag["action"] = "skeleton_duplicate_of_obb"
            wall_diag["skeleton_duplicate_candidate"] = True
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        # STRIP_NOISY / FRAGMENTED: only replace when skeleton is not shorter than OBB
        if clf["classifier"] in ("STRIP_NOISY", "FRAGMENTED") and new_len < existing_len * 0.95:
            diagnostics["segments_retained_obb"] += 1
            wall_diag["action"] = "p5b_shorter_than_obb"
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        # Never replace with a dramatically shorter path (skeleton failure)
        if new_len < existing_len * 0.5 and len(kept) == 1:
            diagnostics["segments_retained_obb"] += 1
            wall_diag["action"] = "p5b_collapsed_keep_obb"
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        # Single-path replace only when clearly longer (avoid micro-drifts that
        # cascade through P3/P5A and drop valid opening hosts).
        if len(kept) == 1 and new_len < existing_len * 1.12:
            diagnostics["segments_retained_obb"] += 1
            wall_diag["action"] = "p5b_not_longer_keep_obb"
            out.append(s)
            diagnostics["per_wall"].append(wall_diag)
            continue

        diagnostics["accepted_paths"] += len(kept)
        if len(kept) == 1:
            ns = dict(s)
            poly = [list(best["polyline_px"][0]), list(best["polyline_px"][-1])]
            ns["centerline_px_pre_snap"] = poly
            ns["polyline_px"] = [list(poly[0]), list(poly[-1])]
            ns["length_px"] = float(new_len)
            ns["length_px_pre_snap"] = float(new_len)
            ns["centerline_method"] = "skeleton_pca"
            ns["p5b_confidence"] = best["confidence"]
            ns["p5b_path_id"] = best["path_id"]
            ns["p5b_mask_support_ratio"] = best["mask_support_ratio"]
            ns["p5b_width_median_px"] = best.get("width_median_px")
            out.append(ns)
            diagnostics["segments_replaced"] += 1
            wall_diag["action"] = "replaced_centerline"
        else:
            diagnostics["segments_split"] += 1
            wall_diag["action"] = f"split_{len(kept)}_paths"
            for pi, p in enumerate(kept):
                ns = dict(s)
                plen = math.hypot(
                    p["polyline_px"][-1][0] - p["polyline_px"][0][0],
                    p["polyline_px"][-1][1] - p["polyline_px"][0][1],
                )
                poly = [list(p["polyline_px"][0]), list(p["polyline_px"][-1])]
                ns["id"] = s["id"] if pi == 0 else f"{s['id']}_p{pi}"
                ns["centerline_px_pre_snap"] = poly
                ns["polyline_px"] = [list(poly[0]), list(poly[-1])]
                ns["length_px"] = float(plen)
                ns["length_px_pre_snap"] = float(plen)
                ns["centerline_method"] = "skeleton_pca"
                ns["p5b_confidence"] = p["confidence"]
                ns["p5b_path_id"] = p["path_id"]
                ns["p5b_mask_support_ratio"] = p["mask_support_ratio"]
                ns["merged_from"] = [s["id"]] if pi == 0 else [s["id"], ns["id"]]
                out.append(ns)

        diagnostics["per_wall"].append(wall_diag)

    def _id_key(sid: str):
        s = str(sid)
        if s.startswith("w"):
            core = s[1:].split("_")[0]
            try:
                return (0, int(core), s[1 + len(core) :])
            except Exception:
                return (1, s)
        return (2, s)

    out.sort(key=lambda seg: _id_key(seg["id"]))
    diagnostics["output_segment_count"] = len(out)
    diagnostics["input_segment_count"] = len(segs)
    return out, diagnostics
