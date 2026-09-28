#!/usr/bin/env python3
"""
HCI_2.1 IFC geometry pipeline (Day 3/4 library).
Builds walls + openings + doors/windows IFC4 from YOLO-seg labels.
Does NOT touch best_gdrive.pt or Hci_1.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from logic.p5b_skeleton import apply_p5b_to_segments, p5b_enabled
from logic.wall_connect import connect_wall_segments, connect_walls_enabled

# Optional IfcOpenShell — import at call time for clearer errors
WALL_H = 3.0
DOOR_H = 2.1
WIN_H = 1.2
WIN_SILL = 0.9
THICK_FALLBACK = 0.23
THICK_MIN, THICK_MAX = 0.10, 0.40  # legacy constants (P1); P2 uses soft bounds below
SNAP_PX = 20.0  # legacy P0/P1 constant — P3 uses hybrid snap (not this alone)
OPEN_MAX_DIST_M = 0.55
OPEN_PROJ_EPS = 0.08  # P4 projection interior/endpoint band

# P2 thickness policy
THICK_N_MIN_SAMPLES = 3
THICK_SOFT_HIGH_MIN_M = 0.08
THICK_SOFT_HIGH_MAX_M = 0.50
THICK_PX_MIN_LOW = 1.0
THICK_PX_MAX_FRAC_LOW = 0.08  # of min(W, H)

# P3 topology policy (approved starting parameters)
SNAP_ALPHA = 0.75
SNAP_M_MIN = 0.05
SNAP_M_MAX = 0.15
# Slight adjustment from approved beta=0.02 → 0.025 so large plans (e.g. Cubi 853px)
# retain ~legacy 20px snap under the image cap, while tiny plans (229px) stay ~4–5px.
SNAP_BETA = 0.025
SNAP_ANGLE_MAX_DEG = 35.0
COLLINEAR_ANGLE_MAX_DEG = 8.0
L_ANGLE_CENTER_DEG = 90.0
L_ANGLE_TOL_DEG = 25.0
ZERO_LENGTH_M = 1e-4
THICKNESS_COMPAT_RATIO = 2.5

# HCI / CubiCasa aligned IDs for architectural IFC
CLASS_WALL = 3
CLASS_DOOR = 2
CLASS_WINDOW = 1


def yolo_polys(lbl: Path, class_id: int, W: int, H: int) -> list[dict]:
    out = []
    text = lbl.read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        if int(float(parts[0])) != class_id:
            continue
        coords = list(map(float, parts[1:]))
        pts = []
        for k in range(0, len(coords) - 1, 2):
            pts.append([coords[k] * W, coords[k + 1] * H])
        if len(pts) < 3:
            continue
        if pts[0] != pts[-1]:
            pts.append(pts[0][:])
        arr = np.array(pts, dtype=np.float32)
        area = float(abs(cv2.contourArea(arr)))
        out.append({"id": f"c{class_id}_{len(out)}", "class": class_id, "points_px": pts, "area_px": area})
    return out


def min_area_centerline(pts_px: list[list[float]]) -> tuple[list[list[float]], float, float]:
    cnt = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0
    rad = math.radians(angle)
    dx, dy = math.cos(rad), math.sin(rad)
    half = w / 2.0
    p1 = [cx - dx * half, cy - dy * half]
    p2 = [cx + dx * half, cy + dy * half]
    thick = max(float(h), 2.0)
    return [p1, p2], thick, float(w)


def obb_short_side_px(pts_px: list[list[float]]) -> float:
    """True OBB short-side length in pixels (no artificial floor)."""
    cnt = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(cnt) < 3:
        return 0.0
    (_c), (w, h), _ang = cv2.minAreaRect(cnt)
    return float(min(w, h))


def thickness_sample_count(length_px: float) -> int:
    return max(5, min(21, int(length_px / 10.0)))


def _contour_from_points(pts_px: list[list[float]]) -> np.ndarray:
    arr = np.array(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    return arr


def _point_inside_contour(x: float, y: float, contour: np.ndarray) -> bool:
    return float(cv2.pointPolygonTest(contour, (float(x), float(y)), False)) >= 0.0


def _ray_exit_distance_px(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    contour: np.ndarray,
    max_r: float,
    step: float = 0.25,
) -> float | None:
    """Distance along unit direction until first exterior sample (polygon exit)."""
    nlen = math.hypot(dx, dy)
    if nlen < 1e-12:
        return None
    ux, uy = dx / nlen, dy / nlen
    # Start slightly inside; if origin is outside, fail this sample
    if not _point_inside_contour(ox, oy, contour):
        # nudge toward centroid of contour
        return None
    prev_r = 0.0
    r = step
    while r <= max_r:
        x, y = ox + ux * r, oy + uy * r
        if not _point_inside_contour(x, y, contour):
            # binary refine between prev_r and r
            lo, hi = prev_r, r
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                mx, my = ox + ux * mid, oy + uy * mid
                if _point_inside_contour(mx, my, contour):
                    lo = mid
                else:
                    hi = mid
            return float(hi)
        prev_r = r
        r += step
    return None


def sample_perpendicular_widths_px(
    pts_px: list[list[float]],
    centerline_px: list[list[float]],
    length_px: float | None = None,
) -> dict[str, Any]:
    """
    Sample boundary-to-boundary width perpendicular to the OBB/PCA centerline.
    Excludes ~first/last 10% of the centerline. Pure observation of the mask.
    """
    if len(centerline_px) < 2 or len(pts_px) < 3:
        return {
            "widths_px": [],
            "valid_sample_count": 0,
            "rejected_sample_count": 0,
            "sample_ts": [],
        }
    ax, ay = float(centerline_px[0][0]), float(centerline_px[0][1])
    bx, by = float(centerline_px[-1][0]), float(centerline_px[-1][1])
    lx, ly = bx - ax, by - ay
    seg_len = math.hypot(lx, ly)
    if length_px is None:
        length_px = seg_len
    if seg_len < 1e-6:
        return {
            "widths_px": [],
            "valid_sample_count": 0,
            "rejected_sample_count": 0,
            "sample_ts": [],
        }
    tx, ty = lx / seg_len, ly / seg_len
    nx, ny = -ty, tx
    contour = _contour_from_points(pts_px)
    k = thickness_sample_count(float(length_px))
    # Interior samples only: t in (0.1, 0.9)
    ts = [0.1 + (0.8 * i / (k - 1)) for i in range(k)] if k > 1 else [0.5]
    max_r = max(8.0, 0.25 * float(length_px), 40.0)
    widths: list[float] = []
    rejected = 0
    for t in ts:
        sx, sy = ax + t * lx, ay + t * ly
        if not _point_inside_contour(sx, sy, contour):
            rejected += 1
            continue
        d_pos = _ray_exit_distance_px(sx, sy, nx, ny, contour, max_r)
        d_neg = _ray_exit_distance_px(sx, sy, -nx, -ny, contour, max_r)
        if d_pos is None or d_neg is None:
            rejected += 1
            continue
        w = float(d_pos + d_neg)
        if w < 0.5:  # unstable / degenerate
            rejected += 1
            continue
        widths.append(w)
    return {
        "widths_px": widths,
        "valid_sample_count": len(widths),
        "rejected_sample_count": rejected,
        "sample_ts": ts,
    }


def robust_median_width_px(widths_px: list[float]) -> tuple[float | None, list[float], int]:
    """
    Median of widths; reject outliers outside [0.5*med, 2.0*med]; re-median.
    Returns (median_or_None, kept_widths, n_rejected_as_outliers).
    """
    if len(widths_px) < THICK_N_MIN_SAMPLES:
        return None, list(widths_px), 0
    arr = np.asarray(widths_px, dtype=np.float64)
    med0 = float(np.median(arr))
    if med0 <= 0:
        return None, list(widths_px), 0
    lo, hi = 0.5 * med0, 2.0 * med0
    kept = [float(w) for w in arr if lo <= float(w) <= hi]
    n_out = len(widths_px) - len(kept)
    if len(kept) < THICK_N_MIN_SAMPLES:
        return None, kept, n_out
    return float(np.median(np.asarray(kept, dtype=np.float64))), kept, n_out


def thickness_px_sane(
    thickness_px: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> bool:
    if thickness_px < THICK_PX_MIN_LOW:
        return False
    max_px = THICK_PX_MAX_FRAC_LOW * float(min(W, H))
    if thickness_px > max_px:
        return False
    if scale_confidence == "high":
        # metre-space soft window checked later; still reject absurd px
        return True
    return True


def soft_clip_thickness_m(
    thickness_m: float,
    scale_confidence: str,
) -> tuple[float, bool]:
    """Apply soft metre bounds. Returns (value, clipped)."""
    if scale_confidence == "high":
        lo, hi = THICK_SOFT_HIGH_MIN_M, THICK_SOFT_HIGH_MAX_M
        if thickness_m < lo:
            return lo, True
        if thickness_m > hi:
            return hi, True
        return float(thickness_m), False
    # Low confidence: do not force into [0.10, 0.40]; light sanity only
    lo, hi = 0.01, 0.80
    if thickness_m < lo:
        return lo, True
    if thickness_m > hi:
        return hi, True
    return float(thickness_m), False


def estimate_wall_thickness_candidate(
    pts_px: list[list[float]],
    centerline_px: list[list[float]],
    length_px: float,
    W: int,
    H: int,
    mpp: float,
    scale_confidence: str = "low",
) -> dict[str, Any]:
    """
    P2 single-wall thickness candidate (before plan-level / global fallback).
    Hierarchy step 1–2 only: perpendicular_median → obb.
    """
    raw_obb = obb_short_side_px(pts_px)
    sample = sample_perpendicular_widths_px(pts_px, centerline_px, length_px)
    med, kept, n_out = robust_median_width_px(sample["widths_px"])
    rejected_total = int(sample["rejected_sample_count"]) + int(n_out)

    method = None
    thick_px = None
    fallback_reason = None

    if med is not None and thickness_px_sane(med, W, H, scale_confidence):
        method = "perpendicular_median"
        thick_px = float(med)
    elif thickness_px_sane(raw_obb, W, H, scale_confidence) and raw_obb >= THICK_PX_MIN_LOW:
        method = "obb"
        thick_px = float(raw_obb)
        fallback_reason = "PERP_INSUFFICIENT_OR_UNSTABLE" if med is None else "PERP_FAILED_SANITY"
    else:
        method = None
        thick_px = None
        if med is None:
            fallback_reason = "PERP_AND_OBB_UNUSABLE"
        else:
            fallback_reason = "THICKNESS_FAILED_SANITY"

    raw_m = float(thick_px) * float(mpp) if thick_px is not None else None
    thick_m = None
    clipped = False
    if thick_px is not None and raw_m is not None:
        if scale_confidence == "high":
            # metre soft bounds; if still absurd after clip path, leave for plan median
            thick_m, clipped = soft_clip_thickness_m(raw_m, scale_confidence)
        else:
            thick_m, clipped = soft_clip_thickness_m(raw_m, scale_confidence)

    return {
        "thickness_px": thick_px,
        "thickness_m": thick_m,
        "thickness_method": method,
        "raw_obb_px": float(raw_obb),
        "sampled_widths_px": [float(w) for w in sample["widths_px"]],
        "kept_widths_px": kept,
        "valid_sample_count": int(sample["valid_sample_count"]),
        "rejected_sample_count": rejected_total,
        "median_sample_width_px": float(med) if med is not None else None,
        "mpp": float(mpp),
        "scale_confidence": scale_confidence,
        "fallback_reason": fallback_reason,
        "clipped": bool(clipped),
        "raw_m": raw_m,
        "reliable": method in ("perpendicular_median", "obb") and thick_m is not None,
    }


def apply_plan_and_global_thickness_fallback(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float | None]:
    """
    Steps 3–4 of fallback hierarchy across the plan.
    Mutates/returns finalized per-wall dicts with thickness_method set.
    """
    reliable_ms = [
        float(c["thickness_m"])
        for c in candidates
        if c.get("reliable") and c.get("thickness_m") is not None
    ]
    plan_med = float(np.median(np.asarray(reliable_ms, dtype=np.float64))) if len(reliable_ms) >= 3 else None

    out: list[dict[str, Any]] = []
    for c in candidates:
        d = dict(c)
        d["plan_median_used"] = False
        d["global_fallback_used"] = False
        if d.get("thickness_m") is not None and d.get("thickness_method") in (
            "perpendicular_median",
            "obb",
        ):
            out.append(d)
            continue
        if plan_med is not None:
            d["thickness_m"] = float(plan_med)
            d["thickness_px"] = float(plan_med) / float(d["mpp"]) if d["mpp"] else None
            d["thickness_method"] = "plan_median"
            d["plan_median_used"] = True
            d["fallback_reason"] = d.get("fallback_reason") or "PLAN_MEDIAN"
            d["reliable"] = False
            out.append(d)
            continue
        d["thickness_m"] = float(THICK_FALLBACK)
        d["thickness_px"] = float(THICK_FALLBACK) / float(d["mpp"]) if d["mpp"] else None
        d["thickness_method"] = "fallback_global"
        d["global_fallback_used"] = True
        d["fallback_reason"] = "GLOBAL_FALLBACK"
        d["reliable"] = False
        out.append(d)
    return out, plan_med


def dist_point_to_segment(px, py, ax, ay, bx, by) -> tuple[float, float, float, float]:
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    if len2 < 1e-12:
        return math.hypot(px - ax, py - ay), 0.0, ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / len2
    t_clamped = max(0.0, min(1.0, t))
    qx, qy = ax + t_clamped * abx, ay + t_clamped * aby
    return math.hypot(px - qx, py - qy), t_clamped, qx, qy


def _classify_projection_t(t: float, eps: float = 1e-6) -> str:
    if t <= eps:
        return "endpoint_start"
    if t >= 1.0 - eps:
        return "endpoint_end"
    return "interior"


def project_t_unclamped(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    len2 = abx * abx + aby * aby
    if len2 < 1e-12:
        return 0.0
    return float(((px - ax) * abx + (py - ay) * aby) / len2)


def opening_d_max(thickness_m: float) -> float:
    return float(max(OPEN_MAX_DIST_M, 1.25 * float(thickness_m) + 0.10))


def orientation_alignment_score(wall_ang_deg: float, opening_ang_deg: float) -> float:
    """Undirected orientation agreement in [0,1] (1 = parallel)."""
    dang = angle_diff_deg(wall_ang_deg, opening_ang_deg)
    return float(max(0.0, min(1.0, 1.0 - dang / 90.0)))


def longitudinal_overlap_ratio(
    cx_m: float,
    cy_m: float,
    width_m: float,
    opening_ang_deg: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """
    Fraction of the opening's longitudinal extent (along its long axis) that
    projects onto the wall segment [0,1]. Returns [0,1].
    """
    # Prefer projecting along the wall axis using opening width as longitudinal extent
    # (door/window width typically spans along the host wall).
    wall_len = math.hypot(bx - ax, by - ay)
    if wall_len < 1e-12:
        return 0.0
    ux, uy = (bx - ax) / wall_len, (by - ay) / wall_len
    half = max(float(width_m) * 0.5, 1e-6)
    # Extent along wall direction from centroid
    p1x, p1y = cx_m + ux * half, cy_m + uy * half
    p2x, p2y = cx_m - ux * half, cy_m - uy * half
    t1 = project_t_unclamped(p1x, p1y, ax, ay, bx, by)
    t2 = project_t_unclamped(p2x, p2y, ax, ay, bx, by)
    t_lo, t_hi = (t1, t2) if t1 <= t2 else (t2, t1)
    span = max(t_hi - t_lo, 1e-9)
    overlap = max(0.0, min(t_hi, 1.0) - max(t_lo, 0.0))
    return float(max(0.0, min(1.0, overlap / span)))


def score_opening_wall_candidate(
    *,
    distance_m: float,
    d_max: float,
    t: float,
    overlap_ratio: float,
    orientation_score: float,
    duplicate_group_size: int,
    proj_eps: float = OPEN_PROJ_EPS,
) -> dict[str, Any]:
    """P4 multi-factor score for one opening×wall pair."""
    d = float(distance_m)
    dm = float(d_max) if d_max > 0 else 1.0
    distance_score = float(max(0.0, min(1.0, 1.0 - d / dm)))
    proj = _classify_projection_t(float(t), eps=proj_eps)
    is_interior = proj == "interior"
    interior_bonus = 1.0 if is_interior else 0.0
    endpoint_penalty = 0.0 if is_interior else 1.0
    dup_n = max(int(duplicate_group_size), 1)
    duplicate_penalty = 0.0 if dup_n <= 1 else min(1.0, (dup_n - 1) / dup_n)

    ov = float(max(0.0, min(1.0, overlap_ratio)))
    ori = float(max(0.0, min(1.0, orientation_score)))

    score = (
        1.0 * distance_score
        + 0.8 * interior_bonus
        + 0.7 * ov
        + 0.3 * ori
        - 0.9 * endpoint_penalty
        - 0.2 * duplicate_penalty
    )
    return {
        "score": float(score),
        "distance_score": distance_score,
        "interior_bonus": interior_bonus,
        "overlap_ratio": ov,
        "orientation_score": ori,
        "endpoint_penalty": endpoint_penalty,
        "duplicate_penalty": duplicate_penalty,
        "projection_class": proj,
        "is_interior": is_interior,
    }


def candidate_acceptance_valid(
    *,
    distance_m: float,
    d_max: float,
    is_interior: bool,
    overlap_ratio: float,
    thickness_m: float,
) -> bool:
    if float(distance_m) > float(d_max):
        return False
    if is_interior:
        return True
    # Endpoint candidates need meaningful longitudinal overlap AND tighter distance
    return bool(
        float(overlap_ratio) >= 0.30
        and float(distance_m) <= 0.5 * float(thickness_m) + 0.15
    )


def _seg_id_numeric_key(sid: str) -> tuple:
    return _seg_id_sort_key(sid)


def _polyline_key_m(poly_m: list, nd: int = 4) -> tuple:
    a, b = poly_m[0], poly_m[-1]
    k1 = (round(a[0], nd), round(a[1], nd), round(b[0], nd), round(b[1], nd))
    k2 = (round(b[0], nd), round(b[1], nd), round(a[0], nd), round(a[1], nd))
    return k1 if k1 <= k2 else k2


def stable_geometry_fingerprint(graph_m: dict, opening_map: dict | None = None) -> str:
    """Deterministic string for P1 geometry/association identity checks (read-only use)."""
    import hashlib

    segs = []
    for s in sorted(graph_m.get("segments", []), key=lambda x: x["id"]):
        segs.append(
            {
                "id": s["id"],
                "polyline_m": [[round(p[0], 6), round(p[1], 6)] for p in s["polyline_m"]],
                "thickness_m": round(float(s["thickness_m"]), 6),
                "length_m": round(float(s["length_m"]), 6),
                "start_node_id": s.get("start_node_id"),
                "end_node_id": s.get("end_node_id"),
            }
        )
    nodes = [
        {"id": n["id"], "x_px": round(float(n["x_px"]), 4), "y_px": round(float(n["y_px"]), 4)}
        for n in sorted(graph_m.get("nodes", []), key=lambda x: x["id"])
    ]
    hosts = []
    if opening_map:
        for m in sorted(opening_map.get("mappings", []), key=lambda x: x["opening_id"]):
            hosts.append(
                {
                    "opening_id": m["opening_id"],
                    "host_wall_id": m["host_wall_id"],
                    "t": round(float(m["t"]), 6),
                    "distance_to_wall": round(float(m["distance_to_wall"]), 6),
                }
            )
        unmapped = sorted(opening_map.get("unmapped", []))
    else:
        unmapped = []
    payload = json.dumps(
        {"nodes": nodes, "segments": segs, "hosts": hosts, "unmapped": unmapped},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# P3 topology helpers
# ---------------------------------------------------------------------------


def _parse_source_index(source_polygon_id: str) -> int:
    try:
        return int(str(source_polygon_id).rsplit("_", 1)[-1])
    except Exception:
        return 0


def deterministic_wall_order(walls: list[dict]) -> list[dict]:
    """Stable order: source index, then -area, then first-vertex geometry."""

    def key(w: dict):
        pts = w.get("points_px") or [[0.0, 0.0]]
        return (
            _parse_source_index(w.get("id", "")),
            -float(w.get("area_px") or 0.0),
            round(float(pts[0][0]), 4),
            round(float(pts[0][1]), 4),
            str(w.get("id") or ""),
        )

    return sorted(walls, key=key)


def segment_angle_deg(p0: list[float], p1: list[float]) -> float:
    return math.degrees(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0])))


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return float(min(d, 180.0 - d))


def compute_snap_radius_px(
    thickness_px: float,
    thickness_m: float,
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str = "low",
) -> float:
    """
    Hybrid snap radius in pixels.

    Tiny images (min(W,H) < 300): thickness-relative hybrid + image cap (fixes 20px collapse).
    Larger images: legacy SNAP_PX=20 (capped by beta*min_dim) so opening hosts/t stay P2-stable;
    zero-length walls are removed by post-snap cull instead of changing the snap regime.
    """
    mpp = float(mpp) if mpp and mpp > 0 else 0.01
    min_dim = float(min(W, H))
    cap = SNAP_BETA * min_dim

    if min_dim < 300:
        # Tiny-plan hybrid path (e.g. 2 BHK 229×220)
        if scale_confidence == "high":
            snap_m = max(SNAP_M_MIN, min(SNAP_M_MAX, SNAP_ALPHA * float(thickness_m)))
            snap_px = snap_m / mpp
        else:
            snap_px = max(SNAP_M_MIN / mpp, SNAP_ALPHA * float(thickness_px))
        return float(min(snap_px, cap))

    # Medium/large plans: preserve legacy snap magnitude for opening invariance
    if scale_confidence == "high":
        snap_m = max(SNAP_M_MIN, min(SNAP_M_MAX, SNAP_ALPHA * float(thickness_m)))
        snap_px = snap_m / mpp
        return float(min(max(snap_px, min(SNAP_PX, cap)), cap if cap >= SNAP_PX else SNAP_PX))
    return float(min(SNAP_PX, max(cap, SNAP_PX)))  # → SNAP_PX when cap≥20, else cap


def _seg_id_sort_key(sid: str) -> tuple:
    """Numeric wall id order: w2 < w12 (not lexicographic). Supports w3_p1 splits."""
    s = str(sid or "")
    if s.startswith("w"):
        core = s[1:].split("_")[0]
        try:
            return (0, int(core), s[1 + len(core) :])
        except Exception:
            pass
    return (1, s)


def _canonical_prefer_key(seg: dict) -> tuple:
    # Prefer smallest numeric id for stable hosts (opening invariance), then area / length.
    return (
        _seg_id_sort_key(seg.get("id") or ""),
        -float(seg.get("area_px") or 0.0),
        -float(seg.get("length_px_pre_snap") or seg.get("length_px") or 0.0),
    )



def pair_snap_radius_px(a: dict, b: dict, mpp: float, W: int, H: int, scale_confidence: str) -> float:
    ra = compute_snap_radius_px(
        a.get("thickness_px", 2.0), a.get("thickness_m", 0.05), mpp, W, H, scale_confidence
    )
    rb = compute_snap_radius_px(
        b.get("thickness_px", 2.0), b.get("thickness_m", 0.05), mpp, W, H, scale_confidence
    )
    return float(max(ra, rb))



def _endpoint_pairs_for_seg(seg: dict) -> tuple[list[float], list[float]]:
    poly = seg["polyline_px"]
    return [float(poly[0][0]), float(poly[0][1])], [float(poly[-1][0]), float(poly[-1][1])]


def segments_are_coincident(a: dict, b: dict, snap_px: float) -> bool:
    a0, a1 = _endpoint_pairs_for_seg(a)
    b0, b1 = _endpoint_pairs_for_seg(b)
    ang = angle_diff_deg(segment_angle_deg(a0, a1), segment_angle_deg(b0, b1))
    if ang > COLLINEAR_ANGLE_MAX_DEG:
        return False
    # Same orientation match or reversed
    d_same = max(math.hypot(a0[0] - b0[0], a0[1] - b0[1]), math.hypot(a1[0] - b1[0], a1[1] - b1[1]))
    d_rev = max(math.hypot(a0[0] - b1[0], a0[1] - b1[1]), math.hypot(a1[0] - b0[0], a1[1] - b0[1]))
    if min(d_same, d_rev) > snap_px:
        return False
    # Near-complete projected overlap: both lengths similar and midpoints close
    la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
    if la < 1e-9 or lb < 1e-9:
        return True  # both degenerate / one degenerate handled by cull
    am = [(a0[0] + a1[0]) * 0.5, (a0[1] + a1[1]) * 0.5]
    d_mid, t_mid, _, _ = dist_point_to_segment(am[0], am[1], b0[0], b0[1], b1[0], b1[1])
    if d_mid > snap_px:
        return False
    if not (0.0 - 1e-6 <= t_mid <= 1.0 + 1e-6):
        return False
    # length ratio within reason for "complete" overlap
    ratio = max(la, lb) / max(min(la, lb), 1e-9)
    return ratio <= 1.35


def _project_1d(px: float, py: float, ox: float, oy: float, ux: float, uy: float) -> float:
    return (px - ox) * ux + (py - oy) * uy


def try_collinear_merge(a: dict, b: dict, snap_px: float) -> dict | None:
    """Return merged segment dict fields if a,b should merge; else None."""
    a0, a1 = _endpoint_pairs_for_seg(a)
    b0, b1 = _endpoint_pairs_for_seg(b)
    ang_a = segment_angle_deg(a0, a1)
    ang_b = segment_angle_deg(b0, b1)
    dang = angle_diff_deg(ang_a, ang_b)
    # L-junction: do not merge
    if abs(dang - L_ANGLE_CENTER_DEG) <= L_ANGLE_TOL_DEG:
        return None
    if dang > COLLINEAR_ANGLE_MAX_DEG:
        return None
    # thickness compatibility
    ta = float(a.get("thickness_px") or 1.0)
    tb = float(b.get("thickness_px") or 1.0)
    if max(ta, tb) / max(min(ta, tb), 1e-9) > THICKNESS_COMPAT_RATIO:
        return None
    # common axis from longer segment
    la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
    if la >= lb:
        ox, oy = a0
        ux, uy = a1[0] - a0[0], a1[1] - a0[1]
    else:
        ox, oy = b0
        ux, uy = b1[0] - b0[0], b1[1] - b0[1]
    ulen = math.hypot(ux, uy)
    if ulen < 1e-9:
        return None
    ux, uy = ux / ulen, uy / ulen
    # lateral separation of midpoints
    am = [(a0[0] + a1[0]) * 0.5, (a0[1] + a1[1]) * 0.5]
    bm = [(b0[0] + b1[0]) * 0.5, (b0[1] + b1[1]) * 0.5]
    # cross distance between axes
    d_lat = abs((bm[0] - am[0]) * (-uy) + (bm[1] - am[1]) * ux)
    if d_lat > snap_px:
        return None
    pts = [a0, a1, b0, b1]
    ts = [_project_1d(p[0], p[1], ox, oy, ux, uy) for p in pts]
    t_min, t_max = min(ts), max(ts)
    # interval gap: if projections overlap or gap <= snap
    ta0 = _project_1d(a0[0], a0[1], ox, oy, ux, uy)
    ta1 = _project_1d(a1[0], a1[1], ox, oy, ux, uy)
    tb0 = _project_1d(b0[0], b0[1], ox, oy, ux, uy)
    tb1 = _project_1d(b1[0], b1[1], ox, oy, ux, uy)
    a_lo, a_hi = min(ta0, ta1), max(ta0, ta1)
    b_lo, b_hi = min(tb0, tb1), max(tb0, tb1)
    gap = max(0.0, max(a_lo, b_lo) - min(a_hi, b_hi))  # 0 if overlap
    # if separated along axis
    if a_hi < b_lo:
        gap = b_lo - a_hi
    elif b_hi < a_lo:
        gap = a_lo - b_hi
    else:
        gap = 0.0
    if gap > snap_px:
        return None
    # Prefer not to merge if this looks like T (endpoint of one on interior of other)
    # — T should stay two segments; only merge true collinear continuations / overlaps
    for (p, other0, other1) in (
        (a0, b0, b1),
        (a1, b0, b1),
        (b0, a0, a1),
        (b1, a0, a1),
    ):
        d, t, _, _ = dist_point_to_segment(p[0], p[1], other0[0], other0[1], other1[0], other1[1])
        if 0.08 < t < 0.92 and d <= snap_px * 0.75:
            # endpoint on interior → T, do not collinear-merge
            return None

    p_start = [ox + ux * t_min, oy + uy * t_min]
    p_end = [ox + ux * t_max, oy + uy * t_max]
    canon = a if _canonical_prefer_key(a) <= _canonical_prefer_key(b) else b
    other = b if canon is a else a
    return {
        "id": canon["id"],
        "polyline_px": [p_start, p_end],
        "length_px": float(math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])),
        "merged_from": sorted({canon["id"], other["id"]} | set(canon.get("merged_from") or []) | set(other.get("merged_from") or [])),
        "source_polygon_ids": sorted(
            set(canon.get("source_polygon_ids") or [canon.get("source_polygon_id")])
            | set(other.get("source_polygon_ids") or [other.get("source_polygon_id")])
        ),
        "area_px": max(float(canon.get("area_px") or 0), float(other.get("area_px") or 0)),
        "thickness_px": float(canon.get("thickness_px")),
        "thickness_m": float(canon.get("thickness_m")),
        "thickness_est": canon.get("thickness_est"),
        "points_px": canon.get("points_px"),
        "centerline_px_pre_snap": canon.get("centerline_px_pre_snap"),
        "length_px_pre_snap": float(canon.get("length_px_pre_snap") or canon.get("length_px") or 0)
        + float(other.get("length_px_pre_snap") or other.get("length_px") or 0),
        "source_polygon_id": canon.get("source_polygon_id"),
    }


def assign_nodes_frozen(segs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Assign node ids from already-finalized polylines without moving endpoints."""
    nodes: list[dict] = []
    key_to_nid: dict[tuple, str] = {}
    snap_groups: list[dict] = []
    bucket: dict[tuple, list[str]] = defaultdict(list)

    def nid_for(x: float, y: float) -> str:
        key = (round(float(x), 4), round(float(y), 4))
        if key not in key_to_nid:
            nid = f"n{len(nodes)}"
            key_to_nid[key] = nid
            nodes.append({"id": nid, "x_px": float(x), "y_px": float(y)})
        return key_to_nid[key]

    for s in segs:
        a = s["polyline_px"][0]
        b = s["polyline_px"][-1]
        na = nid_for(a[0], a[1])
        nb = nid_for(b[0], b[1])
        s["start_node_id"] = na
        s["end_node_id"] = nb
        s["length_px"] = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        bucket[(round(float(a[0]), 4), round(float(a[1]), 4))].append(s["id"])
        bucket[(round(float(b[0]), 4), round(float(b[1]), 4))].append(s["id"])

    for (x, y), seg_ids in sorted(bucket.items(), key=lambda kv: kv[0]):
        nid = key_to_nid[(x, y)]
        snap_groups.append(
            {
                "node_id": nid,
                "endpoint_indices": [],
                "segment_ids": sorted(set(seg_ids)),
                "size": len(set(seg_ids)),
            }
        )
    return nodes, snap_groups


def rebuild_nodes_from_segments(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Cluster endpoints with hybrid angular-aware snap; write start/end node ids back.
    Returns (nodes, snap_groups, near_misses).
    """
    if not segs:
        return [], [], []

    endpoints: list[tuple[float, float, int, str]] = []
    for i, s in enumerate(segs):
        endpoints.append((float(s["polyline_px"][0][0]), float(s["polyline_px"][0][1]), i, "a"))
        endpoints.append((float(s["polyline_px"][-1][0]), float(s["polyline_px"][-1][1]), i, "b"))

    n = len(endpoints)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    near_misses: list[dict] = []
    endpoint_snaps = 0

    # Endpoint–endpoint unions within hybrid snap radius (distance-based node sharing).
    # Angular incompatibility is recorded as NEAR_MISS for diagnostics; node sharing still
    # occurs when d<=snap_r so L junctions and legacy opening hosts remain stable.
    # Collinear *merge* elsewhere still requires angular compatibility.
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = endpoints[i][2], endpoints[j][2]
            if si == sj:
                continue
            d = math.hypot(endpoints[i][0] - endpoints[j][0], endpoints[i][1] - endpoints[j][1])
            snap_r = pair_snap_radius_px(segs[si], segs[sj], mpp, W, H, scale_confidence)
            if d > snap_r:
                continue
            ai = segment_angle_deg(segs[si]["polyline_px"][0], segs[si]["polyline_px"][-1])
            aj = segment_angle_deg(segs[sj]["polyline_px"][0], segs[sj]["polyline_px"][-1])
            dang = angle_diff_deg(ai, aj)
            is_L = abs(dang - L_ANGLE_CENTER_DEG) <= L_ANGLE_TOL_DEG
            is_cont = dang < SNAP_ANGLE_MAX_DEG
            if not (is_cont or is_L):
                near_misses.append(
                    {
                        "seg_a": segs[si]["id"],
                        "seg_b": segs[sj]["id"],
                        "distance_px": float(d),
                        "snap_px": float(snap_r),
                        "angle_diff_deg": float(dang),
                        "topology_type": "NEAR_MISS",
                        "snapped_despite_angle": True,
                    }
                )
            union(i, j)
            endpoint_snaps += 1


    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # Deterministic cluster order by centroid
    cluster_items = []
    for members in clusters.values():
        xs = [endpoints[i][0] for i in members]
        ys = [endpoints[i][1] for i in members]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        cluster_items.append((cx, cy, sorted(members)))
    cluster_items.sort(key=lambda t: (round(t[0], 4), round(t[1], 4), tuple(t[2])))

    nodes = []
    ep_to_node: dict[int, str] = {}
    snap_groups = []
    for ci, (cx, cy, members) in enumerate(cluster_items):
        nid = f"n{ci}"
        nodes.append({"id": nid, "x_px": float(cx), "y_px": float(cy)})
        seg_ids = sorted({segs[endpoints[i][2]]["id"] for i in members})
        snap_groups.append(
            {
                "node_id": nid,
                "endpoint_indices": list(members),
                "segment_ids": seg_ids,
                "size": len(members),
            }
        )
        for i in members:
            ep_to_node[i] = nid

    for i, s in enumerate(segs):
        ia = next(k for k, e in enumerate(endpoints) if e[2] == i and e[3] == "a")
        ib = next(k for k, e in enumerate(endpoints) if e[2] == i and e[3] == "b")
        na, nb = ep_to_node[ia], ep_to_node[ib]
        na_n = next(n_ for n_ in nodes if n_["id"] == na)
        nb_n = next(n_ for n_ in nodes if n_["id"] == nb)
        s["polyline_px"] = [[na_n["x_px"], na_n["y_px"]], [nb_n["x_px"], nb_n["y_px"]]]
        s["length_px"] = math.hypot(nb_n["x_px"] - na_n["x_px"], nb_n["y_px"] - na_n["y_px"])
        s["start_node_id"] = na
        s["end_node_id"] = nb

    # stash count for diagnostics
    for g in snap_groups:
        g["endpoint_snap_ops"] = endpoint_snaps

    return nodes, snap_groups, near_misses


def apply_t_junction_snaps(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> list[dict]:
    """Move endpoints onto host interiors when within hybrid snap; record events on segs."""
    t_events: list[dict] = []
    # Work on mutable endpoint coordinates in segs' polylines
    for si, s in enumerate(segs):
        for which, epi in (("a", 0), ("b", -1)):
            ex, ey = float(s["polyline_px"][epi][0]), float(s["polyline_px"][epi][1])
            best_d, best_sj, best_t, best_q = 1e18, -1, 0.0, None
            for sj, host in enumerate(segs):
                if sj == si:
                    continue
                ax, ay = host["polyline_px"][0]
                bx, by = host["polyline_px"][-1]
                d, t, qx, qy = dist_point_to_segment(ex, ey, ax, ay, bx, by)
                snap_r = pair_snap_radius_px(s, host, mpp, W, H, scale_confidence)
                if 0.05 < t < 0.95 and d <= snap_r and d < best_d:
                    best_d, best_sj, best_t, best_q = d, sj, t, (qx, qy)
            if best_sj >= 0 and best_q is not None:
                s["polyline_px"][epi] = [float(best_q[0]), float(best_q[1])]
                t_events.append(
                    {
                        "endpoint_seg_id": s["id"],
                        "endpoint_which": which,
                        "host_seg_id": segs[best_sj]["id"],
                        "distance_px": float(best_d),
                        "t_on_host": float(best_t),
                        "topology_type": "T",
                        "snap_px": pair_snap_radius_px(s, segs[best_sj], mpp, W, H, scale_confidence),
                    }
                )
        s["length_px"] = math.hypot(
            s["polyline_px"][-1][0] - s["polyline_px"][0][0],
            s["polyline_px"][-1][1] - s["polyline_px"][0][1],
        )
    return t_events


def dedupe_coincident_segments(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> tuple[list[dict], list[dict]]:
    """Keep one canonical per coincident group. Returns (kept, group_records)."""
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            snap_r = pair_snap_radius_px(segs[i], segs[j], mpp, W, H, scale_confidence)
            if segments_are_coincident(segs[i], segs[j], snap_r):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    kept: list[dict] = []
    records: list[dict] = []
    gid = 0
    for members in sorted(groups.values(), key=lambda ms: min(segs[i]["id"] for i in ms)):
        if len(members) == 1:
            s = segs[members[0]]
            s.setdefault("merged_from", [s["id"]])
            s.setdefault("source_polygon_ids", [s.get("source_polygon_id")])
            s["topology_status"] = s.get("topology_status") or "ACTIVE"
            s["duplicate_group_id"] = None
            s["canonical_segment_id"] = s["id"]
            kept.append(s)
            continue
        ranked = sorted(members, key=lambda i: _canonical_prefer_key(segs[i]))
        canon = segs[ranked[0]]
        # Keep canonical id's own P2 thickness (do not adopt another member's thickness —
        # adopting shifted Cubi median and opening d_max).
        ids = [segs[i]["id"] for i in ranked]
        srcs = []
        for i in ranked:
            srcs.extend(segs[i].get("source_polygon_ids") or [segs[i].get("source_polygon_id")])
        canon["merged_from"] = sorted(set(ids) | set(canon.get("merged_from") or []))
        canon["source_polygon_ids"] = sorted({s for s in srcs if s})
        canon["duplicate_group_id"] = f"dup{gid}"
        canon["canonical_segment_id"] = canon["id"]
        canon["topology_status"] = "DEDUPED_CANONICAL"
        canon["area_px"] = max(float(segs[i].get("area_px") or 0) for i in ranked)
        kept.append(canon)
        records.append(
            {
                "duplicate_group_id": f"dup{gid}",
                "canonical_segment_id": canon["id"],
                "merged_from": sorted(ids),
                "count": len(ids),
            }
        )
        gid += 1
    return kept, records


def merge_tol_px_p5a(a: dict, b: dict, mpp: float, W: int, H: int, scale_confidence: str) -> float:
    """
    Gap / projection-contiguity tolerance in pixels (P5A).
    Thickness-relative metres clamped to [0.03, 0.15], converted via mpp; when scale
    confidence is low, prefer existing pixel topology snap (capped) rather than inventing scale.
    """
    ta = float(a.get("thickness_m") or 0.05)
    tb = float(b.get("thickness_m") or 0.05)
    med = 0.5 * (ta + tb)
    tol_m = max(0.03, min(0.15, 0.75 * med))
    tol_px = tol_m / max(float(mpp), 1e-9)
    snap = pair_snap_radius_px(a, b, mpp, W, H, scale_confidence)
    conf = (scale_confidence or "low").lower()
    if conf != "high":
        # Low/medium: do not invent metres — use topology snap, but never above snap.
        return float(min(snap, max(tol_px, snap * 0.5)))
    return float(min(snap, max(tol_px, 3.0)))


def lateral_tol_px_p5a(a: dict, b: dict) -> float:
    """
    Perpendicular (supporting-line) tolerance — tighter than gap tol.
    Prevents chaining parallel offset YOLO bands into one drifted centerline.
    """
    ta = float(a.get("thickness_px") or 1.0)
    tb = float(b.get("thickness_px") or 1.0)
    return float(max(2.0, 0.45 * min(ta, tb)))


def _merge_geom_endpoints(s: dict) -> tuple[list[float], list[float]]:
    """Prefer pre-snap centerline for wall-run reconstruction (post-snap may be collapsed)."""
    pre = s.get("centerline_px_pre_snap")
    if pre and len(pre) >= 2:
        p0, p1 = pre[0], pre[-1]
        if math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])) > 1e-6:
            return [float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]
    return _endpoint_pairs_for_seg(s)


def can_form_wall_run(a: dict, b: dict, tol_px: float, lateral_tol_px: float | None = None) -> bool:
    """
    True if a,b are collinear/near-collinear and overlap or nearly touch along axis.
    Uses pre-snap geometry when available. Rejects L (~90°) and T (endpoint on interior
    of non-collinear partner is already angle-rejected; collinear T-stub blocked below).
    """
    a0, a1 = _merge_geom_endpoints(a)
    b0, b1 = _merge_geom_endpoints(b)
    ang_a = segment_angle_deg(a0, a1)
    ang_b = segment_angle_deg(b0, b1)
    dang = angle_diff_deg(ang_a, ang_b)
    if abs(dang - L_ANGLE_CENTER_DEG) <= L_ANGLE_TOL_DEG:
        return False
    if dang > COLLINEAR_ANGLE_MAX_DEG:
        return False
    ta = float(a.get("thickness_px") or 1.0)
    tb = float(b.get("thickness_px") or 1.0)
    if max(ta, tb) / max(min(ta, tb), 1e-9) > THICKNESS_COMPAT_RATIO:
        return False
    la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
    if la < 1e-9 and lb < 1e-9:
        return False
    # Supporting-line distance: project the shorter (or partner) onto the longer axis
    if la >= lb:
        ox, oy = a0
        ux, uy = a1[0] - a0[0], a1[1] - a0[1]
        q0, q1 = b0, b1
    else:
        ox, oy = b0
        ux, uy = b1[0] - b0[0], b1[1] - b0[1]
        q0, q1 = a0, a1
    ulen = math.hypot(ux, uy)
    if ulen < 1e-9:
        return False
    ux, uy = ux / ulen, uy / ulen
    nx, ny = -uy, ux
    d_lat = 0.5 * (
        abs((q0[0] - ox) * nx + (q0[1] - oy) * ny)
        + abs((q1[0] - ox) * nx + (q1[1] - oy) * ny)
    )
    lat_tol = float(lateral_tol_px) if lateral_tol_px is not None else lateral_tol_px_p5a(a, b)
    if d_lat > lat_tol:
        return False
    ta0 = _project_1d(a0[0], a0[1], ox, oy, ux, uy)
    ta1 = _project_1d(a1[0], a1[1], ox, oy, ux, uy)
    tb0 = _project_1d(b0[0], b0[1], ox, oy, ux, uy)
    tb1 = _project_1d(b1[0], b1[1], ox, oy, ux, uy)
    a_lo, a_hi = min(ta0, ta1), max(ta0, ta1)
    b_lo, b_hi = min(tb0, tb1), max(tb0, tb1)
    if a_hi < b_lo:
        gap = b_lo - a_hi
    elif b_hi < a_lo:
        gap = a_lo - b_hi
    else:
        gap = 0.0
    if gap > tol_px:
        return False
    # Block clear T: short stub endpoint on interior of much longer collinear host
    # with small lateral offset — keep as T topology, not one run.
    for short, long_ in ((a, b), (b, a)):
        s0, s1 = _merge_geom_endpoints(short)
        l0, l1 = _merge_geom_endpoints(long_)
        ls = math.hypot(s1[0] - s0[0], s1[1] - s0[1])
        ll = math.hypot(l1[0] - l0[0], l1[1] - l0[1])
        if ll < 1e-6 or ls >= 0.55 * ll:
            continue
        for p in (s0, s1):
            d, t, _, _ = dist_point_to_segment(p[0], p[1], l0[0], l0[1], l1[0], l1[1])
            if 0.12 < t < 0.88 and d <= lat_tol and ls < 0.35 * ll:
                return False
    return True


def union_wall_run_group(members: list[dict]) -> dict:
    """
    Build one wall run covering the UNION of member projections on a canonical axis.
    Uses pre-snap centerlines when present. Canonical id = smallest numeric segment id.
    Axis is taken from the canonical member so host geometry stays stable.
    """
    assert members
    members_sorted = sorted(members, key=lambda s: _canonical_prefer_key(s))
    canon = members_sorted[0]

    # Axis from canonical member (stable hosts); fall back to longest if degenerate
    c0, c1 = _merge_geom_endpoints(canon)
    ux, uy = c1[0] - c0[0], c1[1] - c0[1]
    ulen = math.hypot(ux, uy)
    ox, oy = c0[0], c0[1]
    if ulen < 1e-9:
        best_len = -1.0
        for s in members:
            p0, p1 = _merge_geom_endpoints(s)
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if L > best_len:
                best_len = L
                ox, oy = p0
                ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        ulen = math.hypot(ux, uy)
    if ulen < 1e-9:
        # fallback: keep canon polyline
        out = dict(canon)
        out["merged_from"] = sorted({m["id"] for m in members}, key=_seg_id_sort_key)
        out["source_polygon_ids"] = sorted(
            {sid for m in members for sid in (m.get("source_polygon_ids") or [m.get("source_polygon_id")]) if sid}
        )
        out["topology_status"] = "P5A_WALL_RUN"
        out["canonical_segment_id"] = canon["id"]
        return out
    ux, uy = ux / ulen, uy / ulen

    ts: list[float] = []
    for s in members:
        p0, p1 = _merge_geom_endpoints(s)
        ts.append(_project_1d(p0[0], p0[1], ox, oy, ux, uy))
        ts.append(_project_1d(p1[0], p1[1], ox, oy, ux, uy))
    t_min, t_max = min(ts), max(ts)
    p_start = [ox + ux * t_min, oy + uy * t_min]
    p_end = [ox + ux * t_max, oy + uy * t_max]

    # Thickness: prefer median of members (stable); keep canon's thickness_est
    thicks_m = [float(m.get("thickness_m") or 0) for m in members]
    thicks_px = [float(m.get("thickness_px") or 0) for m in members]
    thick_m = float(np.median(np.asarray(thicks_m, dtype=np.float64))) if thicks_m else float(canon["thickness_m"])
    thick_px = float(np.median(np.asarray(thicks_px, dtype=np.float64))) if thicks_px else float(canon["thickness_px"])

    merged_ids = sorted({m["id"] for m in members}, key=_seg_id_sort_key)
    srcs = sorted(
        {
            sid
            for m in members
            for sid in (m.get("source_polygon_ids") or [m.get("source_polygon_id")])
            if sid
        }
    )
    # Also union pre-snap for diagnostics continuity
    pre_pts = []
    for m in members:
        p0, p1 = _merge_geom_endpoints(m)
        pre_pts.extend([p0, p1])

    out = {
        "id": canon["id"],
        "source_polygon_id": canon.get("source_polygon_id"),
        "source_polygon_ids": srcs,
        "area_px": max(float(m.get("area_px") or 0) for m in members),
        "points_px": canon.get("points_px"),
        "centerline_px_pre_snap": [list(p_start), list(p_end)],
        "polyline_px": [list(p_start), list(p_end)],
        "thickness_px": thick_px,
        "thickness_m": thick_m,
        "thickness_est": canon.get("thickness_est"),
        "length_px": float(math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])),
        "length_px_pre_snap": float(sum(float(m.get("length_px_pre_snap") or 0) for m in members)),
        "merged_from": merged_ids,
        "topology_status": "P5A_WALL_RUN",
        "canonical_segment_id": canon["id"],
        "duplicate_group_id": canon.get("duplicate_group_id"),
        "merge_reason": "COLLINEAR_OR_COINCIDENT_UNION",
        "p5a_member_count": len(members),
    }
    return out


def reconstruct_wall_runs_p5a(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> tuple[list[dict], list[dict]]:
    """
    P5A: merge coincident/collinear wall fragments into union wall runs.
    Returns (active_segments, merge_records).
    """
    if not segs:
        return [], []

    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Pass 1: collinear / near-collinear union on pre-snap geometry (tight lateral)
    for i in range(n):
        for j in range(i + 1, n):
            tol = merge_tol_px_p5a(segs[i], segs[j], mpp, W, H, scale_confidence)
            lat = lateral_tol_px_p5a(segs[i], segs[j])
            if can_form_wall_run(segs[i], segs[j], tol, lat):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out: list[dict] = []
    records: list[dict] = []
    gid = 0
    for members_idx in sorted(
        groups.values(),
        key=lambda idxs: _seg_id_sort_key(min((segs[i]["id"] for i in idxs), key=_seg_id_sort_key)),
    ):
        members = [segs[i] for i in members_idx]
        if len(members) == 1:
            s = dict(members[0])
            s.setdefault("merged_from", [s["id"]])
            s.setdefault("source_polygon_ids", [s.get("source_polygon_id")])
            s.setdefault("canonical_segment_id", s["id"])
            if s.get("topology_status") in (None, "ACTIVE", "DEDUPED_CANONICAL"):
                s["topology_status"] = "ACTIVE"
            out.append(s)
            continue
        merged = union_wall_run_group(members)
        merged["merge_group_id"] = f"run{gid}"
        out.append(merged)
        records.append(
            {
                "merge_group_id": f"run{gid}",
                "canonical_segment_id": merged["id"],
                "merged_from": list(merged["merged_from"]),
                "member_count": len(members),
                "union_length_px": float(merged["length_px"]),
                "topology_type": "P5A_WALL_RUN",
                "merge_reason": "COLLINEAR_OR_COINCIDENT_UNION",
            }
        )
        gid += 1

    # Pass 2: drop remaining post-snap coincident copies; keep canonical polyline as-is
    out, dedupe_recs = _collapse_post_snap_coincident(
        out, mpp, W, H, scale_confidence, start_gid=gid
    )
    records.extend(dedupe_recs)

    out.sort(key=lambda s: _seg_id_sort_key(s["id"]))
    return out, records


def _collapse_post_snap_coincident(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
    start_gid: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Keep canonical post-snap polyline; drop redundant coincident copies."""
    if len(segs) < 2:
        return segs, []
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            snap = pair_snap_radius_px(segs[i], segs[j], mpp, W, H, scale_confidence)
            if segments_are_coincident(segs[i], segs[j], snap):
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out: list[dict] = []
    records: list[dict] = []
    gid = int(start_gid)
    for members_idx in sorted(
        groups.values(),
        key=lambda idxs: _seg_id_sort_key(min((segs[i]["id"] for i in idxs), key=_seg_id_sort_key)),
    ):
        members = [segs[i] for i in members_idx]
        if len(members) == 1:
            out.append(members[0])
            continue
        members_sorted = sorted(members, key=_canonical_prefer_key)
        canon = dict(members_sorted[0])
        merged_ids = sorted({m["id"] for m in members}, key=_seg_id_sort_key)
        srcs = sorted(
            {
                sid
                for m in members
                for sid in (m.get("source_polygon_ids") or [m.get("source_polygon_id")])
                if sid
            }
        )
        prior_merged: list[str] = []
        for m in members:
            prior_merged.extend(list(m.get("merged_from") or [m["id"]]))
        canon["merged_from"] = sorted(set(prior_merged), key=_seg_id_sort_key)
        canon["source_polygon_ids"] = srcs
        canon["canonical_segment_id"] = canon["id"]
        canon["topology_status"] = "P5A_WALL_RUN"
        canon["merge_reason"] = "POST_SNAP_COINCIDENT_DEDUPE"
        canon["area_px"] = max(float(m.get("area_px") or 0) for m in members)
        thicks_m = [float(m.get("thickness_m") or 0) for m in members]
        thicks_px = [float(m.get("thickness_px") or 0) for m in members]
        canon["thickness_m"] = float(np.median(np.asarray(thicks_m, dtype=np.float64)))
        canon["thickness_px"] = float(np.median(np.asarray(thicks_px, dtype=np.float64)))
        canon["merge_group_id"] = f"run{gid}"
        out.append(canon)
        records.append(
            {
                "merge_group_id": f"run{gid}",
                "canonical_segment_id": canon["id"],
                "merged_from": merged_ids,
                "member_count": len(members),
                "union_length_px": float(canon.get("length_px") or 0),
                "topology_type": "P5A_WALL_RUN",
                "merge_reason": "POST_SNAP_COINCIDENT_DEDUPE",
            }
        )
        gid += 1
    return out, records


def merge_collinear_segments(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
) -> tuple[list[dict], list[dict]]:
    """Backward-compatible name → P5A wall-run reconstruction. """
    return reconstruct_wall_runs_p5a(segs, mpp, W, H, scale_confidence)


def _merge_collinear_segments_impl(
    segs: list[dict],
    mpp: float,
    W: int,
    H: int,
    scale_confidence: str,
    enabled: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Test helper: pairwise legacy merge when enabled; else identity."""
    if not enabled:
        out = []
        for s in segs:
            s = dict(s)
            s.setdefault("merged_from", [s["id"]])
            s.setdefault("source_polygon_ids", [s.get("source_polygon_id")])
            s.setdefault("canonical_segment_id", s["id"])
            s.setdefault("topology_status", "ACTIVE")
            out.append(s)
        out.sort(key=lambda s: _seg_id_sort_key(s["id"]))
        return out, []
    return reconstruct_wall_runs_p5a(segs, mpp, W, H, scale_confidence)


def classify_L_X_junctions(segs: list[dict], mpp: float) -> tuple[list[dict], list[dict]]:
    L_list: list[dict] = []
    X_list: list[dict] = []
    for i, a in enumerate(segs):
        a0, a1 = a["polyline_px"][0], a["polyline_px"][-1]
        ang_a = segment_angle_deg(a0, a1)
        for b in segs[i + 1 :]:
            b0, b1 = b["polyline_px"][0], b["polyline_px"][-1]
            ang_b = segment_angle_deg(b0, b1)
            dang = angle_diff_deg(ang_a, ang_b)
            # L: ~90° and share a near endpoint
            if abs(dang - L_ANGLE_CENTER_DEG) <= L_ANGLE_TOL_DEG:
                dists = [
                    math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))
                    for p in (a0, a1)
                    for q in (b0, b1)
                ]
                if min(dists) <= 1.5:  # after node share, should be ~0
                    L_list.append(
                        {
                            "segment_a": a["id"],
                            "segment_b": b["id"],
                            "angle_diff_deg": round(dang, 3),
                            "topology_type": "L",
                        }
                    )
            # X: interiors cross (diagnostic only)
            amx, amy = (float(a0[0]) + float(a1[0])) / 2.0, (float(a0[1]) + float(a1[1])) / 2.0
            bmx, bmy = (float(b0[0]) + float(b1[0])) / 2.0, (float(b0[1]) + float(b1[1])) / 2.0
            da, ta, _, _ = dist_point_to_segment(amx, amy, float(b0[0]), float(b0[1]), float(b1[0]), float(b1[1]))
            db, tb, _, _ = dist_point_to_segment(bmx, bmy, float(a0[0]), float(a0[1]), float(a1[0]), float(a1[1]))
            # metre-ish thresholds via mpp for stability
            thr = max(2.0, 0.15 / max(mpp, 1e-9))
            if 0.15 < ta < 0.85 and 0.15 < tb < 0.85 and da < thr and db < thr and dang > 25:
                X_list.append(
                    {
                        "segment_a": a["id"],
                        "segment_b": b["id"],
                        "mid_dist_a_to_b_px": round(da, 4),
                        "mid_dist_b_to_a_px": round(db, 4),
                        "topology_type": "X",
                    }
                )
    return L_list, X_list


def build_wall_graph(
    walls: list[dict],
    W: int,
    H: int,
    mpp: float,
    warnings: list[str],
    scale_confidence: str = "low",
) -> dict:
    """
    Build wall graph: P2 thickness + P3 hybrid topology.
    Opening association is performed separately by associate_openings() (P4 scoring).
    """
    thickness_diag: list[dict] = []
    scale_confidence = "high" if str(scale_confidence).lower() == "high" else "low"
    mpp = float(mpp)

    # --- Deterministic ordering (P3) ---
    walls_ordered = deterministic_wall_order(walls)

    raw_segs: list[dict] = []
    for w in walls_ordered:
        poly, thick_px_legacy, length_px = min_area_centerline(w["points_px"])
        raw_segs.append(
            {
                "id": f"w{len(raw_segs)}",
                "source_polygon_id": w["id"],
                "source_polygon_ids": [w["id"]],
                "area_px": float(w.get("area_px") or 0.0),
                "points_px": w["points_px"],
                "centerline_px_pre_snap": [list(poly[0]), list(poly[-1])],
                "polyline_px": [list(poly[0]), list(poly[-1])],
                "thickness_px": float(thick_px_legacy),
                "length_px": float(length_px),
                "length_px_pre_snap": float(length_px),
                "merged_from": [],
                "topology_status": "ACTIVE",
            }
        )

    old_segment_count = len(raw_segs)

    def py_to_m(y):
        return (H - y) * mpp

    # --- P2 thickness BEFORE topology (needed for hybrid snap; sampling uses pre-snap centerline) ---
    candidates: list[dict[str, Any]] = []
    for s in raw_segs:
        cand = estimate_wall_thickness_candidate(
            pts_px=s["points_px"],
            centerline_px=s["centerline_px_pre_snap"],
            length_px=float(s["length_px_pre_snap"]),
            W=W,
            H=H,
            mpp=mpp,
            scale_confidence=scale_confidence,
        )
        cand["segment_id"] = s["id"]
        cand["source_polygon_id"] = s["source_polygon_id"]
        candidates.append(cand)
    finalized, plan_med = apply_plan_and_global_thickness_fallback(candidates)
    by_id = {c["segment_id"]: c for c in finalized}
    for s in raw_segs:
        est = by_id[s["id"]]
        s["thickness_m"] = float(est["thickness_m"])
        s["thickness_px"] = (
            float(est["thickness_px"]) if est["thickness_px"] is not None else float(s["thickness_px"])
        )
        s["thickness_est"] = est

    def _emit_thickness_diag(s: dict, length_m: float, length_px_post: float) -> None:
        est = s.get("thickness_est") or {}
        method = est.get("thickness_method")
        thick_m = float(s["thickness_m"])
        thick_px = float(s["thickness_px"])
        raw_m = float(est["raw_m"]) if est.get("raw_m") is not None else thick_px * mpp
        if method == "fallback_global":
            result_kind, fallback_occurred = "fallback", True
        elif method == "plan_median":
            result_kind, fallback_occurred = "plan_median", False
        elif est.get("clipped"):
            result_kind, fallback_occurred = "clipped", False
        else:
            result_kind = "raw" if method == "perpendicular_median" else "obb"
            fallback_occurred = False
        thickness_diag.append(
            {
                "segment_id": s["id"],
                "source_polygon_id": s["source_polygon_id"],
                "thick_px": thick_px,
                "raw_m": float(raw_m),
                "thickness_m": thick_m,
                "fallback_occurred": fallback_occurred,
                "fallback_reason": est.get("fallback_reason"),
                "result_kind": result_kind,
                "length_px_pre_snap": float(s.get("length_px_pre_snap") or 0),
                "length_px_post_snap": float(length_px_post),
                "length_m": float(length_m),
                "mpp": float(mpp),
                "image_width": int(W),
                "image_height": int(H),
                "thickness_px": thick_px,
                "thickness_method": method,
                "raw_obb_px": float(est.get("raw_obb_px") or 0.0),
                "sampled_widths_px": list(est.get("sampled_widths_px") or []),
                "valid_sample_count": int(est.get("valid_sample_count") or 0),
                "rejected_sample_count": int(est.get("rejected_sample_count") or 0),
                "median_sample_width_px": est.get("median_sample_width_px"),
                "scale_confidence": scale_confidence,
                "clipped": bool(est.get("clipped")),
                "plan_median_used": bool(est.get("plan_median_used")),
                "global_fallback_used": bool(est.get("global_fallback_used")),
                "plan_median_m": plan_med,
            }
        )

    # P2 thickness diagnostics for ALL source walls (regression baseline; before topology cull)
    for s in raw_segs:
        _emit_thickness_diag(s, float(s["length_px_pre_snap"]) * mpp, float(s["length_px_pre_snap"]))

    # --- P5B: optional complex/fragmented centerline recovery (fallback; default OFF) ---
    p5b_diag: dict[str, Any] = {"p5b_version": 1, "enabled": False}
    if p5b_enabled(default=False):
        raw_segs, p5b_diag = apply_p5b_to_segments(raw_segs, W, H, mpp)
        # Re-sync length fields after possible splits (thickness_m retained from P2)
        for s in raw_segs:
            if len(s.get("polyline_px") or []) >= 2:
                s["length_px"] = math.hypot(
                    s["polyline_px"][-1][0] - s["polyline_px"][0][0],
                    s["polyline_px"][-1][1] - s["polyline_px"][0][1],
                )
    else:
        p5b_diag["enabled"] = False
        p5b_diag["activations"] = 0

    # --- P3 topology ---
    t_events = apply_t_junction_snaps(raw_segs, mpp, W, H, scale_confidence)
    nodes, snap_groups, near_misses = rebuild_nodes_from_segments(
        raw_segs, mpp, W, H, scale_confidence
    )
    old_node_count = len(nodes)

    zero_length_before = sum(1 for s in raw_segs if float(s["length_px"]) * mpp < ZERO_LENGTH_M)

    # Count coincident groups before dedupe
    dup_before = 0
    _seen = set()
    for i, a in enumerate(raw_segs):
        if a["id"] in _seen:
            continue
        group = [a["id"]]
        for b in raw_segs[i + 1 :]:
            snap_r = pair_snap_radius_px(a, b, mpp, W, H, scale_confidence)
            if segments_are_coincident(a, b, snap_r):
                group.append(b["id"])
                _seen.add(b["id"])
        if len(group) > 1:
            dup_before += 1
            _seen.add(a["id"])

    # Cull zero-length after snap (live filter)
    culled: list[dict] = []
    active: list[dict] = []
    for s in raw_segs:
        length_m = float(s["length_px"]) * mpp
        if length_m < ZERO_LENGTH_M:
            s["topology_status"] = "CULLED_ZERO_LENGTH"
            s["culled_reason"] = "ZERO_LENGTH_AFTER_SNAP"
            culled.append(s)
            warnings.append(f"{s['id']} culled zero-length after snap")
        else:
            s["topology_status"] = "ACTIVE"
            active.append(s)

    # Duplicate groups: diagnostics (pre-P5A). P5A will remove/union coincident runs.
    dup_records: list[dict] = []
    _seen_d: set[str] = set()
    gid = 0
    for i, a in enumerate(active):
        if a["id"] in _seen_d:
            continue
        group = [a]
        for b in active[i + 1 :]:
            if b["id"] in _seen_d:
                continue
            snap_r = pair_snap_radius_px(a, b, mpp, W, H, scale_confidence)
            if segments_are_coincident(a, b, snap_r):
                group.append(b)
        if len(group) > 1:
            ids = sorted((g["id"] for g in group), key=_seg_id_sort_key)
            for g in group:
                _seen_d.add(g["id"])
                g["duplicate_group_id"] = f"dup{gid}"
            canon_id = ids[0]
            dup_records.append(
                {
                    "duplicate_group_id": f"dup{gid}",
                    "canonical_segment_id": canon_id,
                    "merged_from": ids,
                    "count": len(ids),
                    "action": "DETECTED_PRE_P5A",
                }
            )
            gid += 1

    # --- P5A: coincident + collinear wall-run reconstruction (pre-snap union) ---
    active, collinear_merges = reconstruct_wall_runs_p5a(active, mpp, W, H, scale_confidence)
    for rec in collinear_merges:
        rec["action"] = "P5A_MERGED"

    # Freeze polylines from first snap — do not re-average endpoints (preserves opening hosts)
    final_segs: list[dict] = []
    for s in active:
        s["length_px"] = math.hypot(
            s["polyline_px"][-1][0] - s["polyline_px"][0][0],
            s["polyline_px"][-1][1] - s["polyline_px"][0][1],
        )
        if float(s["length_px"]) * mpp < ZERO_LENGTH_M:
            s["topology_status"] = "CULLED_ZERO_LENGTH"
            s["culled_reason"] = "ZERO_LENGTH_AFTER_DEDUP"
            culled.append(s)
            warnings.append(f"{s['id']} culled zero-length after dedup")
        else:
            final_segs.append(s)
    
     # --- Connected walls: exact centerline intersections + corner extension (L/T) ---
    junctions: list[dict] = []
    connect_diag: dict[str, Any] = {"enabled": False}
    if connect_walls_enabled(default=True):
        junctions, connect_diag = connect_wall_segments(final_segs, mpp)

    nodes, snap_groups = assign_nodes_frozen(final_segs)

    L_list, X_list = classify_L_X_junctions(final_segs, mpp)

    # --- Emit metre segments (thickness_diag already filled for all source walls) ---
    segments_m: list[dict] = []
    topology_seg_diag: list[dict] = []

    for s in final_segs:
        poly_m = [[p[0] * mpp, py_to_m(p[1])] for p in s["polyline_px"]]
        length_m = float(s["length_px"]) * mpp
        snap_px_local = compute_snap_radius_px(
            s["thickness_px"], s["thickness_m"], mpp, W, H, scale_confidence
        )
        public = {
            "id": s["id"],
            "source_polygon_id": s["source_polygon_id"],
            "source_polygon_ids": list(s.get("source_polygon_ids") or [s.get("source_polygon_id")]),
            "merged_from": list(s.get("merged_from") or [s["id"]]),
            "polyline_px": s["polyline_px"],
            "thickness_px": float(s["thickness_px"]),
            "length_px": float(s["length_px"]),
            "start_node_id": s.get("start_node_id"),
            "end_node_id": s.get("end_node_id"),
            "polyline_m": poly_m,
            "thickness_m": float(s["thickness_m"]),
            "length_m": float(length_m),
            "ext_start_m": float(s.get("ext_start_px", 0.0)) * mpp,
            "ext_end_m": float(s.get("ext_end_px", 0.0)) * mpp,
        }
        if s.get("centerline_method"):
            public["centerline_method"] = s.get("centerline_method")
        if s.get("p5b_classifier"):
            public["classifier"] = s.get("p5b_classifier")
        if s.get("p5b_confidence") is not None:
            public["confidence"] = float(s["p5b_confidence"])
        if s.get("p5b_mask_support_ratio") is not None:
            public["mask_support_ratio"] = float(s["p5b_mask_support_ratio"])
        segments_m.append(public)
        topology_seg_diag.append(
            {
                "original_segment_id": s["id"],
                "canonical_segment_id": s.get("canonical_segment_id") or s["id"],
                "source_polygon_ids": list(s.get("source_polygon_ids") or [s.get("source_polygon_id")]),
                "topology_status": s.get("topology_status") or "ACTIVE",
                "duplicate_group_id": s.get("duplicate_group_id"),
                "merged_from": list(s.get("merged_from") or [s["id"]]),
                "snap_px": float(snap_px_local),
                "snap_m": float(snap_px_local) * mpp,
                "original_length_px": float(s.get("length_px_pre_snap") or 0),
                "final_length_px": float(s["length_px"]),
                "final_length_m": float(length_m),
                "culled_reason": None,
            }
        )

    for s in culled:
        length_m = float(s.get("length_px") or 0) * mpp
        topology_seg_diag.append(
            {
                "original_segment_id": s["id"],
                "canonical_segment_id": None,
                "source_polygon_ids": list(s.get("source_polygon_ids") or [s.get("source_polygon_id")]),
                "topology_status": "CULLED_ZERO_LENGTH",
                "duplicate_group_id": None,
                "merged_from": list(s.get("merged_from") or [s["id"]]),
                "snap_px": compute_snap_radius_px(
                    s.get("thickness_px", 2), s.get("thickness_m", 0.05), mpp, W, H, scale_confidence
                ),
                "snap_m": None,
                "original_length_px": float(s.get("length_px_pre_snap") or 0),
                "final_length_px": float(s.get("length_px") or 0),
                "final_length_m": float(length_m),
                "culled_reason": s.get("culled_reason"),
            }
        )

    zero_length_after = [s for s in segments_m if float(s["length_m"]) < ZERO_LENGTH_M]

    # Observational remaining coincident groups (should be ~0)
    coincident: dict[tuple, list[str]] = defaultdict(list)
    for s in segments_m:
        coincident[_polyline_key_m(s["polyline_m"])].append(s["id"])
    duplicate_groups_after = [
        {"polyline_key": list(k), "segment_ids": ids, "count": len(ids)}
        for k, ids in sorted(coincident.items(), key=lambda kv: kv[1][0])
        if len(ids) > 1
    ]

    # Degree distribution
    degree: dict[str, int] = defaultdict(int)
    for s in segments_m:
        degree[s["start_node_id"]] += 1
        degree[s["end_node_id"]] += 1
    degree_hist: dict[str, int] = defaultdict(int)
    for d in degree.values():
        degree_hist[str(d)] += 1

    mean_snap = (
        float(
            np.mean(
                [
                    compute_snap_radius_px(s["thickness_px"], s["thickness_m"], mpp, W, H, scale_confidence)
                    for s in final_segs
                ]
            )
        )
        if final_segs
        else 0.0
    )

    diagnostics = {
        "p1_version": 1,
        "p2_version": 1,
        "p3_version": 1,
        "p5a_version": 1,
        "p5b_version": 1,
        "scale_confidence": scale_confidence,
        "plan_median_m": plan_med,
        "thickness": thickness_diag,
        "p5b": p5b_diag,
        "connect": connect_diag,
        "topology": {
            "segment_count_pre_snap": old_segment_count,
            "segment_count_post_snap": len(segments_m),
            "old_segment_count": old_segment_count,
            "new_segment_count": len(segments_m),
            "old_node_count": old_node_count,
            "new_node_count": len(nodes),
            "node_count": len(nodes),
            "snap_px": float(mean_snap),
            "snap_m": float(mean_snap) * mpp,
            "snap_policy": {
                "alpha": SNAP_ALPHA,
                "snap_m_min": SNAP_M_MIN,
                "snap_m_max": SNAP_M_MAX,
                "beta": SNAP_BETA,
                "legacy_SNAP_PX": SNAP_PX,
            },
            "mpp": float(mpp),
            "image_width": int(W),
            "image_height": int(H),
            "snap_groups": snap_groups,
            "zero_length_before": zero_length_before,
            "zero_length_after": len(zero_length_after),
            "zero_length_segments": [
                {"segment_id": s["id"], "length_m": float(s["length_m"]), "length_px": float(s["length_px"])}
                for s in zero_length_after
            ],
            "duplicate_groups_before": dup_before,
            "duplicate_groups_after": len(duplicate_groups_after),
            "duplicate_coincident_groups": duplicate_groups_after,
            "duplicate_merge_records": dup_records,
            "collinear_merges": collinear_merges,
            "collinear_overlap_candidates": collinear_merges,  # P1 key retained (now real merges)
            "t_junction_candidates": t_events,
            "T_snaps": t_events,
            "endpoint_snaps": sum(g.get("endpoint_snap_ops", 0) for g in snap_groups[:1]) if snap_groups else 0,
            "L_junctions": L_list,
            "X_intersections": X_list,
            "x_junction_candidates": X_list,
            "near_misses": near_misses,
            "culled": [
                {"segment_id": s["id"], "reason": s.get("culled_reason")} for s in culled
            ],
            "culled_count": len(culled),
            "segments": topology_seg_diag,
            "degree_distribution": dict(degree_hist),
            "duplicates_removed": sum(max(0, int(r.get("member_count", r.get("count", 1))) - 1) for r in collinear_merges),
            "duplicates_detected": sum(max(0, r["count"] - 1) for r in dup_records),
            "p5a_wall_runs": collinear_merges,
            "p5a_merge_count": len(collinear_merges),
        },
    }

    return {
        "W": W,
        "H": H,
        "meters_per_pixel": mpp,
        "nodes": nodes,
        "segments": segments_m,
        "junctions": junctions,
        "gt_wall_count": len(walls),
        "diagnostics": diagnostics,
    }


def opening_metrics(pts_px: list[list[float]]) -> dict:
    arr = np.array(pts_px, dtype=np.float32)
    cnt = arr.reshape(-1, 1, 2)
    M = cv2.moments(cnt)
    if M["m00"] > 1e-6:
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    else:
        cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
    rect = cv2.minAreaRect(cnt)
    (rcx, rcy), (w, h), angle = rect
    width_px = float(max(w, h))
    depth_px = float(min(w, h))
    if w < h:
        angle += 90.0
    return {
        "centroid_px": [float(cx), float(cy)],
        "width_px": width_px,
        "depth_px": depth_px,
        "orientation_deg": float(angle),
    }


def associate_openings(graph_m: dict, doors: list[dict], wins: list[dict], H: int, mpp: float, warnings: list[str]) -> dict:
    """
    P4: deterministic multi-factor opening→wall association.
    Does not modify topology or thickness; scoring only.
    """

    def px_to_m(x, y):
        return x * mpp, (H - y) * mpp

    openings = []
    for o in doors:
        o = dict(o)
        o.update(opening_metrics(o["points_px"]))
        o["type"] = "door"
        openings.append(o)
    for o in wins:
        o = dict(o)
        o.update(opening_metrics(o["points_px"]))
        o["type"] = "window"
        openings.append(o)

    segments = list(graph_m.get("segments") or [])

    # Coincident / duplicate groups for scoring: LIVE geometry only.
    # Do NOT seed from pre-P5A duplicate_merge_records — those member IDs were removed
    # from the active graph and would incorrectly re-penalize the canonical wall run.
    coincident_lookup: dict[str, list[str]] = {}
    by_key: dict[tuple, list[str]] = defaultdict(list)
    for s in segments:
        by_key[_polyline_key_m(s["polyline_m"])].append(s["id"])
    for ids in by_key.values():
        if len(ids) > 1:
            for sid in ids:
                coincident_lookup[sid] = sorted(ids, key=_seg_id_sort_key)

    mappings = []
    unmapped = []
    opening_diag: list[dict] = []

    for o in openings:
        cx, cy = o["centroid_px"]
        cx_m, cy_m = px_to_m(cx, cy)
        width_m = float(o["width_px"]) * float(mpp)
        open_ang = float(o["orientation_deg"])

        all_candidates: list[dict] = []
        for s in segments:
            ax, ay = s["polyline_m"][0]
            bx, by = s["polyline_m"][-1]
            d, t_clamped, qx, qy = dist_point_to_segment(cx_m, cy_m, ax, ay, bx, by)
            # Use unclamped t for classification? Spec says projection t on segment —
            # dist_point_to_segment returns clamped t which is correct for endpoint class.
            t = float(t_clamped)
            thick = float(s["thickness_m"])
            d_max = opening_d_max(thick)
            wall_ang = math.degrees(math.atan2(by - ay, bx - ax))
            ov = longitudinal_overlap_ratio(cx_m, cy_m, width_m, open_ang, ax, ay, bx, by)
            ori = orientation_alignment_score(wall_ang, open_ang)
            group = coincident_lookup.get(s["id"]) or [s["id"]]
            scored = score_opening_wall_candidate(
                distance_m=d,
                d_max=d_max,
                t=t,
                overlap_ratio=ov,
                orientation_score=ori,
                duplicate_group_size=len(group),
            )
            valid = candidate_acceptance_valid(
                distance_m=d,
                d_max=d_max,
                is_interior=bool(scored["is_interior"]),
                overlap_ratio=ov,
                thickness_m=thick,
            )
            cand = {
                "wall_id": s["id"],
                "distance_m": float(d),
                "distance_px": float(d / mpp) if mpp > 0 else float("inf"),
                "d_max": float(d_max),
                "t": float(t),
                "projection_class": scored["projection_class"],
                "is_interior": bool(scored["is_interior"]),
                "overlap_ratio": float(ov),
                "orientation_score": float(ori),
                "score": float(scored["score"]),
                "distance_score": float(scored["distance_score"]),
                "interior_bonus": float(scored["interior_bonus"]),
                "endpoint_penalty": float(scored["endpoint_penalty"]),
                "duplicate_penalty": float(scored["duplicate_penalty"]),
                "accepted_by_threshold": bool(valid),
                "wall_length_m": float(s["length_m"]),
                "wall_thickness_m": thick,
                "coincident_group": group if len(group) > 1 else None,
                "orientation_deg": float(wall_ang),
            }
            all_candidates.append(cand)

        # Deterministic ranking (never encounter order):
        # highest score, highest overlap, interior over endpoint, smallest distance, smallest seg id
        def rank_key(c: dict):
            return (
                -float(c["score"]),
                -float(c["overlap_ratio"]),
                0 if c["is_interior"] else 1,
                float(c["distance_m"]),
                _seg_id_sort_key(c["wall_id"]),
            )

        all_candidates.sort(key=rank_key)
        for rank, c in enumerate(all_candidates):
            c["rank_by_score"] = rank

        valid_cands = [c for c in all_candidates if c["accepted_by_threshold"]]
        within_d = [c for c in all_candidates if c["distance_m"] <= c["d_max"] + 1e-12]

        best_cand = valid_cands[0] if valid_cands else None

        if best_cand is None:
            unmapped.append(o["id"])
            warnings.append(f"unmapped opening {o['id']} ({o['type']})")
            if not segments:
                reject_reason = "NO_WALLS"
            elif not within_d:
                reject_reason = "TOO_FAR"
            elif within_d and all(not c["is_interior"] for c in within_d):
                # All in-range are endpoint; check if overlap clause failed
                if all(c["overlap_ratio"] < 0.30 for c in within_d):
                    reject_reason = "ENDPOINT_ONLY"
                else:
                    reject_reason = "ENDPOINT_ONLY"
            elif within_d and all(c["overlap_ratio"] < 0.30 for c in within_d):
                reject_reason = "NO_OVERLAP"
            else:
                reject_reason = "AMBIGUOUS"

            nearest = all_candidates[0] if all_candidates else None
            # Prefer nearest-by-distance for diagnostics display
            nearest_by_d = (
                min(all_candidates, key=lambda c: (c["distance_m"], _seg_id_sort_key(c["wall_id"])))
                if all_candidates
                else None
            )
            opening_diag.append(
                {
                    "opening_id": o["id"],
                    "class": o["type"],
                    "centroid_px": [float(cx), float(cy)],
                    "accepted": False,
                    "rejection_reason": reject_reason,
                    "nearest_candidate_wall": nearest_by_d["wall_id"] if nearest_by_d else None,
                    "nearest_distance_m": nearest_by_d["distance_m"] if nearest_by_d else None,
                    "nearest_distance_px": nearest_by_d["distance_px"] if nearest_by_d else None,
                    "nearest_d_max": nearest_by_d["d_max"] if nearest_by_d else None,
                    "nearest_t": nearest_by_d["t"] if nearest_by_d else None,
                    "nearest_projection_class": nearest_by_d["projection_class"] if nearest_by_d else None,
                    "best_score_wall": nearest["wall_id"] if nearest else None,
                    "best_score": nearest["score"] if nearest else None,
                    "candidate_count": len(all_candidates),
                    "candidates_within_d_max": within_d[:12],
                    "top_candidates": all_candidates[:8],
                }
            )
        else:
            s_host = next(s for s in segments if s["id"] == best_cand["wall_id"])
            mapping = {
                "opening_id": o["id"],
                "opening_type": o["type"],
                "host_wall_id": best_cand["wall_id"],
                "distance_to_wall": float(best_cand["distance_m"]),
                "offset_along_wall": float(best_cand["t"]) * float(s_host["length_m"]),
                "t": float(best_cand["t"]),
                "orientation_deg": float(best_cand["orientation_deg"]),
                "width_m": width_m,
                "wall_length_m": float(s_host["length_m"]),
                "wall_thickness_m": float(s_host["thickness_m"]),
                "association_score": float(best_cand["score"]),
                "overlap_ratio": float(best_cand["overlap_ratio"]),
                "projection_class": best_cand["projection_class"],
            }
            mappings.append(mapping)
            opening_diag.append(
                {
                    "opening_id": o["id"],
                    "class": o["type"],
                    "centroid_px": [float(cx), float(cy)],
                    "accepted": True,
                    "rejection_reason": None,
                    "host_wall_id": best_cand["wall_id"],
                    "distance_m": float(best_cand["distance_m"]),
                    "distance_px": float(best_cand["distance_px"]),
                    "d_max": float(best_cand["d_max"]),
                    "t": float(best_cand["t"]),
                    "projection_class": best_cand["projection_class"],
                    "score": float(best_cand["score"]),
                    "overlap_ratio": float(best_cand["overlap_ratio"]),
                    "orientation_score": float(best_cand["orientation_score"]),
                    "candidate_count": len(all_candidates),
                    "candidates_within_d_max": within_d[:12],
                    "top_candidates": all_candidates[:8],
                    "coincident_group": best_cand.get("coincident_group"),
                }
            )

    result = {
        "total_openings": len(openings),
        "successfully_mapped": len(mappings),
        "unmapped": unmapped,
        "unmapped_count": len(unmapped),
        "mapping_success_rate": len(mappings) / len(openings) if openings else 1.0,
        "mappings": mappings,
        "diagnostics": {"p1_version": 1, "p4_version": 1, "openings": opening_diag},
    }
    result["geometry_fingerprint"] = stable_geometry_fingerprint(graph_m, result)
    return result


def write_ifc4(graph_m: dict, opening_map: dict, out_ifc: Path, basename: str) -> dict:
    import ifcopenshell
    import ifcopenshell.guid

    model = ifcopenshell.file(schema="IFC4")
    tnow = int(time.time())
    person = model.create_entity("IfcPerson", Identification="HCI_2.1", FamilyName="HCI")
    org = model.create_entity("IfcOrganization", Name="HCI Interior 2.1")
    p_and_o = model.create_entity("IfcPersonAndOrganization", ThePerson=person, TheOrganization=org)
    app = model.create_entity(
        "IfcApplication",
        ApplicationDeveloper=org,
        Version="2.1",
        ApplicationFullName="HCI_2.1 IFC Export",
        ApplicationIdentifier="hci_2_1_ifc",
    )
    owner = model.create_entity(
        "IfcOwnerHistory",
        OwningUser=p_and_o,
        OwningApplication=app,
        ChangeAction="ADDED",
        CreationDate=tnow,
    )
    unit_l = model.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    units = model.create_entity("IfcUnitAssignment", Units=[unit_l])
    origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    world_pl = model.create_entity("IfcAxis2Placement3D", Location=origin)
    context = model.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        Precision=1e-05,
        WorldCoordinateSystem=world_pl,
    )
    body_context = model.create_entity(
        "IfcGeometricRepresentationSubContext",
        ContextIdentifier="Body",
        ContextType="Model",
        ParentContext=context,
        TargetView="MODEL_VIEW",
    )
    project = model.create_entity(
        "IfcProject",
        GlobalId=ifcopenshell.guid.new(),
        Name=f"HCI {basename}",
        OwnerHistory=owner,
        RepresentationContexts=[context],
        UnitsInContext=units,
    )
    site = model.create_entity(
        "IfcSite",
        GlobalId=ifcopenshell.guid.new(),
        Name="Site",
        ObjectPlacement=model.create_entity("IfcLocalPlacement", RelativePlacement=world_pl),
    )
    building = model.create_entity(
        "IfcBuilding",
        GlobalId=ifcopenshell.guid.new(),
        Name="Building",
        ObjectPlacement=model.create_entity(
            "IfcLocalPlacement", PlacementRelTo=site.ObjectPlacement, RelativePlacement=world_pl
        ),
    )
    storey_pl = model.create_entity(
        "IfcLocalPlacement", PlacementRelTo=building.ObjectPlacement, RelativePlacement=world_pl
    )
    storey = model.create_entity(
        "IfcBuildingStorey",
        GlobalId=ifcopenshell.guid.new(),
        Name="Ground Floor",
        ObjectPlacement=storey_pl,
        Elevation=0.0,
    )
    model.create_entity(
        "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site]
    )
    model.create_entity(
        "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building]
    )
    model.create_entity(
        "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey]
    )

    def make_extruded_box(length, width, height, z0=0.0,x_start=0.0):
        half = width / 2.0
        x_end = x_start + length
        pts = [
            model.create_entity("IfcCartesianPoint", Coordinates=c)
            for c in [(x_start, -half),(x_end, -half),(x_end, half),(x_start, half),(x_start, -half),]
        ]
        profile = model.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=model.create_entity("IfcPolyline", Points=pts),
        )
        if abs(z0) > 1e-9:
            z_origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, float(z0)))
            pos = model.create_entity("IfcAxis2Placement3D", Location=z_origin)
        else:
            pos = world_pl
        return model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=pos,
            ExtrudedDirection=model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            Depth=float(height),
        )

    def assign_body(product, solid):
        rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body_context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        product.Representation = model.create_entity(
            "IfcProductDefinitionShape", Representations=[rep]
        )

    wall_entities = {}
    wall_placements = {}
    elements = []

    for s in graph_m["segments"]:
        x0, y0 = s["polyline_m"][0]
        x1, y1 = s["polyline_m"][-1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-4:
            continue
        angle = math.atan2(dy, dx)
        thick = float(np.clip(s["thickness_m"], THICK_MIN, THICK_MAX))
        wall_origin = model.create_entity("IfcCartesianPoint", Coordinates=(float(x0), float(y0), 0.0))
        wall_ax = model.create_entity(
            "IfcAxis2Placement3D",
            Location=wall_origin,
            RefDirection=model.create_entity(
                "IfcDirection", DirectionRatios=(math.cos(angle), math.sin(angle), 0.0)
            ),
        )
        wall_loc = model.create_entity(
            "IfcLocalPlacement", PlacementRelTo=storey_pl, RelativePlacement=wall_ax
        )
        ifc_wall = model.create_entity(
            "IfcWall",
            GlobalId=ifcopenshell.guid.new(),
            Name=s["id"],
            OwnerHistory=owner,
            ObjectPlacement=wall_loc,
        )
        # Connected walls: L-ends extended by half the other wall's thickness
        ext0 = float(s.get("ext_start_m") or 0.0)
        ext1 = float(s.get("ext_end_m") or 0.0)
        assign_body(
            ifc_wall,
            make_extruded_box(length + ext0 + ext1, thick, WALL_H, x_start=-ext0),
        )
        wall_entities[s["id"]] = ifc_wall
        wall_placements[s["id"]] = {"placement": wall_loc, "length": length, "thick": thick}
        elements.append(ifc_wall)

    door_count = win_count = opening_count = 0
    for m in opening_map["mappings"]:
        wid = m["host_wall_id"]
        if wid not in wall_entities:
            continue
        wp = wall_placements[wid]
        wall = wall_entities[wid]
        length = wp["length"]
        thick = wp["thick"]
        width = float(max(0.4, min(m["width_m"], length * 0.9)))
        t = m["t"]
        local_x = t * length - width / 2.0
        local_x = max(0.05, min(local_x, length - width - 0.05))
        if m["opening_type"] == "door":
            oh, z0 = DOOR_H, 0.0
        else:
            oh, z0 = WIN_H, WIN_SILL

        op_origin = model.create_entity("IfcCartesianPoint", Coordinates=(float(local_x), 0.0, 0.0))
        op_ax = model.create_entity("IfcAxis2Placement3D", Location=op_origin)
        op_loc = model.create_entity(
            "IfcLocalPlacement", PlacementRelTo=wp["placement"], RelativePlacement=op_ax
        )
        opening = model.create_entity(
            "IfcOpeningElement",
            GlobalId=ifcopenshell.guid.new(),
            Name=f"opening_{m['opening_id']}",
            OwnerHistory=owner,
            ObjectPlacement=op_loc,
        )
        assign_body(opening, make_extruded_box(width, thick * 1.05, oh, z0=z0))
        model.create_entity(
            "IfcRelVoidsElement",
            GlobalId=ifcopenshell.guid.new(),
            RelatingBuildingElement=wall,
            RelatedOpeningElement=opening,
        )
        opening_count += 1

        fill_loc = model.create_entity(
            "IfcLocalPlacement", PlacementRelTo=wp["placement"], RelativePlacement=op_ax
        )
        if m["opening_type"] == "door":
            door = model.create_entity(
                "IfcDoor",
                GlobalId=ifcopenshell.guid.new(),
                Name=m["opening_id"],
                OwnerHistory=owner,
                ObjectPlacement=fill_loc,
                OverallHeight=DOOR_H,
                OverallWidth=width,
            )
            assign_body(door, make_extruded_box(width * 0.98, min(0.05, thick * 0.4), DOOR_H))
            model.create_entity(
                "IfcRelFillsElement",
                GlobalId=ifcopenshell.guid.new(),
                RelatingOpeningElement=opening,
                RelatedBuildingElement=door,
            )
            elements.append(door)
            door_count += 1
        else:
            win = model.create_entity(
                "IfcWindow",
                GlobalId=ifcopenshell.guid.new(),
                Name=m["opening_id"],
                OwnerHistory=owner,
                ObjectPlacement=fill_loc,
                OverallHeight=WIN_H,
                OverallWidth=width,
            )
            assign_body(win, make_extruded_box(width * 0.98, min(0.05, thick * 0.4), WIN_H, z0=WIN_SILL))
            model.create_entity(
                "IfcRelFillsElement",
                GlobalId=ifcopenshell.guid.new(),
                RelatingOpeningElement=opening,
                RelatedBuildingElement=win,
            )
            elements.append(win)
            win_count += 1
    junction_count = 0
    for j in graph_m.get("junctions", []):
        if j.get("kind") not in ("L", "T"):
            continue
        wa, wb = wall_entities.get(j["a"]), wall_entities.get(j["b"])
        if wa is None or wb is None:
            continue
        # T: RelatingElement = host (ATPATH), RelatedElement = the wall that butts into it
        if j["kind"] == "T":
            relating, related = wb, wa
            relating_type, related_type = "ATPATH", j["a_end"]
        else:
            relating, related = wa, wb
            relating_type, related_type = j["a_end"], j["b_end"]
        model.create_entity(
            "IfcRelConnectsPathElements",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name=f"{j['kind']}-junction {j['a']}/{j['b']}",
            RelatingElement=relating,
            RelatedElement=related,
            RelatingPriorities=[],
            RelatedPriorities=[],
            RelatedConnectionType=related_type,
            RelatingConnectionType=relating_type,
        )
        junction_count += 1
        
    if elements:
        model.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            RelatingStructure=storey,
            RelatedElements=elements,
        )

    out_ifc.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_ifc))
    return {
        "walls": len(wall_entities),
        "openings": opening_count,
        "doors": door_count,
        "windows": win_count,
        "connected_junctions": junction_count,
    }


def generate_full_ifc(
    image_path: str | Path,
    label_path: str | Path,
    output_ifc: str | Path,
    meters_per_pixel: float,
    work_dir: str | Path | None = None,
    scale_confidence: str = "low",
) -> dict:
    """
    End-to-end IFC generation from an image + YOLO-seg label file.
    Returns a result dict with counts, mapping stats, warnings, paths.
    """
    warnings: list[str] = []
    image_path = Path(image_path)
    label_path = Path(label_path)
    output_ifc = Path(output_ifc)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"label not found: {label_path}")
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be > 0")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    H, W = img.shape[:2]
    basename = image_path.stem

    walls = yolo_polys(label_path, CLASS_WALL, W, H)
    doors = yolo_polys(label_path, CLASS_DOOR, W, H)
    wins = yolo_polys(label_path, CLASS_WINDOW, W, H)
    if not walls:
        raise ValueError("No Wall (class 3) polygons found in label file")

    graph_m = build_wall_graph(
        walls, W, H, float(meters_per_pixel), warnings, scale_confidence=scale_confidence
    )
    opening_map = associate_openings(graph_m, doors, wins, H, float(meters_per_pixel), warnings)
    counts = write_ifc4(graph_m, opening_map, output_ifc, basename)

    # Optional debug sidecars (Windows-safe names: strip trailing spaces)
    if work_dir is not None:
        wd = Path(work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        safe_base = str(basename).strip() or "unnamed"
        (wd / f"{safe_base}_wall_graph_m.json").write_text(
            json.dumps(graph_m, indent=2), encoding="utf-8"
        )
        (wd / f"{safe_base}_opening_wall_map.json").write_text(
            json.dumps(opening_map, indent=2), encoding="utf-8"
        )

    xs = [p[0] for s in graph_m["segments"] for p in s["polyline_m"]]
    ys = [p[1] for s in graph_m["segments"] for p in s["polyline_m"]]
    bbox = {
        "size_x_m": max(xs) - min(xs) if xs else 0,
        "size_y_m": max(ys) - min(ys) if ys else 0,
        "height_m": WALL_H,
    }

    return {
        "ok": True,
        "basename": basename,
        "ifc_path": str(output_ifc),
        "meters_per_pixel": float(meters_per_pixel),
        "gt_walls": len(walls),
        "gt_doors": len(doors),
        "gt_windows": len(wins),
        "ifc_walls": counts["walls"],
        "ifc_openings": counts["openings"],
        "ifc_doors": counts["doors"],
        "ifc_windows": counts["windows"],
        "ifc_connected_junctions": counts.get("connected_junctions", 0),
        "mapping_success_rate": opening_map["mapping_success_rate"],
        "mapped_openings": opening_map["successfully_mapped"],
        "unmapped_openings": opening_map["unmapped_count"],
        "bbox_m": bbox,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # P1 observational only — extra keys; existing callers ignore them
        "geometry_fingerprint": opening_map.get("geometry_fingerprint"),
        "p1_diagnostics": {
            "wall_graph": graph_m.get("diagnostics"),
            "openings": opening_map.get("diagnostics"),
            "unmapped": opening_map.get("unmapped"),
        },
    }
