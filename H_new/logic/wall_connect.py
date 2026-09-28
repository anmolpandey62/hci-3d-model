#!/usr/bin/env python3
"""
Connected walls (L / T junctions) for HCI_2.1.

One geometric core, two adapters:

  1. connect_wall_polygons()  -> 2D wall polygons (YOLO labels / model predictions)
       Wall rectangles are extended so that at every L / T junction they
       OVERLAP (L: up to the outer face of the other wall, T: up to the host
       centerline).  The union of the polygons is then one connected shape with
       no gaps and no notched outer corners.

  2. connect_wall_segments()  -> centerline segments used by ifc_pipeline
       Segment end points are moved to the *exact* intersection of the two
       centerlines (instead of the average of the two end points), and every
       L-end gets an extension of half the other wall's thickness so the IFC
       boxes fill the outer corner.  A junction list is returned so the IFC
       writer can emit IfcRelConnectsPathElements.

Only numpy + cv2 are needed.  Nothing here touches model weights.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

ENV_FLAG = "HCI_CONNECT_WALLS"

PERP_TOL_DEG = 30.0      # walls whose angle is 90 +/- this are candidates for L/T
GAP_TOL_FACTOR = 2.0     # bridge gaps up to  factor * max(t_a, t_b)  (px along the wall axis)
THROUGH_SLACK = 0.25     # extra slack (x t_other) before a wall counts as "passing through"
MIN_SOLIDITY = 0.80      # polygon area / OBB area to be treated as a straight wall run
MIN_ASPECT = 1.5         # OBB long/short; below this it is a pier / column, left alone
# Duplicate Wall detection settings
DUP_WALL_ANGLE_DEG = 10.0
DUP_WALL_CENTER_TOL_FACTOR = 0.75
DUP_WALL_OVERLAP_RATIO = 0.50
DUP_WALL_THICKNESS_RATIO = 3.0
THICK_MIN_M, THICK_MAX_M = 0.10, 0.40  # same clip as write_ifc4


def connect_walls_enabled(default: bool = True) -> bool:
    raw = os.environ.get(ENV_FLAG)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# geometry core
# --------------------------------------------------------------------------- #
@dataclass
class WallRect:
    wid: str
    cx: float
    cy: float
    ux: float   # unit vector along the wall
    uy: float
    hl: float   # half length (px)
    t: float    # full thickness (px)


@dataclass
class Junction:
    kind: str            # "L" | "T" | "X"
    a: str               # wall id (for T: the butting wall)
    a_end: str           # ATSTART | ATEND | ATPATH
    b: str               # wall id (for T: the host wall)
    b_end: str
    point: tuple         # exact centerline intersection (px)
    gap_px: float        # total gap that was bridged (0 if walls already touched)


def _intersect(a: WallRect, b: WallRect):
    """Intersection of the two infinite centerlines.  Returns (s, r, sin_theta) or None."""
    det = -a.ux * b.uy + b.ux * a.uy
    if abs(det) < 1e-9:
        return None
    dx, dy = b.cx - a.cx, b.cy - a.cy
    s = (-dx * b.uy + b.ux * dy) / det
    r = (a.ux * dy - a.uy * dx) / det
    return s, r, abs(det)


def _status(s: float, hl: float, t_other: float, sin_t: float, gap_tol: float):
    """Where does the intersection point sit relative to this wall's span?"""
    e = hl - abs(s)                                   # >0 inside span, <0 beyond the end
    ext = 0.5 * t_other / max(sin_t, 0.5)             # half-thickness of the other wall along this axis
    if e < -gap_tol:
        return "FAR", e, ext
    if e <= ext + THROUGH_SLACK * t_other:
        return "END", e, ext
    return "THROUGH", e, ext


def find_junctions(
    walls: list[WallRect],
    *,
    perp_tol_deg: float = PERP_TOL_DEG,
    gap_tol_factor: float = GAP_TOL_FACTOR,
):
    """
    Detect L / T / X junctions between wall rectangles.

    Returns (junctions, ends) where
      ends[(wid, end_idx)] = {"kind","other","s_node","ext","gap"}   end_idx 0 = start(-u), 1 = end(+u)
    Each wall end is used by at most one L/T junction (greedy, lowest cost first).
    """
    min_sin = math.sin(math.radians(90.0 - perp_tol_deg))
    cands = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            hit = _intersect(a, b)
            if hit is None:
                continue
            s, r, sin_t = hit
            if sin_t < min_sin:
                continue
            gap_tol = gap_tol_factor * max(a.t, b.t)
            sa, ea, exta = _status(s, a.hl, b.t, sin_t, min(gap_tol, 2 * a.hl))
            sb, eb, extb = _status(r, b.hl, a.t, sin_t, min(gap_tol, 2 * b.hl))
            if "FAR" in (sa, sb):
                continue
            gap = max(0.0, -ea) + max(0.0, -eb)
            px = a.cx + a.ux * s
            py = a.cy + a.uy * s
            if sa == "END" and sb == "END":
                kind = "L"
            elif sa == "END" and sb == "THROUGH":
                kind = "T_a"      # a butts into b
            elif sa == "THROUGH" and sb == "END":
                kind = "T_b"      # b butts into a
            else:
                kind = "X"
            cands.append((gap, i, j, kind, s, r, exta, extb, (px, py)))

    cands.sort(key=lambda c: (round(c[0], 6), c[1], c[2]))
    used: set = set()
    ends: dict = {}
    junctions: list[Junction] = []

    def end_of(s: float) -> int:
        return 1 if s > 0 else 0

    def name(idx: int) -> str:
        return "ATEND" if idx == 1 else "ATSTART"

    for gap, i, j, kind, s, r, exta, extb, pt in cands:
        a, b = walls[i], walls[j]
        if kind == "X":
            junctions.append(Junction("X", a.wid, "ATPATH", b.wid, "ATPATH", pt, 0.0))
            continue
        if kind == "L":
            ka, kb = (a.wid, end_of(s)), (b.wid, end_of(r))
            if ka in used or kb in used:
                continue
            used.update([ka, kb])
            ends[ka] = {"kind": "L", "other": b.wid, "s_node": s, "ext": exta, "gap": gap}
            ends[kb] = {"kind": "L", "other": a.wid, "s_node": r, "ext": extb, "gap": gap}
            junctions.append(Junction("L", a.wid, name(ka[1]), b.wid, name(kb[1]), pt, gap))
        elif kind == "T_a":
            ka = (a.wid, end_of(s))
            if ka in used:
                continue
            used.add(ka)
            ends[ka] = {"kind": "T", "other": b.wid, "s_node": s, "ext": 0.0, "gap": gap}
            junctions.append(Junction("T", a.wid, name(ka[1]), b.wid, "ATPATH", pt, gap))
        else:  # T_b
            kb = (b.wid, end_of(r))
            if kb in used:
                continue
            used.add(kb)
            ends[kb] = {"kind": "T", "other": a.wid, "s_node": r, "ext": 0.0, "gap": gap}
            junctions.append(Junction("T", b.wid, name(kb[1]), a.wid, "ATPATH", pt, gap))
    return junctions, ends


# --------------------------------------------------------------------------- #
# adapter 1: polygons  (labels / model predictions)
# --------------------------------------------------------------------------- #
def _polygon_to_rect(wid: str, pts: np.ndarray):
    """Return (WallRect, solidity) for a polygon, or None if it is not a wall run."""
    if pts is None or len(pts) < 3:
        return None
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), _ = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    if min(w, h) < 1e-3:
        return None
    box = cv2.boxPoints(((cx, cy), (w, h), _))
    e0, e1 = box[1] - box[0], box[2] - box[1]
    edge = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
    L = float(np.hypot(*edge))
    ux, uy = float(edge[0] / L), float(edge[1] / L)
    T = float(min(w, h))
    area = float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))
    solidity = area / max(float(w * h), 1e-9)
    if L / max(T, 1e-9) < MIN_ASPECT:
        return None
    return WallRect(wid, float(cx), float(cy), ux, uy, L / 2.0, T), solidity


def _rebuild_polygon(w: WallRect, s0: float, s1: float) -> np.ndarray:
    nx, ny = -w.uy, w.ux
    h = w.t / 2.0
    p = lambda s, k: [w.cx + w.ux * s + nx * h * k, w.cy + w.uy * s + ny * h * k]
    return np.array([p(s0, -1), p(s1, -1), p(s1, 1), p(s0, 1)], dtype=np.float32)

def _wall_angle_diff_deg(a: WallRect, b: WallRect) -> float:
    """
    Return the smallest angle between two wall directions.

    0 degree  -> parallel / same direction
    90 degree -> perpendicular
    """
    dot = abs(a.ux * b.ux + a.uy * b.uy)
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(math.acos(dot))


def _same_physical_wall(a: WallRect, b: WallRect) -> bool:
    """
    Check whether two WallRect objects are most likely
    duplicate detections of the same physical wall.

    Important:
    - Only nearly-parallel walls are considered.
    - Their center lines must be close.
    - Their lengths must overlap substantially.
    - Very different wall thicknesses are not merged.

    This prevents L/T/X junction walls from being merged.
    """

    # ---------------------------------------------------------
    # 1. Direction must be almost parallel
    # ---------------------------------------------------------
    angle_diff = _wall_angle_diff_deg(a, b)

    if angle_diff > DUP_WALL_ANGLE_DEG:
        return False

    # ---------------------------------------------------------
    # 2. Thickness should be reasonably similar
    # ---------------------------------------------------------
    min_thickness = max(min(a.t, b.t), 1e-6)
    max_thickness = max(a.t, b.t)

    thickness_ratio = max_thickness / min_thickness

    if thickness_ratio > DUP_WALL_THICKNESS_RATIO:
        return False

    # ---------------------------------------------------------
    # 3. Center lines should be close
    # ---------------------------------------------------------
    # Perpendicular direction of wall A
    nx = -a.uy
    ny = a.ux

    dx = b.cx - a.cx
    dy = b.cy - a.cy

    perpendicular_distance = abs(
        dx * nx + dy * ny
    )

    allowed_distance = (
        DUP_WALL_CENTER_TOL_FACTOR
        * max_thickness
    )

    if perpendicular_distance > allowed_distance:
        return False

    # ---------------------------------------------------------
    # 4. Check longitudinal overlap
    # ---------------------------------------------------------
    # Project B's center onto A's direction.
    center_offset = (
        dx * a.ux
        + dy * a.uy
    )

    # Wall A interval
    a_start = -a.hl
    a_end = a.hl

    # Wall B interval expressed in A's coordinate system
    b_start = center_offset - b.hl
    b_end = center_offset + b.hl

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    overlap_length = max(
        0.0,
        overlap_end - overlap_start,
    )

    shorter_length = max(
        1.0,
        min(
            2.0 * a.hl,
            2.0 * b.hl,
        ),
    )

    overlap_ratio = (
        overlap_length / shorter_length
    )

    if overlap_ratio < DUP_WALL_OVERLAP_RATIO:
        return False

    return True


def _deduplicate_wall_polygons(polys_px):
    """
    Merge YOLO predictions that represent the same physical wall.

    Example:

        Wall #1  ====================
        Wall #2  ==================
        Wall #3   ==================

    becomes:

        Wall     =====================

    Only duplicate/overlapping parallel walls are merged.

    Perpendicular L/T/X walls are NOT merged.
    """

    if not polys_px:
        return polys_px, 0

    # ---------------------------------------------------------
    # Convert eligible polygons to WallRect
    # ---------------------------------------------------------
    rects = []

    for index, poly in enumerate(polys_px):

        result = _polygon_to_rect(
            f"p{index}",
            np.asarray(poly, dtype=np.float32).reshape(-1, 2),
        )

        if result is None:
            continue

        rect, solidity = result

        # Keep same eligibility rule as existing wall logic.
        if solidity < MIN_SOLIDITY:
            continue

        rects.append(
            (index, rect)
        )

    if len(rects) < 2:
        return polys_px, 0

    # ---------------------------------------------------------
    # Union-Find structure
    # ---------------------------------------------------------
    parent = {
        index: index
        for index, _ in rects
    }

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    # ---------------------------------------------------------
    # Find duplicate wall pairs
    # ---------------------------------------------------------
    for i in range(len(rects)):

        index_a, wall_a = rects[i]

        for j in range(i + 1, len(rects)):

            index_b, wall_b = rects[j]

            if _same_physical_wall(
                wall_a,
                wall_b,
            ):
                union(
                    index_a,
                    index_b,
                )

    # ---------------------------------------------------------
    # Build groups
    # ---------------------------------------------------------
    groups = {}

    for index, rect in rects:

        root = find(index)

        groups.setdefault(
            root,
            [],
        ).append(
            (index, rect)
        )

    # ---------------------------------------------------------
    # Create replacement polygons
    # ---------------------------------------------------------
    replacements = {}
    skipped_indices = set()

    duplicates_removed = 0

    for group in groups.values():

        # Nothing to merge
        if len(group) < 2:
            continue

        # Use the longest wall as reference.
        # This gives us a stable direction.
        base_index, base = max(
            group,
            key=lambda item: item[1].hl,
        )

        ux = base.ux
        uy = base.uy

        # Perpendicular direction
        nx = -uy
        ny = ux

        s_values = []
        perpendicular_offsets = []

        for _, wall in group:

            dx = wall.cx - base.cx
            dy = wall.cy - base.cy

            center_s = (
                dx * ux
                + dy * uy
            )

            center_p = (
                dx * nx
                + dy * ny
            )

            s_values.append(
                center_s - wall.hl
            )

            s_values.append(
                center_s + wall.hl
            )

            perpendicular_offsets.append(
                center_p
            )

        # -----------------------------------------------------
        # Combined longitudinal range
        # -----------------------------------------------------
        s_start = min(s_values)
        s_end = max(s_values)

        # Median perpendicular position.
        # This avoids shifting the merged wall because of
        # one slightly inaccurate YOLO prediction.
        perpendicular_offset = float(
            np.median(
                perpendicular_offsets
            )
        )

        center_s = (
            s_start + s_end
        ) / 2.0

        merged_cx = (
            base.cx
            + ux * center_s
            + nx * perpendicular_offset
        )

        merged_cy = (
            base.cy
            + uy * center_s
            + ny * perpendicular_offset
        )

        merged_half_length = (
            s_end - s_start
        ) / 2.0

        merged_thickness = max(
            wall.t
            for _, wall in group
        )

        merged_wall = WallRect(
            wid=base.wid,
            cx=merged_cx,
            cy=merged_cy,
            ux=ux,
            uy=uy,
            hl=merged_half_length,
            t=merged_thickness,
        )

        merged_polygon = _rebuild_polygon(
            merged_wall,
            -merged_wall.hl,
            merged_wall.hl,
        )

        # Put merged wall at the first occurrence.
        first_index = min(
            index
            for index, _ in group
        )

        replacements[
            first_index
        ] = merged_polygon

        # All remaining predictions in this group
        # are removed.
        for index, _ in group:

            if index != first_index:
                skipped_indices.add(index)

        duplicates_removed += (
            len(group) - 1
        )

    # ---------------------------------------------------------
    # Rebuild output list in original order
    # ---------------------------------------------------------
    output = []

    for index, poly in enumerate(polys_px):

        if index in skipped_indices:
            continue

        if index in replacements:
            output.append(
                replacements[index]
            )
        else:
            output.append(poly)

    return output, duplicates_removed

def connect_wall_polygons(
    polys_px: list,
    *,
    perp_tol_deg: float = PERP_TOL_DEG,
    gap_tol_factor: float = GAP_TOL_FACTOR,
):
    """
    polys_px: list of (N,2) or (N,1,2) arrays in pixels.
    Returns (new_polys, report).  new_polys has the same length/order as the input;
    walls that are not straight runs (L-shaped blobs, piers) are returned unchanged.
    """
     # Remove duplicate YOLO detections of the same physical wall
    input_wall_count = len(polys_px)
    polys_px, duplicates_removed = _deduplicate_wall_polygons(polys_px)

    rects: list[WallRect] = []
    idx_of: dict[str, int] = {}
    skipped = 0
    for i, poly in enumerate(polys_px):
        got = _polygon_to_rect(f"p{i}", np.asarray(poly, dtype=np.float32).reshape(-1, 2))
        if got is None or got[1] < MIN_SOLIDITY:
            skipped += 1
            continue
        rects.append(got[0])
        idx_of[got[0].wid] = i

    junctions, ends = find_junctions(rects, perp_tol_deg=perp_tol_deg, gap_tol_factor=gap_tol_factor)

    out = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in polys_px]
    changed = 0
    for w in rects:
        e0 = ends.get((w.wid, 0))
        e1 = ends.get((w.wid, 1))
        if not e0 and not e1:
            continue
        s0 = (e0["s_node"] - e0["ext"]) if e0 else -w.hl
        s1 = (e1["s_node"] + e1["ext"]) if e1 else w.hl
        out[idx_of[w.wid]] = _rebuild_polygon(w, s0, s1)
        changed += 1

    report = {
        "walls": len(polys_px),
        "input_walls": input_wall_count,
        "output_walls": len(polys_px),
        "duplicates_removed": duplicates_removed,
        
        "straight_runs": len(rects),
        "skipped_non_rect": skipped,
        "junctions_L": sum(1 for j in junctions if j.kind == "L"),
        "junctions_T": sum(1 for j in junctions if j.kind == "T"),
        "junctions_X": sum(1 for j in junctions if j.kind == "X"),
        "walls_extended": changed,
    }
    return out, report


def connect_yolo_label_lines(lines: list[str], W: int, H: int, wall_class_id: int = 3):
    """
    Convert normalised YOLO-seg lines: rewrite only the Wall lines so they are connected.
    Non-wall lines are passed through untouched (same order).  Returns (new_lines, report).
    """
    wall_pos, wall_polys = [], []
    for k, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 7 or int(float(parts[0])) != wall_class_id:
            continue
        c = list(map(float, parts[1:]))
        pts = np.array([[c[i] * W, c[i + 1] * H] for i in range(0, len(c) - 1, 2)], dtype=np.float32)
        wall_pos.append(k)
        wall_polys.append(pts)
    if not wall_polys:
        return list(lines), {"walls": 0}
    new_polys, report = connect_wall_polygons(wall_polys)
    out = list(lines)
    for k, poly in zip(wall_pos, new_polys):
        poly = np.clip(poly, [0, 0], [W - 1, H - 1])
        coords = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in poly)
        out[k] = f"{wall_class_id} {coords}"
    return out, report


def connect_wall_contours(contours: list, img_w: int, img_h: int):
    """
    cv2-style contours ((N,1,2) int32) in/out.

    Duplicate Wall detections may be removed inside
    connect_wall_polygons(), so do not map the new polygons
    back to the old contours using zip().
    """

    new, report = connect_wall_polygons([np.asarray(c).reshape(-1, 2) for c in contours])
    res = []

    for poly in new:
        poly = np.clip(np.rint(poly),[0, 0],[img_w - 1, img_h - 1]).astype(np.int32)
        res.append(poly.reshape(-1, 1, 2))

    return res, report


# --------------------------------------------------------------------------- #
# adapter 2: centerline segments  (IFC pipeline)
# --------------------------------------------------------------------------- #
def _undisplaced_axis(s: dict):
    """
    Wall axis taken from the PRE-snap centerline (centroid snapping in
    rebuild_nodes_from_segments moves whole walls sideways by up to a few px),
    with the along-axis extent taken from the current (snapped) polyline so
    closed gaps stay closed.  Returns (ox, oy, ux, uy, t0, t1) or None.
    """
    cur0, cur1 = s["polyline_px"][0], s["polyline_px"][-1]
    pre = s.get("centerline_px_pre_snap") or [cur0, cur1]
    a0, a1 = pre[0], pre[-1]
    dx, dy = float(a1[0]) - float(a0[0]), float(a1[1]) - float(a0[1])
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L
    ox, oy = float(a0[0]), float(a0[1])
    t0 = (float(cur0[0]) - ox) * ux + (float(cur0[1]) - oy) * uy
    t1 = (float(cur1[0]) - ox) * ux + (float(cur1[1]) - oy) * uy
    if t1 - t0 < 1e-6:            # snapping flipped / collapsed the run: fall back to pre-snap extent
        t0, t1 = 0.0, L
    return ox, oy, ux, uy, t0, t1


def connect_wall_segments(
    segs: list[dict],
    mpp: float,
    *,
    perp_tol_deg: float = PERP_TOL_DEG,
    gap_tol_factor: float = GAP_TOL_FACTOR,
):
    """
    segs: ifc_pipeline segments (need id, polyline_px, thickness_m; centerline_px_pre_snap optional).
    Mutated in place, only for walls that take part in an L / T junction:
      - the end that meets the junction is moved to the exact centerline intersection,
      - the wall axis is restored to its pre-snap position (undoes sideways centroid drift),
      - ext_start_px / ext_end_px = extra length (px) each end must be extended by so the
        outer corner is filled (L only).
    Returns (junction_dicts, diagnostics).
    """
    rects: list[WallRect] = []
    by_id: dict[str, dict] = {}
    axes: dict[str, tuple] = {}
    for s in segs:
        s["ext_start_px"] = 0.0
        s["ext_end_px"] = 0.0
        ax = _undisplaced_axis(s)
        if ax is None:
            continue
        ox, oy, ux, uy, t0, t1 = ax
        t_m = min(max(float(s.get("thickness_m", 0.23)), THICK_MIN_M), THICK_MAX_M)
        tm = (t0 + t1) / 2.0
        rects.append(WallRect(s["id"], ox + ux * tm, oy + uy * tm, ux, uy, (t1 - t0) / 2.0, t_m / mpp))
        by_id[s["id"]] = s
        axes[s["id"]] = ax

    junctions, ends = find_junctions(rects, perp_tol_deg=perp_tol_deg, gap_tol_factor=gap_tol_factor)

    moved = 0
    for w in rects:
        e0, e1 = ends.get((w.wid, 0)), ends.get((w.wid, 1))
        if not e0 and not e1:
            continue                       # untouched: keep the existing (snapped) polyline
        s = by_id[w.wid]
        p0 = [w.cx - w.ux * w.hl, w.cy - w.uy * w.hl]
        p1 = [w.cx + w.ux * w.hl, w.cy + w.uy * w.hl]
        if e0:
            p0 = [w.cx + w.ux * e0["s_node"], w.cy + w.uy * e0["s_node"]]
            s["ext_start_px"] = float(e0["ext"]) if e0["kind"] == "L" else 0.0
            moved += 1
        if e1:
            p1 = [w.cx + w.ux * e1["s_node"], w.cy + w.uy * e1["s_node"]]
            s["ext_end_px"] = float(e1["ext"]) if e1["kind"] == "L" else 0.0
            moved += 1
        s["polyline_px"] = [p0, p1]
        s["length_px"] = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    jd = [
        {
            "kind": j.kind,
            "a": j.a,
            "a_end": j.a_end,
            "b": j.b,
            "b_end": j.b_end,
            "point_px": [float(j.point[0]), float(j.point[1])],
            "gap_px": float(j.gap_px),
        }
        for j in junctions
    ]
    diag = {
        "enabled": True,
        "L": sum(1 for j in junctions if j.kind == "L"),
        "T": sum(1 for j in junctions if j.kind == "T"),
        "X": sum(1 for j in junctions if j.kind == "X"),
        "ends_connected": moved,
        "max_gap_bridged_px": max([j.gap_px for j in junctions] or [0.0]),
    }
    return jd, diag
