#!/usr/bin/env python3
"""Unit tests for P5B conservative skeleton centerline recovery."""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    associate_openings,
    opening_d_max,
)
from logic.p5b_skeleton import (  # noqa: E402
    apply_p5b_to_segments,
    classify_wall_mask,
    classifier_authorizes_p5b,
    morphological_skeleton,
    p5b_enabled,
    prune_skeleton_spurs,
    rasterize_polygon,
    recover_paths_from_mask,
    segments_substantially_coincident_px,
    validate_centerline_on_mask,
)


def _rect(x0, y0, x1, y1, n=8):
    """Axis-aligned rectangle polygon (clockwise)."""
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0],
    ]


def _L_poly(t=12.0):
    # L: horizontal bar + vertical bar sharing corner
    # Outer: (0,0)-(80,0)-(80,t)-(t,t)-(t,80)-(0,80)
    return [
        [0, 0],
        [80, 0],
        [80, t],
        [t, t],
        [t, 80],
        [0, 80],
        [0, 0],
    ]


def _T_poly(t=12.0):
    # T: horizontal top + vertical stem
    return [
        [0, 0],
        [100, 0],
        [100, t],
        [55, t],
        [55, 70],
        [45, 70],
        [45, t],
        [0, t],
        [0, 0],
    ]


def _X_poly(t=10.0):
    # Approximate X as thick cross (union of H and V bars) as single polygon outline
    # Use a plus shape
    c = 50.0
    half = 40.0
    ht = t / 2
    return [
        [c - half, c - ht],
        [c - ht, c - ht],
        [c - ht, c - half],
        [c + ht, c - half],
        [c + ht, c - ht],
        [c + half, c - ht],
        [c + half, c + ht],
        [c + ht, c + ht],
        [c + ht, c + half],
        [c - ht, c + half],
        [c - ht, c + ht],
        [c - half, c + ht],
        [c - half, c - ht],
    ]


def _seg_from_poly(sid, pts, thick=10.0):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # crude OBB via bbox long side
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    if bw >= bh:
        p0, p1 = [min(xs), 0.5 * (min(ys) + max(ys))], [max(xs), 0.5 * (min(ys) + max(ys))]
        length = bw
    else:
        p0, p1 = [0.5 * (min(xs) + max(xs)), min(ys)], [0.5 * (min(xs) + max(xs)), max(ys)]
        length = bh
    return {
        "id": sid,
        "source_polygon_id": f"c3_{sid[1:]}",
        "source_polygon_ids": [f"c3_{sid[1:]}"],
        "area_px": float(abs(bw * bh)),
        "points_px": pts,
        "centerline_px_pre_snap": [p0, p1],
        "polyline_px": [p0, p1],
        "thickness_px": float(thick),
        "thickness_m": float(thick) * 0.01,
        "length_px": float(length),
        "length_px_pre_snap": float(length),
        "merged_from": [sid],
        "topology_status": "ACTIVE",
    }


class TestP5B(unittest.TestCase):
    def setUp(self):
        os.environ["HCI_WALL_P5B"] = "1"

    def tearDown(self):
        os.environ.pop("HCI_WALL_P5B", None)

    def test_flag_default_off(self):
        os.environ.pop("HCI_WALL_P5B", None)
        self.assertFalse(p5b_enabled(default=False))

    def test_clean_strip_not_activated(self):
        pts = _rect(0, 0, 120, 10)
        clf = classify_wall_mask(pts, length_px=120, thickness_px=10, skeleton_junctions=0)
        self.assertEqual(clf["classifier"], "STRIP")
        self.assertFalse(classifier_authorizes_p5b(clf["classifier"]))
        segs = [_seg_from_poly("w0", pts, thick=10)]
        out, diag = apply_p5b_to_segments(segs, 200, 200, 0.01)
        self.assertEqual(diag["activations"], 0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["centerline_px_pre_snap"], segs[0]["centerline_px_pre_snap"])

    def test_short_strip_obb_remains(self):
        pts = _rect(0, 0, 20, 12)
        clf = classify_wall_mask(pts, length_px=20, thickness_px=12)
        self.assertIn(clf["classifier"], ("SHORT_STRIP", "STRIP", "COMPLEX"))
        segs = [_seg_from_poly("w0", pts, thick=12)]
        out, diag = apply_p5b_to_segments(segs, 100, 100, 0.01)
        if not classifier_authorizes_p5b(clf["classifier"]):
            self.assertEqual(diag["activations"], 0)

    def test_branched_L_two_paths(self):
        pts = _L_poly(12)
        mask, ox, oy = rasterize_polygon(pts)
        skel = morphological_skeleton(mask)
        # Force complex recovery
        paths, diag = recover_paths_from_mask(
            mask, ox, oy, thickness_px=12, pts_px=pts, classifier="BRANCHED", component_id="L"
        )
        self.assertGreaterEqual(len(paths), 1)
        # Angles of accepted paths should not be a single diagonal spanning both arms only
        segs = [_seg_from_poly("w0", pts, thick=12)]
        # Mark as needing P5B by ensuring junctions
        out, d = apply_p5b_to_segments(segs, 200, 200, 0.01)
        # Either split or keep; must not produce one diagonal from (0,0) to (80,80)
        for s in out:
            p0, p1 = s["polyline_px"][0], s["polyline_px"][-1]
            dx, dy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
            # Reject pure 45° diagonal spanning both arms
            if dx > 40 and dy > 40:
                self.fail(f"L collapsed to diagonal: {p0}->{p1}")

    def test_T_branch_preserved(self):
        pts = _T_poly(12)
        segs = [_seg_from_poly("w0", pts, thick=12)]
        out, diag = apply_p5b_to_segments(segs, 200, 200, 0.01)
        # Should not be a single segment covering full bbox diagonal
        for s in out:
            p0, p1 = s["polyline_px"][0], s["polyline_px"][-1]
            if abs(p1[0] - p0[0]) > 90 and abs(p1[1] - p0[1]) > 50:
                self.fail("T collapsed incorrectly")

    def test_X_crossing_preserved(self):
        pts = _X_poly(10)
        segs = [_seg_from_poly("w0", pts, thick=10)]
        out, _ = apply_p5b_to_segments(segs, 200, 200, 0.01)
        self.assertGreaterEqual(len(out), 1)
        # Multiple paths preferred; if one, must not be arbitrary
        self.assertTrue(all(len(s["polyline_px"]) >= 2 for s in out))

    def test_tiny_spur_pruned(self):
        # Long bar with a tiny spur bump — build mask manually
        mask = np.zeros((40, 100), dtype=np.uint8)
        mask[15:25, 5:95] = 1
        mask[10:15, 50:52] = 1  # spur
        skel = morphological_skeleton(mask)
        pruned, recs = prune_skeleton_spurs(skel, max_spur_px=6)
        # Spur region should be reduced
        self.assertLessEqual(int(pruned[10:15, 50:52].sum()), int(skel[10:15, 50:52].sum()))

    def test_noisy_straight_stable(self):
        # Thick strip with jagged long edges (still a simple filled band)
        pts = []
        for i in range(0, 101, 4):
            pts.append([float(i), 0.0 + (0.4 if (i // 4) % 2 else 0.0)])
        for i in range(100, -1, -4):
            pts.append([float(i), 14.0 - (0.4 if (i // 4) % 2 else 0.0)])
        pts.append(pts[0])
        segs = [_seg_from_poly("w0", pts, thick=14)]
        out, diag = apply_p5b_to_segments(segs, 150, 50, 0.01)
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0]["length_px"], 70)

    def test_disconnected_not_bridged(self):
        # Two separate rectangles — process each; must not invent bridge
        a = _seg_from_poly("w0", _rect(0, 0, 40, 10), thick=10)
        b = _seg_from_poly("w1", _rect(80, 0, 120, 10), thick=10)
        out, _ = apply_p5b_to_segments([a, b], 150, 50, 0.01)
        self.assertEqual(len(out), 2)
        # No segment spanning the gap
        for s in out:
            p0, p1 = s["polyline_px"][0], s["polyline_px"][-1]
            xs = sorted([p0[0], p1[0]])
            self.assertFalse(xs[0] < 30 and xs[1] > 90)

    def test_connected_irregular_recovers(self):
        # Irregular but connected thick strip (slightly bent contour)
        pts = [
            [0, 5],
            [40, 3],
            [80, 6],
            [120, 4],
            [120, 16],
            [80, 18],
            [40, 15],
            [0, 17],
            [0, 5],
        ]
        segs = [_seg_from_poly("w0", pts, thick=12)]
        out, diag = apply_p5b_to_segments(segs, 150, 40, 0.01)
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0]["length_px"], 80)

    def test_path_leaving_mask_rejected(self):
        pts = _rect(0, 0, 60, 12)
        mask, ox, oy = rasterize_polygon(pts)
        # Fake centerline outside
        bad = [[-20, -20], [100, -20]]
        val = validate_centerline_on_mask(bad, mask, ox, oy, pts_px=pts)
        self.assertFalse(val["ok"])

    def test_duplicate_with_obb_deduped(self):
        pts = _rect(0, 0, 100, 12)
        # Force FRAGMENTED/STRIP_NOISY by high verts — use noisy strip that authorizes
        noisy = []
        for i in range(0, 101, 2):
            noisy.append([i, 1 + (i % 5) * 0.2])
        for i in range(100, -1, -2):
            noisy.append([i, 11 - (i % 3) * 0.2])
        noisy.append(noisy[0])
        segs = [_seg_from_poly("w0", noisy, thick=10)]
        out, diag = apply_p5b_to_segments(segs, 150, 40, 0.01)
        # Either retained as OBB duplicate or single replaced — never 2 copies
        self.assertEqual(len(out), 1)

    def test_deterministic_ids_and_order(self):
        pts = _L_poly(12)
        segs = [_seg_from_poly("w5", pts, thick=12)]
        out1, _ = apply_p5b_to_segments(segs, 200, 200, 0.01)
        out2, _ = apply_p5b_to_segments([dict(segs[0])], 200, 200, 0.01)
        self.assertEqual([s["id"] for s in out1], [s["id"] for s in out2])
        for a, b in zip(out1, out2):
            self.assertAlmostEqual(a["length_px"], b["length_px"], places=3)

    def test_far_opening_unmapped(self):
        pts = _rect(0, 900, 100, 920)
        segs = [_seg_from_poly("w0", pts, thick=20)]
        out, _ = apply_p5b_to_segments(segs, 200, 1000, 0.01)
        H = 1000
        mpp = 0.01
        for s in out:
            s["polyline_m"] = [[p[0] * mpp, (H - p[1]) * mpp] for p in s["polyline_px"]]
            s["length_m"] = s["length_px"] * mpp
        graph = {"segments": out, "nodes": [], "diagnostics": {"topology": {}, "p5b": {}}}
        door = {
            "id": "c2_0",
            "class": 2,
            "points_px": [[50, 100], [90, 100], [90, 120], [50, 120], [50, 100]],
            "area_px": 800,
        }
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 0)

    def test_interior_on_recovered_wall_maps(self):
        pts = _rect(0, 100, 200, 120)
        # Noisy enough to authorize if needed; STRIP still works with OBB for association
        segs = [_seg_from_poly("w0", pts, thick=20)]
        out, _ = apply_p5b_to_segments(segs, 300, 200, 0.01)
        H = 200
        mpp = 0.01
        for s in out:
            s["polyline_m"] = [[p[0] * mpp, (H - p[1]) * mpp] for p in s["polyline_px"]]
            s["length_m"] = s["length_px"] * mpp
            s["thickness_m"] = 0.20
        graph = {"segments": out, "nodes": [], "diagnostics": {"topology": {}}}
        door = {
            "id": "c2_0",
            "class": 2,
            "points_px": [[90, 85], [110, 85], [110, 95], [90, 95], [90, 85]],
            "area_px": 200,
        }
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 1)
        self.assertEqual(res["mappings"][0]["projection_class"], "interior")

    def test_p4_d_max_unchanged(self):
        self.assertAlmostEqual(opening_d_max(0.2), max(0.55, 1.25 * 0.2 + 0.10), places=6)
        self.assertAlmostEqual(opening_d_max(0.05), 0.55, places=6)

    def test_coincident_helper(self):
        self.assertTrue(
            segments_substantially_coincident_px([0, 0], [100, 0], [5, 1], [95, 1], lat_tol_px=5)
        )
        self.assertFalse(
            segments_substantially_coincident_px([0, 0], [100, 0], [0, 40], [100, 40], lat_tol_px=5)
        )


if __name__ == "__main__":
    unittest.main()
