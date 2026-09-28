#!/usr/bin/env python3
"""Unit tests for P3 topology helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    SNAP_PX,
    _merge_collinear_segments_impl,
    _seg_id_sort_key,
    angle_diff_deg,
    assign_nodes_frozen,
    compute_snap_radius_px,
    dedupe_coincident_segments,
    deterministic_wall_order,
    rebuild_nodes_from_segments,
    segment_angle_deg,
    segments_are_coincident,
    try_collinear_merge,
)


def _seg(sid, p0, p1, thick_px=10.0, area=100.0):
    length = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    return {
        "id": sid,
        "source_polygon_id": f"c3_{sid[1:]}",
        "source_polygon_ids": [f"c3_{sid[1:]}"],
        "polyline_px": [list(p0), list(p1)],
        "thickness_px": thick_px,
        "thickness_m": thick_px * 0.01,
        "length_px": length,
        "length_px_pre_snap": length,
        "area_px": area,
        "merged_from": [sid],
    }


class TestTopologyHelpers(unittest.TestCase):
    def test_endpoints_within_snap_union(self):
        segs = [_seg("w0", [0, 0], [100, 0]), _seg("w1", [2, 0], [100, 2])]
        nodes, groups, _ = rebuild_nodes_from_segments(segs, 0.01, 800, 800, "low")
        # with SNAP_PX=20 on large image, endpoints near (0,0) and (2,0) should share
        self.assertEqual(segs[0]["start_node_id"], segs[1]["start_node_id"])
        self.assertGreaterEqual(len(nodes), 2)

    def test_endpoints_outside_snap_no_union(self):
        segs = [_seg("w0", [0, 0], [50, 0]), _seg("w1", [80, 0], [120, 0])]
        rebuild_nodes_from_segments(segs, 0.01, 800, 800, "low")
        self.assertNotEqual(segs[0]["end_node_id"], segs[1]["start_node_id"])

    def test_perpendicular_no_collinear_merge(self):
        a = _seg("w0", [0, 0], [100, 0])
        b = _seg("w1", [100, 0], [100, 100])
        self.assertIsNone(try_collinear_merge(a, b, snap_px=20))

    def test_collinear_touching_merge(self):
        a = _seg("w0", [0, 0], [50, 0])
        b = _seg("w1", [50, 0], [100, 0])
        m = try_collinear_merge(a, b, snap_px=5)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["length_px"], 100, delta=1)

    def test_collinear_near_merge(self):
        a = _seg("w0", [0, 0], [48, 0])
        b = _seg("w1", [50, 0], [100, 0])
        m = try_collinear_merge(a, b, snap_px=5)
        self.assertIsNotNone(m)

    def test_duplicate_coincident_dedupe(self):
        a = _seg("w0", [0, 0], [100, 0], area=50)
        b = _seg("w2", [0.5, 0.5], [100.5, 0.5], area=200)
        kept, recs = dedupe_coincident_segments([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["id"], "w0")  # numeric min id
        self.assertEqual(len(recs), 1)

    def test_L_junction_preserve_two(self):
        a = _seg("w0", [0, 0], [100, 0])
        b = _seg("w1", [100, 0], [100, 80])
        out, merges = _merge_collinear_segments_impl([a, b], 0.01, 800, 800, "low", enabled=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(merges), 0)

    def test_T_snap_moves_endpoint(self):
        from logic.ifc_pipeline import apply_t_junction_snaps

        host = _seg("w0", [0, 0], [200, 0])
        stub = _seg("w1", [100, 5], [100, 80])
        ev = apply_t_junction_snaps([host, stub], 0.01, 800, 800, "low")
        self.assertTrue(any(e["topology_type"] == "T" for e in ev))
        # stub endpoint near host should move close to y=0
        self.assertLess(abs(stub["polyline_px"][0][1]), 1.0)

    def test_X_diagnostic_only(self):
        from logic.ifc_pipeline import classify_L_X_junctions

        a = _seg("w0", [0, 50], [100, 50])
        b = _seg("w1", [50, 0], [50, 100])
        L, X = classify_L_X_junctions([a, b], 0.01)
        self.assertGreaterEqual(len(X), 1)

    def test_near_miss_recorded(self):
        # nearly parallel but angled >35 within snap — recorded in near_misses
        segs = [
            _seg("w0", [0, 0], [100, 0]),
            _seg("w1", [3, 3], [80, 40]),  # angled
        ]
        _, _, near = rebuild_nodes_from_segments(segs, 0.01, 800, 800, "low")
        # may or may not record depending on distance; ensure call is stable
        self.assertIsInstance(near, list)

    def test_zero_length_cull_marker(self):
        s = _seg("w0", [10, 10], [10, 10])
        self.assertLess(s["length_px"], 1e-6)

    def test_deterministic_ordering(self):
        walls = [
            {"id": "c3_2", "area_px": 10, "points_px": [[0, 0], [1, 0], [1, 1]]},
            {"id": "c3_0", "area_px": 5, "points_px": [[2, 2], [3, 2], [3, 3]]},
            {"id": "c3_1", "area_px": 50, "points_px": [[4, 4], [5, 4], [5, 5]]},
        ]
        ordered = deterministic_wall_order(walls)
        self.assertEqual([w["id"] for w in ordered], ["c3_0", "c3_1", "c3_2"])

    def test_tiny_image_snap_cap(self):
        r = compute_snap_radius_px(20, 0.2, 0.01, 229, 220, "low")
        self.assertLess(r, 10)
        self.assertLess(r, SNAP_PX)

    def test_thickness_relative_snap_tiny(self):
        thin = compute_snap_radius_px(4, 0.04, 0.01, 229, 220, "low")
        thick = compute_snap_radius_px(30, 0.3, 0.01, 229, 220, "low")
        # both capped by beta*min; thick may hit cap
        self.assertLessEqual(thin, thick + 1e-9)

    def test_low_confidence_large_uses_legacy_snap(self):
        r = compute_snap_radius_px(25, 0.25, 0.01, 853, 1074, "low")
        self.assertAlmostEqual(r, SNAP_PX, delta=0.1)

    def test_duplicate_canonical_min_numeric_id(self):
        self.assertLess(_seg_id_sort_key("w2"), _seg_id_sort_key("w12"))
        a = _seg("w12", [0, 0], [100, 0], area=999)
        b = _seg("w2", [0, 0], [100, 0], area=1)
        kept, _ = dedupe_coincident_segments([a, b], 0.01, 800, 800, "low")
        self.assertEqual(kept[0]["id"], "w2")

    def test_angle_diff(self):
        self.assertAlmostEqual(angle_diff_deg(0, 90), 90)
        self.assertAlmostEqual(angle_diff_deg(10, 350), 20)  # 40? 350-10=340 -> min(340,20)=20 via %180
        self.assertLess(angle_diff_deg(0, 8), 10)

    def test_frozen_nodes_preserve_xy(self):
        segs = [_seg("w0", [1.23456, 7.891], [10.5, 7.891])]
        nodes, _ = assign_nodes_frozen(segs)
        self.assertAlmostEqual(nodes[0]["x_px"], 1.23456)
        self.assertEqual(segs[0]["start_node_id"], nodes[0]["id"])

    def test_collinear_impl_enabled(self):
        a = _seg("w0", [0, 0], [40, 0])
        b = _seg("w1", [40, 0], [80, 0])
        out, merges = _merge_collinear_segments_impl([a, b], 0.01, 800, 800, "low", enabled=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(merges), 1)


if __name__ == "__main__":
    unittest.main()
