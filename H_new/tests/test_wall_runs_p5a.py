#!/usr/bin/env python3
"""Unit tests for P5A collinear/coincident wall-run reconstruction."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    associate_openings,
    can_form_wall_run,
    reconstruct_wall_runs_p5a,
    stable_geometry_fingerprint,
    union_wall_run_group,
)


def _seg(sid, p0, p1, thick_px=10.0, thick_m=None, area=100.0, pre=None):
    length = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    if thick_m is None:
        thick_m = thick_px * 0.01
    return {
        "id": sid,
        "source_polygon_id": f"c3_{sid[1:]}",
        "source_polygon_ids": [f"c3_{sid[1:]}"],
        "polyline_px": [list(p0), list(p1)],
        "centerline_px_pre_snap": [list(pre[0]), list(pre[1])] if pre else [list(p0), list(p1)],
        "thickness_px": thick_px,
        "thickness_m": thick_m,
        "length_px": length,
        "length_px_pre_snap": length,
        "area_px": area,
        "merged_from": [sid],
        "topology_status": "ACTIVE",
    }


class TestP5AWallRuns(unittest.TestCase):
    def test_overlapping_collinear_merge(self):
        a = _seg("w0", [0, 0], [60, 0])
        b = _seg("w1", [40, 0], [100, 0])
        out, recs = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 1)
        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(out[0]["length_px"], 100, delta=1)
        self.assertEqual(out[0]["id"], "w0")

    def test_touching_collinear_merge(self):
        a = _seg("w0", [0, 0], [50, 0])
        b = _seg("w1", [50, 0], [100, 0])
        out, recs = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["length_px"], 100, delta=1)

    def test_near_collinear_outside_tol_separate(self):
        a = _seg("w0", [0, 0], [80, 0])
        b = _seg("w1", [0, 40], [80, 40])  # parallel, far
        out, _ = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 2)

    def test_parallel_room_walls_separate(self):
        a = _seg("w0", [0, 0], [200, 0], thick_m=0.2, thick_px=20)
        b = _seg("w1", [0, 150], [200, 150], thick_m=0.2, thick_px=20)
        self.assertFalse(can_form_wall_run(a, b, tol_px=20))
        out, _ = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 2)

    def test_L_junction_two_segments(self):
        a = _seg("w0", [0, 0], [100, 0])
        b = _seg("w1", [100, 0], [100, 80])
        out, recs = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 2)
        self.assertEqual(len(recs), 0)

    def test_T_junction_not_one_run(self):
        host = _seg("w0", [0, 0], [200, 0])
        stub = _seg("w1", [100, 0], [100, 40])  # perpendicular T
        out, _ = reconstruct_wall_runs_p5a([host, stub], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 2)

    def test_X_crossing_not_merged(self):
        a = _seg("w0", [0, 50], [100, 50])
        b = _seg("w1", [50, 0], [50, 100])
        out, _ = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 2)

    def test_duplicate_coincident_one_canonical(self):
        a = _seg("w2", [0, 0], [50, 0], area=10)
        b = _seg("w12", [0.5, 0.2], [50.5, 0.2], area=99)
        out, recs = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["id"], "w2")

    def test_union_not_longest_only(self):
        # Short + long overlapping: union spans full extent
        short = _seg("w0", [40, 0], [60, 0])
        long = _seg("w1", [0, 0], [100, 0])
        merged = union_wall_run_group([short, long])
        self.assertAlmostEqual(merged["length_px"], 100, delta=1)
        self.assertEqual(merged["id"], "w0")  # smallest id

    def test_deterministic_id(self):
        a = _seg("w5", [0, 0], [50, 0])
        b = _seg("w1", [45, 0], [90, 0])
        out1, _ = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        out2, _ = reconstruct_wall_runs_p5a([b, a], 0.01, 800, 800, "low")
        self.assertEqual(out1[0]["id"], "w1")
        self.assertEqual(out2[0]["id"], "w1")
        self.assertAlmostEqual(out1[0]["length_px"], out2[0]["length_px"], places=4)

    def test_presnap_union_extends_collapsed_shorts(self):
        # Post-snap both collapsed to short identical line; pre-snap spans longer wall
        a = _seg(
            "w2",
            [300, 870],
            [360, 870],
            pre=[[100, 870], [500, 870]],
        )
        b = _seg(
            "w4",
            [300, 870],
            [360, 870],
            pre=[[200, 870], [450, 870]],
        )
        out, recs = reconstruct_wall_runs_p5a([a, b], 0.01, 900, 1100, "low")
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0]["length_px"], 300)
        self.assertEqual(out[0]["id"], "w2")

    def test_opening_moves_toward_interior_after_run(self):
        # Short coincident walls → endpoint; after union of long pre-snap → interior
        H = 1000
        mpp = 0.01
        shorts = [
            _seg("w2", [298, 870], [358, 870], thick_m=0.29, thick_px=29, pre=[[100, 870], [600, 870]]),
            _seg("w4", [298, 870], [358, 870], thick_m=0.29, thick_px=29, pre=[[150, 870], [550, 870]]),
        ]
        runs, _ = reconstruct_wall_runs_p5a(shorts, mpp, 900, 1100, "low")
        # Build metre polylines for association
        for s in runs:
            s["polyline_m"] = [[p[0] * mpp, (H - p[1]) * mpp] for p in s["polyline_px"]]
            s["length_m"] = s["length_px"] * mpp
        graph = {"segments": runs, "nodes": [], "diagnostics": {"topology": {}}}
        # Opening near middle of reconstructed run (px y=855 → metres)
        cx_px, cy_px = 350.0, 855.0
        pts = [
            [cx_px - 40, cy_px - 10],
            [cx_px + 40, cy_px - 10],
            [cx_px + 40, cy_px + 10],
            [cx_px - 40, cy_px + 10],
            [cx_px - 40, cy_px - 10],
        ]
        door = {"id": "c2_0", "class": 2, "points_px": pts, "area_px": 1600}
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 1)
        self.assertEqual(res["mappings"][0]["projection_class"], "interior")
        self.assertGreater(res["mappings"][0]["t"], 0.08)
        self.assertLess(res["mappings"][0]["t"], 0.92)

    def test_far_opening_still_unmapped(self):
        H = 1000
        mpp = 0.01
        a = _seg("w0", [0, 900], [200, 900], thick_m=0.23, thick_px=23)
        runs, _ = reconstruct_wall_runs_p5a([a], mpp, 800, 800, "low")
        for s in runs:
            s["polyline_m"] = [[p[0] * mpp, (H - p[1]) * mpp] for p in s["polyline_px"]]
            s["length_m"] = s["length_px"] * mpp
        graph = {"segments": runs, "nodes": [], "diagnostics": {"topology": {}}}
        door = {
            "id": "c2_0",
            "class": 2,
            "points_px": [[100, 500], [180, 500], [180, 520], [100, 520], [100, 500]],
            "area_px": 1600,
        }
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 0)

    def test_fingerprint_stable(self):
        a = _seg("w0", [0, 0], [60, 0], pre=[[0, 0], [60, 0]])
        b = _seg("w1", [50, 0], [120, 0], pre=[[50, 0], [120, 0]])
        g1, _ = reconstruct_wall_runs_p5a([a, b], 0.01, 800, 800, "low")
        g2, _ = reconstruct_wall_runs_p5a([dict(a), dict(b)], 0.01, 800, 800, "low")
        for s in g1 + g2:
            s["polyline_m"] = [[p[0] * 0.01, p[1] * 0.01] for p in s["polyline_px"]]
            s["length_m"] = s["length_px"] * 0.01
        fp1 = stable_geometry_fingerprint({"segments": g1, "nodes": []}, {"mappings": [], "unmapped": []})
        fp2 = stable_geometry_fingerprint({"segments": g2, "nodes": []}, {"mappings": [], "unmapped": []})
        self.assertEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
