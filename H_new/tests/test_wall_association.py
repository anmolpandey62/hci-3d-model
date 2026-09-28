#!/usr/bin/env python3
"""Unit tests for P4 opening↔wall association scoring."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    associate_openings,
    candidate_acceptance_valid,
    longitudinal_overlap_ratio,
    opening_d_max,
    orientation_alignment_score,
    score_opening_wall_candidate,
)


def _wall(wid, a, b, thick=0.23, length=None):
    ax, ay = a
    bx, by = b
    if length is None:
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    return {
        "id": wid,
        "polyline_m": [list(a), list(b)],
        "thickness_m": thick,
        "length_m": length,
        "polyline_px": [[ax / 0.01, ay / 0.01], [bx / 0.01, by / 0.01]],
        "thickness_px": thick / 0.01,
        "length_px": length / 0.01,
        "start_node_id": "n0",
        "end_node_id": "n1",
        "source_polygon_id": f"c3_{wid}",
    }


def _opening_poly(oid, cx, cy, w=0.9, h=0.2, class_id=2):
    # axis-aligned rectangle around centroid in metres → fake px with mpp=0.01 later via points
    # associate_openings expects points_px; use mpp=0.01 ⇒ px = m/0.01
    mpp = 0.01
    H = 1000
    # image y: cy_m = (H - y)*mpp ⇒ y = H - cy_m/mpp
    px = cx / mpp
    py = H - cy / mpp
    hw, hh = (w / 2) / mpp, (h / 2) / mpp
    pts = [
        [px - hw, py - hh],
        [px + hw, py - hh],
        [px + hw, py + hh],
        [px - hw, py + hh],
        [px - hw, py - hh],
    ]
    return {"id": oid, "class": class_id, "points_px": pts, "area_px": (2 * hw) * (2 * hh)}


class TestP4Scoring(unittest.TestCase):
    def test_interior_beats_endpoint(self):
        # Same distance-ish: interior should score higher
        si = score_opening_wall_candidate(
            distance_m=0.2, d_max=0.55, t=0.5, overlap_ratio=0.8, orientation_score=0.9, duplicate_group_size=1
        )
        se = score_opening_wall_candidate(
            distance_m=0.15, d_max=0.55, t=1.0, overlap_ratio=0.1, orientation_score=0.9, duplicate_group_size=1
        )
        self.assertTrue(si["is_interior"])
        self.assertFalse(se["is_interior"])
        self.assertGreater(si["score"], se["score"])

    def test_overlap_beats_nearby_endpoint(self):
        good = score_opening_wall_candidate(
            distance_m=0.25, d_max=0.55, t=0.4, overlap_ratio=0.9, orientation_score=0.7, duplicate_group_size=1
        )
        bad = score_opening_wall_candidate(
            distance_m=0.10, d_max=0.55, t=0.0, overlap_ratio=0.05, orientation_score=0.7, duplicate_group_size=1
        )
        self.assertGreater(good["score"], bad["score"])
        self.assertFalse(
            candidate_acceptance_valid(
                distance_m=0.10, d_max=0.55, is_interior=False, overlap_ratio=0.05, thickness_m=0.23
            )
        )

    def test_endpoint_only_rejected(self):
        self.assertFalse(
            candidate_acceptance_valid(
                distance_m=0.40, d_max=0.55, is_interior=False, overlap_ratio=0.1, thickness_m=0.23
            )
        )
        # even with overlap, distance must be tight for endpoint
        self.assertFalse(
            candidate_acceptance_valid(
                distance_m=0.50, d_max=0.55, is_interior=False, overlap_ratio=0.5, thickness_m=0.23
            )
        )

    def test_endpoint_accepted_with_overlap_and_tight_d(self):
        self.assertTrue(
            candidate_acceptance_valid(
                distance_m=0.20, d_max=0.55, is_interior=False, overlap_ratio=0.5, thickness_m=0.23
            )
        )

    def test_duplicate_no_encounter_order(self):
        # Identical geometry walls: deterministic smallest id wins among equal scores
        H = 1000
        mpp = 0.01
        # long wall and short coincident endpoint wall
        walls = [
            _wall("w28", [0.0, 5.0], [0.5, 5.0], thick=0.29),  # short
            _wall("w2", [0.0, 5.0], [0.5, 5.0], thick=0.29),  # same geom, smaller id
            _wall("w10", [0.0, 5.0], [4.0, 5.0], thick=0.29),  # long interior host
        ]
        graph = {"segments": walls, "nodes": [], "diagnostics": {"topology": {}}}
        # Opening near middle of long wall
        door = _opening_poly("c2_0", cx=2.0, cy=5.15, w=0.9, h=0.25)
        wlist: list = []
        res = associate_openings(graph, [door], [], H, mpp, wlist)
        self.assertEqual(res["successfully_mapped"], 1)
        self.assertEqual(res["mappings"][0]["host_wall_id"], "w10")
        self.assertEqual(res["mappings"][0]["projection_class"], "interior")

    def test_deterministic_tie_break_smallest_id(self):
        H = 1000
        mpp = 0.01
        walls = [
            _wall("w5", [0.0, 1.0], [3.0, 1.0], thick=0.25),
            _wall("w1", [0.0, 1.0], [3.0, 1.0], thick=0.25),
        ]
        graph = {"segments": walls, "nodes": [], "diagnostics": {"topology": {}}}
        door = _opening_poly("c2_0", cx=1.5, cy=1.1, w=0.8, h=0.2)
        r1 = associate_openings(graph, [door], [], H, mpp, [])
        r2 = associate_openings({"segments": list(reversed(walls)), "nodes": [], "diagnostics": {"topology": {}}}, [door], [], H, mpp, [])
        self.assertEqual(r1["mappings"][0]["host_wall_id"], "w1")
        self.assertEqual(r2["mappings"][0]["host_wall_id"], "w1")

    def test_far_opening_unmapped(self):
        H = 1000
        mpp = 0.01
        walls = [_wall("w0", [0.0, 0.0], [5.0, 0.0], thick=0.23)]
        graph = {"segments": walls, "nodes": [], "diagnostics": {"topology": {}}}
        door = _opening_poly("c2_0", cx=2.5, cy=2.0, w=0.9, h=0.2)  # 2m away
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 0)
        self.assertEqual(res["diagnostics"]["openings"][0]["rejection_reason"], "TOO_FAR")

    def test_valid_nearby_still_maps(self):
        H = 1000
        mpp = 0.01
        walls = [_wall("w0", [0.0, 0.0], [5.0, 0.0], thick=0.23)]
        graph = {"segments": walls, "nodes": [], "diagnostics": {"topology": {}}}
        door = _opening_poly("c2_0", cx=2.5, cy=0.15, w=0.9, h=0.2)
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["successfully_mapped"], 1)
        self.assertEqual(res["mappings"][0]["host_wall_id"], "w0")
        self.assertLess(res["mappings"][0]["t"], 1.0 - 0.08)
        self.assertGreater(res["mappings"][0]["t"], 0.08)

    def test_zero_walls_no_walls(self):
        H = 1000
        mpp = 0.01
        graph = {"segments": [], "nodes": [], "diagnostics": {"topology": {}}}
        door = _opening_poly("c2_0", cx=1.0, cy=1.0)
        res = associate_openings(graph, [door], [], H, mpp, [])
        self.assertEqual(res["diagnostics"]["openings"][0]["rejection_reason"], "NO_WALLS")

    def test_d_max_not_loosened(self):
        self.assertAlmostEqual(opening_d_max(0.23), max(0.55, 1.25 * 0.23 + 0.10), places=6)
        self.assertLess(opening_d_max(0.23), 1.0)

    def test_orientation_and_overlap_helpers(self):
        self.assertGreater(orientation_alignment_score(0, 5), 0.9)
        ov = longitudinal_overlap_ratio(1.0, 0.1, 0.8, 0.0, 0.0, 0.0, 3.0, 0.0)
        self.assertGreater(ov, 0.5)


if __name__ == "__main__":
    unittest.main()
