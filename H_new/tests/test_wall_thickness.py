#!/usr/bin/env python3
"""Unit tests for P2 wall thickness helpers (perpendicular + scale-aware)."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.ifc_pipeline import (  # noqa: E402
    THICK_FALLBACK,
    apply_plan_and_global_thickness_fallback,
    estimate_wall_thickness_candidate,
    min_area_centerline,
    robust_median_width_px,
    sample_perpendicular_widths_px,
    soft_clip_thickness_m,
    thickness_px_sane,
    thickness_sample_count,
)


def axis_aligned_strip(x0, y0, length, thickness, horizontal=True):
    """Closed rectangle polygon approximating a wall strip."""
    t = thickness
    L = length
    if horizontal:
        return [
            [x0, y0],
            [x0 + L, y0],
            [x0 + L, y0 + t],
            [x0, y0 + t],
            [x0, y0],
        ]
    return [
        [x0, y0],
        [x0 + t, y0],
        [x0 + t, y0 + L],
        [x0, y0 + L],
        [x0, y0],
    ]


class TestThicknessHelpers(unittest.TestCase):
    def test_sample_count_bounds(self):
        self.assertEqual(thickness_sample_count(10), 5)
        self.assertEqual(thickness_sample_count(100), 10)
        self.assertEqual(thickness_sample_count(500), 21)

    def test_3px_thin_strip_not_global_fallback(self):
        pts = axis_aligned_strip(10, 10, length=120, thickness=3, horizontal=True)
        cl, _obb, length_px = min_area_centerline(pts)
        est = estimate_wall_thickness_candidate(
            pts, cl, length_px, W=400, H=400, mpp=0.01, scale_confidence="low"
        )
        self.assertIsNotNone(est["thickness_m"])
        self.assertNotEqual(est["thickness_method"], "fallback_global")
        self.assertNotAlmostEqual(est["thickness_m"], THICK_FALLBACK, places=3)
        # ~3 px * 0.01 = 0.03 m — must NOT become 0.23
        self.assertLess(est["thickness_m"], 0.08)
        self.assertGreater(est["thickness_m"], 0.015)

    def test_5px_thin_strip_not_global_fallback(self):
        pts = axis_aligned_strip(10, 10, length=150, thickness=5, horizontal=True)
        cl, _, length_px = min_area_centerline(pts)
        est = estimate_wall_thickness_candidate(
            pts, cl, length_px, W=400, H=400, mpp=0.01, scale_confidence="low"
        )
        self.assertIn(est["thickness_method"], ("perpendicular_median", "obb"))
        self.assertAlmostEqual(est["thickness_m"], 0.05, delta=0.02)

    def test_20px_normal_strip(self):
        pts = axis_aligned_strip(20, 20, length=200, thickness=20, horizontal=True)
        cl, _, length_px = min_area_centerline(pts)
        est = estimate_wall_thickness_candidate(
            pts, cl, length_px, W=800, H=800, mpp=0.01, scale_confidence="low"
        )
        self.assertEqual(est["thickness_method"], "perpendicular_median")
        self.assertAlmostEqual(est["thickness_m"], 0.20, delta=0.03)
        self.assertGreaterEqual(est["valid_sample_count"], 3)

    def test_noisy_strip(self):
        # Base 12px strip with jagged edges
        base = axis_aligned_strip(30, 30, length=180, thickness=12, horizontal=True)
        rng = np.random.default_rng(0)
        noisy = []
        for p in base[:-1]:
            noisy.append([p[0] + float(rng.normal(0, 0.4)), p[1] + float(rng.normal(0, 0.6))])
        noisy.append(noisy[0][:])
        cl, _, length_px = min_area_centerline(noisy)
        est = estimate_wall_thickness_candidate(
            noisy, cl, length_px, W=600, H=600, mpp=0.01, scale_confidence="low"
        )
        self.assertIsNotNone(est["thickness_m"])
        self.assertNotEqual(est["thickness_method"], "fallback_global")
        self.assertAlmostEqual(est["thickness_m"], 0.12, delta=0.05)

    def test_corner_endpoint_exclusion(self):
        pts = axis_aligned_strip(0, 0, length=200, thickness=10, horizontal=True)
        cl, _, length_px = min_area_centerline(pts)
        sample = sample_perpendicular_widths_px(pts, cl, length_px)
        # All sample ts must be in (0.1, 0.9)
        for t in sample["sample_ts"]:
            self.assertGreaterEqual(t, 0.1 - 1e-9)
            self.assertLessEqual(t, 0.9 + 1e-9)
        self.assertNotIn(0.0, sample["sample_ts"])
        self.assertNotIn(1.0, sample["sample_ts"])

    def test_insufficient_samples_falls_to_obb(self):
        # Degenerate tiny polygon: sampling should fail, OBB still usable
        pts = [
            [50, 50],
            [52, 50.2],
            [51.5, 53],
            [50, 50],
        ]
        cl, _, length_px = min_area_centerline(pts)
        est = estimate_wall_thickness_candidate(
            pts, cl, max(length_px, 2.0), W=200, H=200, mpp=0.01, scale_confidence="low"
        )
        # Either OBB or later plan/global; must not invent perp median with <3 samples
        if est["thickness_method"] == "perpendicular_median":
            self.assertGreaterEqual(est["valid_sample_count"], 3)
        else:
            self.assertIn(est["thickness_method"], ("obb", None))

    def test_low_confidence_allows_sub_0_05m(self):
        pts = axis_aligned_strip(10, 10, length=100, thickness=4, horizontal=True)
        cl, _, length_px = min_area_centerline(pts)
        est = estimate_wall_thickness_candidate(
            pts, cl, length_px, W=300, H=300, mpp=0.01, scale_confidence="low"
        )
        self.assertTrue(est["reliable"])
        self.assertLess(est["thickness_m"], 0.05)
        self.assertNotAlmostEqual(est["thickness_m"], THICK_FALLBACK)

    def test_high_confidence_soft_bounds(self):
        # Very thick in metres under high confidence should soft-clip toward 0.50
        pts = axis_aligned_strip(10, 10, length=200, thickness=80, horizontal=True)
        cl, _, length_px = min_area_centerline(pts)
        # large image so px sanity allows 80px
        est = estimate_wall_thickness_candidate(
            pts, cl, length_px, W=2000, H=2000, mpp=0.01, scale_confidence="high"
        )
        self.assertTrue(est["reliable"])
        self.assertLessEqual(est["thickness_m"], 0.50 + 1e-9)
        self.assertTrue(est["clipped"] or est["thickness_m"] <= 0.50)

        lo, clipped = soft_clip_thickness_m(0.03, "high")
        self.assertEqual(lo, 0.08)
        self.assertTrue(clipped)

    def test_plan_level_median(self):
        good = []
        for t_px in (10, 12, 11, 13):
            pts = axis_aligned_strip(0, 0, 100, t_px)
            cl, _, lp = min_area_centerline(pts)
            c = estimate_wall_thickness_candidate(pts, cl, lp, 400, 400, 0.01, "low")
            c["segment_id"] = f"g{len(good)}"
            good.append(c)
        # One unusable candidate (empty poly-like)
        bad = {
            "segment_id": "bad",
            "thickness_px": None,
            "thickness_m": None,
            "thickness_method": None,
            "raw_obb_px": 0.0,
            "sampled_widths_px": [],
            "valid_sample_count": 0,
            "rejected_sample_count": 5,
            "median_sample_width_px": None,
            "mpp": 0.01,
            "scale_confidence": "low",
            "fallback_reason": "PERP_AND_OBB_UNUSABLE",
            "clipped": False,
            "raw_m": None,
            "reliable": False,
        }
        finalized, plan_med = apply_plan_and_global_thickness_fallback(good + [bad])
        self.assertIsNotNone(plan_med)
        bad_f = next(x for x in finalized if x["segment_id"] == "bad")
        self.assertEqual(bad_f["thickness_method"], "plan_median")
        self.assertTrue(bad_f["plan_median_used"])
        self.assertAlmostEqual(bad_f["thickness_m"], plan_med, places=6)

    def test_absolute_last_resort_global_fallback(self):
        # No reliable walls → global 0.23
        bad = {
            "segment_id": "x",
            "thickness_px": None,
            "thickness_m": None,
            "thickness_method": None,
            "raw_obb_px": 0.0,
            "sampled_widths_px": [],
            "valid_sample_count": 0,
            "rejected_sample_count": 0,
            "median_sample_width_px": None,
            "mpp": 0.01,
            "scale_confidence": "low",
            "fallback_reason": "PERP_AND_OBB_UNUSABLE",
            "clipped": False,
            "raw_m": None,
            "reliable": False,
        }
        finalized, plan_med = apply_plan_and_global_thickness_fallback([bad, dict(bad, segment_id="y")])
        self.assertIsNone(plan_med)
        for f in finalized:
            self.assertEqual(f["thickness_method"], "fallback_global")
            self.assertTrue(f["global_fallback_used"])
            self.assertAlmostEqual(f["thickness_m"], THICK_FALLBACK)

    def test_px_sanity_low_confidence(self):
        self.assertTrue(thickness_px_sane(3.0, 500, 382, "low"))
        self.assertFalse(thickness_px_sane(0.5, 500, 382, "low"))
        # 0.08 * 382 ≈ 30.5
        self.assertFalse(thickness_px_sane(40.0, 500, 382, "low"))

    def test_robust_median_outlier_rejection(self):
        med, kept, n_out = robust_median_width_px([10, 10.5, 9.5, 11, 50])
        self.assertIsNotNone(med)
        self.assertGreater(n_out, 0)
        self.assertTrue(all(w < 20 for w in kept))


if __name__ == "__main__":
    unittest.main()
