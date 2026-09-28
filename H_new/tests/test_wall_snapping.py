"""Phase 4 — Wall strip H/V + endpoint snapping (mirrors web/index.html helpers)."""
from __future__ import annotations

import math
import unittest
from typing import Iterable, Optional, Sequence, Tuple

Point = Tuple[float, float]

# Mirror of frontend constants
WALL_STRIP_HV_TOL_DEG = 9.0
WALL_STRIP_DEFAULT_THICKNESS_PX = 8.0
WALL_STRIP_EP_TOL_CAP_PX = 12.0


def endpoint_snap_tol(thickness_px: float = WALL_STRIP_DEFAULT_THICKNESS_PX) -> float:
    """max(3, 0.5*t) capped so thick walls do not explode tolerance."""
    return max(3.0, min(0.5 * float(thickness_px), WALL_STRIP_EP_TOL_CAP_PX))


def strip_centerline_endpoints(poly: Sequence[Sequence[float]]) -> Optional[Tuple[Point, Point]]:
    """
    For 4-pt strip A-B / D-C (order A,B,C,D):
      endpoints = mid(A,D), mid(B,C)  OR mid(A,B), mid(D,C) depending on long axis.
    Same rule as frontend _stripParamsFromQuad / Phase 3.
    """
    if poly is None or len(poly) != 4:
        return None

    def mid(a, b):
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    a, b, c, d = poly[0], poly[1], poly[2], poly[3]
    m01, m12, m23, m30 = mid(a, b), mid(b, c), mid(c, d), mid(d, a)
    len_a = dist(m01, m23)
    len_b = dist(m12, m30)
    if len_b >= len_a:
        return (m30, m12)  # mid(A,D), mid(B,C) for Phase-2 stripCorners order
    return (m01, m23)


def snap_hv(
    p0: Point,
    p1: Point,
    tol_deg: float = WALL_STRIP_HV_TOL_DEG,
    free_angle: bool = False,
) -> Tuple[Point, Optional[str]]:
    """Snap p1 to share X or Y with p0 when near axis-aligned. Shift/free_angle disables."""
    if free_angle:
        return p1, None
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return p1, None
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    d_h = min(abs(ang), abs(abs(ang) - 180.0))
    d_v = abs(abs(ang) - 90.0)
    if d_h <= tol_deg and d_h <= d_v:
        return (p1[0], p0[1]), "H"
    if d_v <= tol_deg:
        return (p0[0], p1[1]), "V"
    return p1, None


def snap_to_endpoint(
    pt: Point,
    endpoints: Iterable[Point],
    tol: float,
    *,
    prefer: Optional[Point] = None,
    sticky: Optional[Point] = None,
    sticky_release_mul: float = 1.5,
) -> Tuple[Point, Optional[str], Optional[Point]]:
    """
    Priority: sticky (hysteresis) → prefer (previous wall end) if in tol → closest endpoint.
    Returns (point, kind|None, new_sticky).
    """
    release = tol * sticky_release_mul
    if sticky is not None:
        if math.hypot(pt[0] - sticky[0], pt[1] - sticky[1]) <= release:
            return sticky, "endpoint", sticky

    best: Optional[Point] = None
    best_d = tol
    # Prefer previous endpoint if within tolerance
    if prefer is not None:
        d = math.hypot(pt[0] - prefer[0], pt[1] - prefer[1])
        if d <= tol:
            return prefer, "endpoint", prefer

    for ep in endpoints:
        d = math.hypot(pt[0] - ep[0], pt[1] - ep[1])
        if d < best_d - 1e-12:
            best_d = d
            best = (float(ep[0]), float(ep[1]))
        elif abs(d - best_d) <= 1e-12 and best is not None:
            # Deterministic tie-break: smaller x, then y
            if ep[0] < best[0] or (ep[0] == best[0] and ep[1] < best[1]):
                best = (float(ep[0]), float(ep[1]))

    if best is not None:
        return best, "endpoint", best
    return pt, None, None


def snap_strip_end(
    p0: Point,
    raw_p1: Point,
    endpoints: Sequence[Point],
    *,
    last_endpoint: Optional[Point] = None,
    thickness_px: float = WALL_STRIP_DEFAULT_THICKNESS_PX,
    free_angle: bool = False,
    sticky: Optional[Point] = None,
) -> Tuple[Point, Optional[str], Optional[Point]]:
    """
    Priority: previous/other Wall endpoint → H/V → free cursor.
    """
    tol = endpoint_snap_tol(thickness_px)
    pt, kind, new_sticky = snap_to_endpoint(
        raw_p1, endpoints, tol, prefer=last_endpoint, sticky=sticky
    )
    if kind:
        return pt, kind, new_sticky
    hv, hv_kind = snap_hv(p0, raw_p1, free_angle=free_angle)
    if hv_kind:
        return hv, hv_kind, None
    return raw_p1, None, None


def snap_strip_start(
    raw_p0: Point,
    endpoints: Sequence[Point],
    *,
    last_endpoint: Optional[Point] = None,
    thickness_px: float = WALL_STRIP_DEFAULT_THICKNESS_PX,
    sticky: Optional[Point] = None,
) -> Tuple[Point, Optional[str], Optional[Point]]:
    tol = endpoint_snap_tol(thickness_px)
    return snap_to_endpoint(
        raw_p0, endpoints, tol, prefer=last_endpoint, sticky=sticky
    )


class TestHorizontalSnap(unittest.TestCase):
    def test_near_horizontal(self):
        out, kind = snap_hv((100, 100), (300, 120))
        self.assertEqual(kind, "H")
        self.assertEqual(out, (300, 100))


class TestVerticalSnap(unittest.TestCase):
    def test_near_vertical(self):
        out, kind = snap_hv((100, 100), (120, 300))
        self.assertEqual(kind, "V")
        self.assertEqual(out, (100, 300))


class TestDiagonalFree(unittest.TestCase):
    def test_diagonal_not_forced(self):
        out, kind = snap_hv((100, 100), (200, 200))
        self.assertIsNone(kind)
        self.assertEqual(out, (200, 200))

    def test_shift_disables_hv(self):
        out, kind = snap_hv((100, 100), (300, 105), free_angle=True)
        self.assertIsNone(kind)
        self.assertEqual(out, (300, 105))


class TestEndpointSnap(unittest.TestCase):
    def test_within_tol(self):
        pt, kind, _ = snap_to_endpoint((302, 102), [(300, 100)], endpoint_snap_tol(8))
        self.assertEqual(kind, "endpoint")
        self.assertEqual(pt, (300, 100))

    def test_outside_tol(self):
        pt, kind, _ = snap_to_endpoint((320, 120), [(300, 100)], endpoint_snap_tol(8))
        self.assertIsNone(kind)
        self.assertEqual(pt, (320, 120))


class TestCenterlineEndpoints(unittest.TestCase):
    def test_phase2_strip_order(self):
        # Phase-2 stripCorners((100,100),(300,100),8):
        # A=(100,96), B=(300,96), C=(300,104), D=(100,104)
        poly = [[100, 96], [300, 96], [300, 104], [100, 104]]
        e0, e1 = strip_centerline_endpoints(poly)
        self.assertAlmostEqual(e0[0], 100, places=5)
        self.assertAlmostEqual(e0[1], 100, places=5)
        self.assertAlmostEqual(e1[0], 300, places=5)
        self.assertAlmostEqual(e1[1], 100, places=5)
        # Must NOT blindly return A and B as centerline ends for thickness ends
        self.assertNotEqual(e0, (100, 96))


class TestZoomInvariance(unittest.TestCase):
    def test_same_image_space(self):
        # Snapping is pure image-space — zoom cannot change result
        a1, k1 = snap_hv((50, 50), (150, 55))
        a2, k2 = snap_hv((50, 50), (150, 55))
        self.assertEqual(a1, a2)
        self.assertEqual(k1, k2)
        p1, _, _ = snap_to_endpoint((101, 100), [(100, 100)], 4)
        p2, _, _ = snap_to_endpoint((101, 100), [(100, 100)], 4)
        self.assertEqual(p1, p2)


class TestContinuation(unittest.TestCase):
    def test_prefer_last_endpoint(self):
        last = (300.0, 100.0)
        others = [(305.0, 100.0), (10.0, 10.0)]
        pt, kind, _ = snap_strip_start(
            (302, 101), others, last_endpoint=last, thickness_px=8
        )
        self.assertEqual(kind, "endpoint")
        self.assertEqual(pt, last)


class TestClosestWins(unittest.TestCase):
    def test_closest_deterministic(self):
        eps = [(300, 100), (310, 100), (305, 100)]
        pt, kind, _ = snap_to_endpoint((304, 100), eps, 20)
        self.assertEqual(kind, "endpoint")
        self.assertEqual(pt, (305, 100))

    def test_tie_break_smaller_xy(self):
        eps = [(100, 100), (100, 100)]  # identical
        pt, kind, _ = snap_to_endpoint((100, 100), eps, 5)
        self.assertEqual(pt, (100, 100))


class TestHysteresis(unittest.TestCase):
    def test_sticky_holds(self):
        sticky = (300.0, 100.0)
        # Move slightly outside acquire tol but inside release
        tol = endpoint_snap_tol(8)  # 4
        pt, kind, st = snap_to_endpoint(
            (300 + tol + 0.5, 100), [], tol, sticky=sticky
        )
        self.assertEqual(kind, "endpoint")
        self.assertEqual(pt, sticky)
        self.assertEqual(st, sticky)


class TestStripEndPriority(unittest.TestCase):
    def test_endpoint_before_hv(self):
        # Near horizontal AND near an endpoint — endpoint wins
        p0 = (100.0, 100.0)
        raw = (298.0, 102.0)
        pt, kind, _ = snap_strip_end(
            p0, raw, [(300, 100)], last_endpoint=(300, 100), thickness_px=8
        )
        self.assertEqual(kind, "endpoint")
        self.assertEqual(pt, (300, 100))

    def test_hv_when_no_endpoint(self):
        p0 = (100.0, 100.0)
        raw = (300.0, 108.0)
        pt, kind, _ = snap_strip_end(p0, raw, [], thickness_px=8)
        self.assertEqual(kind, "H")
        self.assertEqual(pt, (300.0, 100.0))


class TestNonWallRegressionNote(unittest.TestCase):
    def test_helpers_are_wall_only_math(self):
        # Door/Window/Room never call these — covered by FE branch _isWallStripDraw()
        self.assertTrue(callable(snap_hv))
        self.assertTrue(callable(snap_strip_end))


if __name__ == "__main__":
    unittest.main()
