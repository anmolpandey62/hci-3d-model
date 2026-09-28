#!/usr/bin/env python3
"""Scale calibration helpers for HCI_2.1 (meters-per-pixel sidecar)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MPP = 0.01


def scale_path(dataset_dir: Path | str, basename: str) -> Path:
    return Path(dataset_dir) / "metadata" / f"{basename}_scale.json"


def load_scale(dataset_dir: Path | str, basename: str) -> dict | None:
    path = scale_path(dataset_dir, basename)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_meters_per_pixel(p1: list[float], p2: list[float], distance_m: float) -> float:
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    d_px = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
    if d_px < 1.0:
        raise ValueError("pixel distance too small; pick two farther points")
    return float(distance_m) / d_px


def save_scale(
    dataset_dir: Path | str,
    basename: str,
    p1: list[float],
    p2: list[float],
    distance_m: float,
    image_wh: list[int] | None = None,
    source: str = "manual",
) -> dict[str, Any]:
    mpp = compute_meters_per_pixel(p1, p2, distance_m)
    data = {
        "basename": basename,
        "meters_per_pixel": mpp,
        "source": source,
        "p1_px": [float(p1[0]), float(p1[1])],
        "p2_px": [float(p2[0]), float(p2[1])],
        "distance_m": float(distance_m),
        "image_wh": image_wh,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = scale_path(dataset_dir, basename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def resolve_mpp(dataset_dir: Path | str, basename: str, fallback: float = DEFAULT_MPP) -> tuple[float, str]:
    """Return (mpp, source_note)."""
    data = load_scale(dataset_dir, basename)
    if data and float(data.get("meters_per_pixel", 0)) > 0:
        return float(data["meters_per_pixel"]), data.get("source", "saved")
    return float(fallback), "default_fallback"
