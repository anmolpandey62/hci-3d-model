#!/usr/bin/env python3
"""
Reusable IFC generation entry for HCI_2.1.

Phase 1 extraction: same steps as web/server.py generate_ifc_geometry,
without FastAPI response types — so Training UI and a future Production
Orchestrator can share one entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from logic.dataset_io import (
    ifc_debug_dir,
    ifc_file_path,
    ifc_output_dir,
    label_train_path,
    resolve_image_for_basename,
)
from logic.ifc_pipeline import generate_full_ifc
from logic.scale_calibration import DEFAULT_MPP, resolve_mpp


class IfcLabelMissingError(FileNotFoundError):
    """No labels/train/{basename}.txt on disk."""


class IfcImageMissingError(FileNotFoundError):
    """No train/raw image found for basename."""


def resolve_meters_per_pixel(
    dataset_dir: Path | str,
    basename: str,
    meters_per_pixel: float | None = None,
    fallback: float = DEFAULT_MPP,
) -> tuple[float, str]:
    """
    Same mpp resolution as former generate_ifc_geometry:
      explicit meters_per_pixel → ("request")
      else resolve_mpp(dataset, basename, fallback)
    """
    if meters_per_pixel is not None:
        return float(meters_per_pixel), "request"
    return resolve_mpp(dataset_dir, basename, fallback)


def generate_ifc_for_basename(
    dataset_dir: Path | str,
    basename: str,
    meters_per_pixel: float | None = None,
    *,
    download_url: str | None = None,
) -> dict[str, Any]:
    """
    Generate IFC4 for an already-labelled floor plan.

    Raises:
      IfcLabelMissingError — no label file
      IfcImageMissingError — no image
      Other exceptions from generate_full_ifc (caller may format as 500)

    Returns:
      generate_full_ifc result dict plus mpp_source and download_url
      (download_url defaults to /api/ifc/file/{basename}).
    """
    dataset_dir = Path(dataset_dir)
    lbl = label_train_path(dataset_dir, basename)
    if not lbl.exists():
        raise IfcLabelMissingError(
            f"No saved labels for {basename}. Auto-label / correct / save first."
        )

    img_path = resolve_image_for_basename(dataset_dir, basename)
    if img_path is None:
        raise IfcImageMissingError(f"No image found for {basename}")

    mpp, mpp_source = resolve_meters_per_pixel(
        dataset_dir, basename, meters_per_pixel=meters_per_pixel
    )

    out_dir = ifc_output_dir(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ifc = ifc_file_path(dataset_dir, basename)
    work_dir = ifc_debug_dir(dataset_dir, basename)

    result = generate_full_ifc(
        image_path=img_path,
        label_path=lbl,
        output_ifc=out_ifc,
        meters_per_pixel=mpp,
        work_dir=work_dir,
    )
    result["mpp_source"] = mpp_source
    result["download_url"] = (
        download_url if download_url is not None else f"/api/ifc/file/{basename}"
    )
    return result
