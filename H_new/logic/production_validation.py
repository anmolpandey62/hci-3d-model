#!/usr/bin/env python3
"""
Production quality gate — Phase 5.

Inspects auto-label outputs before IFC generation.
Does not duplicate YOLO inference, IFC, upload, or scale logic.
Does not alter the Manual Training workflow.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from config.classes import CLASS_IDS, ID_TO_CLASS
from logic.dataset_io import label_train_path, resolve_image_for_basename
from logic.yolo_inference import resolve_hci21_model

# YOLO-seg line: class_id + at least 3 (x, y) pairs → 7 tokens minimum.
_MIN_YOLO_SEG_TOKENS = 7

WALL_CLASS_ID = CLASS_IDS["Wall"]
DOOR_CLASS_ID = CLASS_IDS["Door"]
WINDOW_CLASS_ID = CLASS_IDS["Window"]


class QualityGateError(Exception):
    """Business quality failure; mapped to ProductionError by the orchestrator."""

    def __init__(self, message: str, status_code: int, error_code: str):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


def ensure_production_model_available() -> tuple[str, str]:
    """
    Verify the production HCI_2.1 model can be resolved before inference.

    Reuses resolve_hci21_model() — does not load or run YOLO.
    Raises QualityGateError (503 / model_unavailable) if missing.
    """
    path, source = resolve_hci21_model()
    if not path:
        raise QualityGateError(
            "No HCI_2.1 YOLO model available for production auto-label. "
            "Set HCI21_MODEL_PATH or install the expected checkpoint.",
            status_code=503,
            error_code="model_unavailable",
        )
    return path, source


def _parse_yolo_seg_line(line: str, line_no: int) -> int:
    """
    Validate one YOLO-seg label row. Returns class_id.

    Rejects wrong column counts, non-numeric values, negatives, NaN/Inf.
    """
    raw = line.strip()
    if not raw:
        raise ValueError("empty line")  # caller skips blanks before this

    parts = raw.split()
    if len(parts) < _MIN_YOLO_SEG_TOKENS:
        raise QualityGateError(
            f"Invalid label format at line {line_no}: expected class_id and "
            f"at least 3 polygon points (≥{_MIN_YOLO_SEG_TOKENS} columns), "
            f"got {len(parts)}.",
            status_code=422,
            error_code="invalid_label_format",
        )

    # class_id + even number of coordinates (x,y pairs)
    n_coords = len(parts) - 1
    if n_coords % 2 != 0:
        raise QualityGateError(
            f"Invalid label format at line {line_no}: unpaired coordinate "
            f"({n_coords} values after class_id).",
            status_code=422,
            error_code="invalid_label_format",
        )

    try:
        class_id = int(float(parts[0]))
    except (TypeError, ValueError) as exc:
        raise QualityGateError(
            f"Invalid label format at line {line_no}: class_id is not an integer.",
            status_code=422,
            error_code="invalid_label_format",
        ) from exc

    if class_id < 0 or class_id not in ID_TO_CLASS:
        raise QualityGateError(
            f"Invalid label format at line {line_no}: class_id {class_id} out of range.",
            status_code=422,
            error_code="invalid_label_format",
        )

    for i, tok in enumerate(parts[1:], start=1):
        try:
            val = float(tok)
        except (TypeError, ValueError) as exc:
            raise QualityGateError(
                f"Invalid label format at line {line_no}: non-numeric value at column {i + 1}.",
                status_code=422,
                error_code="invalid_label_format",
            ) from exc
        if not math.isfinite(val):
            raise QualityGateError(
                f"Invalid label format at line {line_no}: NaN/Inf at column {i + 1}.",
                status_code=422,
                error_code="invalid_label_format",
            )
        if val < 0:
            raise QualityGateError(
                f"Invalid label format at line {line_no}: negative value at column {i + 1}.",
                status_code=422,
                error_code="invalid_label_format",
            )

    return class_id


def validate_labels_and_image(dataset_dir: Path | str, basename: str) -> dict[str, Any]:
    """
    Quality gate after auto-label and before IFC.

    Checks:
      - label file exists
      - label file non-empty
      - valid YOLO-seg rows
      - at least one Wall
      - resolved image exists

    Returns a validation report dict (passed=True) or raises QualityGateError.
    Warnings do not fail the request.
    """
    dataset_dir = Path(dataset_dir)
    warnings: list[str] = []

    lbl = label_train_path(dataset_dir, basename)
    if not lbl.exists():
        raise QualityGateError(
            f"No labels generated for {basename}. "
            "Auto-label finished but labels/train/{basename}.txt is missing.",
            status_code=422,
            error_code="labels_missing",
        )

    text = lbl.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise QualityGateError(
            f"Label file for {basename} is empty.",
            status_code=422,
            error_code="empty_labels",
        )

    wall_count = 0
    door_count = 0
    window_count = 0
    other_count = 0

    for idx, line in enumerate(lines, start=1):
        class_id = _parse_yolo_seg_line(line, idx)
        if class_id == WALL_CLASS_ID:
            wall_count += 1
        elif class_id == DOOR_CLASS_ID:
            door_count += 1
        elif class_id == WINDOW_CLASS_ID:
            window_count += 1
        else:
            other_count += 1

    if wall_count < 1:
        raise QualityGateError(
            "No walls detected. IFC generation requires at least one wall.",
            status_code=422,
            error_code="no_walls_detected",
        )

    img_path = resolve_image_for_basename(dataset_dir, basename)
    if img_path is None:
        raise QualityGateError(
            f"No image found for {basename}",
            status_code=404,
            error_code="image_missing",
        )

    if door_count == 0:
        warnings.append("No doors detected.")
    if window_count == 0:
        warnings.append("No windows detected.")

    return {
        "passed": True,
        "wall_count": wall_count,
        "door_count": door_count,
        "window_count": window_count,
        "other_count": other_count,
        "label_rows": len(lines),
        "image_path": str(img_path),
        "warnings": warnings,
    }


class QualityValidator:
    """Reusable production quality validator (Phase 5)."""

    @staticmethod
    def ensure_model_available() -> tuple[str, str]:
        return ensure_production_model_available()

    @staticmethod
    def validate_after_autolabel(dataset_dir: Path | str, basename: str) -> dict[str, Any]:
        return validate_labels_and_image(dataset_dir, basename)
