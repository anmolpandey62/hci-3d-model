#!/usr/bin/env python3
"""
Production floor-plan orchestration for the external frontend.

Phase 2: sequences existing reusable backend logic only.
Phase 3: ProductionOrchestrator.process facade.
Phase 4: production hardening (unique names, envelope, logging, validation).
Phase 5: quality gate before IFC (logic.production_validation).
Phase 6: response standardization, request_id, timing, health, observability.

Does not duplicate YOLO, IFC, upload, or scale implementations.
Does not alter the Manual Training async /api/autolabel contract.
"""
from __future__ import annotations

import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from logic.dataset_io import (
    ifc_output_dir,
    is_allowed_image_name,
    save_raw_image_bytes,
)
from logic.ifc_service import (
    IfcImageMissingError,
    IfcLabelMissingError,
    generate_ifc_for_basename,
)
from logic.production_validation import QualityGateError, QualityValidator
from logic.yolo_inference import resolve_hci21_model

# Default wall-clock wait for the existing auto-label worker (seconds).
DEFAULT_AUTOLABEL_TIMEOUT_S = 600
PRODUCTION_API_VERSION = "2.1"

# Process start for health uptime (set at import of this module).
_PRODUCTION_MODULE_START = time.time()

AutolabelRunner = Callable[[list | None, str], None]

logger = logging.getLogger("hci21.production")


class ProductionError(Exception):
    """Clean production failure with an HTTP status hint."""

    def __init__(self, message: str, status_code: int = 400, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or _default_error_code(status_code)


def _default_error_code(status_code: int) -> str:
    return {
        400: "invalid_request",
        404: "not_found",
        422: "invalid_floorplan",
        503: "model_unavailable",
        504: "timeout",
        500: "internal_error",
    }.get(status_code, "error")


def _production_debug_enabled() -> bool:
    return os.environ.get("HCI21_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _attach_error_context(
    exc: ProductionError,
    *,
    request_id: str | None,
    basename: str | None,
    processing_time_ms: int | None,
    timing: dict[str, int] | None = None,
    warnings: list[str] | None = None,
) -> ProductionError:
    if request_id is not None:
        exc.request_id = request_id  # type: ignore[attr-defined]
    if basename is not None:
        exc.basename = basename  # type: ignore[attr-defined]
    if processing_time_ms is not None:
        exc.processing_time_ms = processing_time_ms  # type: ignore[attr-defined]
    if timing is not None:
        exc.timing = timing  # type: ignore[attr-defined]
    if warnings is not None:
        exc.warnings = warnings  # type: ignore[attr-defined]
    return exc


def make_unique_production_filename(filename: str) -> str:
    """
    Production-only unique upload name.

    floorplan.png → floorplan_20260805_143211_ab12cd.png

    Preserves extension. Training /api/upload naming is untouched.
    """
    path = Path(filename)
    stem = path.stem.strip() or "floorplan"
    # Keep logs readable; strip path separators if a client sends a path.
    stem = stem.replace("\\", "_").replace("/", "_").replace(" ", "_")
    ext = path.suffix.lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{stem}_{ts}_{suffix}{ext}"


def validate_production_request(
    filename: str | None,
    image_bytes: bytes | None,
    meters_per_pixel: float | str | None = None,
) -> float | None:
    """
    Validate production input before orchestration.

    Returns normalized meters_per_pixel (or None).
    Raises ProductionError (400) on invalid request.
    """
    if filename is None or str(filename).strip() == "":
        raise ProductionError(
            "Upload failed: image file is required",
            status_code=400,
            error_code="invalid_request",
        )
    if not image_bytes:
        raise ProductionError(
            "Upload failed: empty image file",
            status_code=400,
            error_code="invalid_request",
        )
    if not is_allowed_image_name(filename):
        raise ProductionError(
            "Upload failed: unsupported image type "
            "(allowed: .jpg, .jpeg, .png, .bmp, .tiff, .webp, .svg)",
            status_code=400,
            error_code="invalid_request",
        )

    if meters_per_pixel is None or meters_per_pixel == "":
        return None
    try:
        mpp = float(meters_per_pixel)
    except (TypeError, ValueError) as exc:
        raise ProductionError(
            "Invalid meters_per_pixel: must be a positive number",
            status_code=400,
            error_code="invalid_request",
        ) from exc
    if not math.isfinite(mpp) or mpp <= 0:
        raise ProductionError(
            "Invalid meters_per_pixel: must be a positive number",
            status_code=400,
            error_code="invalid_request",
        )
    return mpp


def get_production_health(dataset_dir: Path | str) -> dict[str, Any]:
    """
    Pure production health check — no YOLO load, inference, or IFC.
    Reuses resolve_hci21_model() for model availability only.
    """
    root = Path(dataset_dir)
    model_path, model_source = resolve_hci21_model()
    model_available = bool(model_path)
    out_dir = ifc_output_dir(root)
    return {
        "status": "healthy" if model_available and root.is_dir() else "degraded",
        "version": PRODUCTION_API_VERSION,
        "model_available": model_available,
        "model_source": model_source if model_available else None,
        "model_path": model_path or None,
        "dataset_directory_exists": root.is_dir(),
        "output_directory_exists": out_dir.is_dir(),
        "timestamp": _utc_timestamp(),
        "uptime_seconds": int(time.time() - _PRODUCTION_MODULE_START),
    }


def build_production_error_body(
    message: str,
    *,
    status_code: int,
    error_code: str | None = None,
    basename: str | None = None,
    processing_time_ms: int | None = None,
    request_id: str | None = None,
    timing: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    """Normalized production error envelope (no traceback unless DEBUG)."""
    body: dict[str, Any] = {
        "success": False,
        "request_id": request_id,
        "error": message,
        "error_code": error_code or _default_error_code(status_code),
        "processing_time_ms": processing_time_ms,
        "timestamp": _utc_timestamp(),
        "basename": basename,
        "warnings": list(warnings or []),
    }
    if timing is not None:
        body["timing"] = timing
    if _production_debug_enabled() and exc is not None:
        import traceback as _tb

        body["trace"] = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    return body


def build_production_success_body(
    *,
    request_id: str,
    basename: str,
    original_filename: str | None,
    stored_filename: str | None,
    processing_time_ms: int,
    download_url: str,
    validation: dict[str, Any],
    warnings: list[str],
    timing: dict[str, int],
    ifc_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Phase 6 standardized success envelope.

    Preserves existing IFC result fields at the top level while adding the
    consistent schema (request_id, result, timing, …).
    """
    walls = ifc_result.get("ifc_walls")
    doors = ifc_result.get("ifc_doors")
    windows = ifc_result.get("ifc_windows")
    body: dict[str, Any] = {
        **ifc_result,
        "success": True,
        "request_id": request_id,
        "basename": basename,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "processing_time_ms": processing_time_ms,
        "download_url": download_url,
        "mpp_source": ifc_result.get("mpp_source"),
        "validation": validation,
        "warnings": warnings,
        "timing": timing,
        "result": {
            "walls": walls,
            "doors": doors,
            "windows": windows,
        },
        # Keep prior flat keys for backward compatibility with Phase 4/5 clients.
        "ifc_walls": walls,
        "ifc_doors": doors,
        "ifc_windows": windows,
        "timestamp": _utc_timestamp(),
    }
    return body


def process_floorplan(
    dataset_dir: Path | str,
    filename: str | None,
    image_bytes: bytes,
    meters_per_pixel: float | None = None,
    *,
    run_autolabel: AutolabelRunner,
    metadata_choice: str = "local",
    autolabel_timeout_s: float = DEFAULT_AUTOLABEL_TIMEOUT_S,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Orchestrate: save image → auto-label (sync wait) → verify labels → IFC.

    ``run_autolabel`` must be the existing backend worker (e.g. ``_autolabel_worker``),
    invoked directly — not via HTTP or SSE polling.

    Returns a Phase 6 production success payload.
    """
    dataset_dir = Path(dataset_dir)
    warnings: list[str] = []
    t0 = time.perf_counter()
    request_id = request_id or str(uuid.uuid4())
    basename: str | None = None
    unique_name: str | None = None
    timing: dict[str, int] = {
        "upload_ms": 0,
        "autolabel_ms": 0,
        "validation_ms": 0,
        "ifc_ms": 0,
        "total_ms": 0,
    }

    try:
        logger.info(
            "request_start request_id=%s basename=%s duration_ms=0 status=started "
            "original=%s",
            request_id,
            basename,
            filename,
        )

        if not filename or not image_bytes:
            raise ProductionError(
                "Upload failed: image file is required",
                status_code=400,
                error_code="invalid_request",
            )

        t_upload = time.perf_counter()
        unique_name = make_unique_production_filename(filename)
        saved = save_raw_image_bytes(dataset_dir, unique_name, image_bytes)
        if saved is None:
            raise ProductionError(
                "Upload failed: unsupported image type "
                "(allowed: .jpg, .jpeg, .png, .bmp, .tiff, .webp, .svg)",
                status_code=400,
                error_code="invalid_request",
            )
        timing["upload_ms"] = _ms_since(t_upload)

        basename = saved.stem
        logger.info(
            "request_start request_id=%s basename=%s duration_ms=%s status=uploaded "
            "original=%s unique=%s",
            request_id,
            basename,
            timing["upload_ms"],
            filename,
            unique_name,
        )

        # Phase 5: model must be resolvable before inference begins.
        try:
            model_path, model_src = QualityValidator.ensure_model_available()
            logger.info(
                "request_validation request_id=%s basename=%s duration_ms=%s "
                "status=model_ok path=%s source=%s",
                request_id,
                basename,
                _ms_since(t0),
                model_path,
                model_src,
            )
        except QualityGateError as exc:
            raise ProductionError(
                exc.message,
                status_code=exc.status_code,
                error_code=exc.error_code,
            ) from exc

        # Run existing auto-label worker synchronously and wait for completion.
        logger.info(
            "request_autolabel_start request_id=%s basename=%s duration_ms=%s status=started",
            request_id,
            basename,
            _ms_since(t0),
        )
        t_al = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(run_autolabel, [saved.name], metadata_choice)
                future.result(timeout=autolabel_timeout_s)
        except FuturesTimeout as exc:
            timing["autolabel_ms"] = _ms_since(t_al)
            raise ProductionError(
                f"Auto-label timed out after {autolabel_timeout_s:.0f}s for {saved.name}",
                status_code=504,
                error_code="timeout",
            ) from exc
        except ProductionError:
            timing["autolabel_ms"] = _ms_since(t_al)
            raise
        except Exception as exc:
            timing["autolabel_ms"] = _ms_since(t_al)
            raise ProductionError(
                f"Auto-label failed for {saved.name}: {exc}",
                status_code=500,
                error_code="internal_error",
            ) from exc
        timing["autolabel_ms"] = _ms_since(t_al)
        logger.info(
            "request_autolabel_complete request_id=%s basename=%s duration_ms=%s status=ok",
            request_id,
            basename,
            timing["autolabel_ms"],
        )

        # Phase 5: quality gate before IFC generation.
        t_val = time.perf_counter()
        try:
            validation = QualityValidator.validate_after_autolabel(dataset_dir, basename)
        except QualityGateError as exc:
            timing["validation_ms"] = _ms_since(t_val)
            raise ProductionError(
                exc.message,
                status_code=exc.status_code,
                error_code=exc.error_code,
            ) from exc
        timing["validation_ms"] = _ms_since(t_val)
        warnings.extend(validation.get("warnings") or [])
        logger.info(
            "request_validation_complete request_id=%s basename=%s duration_ms=%s "
            "status=ok walls=%s doors=%s windows=%s",
            request_id,
            basename,
            timing["validation_ms"],
            validation.get("wall_count"),
            validation.get("door_count"),
            validation.get("window_count"),
        )

        logger.info(
            "request_ifc_start request_id=%s basename=%s duration_ms=%s status=started",
            request_id,
            basename,
            _ms_since(t0),
        )
        t_ifc = time.perf_counter()
        try:
            result = generate_ifc_for_basename(
                dataset_dir,
                basename,
                meters_per_pixel=meters_per_pixel,
            )
        except IfcLabelMissingError as exc:
            timing["ifc_ms"] = _ms_since(t_ifc)
            raise ProductionError(
                str(exc), status_code=404, error_code="labels_missing"
            ) from exc
        except IfcImageMissingError as exc:
            timing["ifc_ms"] = _ms_since(t_ifc)
            raise ProductionError(
                str(exc), status_code=404, error_code="image_missing"
            ) from exc
        except Exception as exc:
            timing["ifc_ms"] = _ms_since(t_ifc)
            raise ProductionError(
                f"IFC generation failed for {basename}: {exc}",
                status_code=500,
                error_code="ifc_generation_failed",
            ) from exc
        timing["ifc_ms"] = _ms_since(t_ifc)
        logger.info(
            "request_ifc_complete request_id=%s basename=%s duration_ms=%s status=ok "
            "walls=%s doors=%s windows=%s",
            request_id,
            basename,
            timing["ifc_ms"],
            result.get("ifc_walls"),
            result.get("ifc_doors"),
            result.get("ifc_windows"),
        )

        timing["total_ms"] = _ms_since(t0)
        download_url = result.get("download_url", f"/api/ifc/file/{basename}")
        payload = build_production_success_body(
            request_id=request_id,
            basename=basename,
            original_filename=filename,
            stored_filename=unique_name,
            processing_time_ms=timing["total_ms"],
            download_url=download_url,
            validation=validation,
            warnings=warnings,
            timing=timing,
            ifc_result=result,
        )
        logger.info(
            "request_finished request_id=%s basename=%s duration_ms=%s status=ok",
            request_id,
            basename,
            timing["total_ms"],
        )
        return payload

    except ProductionError as exc:
        timing["total_ms"] = _ms_since(t0)
        logger.error(
            "request_finished request_id=%s basename=%s duration_ms=%s status=%s error=%s",
            request_id,
            basename,
            timing["total_ms"],
            exc.status_code,
            exc.message,
            exc_info=True,
        )
        _attach_error_context(
            exc,
            request_id=request_id,
            basename=basename,
            processing_time_ms=timing["total_ms"],
            timing=dict(timing),
            warnings=list(warnings),
        )
        raise
    except Exception as exc:
        timing["total_ms"] = _ms_since(t0)
        logger.exception(
            "request_finished request_id=%s basename=%s duration_ms=%s status=500 error=%s",
            request_id,
            basename,
            timing["total_ms"],
            exc,
        )
        pe = ProductionError(
            f"Unexpected internal failure: {exc}",
            status_code=500,
            error_code="internal_error",
        )
        _attach_error_context(
            pe,
            request_id=request_id,
            basename=basename,
            processing_time_ms=timing["total_ms"],
            timing=dict(timing),
            warnings=list(warnings),
        )
        raise pe from exc


class ProductionOrchestrator:
    """
    Thin Phase 3 API facade over the existing Phase 2 ``process_floorplan`` pipeline.
    Phase 4–6: validation, request_id, timing, standardized envelopes.

    Does not reimplement upload, auto-label, scale, or IFC — delegates entirely.
    """

    def __init__(
        self,
        dataset_dir: Path | str,
        *,
        run_autolabel: AutolabelRunner,
        metadata_choice: str = "local",
        autolabel_timeout_s: float = DEFAULT_AUTOLABEL_TIMEOUT_S,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.run_autolabel = run_autolabel
        self.metadata_choice = metadata_choice
        self.autolabel_timeout_s = autolabel_timeout_s

    def process(
        self,
        filename: str | None,
        image_bytes: bytes,
        meters_per_pixel: float | str | None = None,
    ) -> dict[str, Any]:
        """Validate, then run the production pipeline with request_id observability."""
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        logger.info(
            "request_start request_id=%s basename=%s duration_ms=0 status=accepted "
            "original=%s",
            request_id,
            None,
            filename,
        )
        try:
            mpp = validate_production_request(filename, image_bytes, meters_per_pixel)
            logger.info(
                "request_validation request_id=%s basename=%s duration_ms=%s status=ok",
                request_id,
                None,
                _ms_since(t0),
            )
        except ProductionError as exc:
            elapsed = _ms_since(t0)
            logger.error(
                "request_finished request_id=%s basename=%s duration_ms=%s status=%s error=%s",
                request_id,
                None,
                elapsed,
                exc.status_code,
                exc.message,
            )
            _attach_error_context(
                exc,
                request_id=request_id,
                basename=None,
                processing_time_ms=elapsed,
                timing={
                    "upload_ms": 0,
                    "autolabel_ms": 0,
                    "validation_ms": elapsed,
                    "ifc_ms": 0,
                    "total_ms": elapsed,
                },
                warnings=[],
            )
            raise

        return process_floorplan(
            self.dataset_dir,
            filename,
            image_bytes,
            meters_per_pixel=mpp,
            run_autolabel=self.run_autolabel,
            metadata_choice=self.metadata_choice,
            autolabel_timeout_s=self.autolabel_timeout_s,
            request_id=request_id,
        )
