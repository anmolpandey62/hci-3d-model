# logic/window_inference.py
#
# Window-only inference pipeline.
#
# IMPORTANT:
# - Does NOT modify Wall detection.
# - Does NOT modify Door detection.
# - Uses the same HCI model resolver as yolo_inference.py.
# - Runs Window-only inference on the full image + tiled crops.
# - Converts crop coordinates back to original-image coordinates.
#
# Goal:
# Improve Window recall, especially for small Windows.

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from config.classes import CLASS_IDS
from logic.yolo_inference import (
    _filter_contour,
    _bbox_to_contour,
    contour_to_yolo_seg,
    map_model_class_to_hci,
)

from ultralytics import YOLO

# Separate model cache for Window inference.
# This prevents Window-only prediction settings
# from affecting the normal Wall/Door/Window pipeline.
_window_model_cache = {
    "path": None,
    "model": None,
}

def _get_window_model(model_path: str):
    if not model_path:
        raise ValueError("Window model path is required")

    if (
        _window_model_cache["path"] == model_path
        and _window_model_cache["model"] is not None
    ):
        return _window_model_cache["model"]

    model = YOLO(model_path)

    _window_model_cache["path"] = model_path
    _window_model_cache["model"] = model

    return model
logger = logging.getLogger(__name__)


# ============================================================
# WINDOW SETTINGS
# ============================================================

WINDOW_CONF = 0.05
WINDOW_TILE_CONF = 0.03
WINDOW_FULL_IMGSZ = 1280
WINDOW_TILE_IMGSZ = 1280
WINDOW_GRID_ROWS = 2
WINDOW_GRID_COLS = 2
WINDOW_TILE_OVERLAP = 0.20
WINDOW_MIN_AREA = 30.0
WINDOW_DEDUPE = True


# ============================================================
# MODEL CLASS
# ============================================================

def _find_window_model_id(model):
    """
    Find the actual YOLO class id corresponding to HCI Window.
    """

    for model_id, _model_name in model.names.items():

        hci_name = map_model_class_to_hci(
            model_id,
            model.names,
        )

        if hci_name == "Window":
            return int(model_id)

    return None


# ============================================================
# BASIC WINDOW CONTOUR FILTER
# ============================================================

def _filter_window_contour(
    cnt: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray | None:
    """
    Window-specific contour validation.

    This is intentionally less restrictive than Wall filtering.
    """

    if cnt is None or len(cnt) < 3:
        return None

    cnt = np.asarray(cnt, dtype=np.int32)

    area = float(cv2.contourArea(cnt))

    if area < WINDOW_MIN_AREA:
        return None

    peri = cv2.arcLength(cnt, True)

    if peri <= 0:
        return None

    # Small approximation so Window polygon shape is retained.
    eps = max(0.8, 0.002 * peri)

    approx = cv2.approxPolyDP(
        cnt,
        eps,
        True,
    )

    if approx is None or len(approx) < 3:
        return None

    return approx


# ============================================================
# MASK -> CONTOURS
# ============================================================

def _mask_to_contours(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
) -> list[np.ndarray]:
    """
    Convert one YOLO segmentation mask into Window contours.
    """

    if mask is None:
        return []

    if mask.shape != (img_h, img_w):
        mask = cv2.resize(
            mask,
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bin = (
        mask > 0.5
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    output = []

    for cnt in contours:

        clean = _filter_window_contour(
            cnt,
            img_h,
            img_w,
        )

        if clean is not None:
            output.append(clean)

    return output


# ============================================================
# BOX -> CONTOUR
# ============================================================

def _box_to_window_contour(
    box,
    img_h: int,
    img_w: int,
):
    """
    Convert YOLO box into polygon fallback.
    """

    x1, y1, x2, y2 = box[:4]

    cnt = _bbox_to_contour(
        x1,
        y1,
        x2,
        y2,
    )

    return _filter_window_contour(
        cnt,
        img_h,
        img_w,
    )


# ============================================================
# FULL IMAGE WINDOW INFERENCE
# ============================================================

def _run_window_on_image(
    model,
    img: np.ndarray,
    window_model_id: int,
    conf: float,
    imgsz: int,
) -> list[np.ndarray]:
    """
    Run Window-only YOLO inference on one image.

    Returns contours in that image's coordinate system.
    """

    img_h, img_w = img.shape[:2]

    results = model.predict(
        img,
        imgsz=imgsz,
        conf=conf,
        classes=[window_model_id],
        verbose=False,
    )

    if not results:
        return []

    result = results[0]

    boxes = result.boxes
    masks = result.masks

    detections = []

    # --------------------------------------------------------
    # Segmentation / mask path
    # --------------------------------------------------------

    if (
        masks is not None
        and boxes is not None
        and len(boxes) > 0
    ):

        mask_data = (
            masks.data.cpu().numpy()
        )

        box_data = (
            boxes.data.cpu().numpy()
        )

        for i, box in enumerate(box_data):

            box_conf = float(box[4])

            if box_conf < conf:
                continue

            if i >= len(mask_data):
                continue

            contours = _mask_to_contours(
                mask_data[i],
                img_h,
                img_w,
            )

            detections.extend(contours)

        return detections

    # --------------------------------------------------------
    # Bounding-box fallback
    # --------------------------------------------------------

    if (
        boxes is not None
        and len(boxes) > 0
    ):

        box_data = (
            boxes.data.cpu().numpy()
        )

        for box in box_data:

            box_conf = float(box[4])

            if box_conf < conf:
                continue

            clean = _box_to_window_contour(
                box,
                img_h,
                img_w,
            )

            if clean is not None:
                detections.append(clean)

    return detections


# ============================================================
# TILE GENERATION
# ============================================================

def _build_tiles(
    img_h: int,
    img_w: int,
    rows: int,
    cols: int,
    overlap: float,
):
    """
    Create overlapping image tiles.

    Returns:
        (x1, y1, x2, y2)
    """

    rows = max(1, int(rows))
    cols = max(1, int(cols))

    overlap = float(
        max(0.0, min(overlap, 0.8))
    )

    # Tile dimensions.
    tile_h = int(
        img_h / (
            rows - (rows - 1) * overlap
        )
    )

    tile_w = int(
        img_w / (
            cols - (cols - 1) * overlap
        )
    )

    tile_h = min(tile_h, img_h)
    tile_w = min(tile_w, img_w)

    if rows == 1:
        y_positions = [0]
    else:
        y_step = max(
            1,
            int(tile_h * (1.0 - overlap)),
        )

        y_positions = list(
            range(
                0,
                max(1, img_h - tile_h + 1),
                y_step,
            )
        )

        last_y = img_h - tile_h

        if y_positions[-1] != last_y:
            y_positions.append(last_y)

    if cols == 1:
        x_positions = [0]
    else:
        x_step = max(
            1,
            int(tile_w * (1.0 - overlap)),
        )

        x_positions = list(
            range(
                0,
                max(1, img_w - tile_w + 1),
                x_step,
            )
        )

        last_x = img_w - tile_w

        if x_positions[-1] != last_x:
            x_positions.append(last_x)

    tiles = []

    for y1 in y_positions:

        for x1 in x_positions:

            x2 = min(
                img_w,
                x1 + tile_w,
            )

            y2 = min(
                img_h,
                y1 + tile_h,
            )

            tiles.append(
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )
            )

    return tiles


# ============================================================
# SHIFT TILE CONTOUR -> ORIGINAL IMAGE
# ============================================================

def _shift_contour(
    cnt: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """
    Convert tile-local coordinates to original-image coordinates.
    """

    shifted = cnt.copy()

    shifted[:, 0, 0] += int(offset_x)
    shifted[:, 0, 1] += int(offset_y)

    return shifted


# ============================================================
# OPTIONAL DUPLICATE FUNCTION
# ============================================================

def _bbox_iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    ax1, ay1, aw, ah = cv2.boundingRect(a)
    bx1, by1, bw, bh = cv2.boundingRect(b)

    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = aw * ah
    area_b = bw * bh

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def _dedupe_window_contours(
    contours: list[np.ndarray],
    iou_threshold: float = 0.30,
) -> list[np.ndarray]:

    if not contours:
        return []

    ordered = sorted(
        contours,
        key=lambda cnt: cv2.contourArea(cnt),
        reverse=True,
    )

    kept: list[np.ndarray] = []

    for candidate in ordered:
        is_duplicate = False

        for existing in kept:
            iou = _bbox_iou(candidate, existing)

            if iou >= iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept
#===========================================================
# MAIN WINDOW PIPELINE
# ============================================================

def run_window_inference(
    img: np.ndarray,
    model_path: str | None = None,
) -> tuple[list[np.ndarray], list[str], str]:
    """
    Window-only inference.

    Pipeline:

        Full image Window detection
                    +
        Overlapping tile Window detection
                    ↓
        Convert all detections to original coordinates
                    ↓
        Return Window contours

    Returns:
        (
            window_contours,
            yolo_label_lines,
            model_path_used,
        )

    IMPORTANT:
        This function does NOT run Wall detection.
        This function does NOT run Door detection.
    """

    if img is None:
        raise ValueError(
            "image unreadable"
        )

    img_h, img_w = img.shape[:2]

    # --------------------------------------------------------
    # Load same HCI model used by existing pipeline.
    # --------------------------------------------------------

    used_path = model_path
    model = _get_window_model(used_path)

    window_model_id = _find_window_model_id(
        model
    )

    if window_model_id is None:

        raise RuntimeError(
            "Window class was not found in YOLO model names. "
            f"Available classes: {model.names}"
        )

    logger.info(
        "Window model class id: %s",
        window_model_id,
    )

    all_windows: list[np.ndarray] = []

    # ========================================================
    # PASS 1 — FULL IMAGE
    # ========================================================

    logger.info(
        "Window full-image inference: conf=%.3f imgsz=%d",
        WINDOW_CONF,
        WINDOW_FULL_IMGSZ,
    )

    full_windows = _run_window_on_image(
        model=model,
        img=img,
        window_model_id=window_model_id,
        conf=WINDOW_CONF,
        imgsz=WINDOW_FULL_IMGSZ,
    )

    all_windows.extend(
        full_windows
    )

    logger.info(
        "Window full-image detections: %d",
        len(full_windows),
    )

    # ========================================================
    # PASS 2 — OVERLAPPING TILES
    # ========================================================

    tiles = _build_tiles(
        img_h=img_h,
        img_w=img_w,
        rows=WINDOW_GRID_ROWS,
        cols=WINDOW_GRID_COLS,
        overlap=WINDOW_TILE_OVERLAP,
    )

    logger.info(
        "Window tiled inference: %d tiles",
        len(tiles),
    )

    for tile_index, (
        x1,
        y1,
        x2,
        y2,
    ) in enumerate(tiles):

        crop = img[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]

        logger.info(
            "Window tile %d/%d: x=%d:%d y=%d:%d size=%dx%d",
            tile_index + 1,
            len(tiles),
            x1,
            x2,
            y1,
            y2,
            crop_w,
            crop_h,
        )

        tile_windows = _run_window_on_image(
            model=model,
            img=crop,
            window_model_id=window_model_id,
            conf=WINDOW_TILE_CONF,
            imgsz=WINDOW_TILE_IMGSZ,
        )

        # Convert tile coordinates back to
        # original image coordinates.
        for cnt in tile_windows:

            shifted = _shift_contour(
                cnt,
                offset_x=x1,
                offset_y=y1,
            )

            # Make sure coordinates stay
            # inside original image.
            shifted[:, 0, 0] = np.clip(
                shifted[:, 0, 0],
                0,
                img_w - 1,
            )

            shifted[:, 0, 1] = np.clip(
                shifted[:, 0, 1],
                0,
                img_h - 1,
            )

            clean = _filter_window_contour(
                shifted,
                img_h,
                img_w,
            )

            if clean is not None:
                all_windows.append(
                    clean
                )

        logger.info(
            "Window tile %d detections: %d",
            tile_index + 1,
            len(tile_windows),
        )
        
    # ========================================================
    # WINDOW DEDUPLICATION
    # ========================================================

    if WINDOW_DEDUPE:
        logger.info(
            "Window dedupe: before=%d",
            len(all_windows),
        )
        window_contours = _dedupe_window_contours(
            all_windows,
            iou_threshold=0.30,
        )

        logger.info(
            "Window dedupe: after=%d",
            len(window_contours),
        )

    else:
        window_contours = all_windows
    # ========================================================
    # CREATE YOLO SEGMENTATION LABEL LINES
    # ========================================================

    window_id = CLASS_IDS["Window"]

    label_lines: list[str] = []

    for cnt in window_contours:

        line = contour_to_yolo_seg(
            cnt,
            img_w,
            img_h,
            window_id,
        )

        if line:
            label_lines.append(
                line
            )

    logger.info(
        "Window pipeline completed: %d detections",
        len(window_contours),
    )

    return (
        window_contours,
        label_lines,
        used_path,
    )


# ============================================================
# SIMPLE OVERLAY HELPER
# ============================================================

def draw_window_overlay(
    img: np.ndarray,
    window_contours: list[np.ndarray],
) -> np.ndarray:
    """
    Draw Window-only detections.

    Window HCI color = class 1:
        (255, 0, 255)
    """

    if img is None:
        raise ValueError(
            "image unreadable"
        )

    vis = img.copy()

    img_h, img_w = img.shape[:2]

    max_dim = max(
        img_h,
        img_w,
    )

    font_scale = max(
        0.4,
        max_dim / 2500.0,
    )

    thick = max(
        1,
        int(max_dim * 0.002),
    )

    window_color = (
        255,
        0,
        255,
    )

    for index, cnt in enumerate(
        window_contours,
        start=1,
    ):

        if cnt is None or len(cnt) < 3:
            continue

        mask = np.zeros(
            (
                img_h,
                img_w,
            ),
            dtype=np.uint8,
        )

        cv2.drawContours(
            mask,
            [cnt],
            -1,
            255,
            -1,
        )

        overlay = vis.copy()

        overlay[
            mask > 0
        ] = window_color

        vis = cv2.addWeighted(
            overlay,
            0.30,
            vis,
            0.70,
            0,
        )

        cv2.drawContours(
            vis,
            [cnt],
            -1,
            window_color,
            thick,
        )

        M = cv2.moments(cnt)

        if M["m00"] > 0:

            cx = int(
                M["m10"] /
                M["m00"]
            )

            cy = int(
                M["m01"] /
                M["m00"]
            )

            cv2.putText(
                vis,
                f"Window #{index}",
                (
                    cx,
                    cy,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                window_color,
                thick,
                cv2.LINE_AA,
            )

    return vis