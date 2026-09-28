# logic/wall_cv_assist.py

import cv2
import numpy as np


def _contour_to_mask(cnt, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def _line_to_contour(x1, y1, x2, y2, thickness):
    """
    Convert a detected horizontal/vertical line into
    a small rectangular wall polygon.
    """
    dx = x2 - x1
    dy = y2 - y1

    length = max(1.0, (dx * dx + dy * dy) ** 0.5)

    nx = -dy / length
    ny = dx / length

    half = max(2.0, thickness / 2.0)

    pts = np.array(
        [
            [x1 + nx * half, y1 + ny * half],
            [x2 + nx * half, y2 + ny * half],
            [x2 - nx * half, y2 - ny * half],
            [x1 - nx * half, y1 - ny * half],
        ],
        dtype=np.float32,
    )

    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def _iou(cnt_a, cnt_b, h, w):
    mask_a = _contour_to_mask(cnt_a, h, w)
    mask_b = _contour_to_mask(cnt_b, h, w)

    area_a = int(np.count_nonzero(mask_a))
    area_b = int(np.count_nonzero(mask_b))

    if area_a == 0 or area_b == 0:
        return 0.0

    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def _overlap_ratio(candidate, existing, h, w):
    """
    How much of the CV candidate is already covered
    by an existing YOLO wall.
    """
    c_mask = _contour_to_mask(candidate, h, w)
    e_mask = _contour_to_mask(existing, h, w)

    c_area = int(np.count_nonzero(c_mask))

    if c_area == 0:
        return 0.0

    intersection = int(
        np.count_nonzero((c_mask > 0) & (e_mask > 0))
    )

    return intersection / float(c_area)


def build_wall_candidate_contours(
    img,
    min_line_length_ratio=0.035,
    wall_thickness=8,
):
    """
    Detect long horizontal/vertical wall-like lines using OpenCV.

    This is an ASSIST layer only.
    It is intended to recover walls missed by YOLO.
    """

    if img is None:
        return []

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Floor-plan drawings normally have dark lines on light background.
    # THRESH_BINARY_INV makes those lines white.
    _, binary = cv2.threshold(
        gray,
        210,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # Remove tiny noise while preserving long architectural lines.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(w * 0.025)), 3),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, max(15, int(h * 0.025))),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    line_mask = cv2.bitwise_or(horizontal, vertical)

    # Close small breaks in walls.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    # Hough detects long straight wall portions.
    min_len = max(
        30,
        int(min(h, w) * min_line_length_ratio),
    )

    lines = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_len,
        maxLineGap=18,
    )

    if lines is None:
        return []

    candidates = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)

        dx = x2 - x1
        dy = y2 - y1

        length = (dx * dx + dy * dy) ** 0.5

        if length < min_len:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))

        # Keep mostly horizontal / vertical architectural lines.
        if not (
            angle <= 8
            or angle >= 172
            or 82 <= angle <= 98
        ):
            continue

        cnt = _line_to_contour(
            x1,
            y1,
            x2,
            y2,
            wall_thickness,
        )

        candidates.append(cnt)

    return candidates


def merge_yolo_with_cv_walls(
    yolo_walls,
    cv_walls,
    img_h,
    img_w,
    overlap_threshold=0.55,
):
    """
    Keep YOLO walls as the primary detections.

    Add CV walls only when they are not already covered
    by an existing YOLO wall.

    This prevents CV from creating duplicate walls.
    """

    result = []

    # 1. Keep existing YOLO walls.
    for cnt in yolo_walls or []:
        if cnt is None or len(cnt) < 3:
            continue

        result.append(cnt)

    # 2. Add only missing CV wall pieces.
    for cv_cnt in cv_walls or []:

        if cv_cnt is None or len(cv_cnt) < 3:
            continue

        already_detected = False

        for yolo_cnt in result:

            overlap = _overlap_ratio(
                cv_cnt,
                yolo_cnt,
                img_h,
                img_w,
            )

            if overlap >= overlap_threshold:
                already_detected = True
                break

        if not already_detected:
            result.append(cv_cnt)

    return result


def remove_overlapping_walls(
    contours,
    img_h,
    img_w,
    iou_threshold=0.70,
):
    """
    Remove near-duplicate wall polygons.

    Larger wall polygon is preferred.
    """

    if not contours:
        return []

    items = []

    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue

        area = abs(cv2.contourArea(cnt))

        if area <= 2:
            continue

        items.append((area, cnt))

    # Largest first.
    items.sort(key=lambda x: x[0], reverse=True)

    kept = []

    for area, cnt in items:

        duplicate = False

        for existing in kept:

            if _iou(
                cnt,
                existing,
                img_h,
                img_w,
            ) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(cnt)

    return kept