# logic/yolo_inference.py — shared YOLO inference helpers (Auto Label + Detect)
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from config.classes import CLASS_IDS, ID_TO_CLASS
from logic.wall_connect import connect_wall_contours, connect_walls_enabled



logger = logging.getLogger(__name__)

HCI_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HCI_DIR.parent

# Phase 1 — CubiCasa / HCI_2.1 model defaults (never write best_gdrive.pt)
CUBICASA_OFFLINE_BEST = (
    PROJECT_ROOT / "cubicasa_hqa_500" / "runs" / "hqa500_offline" / "weights" / "best.pt"
)
CUBICASA_SMOKE_BEST = (
    PROJECT_ROOT
    / "cubicasa_server_package_2121"
    / "runs_cubicasa_hqa"
    / "hqa500_smoke10"
    / "weights"
    / "best.pt"
)
HCI21_ACTIVE_MODEL = PROJECT_ROOT / "best_hci21.pt"
PRODUCTION_MODEL = PROJECT_ROOT / "best_gdrive.pt"

# Same palette as /api/detect in server.py
HCI_COLORS = {
    0: (0, 0, 200),
    1: (255, 0, 255),
    2: (0, 165, 255),
    3: (0, 200, 0),
    4: (100, 100, 100),
    5: (80, 80, 80),
    6: (255, 165, 0),
    7: (139, 69, 19),
    8: (100, 100, 255),
    9: (180, 180, 180),
    10: (200, 150, 150),
    11: (200, 200, 0),
    12: (210, 180, 140),
    13: (255, 255, 165),
    14: (165, 165, 255),
    15: (165, 255, 165),
    16: (255, 200, 100),
}

PRIORITY_HCI_CLASSES = frozenset({"Wall", "Door", "Window"})

_NAME_ALIASES = {
    "wall": "Wall",
    "walls": "Wall",
    "door": "Door",
    "doors": "Door",
    "window": "Window",
    "windows": "Window",
    "section door": "Door",
}

_model_cache = {"path": None, "model": None}

# WALL DETECTION SETTINGS
WALL_MIN_CONF = 0.020
WALL_MIN_ASPECT = 1.5
WALL_MIN_LENGTH_RATIO = 0.008
WALL_MIN_AREA_RATIO = 0.00002


def contour_to_yolo_seg(cnt, img_w, img_h, cid):
    if cnt is None or len(cnt) < 3:
        return ""
    flat = cnt.flatten()
    coords = []
    for i in range(0, len(flat), 2):
        coords.append(f"{flat[i] / img_w:.6f} {flat[i + 1] / img_h:.6f}")
    return f"{cid} " + " ".join(coords)


class ModelNotFoundError(FileNotFoundError):
    """Raised when no YOLO weights are available for inference."""


def resolve_hci21_model() -> tuple[str, str]:
    """
    Resolve the HCI_2.1 Auto Label / Test default model.

    Priority (M9b):
      1. HCI21_MODEL_PATH env (if file exists)
      2. HCI_MODEL_PATH env (legacy alias, if file exists)
      3. HCI21_ACTIVE_MODEL / best_hci21.pt (promoted active checkpoint)
      4. CubiCasa HQA500 offline best.pt (fallback)
      5. CubiCasa smoke10 best.pt (warning)
      6. empty — do NOT fall through to best_gdrive.pt

    Returns (path, source_tag). path may be "".
    Never writes any weights file.
    """
    env21 = os.environ.get("HCI21_MODEL_PATH", "").strip()
    if env21:
        if os.path.isfile(env21):
            return env21, "env:HCI21_MODEL_PATH"
        logger.warning("HCI21_MODEL_PATH set but file missing: %s", env21)

    env_legacy = os.environ.get("HCI_MODEL_PATH", "").strip()
    if env_legacy:
        if os.path.isfile(env_legacy):
            return env_legacy, "env:HCI_MODEL_PATH"
        logger.warning("HCI_MODEL_PATH set but file missing: %s", env_legacy)

    # M9b: promoted active HCI21 checkpoint beats CubiCasa offline fallback
    if HCI21_ACTIVE_MODEL.is_file():
        return str(HCI21_ACTIVE_MODEL), "best_hci21.pt"

    if CUBICASA_OFFLINE_BEST.is_file():
        return str(CUBICASA_OFFLINE_BEST), "cubicasa_hqa500_offline"

    if CUBICASA_SMOKE_BEST.is_file():
        logger.warning(
            "Using CubiCasa smoke10 checkpoint (weaker walls); prefer offline or HCI21_MODEL_PATH"
        )
        return str(CUBICASA_SMOKE_BEST), "cubicasa_hqa500_smoke10"

    return "", "not_found"


def invalidate_model_cache() -> None:
    """M9b: drop cached YOLO weights so the next load reads from disk."""
    _model_cache["path"] = None
    _model_cache["model"] = None


def find_production_model_path() -> str:
    """Read-only path to production best_gdrive.pt (compare only; never promote from HCI_2.1)."""
    if PRODUCTION_MODEL.is_file():
        return str(PRODUCTION_MODEL)
    return ""


def find_model_path() -> str:
    """
    HCI_2.1 default inference weights = resolve_hci21_model().
    Does not use best_gdrive.pt as Auto Label default.
    """
    path, _src = resolve_hci21_model()
    return path


def map_model_class_to_hci(model_cls_id: int, model_names: dict) -> str | None:
    """Map a YOLO model class id to an HCI class name (Wall/Door/Window only)."""
    raw = model_names.get(int(model_cls_id), str(model_cls_id))
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    raw_str = str(raw).strip()
    if raw_str in CLASS_IDS and raw_str in PRIORITY_HCI_CLASSES:
        return raw_str
    key = raw_str.lower().replace("_", " ")
    hci = _NAME_ALIASES.get(key)
    if hci and hci in PRIORITY_HCI_CLASSES:
        return hci
    for name in PRIORITY_HCI_CLASSES:
        if raw_str.lower() == name.lower():
            return name
    return None


def _get_model(model_path: str | None = None):
    path = model_path or find_model_path()
    if not path or not os.path.exists(path):
        raise ModelNotFoundError(
            "No HCI_2.1 YOLO model found. Set HCI21_MODEL_PATH or place weights at "
            f"{CUBICASA_OFFLINE_BEST}"
        )
    if _model_cache["path"] == path and _model_cache["model"] is not None:
        return _model_cache["model"], path
    from ultralytics import YOLO

    model = YOLO(path)
    _model_cache["path"] = path
    _model_cache["model"] = model
    return model, path


def _filter_contour(cnt: np.ndarray, img_h: int, img_w: int) -> np.ndarray | None:
    if cnt is None or len(cnt) < 3:
        return None
    area = cv2.contourArea(cnt)
    min_area = max(16.0, 0.00005 * img_h * img_w)
    if area < min_area:
        return None
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.002 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if approx is None or len(approx) < 3:
        return None
    return approx


def _bbox_to_contour(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    pts = np.array(
        [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    return pts


def _filter_wall_candidates(
    contours,
    img_h: int,
    img_w: int,
) -> list[np.ndarray]:
    """
    Strict Wall-only geometry filter.
    """

    if not contours:
        return []

    kept: list[np.ndarray] = []

    min_area = max(
        50.0,
        WALL_MIN_AREA_RATIO * img_h * img_w,
    )

    min_length = max(
        25.0,
        WALL_MIN_LENGTH_RATIO * min(img_h, img_w),
    )

    for cnt in contours:

        if cnt is None or len(cnt) < 3:
            continue

        cnt = np.asarray(cnt, dtype=np.int32)

        area = float(cv2.contourArea(cnt))

        if area < min_area:
            continue

        rect = cv2.minAreaRect(cnt)

        (_, _), (rw, rh), _ = rect

        long_side = max(float(rw), float(rh))
        short_side = min(float(rw), float(rh))

        if long_side <= 0:
            continue

        aspect = long_side / max(short_side, 1.0)

        # Wall should be long and thin
        if long_side < min_length:
            continue

        if aspect < WALL_MIN_ASPECT:
            continue

        # Solidity check
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))

        if hull_area <= 0:
            continue

        solidity = area / hull_area

        if solidity < 0.65:
            continue

        kept.append(cnt)

    return kept


def run_yolo_inference(
    img: np.ndarray,
    model_path: str | None = None,
    conf: float = 0.025,
    imgsz: int = 1280,    
    retry_conf: float = 0.01,
) -> tuple[dict, list[str], str]:
    """
    Run YOLO on img; return (labelled_dict, yolo_label_lines, model_path_used).
    labelled_dict keys use HCI PascalCase: Wall, Door, Window.
    """
    if img is None:
        raise ValueError("image unreadable")

    img_h, img_w = img.shape[:2]
    model, used_path = _get_model(model_path)
    
    # Find the model's Wall class id
    wall_model_id = None

    for model_id, model_name in model.names.items():
        mapped_name = map_model_class_to_hci(model_id,model.names,)

        if mapped_name == "Wall":
            wall_model_id = int(model_id)
            break
        
    logger.info(
        "Wall model class id: %s",
        wall_model_id,
    )
    
    
    labelled: dict = {}
    label_lines: list[str] = []
    confs = [conf, retry_conf]

    for attempt_conf in confs:
        labelled.clear()
        label_lines.clear()
        results = model.predict(img, imgsz=imgsz, conf=attempt_conf, verbose=False)
        result = results[0]
        boxes = result.boxes
        masks = result.masks

        if masks is not None and boxes is not None and len(boxes) > 0:
            mask_data = masks.data.cpu().numpy()
            box_data = boxes.data.cpu().numpy()
            for i, box in enumerate(box_data):
                cls_id = int(box[5])
                box_conf = float(box[4])
                hci_name = map_model_class_to_hci(cls_id, model.names)
                if hci_name is None:
                    continue
                
                if hci_name == "Wall":
                    if box_conf < WALL_MIN_CONF:
                        continue
             
                else:
                    if box_conf < 0.05:
                        continue
                        
                hci_cid = CLASS_IDS[hci_name]
                mask = mask_data[i]
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h))
                mask_bin = (mask > 0.5).astype(np.uint8)
                cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    clean = _filter_contour(cnt, img_h, img_w)
                    if clean is None:
                        continue
                    labelled.setdefault(hci_name, []).append(clean)
                    line = contour_to_yolo_seg(clean, img_w, img_h, hci_cid)
                    if line:
                        label_lines.append(line)

        elif boxes is not None and len(boxes) > 0:
            box_data = boxes.data.cpu().numpy()
            for box in box_data:
                cls_id = int(box[5])
                box_conf = float(box[4])
                hci_name = map_model_class_to_hci(cls_id, model.names)
                if hci_name is None:
                    continue
                
                if hci_name == "Wall":
                    if box_conf < WALL_MIN_CONF:
                        continue
               
                else:
                    if box_conf < 0.05:
                        continue
                    
                hci_cid = CLASS_IDS[hci_name]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cnt = _bbox_to_contour(x1, y1, x2, y2)
                clean = _filter_contour(cnt, img_h, img_w)
                if clean is None:
                    continue
                labelled.setdefault(hci_name, []).append(clean)
                line = contour_to_yolo_seg(clean, img_w, img_h, hci_cid)
                if line:
                    label_lines.append(line)

        if label_lines:
            break

    # ============================================================
    # WALL DETECTION + CONNECTED WALL POST-PROCESSING
    # ============================================================

    wall_id = CLASS_IDS["Wall"]

    try:
        # Only use Wall predictions produced by YOLO.
        # Do NOT create Wall labels from CV lines.
        yolo_walls = list(labelled.get("Wall", []))

        # Apply Wall-only geometry filtering.
        yolo_walls = _filter_wall_candidates(yolo_walls,img_h,img_w,)

        if yolo_walls:

            # ----------------------------------------------------
            # Connect L/T/X wall junctions
            # ----------------------------------------------------
            if connect_walls_enabled(default=True):

                new_walls, conn_report = connect_wall_contours(yolo_walls,img_w,img_h,)

            else:

                new_walls = yolo_walls

                conn_report = {
                    "enabled": False,
                    "input_walls": len(yolo_walls),
                    "output_walls": len(yolo_walls),
                }

            # ----------------------------------------------------
            # Store final Wall contours
            # ----------------------------------------------------
            labelled["Wall"] = new_walls
            labelled["_wall_connect"] = conn_report

            # ----------------------------------------------------
            # Remove old Wall label lines
            # ----------------------------------------------------
            label_lines = [
                ln
                for ln in label_lines
                if not ln.startswith(f"{wall_id} ")
            ]

            # ----------------------------------------------------
            # Write final connected Wall polygons
            # ----------------------------------------------------
            for cnt in new_walls:

                line = contour_to_yolo_seg(cnt,img_w,img_h,wall_id,)

                if line:
                    label_lines.append(line)

            logger.info(
                "Wall detection: YOLO=%d Final=%d",
                len(yolo_walls),
                len(new_walls),
                conn_report.get("duplicates_removed", 0),
            )

        else:

            labelled["Wall"] = []

            labelled["_wall_connect"] = {
                "enabled": connect_walls_enabled(default=True),
                "input_walls": 0,
                "output_walls": 0,
            }

            logger.info(
                "Wall detection: YOLO=0 Final=0"
            )

    except Exception:

        logger.exception(
            "wall detection/connect post-process failed; keeping YOLO walls"
        )
    return labelled, label_lines, used_path

   


def draw_detection_overlay(img: np.ndarray, labelled: dict) -> np.ndarray:
    """Draw colored filled polygons + class labels (same style as /api/detect)."""
    vis = img.copy()
    img_h, img_w = img.shape[:2]
    max_dim = max(img_h, img_w)
    font_scale = max(0.4, max_dim / 2500.0)
    thick = max(1, int(max_dim * 0.002))
    
    for cls_name, contours in labelled.items():
        if cls_name.startswith("_") or not isinstance(contours, list):
            continue
        hci_id = CLASS_IDS.get(cls_name, 3)
        color = HCI_COLORS.get(hci_id, (128, 128, 128))
        for cnt in contours:
            if cnt is None or len(cnt) < 3:
                continue
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            overlay = vis.copy()
            overlay[mask > 0] = color
            vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
            cv2.drawContours(vis, [cnt], -1, color, thick)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    vis,
                    cls_name,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thick,
                    cv2.LINE_AA,
                )
    return vis
