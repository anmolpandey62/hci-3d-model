# auto_label.py
import logging
import os

import cv2

from logic.yolo_inference import (
    ModelNotFoundError,
    contour_to_yolo_seg,
    draw_detection_overlay,
    resolve_hci21_model,
    run_yolo_inference,
)

logger = logging.getLogger(__name__)


def generate_labels(img_path, detector=None):
    """
    Run YOLO inference and return (label_lines, img, labelled) for _autolabel_worker.

    detector is accepted for API compatibility but not used (heuristic stub ignored).
    HCI_2.1 default: CubiCasa via resolve_hci21_model() / HCI21_MODEL_PATH.
    Never writes best_gdrive.pt.
    """
    img = cv2.imread(img_path)
    if img is None:
        labelled = {"_skip_reason": "image unreadable"}
        logger.error("generate_labels: cannot read %s", img_path)
        return [], None, labelled

    model_path, model_src = resolve_hci21_model()
    if not model_path:
        raise ModelNotFoundError(
            "No HCI_2.1 YOLO weights found. Set HCI21_MODEL_PATH to "
            r"D:\HCI_interor\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt"
        )

    try:
        labelled, label_lines, used = run_yolo_inference(img, model_path=model_path)
    except ModelNotFoundError:
        raise
    except Exception as exc:
        logger.exception("generate_labels inference failed for %s", img_path)
        labelled = {"_skip_reason": f"inference error: {exc}"}
        return [], img, labelled

    if not label_lines:
        labelled["_skip_reason"] = (
            f"zero detections (model={os.path.basename(used)}, source={model_src})"
        )
        logger.warning(
            "generate_labels: zero Wall/Door/Window detections for %s using %s (%s)",
            img_path,
            used,
            model_src,
        )
    else:
        logger.info(
            "generate_labels: %s polygons from %s using %s (%s)",
            len(label_lines),
            img_path,
            used,
            model_src,
        )
        labelled["_model_path"] = used
        labelled["_model_source"] = model_src

    return label_lines, img, labelled


def draw_labelled_image(img, labelled, marked_path):
    """Save coloured polygon overlay preview."""
    if img is None:
        return
    vis = draw_detection_overlay(img, labelled)
    cv2.imwrite(marked_path, vis)
