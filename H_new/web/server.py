#!/usr/bin/env python3
# web/server.py - FastAPI backend for Floor Plan Model Trainer Web UI

import os, sys, glob, shutil, threading, json, base64, traceback, asyncio
from pathlib import Path

# ── Interpreter lock (HCI_2.1) ────────────────────────────────────────────────
# Auto Label / YOLO require a working PyTorch. Anaconda base and
# D:\HCI_interor\web_file_v2\.venv both fail with WinError 1114 on c10.dll.
# Escape hatch (tests only): set HCI21_ALLOW_ANY_PYTHON=1
_HCI21_REQUIRED_PYTHON = Path(
    r"C:\Users\DELL\anaconda3\envs\improved_model_train\python.exe"
)

def _enforce_hci21_python() -> None:
    if os.environ.get("HCI21_ALLOW_ANY_PYTHON", "").strip().lower() in ("1", "true", "yes"):
        return
    exe = Path(sys.executable).resolve()
    required = _HCI21_REQUIRED_PYTHON.resolve()
    parts_lower = [p.lower() for p in exe.parts]
    if "improved_model_train" in parts_lower:
        return
    if exe == required:
        return
    msg = (
        "\n"
        "============================================================\n"
        "HCI_2.1 refused to start: wrong Python interpreter.\n"
        f"  Running:  {exe}\n"
        f"  Required: {required}\n"
        "\n"
        "web_file_v2\\.venv and Anaconda base fail to load torch (c10.dll).\n"
        "Fix: run START_HCI_2.1.bat or scripts\\start_server.bat\n"
        "  or: conda activate improved_model_train\n"
        "Override (not recommended): set HCI21_ALLOW_ANY_PYTHON=1\n"
        "============================================================\n"
    )
    sys.stderr.write(msg)
    raise SystemExit(1)

_enforce_hci21_python()

# ── Path setup ────────────────────────────────────────────────────────────────
LOGIC_DIR   = Path(__file__).resolve().parent.parent          # logic/
PROJECT_ROOT = LOGIC_DIR.parent                               # sam_env/
for p in [str(PROJECT_ROOT), str(LOGIC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if sys.platform == "darwin":
    _brew = "/opt/homebrew/lib"
    if os.path.isdir(_brew):
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", _brew)

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np

from config.classes import CLASS_IDS, ID_TO_CLASS
from auto_label import generate_labels, draw_labelled_image, contour_to_yolo_seg
from logic.room_text_mapper import analyse_image, draw_text_mapping_overlay
from logic.floor_plan_analyzer import analyse_floor_plan, draw_analysis_overlay, extract_text_seeds
from logic.dataset_io import (
    IMG_EXTS,
    ensure_dataset_dirs,
    ifc_file_path,
    list_raw_image_names,
    resolve_image_for_basename,
    save_raw_image_bytes,
)

GDRIVE_FOLDER_ID = "18IThRKRGUHFXnSiMtJlhqHSphDIuphNk"
DATASET_DIR      = PROJECT_ROOT / "gdrive_dataset"

# Ensure runtime dataset folders exist on startup (safe if already present).
ensure_dataset_dirs(DATASET_DIR)

# ── Shared state ──────────────────────────────────────────────────────────────
_log_queue: list[str] = []          # SSE log lines
_progress: dict       = {"pct": 0, "status": "Ready", "metrics": {}}
_analysis: dict       = {}          # basename -> {labelled, img, n_labels, label_lines}
_training_lock        = threading.Lock()
_training_active      = False

# ── Corrected-files tracker (for Update Model panel) ─────────────────────────
_corrected_basenames: set = set()   # basenames that have been corrected this session


def _mark_corrected(basename: str) -> None:
    """M9a: register basename for train_from_corrections 'corrected' session scope.

    Call only after a human correction has successfully persisted (e.g. after
    _rebuild_labels / explicit label write). Duplicate adds are harmless (set).
    """
    if basename:
        _corrected_basenames.add(basename)


def _push(msg: str, pct: float = None, status: str = None, metrics: dict = None):
    _log_queue.append(msg)
    if pct is not None:
        _progress["pct"] = round(pct, 1)
    if status:
        _progress["status"] = status
    if metrics:
        _progress["metrics"].update(metrics)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Floor Plan Trainer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _cv_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

# Phase 0 — promotion kill-switch: HCI_2.1 must NEVER write production weights.
PRODUCTION_MODEL_PATH = PROJECT_ROOT / "best_gdrive.pt"
HCI21_ACTIVE_MODEL_PATH = PROJECT_ROOT / "best_hci21.pt"


def _is_production_model_path(path: str | Path) -> bool:
    """True if path resolves to the shared production best_gdrive.pt."""
    try:
        return Path(path).resolve() == PRODUCTION_MODEL_PATH.resolve()
    except Exception:
        return os.path.normcase(os.path.abspath(str(path))) == os.path.normcase(
            os.path.abspath(str(PRODUCTION_MODEL_PATH))
        )


def _promote_hci21_checkpoint(src: str | Path, *, source: str = "") -> dict:
    """
    Promote a checkpoint for HCI_2.1 only.
    Writes best_hci21.pt. Never writes best_gdrive.pt (production).
    """
    src_path = Path(src)
    if not src_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {src_path}")
    dest = HCI21_ACTIVE_MODEL_PATH
    if _is_production_model_path(dest):
        raise RuntimeError("Refusing to promote: HCI21 destination resolves to best_gdrive.pt")
    shutil.copy2(str(src_path), str(dest))
    if not dest.is_file():
        raise FileNotFoundError(f"promotion failed: destination missing after copy: {dest}")
    # M9b: drop stale in-memory weights so Auto Label reloads promoted file
    from logic.yolo_inference import invalidate_model_cache
    invalidate_model_cache()
    note = (
        f"HCI_2.1 promoted → {dest.name}"
        + (f" (from {source})" if source else "")
        + f" | production protected: {PRODUCTION_MODEL_PATH.name} untouched"
    )
    return {
        "ok": True,
        "active": str(dest),
        "source": str(src_path),
        "production_protected": True,
        "production_model": str(PRODUCTION_MODEL_PATH),
        "note": note,
    }


def _find_best_model() -> str:
    """Return path to best_gdrive.pt (active model). Falls back to highest-mAP50 model."""
    active = PROJECT_ROOT / "best_gdrive.pt"
    if active.exists():
        return str(active)
    # Fallback: find highest mAP50 across all runs
    return _find_best_by_map()

def _find_best_by_map() -> str:
    """Scan all runs and return the model with highest mAP50."""
    search_roots = [
        PROJECT_ROOT / "gdrive_dataset" / "runs",
        PROJECT_ROOT / "runs",
        PROJECT_ROOT / "iterations",
        LOGIC_DIR / "gdrive_dataset" / "runs",
    ]
    best_path, best_map = "", 0.0
    for root in search_roots:
        for pt in glob.glob(str(root / "**" / "best.pt"), recursive=True):
            try:
                from ultralytics import YOLO as _YOLO
                m = _YOLO(pt)
                tr = (m.ckpt or {}).get("train_results", {})
                map50 = max(tr.get("metrics/mAP50(B)", [0]))
                if map50 > best_map:
                    best_map = map50
                    best_path = pt
            except Exception:
                pass
    return best_path

def _list_raw_images() -> list[str]:
    return list_raw_image_names(DATASET_DIR)

def _list_labelled_images() -> list[str]:
    return sorted(_analysis.keys())

# ── SSE stream ────────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def stream():
    async def event_gen():
        sent = 0
        while True:
            while sent < len(_log_queue):
                line = _log_queue[sent].replace("\n", " ")
                yield f"data: {json.dumps({'log': line, 'progress': _progress})}\n\n"
                sent += 1
            await asyncio.sleep(0.3)
    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    # Always sync _analysis with any new marked images on disk
    _load_existing_labels()
    from logic.yolo_inference import resolve_hci21_model, find_production_model_path

    hci21_path, hci21_src = resolve_hci21_model()
    return {**_progress, "training": _training_active,
            "raw_images": _list_raw_images(),
            "labelled_images": _list_labelled_images(),
            "best_model": hci21_path or _find_best_model(),
            "hci21_model": hci21_path,
            "hci21_model_source": hci21_src,
            "production_model": find_production_model_path(),
            "production_protected": True,
            "raw_folder": str(DATASET_DIR / "images_raw"),
            "labelled_folder": str(DATASET_DIR / "marked")}

# ── Download GDrive ───────────────────────────────────────────────────────────
@app.post("/api/download")
def download_gdrive(background_tasks: BackgroundTasks):
    background_tasks.add_task(_download_worker)
    return {"ok": True}

def _download_worker():
    dl_dir = DATASET_DIR / "images_raw"
    dl_dir.mkdir(parents=True, exist_ok=True)
    _push("Downloading from Google Drive...", status="Downloading...")
    try:
        import gdown
        url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
        gdown.download_folder(url, output=str(dl_dir), quiet=False, use_cookies=False)
        # Flatten subfolders
        for root_d, _, files in os.walk(dl_dir):
            for f in files:
                if Path(f).suffix.lower() in IMG_EXTS:
                    src, dst = Path(root_d) / f, dl_dir / f
                    if src != dst and not dst.exists():
                        shutil.move(str(src), str(dst))
        count = len([f for f in os.listdir(dl_dir) if Path(f).suffix.lower() in IMG_EXTS])
        _push(f"✅ Downloaded {count} images", pct=100, status=f"Downloaded {count} images")
    except Exception as e:
        _push(f"ERROR: {e}", status="Download failed")

# ── Upload images ─────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    saved = 0
    for f in files:
        data = await f.read()
        if save_raw_image_bytes(DATASET_DIR, f.filename, data) is not None:
            saved += 1
    _push(f"Uploaded {saved} images", status=f"Uploaded {saved} images")
    return {"saved": saved, "images": _list_raw_images(),
            "raw_folder": str(DATASET_DIR / "images_raw")}

# ── Auto-label ────────────────────────────────────────────────────────────────
@app.post("/api/autolabel")
def autolabel(background_tasks: BackgroundTasks, body: dict = None):
    selected         = (body or {}).get("files", [])
    metadata_choice  = (body or {}).get("metadata_choice", "local")
    background_tasks.add_task(_autolabel_worker, selected, metadata_choice)
    return {"ok": True}

def _autolabel_worker(selected_files: list = None, metadata_choice: str = "local"):
    """
    metadata_choice:
      'use'    — load existing JSON, skip re-analysis
      'local'  — run local OCR/heuristic analysis (default)
      'gemini' — placeholder: run local analysis (Gemini requires API key)
    """
    raw_dir = DATASET_DIR / "images_raw"
    if not raw_dir.is_dir():
        _push("ERROR: No images_raw dir", status="Error"); return
    all_files = sorted([raw_dir / f for f in os.listdir(raw_dir) if Path(f).suffix.lower() in IMG_EXTS])
    if not all_files:
        _push("ERROR: No images found", status="Error"); return
    # Filter to selected files if provided
    if selected_files:
        sel_set = set(selected_files)
        files = [f for f in all_files if f.name in sel_set]
        if not files:
            _push("ERROR: None of the selected files found in images_raw", status="Error"); return
        _push(f"Processing {len(files)} selected file(s) of {len(all_files)} total")
    else:
        files = all_files

    img_out  = DATASET_DIR / "images" / "train"
    lbl_out  = DATASET_DIR / "labels" / "train"
    mark_out = DATASET_DIR / "marked"
    for d in [img_out, lbl_out, mark_out]:
        d.mkdir(parents=True, exist_ok=True)

    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET_DIR}\ntrain: images/train\nval: images/train\ntest: images/train\n\n")
        f.write(f"nc: {len(CLASS_IDS)}\n\nnames:\n")
        for name, idx in sorted(CLASS_IDS.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {name}\n")

    from logic.detector import FloorPlanDetector
    from logic.yolo_inference import resolve_hci21_model
    from logic.window_inference import run_window_inference
    
    hci21_path, hci21_src = resolve_hci21_model()
    if not hci21_path:
        _push(
            "ERROR: No HCI_2.1 YOLO model found. Set HCI21_MODEL_PATH to "
            r"D:\HCI_interor\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt",
            status="Error",
        )
        return
    _push(f"HCI_2.1 Auto Label model: {hci21_path} ({hci21_src})")
    detector = FloorPlanDetector(debug_mode=False, output_dir=".",
                                  remove_captions=True, detection_mode="heuristic_only")
    success = 0
    for i, img_path in enumerate(files):
        basename = img_path.stem
        pct = (i + 1) / len(files) * 100
        _push(f"Labelling {i+1}/{len(files)}: {basename}", pct=pct, status=f"Labelling {i+1}/{len(files)}")
        try:
            label_lines, img, labelled = generate_labels(str(img_path), detector)
            # ------------------------------------------------------------
            # WINDOW-ONLY INFERENCE
            # ------------------------------------------------------------
            try:
                window_contours, _, _ = run_window_inference(
                    img,
                    model_path=hci21_path,
                )
                labelled["Window"] = window_contours
                _push(f"  Window pipeline: {len(window_contours)} detections")
                
            except Exception as e:
                _push(
                    f"   Window pipeline failed: {e}"
                )
            
            if img is None:
                _push(f"  SKIP: {basename} (image unreadable)"); 
                continue
            
             # CHECK ALL DETECTIONS
    
            has_any_labels = any(
                isinstance(v, list) and len(v) > 0
                for k, v in labelled.items()
                if not k.startswith("_")
            )
            if not has_any_labels:
                reason = (labelled or {}).get("_skip_reason","no labels")
                _push(
                    f"  SKIP: {basename} ({reason})"
                )
                continue

            # ── Enhanced analyzer: OCR seeds + watershed room splitting ──
            enhanced = analyse_floor_plan(img, labelled)
            # Merge enhanced results back into labelled
            for key in ["rooms", "stairs", "flow_terminals", "furniture"]:
                if enhanced.get(key) and len(enhanced[key]) > len(labelled.get(key, [])):
                    labelled[key] = enhanced[key]
                    _push(f"  Enhanced {key}: {len(enhanced[key])}")
            # Room names from OCR
            room_names = enhanced.get("_room_names", {})
            ocr_seeds  = enhanced.get("_ocr_seeds", [])
            was_corrected = enhanced.get("_analyzer_used", False)

            # Rebuild label lines from enriched labelled dict
            from auto_label import contour_to_yolo_seg
            img_h2, img_w2 = img.shape[:2]
            label_lines = []
            for cls_name, contours in labelled.items():
                if cls_name.startswith("_") or not isinstance(contours, list):
                    continue
                cid = CLASS_IDS.get(cls_name)
                if cid is None: continue
                for cnt in contours:
                    line = contour_to_yolo_seg(cnt, img_w2, img_h2, cid)
                    if line: label_lines.append(line)

            if not label_lines:
                _push(f"  SKIP (no labels after enhance): {basename}"); continue

            ext = img_path.suffix.lower()
            if ext == ".svg":
                cv2.imwrite(str(img_out / (basename + ".png")), img)
            else:
                shutil.copy2(str(img_path), str(img_out / img_path.name))
            with open(lbl_out / (basename + ".txt"), "w") as f:
                f.write("\n".join(label_lines) + "\n")
            marked_path = str(mark_out / (basename + "_labelled.jpg"))
            draw_labelled_image(img, labelled, marked_path)

            # ── Text OCR analysis for pre/post views ────────────────
            room_cnts = labelled.get("Room", [])
            text_analysis = analyse_image(img, room_cnts)
            text_mappings  = text_analysis["mappings"]
            assigned       = text_analysis["assigned"]

            # Log OCR findings
            if text_mappings:
                _push(f"  OCR found {len(text_mappings)} text label(s):")
                for m in assigned:
                    status = "✓" if m.get("inside") else "~"
                    _push(f"    {status} '{m['text']}' → {m['class']} (room #{m.get('room_idx',-1)+1})")

            # Save pre/post label images
            pre_path  = str(mark_out / (basename + "_pre_label.jpg"))
            post_path = str(mark_out / (basename + "_post_label.jpg"))
            cv2.imwrite(pre_path,  text_analysis["pre_label_img"])
            cv2.imwrite(post_path, text_analysis["post_label_img"])

            _analysis[basename] = {
                "labelled": labelled, "marked_path": marked_path,
                "n_labels": len(label_lines), "label_lines": label_lines,
                "img_h": img.shape[0], "img_w": img.shape[1],
                "text_analysis": {
                    "mappings": [{"text": m["text"], "class": m["class"],
                                   "cx": m["cx"], "cy": m["cy"],
                                   "room_idx": m.get("room_idx",-1),
                                   "inside": m.get("inside",False)} for m in assigned],
                    "summary": text_analysis["summary"],
                    "was_corrected": was_corrected,
                    "ocr_rooms": len(room_names),
                    "detector_rooms": len(room_cnts),
                },
                "pre_label_path":  pre_path,
                "post_label_path": post_path,
            }
            _analysis[basename]["img_b64"]        = _cv_to_b64(img)
            _analysis[basename]["marked_b64"]     = _cv_to_b64(cv2.imread(marked_path))
            _analysis[basename]["pre_label_b64"]  = _cv_to_b64(cv2.imread(pre_path))
            _analysis[basename]["post_label_b64"] = _cv_to_b64(cv2.imread(post_path))

            r  = len(labelled.get("Room", []))
            d2 = len(labelled.get("Door", []))
            w  = len(labelled.get("Window", []))
            fu = len(labelled.get("Furniture", []))
            correction_note = " [OCR-corrected]" if was_corrected else ""
            _push(f"  ✓ {basename}: R={r} D={d2} W={w} F={fu}{correction_note}")

            # Save metadata JSON for this image
            try:
                room_names_meta = enhanced.get("_room_names", {})
                ocr_seeds_meta  = enhanced.get("_ocr_seeds", [])
                meta = build_metadata_from_ocr(
                    str(img_path), img, ocr_seeds_meta, room_names_meta,
                    label_lines, labelled, img.shape[0], img.shape[1]
                )
                save_metadata(str(img_path), DATASET_DIR, meta)
                _push(f"  📄 Metadata saved: {basename}.json")
            except Exception as me:
                _push(f"  ⚠️ Metadata save failed: {me}")
            success += 1
        except Exception as e:
            _push(f"  ERROR {basename}: {e}")
    _push(f"\n✅ Done: {success}/{len(files)} labelled", pct=100, status=f"Labelled {success} images")
    # Ensure all new marked images are loaded into _analysis
    _load_existing_labels()

# ── Startup: load existing marked images from disk ───────────────────────────
@app.on_event("startup")
def _load_existing_labels():
    """Load/refresh all labelled images from disk into _analysis.
    Always re-reads — so re-runs pick up the latest marked images.
    Also reconstructs contours from YOLO label lines so corrections work.
    """
    marked_dir = DATASET_DIR / "marked"
    lbl_dir    = DATASET_DIR / "labels" / "train"
    raw_dir    = DATASET_DIR / "images_raw"
    if not marked_dir.is_dir():
        return

    for marked_path in sorted(marked_dir.glob("*_labelled.jpg")):
        basename = marked_path.stem.replace("_labelled", "")

        # Always reload from disk — never skip on re-run
        img = cv2.imread(str(marked_path))
        if img is None:
            continue

        # Load original raw image
        orig = None
        for ext in list(IMG_EXTS):
            for candidate in [raw_dir / (basename + ext), raw_dir / (basename + ext.upper())]:
                if candidate.exists():
                    orig = cv2.imread(str(candidate))
                    break
            if orig is not None:
                break
        h, w = (orig.shape[:2] if orig is not None else img.shape[:2])

        # Read label lines
        lbl_file = lbl_dir / (basename + ".txt")
        label_lines = []
        if lbl_file.exists():
            label_lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]

        # Build counts + reconstruct contours from YOLO polygons
        counts: dict = {}
        labelled: dict = {}
        for line in label_lines:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                cid = int(parts[0])
                cls = ID_TO_CLASS.get(cid, f"cls{cid}")
                coords = list(map(float, parts[1:]))
                pts = []
                for k in range(0, len(coords) - 1, 2):
                    px = int(coords[k] * w)
                    py = int(coords[k+1] * h)
                    pts.append([px, py])
                if len(pts) >= 3:
                    cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    labelled.setdefault(cls, []).append(cnt)
                    counts[cls] = counts.get(cls, 0) + 1
            except Exception:
                pass

        # Load pre/post label images if they exist
        pre_path  = marked_dir / (basename + "_pre_label.jpg")
        post_path = marked_dir / (basename + "_post_label.jpg")

        entry = {
            "labelled":       labelled,
            "marked_path":    str(marked_path),
            "n_labels":       len(label_lines),
            "label_lines":    label_lines,
            "img_h":          h, "img_w": w,
            "img_b64":        _cv_to_b64(orig) if orig is not None else "",
            "marked_b64":     _cv_to_b64(img),
            "pre_label_b64":  _cv_to_b64(cv2.imread(str(pre_path)))  if pre_path.exists()  else "",
            "post_label_b64": _cv_to_b64(cv2.imread(str(post_path))) if post_path.exists() else "",
            "_counts":        counts,
            "from_disk":      True,
        }
        _analysis[basename] = entry

# ── Get labelled image ────────────────────────────────────────────────────────
@app.get("/api/image/{basename}")
def get_image(basename: str):
    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    if info.get("from_disk"):
        # Recount from label_lines in case corrections were applied
        counts: dict = {}
        for line in info.get("label_lines", []):
            try:
                cid = int(line.split()[0])
                cls = ID_TO_CLASS.get(cid, f"cls{cid}")
                counts[cls] = counts.get(cls, 0) + 1
            except Exception:
                pass
        info["_counts"] = counts
        return {"marked_b64":     info.get("marked_b64", ""),
                "pre_label_b64":  info.get("pre_label_b64", ""),
                "post_label_b64": info.get("post_label_b64", ""),
                "text_analysis":  info.get("text_analysis", {}),
                "labels": counts, "n_labels": info["n_labels"]}
    labels_summary = {k: len(v) for k, v in info["labelled"].items() if v}
    return {"marked_b64":     info.get("marked_b64", ""),
            "pre_label_b64":  info.get("pre_label_b64", ""),
            "post_label_b64": info.get("post_label_b64", ""),
            "text_analysis":  info.get("text_analysis", {}),
            "labels": labels_summary, "n_labels": info["n_labels"]}

# ── Correct labels ────────────────────────────────────────────────────────────
@app.post("/api/correct")
def correct_label(body: dict):
    basename  = body.get("basename")
    action    = body.get("action")   # "remove" or "relabel"
    cls_name  = body.get("cls_name")
    idx       = int(body.get("idx", 1)) - 1
    new_cls   = body.get("new_cls", "")

    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "Image not labelled yet"}, status_code=404)

    # For from_disk entries, labelled dict is now populated from contours
    # but if still empty, fall back to label_line manipulation
    items = info["labelled"].get(cls_name, [])

    if len(items) == 0 and info.get("from_disk"):
        # Fallback: manipulate label_lines directly by class+index
        return _correct_label_lines(basename, info, action, cls_name, idx, new_cls)

    if idx < 0 or idx >= len(items):
        return JSONResponse({"error": f"Index out of range (1-{len(items)})"}, status_code=400)

    if action == "remove":
        info["labelled"][cls_name].pop(idx)
        msg = f"Removed {cls_name} #{idx+1}"
    elif action == "relabel":
        if new_cls not in CLASS_IDS:
            return JSONResponse({"error": f"Unknown class: {new_cls}"}, status_code=400)
        cnt = info["labelled"][cls_name].pop(idx)
        info["labelled"].setdefault(new_cls, []).append(cnt)
        msg = f"Relabelled {cls_name} #{idx+1} → {new_cls}"
    else:
        return JSONResponse({"error": "action must be remove or relabel"}, status_code=400)

    _rebuild_labels(basename, info)
    _push(msg, status=msg)
    labels_out = {k: len(v) for k, v in info["labelled"].items() if v}
    info["_counts"] = labels_out
    _mark_corrected(basename)
    return {"ok": True, "msg": msg, "labels": labels_out,
            "marked_b64": info.get("marked_b64", "")}


def _correct_label_lines(basename, info, action, cls_name, idx, new_cls):
    """Correct by directly editing YOLO label lines (fallback for disk entries)."""
    lines = info.get("label_lines", [])
    cid_target = CLASS_IDS.get(cls_name)
    if cid_target is None:
        return JSONResponse({"error": f"Unknown class: {cls_name}"}, status_code=400)

    # Find all lines matching cls_name, pick the idx-th one
    matching = [(i, l) for i, l in enumerate(lines) if l.split()[0] == str(cid_target)]
    if idx < 0 or idx >= len(matching):
        return JSONResponse({"error": f"Index out of range (1-{len(matching)})"}, status_code=400)

    line_idx, line = matching[idx]

    if action == "remove":
        lines.pop(line_idx)
        msg = f"Removed {cls_name} #{idx+1}"
    elif action == "relabel":
        if new_cls not in CLASS_IDS:
            return JSONResponse({"error": f"Unknown class: {new_cls}"}, status_code=400)
        new_cid = CLASS_IDS[new_cls]
        parts = line.split()
        parts[0] = str(new_cid)
        lines[line_idx] = " ".join(parts)
        msg = f"Relabelled {cls_name} #{idx+1} → {new_cls}"
    else:
        return JSONResponse({"error": "action must be remove or relabel"}, status_code=400)

    info["label_lines"] = lines
    info["n_labels"] = len(lines)

    # Save to disk
    lbl_path = DATASET_DIR / "labels" / "train" / (basename + ".txt")
    lbl_path.write_text("\n".join(lines) + "\n")

    # Recount
    counts: dict = {}
    for l in lines:
        try:
            cid = int(l.split()[0])
            cls = ID_TO_CLASS.get(cid, f"cls{cid}")
            counts[cls] = counts.get(cls, 0) + 1
        except Exception:
            pass
    info["_counts"] = counts

    _push(msg, status=msg)
    _mark_corrected(basename)
    return {"ok": True, "msg": msg, "labels": counts,
            "marked_b64": info.get("marked_b64", "")}

def _rebuild_labels(basename: str, info: dict):
    img_h, img_w = info["img_h"], info["img_w"]
    lines = []
    for cls_name, contours in info["labelled"].items():
        cid = CLASS_IDS.get(cls_name)
        if cid is None: continue
        for cnt in contours:
            line = contour_to_yolo_seg(cnt, img_w, img_h, cid)
            if line: lines.append(line)
    info["label_lines"] = lines
    info["n_labels"] = len(lines)
    lbl_path = DATASET_DIR / "labels" / "train" / (basename + ".txt")
    if lbl_path.exists():
        lbl_path.write_text("\n".join(lines) + "\n")
    # Redraw marked image
    marked = info.get("marked_path")
    if marked:
        # Reconstruct img from b64
        img_data = base64.b64decode(info.get("img_b64", ""))
        arr = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            draw_labelled_image(img, info["labelled"], marked)
            info["marked_b64"] = _cv_to_b64(cv2.imread(marked))

# ── Train ─────────────────────────────────────────────────────────────────────
@app.post("/api/train")
def train(body: dict, background_tasks: BackgroundTasks):
    global _training_active
    if _training_active:
        return JSONResponse({"error": "Training already running"}, status_code=409)
    epochs = int(body.get("epochs", 1))
    batch  = int(body.get("batch", 4))
    imgsz  = int(body.get("imgsz", 640))
    background_tasks.add_task(_train_worker, epochs, batch, imgsz)
    return {"ok": True}


@app.get("/api/cubicasa_imports")
def list_cubicasa_imports():
    """List H-managed Cubicasa import versions (no dependency on Cubicasa source)."""
    from logic.cubicasa_dataset_import import list_imported_versions

    return {"ok": True, "imports": list_imported_versions()}


@app.post("/api/train_imported_dataset")
def train_imported_dataset(body: dict, background_tasks: BackgroundTasks):
    """
    Train an isolated experiment on an H-managed Cubicasa import.

    Does NOT promote to best_hci21.pt or best_gdrive.pt.
    Does NOT change auto-label defaults.
    Body: { version, epochs?, batch?, imgsz?, run_name? }
    """
    global _training_active
    if _training_active:
        return JSONResponse({"error": "Training already running"}, status_code=409)
    version = str(body.get("version") or "").strip()
    if not version:
        return JSONResponse({"error": "version is required (H import id)"}, status_code=400)
    try:
        from logic.cubicasa_dataset_import import resolve_training_yaml

        yaml_path = resolve_training_yaml(version)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    epochs = int(body.get("epochs", 1))
    batch = int(body.get("batch", 4))
    imgsz = int(body.get("imgsz", 640))
    run_name = str(body.get("run_name") or "").strip() or None
    background_tasks.add_task(
        _train_imported_worker, str(yaml_path), version, epochs, batch, imgsz, run_name
    )
    return {
        "ok": True,
        "version": version,
        "dataset_yaml": str(yaml_path),
        "promote": False,
        "note": "Isolated experiment only — production / HCI21 weights will not be overwritten",
    }


def _train_imported_worker(
    yaml_path: str,
    version: str,
    epochs: int,
    batch: int,
    imgsz: int,
    run_name: str | None,
):
    """Train from H-managed import yaml; never promote weights."""
    global _training_active
    _training_active = True
    try:
        from ultralytics import YOLO
        import torch
        import datetime as _dt

        model_base = str(PROJECT_ROOT / "yolov8n-seg.pt")
        if not os.path.exists(model_base):
            model_base = "yolov8n-seg.pt"
        model = YOLO(model_base)

        def on_epoch_end(trainer):
            ep = trainer.epoch + 1
            _push(
                f"  [import/{version}] Epoch {ep}/{epochs}",
                pct=ep / epochs * 100,
                status=f"Import train {ep}/{epochs}",
                metrics={"Epoch": f"{ep}/{epochs}"},
            )

        model.add_callback("on_train_epoch_end", on_epoch_end)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        name = run_name or f"cubicasa_furniture_ea_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Isolated runs under hci_datasets — not gdrive_dataset fine-tunes that promote
        project_dir = str(PROJECT_ROOT / "hci_datasets" / "runs")
        Path(project_dir).mkdir(parents=True, exist_ok=True)
        _push(f"  Imported dataset train: version={version}")
        _push(f"  yaml={yaml_path}")
        _push(f"  Device: {device}  run={name}  (NO weight promotion)")
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project_dir,
            name=name,
            workers=0,
            amp=True,
            verbose=False,
        )
        run_dir = Path(project_dir) / name
        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            _push(
                f"\n✅ Import training complete (not promoted): {best}",
                pct=100,
                status="Import training complete (not promoted)",
            )
            _push(f"   🛡️ Untouched: {PRODUCTION_MODEL_PATH.name}, {HCI21_ACTIVE_MODEL_PATH.name}")
        else:
            _push("⚠️ No best.pt in import run", status="Done (no model)")
    except Exception as e:
        _push(f"ERROR: {e}\n{traceback.format_exc()}", status="Import training failed")
    finally:
        _training_active = False


def _train_worker(epochs: int, batch: int, imgsz: int):
    global _training_active
    _training_active = True
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        _push("ERROR: Run auto-label first", status="Error")
        _training_active = False; return
    try:
        from ultralytics import YOLO
        import torch
        model_base = str(PROJECT_ROOT / "yolov8n-seg.pt")
        if not os.path.exists(model_base):
            model_base = "yolov8n-seg.pt"
        model = YOLO(model_base)

        def on_epoch_end(trainer):
            ep = trainer.epoch + 1
            _push(f"  Epoch {ep}/{epochs}", pct=ep/epochs*100,
                  status=f"Epoch {ep}/{epochs}", metrics={"Epoch": f"{ep}/{epochs}"})
            loss = trainer.label_loss_items(trainer.tloss)
            if loss:
                m = {}
                if "train/box_loss" in loss: m["Box Loss"] = f"{loss['train/box_loss']:.4f}"
                if "train/seg_loss" in loss: m["Seg Loss"] = f"{loss['train/seg_loss']:.4f}"
                if m: _progress["metrics"].update(m)

        def on_fit_end(trainer):
            m = trainer.metrics
            rd = m.results_dict if hasattr(m, "results_dict") else (m if isinstance(m, dict) else {})
            out = {}
            if "metrics/mAP50(B)" in rd:    out["mAP50"]    = f"{rd['metrics/mAP50(B)']:.4f}"
            if "metrics/mAP50-95(B)" in rd: out["mAP50-95"] = f"{rd['metrics/mAP50-95(B)']:.4f}"
            if out: _progress["metrics"].update(out)

        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_end)

        device = "cuda" if torch.cuda.is_available() else \
                 ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        _push(f"  Device: {device}")

        # Count training files and log them
        train_img_dir = DATASET_DIR / "images" / "train"
        train_lbl_dir = DATASET_DIR / "labels" / "train"
        img_files = sorted(train_img_dir.glob("*")) if train_img_dir.exists() else []
        img_files = [f for f in img_files if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
        _push(f"  Dataset: {len(img_files)} training images")
        for f in img_files:
            lbl = train_lbl_dir / (f.stem + ".txt")
            n_labels = len(lbl.read_text().splitlines()) if lbl.exists() else 0
            _push(f"    [{f.name}]  labels: {n_labels}")
        _push(f"  Starting {epochs} epoch(s) with batch={batch} imgsz={imgsz}...")

        project_dir = str(DATASET_DIR / "runs")
        model.train(data=str(yaml_path), epochs=epochs, batch=batch, imgsz=imgsz,
                    device=device, project=project_dir, name="train",
                    workers=0, amp=True, verbose=False)

        cands = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
        if cands:
            best = max(cands, key=os.path.getmtime)
            # Phase 0: never write best_gdrive.pt — promote to best_hci21.pt only
            promo = _promote_hci21_checkpoint(best, source="train")
            _push(
                f"\n✅ Training complete! HCI_2.1 model: {promo['active']}",
                pct=100,
                status="Training complete!",
            )
            _push(f"   🛡️ Production protected: {PRODUCTION_MODEL_PATH} untouched")
        else:
            _push("⚠️ No best.pt found", status="Done (no model)")
    except Exception as e:
        _push(f"ERROR: {e}\n{traceback.format_exc()}", status="Training failed")
    finally:
        _training_active = False

# ── Test / Detect ─────────────────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(file: UploadFile = File(...),
                 model_path: str = Form(""),
                 imgsz: int = Form(640),
                 conf_thresh: float = Form(0.1)):
    if not model_path:
        # Phase 1: default Test Model = HCI_2.1 CubiCasa resolver (not best_gdrive.pt)
        from logic.yolo_inference import resolve_hci21_model
        model_path, _src = resolve_hci21_model()
    if not model_path or not os.path.exists(model_path):
        return JSONResponse(
            {
                "error": "No HCI_2.1 model found. Set HCI21_MODEL_PATH or ensure "
                r"cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt exists."
            },
            status_code=400,
        )

    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)

    try:
        from ultralytics import YOLO
        model   = YOLO(model_path)

        # Check model quality from training history
        model_quality = "unknown"
        model_warning = ""
        try:
            ckpt = model.ckpt or {}
            tr   = ckpt.get("train_results", {})
            map50_list = tr.get("metrics/mAP50(B)", [0])
            best_map50 = max(map50_list) if map50_list else 0
            epochs_trained = len(tr.get("epoch", []))
            if best_map50 < 0.01 or epochs_trained == 0:
                model_quality = "undertrained"
                model_warning = (f"⚠️ Model undertrained (mAP50={best_map50:.4f}, {epochs_trained} epochs). "
                                 f"Need more images (100+ per class) for reliable detection. "
                                 f"Showing heuristic fallback.")
            elif best_map50 < 0.3:
                model_quality = "weak"
                model_warning = f"⚠️ Model weak (mAP50={best_map50:.2f}). Results may be inaccurate."
            else:
                model_quality = "good"
        except Exception:
            pass

        COLORS = {
            0:(0,0,200),1:(255,0,255),2:(0,165,255),3:(0,200,0),4:(100,100,100),
            5:(80,80,80),6:(255,165,0),7:(139,69,19),8:(100,100,255),9:(180,180,180),
            10:(200,150,150),11:(200,200,0),12:(210,180,140),13:(255,255,165),
            14:(165,165,255),15:(165,255,165),16:(255,200,100),
        }
        img_h, img_w = img.shape[:2]
        max_dim = max(img_h, img_w)
        font_scale = max(0.4, max_dim / 2500.0)
        thick = max(1, int(max_dim * 0.002))

        # Try YOLO at low confidence
        results = model(img, imgsz=imgsz, conf=conf_thresh, verbose=False)
        result  = results[0]
        vis     = img.copy()
        counts  = {}
        source  = "yolo"

        if result.masks and len(result.boxes) > 0:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(box[5]); conf = float(box[4])
                cls_name = ID_TO_CLASS.get(cls_id, f"cls{cls_id}")
                color    = COLORS.get(cls_id, (128,128,128))
                counts[cls_name] = counts.get(cls_name, 0) + 1
                mask = masks[i]
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h))
                mask_bin = (mask > 0.5).astype(np.uint8)
                overlay  = vis.copy(); overlay[mask_bin > 0] = color
                vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
                cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, color, thick)
                x1, y1 = int(box[0]), int(box[1])
                cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, max(y1-5,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

        # Fallback to heuristic if YOLO found nothing
        if not counts:
            source = "heuristic"
            if not model_warning:
                model_warning = "ℹ️ YOLO found 0 detections — showing heuristic fallback."
            try:
                from logic.detector import FloorPlanDetector
                from logic.floor_plan_analyzer import analyse_floor_plan
                detector = FloorPlanDetector(debug_mode=False, output_dir=".",
                                             remove_captions=True, detection_mode="heuristic_only")
                heur = detector.detect(img)
                enhanced = analyse_floor_plan(img, heur)
                for key in ["rooms","doors","windows","furniture","stairs","flow_terminals"]:
                    if enhanced.get(key):
                        heur[key] = enhanced[key]

                vis = img.copy()
                cls_map = {"rooms":"Room","doors":"Door","windows":"Window",
                           "furniture":"Furniture","stairs":"Stair","flow_terminals":"FlowTerminal"}
                for res_key, cls_name in cls_map.items():
                    color = COLORS.get({"Room":3,"Door":2,"Window":1,"Furniture":11,
                                        "Stair":8,"FlowTerminal":15}.get(cls_name,3), (128,128,128))
                    for cnt in heur.get(res_key, []):
                        overlay = vis.copy()
                        cv2.drawContours(overlay, [cnt], -1, color, -1)
                        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
                        cv2.drawContours(vis, [cnt], -1, color, thick)
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx2, cy2 = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            cv2.putText(vis, cls_name, (cx2, cy2),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
                        n = counts.get(cls_name, 0) + 1
                        counts[cls_name] = n
            except Exception as he:
                model_warning += f" Heuristic also failed: {he}"

        # Add warning banner to image
        if model_warning:
            banner_h = max(36, int(max_dim * 0.025))
            banner = np.zeros((banner_h, img_w, 3), dtype=np.uint8)
            banner[:] = (30, 80, 180) if source == "heuristic" else (30, 30, 80)
            cv2.putText(banner, model_warning[:120], (8, banner_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.35, max_dim/4000.0),
                        (255, 255, 255), 1, cv2.LINE_AA)
            vis = np.vstack([banner, vis])

        orig_b64   = _cv_to_b64(img)
        result_b64 = _cv_to_b64(vis)
        from logic.yolo_inference import resolve_hci21_model, find_production_model_path
        _hci21, _hci21_src = resolve_hci21_model()
        is_prod_compare = bool(
            model_path
            and find_production_model_path()
            and Path(model_path).resolve() == Path(find_production_model_path()).resolve()
        )
        return {
            "orig_b64": orig_b64,
            "result_b64": result_b64,
            "counts": counts,
            "source": source,
            "model_quality": model_quality,
            "warning": model_warning,
            "model_path": model_path,
            "model_role": "production_compare_ro" if is_prod_compare else "hci21_cubicasa",
            "hci21_default": _hci21,
            "hci21_model_source": _hci21_src,
            "production_protected": True,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── On-demand text analysis ──────────────────────────────────────────────────
@app.post("/api/analyse")
async def analyse_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Cannot decode image"}, status_code=400)
    try:
        result = analyse_image(img)
        overlay = draw_text_mapping_overlay(img, result["mappings"])
        return {
            "orig_b64":    _cv_to_b64(img),
            "overlay_b64": _cv_to_b64(overlay),
            "mappings":    [{"text": m["text"], "class": m["class"],
                              "cx": m["cx"], "cy": m["cy"]} for m in result["mappings"]],
            "summary":     result["summary"],
            "ocr_words":   [{"text": r["text"], "clean": r["clean_text"],
                              "conf": r["conf"], "x": r["x"], "y": r["y"]}
                             for r in result["regions"]],
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = WEB_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

# ── Raw image preview ─────────────────────────────────────────────────────────
@app.get("/api/raw/{filename}")
def get_raw_image(filename: str):
    raw_dir = DATASET_DIR / "images_raw"
    # Try exact match first, then case-insensitive
    path = raw_dir / filename
    if not path.exists():
        matches = [f for f in raw_dir.iterdir() if f.name.lower() == filename.lower()]
        if not matches:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = matches[0]
    ext = path.suffix.lower()
    if ext == ".svg":
        try:
            import cairosvg
            png_data = cairosvg.svg2png(url=str(path), output_width=1024)
            arr = np.frombuffer(png_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return JSONResponse({"error": "SVG load failed"}, status_code=500)
    else:
        img = cv2.imread(str(path))
    if img is None:
        return JSONResponse({"error": "cannot decode"}, status_code=500)
    return {"img_b64": _cv_to_b64(img), "filename": filename,
            "width": img.shape[1], "height": img.shape[0]}

# ── Raw image thumbnail (returns JPEG bytes directly) ────────────────────────
@app.get("/api/raw_thumb/{filename}")
def get_raw_thumb(filename: str):
    raw_dir = DATASET_DIR / "images_raw"
    path = raw_dir / filename
    if not path.exists():
        matches = [f for f in raw_dir.iterdir() if f.name.lower() == filename.lower()]
        if not matches:
            return JSONResponse({"error": "not found"}, status_code=404)
        path = matches[0]
    ext = path.suffix.lower()
    if ext == ".svg":
        try:
            import cairosvg
            png_data = cairosvg.svg2png(url=str(path), output_width=120)
            arr = np.frombuffer(png_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return JSONResponse({"error": "SVG load failed"}, status_code=500)
    else:
        img = cv2.imread(str(path))
    if img is None:
        return JSONResponse({"error": "cannot decode"}, status_code=500)
    h, w = img.shape[:2]
    scale = 80 / max(h, w)
    thumb = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
    from fastapi.responses import Response
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# ── Thumbnail (small version of marked image) ─────────────────────────────────
@app.get("/api/thumb/{basename}")
def get_thumb(basename: str):
    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    marked = info.get("marked_path")
    if not marked or not os.path.exists(marked):
        return JSONResponse({"error": "no marked image"}, status_code=404)
    img = cv2.imread(marked)
    if img is None:
        return JSONResponse({"error": "cannot read"}, status_code=500)
    # Resize to thumbnail
    h, w = img.shape[:2]
    scale = 80 / max(h, w)
    thumb = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))))
    from fastapi.responses import Response
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# ── Save corrections (explicit save to disk) ──────────────────────────────────
@app.post("/api/save_corrections")
def save_corrections(body: dict):
    basename = body.get("basename")
    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    lbl_path = DATASET_DIR / "labels" / "train" / (basename + ".txt")
    lines = info.get("label_lines", [])
    lbl_path.write_text("\n".join(lines) + "\n")
    # Redraw marked image if contours available
    if info.get("labelled") and not info.get("from_disk"):
        marked = info.get("marked_path")
        if marked:
            img_data = base64.b64decode(info.get("img_b64", ""))
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                draw_labelled_image(img, info["labelled"], marked)
                info["marked_b64"] = _cv_to_b64(cv2.imread(marked))
    _mark_corrected(basename)
    return {"ok": True, "saved": str(lbl_path), "n_labels": len(lines)}

# ── Revert to original labels from disk backup ────────────────────────────────
@app.post("/api/revert")
def revert_corrections(body: dict):
    basename = body.get("basename")
    lbl_path = DATASET_DIR / "labels" / "train" / (basename + ".txt")
    bak_path = Path(str(lbl_path) + ".bak")
    if bak_path.exists():
        import shutil as _sh
        _sh.copy2(str(bak_path), str(lbl_path))
        # Reload into _analysis
        lines = [l.strip() for l in bak_path.read_text().splitlines() if l.strip()]
        if basename in _analysis:
            _analysis[basename]["label_lines"] = lines
            _analysis[basename]["n_labels"] = len(lines)
            counts: dict = {}
            for line in lines:
                try:
                    cid = int(line.split()[0])
                    cls = ID_TO_CLASS.get(cid, f"cls{cid}")
                    counts[cls] = counts.get(cls, 0) + 1
                except Exception:
                    pass
            _analysis[basename]["_counts"] = counts
        return {"ok": True, "reverted_to": str(bak_path)}
    # No backup — just reload from current file
    if lbl_path.exists():
        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if basename in _analysis:
            _analysis[basename]["label_lines"] = lines
            _analysis[basename]["n_labels"] = len(lines)
        return {"ok": True, "note": "No backup found, reloaded current file"}
    return JSONResponse({"error": "No label file found"}, status_code=404)

# ── Draw annotation: bbox (legacy) or polygon points (Phase 2 wall strip) ─────
# M5a: must match FREE_POLY_CLASSES in web/index.html (Door/Window/Furniture/EA)
_FREE_POLY_RESIZE_CLASSES = frozenset({
    "Door", "Window", "Furniture", "ElectricAppliance",
})


def correct_wall_strip_enabled(default: bool = True) -> bool:
    """HCI_CORRECT_WALL_STRIP=0|1. Default ON for Phase 2 strip drawing (frontend-gated too)."""
    v = os.environ.get("HCI_CORRECT_WALL_STRIP", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _contour_from_bbox(bbox, img_w: int, img_h: int) -> np.ndarray:
    """Axis-aligned rectangle contour from [x, y, w, h] image pixels."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.int32,
    ).reshape(-1, 1, 2)


def _segments_properly_intersect(a, b, c, d) -> bool:
    """True if open segments ab and cd properly cross (shared endpoints ignored)."""

    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def _on_seg(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 == 0 and _on_seg(a, c, b):
        return False
    if o2 == 0 and _on_seg(a, d, b):
        return False
    if o3 == 0 and _on_seg(c, a, d):
        return False
    if o4 == 0 and _on_seg(c, b, d):
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _polygon_self_intersects(pts) -> bool:
    """pts: list[[x,y], ...] closed ring implied. Adjacent edges ignored."""
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        for j in range(i + 1, n):
            # skip adjacent edges and the closing edge pair
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == n - 1:
                continue
            c = pts[j]
            d = pts[(j + 1) % n]
            # edges that share a vertex are adjacent in ring sense when j+1==i
            if (j + 1) % n == i:
                continue
            if _segments_properly_intersect(a, b, c, d):
                return True
    return False


def _contour_from_points(points, img_w: int, img_h: int, *, reject_self_intersect: bool = False):
    """
    Build OpenCV contour from [[x,y], ...] image pixels.
    Returns (cnt, error_message). Does NOT convert to AABB.
    """
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None, "points must be an array of at least 3 [x,y] pairs"
    pts: list[list[int]] = []
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None, f"points[{i}] must be [x, y]"
        try:
            xf, yf = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            return None, f"points[{i}] coordinates must be numeric"
        if not (xf == xf and yf == yf):  # NaN check
            return None, f"points[{i}] coordinates must be finite"
        if abs(xf) == float("inf") or abs(yf) == float("inf"):
            return None, f"points[{i}] coordinates must be finite"
        xi = int(round(xf))
        yi = int(round(yf))
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))
        pts.append([xi, yi])
    if reject_self_intersect and _polygon_self_intersects(pts):
        return None, "polygon is self-intersecting"
    cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    area = float(abs(cv2.contourArea(cnt)))
    if area < 1.0:
        return None, "polygon has zero (or near-zero) area"
    return cnt, None


def _ensure_labelled_contours(info: dict, img_w: int, img_h: int) -> dict:
    """Ensure info['labelled'] has OpenCV contours (rebuild from YOLO lines if needed)."""
    labelled = info.get("labelled", {})
    if labelled:
        return labelled
    if not info.get("from_disk"):
        return labelled
    rebuilt: dict = {}
    for line in info.get("label_lines", []):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            cid = int(parts[0])
            cls = ID_TO_CLASS.get(cid, f"cls{cid}")
            coords = list(map(float, parts[1:]))
            pts = [
                [int(coords[k] * img_w), int(coords[k + 1] * img_h)]
                for k in range(0, len(coords) - 1, 2)
            ]
            if len(pts) >= 3:
                cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                rebuilt.setdefault(cls, []).append(cnt)
        except Exception:
            pass
    info["labelled"] = rebuilt
    return rebuilt


@app.post("/api/section")
def add_section(body: dict):
    """
    Add a user-drawn label contour.

    Accepts either:
      - points: [[x,y], ...]  (Phase 2 oriented wall strip / polygon; preserved as-is)
      - bbox: [x, y, w, h]    (legacy axis-aligned rectangle)

    Coordinates are image pixels. YOLO-seg format unchanged via _rebuild_labels.
    """
    basename = body.get("basename")
    bbox     = body.get("bbox")       # [x, y, w, h] optional if points given
    points   = body.get("points")     # [[x,y], ...] optional if bbox given
    label    = body.get("label", "Room")

    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "Image not labelled yet — run auto-label first"}, status_code=404)
    if label not in CLASS_IDS:
        return JSONResponse({"error": f"Unknown class: {label}"}, status_code=400)

    img_h, img_w = info["img_h"], info["img_w"]
    used_points = False

    if points is not None:
        cnt, err = _contour_from_points(points, img_w, img_h)
        if err:
            return JSONResponse({"error": err}, status_code=400)
        used_points = True
    elif bbox and len(bbox) >= 4:
        cnt = _contour_from_bbox(bbox, img_w, img_h)
    else:
        return JSONResponse(
            {"error": "bbox required: [x, y, w, h]  OR  points: [[x,y], ...]"},
            status_code=400,
        )

    # Add to labelled dict
    if not info.get("from_disk") or info.get("labelled"):
        info["labelled"].setdefault(label, []).append(cnt)
    else:
        # from_disk with no contours — init labelled
        info["labelled"] = {label: [cnt]}

    # Rebuild label lines and save
    _rebuild_labels(basename, info)
    # M9a: human create (free-poly / Wall strip / AABB) counts as a session correction
    _mark_corrected(basename)

    # Recompute counts
    labels_out = {k: len(v) for k, v in info["labelled"].items() if v}
    info["_counts"] = labels_out

    bx, by, bw, bh = [int(v) for v in cv2.boundingRect(cnt)]
    poly_out = cnt.reshape(-1, 2).tolist()

    # Pseudo-GUID
    import hashlib
    guid_src = f"{basename}_{label}_{poly_out}"
    guid = hashlib.md5(guid_src.encode()).hexdigest()[:12].upper()

    out = {
        "ok": True,
        "guid": guid,
        "label": label,
        "bbox": [bx, by, bw, bh],
        "labels": labels_out,
        "n_labels": info["n_labels"],
    }
    if used_points:
        out["points"] = poly_out
    return out


@app.get("/api/features")
def get_ui_features():
    """Feature flags for Correct Labels UI (Phase 2 wall strip, etc.)."""
    return {
        "HCI_CORRECT_WALL_STRIP": correct_wall_strip_enabled(default=True),
    }

# ── Per-label bbox details ────────────────────────────────────────────────────
@app.get("/api/label_details/{basename}")
def get_label_details(basename: str):
    """Return per-label bbox and polygon for the correct-tab editor."""
    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)

    img_h, img_w = info["img_h"], info["img_w"]
    details = {}  # cls -> [{idx, bbox:[x,y,w,h], poly:[[x,y],...], area, n_vertices}]

    labelled = info.get("labelled", {})
    if labelled:
        for cls_name, contours in labelled.items():
            if not isinstance(contours, list) or not contours:
                continue
            items = []
            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                area = float(cv2.contourArea(cnt))
                # Full contour vertices (Phase 3 — do not approxPolyDP; preserves N-gons)
                poly = cnt.reshape(-1, 2).tolist()
                items.append({
                    "idx": i + 1,
                    "bbox": [x, y, w, h],
                    "poly": poly,
                    "area": round(area),
                    "n_vertices": len(poly),
                })
            details[cls_name] = items
    else:
        # from_disk: reconstruct from label_lines
        for line in info.get("label_lines", []):
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                cid = int(parts[0])
                cls = ID_TO_CLASS.get(cid, f"cls{cid}")
                coords = list(map(float, parts[1:]))
                pts = [[int(coords[k]*img_w), int(coords[k+1]*img_h)]
                       for k in range(0, len(coords)-1, 2)]
                if len(pts) < 3:
                    continue
                cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                x, y, w, h = cv2.boundingRect(cnt)
                area = float(cv2.contourArea(cnt))
                details.setdefault(cls, []).append({
                    "idx": len(details.get(cls, [])) + 1,
                    "bbox": [x, y, w, h], "poly": pts, "area": round(area),
                    "n_vertices": len(pts),
                })
            except Exception:
                pass

    return {"details": details, "img_w": img_w, "img_h": img_h}


# ── Resize / move a label (bbox legacy OR polygon points — Phase 3) ───────────
@app.post("/api/resize_label")
def resize_label(body: dict):
    """
    Replace a label's contour.

    Accepts:
      - points: [[x,y], ...]  (geometry-preserving; required for Wall unless force_aabb)
      - bbox: [x,y,w,h]       (legacy AABB; Door/Window/Room; Wall only with force_aabb)

    Wall + bbox-only without force_aabb → 400 (never silently destroy polygon).
    """
    basename = body.get("basename")
    cls_name = body.get("cls_name")
    idx      = int(body.get("idx", 1)) - 1   # 0-based
    new_bbox = body.get("bbox")               # [x, y, w, h] optional
    points   = body.get("points")             # [[x,y], ...] optional
    force_aabb = bool(body.get("force_aabb", False))

    info = _analysis.get(basename)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)

    img_h, img_w = info["img_h"], info["img_w"]
    labelled = _ensure_labelled_contours(info, img_w, img_h)
    items = labelled.get(cls_name, [])

    if idx < 0 or idx >= len(items):
        return JSONResponse({"error": f"Index out of range (1-{len(items)})"}, status_code=400)

    used_points = False
    new_cnt = None
    x = y = w = h = 0

    if points is not None:
        new_cnt, err = _contour_from_points(
            points, img_w, img_h, reject_self_intersect=True
        )
        if err:
            return JSONResponse({"error": err}, status_code=400)
        used_points = True
        x, y, w, h = [int(v) for v in cv2.boundingRect(new_cnt)]
    elif new_bbox and len(new_bbox) >= 4:
        # Phase 3: never silently AABB-stomp Wall polygons
        if cls_name == "Wall" and not force_aabb:
            return JSONResponse(
                {
                    "error": (
                        "Wall polygon edit requires points=[[x,y],...]. "
                        "Pass force_aabb=true only for an explicit destructive AABB conversion."
                    )
                },
                status_code=400,
            )
        # M5a: free-polygon classes — same guard (Door/Window/Furniture/ElectricAppliance)
        if cls_name in _FREE_POLY_RESIZE_CLASSES and not force_aabb:
            return JSONResponse(
                {
                    "error": (
                        f"{cls_name} polygon edit requires points=[[x,y],...]. "
                        "Pass force_aabb=true only for an explicit destructive AABB conversion."
                    )
                },
                status_code=400,
            )
        x, y, w, h = [int(v) for v in new_bbox]
        x = max(0, min(x, img_w - 1)); y = max(0, min(y, img_h - 1))
        w = max(4, min(w, img_w - x)); h = max(4, min(h, img_h - y))
        new_cnt = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
    else:
        return JSONResponse(
            {"error": "points [[x,y],...] or bbox [x,y,w,h] required"},
            status_code=400,
        )

    items[idx] = new_cnt
    _rebuild_labels(basename, info)
    # M9a: human geometry edit (free-poly / Wall / AABB) counts as a session correction
    _mark_corrected(basename)
    labels_out = {k: len(v) for k, v in info["labelled"].items() if v}
    info["_counts"] = labels_out

    poly_out = new_cnt.reshape(-1, 2).tolist()
    if used_points:
        msg = f"Updated {cls_name} #{idx+1} polygon ({len(poly_out)} pts)"
    else:
        msg = f"Resized {cls_name} #{idx+1} → [{x},{y},{w},{h}]"
    _push(msg, status=msg)
    out = {
        "ok": True,
        "msg": msg,
        "labels": labels_out,
        "marked_b64": info.get("marked_b64", ""),
        "bbox": [x, y, w, h],
        "n_vertices": len(poly_out),
    }
    if used_points:
        out["points"] = poly_out
    return out

# ── Model version registry ────────────────────────────────────────────────────
_model_versions: list = []   # [{ts, path, epochs, mAP50, mAP50_95, source, n_images}]

def _register_model(path: str, epochs: int, source: str, n_images: int,
                    mAP50: str = "—", mAP50_95: str = "—"):
    import datetime
    _model_versions.append({
        "ts":       datetime.datetime.now().isoformat(timespec="seconds"),
        "path":     path,
        "name":     Path(path).name,
        "epochs":   epochs,
        "mAP50":    mAP50,
        "mAP50_95": mAP50_95,
        "source":   source,
        "n_images": n_images,
    })

@app.get("/api/model_versions")
def get_model_versions():
    """Scan ALL model locations and return with mAP50, epochs, classes."""
    import datetime as _dt
    from logic.yolo_inference import resolve_hci21_model, find_production_model_path

    search_roots = [
        PROJECT_ROOT / "gdrive_dataset" / "runs",
        PROJECT_ROOT / "runs",
        PROJECT_ROOT / "iterations",
        LOGIC_DIR / "gdrive_dataset" / "runs",
        PROJECT_ROOT / "cubicasa_hqa_500" / "runs",
    ]
    hci21_path, hci21_src = resolve_hci21_model()
    active = hci21_path or _find_best_model()
    scanned = []
    seen = set()
    for root in search_roots:
        for pt in sorted(glob.glob(str(root / "**" / "best.pt"), recursive=True),
                         key=os.path.getmtime, reverse=True):
            if pt in seen: continue
            seen.add(pt)
            ts = _dt.datetime.fromtimestamp(os.path.getmtime(pt)).strftime("%Y-%m-%d %H:%M")
            size_mb = round(os.path.getsize(pt) / 1024 / 1024, 1)
            # Try to read training metadata
            map50, map50_95, epochs, nc, source = "—", "—", "?", "?", "scan"
            try:
                from ultralytics import YOLO as _YOLO
                m = _YOLO(pt)
                ckpt = m.ckpt or {}
                tr   = ckpt.get("train_results", {})
                map50_list = tr.get("metrics/mAP50(B)", [])
                if map50_list:
                    map50    = f"{max(map50_list):.3f}"
                    map50_95 = f"{max(tr.get('metrics/mAP50-95(B)', [0])):.3f}"
                epochs = len(tr.get("epoch", []))
                nc     = m.model.nc if hasattr(m.model, "nc") else "?"
                ta     = ckpt.get("train_args", {})
                source = "corrections" if "finetune" in pt else "full_train"
            except Exception:
                pass
            # Short display name
            parts = Path(pt).parts
            name = "/".join(parts[-4:-1]) if len(parts) >= 4 else pt
            already = any(v["path"] == pt for v in _model_versions)
            if not already:
                scanned.append({
                    "ts": ts, "path": pt, "name": name,
                    "epochs": epochs, "mAP50": map50, "mAP50_95": map50_95,
                    "source": source, "n_images": "?", "size_mb": size_mb,
                    "nc": nc, "is_active": (pt == active or active.endswith(Path(pt).name))
                })
    # Registered versions first, then scanned
    all_versions = list(reversed(_model_versions)) + scanned
    # Mark active
    for v in all_versions:
        v["is_active"] = (v["path"] == active)
    return {
        "versions":    all_versions,
        "best_model":  active,
        "best_exists": bool(active),
        "hci21_model": hci21_path,
        "hci21_model_source": hci21_src,
        "production_model": find_production_model_path(),
        "production_protected": True,
    }

@app.post("/api/set_model")
def set_model(body: dict):
    """
    Set HCI_2.1 active model (best_hci21.pt).
    Phase 0 kill-switch: never writes best_gdrive.pt (production).
    """
    path = body.get("path")
    if not path or not os.path.exists(path):
        return JSONResponse({"error": "Model file not found"}, status_code=404)
    # Refuse requests that explicitly ask to overwrite production
    dest_req = body.get("dest") or body.get("destination")
    if dest_req and _is_production_model_path(dest_req):
        return JSONResponse(
            {
                "error": "Promotion to best_gdrive.pt is disabled in HCI_2.1 (production protected).",
                "production_model": str(PRODUCTION_MODEL_PATH),
            },
            status_code=403,
        )
    try:
        promo = _promote_hci21_checkpoint(path, source="set_model")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    _push(
        f"✅ HCI_2.1 active model set to: {Path(path).name} → {HCI21_ACTIVE_MODEL_PATH.name}",
        status="Model updated",
    )
    _push(f"🛡️ Production protected: {PRODUCTION_MODEL_PATH.name} untouched")
    return {
        "ok": True,
        "active": promo["active"],
        "production_protected": True,
        "production_model": str(PRODUCTION_MODEL_PATH),
        "note": promo["note"],
    }

# ── Fine-tune from corrected labels ──────────────────────────────────────────
@app.get("/api/corrected_files")
def get_corrected_files():
    """Return basenames corrected this session + all labelled basenames."""
    lbl_dir  = DATASET_DIR / "labels" / "train"
    all_lbls = sorted([f.stem for f in lbl_dir.glob("*.txt")]) if lbl_dir.exists() else []
    return {
        "corrected": sorted(_corrected_basenames),
        "all":       all_lbls,
    }

@app.post("/api/train_from_corrections")
def train_from_corrections(body: dict, background_tasks: BackgroundTasks):
    """
    Fine-tune FROM a chosen base model using corrected labels.
    mode:        'incremental' = fine-tune from base_model
                 'scratch'     = train from yolov8n-seg.pt
    train_scope: 'corrected'   = only images corrected this session
                 'all'         = all labelled images in dataset
    train_files: explicit list of basenames (overrides train_scope)
    """
    global _training_active
    if _training_active:
        return JSONResponse({"error": "Training already running"}, status_code=409)

    epochs      = int(body.get("epochs", 5))
    batch       = int(body.get("batch", 2))
    imgsz       = int(body.get("imgsz", 640))
    mode        = body.get("mode", "incremental")
    base_model  = body.get("base_model", "")
    train_scope = body.get("train_scope", "all")   # 'corrected' | 'all'
    train_files = body.get("train_files", [])       # explicit list of basenames

    lbl_dir = DATASET_DIR / "labels" / "train"
    if not lbl_dir.exists() or not list(lbl_dir.glob("*.txt")):
        return JSONResponse({"error": "No labels found. Run auto-label first."}, status_code=400)

    # Resolve which files to train on
    if train_files:
        # Explicit list from UI checkboxes
        selected = [b for b in train_files if (lbl_dir / (b + ".txt")).exists()]
    elif train_scope == "corrected":
        selected = [b for b in _corrected_basenames if (lbl_dir / (b + ".txt")).exists()]
        if not selected:
            return JSONResponse({"error": "No corrected images found this session. "
                                          "Make corrections first, or choose 'All Images'."}, status_code=400)
    else:
        selected = []  # empty = all files

    if mode == "scratch":
        base = str(PROJECT_ROOT / "yolov8n-seg.pt")
        if not os.path.exists(base):
            base = "yolov8n-seg.pt"
    elif base_model and os.path.exists(base_model):
        base = base_model
    else:
        # Prefer HCI_2.1 CubiCasa resolver; never require promoting production weights
        from logic.yolo_inference import resolve_hci21_model
        base, _src = resolve_hci21_model()
        if not base:
            return JSONResponse(
                {"error": "No HCI_2.1 CubiCasa model found. Set HCI21_MODEL_PATH."},
                status_code=400,
            )

    background_tasks.add_task(_finetune_worker, epochs, batch, imgsz, base, selected)
    return {"ok": True, "base_model": base, "mode": mode,
            "train_scope": train_scope, "n_files": len(selected) if selected else "all"}


@app.post("/api/merge_models")
def merge_models(body: dict, background_tasks: BackgroundTasks):
    """
    Merge two models using weight averaging (SWA-style).
    model_a, model_b: paths to two best.pt files
    alpha: weight for model_a (0.0-1.0), model_b gets (1-alpha)
    """
    model_a = body.get("model_a", "")
    model_b = body.get("model_b", "")
    alpha   = float(body.get("alpha", 0.5))
    name    = body.get("name", "merged")

    for p, label in [(model_a, "model_a"), (model_b, "model_b")]:
        if not p or not os.path.exists(p):
            return JSONResponse({"error": f"{label} not found: {p}"}, status_code=400)

    background_tasks.add_task(_merge_worker, model_a, model_b, alpha, name)
    return {"ok": True}


def _merge_worker(model_a: str, model_b: str, alpha: float, name: str):
    """Weight-average two YOLO models: merged = alpha*A + (1-alpha)*B"""
    import torch, datetime as _dt
    _push(f"\n{'='*50}")
    _push(f"Merging models (alpha={alpha})")
    _push(f"  A: {Path(model_a).name}")
    _push(f"  B: {Path(model_b).name}")
    try:
        ckpt_a = torch.load(model_a, map_location="cpu")
        ckpt_b = torch.load(model_b, map_location="cpu")

        sd_a = ckpt_a["model"].state_dict() if hasattr(ckpt_a.get("model",""), "state_dict") else ckpt_a["model"]
        sd_b = ckpt_b["model"].state_dict() if hasattr(ckpt_b.get("model",""), "state_dict") else ckpt_b["model"]

        # Weight average
        merged_sd = {}
        for key in sd_a:
            if key in sd_b:
                ta = sd_a[key].float()
                tb = sd_b[key].float()
                if ta.shape == tb.shape:
                    merged_sd[key] = (alpha * ta + (1 - alpha) * tb).to(sd_a[key].dtype)
                else:
                    merged_sd[key] = ta  # shape mismatch — keep A
                    _push(f"  ⚠️ Shape mismatch for {key}, keeping model A")
            else:
                merged_sd[key] = sd_a[key]

        # Save merged model
        merged_ckpt = dict(ckpt_a)
        if hasattr(ckpt_a.get("model",""), "load_state_dict"):
            ckpt_a["model"].load_state_dict(merged_sd)
            merged_ckpt["model"] = ckpt_a["model"]
        else:
            merged_ckpt["model"] = merged_sd

        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = DATASET_DIR / "runs" / f"merged_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "best.pt")
        torch.save(merged_ckpt, out_path)

        # Phase 0: activate in HCI_2.1 only — never best_gdrive.pt
        promo = _promote_hci21_checkpoint(out_path, source="merge_models")

        _register_model(path=out_path, epochs=0, source="merged",
                        n_images="A+B", mAP50="—", mAP50_95="—")
        _push(f"\n✅ Merged model saved: {out_path}", pct=100, status="Merge complete!")
        _push(f"   HCI_2.1 active model: {promo['active']}")
        _push(f"   🛡️ Production protected: {PRODUCTION_MODEL_PATH} untouched")
    except Exception as e:
        _push(f"ERROR merging: {e}\n{traceback.format_exc()}", status="Merge failed")

def _resolve_finetune_image(stem: str, img_dir: Path, raw_dir: Path) -> Path | None:
    """M9b: locate image for a label stem under images/train, else images_raw."""
    for ext in IMG_EXTS:
        cand = img_dir / (stem + ext)
        if cand.is_file():
            return cand
    for ext in IMG_EXTS:
        cand = raw_dir / (stem + ext)
        if cand.is_file():
            return cand
    return None


def _finetune_worker(epochs: int, batch: int, imgsz: int, base_model: str,
                     train_files: list = None):
    """train_files: list of basenames to train on. Empty/None = all files."""
    global _training_active
    _training_active = True
    yaml_path = DATASET_DIR / "dataset.yaml"
    if not yaml_path.exists():
        _push("ERROR: No dataset.yaml", status="Error")
        _training_active = False; return

    try:
        from ultralytics import YOLO
        import torch

        _push(f"\n{'='*50}")
        _push(f"Fine-tuning FROM: {base_model}")
        _push(f"Epochs: {epochs}  Batch: {batch}  ImgSz: {imgsz}")
        _push(f"Strategy: freeze backbone (10 layers), SGD lr=0.0005")
        _push(f"{'='*50}")

        model = YOLO(base_model)
        size_mb = os.path.getsize(base_model) / 1024 / 1024
        _push(f"  Loaded: {Path(base_model).name}  ({size_mb:.1f}MB)")

        # Validate base model
        try:
            ckpt = model.ckpt or {}
            tr   = ckpt.get("train_results", {})
            epochs_trained = len(tr.get("epoch", []))
            map50_list = tr.get("metrics/mAP50(B)", [])
            best_map50 = max(map50_list) if map50_list else 0
            _push(f"  Base model stats: epochs_trained={epochs_trained}, best_mAP50={best_map50:.4f}")
            if epochs_trained == 0:
                _push(f"  ⚠️ WARNING: Base model has no training history — using pretrained weights only")
            elif best_map50 < 0.01:
                _push(f"  ⚠️ WARNING: mAP50={best_map50:.4f} is very low — need more images (100+ per class)")
        except Exception as ve:
            _push(f"  Could not read model stats: {ve}")

        # ── Resolve which label files to train on ──────────────────────────────────────────────────────────────
        lbl_dir      = DATASET_DIR / "labels" / "train"
        img_dir      = DATASET_DIR / "images" / "train"
        raw_dir      = DATASET_DIR / "images_raw"
        all_lbl      = sorted(lbl_dir.glob("*.txt"))

        if train_files:
            # Filter to only the selected basenames
            sel_set   = set(train_files)
            use_files = [lf for lf in all_lbl if lf.stem in sel_set]
            scope_tag = f"corrected ({len(use_files)} image(s))"
        else:
            use_files = all_lbl
            scope_tag = f"all ({len(use_files)} image(s))"

        n_images = len(use_files)
        _push(f"  Training scope: {scope_tag}")
        for lf in use_files:
            n = len([l for l in lf.read_text().splitlines() if l.strip()])
            _push(f"    [{lf.stem}]  labels: {n}")

        if n_images == 0:
            _push("ERROR: No matching label files found", status="Error")
            _training_active = False; return

        # If training on a subset, create a temporary dataset.yaml pointing to a temp folder
        import tempfile, datetime as _dt
        if train_files:
            tmp_dir  = Path(tempfile.mkdtemp(prefix="ft_subset_"))
            tmp_imgs = tmp_dir / "images" / "train"
            tmp_lbls = tmp_dir / "labels" / "train"
            tmp_imgs.mkdir(parents=True); tmp_lbls.mkdir(parents=True)
            for lf in use_files:
                src_img = _resolve_finetune_image(lf.stem, img_dir, raw_dir)
                if src_img is None:
                    _push(
                        f"ERROR: No image for corrected label '{lf.stem}' "
                        f"under images/train or images_raw",
                        status="Error",
                    )
                    shutil.rmtree(str(tmp_dir), ignore_errors=True)
                    _training_active = False
                    return
                shutil.copy2(str(lf), str(tmp_lbls / lf.name))
                shutil.copy2(str(src_img), str(tmp_imgs / src_img.name))
            # Write temp yaml
            tmp_yaml = tmp_dir / "dataset.yaml"
            orig_yaml = yaml_path.read_text()
            # Replace path line
            new_yaml = "\n".join(
                (f"path: {tmp_dir}" if l.startswith("path:") else l)
                for l in orig_yaml.splitlines()
            )
            tmp_yaml.write_text(new_yaml)
            active_yaml = str(tmp_yaml)
            _push(f"  Temp dataset created at: {tmp_dir}")
        else:
            active_yaml = str(yaml_path)
            tmp_dir = None

        metrics_final = {}

        def on_epoch_end(trainer):
            ep = trainer.epoch + 1
            _push(f"  Epoch {ep}/{epochs}", pct=ep/epochs*100,
                  status=f"Fine-tune epoch {ep}/{epochs}",
                  metrics={"Epoch": f"{ep}/{epochs}"})
            loss = trainer.label_loss_items(trainer.tloss)
            if loss:
                m = {}
                if "train/box_loss" in loss: m["Box Loss"] = f"{loss['train/box_loss']:.4f}"
                if "train/seg_loss" in loss: m["Seg Loss"] = f"{loss['train/seg_loss']:.4f}"
                if m: _progress["metrics"].update(m)

        def on_fit_end(trainer):
            m = trainer.metrics
            rd = m.results_dict if hasattr(m, "results_dict") else (m if isinstance(m, dict) else {})
            if "metrics/mAP50(B)" in rd:
                metrics_final["mAP50"]    = f"{rd['metrics/mAP50(B)']:.4f}"
                metrics_final["mAP50_95"] = f"{rd.get('metrics/mAP50-95(B)', 0):.4f}"
                _progress["metrics"].update({"mAP50": metrics_final["mAP50"]})

        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_end)

        device = "cuda" if torch.cuda.is_available() else \
                 ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        _push(f"  Device: {device}")

        run_name    = f"finetune_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = str(DATASET_DIR / "runs")

        model.train(
            data=active_yaml, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, project=project_dir, name=run_name,
            workers=0, amp=True, verbose=False,
            optimizer="SGD", lr0=0.0005, lrf=0.01,
            momentum=0.937, weight_decay=0.0005,
            warmup_epochs=1, freeze=10, close_mosaic=0,
        )

        # Clean up temp dir
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        cands = glob.glob(os.path.join(project_dir, run_name, "weights", "best.pt"))
        if not cands:
            cands = glob.glob(os.path.join(project_dir, "**", "best.pt"), recursive=True)
        if cands:
            best_new = max(cands, key=os.path.getmtime)
            # Phase 0: never write best_gdrive.pt — promote to best_hci21.pt only
            promo = _promote_hci21_checkpoint(best_new, source="train_from_corrections")
            _register_model(
                path=best_new, epochs=epochs, source="corrections",
                n_images=n_images,
                mAP50=metrics_final.get("mAP50", "—"),
                mAP50_95=metrics_final.get("mAP50_95", "—"),
            )
            _push(
                f"\n✅ Fine-tune complete! HCI_2.1 model: {promo['active']}",
                pct=100,
                status="Fine-tune complete!",
            )
            _push(f"   🛡️ Production protected: {PRODUCTION_MODEL_PATH} untouched")
            _push(f"   mAP50: {metrics_final.get('mAP50','—')}  "
                  f"mAP50-95: {metrics_final.get('mAP50_95','—')}")
        else:
            _push("⚠️ No best.pt produced", status="Done (no model)")
    except Exception as e:
        _push(f"ERROR: {e}\n{traceback.format_exc()}", status="Fine-tune failed")
    finally:
        _training_active = False

# ── Image metadata (JSON cache) ───────────────────────────────────────────────
from logic.image_metadata import (
    get_metadata_path, metadata_exists, load_metadata,
    save_metadata, build_metadata_from_ocr, list_all_metadata
)

@app.get("/api/metadata/check")
def check_metadata(basename: str):
    """Check if metadata JSON exists for an image. Returns exists + summary."""
    img_path = DATASET_DIR / "images_raw" / basename
    # Try with common extensions
    if not img_path.exists():
        for ext in IMG_EXTS:
            candidate = DATASET_DIR / "images_raw" / (basename + ext)
            if candidate.exists():
                img_path = candidate
                break
    exists = metadata_exists(str(img_path), DATASET_DIR)
    if exists:
        data = load_metadata(str(img_path), DATASET_DIR)
        return {
            "exists":      True,
            "source":      data.get("source", "unknown"),
            "saved_at":    data.get("_saved_at", ""),
            "n_labels":    data.get("n_labels", 0),
            "n_rooms":     len(data.get("rooms", [])),
            "label_counts": data.get("label_counts", {}),
            "ifc_classes": data.get("ifc_classes", {}),
            "notes":       data.get("notes", ""),
        }
    return {"exists": False}

@app.get("/api/metadata/{basename}")
def get_metadata(basename: str):
    """Get full metadata JSON for an image."""
    img_path = DATASET_DIR / "images_raw" / basename
    if not img_path.exists():
        for ext in IMG_EXTS:
            c = DATASET_DIR / "images_raw" / (basename + ext)
            if c.exists(): img_path = c; break
    data = load_metadata(str(img_path), DATASET_DIR)
    if not data:
        return JSONResponse({"error": "No metadata found"}, status_code=404)
    return data

@app.get("/api/metadata")
def list_metadata():
    """List all metadata files with summaries."""
    return {"metadata": list_all_metadata(DATASET_DIR)}

@app.post("/api/metadata/save_gemini")
async def save_gemini_metadata(body: dict):
    """Save Gemini AI analysis result as metadata for an image."""
    basename = body.get("basename")
    gemini   = body.get("gemini", {})
    if not basename:
        return JSONResponse({"error": "basename required"}, status_code=400)
    img_path = DATASET_DIR / "images_raw" / basename
    if not img_path.exists():
        for ext in IMG_EXTS:
            c = DATASET_DIR / "images_raw" / (basename + ext)
            if c.exists(): img_path = c; break
    from logic.image_metadata import build_metadata_from_gemini
    data = build_metadata_from_gemini(str(img_path), gemini)
    path = save_metadata(str(img_path), DATASET_DIR, data)
    return {"ok": True, "path": str(path)}

@app.post("/api/metadata/delete")
def delete_metadata(body: dict):
    """Delete metadata JSON for an image (force re-analyse)."""
    basename = body.get("basename")
    img_path = DATASET_DIR / "images_raw" / basename
    meta_path = get_metadata_path(str(img_path), DATASET_DIR)
    if meta_path.exists():
        meta_path.unlink()
        return {"ok": True, "deleted": str(meta_path)}
    return {"ok": False, "error": "No metadata file found"}

# ── IFC Properties ────────────────────────────────────────────────────────────
from logic.ifc_properties import IFC_SCHEMA, MATERIALS, get_default_pset, validate_pset

# In-memory store: basename -> {cls_idx_key -> {pset_name -> {prop -> value}}}
_ifc_props: dict = {}

@app.get("/api/ifc/schema")
def get_ifc_schema():
    """Return full IFC schema for all classes."""
    return {"schema": IFC_SCHEMA, "materials": MATERIALS}

@app.get("/api/ifc/schema/{cls_name}")
def get_ifc_schema_for_class(cls_name: str):
    """Return IFC schema for a specific class."""
    from logic.ifc_properties import get_schema
    schema = get_schema(cls_name)
    if not schema:
        return JSONResponse({"error": f"No schema for {cls_name}"}, status_code=404)
    return {"cls_name": cls_name, "schema": schema,
            "defaults": get_default_pset(cls_name)}

@app.get("/api/ifc/props/{basename}")
def get_ifc_props(basename: str):
    """Get all IFC properties for a labelled image."""
    props = _ifc_props.get(basename, {})
    # Also load from disk if exists
    props_path = DATASET_DIR / "metadata" / (basename + "_ifc_props.json")
    if not props and props_path.exists():
        try:
            props = json.loads(props_path.read_text())
            _ifc_props[basename] = props
        except Exception:
            pass
    return {"basename": basename, "props": props}

@app.post("/api/ifc/props/{basename}")
def save_ifc_props(basename: str, body: dict):
    """Save IFC properties for a specific label (cls + idx)."""
    cls_name  = body.get("cls_name")
    idx       = int(body.get("idx", 1))
    subtype   = body.get("subtype", "")
    pset_data = body.get("psets", {})
    material  = body.get("material", "")
    color     = body.get("color", "")
    dims      = body.get("dimensions", {})

    if not cls_name:
        return JSONResponse({"error": "cls_name required"}, status_code=400)

    # Validate psets against schema
    cleaned = validate_pset(cls_name, pset_data)

    key = f"{cls_name}_{idx}"
    if basename not in _ifc_props:
        _ifc_props[basename] = {}

    _ifc_props[basename][key] = {
        "cls_name":   cls_name,
        "idx":        idx,
        "ifc_class":  IFC_SCHEMA.get(cls_name, {}).get("ifc_class", "IfcBuildingElement"),
        "subtype":    subtype,
        "psets":      cleaned,
        "material":   material,
        "color":      color,
        "dimensions": dims,
        "updated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }

    # Persist to disk
    props_path = DATASET_DIR / "metadata" / (basename + "_ifc_props.json")
    props_path.parent.mkdir(parents=True, exist_ok=True)
    props_path.write_text(json.dumps(_ifc_props[basename], indent=2))

    return {"ok": True, "key": key, "ifc_class": _ifc_props[basename][key]["ifc_class"]}

@app.delete("/api/ifc/props/{basename}/{key}")
def delete_ifc_prop(basename: str, key: str):
    """Delete IFC properties for a specific label."""
    if basename in _ifc_props and key in _ifc_props[basename]:
        del _ifc_props[basename][key]
        props_path = DATASET_DIR / "metadata" / (basename + "_ifc_props.json")
        if props_path.exists():
            props_path.write_text(json.dumps(_ifc_props.get(basename, {}), indent=2))
        return {"ok": True}
    return JSONResponse({"error": "Not found"}, status_code=404)

@app.get("/api/ifc/export/{basename}")
def export_ifc_props(basename: str):
    """Export IFC properties as structured JSON for IFC builder."""
    props = _ifc_props.get(basename, {})
    if not props:
        props_path = DATASET_DIR / "metadata" / (basename + "_ifc_props.json")
        if props_path.exists():
            props = json.loads(props_path.read_text())

    export = {
        "basename": basename,
        "elements": [],
        "summary": {}
    }
    for key, data in props.items():
        export["elements"].append(data)
        cls = data.get("cls_name", "Unknown")
        export["summary"][cls] = export["summary"].get(cls, 0) + 1

    return export


# ── HCI_2.1 IFC geometry export (Day 3/4 library) ─────────────────────────────
# Reads corrected YOLO labels; never overwrites best_gdrive.pt; never changes Hci_1.
# Phase 1: IFC entry / path helpers live in logic.ifc_service + logic.dataset_io
# (API contracts unchanged; Training UI behaviour unchanged).
from logic.scale_calibration import save_scale, load_scale, DEFAULT_MPP
from logic.ifc_service import (
    IfcImageMissingError,
    IfcLabelMissingError,
    generate_ifc_for_basename,
)

IFC_OUTPUT_DIR = DATASET_DIR / "output"


def _resolve_image_for_basename(basename: str) -> Path | None:
    """Prefer train image, then raw image. Thin wrapper over dataset_io helper."""
    return resolve_image_for_basename(DATASET_DIR, basename)


@app.post("/api/scale/calibrate")
def calibrate_scale(body: dict):
    """
    Two-point scale calibration.
    Body: { basename, p1: [x,y], p2: [x,y], distance_m, image_wh?: [W,H] }
    Writes gdrive_dataset/metadata/{basename}_scale.json
    """
    basename = (body or {}).get("basename")
    p1 = (body or {}).get("p1")
    p2 = (body or {}).get("p2")
    distance_m = (body or {}).get("distance_m")
    image_wh = (body or {}).get("image_wh")
    if not basename or not p1 or not p2 or distance_m is None:
        return JSONResponse(
            {"error": "basename, p1, p2, and distance_m are required"}, status_code=400
        )
    try:
        data = save_scale(
            DATASET_DIR,
            basename,
            p1=list(p1),
            p2=list(p2),
            distance_m=float(distance_m),
            image_wh=image_wh,
            source="manual_ui",
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    _push(
        f"✅ Scale calibrated for {basename}: mpp={data['meters_per_pixel']:.6f}",
        status="Scale calibrated",
    )
    return {"ok": True, "scale": data}


@app.get("/api/scale/{basename}")
def get_scale(basename: str):
    data = load_scale(DATASET_DIR, basename)
    if not data:
        return {"ok": True, "exists": False, "meters_per_pixel": DEFAULT_MPP, "source": "default_fallback"}
    return {"ok": True, "exists": True, "scale": data}


@app.post("/api/ifc/generate/{basename}")
def generate_ifc_geometry(basename: str, body: dict = None):
    """
    Generate IFC4 walls+openings+doors+windows from saved YOLO labels.
    Body optional: { meters_per_pixel?: float } — otherwise uses saved scale or 0.01 fallback.
    Writes to gdrive_dataset/output/{basename}.ifc
    Does NOT modify best_gdrive.pt or training pipeline.
    """
    body = body or {}
    mpp_arg = body.get("meters_per_pixel")
    try:
        mpp_kw = float(mpp_arg) if mpp_arg is not None else None
        result = generate_ifc_for_basename(
            DATASET_DIR,
            basename,
            meters_per_pixel=mpp_kw,
        )
    except IfcLabelMissingError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except IfcImageMissingError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except Exception as exc:
        _push(f"IFC generate failed for {basename}: {exc}", status="IFC failed")
        return JSONResponse({"error": str(exc), "trace": traceback.format_exc()}, status_code=500)

    _push(
        f"✅ IFC generated: {basename}.ifc  walls={result.get('ifc_walls')} "
        f"doors={result.get('ifc_doors')} windows={result.get('ifc_windows')}",
        status="IFC ready",
    )
    return result


@app.get("/api/ifc/file/{basename}")
def download_ifc_file(basename: str):
    """Download generated IFC from gdrive_dataset/output/{basename}.ifc"""
    path = ifc_file_path(DATASET_DIR, basename)
    if not path.exists():
        return JSONResponse(
            {"error": "IFC not found. Generate it first via POST /api/ifc/generate/{basename}."},
            status_code=404,
        )
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=f"{basename}.ifc",
    )


# ── Phase 6: production API (health + hardened process-floorplan) ─────────────
# Additive endpoints. Does not alter Manual Training routes or workers.
import logging

from logic.production_orchestrator import (
    ProductionError,
    ProductionOrchestrator,
    build_production_error_body,
    get_production_health,
)

# Ensure production structured logs are visible when the root logger is quiet.
logging.getLogger("hci21.production").setLevel(logging.INFO)
if not logging.getLogger("hci21.production").handlers:
    _prod_handler = logging.StreamHandler()
    _prod_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [hci21.production] %(message)s")
    )
    logging.getLogger("hci21.production").addHandler(_prod_handler)
    logging.getLogger("hci21.production").propagate = False


@app.get("/api/production/health")
def production_health():
    """
    Pure production health check.
    Reuses model resolution only — no YOLO load, inference, or IFC.
    """
    return get_production_health(DATASET_DIR)


@app.post("/api/process-floorplan")
def process_floorplan_endpoint(
    file: UploadFile = File(...),
    meters_per_pixel: str | None = Form(None),
):
    """
    Thin production API: receive multipart → orchestrator.process → envelope.
    Validation, unique naming, timing, and pipeline live in ProductionOrchestrator.
    """
    data = file.file.read()
    orchestrator = ProductionOrchestrator(
        DATASET_DIR,
        run_autolabel=_autolabel_worker,
    )
    try:
        return orchestrator.process(
            file.filename,
            data,
            meters_per_pixel=meters_per_pixel,
        )
    except ProductionError as exc:
        body = build_production_error_body(
            exc.message,
            status_code=exc.status_code,
            error_code=exc.error_code,
            basename=getattr(exc, "basename", None),
            processing_time_ms=getattr(exc, "processing_time_ms", None),
            request_id=getattr(exc, "request_id", None),
            timing=getattr(exc, "timing", None),
            warnings=getattr(exc, "warnings", None),
            exc=exc,
        )
        return JSONResponse(body, status_code=exc.status_code)
    except Exception as exc:
        body = build_production_error_body(
            "Unexpected internal failure",
            status_code=500,
            error_code="internal_error",
            request_id=None,
            exc=exc,
        )
        return JSONResponse(body, status_code=500)

