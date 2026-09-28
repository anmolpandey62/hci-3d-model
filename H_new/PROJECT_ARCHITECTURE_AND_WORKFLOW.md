# HCI Project Documentation (File-by-File + Full Workflow)

This document explains the current project under `D:\HCI_interor\Hci_1`:

- what each core file does,
- how modules connect,
- runtime behavior,
- and complete workflow flowcharts.

---

## 1) Scope: Which files are the active app

This repository currently contains multiple nested copies/archives (`hci-3d-model`, repeated `Hci_1`, legacy bundles, etc.).  
The **active runtime app** (the one you start for local web usage) is:

- `D:\HCI_interor\Hci_1\web\`
- `D:\HCI_interor\Hci_1\logic\`
- `D:\HCI_interor\Hci_1\config\`
- `D:\HCI_interor\Hci_1\auto_label.py`
- `D:\HCI_interor\Hci_1\scripts\`
- `D:\HCI_interor\Hci_1\requirements.txt`

Everything else is supplementary, duplicated, archived, or separate-model project material.

---

## 2) Core runtime architecture

At runtime, `web/server.py` is the orchestrator:

- exposes all API routes,
- coordinates upload/autolabel/correct/train/detect/metadata/IFC,
- stores in-memory UI state (`_analysis`, `_progress`, logs),
- writes generated artifacts to dataset folders.

`auto_label.py` + `logic/yolo_inference.py` provide inference and polygon generation.

`logic/*` modules provide specialized services (metadata, IFC schema, stubs for OCR/analysis/detection).

---

## 3) File-by-file documentation (core app)

## `auto_label.py`
- **Purpose:** Entry point for Auto Label inference from server worker.
- **Key functions:**
  - `generate_labels(img_path, detector=None)`
    - reads image with OpenCV,
    - resolves model path (`HCI_MODEL_PATH` override or automatic lookup),
    - runs YOLO inference via `logic/yolo_inference.py`,
    - returns `(label_lines, img, labelled_dict)`,
    - adds `_skip_reason` on failures/zero detections.
  - `draw_labelled_image(img, labelled, marked_path)`
    - draws class-colored overlays and saves preview image.
- **Behavior notes:**
  - `detector` parameter is kept for compatibility but not used for inference.
  - Raises `ModelNotFoundError` if no usable model exists.

## `config/classes.py`
- **Purpose:** Canonical class taxonomy for HCI.
- **Exports:**
  - `CLASS_NAMES` (17 classes),
  - `CLASS_IDS` (name -> id),
  - `ID_TO_CLASS` (id -> name).
- **Used by:** server, inference, label generation/reconstruction.

## `logic/yolo_inference.py`
- **Purpose:** Shared YOLO segmentation inference engine and overlay rendering.
- **Key functions/classes:**
  - `ModelNotFoundError`
  - `find_model_path()`
  - `run_yolo_inference(...)`
  - `contour_to_yolo_seg(...)`
  - `draw_detection_overlay(...)`
  - helper mapping/filter functions.
- **Behavior:**
  - Searches model in prioritized order (active `best_gdrive.pt`, known improved model, then run scans).
  - Loads model once with in-process cache (`_model_cache`).
  - Performs segmentation, extracts contours, filters/simplifies polygons.
  - Maps model classes to HCI classes (`Wall`, `Door`, `Window`).
  - Uses fallback confidence (`retry_conf=0.001`) if regular confidence gives no detections.

## `logic/detector.py`
- **Purpose:** Legacy detector compatibility placeholder.
- **Current behavior:** Returns empty structures (stub).
- **Why still needed:** server wiring still instantiates `FloorPlanDetector` in autolabel path.

## `logic/floor_plan_analyzer.py`
- **Purpose:** Post-processing analyzer hook (currently minimal).
- **Current behavior:** Pass-through + metadata flags (`_room_names`, `_ocr_seeds`, `_analyzer_used=False`).
- **Connection:** called inside `_autolabel_worker` for enrichment step.

## `logic/room_text_mapper.py`
- **Purpose:** OCR/text-to-room mapping hook (currently stub).
- **Current behavior:** returns empty mappings and image passthrough pre/post overlays.
- **Connection:** used by autolabel pipeline and `/api/analyse`.

## `logic/image_metadata.py`
- **Purpose:** Metadata file pathing, load/save, list.
- **Key functions:**
  - `metadata_exists`, `load_metadata`, `save_metadata`,
  - `build_metadata_from_ocr` (stub payload),
  - `build_metadata_from_gemini` (stub payload),
  - `list_all_metadata`.
- **Connection:** metadata APIs in server + autolabel metadata save step.

## `logic/ifc_properties.py`
- **Purpose:** IFC schema + material catalog + validators/defaults.
- **Contains:**
  - `IFC_SCHEMA` (large structured schema by class),
  - `MATERIALS`,
  - `get_schema`, `get_default_pset`, `validate_pset`.
- **Connection:** `/api/ifc/*` routes for per-image IFC property editing/export.

## `web/server.py`
- **Purpose:** Main FastAPI backend and full workflow coordinator.
- **Major route groups:**
  - **status/logging:** `/api/status`, `/api/stream`
  - **data ingest:** `/api/download`, `/api/upload`
  - **autolabel:** `/api/autolabel` (+ worker)
  - **image retrieval:** `/api/raw`, `/api/raw_thumb`, `/api/thumb`, `/api/image`
  - **correction:** `/api/correct`, `/api/save_corrections`, `/api/revert`, `/api/section`, `/api/resize_label`, `/api/label_details`
  - **detect/analyse:** `/api/detect`, `/api/analyse`
  - **training/model mgmt:** `/api/train`, `/api/train_from_corrections`, `/api/model_versions`, `/api/set_model`, `/api/merge_models`, `/api/corrected_files`
  - **metadata:** `/api/metadata/*`
  - **ifc:** `/api/ifc/*`
  - **frontend index:** `/`
- **Core in-memory states:**
  - `_analysis`, `_progress`, `_log_queue`, `_training_active`, `_corrected_basenames`.
- **Worker functions:**
  - `_autolabel_worker`, `_train_worker`, `_finetune_worker`, `_merge_worker`, `_load_existing_labels`.
- **Startup behavior:**
  - loads existing marked/label files into `_analysis` so Correct Labels can work after restart.

## `web/index.html`
- **Purpose:** Single-page UI client for all app features.
- **Behavior:**
  - Calls backend via same-origin API (`const API = ''`),
  - Handles tabs: upload, autolabel, corrections, detect/test, training, metadata/IFC interactions,
  - Streams progress/logs via SSE endpoint (`/api/stream`).

## `web/ifc_properties.py`
- **Purpose:** Web-side helper schema data mirror/adapter for IFC routes.

## `scripts/start_server.bat`
- **Purpose:** quick launcher using a predefined conda python path.
- **Behavior:** changes to `web` directory and runs `python server.py`.

## `scripts/verify_autolabel_integration.py`
- **Purpose:** one-image offline smoke test for model load + autolabel outputs.

## `scripts/verify_full_pipeline.py`
- **Purpose:** end-to-end functional check of autolabel + analysis cache + corrections save.

## `requirements.txt`
- **Purpose:** minimal external dependency list for the active HCI app runtime.

---

## 4) Runtime data folders and what writes to them

Assuming project root is `D:\HCI_interor\Hci_1` and data root resolves as parent + `gdrive_dataset` in your setup:

- `gdrive_dataset/images_raw`
  - written by upload/download APIs.
- `gdrive_dataset/images/train`
  - copied during autolabel success.
- `gdrive_dataset/labels/train`
  - YOLO segmentation txt labels (autolabel, correction save, section/resize operations).
- `gdrive_dataset/marked`
  - labelled preview + pre/post mapping images.
- `gdrive_dataset/metadata`
  - metadata json files.
- `gdrive_dataset/runs`
  - training/fine-tuning runs and weights.
- `best_gdrive.pt`
  - active model (manual placement or copied from training completion).

---

## 5) End-to-end workflow

1. Start server.
2. Upload images to `images_raw`.
3. Run Auto Label.
4. Model loads and performs segmentation.
5. Polygons are converted to YOLO label lines.
6. Artifacts are written (`images/train`, `labels/train`, `marked`, metadata).
7. `_analysis` cache is populated/reloaded.
8. Correct Labels tab reads polygons and allows edits.
9. Save Corrections persists updated label files.
10. Optional: run training/fine-tuning and update active model.

---

## 6) Flowchart (Mermaid)

```mermaid
flowchart TD
    A[Web UI: index.html] --> B[FastAPI: web/server.py]
    B --> C[/api/upload or /api/download/]
    C --> D[gdrive_dataset/images_raw]

    B --> E[/api/autolabel -> _autolabel_worker]
    E --> F[auto_label.generate_labels]
    F --> G[logic.yolo_inference.find_model_path]
    G --> H[Load YOLO model]
    H --> I[run_yolo_inference]
    I --> J[Masks -> Contours -> Polygon filtering]
    J --> K[YOLO seg label lines]
    K --> L[gdrive_dataset/labels/train/*.txt]
    F --> M[draw_labelled_image]
    M --> N[gdrive_dataset/marked/*_labelled.jpg]
    E --> O[gdrive_dataset/images/train/*]
    E --> P[metadata save]
    E --> Q[_analysis cache update]

    Q --> R[Correct Labels APIs]
    R --> S[/api/correct /api/resize_label /api/section]
    S --> T[/api/save_corrections]
    T --> L

    B --> U[/api/train or /api/train_from_corrections]
    U --> V[_train_worker/_finetune_worker]
    V --> W[gdrive_dataset/runs/*/weights/best.pt]
    W --> X[copy to best_gdrive.pt]
    X --> G
```

---

## 7) Flowchart (ASCII)

```text
index.html (UI)
   |
   v
web/server.py (FastAPI)
   |
   +--> Upload/Download --> images_raw/
   |
   +--> AutoLabel worker
          |
          v
      auto_label.py
          |
          v
      logic/yolo_inference.py
          |
          +--> load model (best_gdrive.pt)
          +--> predict masks
          +--> contours -> polygons
          +--> yolo label lines
          |
          +--> labels/train/*.txt
          +--> marked/*_labelled.jpg
          +--> images/train/*
          +--> metadata/*.json
          +--> _analysis cache
                    |
                    v
               Correct Labels UI
                    |
                    +--> edits (correct/resize/section)
                    +--> save_corrections -> labels/train/*.txt

   +--> Train/Fine-tune
          |
          v
      runs/*/weights/best.pt
          |
          v
      best_gdrive.pt (active model)
```

---

## 8) Connections map (quick reference)

- `web/server.py` -> `auto_label.py`
- `web/server.py` -> `logic/yolo_inference.py`
- `web/server.py` -> `logic/detector.py` (stub compatibility)
- `web/server.py` -> `logic/floor_plan_analyzer.py` (stub enrichment)
- `web/server.py` -> `logic/room_text_mapper.py` (stub OCR mapping)
- `web/server.py` -> `logic/image_metadata.py`
- `web/server.py` -> `logic/ifc_properties.py`
- `auto_label.py` -> `logic/yolo_inference.py`
- `logic/yolo_inference.py` -> `config/classes.py`
- `web/index.html` -> all `/api/*` routes in `web/server.py`

---

## 9) Current behavior maturity

- **Fully functional:** upload, autolabel pipeline wiring, correction persistence, train endpoint wiring.
- **Partially functional / basic:** metadata (stub builders), floor plan analyzer (stub), room text mapping (stub), detector (stub).
- **Operational dependency:** model file availability (`best_gdrive.pt`) and valid Python env.

---

## 10) Recommended next cleanup (optional)

The repository contains deeply nested duplicate copies (`hci-3d-model/Hci_1/hci-3d-model/...`).  
For maintainability, keep one canonical app tree and archive/remove duplicates.

