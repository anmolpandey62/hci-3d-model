# HCI_2.1 Web UI — Official Run & Deployment Guide

Complete deployment and operations guide for the **HCI_2.1** Floor Plan Model Trainer Web UI.

This document assumes a **brand-new machine** with no prior HCI setup.

> **Related:** `DEPLOY.md` (layout & checklist), `requirements.txt` (pip deps), `.gitignore` (what not to commit).

---

# 1. Project Overview

**HCI_2.1** is a FastAPI-based Web application for floor-plan machine learning and BIM export. Operators upload plan images, auto-label walls/doors/windows with YOLO segmentation, correct labels in the browser, train or fine-tune models, calibrate real-world scale, attach metadata / IFC properties, and generate IFC4 geometry for tools such as BIMVision.

## Capabilities

| Feature | What it does |
|---------|----------------|
| **Web UI** | Single-page app (`web/index.html`) served by FastAPI at port **8000** |
| **Auto Label** | Runs YOLO-seg on `images_raw`, writes YOLO labels + marked overlays |
| **Correct Labels** | Interactive edit of polygons; save / revert; feeds fine-tune |
| **Training** | Train from scratch, fine-tune from corrections, merge checkpoints |
| **IFC Generation** | Builds IFC4 walls, doors, windows from corrected labels + scale |
| **Scale Calibration** | Two-point pixel→meter calibration (`*_scale.json`) |
| **Metadata** | Per-image JSON cache (OCR / Gemini stubs + IFC property sets) |
| **Room Analysis** | Floor-plan / room-text analysis endpoints (overlay helpers in UI) |

## End-to-end workflow (image → IFC)

```text
Upload floor plan
       ↓
  images_raw/
       ↓
   Auto Label  →  images/train + labels/train + marked/
       ↓
 Correct Labels  →  refined .txt labels
       ↓
 Scale Calibration  →  metadata/{name}_scale.json
       ↓
 Metadata / IFC Properties  →  metadata/*.json
       ↓
 Generate IFC  →  output/{name}.ifc
       ↓
 Download IFC  →  BIM viewer
```

---

# 2. Folder Structure

## Expected production layout

`PROJECT_ROOT` **must** be the **parent directory of `HCI_2.1`**. Paths for datasets and default models are resolved from that parent.

```text
PROJECT_ROOT/
├── HCI_2.1/                          # Application (from Git)
│   ├── web/                          # FastAPI server + UI
│   ├── logic/                        # YOLO, IFC, scale, metadata
│   ├── config/                       # Class IDs
│   ├── scripts/                      # Launch / verify helpers
│   ├── auto_label.py
│   ├── requirements.txt
│   ├── DEPLOY.md
│   ├── RUN_GUIDE.md                  # This file
│   └── .gitignore
├── gdrive_dataset/                   # Runtime data (auto-created)
│   ├── images_raw/
│   ├── images/train/
│   ├── labels/train/
│   ├── marked/
│   ├── metadata/
│   ├── output/
│   └── runs/
├── cubicasa_hqa_500/                 # Optional default weights tree
│   └── runs/hqa500_offline/weights/best.pt
├── yolov8n-seg.pt                    # Base weights for train-from-scratch
├── best_hci21.pt                     # Optional HCI promote target
└── best_gdrive.pt                    # Optional production compare (read-only)
```

### Why this layout matters

| Code | Resolves `PROJECT_ROOT` as |
|------|----------------------------|
| `web/server.py` | Parent of `HCI_2.1` |
| `logic/yolo_inference.py` | Parent of `HCI_2.1` |

```text
DATASET_DIR = PROJECT_ROOT / "gdrive_dataset"
```

If you flatten the clone so `web/` sits at the filesystem root without a parent `HCI_2.1` folder name, **dataset and model paths break**.

### Created automatically on startup

When the server module loads, these directories are created with `mkdir(parents=True, exist_ok=True)` if missing:

- `gdrive_dataset/images_raw`
- `gdrive_dataset/images/train`
- `gdrive_dataset/labels/train`
- `gdrive_dataset/metadata`
- `gdrive_dataset/marked`
- `gdrive_dataset/output`
- `gdrive_dataset/runs`

You do **not** need to create them by hand for a normal start. Models (`.pt`) and CubiCasa weights are **not** auto-downloaded.

---

# 3. System Requirements

| Item | Requirement |
|------|-------------|
| **Python** | **3.11** (64-bit) |
| **Conda env name** | `improved_model_train` (enforced at startup unless overridden) |
| **OS** | Windows 10/11 fully supported; Linux suitable for headless/server |
| **Disk** | Space for images, runs, and IFC outputs (datasets grow quickly) |
| **GPU** | Optional; CPU torch works; CUDA needs matching torch wheel |

## Required packages

Install from `HCI_2.1/requirements.txt` (includes FastAPI, uvicorn, OpenCV, Ultralytics, torch, **ifcopenshell**, etc.).

## Windows support

- Preferred: Anaconda/Miniconda + `improved_model_train`
- Optional helpers: `START_HCI_2.1.bat`, `scripts/start_server.bat` (may contain developer-machine paths — prefer the `uvicorn` command in §9 on new servers)

## Linux notes

- Same conda + pip flow
- Bind `0.0.0.0:8000` behind a firewall or reverse proxy
- For SVG thumbs, install system Cairo if using `cairosvg`
- Set `HCI21_MODEL_PATH` to an absolute Linux path

---

# 4. Clone From Git

Clone so that **`HCI_2.1` remains a subdirectory of `PROJECT_ROOT`**.

```bash
# Choose your project root
mkdir -p /opt/hci          # or D:\hci_deploy on Windows
cd /opt/hci

# Replace with your actual remote URL
git clone <YOUR_REMOTE_URL> HCI_2.1

# Result:
#   /opt/hci/HCI_2.1/...
#   /opt/hci/          ← this is PROJECT_ROOT
```

**Where to clone**

| Path | Role |
|------|------|
| `/opt/hci` or `D:\hci_deploy` | `PROJECT_ROOT` |
| `/opt/hci/HCI_2.1` | Application code only |

Do **not** clone into a location and then move only the *contents* of `HCI_2.1` up one level without keeping the folder name `HCI_2.1`.

---

# 5. Create Environment

```bash
conda create -n improved_model_train python=3.11 -y
conda activate improved_model_train

cd PROJECT_ROOT/HCI_2.1

# Recommended: install the correct torch wheel for CPU or CUDA first
# https://pytorch.org

pip install -r requirements.txt
```

### ifcopenshell

Required for **Generate IFC**. Included in `requirements.txt`. If missing after install:

```bash
pip install ifcopenshell
```

### Optional packages

| Package | Feature |
|---------|---------|
| **gdown** | Google Drive folder download (`/api/download`) |
| **cairosvg** | SVG preview / thumbnails (needs system Cairo) |

Both are listed in `requirements.txt` as optional UI features.

---

# 6. Models

Weights are **not** stored in Git (`*.pt` is gitignored). Copy them onto the server or set an environment variable.

## Model table

| Model | Purpose | Required? | In Git? | On server? | `HCI21_MODEL_PATH`? |
|-------|---------|-----------|---------|------------|---------------------|
| **`best.pt`** (CubiCasa offline) at `cubicasa_hqa_500/runs/hqa500_offline/weights/best.pt` | Default Auto Label / Test if env unset | **Yes** *or* set env | No | Copy path or use env | Preferred override points here or any `.pt` |
| **`best_hci21.pt`** | HCI promote target; resolver fallback | Optional | No | Optional | Can point env here |
| **`best_gdrive.pt`** | Production compare only; **never** Auto Label default; never overwritten by HCI_2.1 | Optional | No | Optional | Do not rely on this for Auto Label |
| **`yolov8n-seg.pt`** | Base for train-from-scratch / scratch fine-tune | Required for **training** | No | Yes, under `PROJECT_ROOT` | N/A |

## Preferred configuration

```bash
export HCI21_MODEL_PATH=/absolute/path/to/best.pt
```

## Resolver fallback order (Auto Label)

1. `HCI21_MODEL_PATH` (file must exist)
2. `HCI_MODEL_PATH` (legacy)
3. `PROJECT_ROOT/cubicasa_hqa_500/runs/hqa500_offline/weights/best.pt`
4. `PROJECT_ROOT/best_hci21.pt`
5. CubiCasa smoke10 package (last resort)
6. Not found — **does not** fall through to `best_gdrive.pt`

---

# 7. Runtime Dataset

## Automatic creation

On server startup, `gdrive_dataset/` and its standard subfolders are created under `PROJECT_ROOT` if they do not already exist.

## Final structure

```text
PROJECT_ROOT/gdrive_dataset/
├── images_raw/          # Uploaded / downloaded source plans
├── images/
│   └── train/           # Images used for training after Auto Label
├── labels/
│   └── train/           # YOLO segmentation .txt labels
├── marked/              # Labelled overlays (*_labelled.jpg, etc.)
├── metadata/            # Scale, metadata JSON, IFC property JSON
├── output/              # Generated .ifc files
├── runs/                # Ultralytics training / merge runs
└── dataset.yaml         # Written by Auto Label when needed
```

## Folder roles

| Folder | Role |
|--------|------|
| **images_raw** | Source uploads; Auto Label input |
| **images/train** | Training image copies |
| **labels/train** | Editable YOLO label files |
| **marked** | Visual overlays for Correct Labels UI |
| **metadata** | `{name}.json`, `{name}_scale.json`, `{name}_ifc_props.json` |
| **output** | `{name}.ifc` BIM exports |
| **runs** | Train/finetune/merge checkpoints and metrics |

---

# 8. Environment Variables

| Variable | Purpose |
|----------|---------|
| **`HCI21_MODEL_PATH`** | Absolute path to YOLO `.pt` for Auto Label / Test. **Recommended in production.** |
| **`HCI_MODEL_PATH`** | Legacy alias (lower priority than `HCI21_MODEL_PATH`) |
| **`HCI21_ALLOW_ANY_PYTHON`** | Set to `1` / `true` / `yes` to skip the `improved_model_train` interpreter check. Use only if torch loads correctly in that Python. |

### Windows (PowerShell)

```powershell
$env:HCI21_MODEL_PATH = "D:\hci_deploy\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt"
# Optional escape hatch:
# $env:HCI21_ALLOW_ANY_PYTHON = "1"
```

### Windows (CMD)

```bat
set HCI21_MODEL_PATH=D:\hci_deploy\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt
```

### Linux / macOS

```bash
export HCI21_MODEL_PATH=/opt/hci/cubicasa_hqa_500/runs/hqa500_offline/weights/best.pt
```

---

# 9. Starting the Server

## Preferred command

```bash
conda activate improved_model_train
cd PROJECT_ROOT/HCI_2.1/web
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Then open: `http://localhost:8000/` (or `http://<server-ip>:8000/`).

## Startup order

1. Activate `improved_model_train`
2. Set `HCI21_MODEL_PATH` (recommended)
3. `cd` into `HCI_2.1/web`
4. Run uvicorn
5. Confirm browser loads the Floor Plan Model Trainer UI

## What happens during startup

1. **Interpreter gate** — exits if Python is not under `improved_model_train` (unless `HCI21_ALLOW_ANY_PYTHON` is set)
2. **Path setup** — `PROJECT_ROOT` = parent of `HCI_2.1`; `DATASET_DIR` = `gdrive_dataset`
3. **Folder ensure** — creates the seven dataset subdirs if missing
4. **App bind** — FastAPI listens on host/port; UI served from `/`

---

# 10. Web UI Workflow

Typical operator path:

```text
Upload
  ↓
Auto Label
  ↓
Correct Labels
  ↓
Scale Calibration
  ↓
Metadata / IFC Properties
  ↓
Generate IFC
  ↓
Download IFC
```

### Step details

1. **Upload** — Train tab → Upload Images (or Drive download). Files land in `images_raw/`.
2. **Auto Label** — Select images → Auto Label. Produces labels, train images, and marked overlays. Status/log stream updates in the UI.
3. **Correct Labels** — Correct Labels tab → edit polygons → Save. Optionally **Update Model** to fine-tune.
4. **Scale Calibration** — Click two points on the plan, enter real-world distance (meters). Writes `metadata/{basename}_scale.json`.
5. **Metadata** — Review / save room or AI metadata; edit IFC property sets in the IFC Properties panel.
6. **Generate IFC** — Builds geometry from labels + meters-per-pixel.
7. **Download IFC** — Fetches `gdrive_dataset/output/{basename}.ifc`.

---

# 11. Training Workflow

Accessible from the Train tab and the **Update Model** dialog on Correct Labels.

| Mode | Behavior |
|------|----------|
| **Train from scratch** | Starts from `PROJECT_ROOT/yolov8n-seg.pt` |
| **Fine-tune** | Continues from the selected / active HCI model using corrected labels |
| **Merge models** | Combines two checkpoints into a new run under `gdrive_dataset/runs/` |
| **Promote** | Copies a successful checkpoint to **`best_hci21.pt` only** — never writes `best_gdrive.pt` |

### Checkpoint locations

| Location | Contents |
|----------|----------|
| `gdrive_dataset/runs/.../weights/best.pt` | Ultralytics training outputs |
| `PROJECT_ROOT/best_hci21.pt` | HCI_2.1 promoted active weights |
| `PROJECT_ROOT/best_gdrive.pt` | Protected production weights (compare only) |

---

# 12. IFC Workflow

| Step | Detail |
|------|--------|
| **Generate IFC** | API/UI calls `logic/ifc_pipeline.generate_full_ifc` using corrected YOLO labels |
| **Room properties** | Stored/edited via IFC Properties panel → `metadata/{basename}_ifc_props.json` |
| **Wall generation** | Wall polygons → IFC wall entities (thickness / height from schema defaults + props) |
| **Door / Window** | Opening classes mapped into IFC openings / door / window entities |
| **Scale** | Uses calibrated `meters_per_pixel` or a documented default fallback |
| **Output folder** | `gdrive_dataset/output/{basename}.ifc` |

Requires **`ifcopenshell`** installed in the active environment.

---

# 13. Deployment on a New Server

Checklist:

1. **Clone** — `git clone <url> HCI_2.1` under a chosen `PROJECT_ROOT`
2. **Install** — create `improved_model_train` (Python 3.11), `pip install -r requirements.txt` (correct torch for CPU/CUDA)
3. **Copy models** — place CubiCasa `best.pt` and/or set `HCI21_MODEL_PATH`; add `yolov8n-seg.pt` if training
4. **Set environment variables** — at minimum `HCI21_MODEL_PATH`
5. **Run** — `cd HCI_2.1/web && python -m uvicorn server:app --host 0.0.0.0 --port 8000`
6. **Verify** — browser opens UI; upload → Auto Label smoke test; optional IFC generate

`gdrive_dataset` folders are created automatically on start.

---

# 14. Troubleshooting

| Issue | Likely cause | Fix |
|-------|--------------|-----|
| **Model not found** | No env path and no CubiCasa / `best_hci21.pt` | Set `HCI21_MODEL_PATH` or copy offline `best.pt` |
| **Wrong Python environment** | Server exits: interpreter refused | `conda activate improved_model_train` or set `HCI21_ALLOW_ANY_PYTHON=1` |
| **ifcopenshell missing** | IFC import error | `pip install ifcopenshell` |
| **No images found** | Empty `images_raw` | Upload images first |
| **Missing gdrive_dataset** | Wrong `PROJECT_ROOT` / app not started | Start server once (auto-creates) or fix layout |
| **Wrong PROJECT_ROOT** | Flattened clone | Keep `PROJECT_ROOT/HCI_2.1/...` |
| **CUDA / CPU mismatch** | Wrong torch wheel / DLL errors | Reinstall torch from pytorch.org for this machine |
| **Port already in use** | Another process on 8000 | Change `--port` or stop the other service |

---

# 15. Verification Checklist

After deployment, confirm:

- [ ] Server starts without interpreter / import errors
- [ ] Upload works (`images_raw` receives files)
- [ ] Auto Label works (labels + marked overlays appear)
- [ ] Correct Labels works (save / edit polygons)
- [ ] Training works (optional; needs `yolov8n-seg.pt` + labels)
- [ ] Scale calibration works (`*_scale.json` written)
- [ ] Metadata works (JSON under `metadata/`)
- [ ] IFC generation works (`output/*.ifc` created)
- [ ] IFC downloads correctly in the browser
- [ ] Logs / SSE stream show no unexpected errors

---

# 16. Notes for Developers

- **Do not commit datasets** — keep `gdrive_dataset/` contents out of Git.
- **Do not commit generated IFC** — `*.ifc` is ignored.
- **Do not commit `runs/`** — training artifacts stay local/server-side.
- **Do not commit `*.pt` weights** — ship models separately or via `HCI21_MODEL_PATH`.
- **Prefer environment variables** for model paths in production.
- **Keep `PROJECT_ROOT` layout unchanged** — sibling `gdrive_dataset` and optional CubiCasa weights.
- **Never promote to `best_gdrive.pt` from HCI_2.1** — production weights are protected; promote only to `best_hci21.pt`.
- Nested trees such as `hci-3d-model/` and `IMPROVED_MODEL_1.1/` are **not** part of the live Web UI runtime.

---

## Quick reference

```bash
conda activate improved_model_train
export HCI21_MODEL_PATH=/absolute/path/to/best.pt
cd PROJECT_ROOT/HCI_2.1/web
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

**UI:** `http://localhost:8000/`
