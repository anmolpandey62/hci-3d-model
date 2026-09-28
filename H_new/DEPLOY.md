# HCI_2.1 Web UI — Production Deployment Guide

Deploy the latest HCI_2.1 FastAPI Web UI on a fresh machine so it behaves like the local setup.

**Scope:** application code under `HCI_2.1/` plus sibling runtime data/models under `PROJECT_ROOT`.

---

## 1. Repository layout

Expected structure on the server:

```text
<PROJECT_ROOT>/
    HCI_2.1/                          # application (Git clone of this app)
        web/
        logic/
        config/
        scripts/
        auto_label.py
        requirements.txt
        DEPLOY.md
        START_HCI_2.1.bat             # Windows helper (local paths may need edit)
    gdrive_dataset/                   # runtime data (not in Git)
        images_raw/
        images/train/
        labels/train/
        marked/
        metadata/
        output/
        runs/
    cubicasa_hqa_500/                 # at least weights path, or use HCI21_MODEL_PATH
        runs/hqa500_offline/weights/best.pt
    yolov8n-seg.pt                    # required for Train / scratch finetune
    best_hci21.pt                     # optional (promote target / fallback)
    best_gdrive.pt                    # optional (Test compare only; never overwritten by HCI_2.1)
```

### Why `PROJECT_ROOT` must be the parent of `HCI_2.1`

The Web UI resolves paths from the **parent** of the `HCI_2.1` folder:

| Module | How `PROJECT_ROOT` is derived |
|--------|-------------------------------|
| `web/server.py` | `HCI_2.1/web/server.py` → `HCI_2.1` → **parent** |
| `logic/yolo_inference.py` | `HCI_2.1/logic/` → `HCI_2.1` → **parent** |

Then:

```text
DATASET_DIR = PROJECT_ROOT / "gdrive_dataset"
```

If you clone so that `HCI_2.1` contents become the repo root (no parent folder named `HCI_2.1`), `gdrive_dataset` and model defaults will resolve to the **wrong place**. Always keep:

```text
<PROJECT_ROOT>/HCI_2.1/...
<PROJECT_ROOT>/gdrive_dataset/...
```

---

## 2. Python

**Required:** Python **3.11** (64-bit).

**Recommended conda environment** (name must contain `improved_model_train` unless you override the interpreter gate):

```bash
conda create -n improved_model_train python=3.11
conda activate improved_model_train
```

The server refuses to start unless `sys.executable` is under an env named `improved_model_train`, or you set `HCI21_ALLOW_ANY_PYTHON=1` (see §6).

---

## 3. Install dependencies

From the app directory:

```bash
cd <PROJECT_ROOT>/HCI_2.1
pip install -r requirements.txt
```

### Torch (CPU vs CUDA)

`requirements.txt` lists `torch`, but the **correct wheel depends on the machine**:

- **CPU-only servers:** install a CPU build from [https://pytorch.org](https://pytorch.org) first, then `pip install -r requirements.txt`.
- **GPU servers:** install the matching CUDA torch/torchvision wheels from pytorch.org, then install the rest of `requirements.txt`.

Mismatch (wrong CUDA / broken DLL) causes YOLO load failures (e.g. WinError 1114 on `c10.dll` on Windows).

### IFC

`ifcopenshell` is required for Generate IFC. It is listed in `requirements.txt`. If IFC fails with `ModuleNotFoundError: ifcopenshell`, reinstall:

```bash
pip install ifcopenshell
```

### Optional

- **gdown** — Google Drive download API  
- **cairosvg** — SVG thumbnails (also needs system Cairo libraries)

---

## 4. Required models

### Preferred: `HCI21_MODEL_PATH`

Set an absolute path to a YOLO segmentation `.pt` file:

```bash
# Linux / macOS
export HCI21_MODEL_PATH=/data/hci/models/best.pt

# Windows PowerShell
$env:HCI21_MODEL_PATH = "D:\models\best.pt"
```

This is the **preferred** production setting: one clear weight file, independent of CubiCasa folder layout.

### Fallback order (Auto Label / Test default)

Implemented in `logic/yolo_inference.resolve_hci21_model()`:

1. **`HCI21_MODEL_PATH`** (file must exist)
2. **`HCI_MODEL_PATH`** (legacy alias)
3. **`PROJECT_ROOT/cubicasa_hqa_500/runs/hqa500_offline/weights/best.pt`**
4. **`PROJECT_ROOT/best_hci21.pt`**
5. CubiCasa smoke10 checkpoint under `cubicasa_server_package_2121/...` (weaker; last resort)
6. **Not found** — does **not** fall through to `best_gdrive.pt`

### Model roles

| File | Role | Required? |
|------|------|-----------|
| CubiCasa `.../hqa500_offline/weights/best.pt` | Default Auto Label if env unset | Yes *or* set `HCI21_MODEL_PATH` |
| `best_hci21.pt` | HCI promote target / resolver fallback | Optional |
| `yolov8n-seg.pt` | Base weights for Train / scratch finetune | Required for training features |
| `best_gdrive.pt` | Production compare (read-only); never written by HCI_2.1 | Optional |

Weights are **not** committed (see `.gitignore`). Copy them onto the server separately.

---

## 5. Required runtime folders

Create the dataset tree under `PROJECT_ROOT` (empty dirs are fine):

```bash
# Linux / macOS
cd <PROJECT_ROOT>
mkdir -p gdrive_dataset/images_raw \
         gdrive_dataset/images/train \
         gdrive_dataset/labels/train \
         gdrive_dataset/marked \
         gdrive_dataset/metadata \
         gdrive_dataset/output \
         gdrive_dataset/runs
```

```powershell
# Windows PowerShell
cd <PROJECT_ROOT>
@(
  "gdrive_dataset\images_raw",
  "gdrive_dataset\images\train",
  "gdrive_dataset\labels\train",
  "gdrive_dataset\marked",
  "gdrive_dataset\metadata",
  "gdrive_dataset\output",
  "gdrive_dataset\runs"
) | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
```

| Folder | Purpose |
|--------|---------|
| `images_raw/` | Uploads / Drive downloads |
| `images/train/` | Training images after Auto Label |
| `labels/train/` | YOLO label `.txt` files |
| `marked/` | Overlay / corrected visuals |
| `metadata/` | Scale + IFC property JSON |
| `output/` | Generated `.ifc` files |
| `runs/` | Ultralytics training runs |

`dataset.yaml` is created by Auto Label when needed.

---

## 6. Environment variables

| Variable | Purpose |
|----------|---------|
| **`HCI21_MODEL_PATH`** | Absolute path to YOLO `.pt` used for Auto Label / Test default. **Recommended in production.** |
| **`HCI_MODEL_PATH`** | Legacy alias for the same purpose (lower priority than `HCI21_MODEL_PATH`). |
| **`HCI21_ALLOW_ANY_PYTHON`** | Set to `1` / `true` / `yes` to bypass the `improved_model_train` interpreter check. Use only if you know torch loads correctly in that interpreter. |

Example (Linux):

```bash
export HCI21_MODEL_PATH=/opt/hci/weights/best.pt
# export HCI21_ALLOW_ANY_PYTHON=1   # only if needed
```

---

## 7. Running the server

Activate the env, then start from `web/`:

```bash
conda activate improved_model_train
cd <PROJECT_ROOT>/HCI_2.1/web
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

- **Host `0.0.0.0`** — accept remote connections (use a firewall / reverse proxy in production).
- **Port `8000`** — default UI URL: `http://<server>:8000/`

Windows helpers (`START_HCI_2.1.bat`, `scripts/start_server.bat`) may still point at a developer machine path; on a new server prefer the `uvicorn` command above with the local conda Python.

### Quick verify

1. Open `http://<server>:8000/`
2. Check status / logs in the UI
3. Confirm Auto Label finds a model (`HCI21_MODEL_PATH` or CubiCasa `best.pt`)

---

## 8. Deployment checklist

- [ ] Clone / pull so layout is `<PROJECT_ROOT>/HCI_2.1/...` (parent is `PROJECT_ROOT`)
- [ ] Create conda env: `conda create -n improved_model_train python=3.11` and activate it
- [ ] Install deps: `pip install -r HCI_2.1/requirements.txt` (correct torch wheel for CPU/CUDA)
- [ ] Copy model weights (`best.pt` and/or set `HCI21_MODEL_PATH`; add `yolov8n-seg.pt` if training)
- [ ] Create runtime folders under `gdrive_dataset/` (§5)
- [ ] Set environment variables (`HCI21_MODEL_PATH`; `HCI21_ALLOW_ANY_PYTHON` only if needed)
- [ ] Start server: `cd HCI_2.1/web && python -m uvicorn server:app --host 0.0.0.0 --port 8000`
- [ ] Open browser: `http://<server>:8000/`

---

## 9. Troubleshooting

### Missing `ifcopenshell`

**Symptom:** Generate IFC fails with `ModuleNotFoundError: ifcopenshell`.  
**Fix:** `pip install ifcopenshell` (also listed in `requirements.txt`).

### Model not found

**Symptom:** Auto Label / Test errors about no HCI_2.1 YOLO model.  
**Fix:**

1. Set `HCI21_MODEL_PATH` to an existing `.pt` file, **or**
2. Place CubiCasa offline weights at  
   `PROJECT_ROOT/cubicasa_hqa_500/runs/hqa500_offline/weights/best.pt`, **or**
3. Place `best_hci21.pt` under `PROJECT_ROOT`.

Remember: `best_gdrive.pt` is **not** the Auto Label default.

### Dataset path missing

**Symptom:** uploads / labels / IFC write failures.  
**Fix:** ensure `<PROJECT_ROOT>/gdrive_dataset/` exists with the folders in §5. Confirm you did not start the app from a layout where `PROJECT_ROOT` is wrong.

### Wrong `PROJECT_ROOT`

**Symptom:** app looks for `gdrive_dataset` next to the wrong directory; empty UI folders; models “missing” even though files exist elsewhere.  
**Fix:** keep `HCI_2.1` as a **subdirectory** of `PROJECT_ROOT`. Do not flatten the clone so that `web/` sits at the git root without the `HCI_2.1` parent folder name.

### Wrong Python environment

**Symptom:** process exits immediately with “HCI_2.1 refused to start: wrong Python interpreter”.  
**Fix:** `conda activate improved_model_train` and run uvicorn with that Python.  
**Escape hatch:** `HCI21_ALLOW_ANY_PYTHON=1` (only if torch works in that interpreter).

### Torch install mismatch

**Symptom:** `c10.dll` / CUDA errors / ultralytics fails to load.  
**Fix:** reinstall torch from pytorch.org for this OS + CPU/CUDA combo; avoid mixing broken venvs (e.g. wrong local `.venv`) with the conda env.

### SVG preview fails

**Symptom:** SVG thumbnails return “SVG load failed”.  
**Fix:** install system Cairo for `cairosvg`, or use JPEG/PNG uploads (core UI does not require SVG).

---

## Related files

| File | Role |
|------|------|
| `requirements.txt` | Pip dependencies for the Web UI |
| `.gitignore` | Keeps datasets, weights, and nested archives out of Git |
| `scripts/start_server.bat` / `.ps1` | Local Windows launchers (may need path edits on new hosts) |
