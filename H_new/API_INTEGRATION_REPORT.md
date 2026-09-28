# HCI Image → IFC External API Integration Report

**Document purpose:** Hand this to senior leadership and a separate frontend team to integrate with the existing one-shot Image → IFC API.

**Scope:** `C:\H` HCI backend only. External frontend → HCI API → IFC download.

**Mode:** Read-only analysis of existing code. No server was started, no API was executed, and no IFC was generated for this report.

**Legend:**
- **Verified** — confirmed from source/config in `C:\H`
- **Not verified in code** — not found / cannot be confirmed from inspected files
- **Inference** — logical conclusion from verified facts (called out explicitly)

---

## 1. Executive Summary

The separate frontend talks to **one main API**:

**`POST /api/process-floorplan`**

1. Frontend uploads a 2D floor-plan image (**PNG / JPG / JPEG**, among other allowed extensions) as **multipart form data**.
2. The HCI FastAPI backend **waits** while it:
   - saves the image,
   - runs **YOLO** auto-labeling,
   - validates labels,
   - **generates and saves** an IFC file on disk.
3. Only **after** that work finishes does the backend return a **JSON** response (not the IFC file itself).
4. That JSON includes a **`download_url`** (relative path).
5. The frontend then makes a **second request**: **`GET /api/ifc/file/{basename}`**, and receives the **IFC binary**.

**What the frontend ultimately receives:** a downloadable `.ifc` file (walls, doors, windows, openings), obtained via GET after a successful POST.

**Verified:** This production path is intended for an external frontend (`logic/production_orchestrator.py` module docstring: *“Production floor-plan orchestration for the external frontend.”*).

---

## 2. Exact API Endpoint

### Process (one-shot)

| Item | Value |
|------|--------|
| Method | `POST` |
| Route | `/api/process-floorplan` |
| Handler | `process_floorplan_endpoint` |
| Defined in | `C:\H\web\server.py` |
| Default host (start script) | `127.0.0.1` |
| Default port (start script) | `8000` |
| Example URL | `http://127.0.0.1:8000/api/process-floorplan` |
| Authentication | **None** (verified: no API key / Bearer / session on this route) |
| CORS | `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]` (`web/server.py`) |

```
POST /api/process-floorplan
```

**Note (verified):** `scripts\start_server.bat` binds `127.0.0.1:8000`. If `web/server.py` is started via `if __name__ == "__main__"`, it uses `host="0.0.0.0", port=8000`. The standard launcher uses **127.0.0.1**.

### Download (second step)

| Item | Value |
|------|--------|
| Method | `GET` |
| Route | `/api/ifc/file/{basename}` |
| Handler | `download_ifc_file` |
| Defined in | `C:\H\web\server.py` |
| Example URL | `http://127.0.0.1:8000/api/ifc/file/<basename>` |

```
GET /api/ifc/file/{basename}
```

### Optional health (ops / preflight)

```
GET /api/production/health
```

Handler: `production_health` → `get_production_health` (`web/server.py`, `logic/production_orchestrator.py`).

---

## 3. Complete End-to-End Flow

**Verified from code:**

```
Separate Frontend
        |
        | POST multipart: file (+ optional meters_per_pixel)
        v
HCI FastAPI  (web/server.py)
        |
        v
process_floorplan_endpoint
        |
        v
ProductionOrchestrator.process
        |
        v
validate_production_request
        |
        v
make_unique_production_filename + save_raw_image_bytes
        |   → gdrive_dataset/images_raw/{unique_name}
        v
QualityValidator.ensure_model_available
        |
        v
_autolabel_worker  (sync wait via ThreadPoolExecutor + future.result)
        |   → auto_label.generate_labels → YOLO (yolo_inference)
        |   → labels/train/{basename}.txt
        v
QualityValidator.validate_after_autolabel
        |
        v
generate_ifc_for_basename  (logic/ifc_service.py)
        |
        v
generate_full_ifc  (logic/ifc_pipeline.py)
        |   → gdrive_dataset/output/{basename}.ifc
        v
JSON response  (includes download_url)
        |
        | Frontend reads download_url
        v
Separate Frontend
        |
        | GET /api/ifc/file/{basename}
        v
download_ifc_file → FileResponse (IFC binary)
```

---

## 4. VERY IMPORTANT: Explain POST Timing

### Answer (verified)

**When the frontend sends `POST /api/process-floorplan`, the backend does NOT return JSON immediately.**

It **waits until IFC generation has completed and the IFC file has been written**, then returns JSON.

### Exact sequence (verified)

1. Frontend sends POST  
2. Backend receives and reads image bytes  
3. Backend validates and saves image  
4. Backend checks YOLO model availability  
5. Backend runs YOLO auto-label **and waits** (`future.result(timeout=...)`)  
6. Backend validates labels  
7. Backend generates IFC and saves it to disk  
8. Backend builds success JSON (with `download_url`)  
9. **Only then** does the POST HTTP response return  

### Synchronous vs asynchronous (verified)

- The **POST is synchronous** from the client’s point of view: one request, one response after the full pipeline.
- Comment in code: *“Run existing auto-label worker synchronously and wait for completion.”* (`process_floorplan` in `production_orchestrator.py`).
- Endpoint returns `orchestrator.process(...)` directly — no `BackgroundTasks` on this route.
- There is **no** job queue / webhook / deferred “job id → poll later” pattern on this one-shot endpoint (**verified**: not present in this flow).

### Progress / loading (verified)

- `/api/stream` exists for the **HCI training UI** (SSE), but it is **not** wired into `POST /api/process-floorplan`.
- **No progress streaming / polling / webhook was verified for this one-shot endpoint.**
- Frontend should show its **own** loading state while awaiting the POST response.

---

## 5. POST Request Contract

| Item | Verified |
|------|----------|
| Content-Type | `multipart/form-data` |
| Field `file` | **Required** — `UploadFile = File(...)` |
| Field `meters_per_pixel` | **Optional** — `Form(None)` |
| JSON body upload | **Not supported** on this endpoint |
| Base64 upload | **Not supported** on this endpoint |
| Authentication | **None** |
| App-level max upload size | **Not verified in code** (no limit configured in inspected app code) |

### Accepted image extensions (verified)

From `logic/dataset_io.py` → `IMG_EXTS`:

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.svg`

### Scale behavior (verified)

- If `meters_per_pixel` is provided and valid (finite float **> 0**) → used; `mpp_source` becomes `"request"`.
- If omitted / empty → `resolve_meters_per_pixel` → `resolve_mpp` → fallback **`DEFAULT_MPP = 0.01`** (`logic/scale_calibration.py`).

### curl example

```bash
curl -X POST "http://127.0.0.1:8000/api/process-floorplan" \
  -F "file=@floorplan.png" \
  -F "meters_per_pixel=0.01"
```

### JavaScript FormData / fetch example (contract-accurate)

```javascript
const API_BASE = "http://127.0.0.1:8000"; // replace with HCI API origin

async function processFloorplan(file, metersPerPixel) {
  const form = new FormData();
  form.append("file", file);
  if (metersPerPixel != null && metersPerPixel !== "") {
    form.append("meters_per_pixel", String(metersPerPixel));
  }

  // Do NOT set Content-Type manually when using FormData
  const res = await fetch(`${API_BASE}/api/process-floorplan`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok || data.success === false) {
    throw new Error(data.error || data.error_code || `HTTP ${res.status}`);
  }
  return data;
}
```

---

## 6. Backend Processing Pipeline

### External API contract (frontend-facing)

1. `POST /api/process-floorplan` with multipart image  
2. Wait for JSON  
3. `GET` `API_BASE + download_url` for IFC bytes  

### Internal backend processing (verified call chain)

| Step | Function / class | File | Purpose | Output / next input |
|------|------------------|------|---------|---------------------|
| 1 | `process_floorplan_endpoint` | `C:\H\web\server.py` | HTTP entry; reads upload bytes | filename + bytes → orchestrator |
| 2 | `ProductionOrchestrator.process` | `C:\H\logic\production_orchestrator.py` | Validate + orchestrate | calls `process_floorplan` |
| 3 | `validate_production_request` | same | Extension / empty / mpp checks | normalized mpp or `None` |
| 4 | `make_unique_production_filename` | same | Unique filename | e.g. `plan_20260811_143211_ab12cd.png` |
| 5 | `save_raw_image_bytes` | `C:\H\logic\dataset_io.py` | Write raw image | `gdrive_dataset/images_raw/...` |
| 6 | `QualityValidator.ensure_model_available` | `C:\H\logic\production_validation.py` | Resolve YOLO weights path | path + source or 503 |
| 7 | `_autolabel_worker` | `C:\H\web\server.py` | Auto-label selected file | labels under `labels/train/` |
| 8 | `generate_labels` | `C:\H\auto_label.py` | YOLO entry for labels | label lines + image |
| 9 | `run_yolo_inference` / `resolve_hci21_model` | `C:\H\logic\yolo_inference.py` | Model resolve + inference | detections / YOLO-seg lines |
| 10 | `QualityValidator.validate_after_autolabel` | `C:\H\logic\production_validation.py` | Label quality gate | validation dict or error |
| 11 | `generate_ifc_for_basename` | `C:\H\logic\ifc_service.py` | IFC service entry | calls `generate_full_ifc` |
| 12 | `generate_full_ifc` | `C:\H\logic\ifc_pipeline.py` | Build IFC4 geometry | `.ifc` on disk + result dict |
| 13 | `build_production_success_body` | `C:\H\logic\production_orchestrator.py` | JSON envelope | response with `download_url` |
| 14 | `download_ifc_file` | `C:\H\web\server.py` | Serve IFC binary | `FileResponse` |

---

## 7. Exact IFC Generator

| Question | Verified answer |
|----------|-----------------|
| Exact function | `generate_full_ifc` |
| Exact file | `C:\H\logic\ifc_pipeline.py` |
| Called via | `generate_ifc_for_basename` in `C:\H\logic\ifc_service.py` |
| Uses `logic/ifc_pipeline.py`? | **Yes** |
| Uses `C:\floorplan_ai_ifc`? | **No** (no references under `C:\H\logic`) |
| DetectionIR involved? | **No** |
| BuildingModelIR involved? | **No** |
| Furniture-aware exporter? | **No** |

### Entities generated (verified from `ifc_pipeline.py`)

- `IfcWall`
- `IfcDoor`
- `IfcWindow`
- `IfcOpeningElement` (+ void/fill relationships)

Label classes used: Wall=`3`, Door=`2`, Window=`1`.

### Not generated by this API path (verified)

- Furniture / `IfcFurnishingElement` (not created in `generate_full_ifc`)
- Electric appliances / `IfcElectricAppliance` (not created in `generate_full_ifc`)
- DetectionIR / BuildingModelIR pipelines

---

## 8. POST Response Timing and JSON Response

**When JSON is created (verified):** After IFC file write succeeds and `build_production_success_body` runs. POST body is **JSON metadata**, **not** IFC binary.

### Example success response (fields from code)

```json
{
  "success": true,
  "request_id": "<uuid>",
  "basename": "floorplan_20260811_143211_ab12cd",
  "original_filename": "floorplan.png",
  "stored_filename": "floorplan_20260811_143211_ab12cd.png",
  "processing_time_ms": 18420,
  "download_url": "/api/ifc/file/floorplan_20260811_143211_ab12cd",
  "ifc_path": "C:\\H\\gdrive_dataset\\output\\floorplan_20260811_143211_ab12cd.ifc",
  "ok": true,
  "meters_per_pixel": 0.01,
  "mpp_source": "request",
  "gt_walls": 15,
  "gt_doors": 7,
  "gt_windows": 4,
  "ifc_walls": 15,
  "ifc_openings": 11,
  "ifc_doors": 7,
  "ifc_windows": 4,
  "mapping_success_rate": 1.0,
  "mapped_openings": 11,
  "unmapped_openings": 0,
  "bbox_m": { "size_x_m": 12.3, "size_y_m": 8.1, "height_m": 3.0 },
  "result": { "walls": 15, "doors": 7, "windows": 4 },
  "validation": { },
  "warnings": [],
  "timing": {
    "upload_ms": 12,
    "autolabel_ms": 15000,
    "validation_ms": 5,
    "ifc_ms": 3400,
    "total_ms": 18420
  },
  "generated_at": "<iso8601>",
  "timestamp": "<iso8601>"
}
```

Numeric counts above are **illustrative**; exact numbers depend on the image/model.

### Important fields

| Field | Meaning | Frontend needs? |
|-------|---------|-----------------|
| `success` | Overall success flag | **Yes** |
| `request_id` | Correlation ID | Recommended |
| `basename` | Server id / IFC stem | Useful |
| `download_url` | Relative IFC download path | **Yes — primary** |
| `ifc_path` | Absolute server filesystem path | **Do not use in browser** |
| `processing_time_ms` | End-to-end duration | Optional |
| `meters_per_pixel` | Scale used | Optional |
| `mpp_source` | How scale was chosen | Optional |
| `ifc_walls` / `ifc_doors` / `ifc_windows` / `ifc_openings` | IFC entity counts | Optional UI |
| `warnings` | Non-fatal messages | Recommended |
| `timing` | Stage timings | Optional |
| `validation` | Quality-gate report | Optional |
| `result` | Nested wall/door/window counts | Optional |

Also present when IFC succeeds: `ok`, `gt_*`, mapping stats, `bbox_m`, `generated_at` (from `generate_full_ifc` + envelope).

---

## 9. What is `basename`?

| Topic | Verified |
|-------|----------|
| Where generated | After save: `basename = saved.stem` in `process_floorplan` |
| Filename construction | `make_unique_production_filename`: `{stem}_{YYYYMMDD_HHMMSS}_{6hex}{ext}` |
| Example | `floorplan.png` → `floorplan_20260811_143211_ab12cd.png` → basename `floorplan_20260811_143211_ab12cd` |
| IFC filename | `gdrive_dataset/output/{basename}.ifc` |
| Download route | `/api/ifc/file/{basename}` |
| Why frontend gets it | Identifies which IFC belongs to this request; also embedded in `download_url` |

Stem sanitization (verified): path separators and spaces in stem replaced with `_`.

---

## 10. IFC Download Flow

```
POST /api/process-floorplan
  → wait for full processing
  → JSON response
  → frontend reads download_url
  → frontend SECOND request: GET download_url (with API_BASE)
  → backend returns IFC binary
  → frontend saves / opens / views IFC
```

### Why two requests? (verified)

POST returns a **JSON envelope** (metadata + relative URL). The IFC file is served separately via **`FileResponse`** on GET.

### Download details

| Item | Verified |
|------|----------|
| Route | `GET /api/ifc/file/{basename}` |
| Handler | `download_ifc_file` |
| Storage | `C:\H\gdrive_dataset\output\{basename}.ifc` (`ifc_file_path`) |
| Content-Type | `application/octet-stream` |
| Download filename | `{basename}.ifc` |
| `download_url` | **Relative**, e.g. `/api/ifc/file/{basename}` |
| IFC TTL / expiry / cleanup | **Not verified in code** (no retention/cleanup found under `logic/`) |

**Do NOT expose/use `ifc_path` in the browser because it is a server filesystem path.**

Frontend construction:

```text
ifcUrl = API_BASE + download_url
```

---

## 11. Frontend Integration

1. User selects PNG/JPG/JPEG (or other allowed extension).  
2. Create `FormData`.  
3. `form.append("file", file)`.  
4. Optionally `form.append("meters_per_pixel", "0.01")`.  
5. `POST` to `{API_BASE}/api/process-floorplan`.  
6. Wait for response (show loading UI).  
7. Check HTTP status.  
8. Check `success === true`.  
9. Read `download_url`.  
10. Build `API_BASE + download_url`.  
11. `GET` that URL.  
12. Receive binary / Blob.  
13. Save / open / pass to IFC viewer.  

### Complete fetch example

```javascript
const API_BASE = "http://127.0.0.1:8000";

async function imageToIfc(file, metersPerPixel) {
  const form = new FormData();
  form.append("file", file);
  if (metersPerPixel != null && metersPerPixel !== "") {
    form.append("meters_per_pixel", String(metersPerPixel));
  }

  // Client timeout: inference — YOLO wait alone is 600s verified in backend
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15 * 60 * 1000);

  let processRes;
  try {
    processRes = await fetch(`${API_BASE}/api/process-floorplan`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }

  const data = await processRes.json();
  if (!processRes.ok || data.success === false) {
    throw new Error(data.error || data.error_code || `HTTP ${processRes.status}`);
  }

  const ifcRes = await fetch(API_BASE + data.download_url);
  if (!ifcRes.ok) {
    throw new Error(`IFC download failed: HTTP ${ifcRes.status}`);
  }

  const blob = await ifcRes.blob();
  return { blob, basename: data.basename, meta: data };
}
```

---

## 12. Loading / Waiting Behavior

| Question | Verified answer |
|----------|-----------------|
| Is POST synchronous? | **Yes** |
| Does frontend wait? | **Yes**, until JSON after IFC save |
| Backend stream progress on this endpoint? | **No** (not wired) |
| Polling endpoint for this one-shot job? | **No** |
| Webhook? | **No** |
| Job ID / queue? | **No** (`request_id` is for logging/correlation only) |
| JSON only after IFC generation? | **Yes** (on success path) |

**No progress streaming/polling was verified for this one-shot endpoint. The frontend should show its own loading state while awaiting the POST response.**

*(Note: `/api/stream` SSE exists for the HCI training UI; it is unrelated to this production one-shot contract.)*

---

## 13. Error Handling

### Error JSON structure (verified — `build_production_error_body`)

```json
{
  "success": false,
  "request_id": "<uuid or null>",
  "error": "<message>",
  "error_code": "<code>",
  "processing_time_ms": 123,
  "timestamp": "<iso8601>",
  "basename": null,
  "warnings": [],
  "timing": { "upload_ms": 0, "autolabel_ms": 0, "validation_ms": 0, "ifc_ms": 0, "total_ms": 123 }
}
```

`trace` only if `HCI21_DEBUG` is set.

### Verified status / error_code examples

| HTTP | `error_code` |
|------|----------------|
| 400 | `invalid_request` |
| 404 | `labels_missing`, `image_missing` (also default map `not_found`) |
| 422 | `invalid_label_format`, `labels_missing`, `empty_labels`, `no_walls_detected` |
| 503 | `model_unavailable` |
| 504 | `timeout` |
| 500 | `internal_error`, `ifc_generation_failed` |

### Common failures (from code)

- Missing/empty file or bad extension  
- Invalid `meters_per_pixel`  
- No YOLO weights (`model_unavailable`)  
- Auto-label timeout (600s)  
- No walls / bad labels (`no_walls_detected`, `invalid_label_format`, …)  
- IFC generation exception  

**Frontend should display:** HTTP status + `error` message (+ `error_code` for support).

Download miss (GET) returns a **different** shape: `{"error": "IFC not found. ..."}` with status 404.

---

## 14. Health Check

```
GET /api/production/health
```

| Item | Verified |
|------|----------|
| Function | `get_production_health` |
| YOLO inference? | **No** |
| IFC generation? | **No** |
| Side effects | None significant (reads model path / dirs) |

**Response fields:** `status` (`healthy`|`degraded`), `version` (`"2.1"`), `model_available`, `model_source`, `model_path`, `dataset_directory_exists`, `output_directory_exists`, `timestamp`, `uptime_seconds`.

Use before integration: expect `model_available: true`.

---

## 15. Server Deployment Requirements

### Required for API execution (verified)

| Requirement | Detail |
|-------------|--------|
| FastAPI app | `web/server.py` |
| Uvicorn | Start via `scripts\start_server.bat` / `START_HCI_2.1.bat` |
| Python env | Bat expects conda `improved_model_train` |
| Packages | See `C:\H\requirements.txt` (fastapi, uvicorn, python-multipart, torch, ultralytics, opencv, ifcopenshell, …) |
| YOLO weights | `HCI21_MODEL_PATH` or `resolve_hci21_model` fallbacks |
| Dataset dirs | `gdrive_dataset/` tree; `ensure_dataset_dirs` on startup |
| Output dir | `gdrive_dataset/output/` writable |
| Port | `8000` |

### Env vars (verified names)

- `HCI21_MODEL_PATH` — set by start script (practical requirement)  
- `HCI_MODEL_PATH` — legacy alias in resolver  
- `HCI21_DEBUG` — optional error traces  
- `HCI21_ALLOW_ANY_PYTHON` — optional interpreter override  

### Required only for remote access (not for local same-machine)

- Bind beyond `127.0.0.1` and/or reverse proxy  
- Public DNS / TLS as needed  

**Not verified in code:** cloud object storage, CDN URLs, Docker Compose for this API.

---

## 16. Same-Server vs Remote Browser

### Meaning of `127.0.0.1` (verified)

With the standard start script, the API listens **only on the HCI machine’s loopback**. Processes on that machine can connect; remote machines cannot reach that socket.

### A. Separate frontend **backend/proxy** on same machine as HCI

- **Yes**, call `http://127.0.0.1:8000` or `http://localhost:8000`.

### B. Frontend **browser on same machine** as HCI

- **Yes**, browser can call `http://127.0.0.1:8000`.

### C. Frontend browser on **another user’s laptop**

- **No**, not with current `127.0.0.1` binding — browser cannot reach the server’s loopback.
- CORS being open does **not** fix reachability.

### D. Reverse proxy / same domain

- Frontend uses the **public API origin** as `API_BASE`.
- Still uses `API_BASE + download_url`.
- **Inference:** proxy must forward `/api/process-floorplan` and `/api/ifc/file/*` with long upstream timeouts.

---

## 17. CORS

**Verified** (`web/server.py`):

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

**CORS does:** allow browsers on other origins/ports to make cross-origin requests **if the host is reachable**.

**CORS does not:** make `127.0.0.1` on the server reachable from a remote user’s browser.

---

## 18. Authentication / Security — current state only

| Control | Current state |
|---------|----------------|
| API key | **None** |
| Bearer token | **None** |
| Session auth | **None** |
| CORS | Open `*` |
| Upload validation | Extension allow-list + non-empty bytes + mpp rules |
| Upload size limit | **Not verified in app code** |
| Unique upload naming | Yes (`make_unique_production_filename`) |
| GET basename path-traversal guard | **Not verified** as explicit sanitization in `download_ifc_file` |

**Future recommendation (not implemented):** add auth before public internet exposure.

---

## 19. Performance / Timeout

| Item | Classification | Value |
|------|----------------|-------|
| YOLO / autolabel wait | **Verified timeout** | `DEFAULT_AUTOLABEL_TIMEOUT_S = 600` seconds (`production_orchestrator.py`) |
| Separate IFC-stage timeout | **Not verified in code** | None found |
| Uvicorn / HTTP server timeout | **Not verified in code** | Not set in start script / inspected app config |
| POST behavior | **Verified** | Synchronous until pipeline completes |
| Client timeout | **Inference / recommendation** | Configure **> 600s** (e.g. 10–15 minutes); not a backend HTTP timeout setting |

Timing stages returned in JSON: `upload_ms`, `autolabel_ms`, `validation_ms`, `ifc_ms`, `total_ms`.

---

## 20. Current IFC Capabilities and Limitations

### Contains (verified)

- Walls  
- Doors  
- Windows  
- Openings (voids/fills)  

### Does not contain (verified for this API path)

- Furniture  
- Electric appliances  
- `floorplan_ai_ifc` DetectionIR → BuildingModelIR exporter  

### Other limitations

- Relative `download_url` only  
- Default localhost bind  
- No auth  
- Long synchronous POST  
- Disk storage under `gdrive_dataset/output/`  

---

## 21. What the Senior Needs to Know

Simple meeting wording:

- Frontend **image POST** karega to **`POST /api/process-floorplan`**.  
- Backend image receive karke **YOLO processing** karega.  
- Phir **IFC generate** hogi aur server pe save hogi.  
- IFC banne ke **baad** backend **JSON return** karega.  
- JSON mein **`download_url`** hoga.  
- Frontend us `download_url` se **second GET** call karega: **`GET /api/ifc/file/{basename}`**.  
- Us GET se **actual IFC binary** milegi.  
- POST pehle se IFC file nahi bhejta — pehle JSON, phir GET.  
- Abhi IFC mein mainly **walls / doors / windows** hain; furniture/appliances is path mein nahi.

---

## 22. One-Minute Meeting Explanation

“External frontend `POST /api/process-floorplan` pe PNG/JPG bhejta hai. Backend **wait** karta hai jab tak YOLO + IFC complete na ho; phir JSON deta hai jisme `download_url` hota hai. Frontend us URL pe second `GET` karta hai aur `.ifc` binary download karta hai. Default URL `http://127.0.0.1:8000` hai. Auth nahi hai. IFC abhi walls/doors/windows wali HCI pipeline se banti hai — furniture-aware `floorplan_ai_ifc` is endpoint pe use nahi hoti.”

---

## 23. Final Integration Checklist

- [ ] API base URL configured  
- [ ] `POST /api/process-floorplan` configured  
- [ ] `multipart/form-data` used  
- [ ] `file` field used  
- [ ] Optional `meters_per_pixel` handled  
- [ ] Long client timeout configured (**recommendation:** > 600s)  
- [ ] Success response handled (`success === true`)  
- [ ] `download_url` handled  
- [ ] Second GET implemented  
- [ ] IFC blob handled  
- [ ] Error handling implemented (`error` / `error_code`)  
- [ ] Health check tested (`GET /api/production/health`)  
- [ ] API host reachable from browser or frontend server  
- [ ] CORS/network verified for your deployment topology  

---

## 24. Important Clarifications

| # | Question | Answer (verified unless noted) |
|---|----------|--------------------------------|
| 1 | Does POST return IFC directly? | **No** — JSON only |
| 2 | Does POST return JSON before IFC generation? | **No** — waits until IFC saved (success path) |
| 3 | Does POST wait for IFC generation? | **Yes** |
| 4 | Does frontend need a second GET? | **Yes** |
| 5 | Is there polling? | **No** for this one-shot API |
| 6 | Is there a webhook? | **No** |
| 7 | Is there a job queue? | **No** |
| 8 | Where is IFC stored? | `C:\H\gdrive_dataset\output\{basename}.ifc` |
| 9 | How does frontend know which IFC is theirs? | `basename` / `download_url` in same POST response |
| 10 | What key/identifier is used? | `basename` (also in `download_url`); `request_id` for correlation |
| 11 | Is `download_url` relative or absolute? | **Relative** |
| 12 | Can a browser use `ifc_path`? | **Should not** — server filesystem path |
| 13 | YOLO/model unavailable? | **503** / `model_unavailable` |
| 14 | IFC generation fails? | **500** / `ifc_generation_failed` (or related) |
| 15 | Supported image formats? | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`, `.svg` |
| 16 | Authentication required? | **No** |
| 17 | Remote browser with `127.0.0.1` bind? | **No** — cannot reach server loopback |

---

## 25. Source Code References

| Concern | File | Function / Class | Relevant route / logic |
|---------|------|------------------|------------------------|
| POST API | `C:\H\web\server.py` | `process_floorplan_endpoint` | `POST /api/process-floorplan` |
| IFC download | `C:\H\web\server.py` | `download_ifc_file` | `GET /api/ifc/file/{basename}` |
| Health | `C:\H\web\server.py` | `production_health` | `GET /api/production/health` |
| Orchestrator | `C:\H\logic\production_orchestrator.py` | `ProductionOrchestrator`, `process_floorplan`, `build_production_success_body` | External production pipeline |
| Request validation | `C:\H\logic\production_orchestrator.py` | `validate_production_request`, `make_unique_production_filename` | Upload rules / naming |
| Dataset I/O | `C:\H\logic\dataset_io.py` | `save_raw_image_bytes`, `ifc_file_path`, `IMG_EXTS` | Paths / extensions |
| YOLO resolve / inference | `C:\H\logic\yolo_inference.py` | `resolve_hci21_model`, `run_yolo_inference` | Model + inference |
| Auto-label entry | `C:\H\auto_label.py` | `generate_labels` | Label generation |
| Auto-label worker | `C:\H\web\server.py` | `_autolabel_worker` | Invoked sync from orchestrator |
| Quality gate | `C:\H\logic\production_validation.py` | `QualityValidator`, `QualityGateError` | Pre-IFC checks |
| IFC service | `C:\H\logic\ifc_service.py` | `generate_ifc_for_basename` | IFC entry + `download_url` |
| IFC generation | `C:\H\logic\ifc_pipeline.py` | `generate_full_ifc`, `write_ifc4` | Walls/doors/windows IFC |
| Scale | `C:\H\logic\scale_calibration.py` | `DEFAULT_MPP`, `resolve_mpp` | Default 0.01 m/px |
| Server startup | `C:\H\scripts\start_server.bat`, `C:\H\START_HCI_2.1.bat` | uvicorn launch | `127.0.0.1:8000`, `HCI21_MODEL_PATH` |
| Dependencies | `C:\H\requirements.txt` | — | fastapi, torch, ultralytics, ifcopenshell, … |
| CORS | `C:\H\web\server.py` | `CORSMiddleware` | `allow_origins=["*"]` |

---

## Appendix — Frontend Team Quick Card

```
API_BASE = http://127.0.0.1:8000   (or your HCI API origin)

POST {API_BASE}/api/process-floorplan
  multipart/form-data
  file=REQUIRED
  meters_per_pixel=OPTIONAL

Wait until JSON (IFC already generated on success).

Then:
GET {API_BASE}{download_url}  →  .ifc binary

Auth: none
IFC contents: walls, doors, windows, openings
Not included: furniture / appliances / floorplan_ai_ifc IR pipeline
```

---

*End of report. Generated from static inspection of `C:\H` only.*
