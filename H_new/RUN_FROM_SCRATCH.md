# Run From Scratch — HCI_2.1 (Windows / PowerShell)

Project root: `D:\HCI_interor\HCI_2.1`  
Server entry: `D:\HCI_interor\HCI_2.1\web\server.py`

**Required Python (only env with working PyTorch on this machine):**

```text
C:\Users\DELL\anaconda3\envs\improved_model_train\python.exe
```

Do **not** use:

- `D:\HCI_interor\web_file_v2\.venv` → WinError 1114 on `c10.dll`
- Anaconda `base` → same `c10.dll` failure

---

## Recommended: double-click / batch launcher

```powershell
D:\HCI_interor\HCI_2.1\START_HCI_2.1.bat
```

or:

```powershell
cd D:\HCI_interor\HCI_2.1\scripts
.\start_server.bat
```

PowerShell:

```powershell
cd D:\HCI_interor\HCI_2.1\scripts
.\start_server.ps1
```

Open: `http://127.0.0.1:8000`

---

## Manual start (must be improved_model_train)

```powershell
conda deactivate
conda deactivate
conda activate improved_model_train

# Confirm — must print improved_model_train path and torch 2.8.0+cpu
python -c "import sys,torch; print(sys.executable); print(torch.__version__)"

cd D:\HCI_interor\HCI_2.1\web
$env:HCI21_MODEL_PATH = "D:\HCI_interor\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt"
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

If you still see `(.venv)` in the prompt, deactivate it first — that venv is usually `web_file_v2` and will break Auto Label.

---

## VS Code / Cursor

Open the folder `D:\HCI_interor\HCI_2.1` (not `web_file_v2`).

`.vscode/settings.json` locks the interpreter to `improved_model_train`.

Run / Debug: **HCI_2.1 FastAPI (improved_model_train)**  
or Tasks: **Start HCI_2.1 Server (improved_model_train)**

---

## Install dependencies (only if needed, in improved_model_train)

```powershell
conda activate improved_model_train
cd D:\HCI_interor\HCI_2.1
pip install -r requirements.txt
```

Optional: `pip install gdown cairosvg`

---

## Port 8000 busy

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## Common error: WinError 1114 / c10.dll

Cause: server started with `web_file_v2\.venv` or Anaconda base.

Fix: stop the server, start via `START_HCI_2.1.bat` only.
