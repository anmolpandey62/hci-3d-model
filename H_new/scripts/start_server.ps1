# HCI_2.1 FastAPI — ALWAYS use conda improved_model_train (working PyTorch).
# Do NOT use web_file_v2\.venv or Anaconda base — both fail on c10.dll.

$ErrorActionPreference = "Stop"

$Python = "C:\Users\DELL\anaconda3\envs\improved_model_train\python.exe"
$Model  = "D:\HCI_interor\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt"
$WebDir = Join-Path $PSScriptRoot "..\web" | Resolve-Path

if (-not (Test-Path $Python)) {
    Write-Error "improved_model_train env not found at: $Python"
    exit 1
}

# Prevent inherited broken venv from affecting child process
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
$env:HCI21_MODEL_PATH = $Model

Write-Host "[HCI_2.1] Verifying interpreter and torch..."
& $Python -c "import sys; print('Python:', sys.executable)"
& $Python -c "import torch; print('torch:', torch.__version__)"
& $Python -c "import ultralytics; print('ultralytics:', ultralytics.__version__)"

Set-Location $WebDir
Write-Host "[HCI_2.1] Starting: $Python -m uvicorn server:app"
Write-Host "[HCI_2.1] cwd=$WebDir"
& $Python -m uvicorn server:app --host 127.0.0.1 --port 8000
