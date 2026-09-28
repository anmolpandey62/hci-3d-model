@echo off
REM HCI_2.1 FastAPI — ALWAYS use conda improved_model_train (working PyTorch).
REM Do NOT use web_file_v2\.venv or Anaconda base — both fail on c10.dll.

setlocal
@REM set "PYTHON=C:\Users\DELL\anaconda3\envs\improved_model_train\python.exe"
set "PYTHON=C:\Users\anmol\miniconda3\envs\improved_model_train\python.exe"
set "HCI21_MODEL_PATH=C:\Users\anmol\OneDrive\project_off\hci-3d-model-vaishnavi\hci-3d-model-vaishnavi\best.pt"
@REM set "HCI21_MODEL_PATH=D:\HCI_interor\cubicasa_hqa_500\runs\hqa500_offline\weights\best.pt"
REM Clear inherited venv so child process cannot prefer web_file_v2\.venv
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

if not exist "%PYTHON%" (
  echo ERROR: improved_model_train env not found at:
  echo   %PYTHON%
  echo Create/activate it, or install torch there before starting HCI_2.1.
  exit /b 1
)

echo [HCI_2.1] Verifying interpreter and torch...
"%PYTHON%" -c "import sys; print('Python:', sys.executable)" || exit /b 1
"%PYTHON%" -c "import torch; print('torch:', torch.__version__)" || (
  echo ERROR: torch import failed in improved_model_train
  exit /b 1
)
"%PYTHON%" -c "import ultralytics; print('ultralytics:', ultralytics.__version__)" || (
  echo ERROR: ultralytics import failed in improved_model_train
  exit /b 1
)

cd /d "%~dp0..\web"
echo [HCI_2.1] Starting server with:
echo   %PYTHON%
echo   cwd=%CD%
echo   HCI21_MODEL_PATH=%HCI21_MODEL_PATH%
"%PYTHON%" -m uvicorn server:app --host 127.0.0.1 --port 8000
endlocal
