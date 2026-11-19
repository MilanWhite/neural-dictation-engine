@echo off
rem One-time setup: Python environment + models. Requires Python 3.12 and an NVIDIA GPU.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    py -3.12 -m venv .venv || python -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" download_models.py || goto :fail
echo.
echo Setup complete. Double-click "Start Dictation.vbs" to run.
pause
exit /b 0
:fail
echo.
echo Setup failed.
pause
exit /b 1
