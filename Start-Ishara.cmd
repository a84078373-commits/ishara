@echo off
cd /d "%~dp0"
echo.
echo  Ishara Flask server
echo  Open http://127.0.0.1:5173 in Chrome or Edge
echo  Keep this window open.
echo.
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtualenv...
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
start "" "http://127.0.0.1:5173"
.venv\Scripts\python.exe app.py
pause
