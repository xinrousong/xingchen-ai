@echo off
setlocal
REM XingChen: start training WITH a live console window (watchdog + panel in background).
REM Double-click this file. Closing this window stops training (restart resumes from latest ckpt).
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
if not exist logs mkdir logs

REM guard: if a python process already holds the GPU, do not start a second training
set BUSY=
for /f "usebackq tokens=*" %%L in (`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2^>nul ^| findstr /i "python"`) do set BUSY=1
if defined BUSY (
  echo.
  echo [WARN] a python process is already using the GPU - probably training is running.
  echo        Close it first, then run this again.
  echo.
  pause
  exit /b 1
)

start "xc-watch" /min cmd /c ""%PY%" -u scripts\temp_watch.py >> logs\watchdog.out.log 2>&1"
start "xc-mon"   /min cmd /c ""%PY%" -u scripts\monitor.py --log checkpoints-base-wiki/train_log.jsonl >> logs\monitor.out.log 2>&1"

title XingChen base training  (close this window = stop)
echo ============================================================
echo   Live training output below (one line per 10 steps)
echo   Watchdog + panel started in background: http://127.0.0.1:8765
echo   Closing this window stops training; restart resumes.
echo ============================================================
echo.

"%PY%" -u scripts\train.py --config configs/base-wiki.yaml --resume checkpoints-base-wiki/latest.pt --device cuda

echo.
echo [training exited] press any key to close this window
pause >nul
