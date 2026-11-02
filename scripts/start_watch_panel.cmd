@echo off
setlocal
REM Start ONLY the watchdog + monitor panel (does not touch any training job).
REM Use this after closing the editor/agent: training keeps running, but these two are lost.
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1
if not exist logs mkdir logs

echo starting watchdog (pause at 80C, kill at 90C)...
start "xc-watch" /min cmd /c ""%PY%" -u scripts\temp_watch.py >> logs\watchdog.out.log 2>&1"

netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if errorlevel 1 (
  echo starting monitor on http://127.0.0.1:8765 ...
  start "xc-mon" /min cmd /c ""%PY%" -u scripts\monitor.py --log checkpoints-base-dpo/train_log.jsonl >> logs\monitor.out.log 2>&1"
) else (
  echo monitor already listening on 8765, skipped
)

echo done. panel: http://127.0.0.1:8765
timeout /t 4 >nul
