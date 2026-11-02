@echo off
setlocal
REM XingChen generic trainer launcher with watchdog + panel
REM Usage:  start_train.cmd <config.yaml> <out_dir> [extra train.py args...]
REM Example: start_train.cmd configs\base-300m.yaml checkpoints-base-300m
REM Resumes automatically if <out_dir>\latest.pt exists. Ctrl+C or close window = stop.
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1

set CFG=%1
set OUT=%2
if "%CFG%"=="" set CFG=configs\base-wiki.yaml
if "%OUT%"=="" set OUT=checkpoints-run
shift
shift

echo ============================================================
echo   config : %CFG%
echo   out_dir: %OUT%
echo ============================================================
if not exist logs mkdir logs

echo [watchdog] starting (pause at 80C, kill at 90C)
start "xc-watch" /min cmd /c ""%PY%" -u scripts\temp_watch.py >> logs\watchdog.out.log 2>&1"

netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if errorlevel 1 (
  echo [panel] http://127.0.0.1:8765
  start "xc-mon" /min cmd /c ""%PY%" -u scripts\monitor.py --log %OUT%/train_log.jsonl >> logs\monitor.out.log 2>&1"
) else (
  echo [panel] 8765 already in use, skipped
)

set RESUME=
if exist "%OUT%\latest.pt" (
  echo [resume] found %OUT%\latest.pt, continuing from it
  set RESUME=--resume %OUT%\latest.pt
) else (
  echo [fresh] starting from scratch
)
echo.

"%PY%" -u scripts\train.py --config %CFG% --out_dir %OUT% --device cuda %RESUME% %1 %2 %3 %4
echo.
echo [training exited] press any key to close
pause >nul
