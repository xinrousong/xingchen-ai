@echo off
setlocal
REM XingChen M1.6: continue building preference pairs -> DPO -> eval -> cognitive check
REM Double-click this file. Keep the window open; stage 1 prints live progress.
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1
if not exist logs mkdir logs
if not exist data\dpo mkdir data\dpo

REM --- watchdog: pause training above 80C, kill above 90C (attaches to whatever holds the GPU)
start "xc-watch" /min cmd /c ""%PY%" -u scripts\temp_watch.py >> logs\watchdog.out.log 2>&1"

REM --- monitor panel on :8765 (skip if something already listens there)
netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if errorlevel 1 (
  start "xc-mon" /min cmd /c ""%PY%" -u scripts\monitor.py --log checkpoints-base-dpo/train_log.jsonl >> logs\monitor.out.log 2>&1"
  echo panel: http://127.0.0.1:8765  (shows the DPO curve once stage 2 starts)
) else (
  echo panel: already running on :8765, not starting a second one
)
echo.

echo ============================================================
echo   M1.6  pairs -> DPO -> eval
echo   stage 1 resumes from data\dpo\pairs.jsonl (skips done prompts)
echo   closing this window = stop (pairs already written are kept)
echo ============================================================
echo.

echo [1/4] building preference pairs - watch progress below
"%PY%" -u scripts\build_dpo_pairs.py --prompts 150 --samples 3 --max_new 60 --resume --out data\dpo\pairs.jsonl
if errorlevel 1 goto fail

echo.
echo [2/4] DPO training
"%PY%" -u scripts\train_dpo.py --init checkpoints-base-sft/latest.pt --pairs data\dpo\pairs.jsonl --out_dir checkpoints-base-dpo --epochs 2 --save_every 50 >> logs\m16_dpo.log 2>&1
if errorlevel 1 goto fail

echo [3/4] fixed exam sheet
"%PY%" -u scripts\eval_suite.py --ckpt checkpoints-base-dpo/latest.pt --sft --temperature 0 --max_new 60 >> logs\m16_eval.log 2>&1

echo [4/4] cognitive check
"%PY%" -u scripts\core_check.py --ckpt checkpoints-base-dpo/latest.pt >> logs\m16_check.log 2>&1

echo.
echo ============================================================
echo  DONE. results:
echo    logs\m16_eval.log    fixed exam sheet
echo    logs\m16_check.log   cognitive check
echo    logs\eval_history.log score history
echo ============================================================
pause
exit /b 0

:fail
echo.
echo FAILED - see logs\m16_dpo.log / logs\m16_eval.log
pause
exit /b 1
