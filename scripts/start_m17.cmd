@echo off
setlocal
REM XingChen: mixed training (60% SFT / 20% core memory / 20% pretrain) + eval
REM Double-click. Keep the window open. Watchdog + panel start automatically.
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1
set OUT=checkpoints-sft-mix
if not exist logs mkdir logs

echo ============================================================
echo   Mixed training: 60%% SFT / 20%% core / 20%% pretrain
echo   init = checkpoints-sft-v3 (healthy base, val 3.35)
echo   3000 steps, lr 2e-5  -- roughly 15-70 min depending on GPU load
echo ============================================================
echo.

echo [watchdog] starting (pause >=80C, kill >=90C)
start "xc-watch" /min cmd /c ""%PY%" -u scripts\temp_watch.py >> logs\watchdog.out.log 2>&1"

netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if errorlevel 1 (
  echo [panel] http://127.0.0.1:8765
  start "xc-mon" /min cmd /c ""%PY%" -u scripts\monitor.py --log %OUT%/train_log.jsonl >> logs\monitor.out.log 2>&1"
) else (
  echo [panel] 8765 already in use, skipped
)
echo.

"%PY%" -u scripts\train_sft.py --init checkpoints-sft-v3/latest.pt --out_dir %OUT% ^
  --batch_size 2 --max_iters 3000 --lr 2e-5 ^
  --p_core 0.2 --p_pretrain 0.2 --eval_every 500 --device cuda
if errorlevel 1 goto fail

echo.
echo [eval 1/2] cognitive check
"%PY%" -u scripts\core_check.py --ckpt %OUT%/latest.pt >> logs\mix_check.log 2>&1

echo [eval 2/2] fixed exam sheet
"%PY%" -u scripts\eval_suite.py --ckpt %OUT%/latest.pt --sft --temperature 0 --max_new 60 >> logs\mix_eval.log 2>&1

echo.
echo ============================================================
echo  DONE
echo    logs\mix_check.log   cognitive check (14 questions)
echo    logs\mix_eval.log    fixed exam sheet (35 questions)
echo    logs\eval_history.log  score history
echo ============================================================
pause
exit /b 0

:fail
echo.
echo FAILED - see the console output above
pause
exit /b 1
