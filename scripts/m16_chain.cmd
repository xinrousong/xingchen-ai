@echo off
setlocal
REM XingChen M1.6 chain: build preference pairs -> DPO -> eval -> cognitive check
REM Double-click this file. Total ~55 min on the 5060 laptop GPU.
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1
if not exist logs mkdir logs
if not exist data\dpo mkdir data\dpo

echo [%DATE% %TIME%] === M1.6 chain start === >> logs\m16_chain.log

echo [%DATE% %TIME%] stage 1/4: build preference pairs (300 prompts x 4 samples) >> logs\m16_chain.log
"%PY%" -u scripts\build_dpo_pairs.py --prompts 300 --samples 4 --max_new 72 --out data\dpo\pairs.jsonl >> logs\m16_pairs.log 2>&1
if errorlevel 1 (
  echo [%DATE% %TIME%] stage 1 FAILED >> logs\m16_chain.log
  exit /b 1
)

echo [%DATE% %TIME%] stage 2/4: DPO training >> logs\m16_chain.log
"%PY%" -u scripts\train_dpo.py --init checkpoints-base-sft/latest.pt --pairs data\dpo\pairs.jsonl --out_dir checkpoints-base-dpo --epochs 2 --save_every 100 >> logs\m16_dpo.log 2>&1
if errorlevel 1 (
  echo [%DATE% %TIME%] stage 2 FAILED >> logs\m16_chain.log
  exit /b 1
)

echo [%DATE% %TIME%] stage 3/4: fixed exam sheet >> logs\m16_chain.log
"%PY%" -u scripts\eval_suite.py --ckpt checkpoints-base-dpo/latest.pt --sft --temperature 0 --max_new 60 >> logs\m16_eval.log 2>&1

echo [%DATE% %TIME%] stage 4/4: cognitive check >> logs\m16_chain.log
"%PY%" -u scripts\core_check.py --ckpt checkpoints-base-dpo/latest.pt >> logs\m16_check.log 2>&1

echo [%DATE% %TIME%] === M1.6 chain DONE === >> logs\m16_chain.log
echo.
echo M1.6 finished. Results:
echo   logs\m16_eval.log   (fixed exam sheet)
echo   logs\m16_check.log  (cognitive check)
echo   logs\eval_history.log (score history, comparable across models)
timeout /t 8 >nul
