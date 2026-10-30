@echo off
setlocal
REM XingChen M1.5 stage 2+3: memory top-up -> cognitive check
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1

echo [%DATE% %TIME%] === stage2 chain start === >> logs\m15_chain.log

"%PY%" -u scripts\train.py --config configs/base-wiki.yaml --resume checkpoints-base-sft/latest.pt --out_dir checkpoints-base-sft --train_bin data/wiki/core_topup.bin --lr 3.0e-4 --max_iters 10200 --device cuda >> logs\m15_topup2.log 2>&1
if errorlevel 1 (
  echo [%DATE% %TIME%] stage 2 FAILED (rc=%errorlevel%) >> logs\m15_chain.log
  exit /b 1
)

echo [%DATE% %TIME%] stage 3/3: cognitive check >> logs\m15_chain.log
"%PY%" -u scripts\core_check.py --ckpt checkpoints-base-sft/latest.pt >> logs\m15_check.log 2>&1

echo [%DATE% %TIME%] === M1.5 chain DONE === >> logs\m15_chain.log
