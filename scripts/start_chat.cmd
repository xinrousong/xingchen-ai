@echo off
setlocal
REM XingChen chat: double-click to talk to the model (checkpoints-sft-mix2)
cd /d D:\dsh-demo\xingchen
set PY=D:\dsh-demo\ComfyUI_windows_portable\python_embeded\python.exe
set PYTHONUTF8=1

echo ============================================================
echo   XingChen chat  (166M, sft-mix2)
echo   type your message and press Enter; Ctrl+C to quit
echo ============================================================
echo.

"%PY%" scripts\chat.py --ckpt checkpoints-sft-mix2\latest.pt --sft

echo.
echo [chat exited]
pause
