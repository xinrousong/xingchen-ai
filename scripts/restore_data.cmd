@echo off
setlocal
REM Move training data back from E: to D: before starting a new training run.
REM (Cleanup moved train.bin / tokens.bin / mask.bin to E: to save D: space.)
cd /d D:\dsh-demo\xingchen
if not exist data\wiki mkdir data\wiki
if not exist data\sft mkdir data\sft

for %%F in (wiki\train.bin wiki\val.bin sft\tokens.bin sft\mask.bin) do (
  if exist "E:\xingchen\data\%%F" (
    if not exist "data\%%F" (
      move "E:\xingchen\data\%%F" "data\%%F" >nul
      echo [restore] %%F
    )
  )
)
echo.
echo data now:
dir /b data\wiki data\sft
pause
