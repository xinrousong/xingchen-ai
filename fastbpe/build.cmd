@echo off
REM 重新编译 C 版 BPE 引擎（需要 MinGW gcc；本机在 D:\mingw64\bin）
setlocal
cd /d D:\dsh-demo\xingchen
set GCC=D:\mingw64\bin\gcc.exe
"%GCC%" -O3 -shared -static-libgcc -o xingchen\fastbpe.dll fastbpe\bpe.c
if errorlevel 1 (
  echo [FAILED] build failed
  exit /b 1
)
echo [OK] xingchen\fastbpe.dll rebuilt
echo verify:  python fastbpe\verify.py   (see logs\fastbpe_check.txt)
