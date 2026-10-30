@echo off
setlocal
REM =====================================================================
REM  XingChen toolchain installer: MSVC Build Tools + CUDA Toolkit 13.4
REM  Run this file with "Run as administrator" (right click -> Run as administrator).
REM  Needed only for building PyTorch C++/CUDA extensions (fused MoE / MLA kernels).
REM  Size: Build Tools ~2.5 GB + CUDA ~4 GB. Takes 20-40 min on a normal link.
REM =====================================================================
net session >nul 2>&1
if errorlevel 1 (
  echo [ERROR] not running as administrator. Right click this file and choose
  echo         "Run as administrator".
  pause
  exit /b 1
)

echo [1/2] Visual Studio 2022 Build Tools (C++ workload)...
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --silent ^
  --accept-package-agreements --accept-source-agreements ^
  --override "--wait --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
if errorlevel 1 echo [WARN] BuildTools install returned %errorlevel%

echo [2/2] CUDA Toolkit 13.4 (nvcc)...
winget install --id Nvidia.CUDA --version 13.4 --exact --silent ^
  --accept-package-agreements --accept-source-agreements
if errorlevel 1 echo [WARN] CUDA install returned %errorlevel%

echo.
echo Done. Open a NEW terminal and check:
echo    cl        (should print MSVC banner)
echo    nvcc -V   (should print CUDA 13.x)
echo Then tell the assistant "toolchain ready".
pause
