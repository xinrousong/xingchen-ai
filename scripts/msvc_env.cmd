@echo off
REM Enter the MSVC + CUDA build environment, then run python with the given arguments.
REM Usage:  scripts\msvc_env.cmd <script.py> [args...]
setlocal
set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [ERROR] vcvars64.bat failed - Build Tools missing?
  exit /b 1
)
set "PYDIR=D:\dsh-demo\ComfyUI_windows_portable\python_embeded"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%PYDIR%\Scripts;%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
REM python_embeded is an embeddable build: it ships no include/ or libs/.
REM Borrow headers + import libs from the same-version system CPython (3.13.14).
set "PYSYS=C:\Users\Xinrou\AppData\Local\Programs\Python\Python313"
if exist "%PYSYS%\include\Python.h" (
  set "INCLUDE=%PYSYS%\include;%INCLUDE%"
  set "LIB=%PYSYS%\libs;%LIB%"
) else (
  echo [WARN] system CPython headers not found at %PYSYS%
)
REM Force English compiler messages: torch decodes `cl` banner with the OEM code
REM page, and localized (Chinese) output makes it raise UnicodeDecodeError.
set "VSLANG=1033"
chcp 437 >nul
"%PYDIR%\python.exe" %*
