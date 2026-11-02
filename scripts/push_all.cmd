@echo off
setlocal
REM Push this repo to BOTH GitHub (origin) and Gitee (gitee).
REM
REM First time:  scripts\push_all.cmd https://gitee.com/<你的用户名>/xingchen-ai.git
REM Later:       double-click this file (it reuses the saved gitee remote)
cd /d D:\dsh-demo\xingchen

if not "%~1"=="" (
  git remote remove gitee 2>nul
  git remote add gitee %~1
  echo [setup] gitee = %~1
)

echo.
echo === remotes ===
git remote -v
echo.

echo [1/2] push to GitHub (origin) ...
git push origin main
if errorlevel 1 echo [warn] GitHub push failed (network?), continuing with Gitee

echo.
echo [2/2] push to Gitee (gitee) ...
git push gitee main
if errorlevel 1 (
  echo [error] Gitee push failed - check the remote URL / token
) else (
  echo [ok] pushed to Gitee
)

echo.
pause
