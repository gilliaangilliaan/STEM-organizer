@echo off
setlocal
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
  set "PY=venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo Training vocal dry/wet reverb model...
echo   dry:  reverb_data\dry
echo   wet:  reverb_data\wet
echo   out:  models\vocal_reverb.pt
echo.
"%PY%" train_vocal_reverb.py %*
if errorlevel 1 (
  echo.
  echo FAILED
  pause
  exit /b 1
)
echo.
pause
endlocal
