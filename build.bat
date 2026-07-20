@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ========================================
echo   STEM organizer - Windows build
echo ========================================
echo.

set "VENV=.build-venv"
set "PY=%VENV%\Scripts\python.exe"

echo [1/4] Checking Python ...
REM Windows Store alias makes "where python" succeed with a stub that is not a real interpreter.
python -c "import sys" >nul 2>&1
if errorlevel 1 goto no_python

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Wrong Python version:
    python --version
    echo Download 3.10 or 3.11 from here:
    echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo NOTE: During install, Add python.exe to PATH
    echo After installed, run build.bat again
    pause
    exit /b 1
)
python --version
echo.
goto python_ok

:no_python
echo Python was not found; Download and install it from here:
echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo NOTE: During install, Add python.exe to PATH
echo After installed, run build.bat again
pause
exit /b 1

:python_ok

echo [2/4] Preparing build environment ...
if not exist "%PY%" (
    echo   Creating %VENV% ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo   Installing PyInstaller and packager deps ...
"%PY%" -m pip install -q -U pip
"%PY%" -m pip install -q pyinstaller requests packaging customtkinter psutil
if errorlevel 1 (
    echo ERROR: Failed to install build dependencies.
    pause
    exit /b 1
)
"%PY%" --version
echo   Ready.
echo.

echo [3/4] Running PyInstaller ...
echo   Bundling UI into dist\STEM-organizer.exe
echo   This usually takes ~1 minute - output below:
echo.

"%PY%" -m PyInstaller --noconfirm --clean --log-level=INFO stem_organizer.spec 2>&1
if errorlevel 1 goto failed

echo.
echo [4/4] Finishing dist folder ...
"%PY%" -c "import sys; open('python-version.txt','w',encoding='utf-8').write(f'{sys.version_info[0]}.{sys.version_info[1]}\n')"
copy /Y install-deps.bat dist\install-deps.bat >nul
copy /Y python-version.txt dist\python-version.txt >nul
copy /Y verify_torch_install.py dist\verify_torch_install.py >nul

echo   Copying genre_gender_tagger\ ^(bundled tagger, no venv^) ...
if exist "dist\genre_gender_tagger" rmdir /S /Q "dist\genre_gender_tagger"
mkdir "dist\genre_gender_tagger" >nul
mkdir "dist\genre_gender_tagger\models" >nul
copy /Y "genre_gender_tagger\genre_gender_tagger.py" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\vocal_reverb.py" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\train_vocal_reverb.py" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\train-reverb.bat" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\install-deps.bat" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\run.bat" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\requirements.txt" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\readme.md" "dist\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\models\*.pb" copy /Y "genre_gender_tagger\models\*.pb" "dist\genre_gender_tagger\models\" >nul
if exist "genre_gender_tagger\models\vocal_reverb.pt" copy /Y "genre_gender_tagger\models\vocal_reverb.pt" "dist\genre_gender_tagger\models\" >nul

echo   Copying instrument_tagger\ ^(Rename Auto-detect, no venv^) ...
if exist "dist\instrument_tagger" rmdir /S /Q "dist\instrument_tagger"
mkdir "dist\instrument_tagger" >nul
copy /Y "instrument_tagger\instrument_tagger.py" "dist\instrument_tagger\" >nul
copy /Y "instrument_tagger\passt_mel.py" "dist\instrument_tagger\" >nul
copy /Y "instrument_tagger\install-deps.bat" "dist\instrument_tagger\" >nul

echo.
echo ========================================
echo   SUCCESS
echo ========================================
echo   dist\STEM-organizer.exe
echo   dist\install-deps.bat  ^(run this now^)
echo   dist\genre_gender_tagger\  + dist\instrument_tagger\
echo.
pause
exit /b 0

:failed
echo.
echo ========================================
echo   BUILD FAILED
echo ========================================
echo   See messages above.
echo.
pause
exit /b 1
