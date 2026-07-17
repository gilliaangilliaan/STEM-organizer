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
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo ERROR: Build requires Python 3.10.x or 3.11.x on PATH.
    pause
    exit /b 1
)
python --version
echo.

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
echo   This usually takes 1-3 minutes - output below:
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
copy /Y "genre_gender_tagger\install-deps.bat" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\run.bat" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\requirements.txt" "dist\genre_gender_tagger\" >nul
copy /Y "genre_gender_tagger\readme.md" "dist\genre_gender_tagger\" >nul
if exist "genre_gender_tagger\models\*.pb" copy /Y "genre_gender_tagger\models\*.pb" "dist\genre_gender_tagger\models\" >nul

echo.
echo ========================================
echo   SUCCESS
echo ========================================
echo   dist\STEM-organizer.exe
echo   dist\install-deps.bat  ^(run once for PyTorch/Demucs^)
echo   dist\genre_gender_tagger\  ^(run its install-deps.bat for Genre ^& Gender^)
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
