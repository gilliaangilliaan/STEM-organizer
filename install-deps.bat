@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.10.x or 3.11.x from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this script again.
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: install-deps.bat needs Python 3.10.x or 3.11.x on PATH.
    echo.
    python --version
    echo.
    echo Packages must match the .exe embedded Python. 3.12+ will not work.
    echo If multiple Pythons are installed: py -3.10 install-deps.bat  or  py -3.11 install-deps.bat
    pause
    exit /b 1
)

if exist "%~dp0python-version.txt" (
    python -c "import sys; from pathlib import Path; req=Path(r'%~dp0python-version.txt').read_text(encoding='utf-8').strip(); cur=f'{sys.version_info[0]}.{sys.version_info[1]}'; sys.exit(0 if cur==req else 1)"
    if errorlevel 1 (
        echo.
        echo ERROR: Python version mismatch for this app folder.
        echo.
        type "%~dp0python-version.txt"
        echo   ^<-- required for the .exe beside this script
        echo.
        python --version
        echo.
        echo Run install-deps.bat with that Python, e.g. py -3.10 install-deps.bat
        pause
        exit /b 1
    )
) else if exist "%~dp0STEM-organizer.exe" (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
    if errorlevel 1 (
        echo.
        echo ERROR: This .exe was built with Python 3.11.x.
        echo Run: py -3.11 install-deps.bat
        echo Or rebuild the .exe with build.bat using your Python 3.10.
        pause
        exit /b 1
    )
)

set "DEST=%~dp0site-packages"
if not exist "%DEST%" mkdir "%DEST%"

echo Installing STEM organizer dependencies into:
echo   %DEST%
echo.
echo Using:
python --version
echo.

if exist "%DEST%\torch\__init__.py" (
    echo site-packages already contains a PyTorch install.
    echo This script reinstalls/upgrades all dependencies each time it runs.
    echo.
    choice /C YN /N /M "Continue with reinstall? [Y/N]: "
    if errorlevel 2 (
        echo Cancelled.
        pause
        exit /b 0
    )
    echo.
)

echo This is a one-time setup ^(or re-run when changing CPU/GPU PyTorch^).
echo.

:ask_gpu
echo Which PyTorch build do you need?
echo   1 = NVIDIA GPU - RTX 20/30/40 series ^(CUDA 12.4 / cu124^)
echo   2 = No NVIDIA GPU - CPU only
echo   3 = NVIDIA GPU - RTX 50-series / Blackwell ^(CUDA 12.8 / cu128, e.g. 5090^)
echo.
choice /C 123 /N /M "Enter 1, 2, or 3: "
if errorlevel 3 goto cuda_new_torch
if errorlevel 2 goto cpu_torch
if errorlevel 1 goto cuda_torch
goto ask_gpu

:cuda_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "TORCH_LABEL=CUDA 12.4 (RTX 20/30/40)"
goto install_torch

:cuda_new_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_LABEL=CUDA 12.8 (RTX 50-series)"
goto install_torch

:cpu_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU"
goto install_torch

:install_torch
python -m pip install --upgrade pip
if errorlevel 1 goto failed

echo.
echo [1/3] PyTorch (%TORCH_LABEL%) ...
python -m pip install torch torchaudio --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo.
echo [2/3] cffi ^(needed by soundfile^) ...
python -m pip install "cffi>=1.16" -t "%DEST%" --only-binary=:all: --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo.
echo [3/3] demucs, soundfile, numpy, sounddevice ...
python -m pip install soundfile numpy sounddevice -t "%DEST%" --upgrade --no-cache-dir
echo Installing Pair Finder dependencies (mutagen, scipy, librosa)...
python -m pip install mutagen scipy librosa resampy audioread -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed
python -m pip install omegaconf retrying submitit treetable cloudpickle colorama -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed
python -m pip install dora-search -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
python -m pip install lameenc einops julius openunmix pyyaml tqdm -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
python -m pip install demucs -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed

if /I "%TORCH_LABEL%"=="CPU" (
    for /D %%D in ("%DEST%\torch-*.dist-info") do (
        echo %%~nxD | findstr /C:"+cpu" >nul || rmdir /S /Q "%%D" 2>nul
    )
)

echo.
echo Verifying install ...
if /I "%TORCH_LABEL%"=="CPU" (
    python -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import torch; import soundfile; import demucs; v=torch.__version__; assert '+cpu' in v, f'expected CPU torch, got {v}'; print('OK - torch', v)"
) else (
    python -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import torch; import soundfile; import demucs; print('OK - torch', torch.__version__)"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    echo.
    echo Checking GPU compatibility ...
    python "%~dp0verify_torch_install.py"
    if errorlevel 1 (
        echo.
        echo WARNING: Installed PyTorch cannot run on your GPU.
        if /I "%TORCH_LABEL%"=="CUDA 12.4 (RTX 20/30/40)" (
            echo For RTX 5090 / 50-series, delete site-packages\ and re-run this script with option 3.
        ) else (
            echo Try option 2 ^(CPU^) or update NVIDIA drivers, then re-run this script.
        )
        echo STEM organizer will still run on CPU until PyTorch matches your GPU.
        echo.
    )
)

echo.
echo Removing setuptools metadata ^(not needed at runtime^) ...
rmdir /S /Q "%DEST%\setuptools" 2>nul
rmdir /S /Q "%DEST%\pkg_resources" 2>nul
rmdir /S /Q "%DEST%\_distutils_hack" 2>nul
del /Q "%DEST%\distutils-precedence.pth" 2>nul
for /D %%D in ("%DEST%\setuptools-*") do rmdir /S /Q "%%~D" 2>nul

echo.
echo Preparing libsndfile DLL alias ...
for %%F in ("%DEST%\_soundfile_data\libsndfile*.dll") do (
    if not exist "%DEST%\_soundfile_data\libsndfile.dll" copy /Y "%%F" "%DEST%\_soundfile_data\libsndfile.dll" >nul
)

python -c "import sys; v=f'{sys.version_info[0]}.{sys.version_info[1]}\n'; open(r'%DEST%\.python-version-used','w',encoding='utf-8').write(v); open(r'%~dp0python-version.txt','w',encoding='utf-8').write(v)"

echo.
echo Installing ffmpeg into ffmpeg\ beside this app ...
set "FFMPEG_DIR=%~dp0ffmpeg"
if exist "%FFMPEG_DIR%\ffmpeg.exe" (
    echo ffmpeg already present:
    echo   %FFMPEG_DIR%
    goto ffmpeg_done
)

set "FFMPEG_ZIP=%TEMP%\stem-organizer-ffmpeg.zip"
set "FFMPEG_URL=https://github.com/GyanD/codexffmpeg/releases/download/8.1/ffmpeg-8.1-essentials_build.zip"
set "FFMPEG_URL_FALLBACK=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

echo Downloading ffmpeg essentials (GitHub mirror) ...
echo   %FFMPEG_URL%
echo.

where curl >nul 2>&1
if errorlevel 1 goto ffmpeg_download_ps

curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL%"
if errorlevel 1 (
    echo.
    echo Primary mirror failed, trying gyan.dev ...
    echo   %FFMPEG_URL_FALLBACK%
    echo.
    curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL_FALLBACK%"
)
if not errorlevel 1 goto ffmpeg_download_ok
goto ffmpeg_download_failed

:ffmpeg_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$urls=@('%FFMPEG_URL%','%FFMPEG_URL_FALLBACK%');" ^
  "$out='%FFMPEG_ZIP%';" ^
  "$ok=$false;" ^
  "foreach ($url in $urls) {" ^
  "  try {" ^
  "    Write-Host ('Downloading ' + $url);" ^
  "    $wc = New-Object System.Net.WebClient;" ^
  "    $wc.DownloadFile($url, $out);" ^
  "    $ok=$true; break" ^
  "  } catch { Write-Host $_.Exception.Message }" ^
  "};" ^
  "if (-not $ok) { exit 1 }"
if errorlevel 1 goto ffmpeg_download_failed

:ffmpeg_download_ok
if not exist "%FFMPEG_ZIP%" goto ffmpeg_download_failed
goto ffmpeg_extract

:ffmpeg_download_failed
echo WARNING: ffmpeg download failed. STEM organizer will still run, but some stems may not decode.
goto ffmpeg_done

:ffmpeg_extract

if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"

echo Extracting ffmpeg ...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-ffmpeg-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath $tmp -Force;" ^
  "$bin = (Get-ChildItem -Path $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1).Directory.FullName;" ^
  "if (-not $bin) { throw 'ffmpeg.exe not found in archive' };" ^
  "Copy-Item -Path (Join-Path $bin '*') -Destination '%FFMPEG_DIR%' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"

if errorlevel 1 (
    echo WARNING: ffmpeg extract failed. STEM organizer will still run, but some stems may not decode.
    goto ffmpeg_done
)

del /Q "%FFMPEG_ZIP%" 2>nul

echo Installed:
echo   %FFMPEG_DIR%

:ffmpeg_done

echo.
echo Done. You can start STEM-organizer.exe now.
echo Or run from source: python stem_organizer_ui.py ^(same Python as above^).
echo Demucs models download on first run into torch_home\ beside this app.
echo.

if /I "%TORCH_LABEL%"=="CPU" (
    echo Installed CPU PyTorch. Re-run this script if you later add an NVIDIA GPU.
) else (
    echo Installed %TORCH_LABEL% PyTorch. Restart STEM organizer after install.
)

echo.
pause
exit /b 0

:failed
echo.
echo ERROR: install failed. See messages above.
echo Use Python 3.10.x or 3.11.x matching your .exe, and keep install-deps.bat next to the app.
pause
exit /b 1
