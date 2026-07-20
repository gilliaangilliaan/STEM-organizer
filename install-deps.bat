@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo STEM organizer - install dependencies
echo.
echo You need:
echo   - Python 3.10 or 3.11 on PATH ^(same version as the .exe if present^)
echo   - Internet ^(PyTorch, demucs, ffmpeg^)
echo.
echo You choose once: GPU or CPU PyTorch.
echo Then optional: Genre ^& Gender, Rename Auto-detect.
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not on PATH.
    echo Install 3.10/3.11 from https://www.python.org/downloads/ - tick "Add to PATH".
    pause
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo ERROR: need Python 3.10 or 3.11. Got:
    python --version
    echo Tip: py -3.11 "%~f0"
    pause
    exit /b 1
)

if exist "%~dp0python-version.txt" (
    python -c "import sys; from pathlib import Path; req=Path(r'%~dp0python-version.txt').read_text(encoding='utf-8').strip(); cur=f'{sys.version_info[0]}.{sys.version_info[1]}'; sys.exit(0 if cur==req else 1)"
    if errorlevel 1 (
        echo ERROR: Python mismatch. This folder expects:
        type "%~dp0python-version.txt"
        echo You have:
        python --version
        echo Tip: py -3.11 "%~f0"
        pause
        exit /b 1
    )
) else if exist "%~dp0STEM-organizer.exe" (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
    if errorlevel 1 (
        echo ERROR: this .exe needs Python 3.11. Tip: py -3.11 "%~f0"
        pause
        exit /b 1
    )
)

set "DEST=%~dp0site-packages"
if not exist "%DEST%" mkdir "%DEST%"

echo Python:
python --version
echo Install into: %DEST%
echo.

if exist "%DEST%\torch\__init__.py" (
    choice /C YN /N /M "site-packages already has PyTorch. Reinstall? [Y/N]: "
    if errorlevel 2 (
        echo Cancelled.
        pause
        exit /b 0
    )
    echo.
)

:ask_gpu
echo PyTorch build:
echo   1 = NVIDIA RTX 20/30/40  ^(CUDA 12.4^)
echo   2 = CPU only
echo   3 = NVIDIA RTX 50-series ^(CUDA 12.8^)
echo.
echo NOTE: Full installation incl. all dependencies should take ~9-13 minutes,
echo depending on your hardware ^& internet connection.
echo.
choice /C 123 /N /M "Enter 1, 2, or 3: "
if errorlevel 3 goto cuda_new_torch
if errorlevel 2 goto cpu_torch
if errorlevel 1 goto cuda_torch
goto ask_gpu

:cuda_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
set "TORCH_LABEL=CUDA 12.4"
set "STEM_GG_TORCH=1"
goto install_torch

:cuda_new_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "TORCH_LABEL=CUDA 12.8"
set "STEM_GG_TORCH=3"
goto install_torch

:cpu_torch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU"
set "STEM_GG_TORCH=2"
goto install_torch

:install_torch
REM Clean pip helper — system Python often has leftover ML pkgs (bs-roformer,
REM audiomentations, …) that make pip print fake conflicts on `pip install -t`.
set "PIP_VENV=%TEMP%\stem-organizer-pip-venv"
if not exist "%PIP_VENV%\Scripts\python.exe" (
    echo Preparing clean pip helper ...
    python -m venv "%PIP_VENV%"
    if errorlevel 1 goto failed
)
set "PIP_PY=%PIP_VENV%\Scripts\python.exe"
"%PIP_PY%" -m pip install -q -U pip
if errorlevel 1 goto failed

echo.
echo [1/4] PyTorch (%TORCH_LABEL%) ...
"%PIP_PY%" -m pip install torch --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install filelock "typing-extensions>=4.10" "setuptools>=77" "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo [2/4] audio + UI deps ...
"%PIP_PY%" -m pip install "cffi>=1.16" -t "%DEST%" --only-binary=:all: --upgrade --no-cache-dir
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install soundfile numpy sounddevice customtkinter psutil mutagen scipy librosa resampy audioread -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo [3/4] demucs ...
"%PIP_PY%" -m pip install omegaconf retrying submitit treetable cloudpickle colorama -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install dora-search -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install lameenc "einops>=0.8.0" julius openunmix pyyaml tqdm -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install demucs -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed

python -c "import sys; sys.path.insert(0, r'%DEST%'); import demucs" 2>nul
if errorlevel 1 (
    echo demucs needs torchaudio - installing ...
    "%PIP_PY%" -m pip install torchaudio --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
    if errorlevel 1 goto failed
)

if /I "%TORCH_LABEL%"=="CPU" (
    for /D %%D in ("%DEST%\torch-*.dist-info") do (
        echo %%~nxD | findstr /C:"+cpu" >nul || rmdir /S /Q "%%D" 2>nul
    )
)

echo [4/4] verify + ffmpeg ...
if /I "%TORCH_LABEL%"=="CPU" (
    python -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import torch; import soundfile; import demucs; v=torch.__version__; assert '+cpu' in v, f'expected CPU torch, got {v}'; print('OK torch', v)"
) else (
    python -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import torch; import soundfile; import demucs; print('OK torch', torch.__version__)"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    python "%~dp0verify_torch_install.py"
    if errorlevel 1 (
        echo WARNING: PyTorch may not match this GPU. App can still run on CPU.
        if /I "%TORCH_LABEL%"=="CUDA 12.4" echo For RTX 50-series: delete site-packages\ and re-run, pick 3.
    )
)

rmdir /S /Q "%DEST%\setuptools" 2>nul
rmdir /S /Q "%DEST%\pkg_resources" 2>nul
rmdir /S /Q "%DEST%\_distutils_hack" 2>nul
del /Q "%DEST%\distutils-precedence.pth" 2>nul
for /D %%D in ("%DEST%\setuptools-*") do rmdir /S /Q "%%~D" 2>nul

for %%F in ("%DEST%\_soundfile_data\libsndfile*.dll") do (
    if not exist "%DEST%\_soundfile_data\libsndfile.dll" copy /Y "%%F" "%DEST%\_soundfile_data\libsndfile.dll" >nul
)

python -c "import sys; v=f'{sys.version_info[0]}.{sys.version_info[1]}\n'; open(r'%DEST%\.python-version-used','w',encoding='utf-8').write(v); open(r'%~dp0python-version.txt','w',encoding='utf-8').write(v)"

set "FFMPEG_DIR=%~dp0ffmpeg"
if exist "%FFMPEG_DIR%\ffmpeg.exe" goto ffmpeg_done

set "FFMPEG_ZIP=%TEMP%\stem-organizer-ffmpeg.zip"
set "FFMPEG_URL=https://github.com/GyanD/codexffmpeg/releases/download/8.1/ffmpeg-8.1-essentials_build.zip"
set "FFMPEG_URL_FALLBACK=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

echo Downloading ffmpeg ...
where curl >nul 2>&1
if errorlevel 1 goto ffmpeg_download_ps

curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL%"
if errorlevel 1 curl -L --fail --progress-bar -o "%FFMPEG_ZIP%" "%FFMPEG_URL_FALLBACK%"
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
echo WARNING: ffmpeg download failed - some stems may not decode.
goto ffmpeg_done

:ffmpeg_extract
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-ffmpeg-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath $tmp -Force;" ^
  "$bin = (Get-ChildItem -Path $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1).Directory.FullName;" ^
  "if (-not $bin) { throw 'ffmpeg.exe not found in archive' };" ^
  "Copy-Item -Path (Join-Path $bin '*') -Destination '%FFMPEG_DIR%' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: ffmpeg extract failed - some stems may not decode.
    goto ffmpeg_done
)
del /Q "%FFMPEG_ZIP%" 2>nul

:ffmpeg_done

echo.
echo Core install OK (%TORCH_LABEL%). Start STEM-organizer.exe when finished.
echo.

if not exist "%~dp0genre_gender_tagger\install-deps.bat" goto after_gg
choice /C YN /N /M "Install Genre & Gender deps? [Y/N]: "
if errorlevel 2 goto skip_gg
set "STEM_GG_TORCH=1"
if /I "%TORCH_LABEL%"=="CPU" set "STEM_GG_TORCH=2"
if /I "%TORCH_LABEL%"=="CUDA 12.8" set "STEM_GG_TORCH=3"
set "STEM_GG_BUNDLED=1"
call "%~dp0genre_gender_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_GG_BUNDLED="
goto after_gg

:skip_gg
echo Skipped Genre ^& Gender.

:after_gg
if not exist "%~dp0instrument_tagger\install-deps.bat" goto after_inst
echo.
choice /C YN /N /M "Install Rename Auto-detect deps? [Y/N]: "
if errorlevel 2 goto skip_inst
set "STEM_INST_BUNDLED=1"
set "STEM_GG_TORCH=1"
if /I "%TORCH_LABEL%"=="CPU" set "STEM_GG_TORCH=2"
if /I "%TORCH_LABEL%"=="CUDA 12.8" set "STEM_GG_TORCH=3"
call "%~dp0instrument_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_INST_BUNDLED="
goto after_inst

:skip_inst
echo Skipped Rename Auto-detect.

:after_inst
echo.
echo All done.
echo Start STEM-organizer.exe in /dist folder
pause
exit /b 0

:failed
echo.
echo ERROR: install failed. Check messages above.
pause
exit /b 1
