@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM =============================================================================
REM STEM organizer (PySide6) - install dependencies
REM
REM Double-click or run from project root / dist folder.
REM
REM PyTorch choice (same as CTk STEM-organizer):
REM   1 = NVIDIA RTX 20/30/40  -> CUDA 12.4  (https://download.pytorch.org/whl/cu124)
REM   2 = CPU only             -> CPU wheels (https://download.pytorch.org/whl/cpu)
REM   3 = NVIDIA RTX 50-series -> CUDA 12.8  (https://download.pytorch.org/whl/cu128)
REM
REM =============================================================================

echo.
echo STEM organizer (PySide6) - install dependencies
echo.
echo You need:
echo   - Python 3.10 or 3.11 on PATH ^(same version as the .exe if present^)
echo   - Internet ^(PyTorch, demucs, ffmpeg, mp3val, flac, tagger models^)
echo.
echo You choose once: GPU or CPU PyTorch.
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not on PATH.
    echo Install 3.10/3.11 from https://www.python.org/downloads/ - tick "Add to PATH".
    pause
    exit /b 1
)

REM Resolve host Python once (full path). Frozen + source installs use this.
set "HOST_PY="
for /f "delims=" %%P in ('where python 2^>nul') do (
    set "HOST_PY=%%P"
    goto host_py_found
)
:host_py_found
if not defined HOST_PY (
    echo ERROR: python not on PATH.
    pause
    exit /b 1
)
REM Avoid nested quotes around "%HOST_PY%" inside for /f (breaks cmd parsing).
for /f "tokens=1,2" %%A in ('python -c "import sys; print(sys.version_info[0], sys.version_info[1])"') do set "HOST_VER=%%A.%%B"
if not defined HOST_VER (
    echo ERROR: could not read Python version from:
    echo   %HOST_PY%
    pause
    exit /b 1
)

"%HOST_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,10),(3,11)) else 1)"
if errorlevel 1 (
    echo ERROR: need Python 3.10 or 3.11. Got:
    "%HOST_PY%" --version
    echo Install from https://www.python.org/downloads/ - tick Add to PATH.
    echo Then re-run with that interpreter, e.g. py -3.10 "%~f0"
    pause
    exit /b 1
)

set "REQ_VER="
if exist "%~dp0python-version.txt" (
    for /f "usebackq delims=" %%R in ("%~dp0python-version.txt") do set "REQ_VER=%%R"
)
if defined REQ_VER (
    if /I not "%HOST_VER%"=="%REQ_VER%" (
        echo ERROR: Python mismatch. This folder expects Python %REQ_VER%.
        echo You have: %HOST_VER%  ^(%HOST_PY%^)
        echo Install Python %REQ_VER% from https://www.python.org/downloads/ - tick Add to PATH.
        echo Then re-run: py -%REQ_VER% "%~f0"
        pause
        exit /b 1
    )
) else if exist "%~dp0STEM-organizer.exe" (
    REM Legacy unmarked builds default to 3.11 ABI (see deps_bootstrap.LEGACY_PREBUILT_PYTHON).
    if /I not "%HOST_VER%"=="3.11" (
        echo ERROR: this .exe needs Python 3.11 ^(no python-version.txt; legacy default^).
        echo You have: %HOST_VER%  ^(%HOST_PY%^)
        echo Install Python 3.11 from https://www.python.org/downloads/ - tick Add to PATH.
        echo Then re-run: py -3.11 "%~f0"
        echo Do not point at a missing Python311 path - install 3.11 first.
        pause
        exit /b 1
    )
)

REM --- Detect GPU hint (nvidia-smi); user still picks 1/2/3 ---
REM Older nvidia-smi may reject csv,noheader and print ERROR on stdout;
REM never treat that text as a GPU name.
set "GPU_HINT=no NVIDIA GPU detected - option 2 (CPU) is safest"
set "GPU_NAME="
where nvidia-smi >nul 2>&1
if errorlevel 1 goto gpu_hint_done

REM 1) Modern: csv,noheader
for /f "tokens=* delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
    set "GPU_NAME=%%G"
    goto gpu_validate_modern
)
goto gpu_try_csv

:gpu_validate_modern
call :gpu_name_ok
if errorlevel 1 (
    set "GPU_NAME="
    goto gpu_try_csv
)
goto gpu_named

:gpu_try_csv
REM 2) csv with header - skip first row
for /f "skip=1 tokens=* delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv 2^>nul') do (
    set "GPU_NAME=%%G"
    goto gpu_validate_csv
)
goto gpu_try_plain

:gpu_validate_csv
call :gpu_name_ok
if errorlevel 1 (
    set "GPU_NAME="
    goto gpu_try_plain
)
goto gpu_named

:gpu_try_plain
REM 3) Plain nvidia-smi - first product-like line (findstr avoids | pipes in echo)
for /f "tokens=* delims=" %%G in ('nvidia-smi 2^>nul ^| findstr /I "GeForce Quadro Tesla Titan RTX GTX"') do (
    set "GPU_NAME=%%G"
    goto gpu_named
)
goto gpu_name_unknown

:gpu_name_unknown
REM nvidia-smi on PATH but no reliable name - do not echo error text as GPU
set "GPU_HINT=could not read GPU name - option 2 (CPU) is safest; NVIDIA: try 1 (CUDA 12.4) or 3 for 50-series"
goto gpu_hint_done

:gpu_named
REM Strip table pipes so echo/findstr are safe
set "GPU_NAME=%GPU_NAME:|=%"
call :gpu_name_ok
if errorlevel 1 (
    set "GPU_NAME="
    goto gpu_name_unknown
)
echo Detected GPU: %GPU_NAME%
echo %GPU_NAME% | findstr /I /C:"RTX 50" >nul
if not errorlevel 1 (
    set "GPU_HINT=RTX 50-series detected - prefer option 3 (CUDA 12.8)"
    goto gpu_hint_done
)
echo %GPU_NAME% | findstr /I /C:"RTX 40" /C:"RTX 30" /C:"RTX 20" >nul
if not errorlevel 1 (
    set "GPU_HINT=RTX 20/30/40 detected - prefer option 1 (CUDA 12.4)"
    goto gpu_hint_done
)
REM Other NVIDIA (e.g. GTX / A-series): CUDA 12.4 usually works
set "GPU_HINT=NVIDIA GPU detected - option 1 (CUDA 12.4) usually works; 50-series needs 3"

:gpu_hint_done
echo Hint: %GPU_HINT%
echo.

REM --- Destination: site-packages beside exe, else project .venv ---
REM NOTE: set DEST inside IF (...), then use %DEST% AFTER the block ends
REM (cmd expands %VAR% at parse time inside parentheses).
REM
REM Frozen path contract (must match deps_bootstrap.external_site_dirs):
REM   install here:  <folder-with-STEM-organizer.exe>\site-packages\
REM   exe looks in:  Path(sys.executable).parent / "site-packages"
REM Put install-deps.bat in the SAME folder as STEM-organizer.exe (the
REM COLLECT output folder, e.g. dist\STEM-organizer\), then run it there.
set "USE_SITE=0"
set "DEST="
set "PY="
if exist "%~dp0STEM-organizer.exe" set "USE_SITE=1"

if "%USE_SITE%"=="1" (
    set "DEST=%~dp0site-packages"
    set "PY=%HOST_PY%"
) else (
    if exist "%~dp0.venv\Scripts\python.exe" (
        "%~dp0.venv\Scripts\python.exe" -c "import sys; v='%HOST_VER%'.split('.'); raise SystemExit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" 1>nul 2>nul
        if errorlevel 1 (
            echo Existing .venv is broken or wrong Python - recreating with %HOST_VER% ...
            rmdir /S /Q "%~dp0.venv" 2>nul
        )
    )
    if not exist "%~dp0.venv\Scripts\python.exe" (
        echo Creating .venv ...
        "%HOST_PY%" -m venv "%~dp0.venv"
        if errorlevel 1 (
            echo ERROR: failed to create .venv
            pause
            exit /b 1
        )
    )
    set "PY=%~dp0.venv\Scripts\python.exe"
    set "DEST=%~dp0.venv\Lib\site-packages"
)

if not defined DEST goto bad_dest
if "%DEST%"=="" goto bad_dest
if not exist "%DEST%" (
    mkdir "%DEST%"
    if errorlevel 1 (
        echo ERROR: could not create "%DEST%"
        pause
        exit /b 1
    )
)
if not exist "%DEST%" (
    echo ERROR: install target missing: "%DEST%"
    pause
    exit /b 1
)

if "%USE_SITE%"=="1" (
REM    echo Mode: frozen / dist - install into site-packages
) else (
REM    echo Mode: source - install into .venv
)
REM echo Install into: %DEST%
REM echo Python: %HOST_PY%  ^(%HOST_VER% detected^)
REM if /I not "%USE_SITE%"=="1" echo Venv: %PY%
REM echo.
goto dest_ok

:bad_dest
echo ERROR: install target path is empty. Aborting.
pause
exit /b 1

:dest_ok

if "%USE_SITE%"=="1" if exist "%DEST%\torch\__init__.py" (
    choice /C YN /N /M "site-packages already has PyTorch. Reinstall? [Y/N]: "
    if errorlevel 2 (
        echo Keeping existing PyTorch / core packages.
        echo Still ensuring torchvision/torchaudio + Genre ^& Gender + Rename Auto-detect...
        set "TORCH_LABEL=existing"
        set "STEM_GG_TORCH=2"
        call :detect_torch_index
        call :ensure_pip_helper
        if errorlevel 1 goto failed
        call :ensure_torch_extras
        if errorlevel 1 goto failed
        goto ffmpeg_section
    )
    echo.
)

:ask_gpu
echo PyTorch build:
echo   1 = NVIDIA RTX 20/30/40  ^(CUDA 12.4^)
echo   2 = CPU only
echo   3 = NVIDIA RTX 50-series ^(CUDA 12.8^)
echo.
echo NOTE: Full installation incl. all dependencies should take ~10-15 minutes,
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
if "%USE_SITE%"=="1" goto install_site
goto install_venv

REM ---------- SOURCE: install into .venv ----------
:install_venv
set "PY=%~dp0.venv\Scripts\python.exe"

echo.
echo [1/4] PyTorch (%TORCH_LABEL%) ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed
REM Install torch + torchaudio from the chosen index FIRST. demucs hard-requires
REM torchaudio (demucs/audio.py: "import torchaudio as ta"), so both must be
REM present before demucs is resolved. Installing them first also means demucs
REM won't drag in a different-flavored torch wheel from PyPI - the default
REM only-if-needed upgrade strategy leaves the already-satisfied torch alone.
"%PY%" -m pip install torch torchaudio --index-url %TORCH_INDEX% --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo [2/4] Core GUI + audio deps ...
"%PY%" -m pip install -r "%~dp0requirements.txt" --upgrade
if errorlevel 1 goto failed

echo [3/4] verify ...
if /I "%TORCH_LABEL%"=="CPU" (
    "%PY%" -c "import torch, soundfile, scipy, flac_detective, demucs; v=torch.__version__; assert '+cpu' in v or 'cpu' in v.lower(), f'expected CPU torch, got {v}'; print('OK torch', v)"
) else (
    "%PY%" -c "import torch, soundfile, scipy, flac_detective, demucs; print('OK torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    "%PY%" "%~dp0verify_torch_install.py"
    if errorlevel 1 (
        echo WARNING: PyTorch may not match this GPU. App can still run on CPU.
        if /I "%TORCH_LABEL%"=="CUDA 12.4" echo For RTX 50-series: re-run install-deps.bat and pick 3.
    )
)

"%PY%" -c "import sys; v=f'{sys.version_info[0]}.{sys.version_info[1]}\n'; open(r'%~dp0python-version.txt','w',encoding='utf-8').write(v)"
goto ffmpeg_section

REM ---------- FROZEN: install into site-packages (CTk pattern) ----------
:install_site
REM Clean pip helper - system Python often has leftover ML pkgs that make
REM pip print fake conflicts on `pip install -t`.
if not defined DEST goto bad_dest
if "%DEST%"=="" goto bad_dest
call :ensure_pip_helper
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install -q -U pip
if errorlevel 1 goto failed

echo.
echo [1/4] PyTorch (%TORCH_LABEL%) ...
"%PIP_PY%" -m pip install torch --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
REM torchvision (hear21passt/timm) + torchaudio (demucs) from same CUDA/CPU index
"%PIP_PY%" -m pip install torchvision torchaudio --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
if errorlevel 1 goto failed
REM Pillow: torchvision imports PIL; --no-deps above skips it
"%PIP_PY%" -m pip install Pillow -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install filelock "typing-extensions>=4.10" "setuptools>=77" "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo [2/4] audio + UI deps ...
"%PIP_PY%" -m pip install "cffi>=1.16" -t "%DEST%" --only-binary=:all: --upgrade --no-cache-dir
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install soundfile numpy sounddevice PySide6 "PySide6-Fluent-Widgets>=1.11,<2" psutil mutagen scipy librosa soxr audioread requests -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 goto failed
"%PIP_PY%" -m pip install flac-detective==1.7.0 -t "%DEST%" --upgrade --no-cache-dir
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

if /I "%TORCH_LABEL%"=="CPU" (
    for /D %%D in ("%DEST%\torch-*.dist-info") do (
        echo %%~nxD | findstr /C:"+cpu" >nul || rmdir /S /Q "%%D" 2>nul
    )
)

echo [4/4] verify ...
if /I "%TORCH_LABEL%"=="CPU" (
    "%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import PIL; import torch; import torchvision; import torchaudio; import soundfile; import scipy; import flac_detective; import demucs; v=torch.__version__; assert '+cpu' in v, f'expected CPU torch, got {v}'; print('OK torch', v, 'torchvision', torchvision.__version__, 'PIL', PIL.__version__)"
) else (
    "%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import _cffi_backend; import PIL; import torch; import torchvision; import torchaudio; import soundfile; import scipy; import flac_detective; import demucs; print('OK torch', torch.__version__, 'torchvision', torchvision.__version__, 'PIL', PIL.__version__)"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    "%HOST_PY%" "%~dp0verify_torch_install.py"
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

"%HOST_PY%" -c "import sys; v=f'{sys.version_info[0]}.{sys.version_info[1]}\n'; open(r'%DEST%\.python-version-used','w',encoding='utf-8').write(v); open(r'%~dp0python-version.txt','w',encoding='utf-8').write(v)"

:ffmpeg_section
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

REM --- mp3val (Detect corruption Fix / Fast second opinion) ---
set "MP3VAL_DIR=%~dp0mp3val"
if exist "%MP3VAL_DIR%\mp3val.exe" goto mp3val_done

set "MP3VAL_ZIP=%TEMP%\stem-organizer-mp3val.zip"
set "MP3VAL_URL=https://downloads.sourceforge.net/project/mp3val/mp3val-bundle/MP3val%%200.1.8%%20with%%20MP3val-frontend%%200.1.1%%20included/mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip"
set "MP3VAL_URL_FALLBACK=https://sourceforge.net/projects/mp3val/files/mp3val-bundle/MP3val%%200.1.8%%20with%%20MP3val-frontend%%200.1.1%%20included/mp3val-0.1.8_with_frontend-0.1.1-bin-win32.zip/download"

echo Downloading mp3val ...
where curl >nul 2>&1
if errorlevel 1 goto mp3val_download_ps

curl -L --fail --progress-bar -o "%MP3VAL_ZIP%" "%MP3VAL_URL%"
if errorlevel 1 curl -L --fail --progress-bar -o "%MP3VAL_ZIP%" "%MP3VAL_URL_FALLBACK%"
if not errorlevel 1 goto mp3val_download_ok
goto mp3val_download_failed

:mp3val_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$urls=@('%MP3VAL_URL%','%MP3VAL_URL_FALLBACK%');" ^
  "$out='%MP3VAL_ZIP%';" ^
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
if errorlevel 1 goto mp3val_download_failed

:mp3val_download_ok
if not exist "%MP3VAL_ZIP%" goto mp3val_download_failed
goto mp3val_extract

:mp3val_download_failed
echo WARNING: mp3val download failed - Detect corruption Fix will use ffmpeg fallback.
goto mp3val_done

:mp3val_extract
if not exist "%MP3VAL_DIR%" mkdir "%MP3VAL_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-mp3val-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -Path '%MP3VAL_ZIP%' -DestinationPath $tmp -Force;" ^
  "$exe = Get-ChildItem -Path $tmp -Recurse -Filter mp3val.exe | Select-Object -First 1;" ^
  "if (-not $exe) { throw 'mp3val.exe not found in archive' };" ^
  "Copy-Item -Path $exe.FullName -Destination '%MP3VAL_DIR%\mp3val.exe' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: mp3val extract failed - Detect corruption Fix will use ffmpeg fallback.
    goto mp3val_done
)
del /Q "%MP3VAL_ZIP%" 2>nul
if exist "%MP3VAL_DIR%\mp3val.exe" echo OK mp3val -^> %MP3VAL_DIR%\mp3val.exe

:mp3val_done

REM --- flac (Detect corruption Deep — STREAMINFO MD5 / foobar parity) ---
set "FLAC_DIR=%~dp0flac"
if exist "%FLAC_DIR%\flac.exe" goto flac_done

set "FLAC_ZIP=%TEMP%\stem-organizer-flac.zip"
set "FLAC_URL=https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.5.0-win.zip"
set "FLAC_URL_FALLBACK=https://downloads.xiph.org/releases/flac/flac-1.5.0-win.zip"

echo Downloading flac ...
curl -L --fail --retry 2 -o "%FLAC_ZIP%" "%FLAC_URL%" 2>nul
if errorlevel 1 goto flac_download_ps
if exist "%FLAC_ZIP%" goto flac_download_ok
curl -L --fail --retry 2 -o "%FLAC_ZIP%" "%FLAC_URL_FALLBACK%" 2>nul
if not errorlevel 1 goto flac_download_ok
goto flac_download_failed

:flac_download_ps
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try {" ^
  "  $ProgressPreference = 'SilentlyContinue';" ^
  "  Invoke-WebRequest -Uri '%FLAC_URL%' -OutFile '%FLAC_ZIP%' -UseBasicParsing;" ^
  "} catch {" ^
  "  try {" ^
  "    Invoke-WebRequest -Uri '%FLAC_URL_FALLBACK%' -OutFile '%FLAC_ZIP%' -UseBasicParsing;" ^
  "  } catch { exit 1 }" ^
  "}"
if errorlevel 1 goto flac_download_failed

:flac_download_ok
if not exist "%FLAC_ZIP%" goto flac_download_failed
goto flac_extract

:flac_download_failed
echo WARNING: flac download failed - Deep FLAC verify will use ffmpeg only ^(no MD5^).
goto flac_done

:flac_extract
mkdir "%FLAC_DIR%" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$tmp = Join-Path $env:TEMP ('stem-organizer-flac-' + [guid]::NewGuid().ToString());" ^
  "New-Item -ItemType Directory -Path $tmp | Out-Null;" ^
  "Expand-Archive -LiteralPath '%FLAC_ZIP%' -DestinationPath $tmp -Force;" ^
  "$exe = Get-ChildItem -Path $tmp -Recurse -Filter flac.exe | Where-Object { $_.FullName -match 'win64' } | Select-Object -First 1;" ^
  "if (-not $exe) { $exe = Get-ChildItem -Path $tmp -Recurse -Filter flac.exe | Select-Object -First 1 };" ^
  "if (-not $exe) { throw 'flac.exe not found in archive' };" ^
  "$srcDir = $exe.Directory.FullName;" ^
  "Copy-Item -Path (Join-Path $srcDir 'flac.exe') -Destination '%FLAC_DIR%\flac.exe' -Force;" ^
  "Get-ChildItem -Path $srcDir -Filter '*.dll' | Copy-Item -Destination '%FLAC_DIR%\' -Force;" ^
  "Remove-Item -Recurse -Force $tmp"
if errorlevel 1 (
    echo WARNING: flac extract failed - Deep FLAC verify will use ffmpeg only ^(no MD5^).
    goto flac_done
)
del /Q "%FLAC_ZIP%" 2>nul
if exist "%FLAC_DIR%\flac.exe" echo OK flac -^> %FLAC_DIR%\flac.exe

:flac_done

echo.
echo Core install OK (%TORCH_LABEL%).
echo.

REM --- Genre & Gender + Rename Auto-detect ---
REM Frozen: extra wheels into site-packages\ (no nested genre_gender_tagger\venv).
REM Source: nested genre_gender_tagger\venv shared by both taggers.
if "%USE_SITE%"=="1" goto install_tagger_site
if not exist "%~dp0genre_gender_tagger\install-deps.bat" goto after_gg
echo === Genre ^& Gender deps ^(always^) ===
set "STEM_GG_BUNDLED=1"
call "%~dp0genre_gender_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_GG_BUNDLED="
if errorlevel 1 (
    echo WARNING: Genre ^& Gender install reported errors - continuing.
)

:after_gg
if not exist "%~dp0instrument_tagger\install-deps.bat" goto after_inst
echo.
echo === Rename Auto-detect deps ^(always^) ===
set "STEM_INST_BUNDLED=1"
set "STEM_GG_BUNDLED=1"
call "%~dp0instrument_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_INST_BUNDLED="
set "STEM_GG_BUNDLED="
if errorlevel 1 (
    echo WARNING: Rename Auto-detect install reported errors - continuing.
)
if not exist "%~dp0panns_tagger\install-deps.bat" goto after_panns
echo.
echo === PANNs tagger deps ^(AudioSet Cnn14^) ===
set "STEM_PANNS_BUNDLED=1"
set "STEM_GG_BUNDLED=1"
call "%~dp0panns_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_PANNS_BUNDLED="
set "STEM_GG_BUNDLED="
if errorlevel 1 (
    echo WARNING: PANNs install reported errors - continuing.
)
:after_panns
if not exist "%~dp0key_tagger\install-deps.bat" goto after_inst
echo.
echo === Key Detect deps ^(MusicalKeyCNN^) ===
set "STEM_KEY_BUNDLED=1"
set "STEM_GG_BUNDLED=1"
call "%~dp0key_tagger\install-deps.bat" %STEM_GG_TORCH%
set "STEM_KEY_BUNDLED="
set "STEM_GG_BUNDLED="
if errorlevel 1 (
    echo WARNING: Key Detect install reported errors - continuing.
)
goto after_inst

:install_tagger_site
echo === Genre ^& Gender + Rename deps into site-packages ===
if not defined PIP_PY (
    call :ensure_pip_helper
    if errorlevel 1 (
        echo WARNING: could not create pip helper - skipping tagger extras.
        goto after_inst
    )
)
if not defined PIP_PY goto after_inst
if "%PIP_PY%"=="" goto after_inst
"%PIP_PY%" -m pip install -q -U pip
REM PaSST FIRST: heavy Genre^&Gender wheels (transformers/onnx) often fail or
REM get cancelled on VMs — previously that left Auto-detect without hear21passt.
set "PASST_INSTALLED=0"
echo Rename Auto-detect ^(hear21passt + helpers^) ...
"%PIP_PY%" -m pip install hear21passt --no-deps -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 (
    echo ERROR: hear21passt pip install failed.
    echo Fix: check network / re-run install-deps.bat beside STEM-organizer.exe.
    goto after_inst
)
"%PIP_PY%" -m pip install timm --no-deps -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 echo WARNING: timm install reported errors - continuing.
"%PIP_PY%" -m pip install pyyaml huggingface_hub safetensors packaging Pillow -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 echo WARNING: PaSST helper deps reported errors - continuing.
REM hear21passt/timm need torchvision - fill gap if Keep=N left it out.
if not defined TORCH_INDEX call :detect_torch_index
call :ensure_torch_extras
if errorlevel 1 (
    echo ERROR: torchvision missing ^(required by hear21passt/timm^).
    echo Fix: re-run install-deps.bat beside STEM-organizer.exe ^(Keep PyTorch = N is fine^).
    goto after_inst
)
REM Same env shape as tagger_launch (PYTHONPATH=site-packages, host Python).
set "STEM_OLD_PP=%PYTHONPATH%"
set "PYTHONPATH=%DEST%"
"%HOST_PY%" -c "import hear21passt; print('OK hear21passt', hear21passt.__file__)"
if errorlevel 1 (
    if defined STEM_OLD_PP (set "PYTHONPATH=%STEM_OLD_PP%") else (set "PYTHONPATH=")
    set "STEM_OLD_PP="
    echo ERROR: hear21passt not importable from:
    echo   %DEST%
    echo Expected folder: %DEST%\hear21passt\
    echo Fix: re-run this install-deps.bat beside STEM-organizer.exe.
    goto after_inst
)
"%HOST_PY%" -c "import torchvision; print('OK torchvision', torchvision.__version__)"
if errorlevel 1 (
    if defined STEM_OLD_PP (set "PYTHONPATH=%STEM_OLD_PP%") else (set "PYTHONPATH=")
    set "STEM_OLD_PP="
    echo ERROR: torchvision not importable from:
    echo   %DEST%
    echo hear21passt/timm need torchvision. Re-run install-deps.bat ^(Keep PyTorch = N is fine^).
    goto after_inst
)
if defined STEM_OLD_PP (set "PYTHONPATH=%STEM_OLD_PP%") else (set "PYTHONPATH=")
set "STEM_OLD_PP="
set "PASST_INSTALLED=1"
if exist "%~dp0genre_gender_tagger\requirements.txt" (
    echo Genre ^& Gender extras ^(transformers / onnxruntime^) ...
    "%PIP_PY%" -m pip install -r "%~dp0genre_gender_tagger\requirements.txt" -t "%DEST%" --upgrade --no-cache-dir
    if errorlevel 1 echo WARNING: Genre ^& Gender extras reported errors - continuing.
) else (
    echo WARNING: genre_gender_tagger\requirements.txt missing - skipping.
)
echo PANNs tagger ^(panns-inference / Cnn14^) ...
"%PIP_PY%" -m pip install panns-inference -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 (
    echo WARNING: panns-inference install reported errors - continuing.
) else (
    set "STEM_OLD_PP=%PYTHONPATH%"
    set "PYTHONPATH=%DEST%"
    "%HOST_PY%" -c "import panns_inference; print('OK panns_inference', panns_inference.__file__)"
    if errorlevel 1 (
        echo WARNING: panns_inference not importable from site-packages - continuing.
    )
    if defined STEM_OLD_PP (set "PYTHONPATH=%STEM_OLD_PP%") else (set "PYTHONPATH=")
    set "STEM_OLD_PP="
)
echo Key Detect ^(librosa / more-itertools / tqdm / soxr^) ...
"%PIP_PY%" -m pip install more-itertools tqdm librosa soxr -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 (
    echo WARNING: Key Detect extras reported errors - continuing.
) else (
    set "STEM_OLD_PP=%PYTHONPATH%"
    set "PYTHONPATH=%DEST%"
    "%HOST_PY%" -c "import librosa, more_itertools; print('OK key_tagger deps')"
    if errorlevel 1 (
        echo WARNING: Key Detect deps not importable from site-packages - continuing.
    )
    if defined STEM_OLD_PP (set "PYTHONPATH=%STEM_OLD_PP%") else (set "PYTHONPATH=")
    set "STEM_OLD_PP="
)

:after_inst
echo.
echo All done (%TORCH_LABEL%).
if "%USE_SITE%"=="1" (
    if not "%PASST_INSTALLED%"=="1" (
        echo Rename Auto-detect: NOT READY - hear21passt and/or torchvision missing.
        echo Re-run install-deps.bat ^(Keep PyTorch = N is fine^) and confirm
        echo "OK hear21passt" and "OK torchvision" appear.
    )
    echo.
    echo Start STEM-organizer.exe in this folder.
) else (
    echo Run from source:
    echo     .venv\Scripts\python.exe run_stem_organizer.py
)
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: install failed. Check messages above.
pause
exit /b 1

REM --- Reject empty / header / nvidia-smi error text as GPU_NAME ---
:gpu_name_ok
if not defined GPU_NAME exit /b 1
if "%GPU_NAME%"=="" exit /b 1
if /I "%GPU_NAME%"=="name" exit /b 1
echo %GPU_NAME% | findstr /I /C:"ERROR" /C:"not recognized" >nul
if not errorlevel 1 exit /b 1
exit /b 0

REM --- Match TORCH_INDEX to existing site-packages torch (+cu128/+cu124/+cpu) ---
:detect_torch_index
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
for /f "delims=" %%F in ('"%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import torch; v=torch.__version__; print('cu128' if '+cu128' in v else 'cu124' if '+cu' in v else 'cpu')" 2^>nul') do set "TORCH_FLAVOR=%%F"
if /I "%TORCH_FLAVOR%"=="cu128" set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
if /I "%TORCH_FLAVOR%"=="cu124" set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
if /I "%TORCH_FLAVOR%"=="cpu" set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
echo Detected torch flavor: %TORCH_FLAVOR% - using %TORCH_INDEX%
exit /b 0

REM --- Install torchvision/torchaudio into DEST if import fails (same index as torch) ---
:ensure_torch_extras
if not defined PIP_PY (
    echo ERROR: pip helper not ready - cannot install torchvision.
    exit /b 1
)
if not defined TORCH_INDEX set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
REM Pillow required by torchvision (from PIL import Image); install before verify
"%PIP_PY%" -m pip install Pillow -t "%DEST%" --upgrade --no-cache-dir
if errorlevel 1 (
    echo ERROR: Pillow install failed ^(needed by torchvision^).
    exit /b 1
)
"%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import torchvision" 1>nul 2>nul
if errorlevel 1 (
    echo torchvision missing - installing from %TORCH_INDEX% ...
    "%PIP_PY%" -m pip install torchvision --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
    if errorlevel 1 (
        echo ERROR: torchvision install failed ^(needed by hear21passt/timm^).
        exit /b 1
    )
) else (
    echo torchvision already present.
)
"%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import torchaudio" 1>nul 2>nul
if errorlevel 1 (
    echo torchaudio missing - installing from %TORCH_INDEX% ...
    "%PIP_PY%" -m pip install torchaudio --index-url %TORCH_INDEX% -t "%DEST%" --upgrade --no-cache-dir --no-deps
    if errorlevel 1 (
        echo WARNING: torchaudio install failed - demucs may break.
    )
) else (
    echo torchaudio already present.
)
"%HOST_PY%" -c "import sys; sys.path.insert(0, r'%DEST%'); import PIL; import torchvision; print('OK PIL', PIL.__version__, 'torchvision', torchvision.__version__)"
if errorlevel 1 (
    echo ERROR: torchvision still not importable from:
    echo   %DEST%
    exit /b 1
)
exit /b 0

REM --- Ensure %TEMP%\stem-organizer-pip-venv matches HOST_PY (major.minor) ---
REM Stale helper from another Python (e.g. 3.11 home while PATH is 3.10) makes
REM the Windows venv launcher print: No Python at '"C:\...\Python311\python.exe"'
:ensure_pip_helper
set "PIP_VENV=%TEMP%\stem-organizer-pip-venv"
set "PIP_PY=%PIP_VENV%\Scripts\python.exe"
if not exist "%PIP_PY%" goto pip_helper_create
"%PIP_PY%" -c "import sys; v='%HOST_VER%'.split('.'); raise SystemExit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" 1>nul 2>nul
if not errorlevel 1 exit /b 0
echo Pip helper venv is broken or wrong Python - recreating with %HOST_VER% ...
rmdir /S /Q "%PIP_VENV%" 2>nul

:pip_helper_create
echo Preparing clean pip helper ...
"%HOST_PY%" -m venv "%PIP_VENV%"
if errorlevel 1 (
    echo ERROR: could not create pip helper with:
    echo   %HOST_PY%
    exit /b 1
)
set "PIP_PY=%PIP_VENV%\Scripts\python.exe"
"%PIP_PY%" -c "import sys; v='%HOST_VER%'.split('.'); raise SystemExit(0 if sys.version_info[:2]==(int(v[0]),int(v[1])) else 1)" 1>nul 2>nul
if errorlevel 1 (
    echo ERROR: pip helper still does not run under Python %HOST_VER%.
    echo No Python at %HOST_PY% - install that version, or delete:
    echo   %PIP_VENV%
    exit /b 1
)
exit /b 0
