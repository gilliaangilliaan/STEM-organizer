@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==============================================
echo  Genre / Gender Tagger - dependency installer
echo ==============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.10.x or 3.11.x from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then run this script again.
    echo.
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
    echo 3.12+ is not recommended for this project.
    echo If multiple Pythons are installed:  py -3.10 install-deps.bat  or  py -3.11 install-deps.bat
    echo.
    pause
    exit /b 1
)

echo Using:
python --version
echo.

set "VENV=%~dp0venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo Creating venv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo ERROR: could not create venv.
        pause
        exit /b 1
    )
) else (
    echo venv already present:
    echo   "%VENV%"
)

set "PY=%VENV%\Scripts\python.exe"

:ask_gpu
echo.
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
echo.
echo Installing PyTorch (%TORCH_LABEL%) ...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto failed

"%PY%" -m pip install torch torchaudio --index-url "%TORCH_INDEX%" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo.
echo Installing project dependencies ...
"%PY%" -m pip install -r "%~dp0requirements.txt" --upgrade --no-cache-dir
if errorlevel 1 goto failed

echo.
echo Verifying install ...
if /I "%TORCH_LABEL%"=="CPU" (
    "%PY%" -c "import torch, torchaudio, soundfile, pandas, tqdm, mutagen, transformers, tensorflow, librosa; v=torch.__version__; assert '+cpu' in v or 'cpu' in v.lower(), f'expected CPU torch, got {v}'; print('OK - torch', v, '| device = cpu')"
) else (
    "%PY%" -c "import torch, torchaudio, soundfile, pandas, tqdm, mutagen, transformers, tensorflow, librosa; print('OK - torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
)
if errorlevel 1 goto failed

if /I not "%TORCH_LABEL%"=="CPU" (
    echo.
    echo Checking GPU visibility ...
    "%PY%" -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>nul
    if errorlevel 1 (
        echo WARNING: PyTorch installed but no CUDA GPU was detected.
        echo If you are in a VM without GPU passthrough, re-run this script and pick option 2 ^(CPU^).
        echo The tagger will fall back to CPU automatically either way.
    )
)

echo.
echo ==============================================
echo  Done.
echo ==============================================
if /I "%TORCH_LABEL%"=="CPU" (
    echo Installed CPU PyTorch. The tagger runs on CPU.
) else (
    echo Installed %TORCH_LABEL% PyTorch.
)
echo.
echo To run:
echo   run.bat
echo.
echo Models download on first run (Hugging Face MAEST / Essentia .pb).
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: install failed. See messages above.
echo Use Python 3.10.x or 3.11.x and keep install-deps.bat beside requirements.txt.
echo Folder paths with spaces are supported ^(e.g. "Genre Gender Tagger"^).
pause
exit /b 1
