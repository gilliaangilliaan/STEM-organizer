# -*- mode: python ; coding: utf-8 -*-
"""Slim PyInstaller spec — UI only; PyTorch/Demucs live in site-packages beside the exe."""
from pathlib import Path
import sys

block_cipher = None
root = Path(SPECPATH)

# Stdlib modules pulled in by external numpy/torch/soundfile/demucs at runtime.
_ML_STDLIB_SKIP = {
    'test', 'tests', 'idlelib', 'turtledemo', 'lib2to3', 'ensurepip', 'venv',
    'tkinter', '__pycache__',
}


def _iter_stdlib_module_names(lib_root: Path) -> list[str]:
    names: list[str] = []
    for entry in sorted(lib_root.iterdir()):
        if entry.name in _ML_STDLIB_SKIP or entry.name.startswith('test'):
            continue
        if entry.is_dir():
            if not (entry / '__init__.py').is_file():
                continue
            names.append(entry.name)
            for py in sorted(entry.rglob('*.py')):
                if py.name == '__init__.py':
                    continue
                rel = py.relative_to(lib_root).with_suffix('')
                names.append('.'.join(rel.parts))
        elif entry.suffix == '.py' and entry.name != '__init__.py':
            names.append(entry.stem)
    return names


_ml_stdlib_imports = _iter_stdlib_module_names(Path(sys.base_prefix) / 'Lib')

datas = [
    (str(root / 'logo.ico'), '.'),
    (str(root / 'logo.png'), '.'),
    (str(root / 'install-deps.bat'), '.'),
    (str(root / 'verify_torch_install.py'), '.'),
]
# ffmpeg is installed beside the exe by install-deps.bat — do not embed it in the onefile build.

hiddenimports = [
    'update_checker',
    'stem_player',
    'pair_matcher',
    'stem_align',
    'pair_finder_panel',
    'ui_theme',
    'restore_align_backups',
    'requests',
    'packaging',
    'packaging.version',
    'deps_bootstrap',
    'ffmpeg_bootstrap',
    'resource_monitor',
    'single_instance',
] + _ml_stdlib_imports

a = Analysis(
    [str(root / 'run_stem_organizer.py')],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchaudio',
        'demucs',
        'numpy',
        'soundfile',
        '_soundfile_data',
        'matplotlib',
        'pandas',
        'scipy',
        'IPython',
        'notebook',
        'pytest',
        'customtkinter',
        'torchvision',
        'transformers',
        'tensorflow',
        'tensorboard',
        'sklearn',
        'numba',
        'librosa',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='STEM-organizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(root / 'logo.ico'),
)
