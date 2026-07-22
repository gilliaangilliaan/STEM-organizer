# PyInstaller spec for STEM organizer (PySide6)
# Build: pyinstaller stem_organizer_py6.spec
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Pull in PySide6 plugins + translations
datas = []
datas += collect_data_files('PySide6', include_py_files=False)
binaries = []

# Bundle the logo + tagger scripts (not the venvs/models).
datas += [('logo.png', '.')]
datas += [('logo.ico', '.')]
# settings.json is user-local (created at runtime by SettingsStore); do not bundle.
# ffmpeg is installed next to the exe by install-deps.bat; do not bundle.
datas += [('genre_gender_tagger/genre_gender_tagger.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/vocal_reverb.py', 'genre_gender_tagger')]
datas += [('genre_gender_tagger/requirements.txt', 'genre_gender_tagger')]
datas += [('instrument_tagger/instrument_tagger.py', 'instrument_tagger')]
datas += [('instrument_tagger/passt_mel.py', 'instrument_tagger')]

hiddenimports = []
# Avoid PySide6.scripts (deploy tooling); collect_submodules hits missing project_lib.
hiddenimports += [
    m for m in collect_submodules('PySide6')
    if not m.startswith('PySide6.scripts')
]
hiddenimports += ['classify_backend', 'pair_matcher', 'stem_align',
                  'ffmpeg_bootstrap', 'deps_bootstrap', 'resource_monitor',
                  'update_checker', 'single_instance', 'done_sound',
                  'sounddevice', 'soundfile', 'resampy', 'numpy',
                  'track_renamer.engine', 'track_renamer.folder_scanner',
                  'track_renamer.audio_preview', 'track_renamer.instrument_enrich',
                  'track_renamer.category_palette']

a = Analysis(
    ['run_stem_organizer.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'customtkinter', 'PySide6.scripts'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='STEM-organizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='STEM-organizer',
)
