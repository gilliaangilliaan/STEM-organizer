"""Stdlib modules needed by external numpy/torch/soundfile/demucs in the frozen exe.

PyInstaller onedir apps omit unused stdlib. External torch (package_exporter,
strobelight, etc.) then fails with ModuleNotFoundError. Prefer a slightly
larger bundle over rebuild whack-a-mole.

- ``_ML_STDLIB_MODULES``: curated set force-imported at runtime.
- ``iter_ml_stdlib_module_names()``: nearly all stdlib names for .spec
  hiddenimports (same strategy as STEM-organizer CTk ``stem_organizer.spec``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Keep in sync with hiddenimports in stem_organizer_py6.spec (imports this list /
# iter_ml_stdlib_module_names).
_ML_STDLIB_SKIP = frozenset({
    'test',
    'tests',
    'idlelib',
    'turtledemo',
    'lib2to3',
    'ensurepip',
    'venv',
    'tkinter',
    '__pycache__',
})

# Generous curated set for runtime ensure_* (and Analysis when traced from entry).
_ML_STDLIB_MODULES = (
    # --- already required / previously missing ---
    'platform',
    'sysconfig',
    'timeit',
    'ctypes',
    'ctypes.util',
    'pickle',
    'pickletools',
    'gzip',
    'bz2',
    'lzma',
    'zlib',
    'zipfile',
    'tarfile',
    'logging',
    'logging.config',
    'logging.handlers',
    'mmap',
    'multiprocessing',
    'multiprocessing.spawn',
    'multiprocessing.pool',
    'multiprocessing.shared_memory',
    'socket',
    'ssl',
    'struct',
    'select',
    'selectors',
    # --- inspect / codegen (torch, numpy) ---
    'inspect',
    'dis',
    'ast',
    'token',
    'tokenize',
    'keyword',
    'linecache',
    'pydoc',
    'doctest',
    'difflib',
    'pprint',
    'code',
    'codeop',
    'py_compile',
    'compileall',
    # --- core helpers ---
    'copy',
    'copyreg',
    'functools',
    'itertools',
    'operator',
    'collections',
    'collections.abc',
    'typing',
    'enum',
    'traceback',
    'warnings',
    'contextlib',
    'contextvars',
    'threading',
    'types',
    'dataclasses',
    'abc',
    'io',
    'codecs',
    'weakref',
    'gc',
    'atexit',
    # --- importlib ---
    'importlib',
    'importlib.metadata',
    'importlib.resources',
    'importlib.machinery',
    'importlib.util',
    'importlib.abc',
    'pkgutil',
    'modulefinder',
    # --- crypto / encoding ---
    'hashlib',
    'hmac',
    'secrets',
    'base64',
    'binascii',
    'binhex',
    'quopri',
    'uu',
    # --- numbers / text ---
    'numbers',
    'decimal',
    'fractions',
    'statistics',
    'math',
    'cmath',
    'random',
    'textwrap',
    'string',
    'stringprep',
    're',
    'json',
    'csv',
    'configparser',
    'netrc',
    # --- argparse / CLI ---
    'argparse',
    'getopt',
    'shlex',
    'cmd',
    # --- pathlib / fs ---
    'pathlib',
    'os',
    'os.path',
    'ntpath',
    'posixpath',
    'genericpath',
    'tempfile',
    'glob',
    'fnmatch',
    'stat',
    'fileinput',
    'filecmp',
    'shutil',
    'errno',
    # --- concurrency ---
    'concurrent',
    'concurrent.futures',
    'concurrent.futures.thread',
    'concurrent.futures.process',
    'asyncio',
    'asyncio.events',
    'asyncio.base_events',
    'asyncio.coroutines',
    'asyncio.futures',
    'asyncio.tasks',
    'asyncio.locks',
    'asyncio.queues',
    'asyncio.subprocess',
    'asyncio.streams',
    'queue',
    'sched',
    'signal',
    'subprocess',
    # --- net / email / http / urllib ---
    'email',
    'email.message',
    'email.parser',
    'email.policy',
    'email.header',
    'email.utils',
    'email.mime',
    'email.mime.text',
    'email.mime.multipart',
    'email.mime.base',
    'http',
    'http.client',
    'http.server',
    'http.cookiejar',
    'http.cookies',
    'urllib',
    'urllib.parse',
    'urllib.request',
    'urllib.error',
    'urllib.response',
    'urllib.robotparser',
    'ipaddress',
    'ftplib',
    'smtplib',
    'poplib',
    'imaplib',
    # --- xml / html / markup ---
    'html',
    'html.parser',
    'html.entities',
    'xml',
    'xml.etree',
    'xml.etree.ElementTree',
    'xml.parsers',
    'xml.parsers.expat',
    'xml.sax',
    'xml.dom',
    'xml.dom.minidom',
    'xmlrpc',
    'xmlrpc.client',
    # --- db / persistence ---
    'sqlite3',
    'dbm',
    'shelve',
    # --- misc commonly pulled by ML stacks ---
    'uuid',
    'locale',
    'gettext',
    'calendar',
    'datetime',
    'time',
    'zoneinfo',
    'graphlib',
    'heapq',
    'bisect',
    'array',
    'unicodedata',
    'encodings',
    'mimetypes',
    'wsgiref',
    'wsgiref.simple_server',
    'unittest',
    'unittest.mock',
)


def iter_ml_stdlib_module_names(lib_root: Path | None = None) -> list[str]:
    """Nearly all stdlib module names under Lib/ for PyInstaller hiddenimports.

    Skips test suites, idle, venv, tkinter. Call from the .spec at build time.
    """
    root = lib_root if lib_root is not None else Path(sys.base_prefix) / 'Lib'
    if not root.is_dir():
        return list(_ML_STDLIB_MODULES)

    names: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name in _ML_STDLIB_SKIP or entry.name.startswith('test'):
            continue
        if entry.is_dir():
            if not (entry / '__init__.py').is_file():
                continue
            names.append(entry.name)
            for py in sorted(entry.rglob('*.py')):
                if py.name == '__init__.py':
                    continue
                if any(part in _ML_STDLIB_SKIP or part.startswith('test')
                       for part in py.relative_to(root).parts):
                    continue
                rel = py.relative_to(root).with_suffix('')
                names.append('.'.join(rel.parts))
        elif entry.suffix == '.py' and entry.name != '__init__.py':
            names.append(entry.stem)
    return names


def ensure_stdlib_for_external_ml() -> None:
    for name in _ML_STDLIB_MODULES:
        try:
            __import__(name)
        except ImportError:
            pass
