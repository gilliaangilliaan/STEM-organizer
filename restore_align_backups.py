#!/usr/bin/env python3
"""Restore instrumental/acapella from _backup_before_align and remove backup folders."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKUP_DIR_NAME = '_backup_before_align'


def find_backup_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    if root.name == BACKUP_DIR_NAME:
        return [root]
    return sorted(p for p in root.rglob(BACKUP_DIR_NAME) if p.is_dir())


def restore_backup_dir(backup_dir: Path, *, dry_run: bool = False) -> tuple[int, list[str]]:
    song_dir = backup_dir.parent
    restored = 0
    messages: list[str] = []

    for src in sorted(backup_dir.iterdir()):
        if not src.is_file():
            continue
        dest = song_dir / src.name
        if dry_run:
            messages.append(f'[dry-run] {src.name}  →  {song_dir.name}/')
        else:
            shutil.copy2(src, dest)
            messages.append(f'✓ {song_dir.name}/{src.name}')
        restored += 1

    if restored == 0:
        messages.append(f'· empty backup, skipping: {backup_dir}')
        return 0, messages

    if dry_run:
        messages.append(f'[dry-run] remove {backup_dir}')
    else:
        shutil.rmtree(backup_dir)
        messages.append(f'✓ removed {backup_dir.name} in {song_dir.name}')

    return restored, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Copy stems from _backup_before_align back into each song folder, then delete the backup.',
    )
    parser.add_argument(
        'root',
        type=Path,
        help='Folder to scan (e.g. with_original or stems root). Searches recursively.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be restored without copying or deleting.',
    )
    args = parser.parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        print(f'Not a folder: {root}', file=sys.stderr)
        return 1

    backup_dirs = find_backup_dirs(root)
    if not backup_dirs:
        print(f'No {BACKUP_DIR_NAME} folders found under {root}')
        return 0

    total_files = 0
    for backup_dir in backup_dirs:
        count, lines = restore_backup_dir(backup_dir, dry_run=args.dry_run)
        total_files += count
        for line in lines:
            print(line)

    label = 'Would restore' if args.dry_run else 'Restored'
    print(f'\n{label} {total_files:,} file(s) from {len(backup_dirs):,} backup folder(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
