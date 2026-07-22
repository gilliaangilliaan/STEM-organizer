"""PyInstaller / dev entry — mirrors the Tk original's run_stem_organizer.py."""
from __future__ import annotations

import sys


def main() -> None:
    from stem_organizer.main_entry import run

    run(sys.argv)


if __name__ == "__main__":
    main()
