"""PyInstaller entry point — checks external ML deps before loading the UI."""
from deps_bootstrap import ensure_ml_deps, is_frozen


def main() -> None:
    if is_frozen():
        from frozen_stdlib_imports import ensure_stdlib_for_external_ml

        ensure_stdlib_for_external_ml()
    import stem_organizer_ui

    stem_organizer_ui.main()


if __name__ == '__main__':
    main()
