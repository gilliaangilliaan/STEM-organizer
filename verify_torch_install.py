"""Post-install check for install-deps.bat (GPU arch vs PyTorch build)."""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    dest = Path(__file__).resolve().parent / 'site-packages'
    if dest.is_dir():
        sys.path.insert(0, str(dest))

    import torch

    if not torch.cuda.is_available():
        print('no NVIDIA GPU detected (CPU mode is fine)')
        return 0

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    tag = f'sm_{major}{minor}'
    archs = []
    try:
        archs = torch.cuda.get_arch_list()
    except Exception:
        pass

    if archs and tag not in archs:
        print(f'{name} ({tag}) not supported by torch {torch.__version__}')
        print(f'  PyTorch arch list: {", ".join(archs)}')
        return 1

    try:
        x = torch.zeros(1, device='cuda')
        x.add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        print(f'{name} ({tag}) CUDA probe failed: {exc}')
        return 1

    print(f'{name} ({tag}) OK with torch {torch.__version__}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
