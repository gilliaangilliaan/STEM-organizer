#!/usr/bin/env python3
"""Fair speed compare: our sequential Key Detect vs MusicalKeyCNN predict3-style pool.

Uses the same checkpoint, same 8 s chunks, same BATCH_SIZE. Only the CQT
preprocess parallelism differs (Pool vs one-file-at-a-time).

Example:
  python bench_vs_predict3.py "K:\\Consensus Dataset_small" --n 100
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Silence mpg123 ID3 chatter on Windows consoles.
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _pick_files(folder: Path, n: int, *, recursive: bool) -> list[Path]:
    from inference import collect_audio_files

    files = collect_audio_files(folder, recursive=recursive)
    if n > 0:
        files = files[:n]
    return files


def _run_ours(paths: list[Path], ckpt: Path, device: str, batch_size: int) -> float:
    import torch

    from inference import BATCH_SIZE, load_model, process_batched

    # Temporarily bump batch size to match predict3 if requested.
    import inference as inf

    old = inf.BATCH_SIZE
    inf.BATCH_SIZE = batch_size
    try:
        torch.set_num_threads(1)
        model = load_model(ckpt, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        process_batched(model, paths, device)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0
    finally:
        inf.BATCH_SIZE = old


def _pool_preproc(path_s: str):
    """Worker entry — must be top-level for Windows spawn."""
    from inference import preproc

    return preproc(path_s)


def _run_predict3_style(
    paths: list[Path],
    ckpt: Path,
    device: str,
    batch_size: int,
    workers: int,
) -> float:
    """Mirror MusicalKeyCNN predict3: Pool CQT + GPU chunk batches."""
    import torch

    from inference import _chunk_frames, load_model, logits_to_key

    torch.set_num_threads(1)
    model = load_model(ckpt, device)
    chunk = _chunk_frames()
    path_strs = [str(p) for p in paths]

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    results: dict[str, list[torch.Tensor]] = {}
    batch_paths: list[str] = []
    batch_chunks: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal batch_paths, batch_chunks
        if not batch_chunks:
            return
        tensor = torch.concat([c.unsqueeze(0) for c in batch_chunks], dim=0)
        with torch.no_grad():
            out = model(tensor)
        for i, pth in enumerate(batch_paths):
            results.setdefault(pth, []).append(out[i].detach().cpu())
        batch_paths = []
        batch_chunks = []

    n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(_pool_preproc, path_strs, chunksize=1):
            path_s, data, err = r
            if err or data is None:
                continue
            n_chunks = int(data.shape[1]) // chunk
            if n_chunks <= 0:
                continue
            for i in range(n_chunks):
                sl = data[:, i * chunk : (i + 1) * chunk]
                t = torch.tensor(sl, dtype=torch.float32, device=device).unsqueeze(0)
                batch_paths.append(path_s)
                batch_chunks.append(t)
                if len(batch_chunks) >= batch_size:
                    flush()
    flush()

    # Mean logits (same as both pipelines) — cheap, include for parity.
    for pth, parts in results.items():
        logits_to_key(torch.mean(torch.stack(parts), dim=0))

    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=str)
    ap.add_argument("--n", type=int, default=100, help="How many files (0 = all)")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=str(_HERE / "checkpoints" / "nf50-q05-221125.pt"),
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="GPU chunk batch (predict3 uses 1024; ours default 320)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Pool workers for predict3-style (0 = cpu_count-1)",
    )
    ap.add_argument(
        "--only",
        choices=("both", "ours", "pool"),
        default="both",
    )
    args = ap.parse_args(argv)

    folder = Path(args.folder).expanduser().resolve()
    ckpt = Path(args.checkpoint)
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}")
        return 2
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 2

    print("Collecting files…", flush=True)
    paths = _pick_files(folder, args.n, recursive=not args.no_recursive)
    if not paths:
        print("No audio files found.")
        return 1

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to cpu")
        device = "cpu"

    print(f"Files: {len(paths):,}", flush=True)
    print(f"Checkpoint: {ckpt.name}", flush=True)
    print(f"Device: {device} · GPU batch={args.batch_size}", flush=True)
    print(f"First: {paths[0].name}", flush=True)
    print(f"Last:  {paths[-1].name}", flush=True)
    print("", flush=True)

    # Warmup CUDA / import cost outside timed section for the first run.
    if device == "cuda":
        print("Warming CUDA…", flush=True)
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()

    times: dict[str, float] = {}

    if args.only in ("both", "ours"):
        print("=== STEM key_tagger (sequential CQT) ===", flush=True)
        t = _run_ours(paths, ckpt, device, args.batch_size)
        times["ours"] = t
        print(f"Took {t:.2f}s  ({len(paths)/t:.2f} files/s)", flush=True)
        print("", flush=True)

    if args.only in ("both", "pool"):
        print("=== predict3-style (multiprocess CQT pool) ===", flush=True)
        t = _run_predict3_style(
            paths, ckpt, device, args.batch_size, args.workers
        )
        times["pool"] = t
        print(f"Took {t:.2f}s  ({len(paths)/t:.2f} files/s)", flush=True)
        print("", flush=True)

    if "ours" in times and "pool" in times and times["pool"] > 0:
        speedup = times["ours"] / times["pool"]
        print(
            f"Pool is {speedup:.2f}× faster than sequential "
            f"({times['ours']:.1f}s vs {times['pool']:.1f}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    # Required on Windows for Pool spawn.
    mp.freeze_support()
    raise SystemExit(main())
