"""Train vocal dry/wet reverb classifier from reverb_data/dry and reverb_data/wet.

Usage (from genre_gender_tagger, with venv active):

  python train_vocal_reverb.py
  python train_vocal_reverb.py --data reverb_data --out models/vocal_reverb.pt --epochs 20

Requires folders:
  reverb_data/dry/*.wav|flac|mp3|...
  reverb_data/wet/*.wav|flac|mp3|...
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vocal_reverb import (
    CLASS_NAMES,
    DEFAULT_CONFIG,
    MelReverbNet,
    audio_to_logmel,
    crop_or_pad_logmel,
    frames_for_clip,
    load_mono,
    probe_audio,
)

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}


def _collect(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    files.sort()
    return files


def _filter_readable(
    items: list[tuple[Path, int]], bad_log: Path | None = None
) -> list[tuple[Path, int]]:
    """Drop files that fail a short decode probe (corrupt FLAC sync, etc.)."""
    good: list[tuple[Path, int]] = []
    bad_lines: list[str] = []
    for path, label in tqdm(items, desc="Scan audio", unit="file"):
        try:
            probe_audio(path)
            good.append((path, label))
        except Exception as exc:
            bad_lines.append(f"{path}\t{exc}")
    if bad_lines:
        print(f"Skipping {len(bad_lines)} unreadable file(s).")
        if bad_log is not None:
            bad_log.parent.mkdir(parents=True, exist_ok=True)
            bad_log.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")
            print(f"Bad list: {bad_log}")
    return good


class ReverbMelDataset(Dataset):
    def __init__(
        self,
        items: list[tuple[Path, int]],
        cfg: dict,
        *,
        train: bool,
        seed: int,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.train = train
        self.target_frames = frames_for_clip(cfg)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        # Retry a few times if a file slips past the scan or fails mid-decode.
        last_err: Exception | None = None
        for attempt in range(6):
            idx = index if attempt == 0 else int(self._rng.integers(0, len(self.items)))
            path, label = self.items[idx]
            try:
                audio = load_mono(path, int(self.cfg["sample_rate"]))

                # Light train augment: random gain + optional time shift
                if self.train and audio.size:
                    gain = float(self._rng.uniform(0.7, 1.15))
                    audio = np.clip(audio * gain, -1.0, 1.0)
                    if audio.size > 1 and self._rng.random() < 0.5:
                        shift = int(self._rng.integers(0, min(audio.size, 4000)))
                        audio = np.roll(audio, shift)

                logmel = audio_to_logmel(
                    audio,
                    sample_rate=int(self.cfg["sample_rate"]),
                    n_mels=int(self.cfg["n_mels"]),
                    n_fft=int(self.cfg["n_fft"]),
                    hop_length=int(self.cfg["hop_length"]),
                )
                rng = self._rng if self.train else None
                logmel = crop_or_pad_logmel(logmel, self.target_frames, rng=rng)
                x = torch.from_numpy(logmel).unsqueeze(0)  # 1, M, T
                y = torch.tensor(label, dtype=torch.long)
                return x, y
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"failed to load audio after retries (last={self.items[index][0]}): {last_err}"
        )


def _split(
    items: list[tuple[Path, int]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    by_class: dict[int, list[tuple[Path, int]]] = {0: [], 1: []}
    for item in items:
        by_class[item[1]].append(item)

    train: list[tuple[Path, int]] = []
    val: list[tuple[Path, int]] = []
    rng = random.Random(seed)
    for cls, rows in by_class.items():
        rng.shuffle(rows)
        if len(rows) < 2:
            train.extend(rows)
            continue
        n_val = max(1, int(round(len(rows) * val_ratio)))
        n_val = min(n_val, len(rows) - 1)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss_sum += float(crit(logits, y).item()) * y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train dry/wet vocal reverb mel-CNN for STEM Genre/Gender."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "reverb_data",
        help="Root with dry/ and wet/ subfolders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "models" / "vocal_reverb.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    dry_dir = args.data / "dry"
    wet_dir = args.data / "wet"
    dry_files = _collect(dry_dir)
    wet_files = _collect(wet_dir)

    print("==============================")
    print("Train vocal reverb (mel-CNN)")
    print("==============================")
    print(f"data:  {args.data}")
    print(f"dry:   {len(dry_files)} files  ({dry_dir})")
    print(f"wet:   {len(wet_files)} files  ({wet_dir})")
    print()

    if len(dry_files) < 2 or len(wet_files) < 2:
        print(
            "ERROR: need at least 2 audio files in each of dry/ and wet/.\n"
            f"  put packs in:\n    {dry_dir}\n    {wet_dir}",
            file=sys.stderr,
        )
        return 1

    items = [(p, 0) for p in dry_files] + [(p, 1) for p in wet_files]
    print("Probing files (skips corrupt FLAC / decode errors)...")
    items = _filter_readable(items, bad_log=args.data / "bad_files.txt")
    n_dry = sum(1 for _, label in items if label == 0)
    n_wet = sum(1 for _, label in items if label == 1)
    print(f"readable: dry={n_dry}  wet={n_wet}")
    print()

    if n_dry < 2 or n_wet < 2:
        print(
            "ERROR: need at least 2 readable files in each of dry/ and wet/ "
            "after the scan.",
            file=sys.stderr,
        )
        return 1

    train_items, val_items = _split(items, args.val_ratio, args.seed)
    print(f"train: {len(train_items)}  val: {len(val_items)}")
    print(f"classes: {CLASS_NAMES[0]}=0  {CLASS_NAMES[1]}=1")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")
    print()

    cfg = dict(DEFAULT_CONFIG)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_ds = ReverbMelDataset(train_items, cfg, train=True, seed=args.seed)
    val_ds = ReverbMelDataset(val_items, cfg, train=False, seed=args.seed + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MelReverbNet(
        n_mels=int(cfg["n_mels"]),
        channels=tuple(cfg["channels"]),
        n_classes=len(CLASS_NAMES),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.size(0)
            n += int(y.size(0))
        train_loss = running / max(n, 1)
        val_loss, val_acc = _eval(model, val_loader, device)
        marker = ""
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{marker}"
        )

    if best_state is None:
        best_state = model.state_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "classes": CLASS_NAMES,
        "config": cfg,
        "metrics": {
            "best_val_acc": best_acc,
            "n_dry": n_dry,
            "n_wet": n_wet,
            "epochs": args.epochs,
        },
    }
    torch.save(payload, args.out)
    elapsed = time.perf_counter() - t0
    print()
    print("==============================")
    print("DONE")
    print("==============================")
    print(f"saved: {args.out}")
    print(f"best val acc: {best_acc:.3f}")
    print(f"time: {elapsed:.1f}s")
    print()
    print("STEM Gender tab will use this file automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
