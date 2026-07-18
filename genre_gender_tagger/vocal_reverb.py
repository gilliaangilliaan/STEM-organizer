"""Lightweight dry/wet vocal reverb classifier (mel-CNN).

Ships as models/vocal_reverb.pt — no Whisper / Hugging Face at runtime.
Train with train_vocal_reverb.py from reverb_data/dry and reverb_data/wet.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn

MODEL_NAME = "vocal_reverb.pt"
CLASS_NAMES = ("dry", "wet")

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 256,
    "clip_seconds": 4.0,
    "channels": (32, 64, 128),
}


class MelReverbNet(nn.Module):
    """Small Conv2d stack over log-mel → dry/wet logits."""

    def __init__(
        self,
        n_mels: int = 64,
        channels: tuple[int, ...] = (32, 64, 128),
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, n_classes),
        )
        self.n_mels = n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, time)
        return self.head(self.features(x))


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Mono float32 @ sample_rate. soundfile first, librosa fallback for bad FLACs."""
    path = Path(path)
    audio = None
    sr = sample_rate
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = data.mean(axis=1)
    except Exception:
        # Corrupt / sync-lost FLACs often fail in libsndfile; librosa/audioread
        # can still decode some of them. If both fail, let caller handle.
        audio, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)

    if audio is None or audio.size == 0:
        raise ValueError(f"empty audio: {path}")

    if int(sr) != int(sample_rate):
        audio = librosa.resample(
            audio, orig_sr=int(sr), target_sr=sample_rate, res_type="soxr_hq"
        )
    return np.asarray(audio, dtype=np.float32)


def probe_audio(path: str | Path) -> None:
    """Raise if the file cannot be opened/decoded (short read)."""
    path = Path(path)
    try:
        with sf.SoundFile(str(path)) as handle:
            n = min(int(handle.frames), 4096)
            if n > 0:
                handle.read(frames=n, dtype="float32", always_2d=True)
            return
    except Exception:
        pass
    # Fallback decode of a short slice
    audio, _sr = librosa.load(str(path), sr=None, mono=True, duration=0.25)
    if audio is None or np.asarray(audio).size == 0:
        raise ValueError(f"unreadable or empty: {path}")


def audio_to_logmel(
    audio: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Return log-mel (n_mels, time) float32."""
    if audio.size == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # Stabilize empty / silent clips
    if not np.isfinite(logmel).all():
        logmel = np.nan_to_num(logmel, nan=-80.0, posinf=0.0, neginf=-80.0)
    return logmel


def crop_or_pad_logmel(
    logmel: np.ndarray, target_frames: int, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Random crop (train) or center crop / pad (infer) to target_frames."""
    n_mels, n_frames = logmel.shape
    if n_frames == target_frames:
        return logmel
    if n_frames > target_frames:
        if rng is not None:
            start = int(rng.integers(0, n_frames - target_frames + 1))
        else:
            start = max(0, (n_frames - target_frames) // 2)
        return logmel[:, start : start + target_frames]
    pad = target_frames - n_frames
    left = pad // 2
    right = pad - left
    return np.pad(logmel, ((0, 0), (left, right)), mode="constant", constant_values=-80.0)


def frames_for_clip(cfg: dict) -> int:
    sr = int(cfg["sample_rate"])
    hop = int(cfg["hop_length"])
    seconds = float(cfg["clip_seconds"])
    return max(1, int(round(seconds * sr / hop)))


class VocalReverbRouter:
    """Load vocal_reverb.pt and predict dry/wet for an audio file."""

    def __init__(self, checkpoint: Path, device: str | None = None, status=print):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        status(f"  loading {Path(checkpoint).name} ({self.device}) ...")
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(saved.get("config") or {})
        self.cfg = cfg
        self.classes = tuple(saved.get("classes") or CLASS_NAMES)
        channels = tuple(cfg.get("channels") or DEFAULT_CONFIG["channels"])
        self.model = MelReverbNet(
            n_mels=int(cfg["n_mels"]),
            channels=channels,
            n_classes=len(self.classes),
        ).to(self.device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.target_frames = frames_for_clip(cfg)
        self._n_crops = 3

    def _crops_for_file(self, filename: str) -> list[np.ndarray]:
        """Decode + log-mel crops for one file (CPU)."""
        cfg = self.cfg
        audio = load_mono(filename, int(cfg["sample_rate"]))
        logmel = audio_to_logmel(
            audio,
            sample_rate=int(cfg["sample_rate"]),
            n_mels=int(cfg["n_mels"]),
            n_fft=int(cfg["n_fft"]),
            hop_length=int(cfg["hop_length"]),
        )
        n_frames = logmel.shape[1]
        if n_frames <= self.target_frames:
            return [crop_or_pad_logmel(logmel, self.target_frames)]
        starts = np.linspace(
            0,
            n_frames - self.target_frames,
            num=self._n_crops,
            dtype=np.int64,
        )
        return [
            logmel[:, int(start) : int(start) + self.target_frames]
            for start in starts
        ]

    def _result_from_logits(self, logits: torch.Tensor) -> dict:
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        class_probs = dict(zip(self.classes, probs, strict=True))
        dry_p = float(class_probs.get("dry", 0.0))
        wet_p = float(class_probs.get("wet", 0.0))
        if wet_p >= dry_p:
            label = "wet"
            confidence = wet_p
        else:
            label = "dry"
            confidence = dry_p
        return {
            "reverb": label,
            "reverb_confidence": confidence,
            "wet": wet_p,
            "dry": dry_p,
        }

    @torch.inference_mode()
    def predict(self, filename: str) -> dict:
        crops = self._crops_for_file(filename)
        x = torch.from_numpy(np.stack(crops, axis=0)).unsqueeze(1)
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
        logits = self.model(x).mean(dim=0)
        return self._result_from_logits(logits)

    @torch.inference_mode()
    def predict_many(
        self,
        filenames: list[str],
        *,
        gpu_batch_size: int = 32,
        num_workers: int = 8,
    ) -> list[dict | BaseException]:
        """Parallel CPU decode/mel, then batched GPU (or CPU) forwards.

        Returns one entry per input path: result dict, or the exception raised.
        """
        from concurrent.futures import ThreadPoolExecutor

        n = len(filenames)
        out: list[dict | BaseException | None] = [None] * n
        if n == 0:
            return []

        def _prep(item: tuple[int, str]):
            index, path = item
            try:
                return index, self._crops_for_file(path), None
            except BaseException as exc:
                return index, None, exc

        workers = max(1, min(num_workers, n))
        prepared: list[tuple[int, list[np.ndarray]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, crops, err in pool.map(_prep, enumerate(filenames)):
                if err is not None:
                    out[index] = err
                else:
                    prepared.append((index, crops))

        if not prepared:
            return [e if e is not None else RuntimeError("no crops") for e in out]

        crop_tensors: list[np.ndarray] = []
        owners: list[int] = []
        for index, crops in prepared:
            for crop in crops:
                crop_tensors.append(crop)
                owners.append(index)

        logit_sum: dict[int, torch.Tensor] = {}
        logit_count: dict[int, int] = {}
        bs = max(1, int(gpu_batch_size))

        for start in range(0, len(crop_tensors), bs):
            chunk = crop_tensors[start : start + bs]
            own = owners[start : start + bs]
            x = torch.from_numpy(np.stack(chunk, axis=0)).unsqueeze(1)
            if self.device == "cuda":
                x = x.pin_memory().to(self.device, non_blocking=True)
            else:
                x = x.to(self.device)
            logits = self.model(x)
            for j, index in enumerate(own):
                row = logits[j].detach()
                if index in logit_sum:
                    logit_sum[index] = logit_sum[index] + row
                    logit_count[index] += 1
                else:
                    logit_sum[index] = row
                    logit_count[index] = 1

        for index, total in logit_sum.items():
            out[index] = self._result_from_logits(total / logit_count[index])

        return [
            e if e is not None else RuntimeError("reverb predict missing")
            for e in out
        ]


def ensure_vocal_reverb(model_dir: Path, status=print) -> Path:
    """Return path to vocal_reverb.pt or exit with train instructions."""
    model_dir = Path(model_dir)
    path = model_dir / MODEL_NAME
    if path.exists() and path.stat().st_size > 0:
        return path

    data_hint = Path(__file__).resolve().parent / "reverb_data"
    raise SystemExit(
        f"\nERROR: missing {path.name}\n"
        f"  expected: {path}\n\n"
        f"Train it from dry/wet vocal packs:\n"
        f"  1. Put audio in:\n"
        f"       {data_hint / 'dry'}\n"
        f"       {data_hint / 'wet'}\n"
        f"  2. Activate genre_gender_tagger\\venv and run:\n"
        f"       python train_vocal_reverb.py\n"
        f"  3. Re-run Gender tagging.\n"
    )


def load_vocal_reverb(model_dir: Path, status=print) -> VocalReverbRouter:
    checkpoint = ensure_vocal_reverb(model_dir, status=status)
    return VocalReverbRouter(checkpoint, status=status)
