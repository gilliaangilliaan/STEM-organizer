"""FFmpeg-backed waveform extraction and FFplay lifecycle management."""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Literal

import psutil

WaveformPeaks = tuple[tuple[float, float], ...]
AudioEvent = tuple[int, Literal["waveform", "duration", "error"], object]

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_STARTUPINFO = None
if sys.platform == "win32":
    _STARTUPINFO = subprocess.STARTUPINFO()
    _STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW


@dataclass(frozen=True, slots=True)
class AudioTools:
    ffmpeg: Path
    ffplay: Path
    ffprobe: Path


def resolve_audio_tools(project_root: Path | None = None) -> AudioTools | None:
    """Resolve an explicit bundle, STEM's shared tools, then PATH."""
    if project_root is None:
        try:
            from ffmpeg_bootstrap import (
                ffmpeg_path,
                ffplay_path,
                ffprobe_path,
            )

            shared = {
                "ffmpeg": ffmpeg_path(),
                "ffplay": ffplay_path(),
                "ffprobe": ffprobe_path(),
            }
            if all(shared.values()):
                return AudioTools(
                    ffmpeg=Path(shared["ffmpeg"]),
                    ffplay=Path(shared["ffplay"]),
                    ffprobe=Path(shared["ffprobe"]),
                )
        except ImportError:
            pass

    root = project_root or Path(__file__).resolve().parents[1]
    bundled = root / "ffmpeg"
    suffix = ".exe" if sys.platform == "win32" else ""
    bundled_paths = {
        name: bundled / f"{name}{suffix}"
        for name in ("ffmpeg", "ffplay", "ffprobe")
    }
    if all(path.is_file() for path in bundled_paths.values()):
        return AudioTools(**bundled_paths)

    discovered = {name: shutil.which(name) for name in bundled_paths}
    if all(discovered.values()):
        return AudioTools(
            ffmpeg=Path(discovered["ffmpeg"]),
            ffplay=Path(discovered["ffplay"]),
            ffprobe=Path(discovered["ffprobe"]),
        )
    return None


def reduce_pcm_peaks(samples: array, target_bins: int = 900) -> WaveformPeaks:
    """Reduce mono float PCM to normalized min/max waveform bins."""
    if not samples or target_bins <= 0:
        return ()
    bucket_size = max(1, (len(samples) + target_bins - 1) // target_bins)
    raw: list[tuple[float, float]] = []
    maximum = 0.0
    for start in range(0, len(samples), bucket_size):
        chunk = samples[start : start + bucket_size]
        low = min(chunk)
        high = max(chunk)
        raw.append((low, high))
        maximum = max(maximum, abs(low), abs(high))
    if maximum <= 1e-12:
        return tuple((0.0, 0.0) for _ in raw)
    scale = 1.0 / maximum
    return tuple((low * scale, high * scale) for low, high in raw)


class WaveformCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self._items: OrderedDict[tuple[str, int, int], WaveformPeaks] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)

    def get(self, path: Path) -> WaveformPeaks | None:
        key = self.key(path)
        with self._lock:
            peaks = self._items.get(key)
            if peaks is not None:
                self._items.move_to_end(key)
            return peaks

    def put(self, path: Path, peaks: WaveformPeaks) -> None:
        key = self.key(path)
        with self._lock:
            self._items[key] = peaks
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


class AudioPreviewService:
    """Own waveform jobs and the single FFplay preview process."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        tools: AudioTools | None = None,
        cache: WaveformCache | None = None,
    ) -> None:
        self.tools = tools if tools is not None else resolve_audio_tools(project_root)
        self.cache = cache or WaveformCache()
        self.events: SimpleQueue[AudioEvent] = SimpleQueue()
        self.generation = 0
        self.active_path: Path | None = None
        self.duration = 0.0
        self._waveform_process: subprocess.Popen[bytes] | None = None
        self._waveform_lock = threading.Lock()
        self._probe_process: subprocess.Popen[bytes] | None = None
        self._probe_lock = threading.Lock()
        self._play_process: subprocess.Popen[bytes] | None = None
        self._paused = False
        self._position_seconds = 0.0
        self._play_started_at: float | None = None

    @property
    def available(self) -> bool:
        return self.tools is not None

    @property
    def unavailable_message(self) -> str:
        return "Add ffmpeg, ffplay, and ffprobe to the ffmpeg folder."

    def load(self, path: Path) -> int:
        """Stop current audio and asynchronously load one waveform."""
        self.generation += 1
        generation = self.generation
        self.stop()
        self._cancel_waveform()
        self._cancel_probe()
        self.active_path = path
        self.duration = 0.0

        if not self.available:
            self.events.put((generation, "error", self.unavailable_message))
            return generation
        if not path.is_file():
            self.events.put((generation, "error", "Audio file is missing."))
            return generation
        threading.Thread(
            target=self._probe_duration,
            args=(generation, path),
            daemon=True,
        ).start()
        try:
            cached = self.cache.get(path)
        except OSError as exc:
            self.events.put((generation, "error", str(exc)))
            return generation
        if cached is not None:
            self.events.put((generation, "waveform", cached))
            return generation

        threading.Thread(
            target=self._extract_waveform,
            args=(generation, path),
            daemon=True,
        ).start()
        return generation

    def _probe_duration(self, generation: int, path: Path) -> None:
        assert self.tools is not None
        command = [
            str(self.tools.ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUPINFO,
            )
            with self._probe_lock:
                if generation != self.generation:
                    self._terminate_process(process)
                    return
                self._probe_process = process
            stdout, _stderr = process.communicate(timeout=10)
            if generation != self.generation or process.returncode:
                return
            duration = float(stdout.decode("ascii", errors="ignore").strip())
            if duration > 0:
                self.duration = duration
                self.events.put((generation, "duration", duration))
        except subprocess.TimeoutExpired:
            if process is not None:
                self._terminate_process(process)
        except (OSError, ValueError):
            pass
        finally:
            with self._probe_lock:
                if self._probe_process is process:
                    self._probe_process = None

    def _extract_waveform(self, generation: int, path: Path) -> None:
        assert self.tools is not None
        command = [
            str(self.tools.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-ac",
            "1",
            "-ar",
            "4000",
            "-f",
            "f32le",
            "pipe:1",
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUPINFO,
            )
            with self._waveform_lock:
                if generation != self.generation:
                    process.terminate()
                    return
                self._waveform_process = process
            stdout, stderr = process.communicate()
            if generation != self.generation:
                return
            if process.returncode:
                detail = stderr.decode("utf-8", errors="replace").strip()
                self.events.put(
                    (generation, "error", detail or "Unable to decode this audio file.")
                )
                return
            samples = array("f")
            samples.frombytes(stdout)
            if sys.byteorder != "little":
                samples.byteswap()
            peaks = reduce_pcm_peaks(samples)
            self.cache.put(path, peaks)
            self.events.put((generation, "waveform", peaks))
        except (OSError, ValueError) as exc:
            if generation == self.generation:
                self.events.put((generation, "error", str(exc)))
        finally:
            with self._waveform_lock:
                if self._waveform_process is process:
                    self._waveform_process = None

    def _cancel_waveform(self) -> None:
        with self._waveform_lock:
            process = self._waveform_process
            self._waveform_process = None
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    def _cancel_probe(self) -> None:
        with self._probe_lock:
            process = self._probe_process
            self._probe_process = None
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        try:
            process.terminate()
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def play_pause(self) -> Literal["playing", "paused", "stopped"]:
        if not self.available or self.active_path is None or not self.active_path.is_file():
            return "stopped"
        process = self._play_process
        if process is not None and process.poll() is None:
            try:
                handle = psutil.Process(process.pid)
                if self._paused:
                    handle.resume()
                    self._paused = False
                    self._play_started_at = time.monotonic()
                    return "playing"
                self._position_seconds = self.playback_position()
                handle.suspend()
                self._paused = True
                self._play_started_at = None
                return "paused"
            except (psutil.Error, OSError):
                self.stop()
                return "stopped"
        return self._start_playback(self._position_seconds)

    def _start_playback(
        self,
        start_position: float = 0.0,
    ) -> Literal["playing", "stopped"]:
        assert self.tools is not None and self.active_path is not None
        self.stop()
        start_position = max(0.0, start_position)
        command = [
            str(self.tools.ffplay),
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-ss",
            f"{start_position:.3f}",
            str(self.active_path),
        ]
        try:
            self._play_process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUPINFO,
            )
            self._paused = False
            self._position_seconds = start_position
            self._play_started_at = time.monotonic()
            return "playing"
        except OSError:
            self._play_process = None
            return "stopped"

    def playback_state(self) -> Literal["playing", "paused", "stopped"]:
        process = self._play_process
        if process is None or process.poll() is not None:
            self._play_process = None
            self._paused = False
            self._position_seconds = 0.0
            self._play_started_at = None
            return "stopped"
        return "paused" if self._paused else "playing"

    def playback_position(self) -> float:
        if self._play_started_at is None:
            return self._position_seconds
        return self._position_seconds + max(0.0, time.monotonic() - self._play_started_at)

    def seek(self, seconds: float) -> float:
        if self.active_path is None:
            return 0.0
        state = self.playback_state()
        limit = self.duration if self.duration > 0 else float("inf")
        target = max(0.0, min(limit, self.playback_position() + seconds))
        if state == "stopped":
            self._position_seconds = target
            return target

        was_paused = state == "paused"
        if self._start_playback(target) != "playing":
            return self._position_seconds
        if was_paused and self._play_process is not None:
            try:
                psutil.Process(self._play_process.pid).suspend()
                self._paused = True
                self._position_seconds = target
                self._play_started_at = None
            except (psutil.Error, OSError):
                self.stop()
        return self.playback_position()

    def stop(self) -> None:
        process = self._play_process
        self._play_process = None
        self._paused = False
        self._position_seconds = 0.0
        self._play_started_at = None
        if process is not None and process.poll() is None:
            self._terminate_process(process)

    def reset(self) -> None:
        self.generation += 1
        self.stop()
        self._cancel_waveform()
        self._cancel_probe()
        self.active_path = None
        self.duration = 0.0

    def shutdown(self) -> None:
        self.reset()
