"""Audio decoding, resampling and writing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .types import Audio

DEFAULT_SR = 44100


def load(path: str | Path, sr: int = DEFAULT_SR, channels: int = 2) -> Audio:
    """Decode any audio file to float32 at `sr`.

    Uses ffmpeg when available (handles mp3/m4a/ogg/flac/wav and honors MP3
    gapless info so timestamps are exact), otherwise falls back to soundfile.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if shutil.which("ffmpeg"):
        return _load_ffmpeg(path, sr, channels)
    return _load_soundfile(path, sr, channels)


def _load_ffmpeg(path: Path, sr: int, channels: int) -> Audio:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    data = data.reshape(-1, channels).T.copy()
    return Audio(data, sr)


def _load_soundfile(path: Path, sr: int, channels: int) -> Audio:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T
    if data.shape[0] == 1 and channels == 2:
        data = np.repeat(data, 2, axis=0)
    elif data.shape[0] > channels:
        data = data[:channels] if channels > 1 else data.mean(axis=0, keepdims=True)
    return resample(Audio(data, file_sr), sr)


def resample(audio: Audio, sr: int) -> Audio:
    if audio.sr == sr:
        return audio
    import librosa

    return Audio(librosa.resample(audio.samples, orig_sr=audio.sr, target_sr=sr, axis=-1), sr)


def to_mono(audio: Audio, sr: int | None = None) -> np.ndarray:
    a = resample(audio, sr) if sr else audio
    return a.mono()


def save(audio: Audio, path: str | Path) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.samples.T, audio.sr, subtype="FLOAT")
    return path
