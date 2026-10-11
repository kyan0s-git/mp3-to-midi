"""Monophonic f0 tracking: pYIN (always available) and CREPE (optional)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import to_mono
from ..types import Audio

# Typical ranges per role, in Hz.
RANGES = {
    "voice": (65.0, 1100.0),  # ~C2..C6
    "bass": (30.0, 400.0),  # ~B0..G4
    "default": (50.0, 2000.0),
}


@dataclass
class F0Curve:
    times: np.ndarray  # seconds, frame centres
    f0: np.ndarray  # Hz, NaN where unvoiced
    voiced_prob: np.ndarray  # 0..1
    rms: np.ndarray  # frame RMS (linear) of the analysed signal

    @property
    def hop(self) -> float:
        return float(self.times[1] - self.times[0]) if len(self.times) > 1 else 0.01

    def midi(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return 69.0 + 12.0 * np.log2(self.f0 / 440.0)


def track(audio: Audio, method: str = "pyin", role: str = "voice") -> F0Curve:
    fmin, fmax = RANGES.get(role, RANGES["default"])
    if method == "crepe":
        return _crepe(audio, fmin, fmax)
    return _pyin(audio, fmin, fmax)


def _frame_rms(y: np.ndarray, frame_length: int, hop: int, n_frames: int) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop, center=True)[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    return rms[:n_frames]


def _pyin(audio: Audio, fmin: float, fmax: float, sr: int = 16000, hop: int = 160) -> F0Curve:
    import librosa

    y = to_mono(audio, sr)
    # Need >= ~2 periods of fmin inside a frame.
    frame_length = 1024 if fmin >= 60 else 2048
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop,
        center=True, fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    f0 = np.where(voiced_flag, f0, np.nan)
    rms = _frame_rms(y, frame_length, hop, len(f0))
    return F0Curve(times, f0, np.nan_to_num(voiced_prob), rms)


def _crepe(audio: Audio, fmin: float, fmax: float, sr: int = 16000, hop: int = 160) -> F0Curve:
    import torch
    import torchcrepe

    y = to_mono(audio, sr)
    x = torch.from_numpy(y)[None]
    f0, periodicity = torchcrepe.predict(
        x, sr, hop, fmin, fmax, model="full", batch_size=512,
        device="cuda" if torch.cuda.is_available() else "cpu", return_periodicity=True,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)[0].numpy()
    f0 = torchcrepe.filter.mean(f0, 3)[0].numpy()
    f0 = np.where(periodicity > 0.21, f0, np.nan)
    times = np.arange(len(f0)) * hop / sr
    rms = _frame_rms(y, 1024, hop, len(f0))
    return F0Curve(times, f0, periodicity, rms)
