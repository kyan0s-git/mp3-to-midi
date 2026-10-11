"""Beat / downbeat tracking -> the tempo map the MIDI file is built on."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import resample
from ..types import Audio


@dataclass
class BeatGrid:
    beats: np.ndarray
    downbeats: np.ndarray
    beats_per_bar: int = 4
    tempo: float = 120.0
    method: str = ""
    notes: list[str] = field(default_factory=list)


def track_beats(audio: Audio, method: str = "librosa-beats", rhythm_audio: Audio | None = None) -> BeatGrid:
    """`rhythm_audio` (drums+bass stems) gives cleaner onsets than the full mix when available."""
    if method == "beat-this":
        try:
            return _beat_this(audio)
        except Exception as exc:  # noqa: BLE001 - weights unavailable, etc.
            grid = _librosa(rhythm_audio or audio)
            grid.notes.append(f"beat-this failed ({exc.__class__.__name__}: {exc}); used librosa")
            return grid
    return _librosa(rhythm_audio or audio)


def _librosa(audio: Audio, sr: int = 22050, hop: int = 256) -> BeatGrid:
    import librosa

    y = resample(audio, sr).mono()
    if not np.any(y):
        return BeatGrid(np.array([]), np.array([]), method="librosa-beats")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    if len(beat_times) < 2:
        return BeatGrid(beat_times, beat_times[:1], tempo=tempo or 120.0, method="librosa-beats")

    # Downbeats: the bar phase whose beats carry the most low-frequency (kick/bass) onset energy.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    low = librosa.fft_frequencies(sr=sr, n_fft=2048) < 200
    low_env = np.maximum(0, np.diff(np.log1p(S[low]), axis=1, prepend=0)).sum(axis=0)
    strength = np.array([low_env[max(0, f - 2):f + 3].max() for f in beats])
    meter = _meter(strength)
    phase = max(range(meter), key=lambda p: strength[p::meter].mean())
    return BeatGrid(beat_times, beat_times[phase::meter], meter, tempo, "librosa-beats")


def _meter(strength: np.ndarray) -> int:
    """3 or 4 beats per bar from the periodicity of per-beat accent strength.

    Comparing autocorrelation at lags 3 and 4 copes with kick-on-1-and-3
    patterns, which look like 2/4 to a simple "strongest phase" test.
    """
    s = strength - strength.mean()
    if len(s) < 9 or not np.any(s):
        return 4
    ac = {lag: float(np.dot(s[:-lag], s[lag:]) / (len(s) - lag)) for lag in (3, 4)}
    return 3 if ac[3] > 1.2 * max(ac[4], 0.0) and ac[3] > 0 else 4


def _beat_this(audio: Audio) -> BeatGrid:
    from beat_this.inference import Audio2Beats

    a2b = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    a = resample(audio, 22050)
    beats, downbeats = a2b(a.mono(), a.sr)
    beats, downbeats = np.asarray(beats), np.asarray(downbeats)
    tempo = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else 120.0
    meter = 4
    if len(downbeats) > 2:
        counts = [np.sum((beats >= a0) & (beats < a1)) for a0, a1 in zip(downbeats[:-1], downbeats[1:])]
        meter = int(np.bincount(counts).argmax()) or 4
    return BeatGrid(beats, downbeats, meter, tempo, "beat-this")


def detect_onsets(audio: Audio, sr: int = 22050, hop: int = 128, delta: float = 0.1) -> np.ndarray:
    """Note onset times (seconds) from spectral flux, for splitting re-attacked notes."""
    import librosa

    y = resample(audio, sr).mono()
    if not np.any(y):
        return np.array([])
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, lag=2, max_size=3)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, delta=delta,
                                        wait=int(0.05 * sr / hop), normalize=True)
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)
