"""Model-free drum transcription baseline (kick / snare / hi-hat).

Detect onsets on a band-normalised spectral-flux envelope, then explain the flux
at each onset as a non-negative mix of three spectral templates (NNLS). This
lets simultaneous hits (snare + hi-hat on the backbeat) both be reported.
A neural ADT model (ADTOF, Noise-to-Notes) should replace this where licensing
allows; see research/03.
"""

from __future__ import annotations

import numpy as np

from ..audio import resample
from ..types import Audio, NoteEvent

GM = {"kick": 36, "snare": 38, "hihat": 42}

# Band edges (Hz) and the rough share of onset flux each drum puts in each band.
BANDS = [(30, 100), (100, 250), (250, 1000), (1000, 3000), (3000, 7000), (7000, 11000)]
TEMPLATES = {
    "kick": [1.0, 0.6, 0.15, 0.05, 0.02, 0.01],
    "snare": [0.1, 0.6, 0.8, 0.7, 0.5, 0.3],
    "hihat": [0.0, 0.0, 0.02, 0.1, 0.5, 1.0],
}


def transcribe_drums(
    audio: Audio,
    sr: int = 22050,
    hop: int = 128,
    delta: float = 0.08,
    min_spacing: float = 0.04,
    class_threshold: float = 0.4,
    coincide_ratio: float = 0.2,
) -> list[NoteEvent]:
    import librosa
    from scipy.optimize import nnls

    y = resample(audio, sr).mono()
    if not np.any(y):
        return []
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    logS = np.log1p(100 * S / (S.max() + 1e-9))
    flux = np.maximum(0.0, np.diff(logS, axis=1, prepend=logS[:, :1]))
    band_flux = np.stack([flux[(freqs >= lo) & (freqs < hi)].sum(axis=0) for lo, hi in BANDS])
    # Normalise bands so quiet hi-hats count as much as loud kicks for detection.
    scale = np.percentile(band_flux, 99, axis=1, keepdims=True) + 1e-9
    norm = band_flux / scale
    env = norm.sum(axis=0)
    env = env / (env.max() + 1e-9)

    wait = max(1, int(min_spacing * sr / hop))
    peaks = librosa.util.peak_pick(env, pre_max=wait, post_max=wait, pre_avg=int(0.1 * sr / hop),
                                   post_avg=int(0.1 * sr / hop), delta=delta, wait=wait)
    if len(peaks) == 0:
        return []

    names = list(TEMPLATES)
    T = np.array([TEMPLATES[n] for n in names]).T  # (bands, drums)
    T = T / np.linalg.norm(T, axis=0, keepdims=True)
    span = max(1, int(0.03 * sr / hop))
    acts = np.zeros((len(peaks), len(names)))
    for i, p in enumerate(peaks):
        v = norm[:, p:p + span].sum(axis=1)
        acts[i], _ = nnls(T, v)

    notes: list[NoteEvent] = []
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    for j, name in enumerate(names):
        a = acts[:, j]
        ref = np.percentile(a[a > 0], 95) if np.any(a > 0) else 0
        if ref <= 0:
            continue
        for t, v in zip(times, a):
            if v >= class_threshold * ref and v >= coincide_ratio * acts[times == t].max():
                vel = int(np.clip(100 + 25 * np.log2(v / ref), 20, 127))
                notes.append(NoteEvent(GM[name], float(t), float(t) + 0.05, vel, float(min(1.0, v / ref)),
                                       None, ["drums-dsp"]))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes
