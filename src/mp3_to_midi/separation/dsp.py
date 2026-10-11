"""Model-free separation fallback.

Classic signal processing, far below neural separators (see research/02), but
it needs nothing beyond numpy/librosa and keeps the pipeline runnable anywhere:

* drums  = percussive part of harmonic/percussive separation (HPSS)
* bass   = harmonic part, low-passed
* vocals = harmonic part, centre-panned time-frequency bins in the vocal band
           (stereo panning-index mask; no effect on mono input)
* other  = harmonic part minus bass and vocals
"""

from __future__ import annotations

import numpy as np

from ..types import Audio

N_FFT = 2048
HOP = 512


def _stft(x: np.ndarray) -> np.ndarray:
    import librosa
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP)


def _istft(X: np.ndarray, length: int) -> np.ndarray:
    import librosa
    return librosa.istft(X, hop_length=HOP, length=length).astype(np.float32)


def separate(audio: Audio) -> dict[str, Audio]:
    import librosa

    sr, n = audio.sr, audio.samples.shape[1]
    specs = [_stft(ch) for ch in audio.samples]
    H, P = zip(*(librosa.decompose.hpss(S, margin=(1.0, 2.0)) for S in specs))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    bass_band = (freqs < 250)[:, None].astype(np.float32)
    vocal_band = ((freqs > 150) & (freqs < 7000))[:, None].astype(np.float32)
    if len(H) >= 2:
        L, R = H[0], H[1]
        # Panning-index similarity: 1 where L and R are equal (centre), 0 when hard-panned.
        sim = 2 * np.abs(L * np.conj(R)) / (np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-12)
        centre = (sim ** 4) * vocal_band
    else:
        centre = vocal_band

    out: dict[str, list[np.ndarray]] = {"drums": [], "bass": [], "vocals": [], "other": []}
    for Hc, Pc in zip(H, P):
        bass = Hc * bass_band
        vocals = Hc * centre * (1 - bass_band)
        other = Hc - bass - vocals
        out["drums"].append(_istft(Pc, n))
        out["bass"].append(_istft(bass, n))
        out["vocals"].append(_istft(vocals, n))
        out["other"].append(_istft(other, n))
    return {k: Audio(np.stack(v), sr) for k, v in out.items()}


def split_mid_side(vocals: Audio) -> dict[str, Audio]:
    """Lead = centre of the vocal stem, backing = side content (panned harmonies).

    Weak heuristic: centre-panned harmonies stay in "lead". The harmony stage
    therefore also analyses the full vocal stem in the pitch domain.
    """
    return {"lead": Audio(vocals.mid()[None], vocals.sr), "backing": Audio(vocals.side()[None], vocals.sr)}
