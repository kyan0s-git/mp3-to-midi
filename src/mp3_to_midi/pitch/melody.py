"""Predominant-melody f0 tracking for signals with several voices.

pYIN assumes one voice; on a vocal stem with harmonies it can latch onto a
harmony. This tracker follows the most salient *continuous* voice instead:
harmonic-sum salience on a CQT, then a Viterbi path that charges for pitch
jumps (Melodia-style). `hybrid` keeps pYIN's finer pitch wherever both agree.
"""

from __future__ import annotations

import numpy as np

from ..types import Audio
from .f0 import RANGES, F0Curve


def _hz_to_midi(hz: float) -> float:
    return 69 + 12 * np.log2(hz / 440.0)


def predominant_f0(
    audio: Audio,
    role: str = "voice",
    sr: int = 22050,
    hop: int = 256,
    jump_cost: float = 0.12,
    max_jump_cost: float = 3.0,
    voicing_threshold: float = 0.25,
) -> F0Curve:
    from scipy.ndimage import maximum_filter1d

    from ..transcription.poly import BINS_PER_SEMITONE as BPS
    from ..transcription.poly import compressed_cqt, harmonic_sum

    fmin, fmax = RANGES.get(role, RANGES["default"])
    lo, hi = int(np.floor(_hz_to_midi(fmin))), int(np.ceil(_hz_to_midi(fmax)))
    X, times = compressed_cqt(audio, lo, hi, sr=sr, hop=hop)
    n_cand = (hi - lo) * BPS + 1
    sal = harmonic_sum(X, n_cand)
    sal *= X[:n_cand] > 0.08 * np.maximum(X.max(axis=0), 1e-9)
    frame_peak = sal.max(axis=0)
    S = sal / (frame_peak + 1e-9)

    # Viterbi: emission = log normalised salience, transition = -cost * |jump in bins| (capped).
    B, T = S.shape
    logE = np.log(S + 1e-3)
    d = np.abs(np.arange(B)[:, None] - np.arange(B)[None, :])
    trans = -np.minimum(jump_cost * d, max_jump_cost)
    back = np.zeros((T, B), dtype=np.int32)
    delta = logE[:, 0].copy()
    for t in range(1, T):
        scores = delta[:, None] + trans
        back[t] = scores.argmax(axis=0)
        delta = scores[back[t], np.arange(B)] + logE[:, t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # Parabolic interpolation around the chosen bin for sub-bin pitch.
    cols = np.arange(T)
    k = np.clip(path, 1, B - 2)
    a, b, c = sal[k - 1, cols], sal[k, cols], sal[k + 1, cols]
    denom = a - 2 * b + c
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
    midi = lo + (k + np.clip(shift, -0.5, 0.5)) / BPS

    # Voicing: the path's salience relative to a sliding ~3 s peak.
    on_path = sal[path, cols]
    local = maximum_filter1d(frame_peak, size=max(1, int(3.0 * sr / hop)), mode="nearest")
    conf = on_path / (local + 1e-9)
    voiced = conf > voicing_threshold
    f0 = np.where(voiced, 440.0 * 2 ** ((midi - 69) / 12), np.nan)
    rms = X.mean(axis=0) ** 2  # energy proxy on the same frame grid (sqrt-compressed CQT)
    return F0Curve(times, f0, np.clip(conf, 0, 1), rms)


def hybrid(pyin: F0Curve, melody: F0Curve, agree_semitones: float = 0.6) -> F0Curve:
    """pYIN pitch where it agrees with the predominant melody, the melody elsewhere.

    Output is on pYIN's time grid; a frame is voiced only if the predominant
    melody says so (pYIN alone may be following a harmony voice).
    """
    t = pyin.times
    m_mid = np.interp(t, melody.times, np.nan_to_num(melody.midi(), nan=-1000.0))
    m_voiced = np.interp(t, melody.times, np.isfinite(melody.f0).astype(float)) > 0.5
    p_mid = pyin.midi()
    use_pyin = np.isfinite(p_mid) & (np.abs(p_mid - m_mid) <= agree_semitones)
    mid = np.where(use_pyin, p_mid, m_mid)
    f0 = np.where(m_voiced, 440.0 * 2 ** ((mid - 69) / 12), np.nan)
    prob = np.where(m_voiced, np.maximum(pyin.voiced_prob, np.interp(t, melody.times, melody.voiced_prob)), 0.0)
    return F0Curve(t, f0, prob, pyin.rms)
