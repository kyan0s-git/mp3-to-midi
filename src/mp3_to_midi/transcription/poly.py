"""Polyphonic note transcription.

Two back-ends with the same output (a list of NoteEvent):

* ``salience`` - no model. CQT + harmonic summation with iterative
  estimation-and-cancellation (Klapuri-style): find the strongest pitch in each
  frame, remove its harmonics from the spectrum, repeat. Good enough for clean
  stems with a few voices (backing vocals, pads); weak on dense mixes.
* ``basic-pitch`` - Spotify's Basic Pitch (Apache-2.0), when installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from ..audio import resample, save
from ..types import Audio, NoteEvent

BINS_PER_SEMITONE = 3


@dataclass
class PitchCandidates:
    times: np.ndarray  # (frames,)
    activation: np.ndarray  # (frames, 128) salience per MIDI pitch, 0 where absent
    fine_pitch: np.ndarray  # (frames, 128) fractional MIDI pitch of the detected peak


def compressed_cqt(audio: Audio, fmin_midi: int, fmax_midi: int, n_harmonics: int = 6, sr: int = 22050,
                   hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """sqrt-compressed CQT magnitude, BINS_PER_SEMITONE bins per semitone, bin 0 centred on `fmin_midi`.

    Covers enough range above `fmax_midi` for `n_harmonics` harmonics (Nyquist permitting).
    """
    import librosa

    y = resample(audio, sr).mono()
    bps = BINS_PER_SEMITONE
    top_midi = min(fmax_midi + 12 * np.log2(n_harmonics) + 1, 12 * np.log2(0.45 * sr / 440) + 69)
    n_bins = int((top_midi - fmin_midi) * bps)
    fmin_hz = 440.0 * 2 ** ((fmin_midi - 69) / 12) * 2 ** (-1 / (2 * 12 * bps))  # centre bins on semitones
    X = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=fmin_hz, n_bins=n_bins, bins_per_octave=12 * bps))
    X = X / (X.max() + 1e-9)
    X = np.sqrt(X)  # mild compression so weak harmonics still count
    return X, np.arange(X.shape[1]) * hop / sr


def harmonic_offsets(n_harmonics: int = 6) -> tuple[list[int], np.ndarray]:
    offsets = [int(round(12 * BINS_PER_SEMITONE * np.log2(h))) for h in range(1, n_harmonics + 1)]
    weights = np.array([0.85 ** (h - 1) for h in range(1, n_harmonics + 1)])
    return offsets, weights


def harmonic_sum(X: np.ndarray, n_cand: int, n_harmonics: int = 6) -> np.ndarray:
    offsets, weights = harmonic_offsets(n_harmonics)
    n_bins = X.shape[0]
    sal = np.zeros((n_cand, X.shape[1]), dtype=np.float32)
    for off, w in zip(offsets, weights):
        hi = min(n_cand, n_bins - off)
        if hi > 0:
            sal[:hi] += w * X[off:off + hi]
    return sal


def multipitch_salience(
    audio: Audio,
    fmin_midi: int = 36,
    fmax_midi: int = 84,
    max_voices: int = 4,
    n_harmonics: int = 6,
    sr: int = 22050,
    hop: int = 256,
    rel_threshold: float = 0.15,
    fundamental_floor: float = 0.08,
    cancel: float = 0.9,
) -> PitchCandidates:
    """Frame-wise multi-f0 candidates via harmonic sum + iterative cancellation."""
    bps = BINS_PER_SEMITONE
    X, times = compressed_cqt(audio, fmin_midi, fmax_midi, n_harmonics, sr, hop)
    n_bins, n_frames = X.shape
    frame_energy = X.max(axis=0)
    n_cand = (fmax_midi - fmin_midi) * bps + 1
    offsets, _ = harmonic_offsets(n_harmonics)

    activation = np.zeros((n_frames, 128), dtype=np.float32)
    fine = np.zeros((n_frames, 128), dtype=np.float32)
    first_peak = None
    Xw = X.copy()
    cols = np.arange(n_frames)
    blocked = np.zeros((n_cand, n_frames), dtype=bool)
    for _ in range(max_voices):
        sal = harmonic_sum(Xw, n_cand, n_harmonics)
        # Require real energy at the fundamental itself (suppresses sub-octave ghosts).
        sal *= Xw[:n_cand] > fundamental_floor * np.maximum(frame_energy, 1e-9)
        # Imperfect cancellation leaves residue right next to accepted pitches;
        # voices a semitone apart are rare, so block that neighbourhood.
        sal[blocked] = 0
        k = sal.argmax(axis=0)
        val = sal[k, cols]
        if first_peak is None:
            first_peak = val.copy()
        ok = (val > rel_threshold * first_peak) & (val > 0.05)
        pitch = fmin_midi + k / bps
        semis = np.clip(np.round(pitch).astype(int), 0, 127)
        f = cols[ok]
        activation[f, semis[ok]] = np.maximum(activation[f, semis[ok]], val[ok])
        fine[f, semis[ok]] = pitch[ok]
        for d in range(-bps - 1, bps + 2):
            b = k + d
            valid = ok & (b >= 0) & (b < n_cand)
            blocked[b[valid], cols[valid]] = True
        # Cancel the harmonics of the accepted pitches (+-1 bin for vibrato/detuning).
        for off in offsets:
            for d in (-1, 0, 1):
                b = k + off + d
                valid = ok & (b >= 0) & (b < n_bins)
                Xw[b[valid], cols[valid]] *= 1 - cancel
    return PitchCandidates(times, activation, fine)


def candidates_to_notes(
    cand: PitchCandidates,
    threshold: float = 0.4,
    min_note: float = 0.08,
    max_gap: float = 0.035,
    reattack_ratio: float = 1.6,
    source: str = "salience",
) -> list[NoteEvent]:
    """Group per-pitch activations into notes; split repeated notes at re-attacks."""
    hop = float(cand.times[1] - cand.times[0]) if len(cand.times) > 1 else 0.01
    from scipy.ndimage import maximum_filter1d

    act = cand.activation
    if not act.size or act.max() <= 0:
        return []
    # Normalise against a sliding local peak (~3 s) rather than the global peak,
    # so quiet sections are not thresholded away.
    local_peak = maximum_filter1d(act.max(axis=1), size=max(1, int(3.0 / hop)), mode="nearest")
    act = act / np.maximum(local_peak, 0.25 * act.max())[:, None]
    gap = max(1, int(round(max_gap / hop)))
    min_frames = max(1, int(round(min_note / hop)))
    notes: list[NoteEvent] = []
    for p in np.flatnonzero(act.max(axis=0) > threshold):
        on = act[:, p] > threshold
        # close small gaps
        idx = np.flatnonzero(on)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap + 1:
                on[a:b] = True
        d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            for s2, e2 in _split_reattacks(act[s:e, p], s, reattack_ratio, min_frames):
                if e2 - s2 < min_frames:
                    continue
                strength = float(np.median(act[s2:e2, p]))
                notes.append(NoteEvent(
                    int(p), float(cand.times[s2]), float(cand.times[e2 - 1] + hop),
                    int(np.clip(40 + 87 * strength, 1, 127)), float(np.clip(strength, 0, 1)), None, [source],
                ))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def _split_reattacks(x: np.ndarray, offset: int, ratio: float, min_frames: int) -> list[tuple[int, int]]:
    cuts = [0]
    run_min = x[0]
    for i in range(1, len(x)):
        run_min = min(run_min, x[i])
        if x[i] > ratio * run_min and x[i - 1] <= run_min * 1.05 and i - cuts[-1] >= min_frames:
            cuts.append(i)
            run_min = x[i]
    cuts.append(len(x))
    return [(offset + a, offset + b) for a, b in zip(cuts[:-1], cuts[1:])]


def salience_notes(audio: Audio, fmin_midi: int = 36, fmax_midi: int = 84, max_voices: int = 4,
                   threshold: float = 0.4) -> list[NoteEvent]:
    cand = multipitch_salience(audio, fmin_midi, fmax_midi, max_voices)
    return candidates_to_notes(cand, threshold=threshold)


def basic_pitch_notes(audio: Audio, fmin_midi: int | None = None, fmax_midi: int | None = None,
                      onset_threshold: float = 0.5, frame_threshold: float = 0.3,
                      min_note_ms: float = 80.0) -> list[NoteEvent]:
    """Run Spotify Basic Pitch. Pitch bends are kept as cents curves."""
    import contextlib
    import io

    # basic_pitch logs a warning for every runtime it cannot find at import time.
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        from basic_pitch.inference import predict
    finally:
        root.setLevel(level)

    to_hz = (lambda m: None if m is None else 440.0 * 2 ** ((m - 69) / 12))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "in.wav")
        save(Audio(audio.mono()[None], audio.sr), path)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, events = predict(
                path, onset_threshold=onset_threshold, frame_threshold=frame_threshold,
                minimum_note_length=min_note_ms, minimum_frequency=to_hz(fmin_midi),
                maximum_frequency=to_hz(fmax_midi), multiple_pitch_bends=True, melodia_trick=True,
            )
    notes = []
    for start, end, pitch, amp, bends in events:
        bend = None
        if bends:
            # Basic Pitch bends: 3 bins per semitone at its 86.13 Hz frame rate.
            fr = 22050 / 256
            bend = [(float(start + i / fr), float(b * 100.0 / 3.0)) for i, b in enumerate(bends)]
        notes.append(NoteEvent(int(pitch), float(start), float(end), int(np.clip(amp * 127 * 1.5, 1, 127)),
                               float(np.clip(amp * 1.5, 0, 1)), bend, ["basic-pitch"]))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def poly_notes(audio: Audio, method: str = "salience", fmin_midi: int = 36, fmax_midi: int = 84,
               max_voices: int = 4) -> list[NoteEvent]:
    if method == "basic-pitch":
        return basic_pitch_notes(audio, fmin_midi, fmax_midi)
    return salience_notes(audio, fmin_midi, fmax_midi, max_voices)
