"""Monophonic f0 curve -> note events (singing voice / bass transcription).

Turning a continuous pitch curve into notes is a segmentation problem: where do
notes start and end, and which semitone does a scooped, vibrato-laden note
"mean"? The approach here:

1. Smooth the pitch curve (~120 ms) so vibrato does not trigger note changes.
2. Hysteresis: within a voiced run, start a new note only when the smoothed
   pitch stays > `switch_semitones` away from the current note for
   `switch_time` seconds.
3. Split repeated same-pitch notes at energy dips (syllables, re-plucks).
4. Note pitch = median of the stable middle of the segment, rounded.
5. Keep the deviation from that semitone as a pitch-bend curve (cents).
"""

from __future__ import annotations

import numpy as np

from ..types import NoteEvent
from .f0 import F0Curve


def _voiced_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _nan_moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    k = np.ones(width)
    num = np.convolve(v, k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def _energy_splits(rms_db: np.ndarray, lo: int, hi: int, depth_db: float, win: int, min_len: int) -> list[int]:
    """Indices inside [lo, hi) where energy dips by `depth_db` and recovers."""
    splits: list[int] = []
    if hi - lo < 2 * min_len:
        return splits
    seg = rms_db[lo:hi]
    for i in range(min_len, len(seg) - min_len):
        if seg[i] > seg[i - 1] or seg[i] > seg[i + 1]:
            continue  # not a local minimum
        left = seg[max(0, i - win):i].max()
        right = seg[i + 1:i + 1 + win].max()
        if left - seg[i] >= depth_db and right - seg[i] >= depth_db:
            if not splits or (lo + i) - splits[-1] >= min_len:
                splits.append(lo + i)
    return splits


def f0_to_notes(
    curve: F0Curve,
    *,
    min_note: float = 0.06,
    switch_semitones: float = 0.7,
    switch_time: float = 0.05,
    smooth_time: float = 0.12,
    merge_gap: float = 0.04,
    energy_split_db: float = 6.0,
    silence_db: float = -50.0,
    tail_db: float = 4.0,
    key_pitch_classes: set[int] | None = None,
    onsets: np.ndarray | list[float] | None = None,
    with_bend: bool = True,
    source: str = "f0",
) -> list[NoteEvent]:
    hop = curve.hop
    midi = curve.midi()
    rms_db = 20 * np.log10(np.maximum(curve.rms, 1e-6))
    voiced = np.isfinite(midi) & (rms_db > silence_db)
    smooth_w = max(1, int(round(smooth_time / hop)))
    smooth = _nan_moving_average(np.where(voiced, midi, np.nan), smooth_w)
    switch_frames = max(1, int(round(switch_time / hop)))
    min_frames = max(1, int(round(min_note / hop)))

    # (start, end, split_by_energy) segments in frame indices
    segments: list[tuple[int, int, bool]] = []
    for lo, hi in _voiced_runs(voiced):
        bounds = [lo]
        q = round(float(smooth[lo]))
        count = 0
        for i in range(lo, hi):
            if abs(smooth[i] - q) > switch_semitones:
                count += 1
                if count >= switch_frames:
                    start = i - count + 1
                    new_q = round(float(np.nanmedian(smooth[start:i + 1])))
                    # The smoothed curve leads the real change by up to half the
                    # smoothing window: move the boundary to where the raw pitch
                    # actually gets closer to the new note.
                    for j in range(max(bounds[-1] + 1, start - smooth_w), i + 1):
                        if np.isfinite(midi[j]) and abs(midi[j] - new_q) < abs(midi[j] - q):
                            start = j
                            break
                    if start > bounds[-1]:
                        bounds.append(start)
                    q = new_q
                    count = 0
            else:
                count = 0
        bounds.append(hi)
        for a, b in zip(bounds[:-1], bounds[1:]):
            cuts = _energy_splits(rms_db, a, b, energy_split_db, max(2, int(0.1 / hop)), min_frames)
            if onsets is not None and len(onsets):
                # Re-attacks found by an onset detector (plucks, re-sung syllables).
                for t in onsets:
                    k = int(np.searchsorted(curve.times, t))
                    if a + min_frames <= k <= b - min_frames and all(abs(k - c) >= min_frames for c in cuts):
                        cuts.append(k)
                cuts.sort()
            pts = [a, *cuts, b]
            for j, (s, e) in enumerate(zip(pts[:-1], pts[1:])):
                segments.append((s, e, j > 0))

    # Absorb too-short segments into a contiguous neighbour (glides between notes).
    merged: list[list] = []
    for s, e, esplit in segments:
        if e - s < min_frames and merged and merged[-1][1] == s:
            merged[-1][1] = e
            continue
        merged.append([s, e, esplit])
    segments = [(s, e, es) for s, e, es in merged if e - s >= min_frames]

    ref_db = np.percentile(rms_db[voiced], 95) if voiced.any() else 0.0
    notes: list[NoteEvent] = []
    for s, e, esplit in segments:
        n = e - s
        trim = int(n * 0.2)
        core = midi[s + trim:e - trim] if n - 2 * trim >= 1 else midi[s:e]
        core = core[np.isfinite(core)]
        if core.size == 0:
            continue
        est = float(np.median(core))
        pitch = int(round(est))
        if key_pitch_classes and pitch % 12 not in key_pitch_classes and abs(est - pitch) > 0.3:
            alt = pitch + (1 if est > pitch else -1)
            if alt % 12 in key_pitch_classes:
                pitch = alt
        # Trim low-energy edges: the analysis window smears the previous note's
        # release and this note's own tail across the boundaries.
        body_db = float(np.median(rms_db[s:e]))
        s_trim, e_trim = s, e
        max_lead = s + int(0.06 / hop)
        while s_trim < max_lead and e_trim - s_trim > min_frames and rms_db[s_trim] < body_db - tail_db:
            s_trim += 1
        while e_trim - s_trim > min_frames and rms_db[e_trim - 1] < body_db - tail_db:
            e_trim -= 1
        onset = float(curve.times[s_trim] - hop / 2)
        offset = float(curve.times[e_trim - 1] + hop / 2)
        seg_db = float(np.mean(rms_db[s:e]))
        velocity = int(np.clip(100 + 2.0 * (seg_db - ref_db), 20, 127))
        conf = float(np.clip(np.mean(curve.voiced_prob[s:e]), 0, 1))
        bend = _bend_curve(curve.times[s:e], midi[s:e], pitch) if with_bend else None

        prev = notes[-1] if notes else None
        if prev and prev.pitch == pitch and not esplit and onset - prev.offset <= merge_gap:
            prev.offset = offset
            if prev.bend is not None and bend:
                prev.bend.extend(bend)
            continue
        notes.append(NoteEvent(pitch, max(0.0, onset), offset, velocity, conf, bend, [source]))
    return notes


def _bend_curve(times: np.ndarray, midi: np.ndarray, pitch: int, min_step_cents: float = 4.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    last = None
    for t, m in zip(times, midi):
        if not np.isfinite(m):
            continue
        c = float(np.clip((m - pitch) * 100.0, -1200, 1200))
        if last is None or abs(c - last) >= min_step_cents:
            out.append((float(t), round(c, 1)))
            last = c
    return out
