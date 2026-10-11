"""Combining note lists: consensus between models, de-duplication, bleed removal."""

from __future__ import annotations

import numpy as np

from .types import NoteEvent, Part


def match_notes(a: list[NoteEvent], b: list[NoteEvent], onset_tol: float = 0.05,
                pitch_tol: int = 0) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by pitch and onset proximity. Returns index pairs."""
    pairs = []
    used = set()
    b_by_pitch: dict[int, list[int]] = {}
    for j, n in enumerate(b):
        b_by_pitch.setdefault(n.pitch, []).append(j)
    for i, n in sorted(enumerate(a), key=lambda x: x[1].onset):
        best, best_d = None, onset_tol
        for p in range(n.pitch - pitch_tol, n.pitch + pitch_tol + 1):
            for j in b_by_pitch.get(p, []):
                d = abs(b[j].onset - n.onset)
                if j not in used and d <= best_d:
                    best, best_d = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def merge_duplicates(notes: list[NoteEvent], gap: float = 0.0) -> list[NoteEvent]:
    """Merge same-pitch notes that overlap (or are within `gap` seconds)."""
    out: list[NoteEvent] = []
    last_by_pitch: dict[int, NoteEvent] = {}
    for n in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        prev = last_by_pitch.get(n.pitch)
        if prev is not None and n.onset <= prev.offset + gap:
            prev.offset = max(prev.offset, n.offset)
            prev.velocity = max(prev.velocity, n.velocity)
            prev.confidence = max(prev.confidence, n.confidence)
            prev.sources = sorted(set(prev.sources) | set(n.sources))
            continue
        m = NoteEvent(n.pitch, n.onset, n.offset, n.velocity, n.confidence,
                      list(n.bend) if n.bend else None, list(n.sources))
        out.append(m)
        last_by_pitch[n.pitch] = m
    return out


def consensus(lists: list[list[NoteEvent]], min_votes: int = 2, keep_confidence: float = 0.85,
              onset_tol: float = 0.05) -> list[NoteEvent]:
    """Fuse several transcriptions of the same stem.

    A note survives when at least `min_votes` lists contain it (same pitch,
    onset within `onset_tol`), or when one list reports it with confidence >=
    `keep_confidence`. Matched notes take the median onset/offset and the
    first available pitch-bend curve.
    """
    lists = [l for l in lists if l]
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    pool = [(k, n) for k, l in enumerate(lists) for n in l]
    pool.sort(key=lambda x: x[1].onset)
    taken = [False] * len(pool)
    fused: list[NoteEvent] = []
    for i, (k, n) in enumerate(pool):
        if taken[i]:
            continue
        group = [(k, n)]
        taken[i] = True
        seen = {k}
        for j in range(i + 1, len(pool)):
            k2, m = pool[j]
            if m.onset - n.onset > onset_tol:
                break
            if not taken[j] and k2 not in seen and m.pitch == n.pitch:
                group.append((k2, m))
                seen.add(k2)
                taken[j] = True
        notes = [g[1] for g in group]
        votes = len(group)
        if votes < min_votes and max(x.confidence for x in notes) < keep_confidence:
            continue
        fused.append(NoteEvent(
            n.pitch,
            float(np.median([x.onset for x in notes])),
            float(np.median([x.offset for x in notes])),
            int(np.mean([x.velocity for x in notes])),
            float(min(1.0, np.mean([x.confidence for x in notes]) * (0.7 + 0.3 * votes / len(lists)))),
            next((x.bend for x in notes if x.bend), None),
            sorted({s for x in notes for s in x.sources}),
        ))
    return fused


def fit_range(notes: list[NoteEvent], lo: int, hi: int, fold_octaves: bool = True) -> list[NoteEvent]:
    """Keep notes in [lo, hi]; optionally octave-fold near misses (octave errors)."""
    out = []
    for n in notes:
        p = n.pitch
        if fold_octaves:
            while p < lo and p + 12 <= hi:
                p += 12
            while p > hi and p - 12 >= lo:
                p -= 12
        if lo <= p <= hi:
            n.pitch = p
            out.append(n)
    return out


def remove_cross_stem_bleed(parts: list[Part], onset_tol: float = 0.04, ratio: float = 1.5) -> int:
    """Drop a note from a part when another part has the same note clearly louder.

    Separation bleed shows up as a quieter copy of a note in a neighbouring stem.
    Returns the number of notes removed. Drum parts are left alone.
    """
    removed = 0
    pitched = [p for p in parts if not p.is_drum]
    for a in pitched:
        for b in pitched:
            if a is b or not a.notes or not b.notes:
                continue
            drop = set()
            for i, j in match_notes(a.notes, b.notes, onset_tol):
                if b.notes[j].velocity >= ratio * a.notes[i].velocity:
                    drop.add(i)
            if drop:
                a.notes = [n for k, n in enumerate(a.notes) if k not in drop]
                removed += len(drop)
    return removed
