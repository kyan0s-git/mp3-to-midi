"""Backing-vocal notes -> separate harmony voices.

1. `split_lead_doubles`: backing notes at the lead's pitch while the lead sings
   them are either bleed from the lead or a unison double. Either way they are
   not a harmony line, so take them out (and keep them for doubling stats).
2. `assign_voices`: group the remaining notes into monophonic voices. Notes that
   start together form a "slice"; each slice is matched to existing voices with
   the Hungarian algorithm on pitch distance (voice leading), so parallel
   harmonies stay in their own lines and voices rarely cross.
"""

from __future__ import annotations

import numpy as np

from ..types import NoteEvent


def _overlap(a: NoteEvent, b: NoteEvent) -> float:
    return max(0.0, min(a.offset, b.offset) - max(a.onset, b.onset))


def split_lead_doubles(
    backing: list[NoteEvent], lead: list[NoteEvent], min_overlap: float = 0.5
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Return (harmony_notes, doubles). A double = same pitch as an overlapping lead note."""
    harmony, doubles = [], []
    lead_sorted = sorted(lead, key=lambda n: n.onset)
    onsets = np.array([n.onset for n in lead_sorted])
    for n in backing:
        lo = np.searchsorted(onsets, n.onset - 10.0)
        shared = sum(
            _overlap(n, l) for l in lead_sorted[lo:]
            if l.onset < n.offset and l.pitch == n.pitch
        )
        if n.duration > 0 and shared / n.duration >= min_overlap:
            doubles.append(n)
        else:
            harmony.append(n)
    return harmony, doubles


def assign_voices(
    notes: list[NoteEvent],
    max_voices: int = 4,
    slice_tolerance: float = 0.06,
    new_voice_cost: float = 6.0,
    overlap_tolerance: float = 0.03,
) -> list[list[NoteEvent]]:
    """Group notes into monophonic voices, ordered from highest to lowest."""
    from scipy.optimize import linear_sum_assignment

    voices: list[list[NoteEvent]] = []
    ordered = sorted(notes, key=lambda n: (n.onset, -n.pitch))
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j].onset - ordered[i].onset <= slice_tolerance:
            j += 1
        chunk = sorted(ordered[i:j], key=lambda n: -n.pitch)
        i = j

        BIG = 1e6
        n_new = max(0, min(len(chunk), max_voices - len(voices)))
        cost = np.full((len(chunk), len(voices) + n_new), BIG)
        for a, note in enumerate(chunk):
            for v, voice in enumerate(voices):
                last = voice[-1]
                busy = last.offset - overlap_tolerance > note.onset
                gap = max(0.0, note.onset - last.offset)
                cost[a, v] = abs(note.pitch - last.pitch) + min(gap, 4.0) * 0.5 + (50.0 if busy else 0.0)
            cost[a, len(voices):] = new_voice_cost + a * 1e-3  # keep slice order for new voices
        rows, cols = linear_sum_assignment(cost)
        n_before = len(voices)
        created: dict[int, list[NoteEvent]] = {}
        for a, c in zip(rows, cols):
            if c < n_before:
                voices[c].append(chunk[a])
            else:
                created[c] = [chunk[a]]
        for c in sorted(created):
            voices.append(created[c])
        # Notes left over (more simultaneous notes than voices): give each to the
        # nearest voice so nothing is silently dropped.
        assigned = set(rows.tolist())
        for a, note in enumerate(chunk):
            if a not in assigned and voices:
                v = int(np.argmin([abs(note.pitch - vc[-1].pitch) for vc in voices]))
                voices[v].append(note)

    voices = [sorted(v, key=lambda n: n.onset) for v in voices]
    voices.sort(key=lambda v: -np.average([n.pitch for n in v], weights=[n.duration + 1e-3 for n in v]))
    return voices


def sync_to_lead(voice: list[NoteEvent], lead: list[NoteEvent], align_tol: float = 0.07,
                 min_aligned: float = 0.5, min_piece: float = 0.1) -> list[NoteEvent]:
    """Split held notes at the lead's note onsets when the voice moves with the lead.

    Polyphonic analysis smears short gaps between repeated same-pitch notes,
    so a harmony singing "da-da-da" on one pitch comes out as one long note.
    If most of this voice's onsets coincide with lead onsets, it is singing
    the lead's rhythm (same lyrics), so re-articulate it on lead onsets.
    Sustained pads (few aligned onsets) are left alone.
    """
    if not voice or not lead:
        return voice
    lead_on = np.array(sorted(n.onset for n in lead))
    aligned = np.mean([np.min(np.abs(lead_on - n.onset)) <= align_tol for n in voice])
    if aligned < min_aligned:
        return voice
    lead_sorted = sorted(lead, key=lambda n: n.onset)
    out: list[NoteEvent] = []
    for n in voice:
        cuts = [l for l in lead_sorted if n.onset + min_piece < l.onset < n.offset - min_piece]
        start = n.onset
        for l in cuts:
            prev_off = max((p.offset for p in lead_sorted if p.offset <= l.onset), default=l.onset)
            end = prev_off if start + min_piece <= prev_off <= l.onset else l.onset
            out.append(NoteEvent(n.pitch, start, end, n.velocity, n.confidence, None, list(n.sources)))
            start = l.onset
        out.append(NoteEvent(n.pitch, start, n.offset, n.velocity, n.confidence, n.bend, list(n.sources)))
    return out


def _lead_at(note: NoteEvent, lead: list[NoteEvent]) -> NoteEvent | None:
    best, best_ov = None, 0.0
    for l in lead:
        if l.onset >= note.offset:
            break
        ov = _overlap(note, l)
        if ov > best_ov:
            best, best_ov = l, ov
    return best if best is not None and best_ov >= 0.3 * note.duration else None


def assign_voices_to_lead(
    notes: list[NoteEvent],
    lead: list[NoteEvent],
    key=None,
    max_voices: int = 3,
    min_share: float = 0.1,
) -> list[list[NoteEvent]]:
    """Group harmony notes by their interval to the lead (parallel harmony lines).

    Backing singers usually hold a steady relation to the lead ("a third
    above"), so the interval in scale steps identifies the voice far more
    reliably than pitch continuity when notes are missing. Steps are clustered
    into at most `max_voices` lines; each note joins the nearest line that is
    free at that time. Notes with no concurrent lead note fall back to
    voice-leading assignment.
    """
    from collections import Counter

    from .classify import scale_steps

    lead = sorted(lead, key=lambda n: n.onset)
    rel: list[int | None] = []
    weights: Counter = Counter()
    for n in notes:
        l = _lead_at(n, lead)
        s = None if l is None else scale_steps(n.pitch, l.pitch, key)
        rel.append(s)
        if s is not None:
            weights[s] += n.duration
    total = sum(weights.values())
    if total == 0:
        return assign_voices(notes, max_voices)

    centres: list[int] = []
    for s, w in weights.most_common():
        if len(centres) >= max_voices or w < min_share * total:
            break
        if all(abs(s - c) > 1 for c in centres):  # 3rd vs 4th above are the same line
            centres.append(s)
    centres.sort(reverse=True)
    lines: list[list[NoteEvent]] = [[] for _ in centres]
    unplaced: list[NoteEvent] = []
    for n, s in sorted(zip(notes, rel), key=lambda x: x[0].onset):
        if s is None:
            unplaced.append(n)
            continue
        order = sorted(range(len(centres)), key=lambda i: (abs(s - centres[i]), i))
        for i in order:
            if not lines[i] or lines[i][-1].offset - 0.03 <= n.onset:
                lines[i].append(n)
                break
        else:
            unplaced.append(n)
    # Notes outside the lead (responses, ad-libs) or extra simultaneous notes.
    extra = assign_voices(unplaced, max_voices=max(1, max_voices - len(centres))) if unplaced else []
    voices = [v for v in lines if v] + extra
    return [sorted(v, key=lambda n: n.onset) for v in voices]
