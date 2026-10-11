"""Label harmony voices relative to the lead ("3rd above", "octave double", ...).

Intervals are measured in *scale steps* of the detected key, so a diatonic
third above counts as one label whether it is 3 or 4 semitones on a given
degree. Without a key, semitones are mapped to the nearest generic interval.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..analysis.key import Key
from ..types import NoteEvent

ORDINALS = {1: "2nd", 2: "3rd", 3: "4th", 4: "5th", 5: "6th", 6: "7th"}
# semitones (mod 12) -> generic steps, used for chromatic notes / no key
SEMITONE_TO_STEPS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}


def scale_steps(upper: int, lower: int, key: Key | None) -> int:
    """Signed generic interval (in scale steps) from `lower` to `upper`."""
    if key is not None:
        du, dl = key.degree(upper), key.degree(lower)
        if du is not None and dl is not None:
            iu = du + 7 * ((upper - key.tonic) // 12)
            il = dl + 7 * ((lower - key.tonic) // 12)
            return iu - il
    semis = upper - lower
    sign = 1 if semis >= 0 else -1
    octaves, rem = divmod(abs(semis), 12)
    return sign * (7 * octaves + SEMITONE_TO_STEPS[rem])


def interval_label(steps: int) -> str:
    if steps == 0:
        return "unison"
    direction = "above" if steps > 0 else "below"
    octaves, rem = divmod(abs(steps), 7)
    if rem == 0:
        return f"{'octave' if octaves == 1 else f'{octaves} octaves'} {direction}"
    base = ORDINALS[rem]
    return f"{base} {direction}" if octaves == 0 else f"{base} {direction} (+{octaves} oct)"


@dataclass
class VoiceAnalysis:
    label: str  # e.g. "3rd above", "octave below", "pad", "response / ad-lib"
    role: str  # "harmony" | "backing_vocal"
    steps: int | None  # dominant signed scale-step interval vs lead
    consistency: float  # share of overlapped time at the dominant interval
    lead_overlap: float  # share of this voice's duration that overlaps the lead
    histogram: dict[str, float] = field(default_factory=dict)


def analyze_voice(voice: list[NoteEvent], lead: list[NoteEvent], key: Key | None) -> VoiceAnalysis:
    total = sum(n.duration for n in voice) or 1e-9
    weights: Counter = Counter()
    for n in voice:
        for l in lead:
            ov = min(n.offset, l.offset) - max(n.onset, l.onset)
            if ov > 0:
                weights[scale_steps(n.pitch, l.pitch, key)] += ov
    overlapped = sum(weights.values())
    lead_overlap = overlapped / total
    hist = {interval_label(s): round(w / overlapped, 3) for s, w in weights.most_common()} if overlapped else {}

    if lead_overlap < 0.3:
        return VoiceAnalysis("response / ad-lib", "backing_vocal", None, 0.0, lead_overlap, hist)
    steps, w = weights.most_common(1)[0]
    consistency = w / overlapped
    label = "unison double" if steps == 0 else interval_label(steps)
    if consistency < 0.5:
        label = f"free harmony (mostly {label})"

    mean_voice = np.mean([n.duration for n in voice])
    mean_lead = np.mean([n.duration for n in lead]) if lead else mean_voice
    role = "harmony"
    if mean_voice > 1.8 * mean_lead and len(voice) < len(lead):
        label = f"pad ({label})"
        role = "backing_vocal"
    return VoiceAnalysis(label, role, steps, consistency, lead_overlap, hist)


def lead_doubling_ratio(lead: list[NoteEvent], doubles: list[NoteEvent]) -> float:
    """Share of the lead's sung time that is doubled at unison."""
    total = sum(n.duration for n in lead)
    if total <= 0:
        return 0.0
    covered = 0.0
    for l in lead:
        covered += min(l.duration, sum(max(0.0, min(l.offset, d.offset) - max(l.onset, d.onset))
                                       for d in doubles if d.pitch == l.pitch))
    return covered / total


def section_summary(lead: list[NoteEvent], voices: list[list[NoteEvent]], key: Key | None,
                    sections: list[tuple[float, float, str]]) -> list[dict]:
    """Per-section harmony description, e.g. chorus: ['3rd above', '5th below']."""
    def clip(ns: list[NoteEvent], start: float, end: float) -> list[NoteEvent]:
        return [n for n in ns if n.onset < end and n.offset > start]

    out = []
    for start, end, name in sections:
        lead_s = clip(lead, start, end)
        labels = []
        for v in voices:
            vs = clip(v, start, end)
            if len(vs) >= 2 and lead_s:
                labels.append(analyze_voice(vs, lead_s, key).label)
        out.append({"start": round(start, 3), "end": round(end, 3), "section": name, "harmonies": labels})
    return out
