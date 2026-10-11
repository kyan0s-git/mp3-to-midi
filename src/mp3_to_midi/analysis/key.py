"""Global key estimation (Krumhansl-Kessler profile correlation on chroma)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..audio import resample
from ..types import Audio, NoteEvent

PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)  # natural minor


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class 0..11
    mode: str  # "major" | "minor"
    confidence: float = 0.0

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.tonic]} {self.mode}"

    @property
    def scale(self) -> tuple[int, ...]:
        steps = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return tuple((self.tonic + s) % 12 for s in steps)

    def degree(self, pitch: int) -> int | None:
        """Scale degree index 0..6 of a MIDI pitch, or None if chromatic."""
        try:
            return self.scale.index(pitch % 12)
        except ValueError:
            return None

    @classmethod
    def parse(cls, name: str) -> Key:
        tonic, mode = name.split()
        alias = {"C#": "Db", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}
        return cls(PITCH_NAMES.index(alias.get(tonic, tonic)), mode)


def _best_key(chroma: np.ndarray) -> Key:
    chroma = chroma / (chroma.sum() + 1e-9)
    scores = []
    for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for t in range(12):
            scores.append((np.corrcoef(chroma, np.roll(prof, t))[0, 1], t, mode))
    scores.sort(reverse=True)
    (s1, t, mode), (s2, *_) = scores[0], scores[1]
    return Key(t, mode, float(max(0.0, s1 - s2)))


def estimate_key(audio: Audio) -> Key:
    import librosa

    y = resample(audio, 22050).mono()
    y = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050).sum(axis=1)
    return _best_key(chroma)


def key_from_notes(notes: list[NoteEvent]) -> Key | None:
    """Key from a transcription: duration-weighted pitch-class histogram."""
    if not notes:
        return None
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.duration
    return _best_key(hist)
