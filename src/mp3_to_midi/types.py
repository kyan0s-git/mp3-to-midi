"""Core data model shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Audio:
    """Multichannel float32 audio, shaped (channels, samples)."""

    samples: np.ndarray
    sr: int

    def __post_init__(self) -> None:
        s = np.asarray(self.samples, dtype=np.float32)
        if s.ndim == 1:
            s = s[None, :]
        self.samples = s

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sr

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def mid(self) -> np.ndarray:
        """Center-panned content (L+R)/2. Equals mono for mono input."""
        return self.mono()

    def side(self) -> np.ndarray:
        """Content that differs between channels (L-R)/2. Zero for mono input."""
        if self.channels < 2:
            return np.zeros(self.samples.shape[1], dtype=np.float32)
        return 0.5 * (self.samples[0] - self.samples[1])

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.samples**2) + 1e-12))


@dataclass
class NoteEvent:
    pitch: int  # MIDI note number
    onset: float  # seconds
    offset: float  # seconds
    velocity: int = 80
    confidence: float = 1.0
    # Optional expressive pitch curve: (absolute time in seconds, deviation in cents).
    bend: list[tuple[float, float]] | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.offset - self.onset


# Roles a part can play. Used to choose GM programs, colors in reports, etc.
ROLES = (
    "lead_vocal",
    "harmony",
    "backing_vocal",
    "melody",
    "bass",
    "drums",
    "piano",
    "guitar",
    "other",
)


@dataclass
class Part:
    name: str
    role: str
    notes: list[NoteEvent] = field(default_factory=list)
    program: int = 0  # General MIDI program (0-127), ignored for drums
    is_drum: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Transcription:
    parts: list[Part] = field(default_factory=list)
    duration: float = 0.0
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    beats_per_bar: int = 4
    tempo_bpm: float = 120.0
    key: str | None = None  # e.g. "C major", "A minor"
    sections: list[tuple[float, float, str]] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    def part(self, name: str) -> Part | None:
        return next((p for p in self.parts if p.name == name), None)

    def parts_by_role(self, role: str) -> list[Part]:
        return [p for p in self.parts if p.role == role]
