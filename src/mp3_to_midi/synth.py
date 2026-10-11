"""Tiny additive synthesizer.

Used to (a) build synthetic test songs with exact ground truth and (b) render
transcriptions back to audio for analysis-by-synthesis checks.
"""

from __future__ import annotations

import numpy as np

from .types import Audio, NoteEvent, Part, Transcription

# Harmonic amplitude profiles (relative to the fundamental).
TIMBRES = {
    "voice": [1.0, 0.55, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05],
    "bass": [1.0, 0.45, 0.2, 0.1, 0.05],
    "piano": [1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05],
    "pad": [1.0, 0.3, 0.12, 0.05],
    "sine": [1.0],
}

GM_DRUMS = {"kick": 36, "snare": 38, "hihat": 42, "open_hihat": 46, "crash": 49, "ride": 51}


def midi_to_hz(m: float | np.ndarray) -> float | np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(m) - 69.0) / 12.0)


def render_note(
    note: NoteEvent,
    sr: int,
    timbre: str = "voice",
    vibrato_cents: float = 0.0,
    vibrato_hz: float = 5.5,
    attack: float = 0.015,
    release: float = 0.03,
) -> tuple[int, np.ndarray]:
    """Return (start_sample, mono signal) for one note."""
    n = max(1, int(round((note.duration + release) * sr)))
    t = np.arange(n) / sr
    cents = np.zeros(n)
    if note.bend:
        bt = np.array([b[0] - note.onset for b in note.bend])
        bc = np.array([b[1] for b in note.bend])
        cents += np.interp(t, bt, bc)
    if vibrato_cents:
        ramp = np.clip(t / 0.25, 0, 1)  # vibrato fades in, like a singer
        cents += vibrato_cents * ramp * np.sin(2 * np.pi * vibrato_hz * t)
    f = midi_to_hz(note.pitch) * 2.0 ** (cents / 1200.0)
    phase = 2 * np.pi * np.cumsum(f) / sr
    y = np.zeros(n)
    nyq = sr / 2
    for h, amp in enumerate(TIMBRES[timbre], start=1):
        if (f * h).max() >= nyq:
            break
        y += amp * np.sin(h * phase)
    env = np.ones(n)
    a = min(n, int(attack * sr))
    env[:a] = np.linspace(0, 1, a, endpoint=False)
    hold = int(note.duration * sr)
    if hold < n:
        env[hold:] *= np.linspace(1, 0, n - hold)
    gain = (note.velocity / 127.0) ** 2
    return int(round(note.onset * sr)), (y * env * gain).astype(np.float32)


def render_drum(name: str, onset: float, velocity: int, sr: int, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    gain = (velocity / 127.0) ** 1.5
    if name == "kick":
        n = int(0.25 * sr)
        t = np.arange(n) / sr
        f = 50 + 100 * np.exp(-t / 0.03)
        y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.08)
    elif name == "snare":
        n = int(0.2 * sr)
        t = np.arange(n) / sr
        noise = rng.standard_normal(n) * np.exp(-t / 0.05)
        tone = 0.5 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04)
        y = 0.6 * noise + tone
    else:  # hi-hats / cymbals: high-passed noise
        n = int((0.4 if name in ("open_hihat", "crash", "ride") else 0.06) * sr)
        t = np.arange(n) / sr
        noise = rng.standard_normal(n)
        noise = np.diff(np.diff(noise, prepend=0), prepend=0)  # crude high-pass
        y = 0.25 * noise * np.exp(-t / (0.12 if n > 0.1 * sr else 0.015))
    return int(round(onset * sr)), (gain * y).astype(np.float32)


def render_part(part: Part, sr: int, n_samples: int, timbre: str | None = None, vibrato_cents: float = 0.0,
                seed: int = 0) -> np.ndarray:
    out = np.zeros(n_samples, dtype=np.float32)
    rng = np.random.default_rng(seed)
    inv_drums = {v: k for k, v in GM_DRUMS.items()}
    for note in part.notes:
        if part.is_drum:
            start, y = render_drum(inv_drums.get(note.pitch, "snare"), note.onset, note.velocity, sr, rng)
        else:
            start, y = render_note(note, sr, timbre or _default_timbre(part.role), vibrato_cents)
        end = min(n_samples, start + len(y))
        if start < n_samples:
            out[start:end] += y[: end - start]
    return out


def _default_timbre(role: str) -> str:
    return {"lead_vocal": "voice", "harmony": "voice", "backing_vocal": "voice", "bass": "bass",
            "piano": "piano"}.get(role, "pad")


def render(tr: Transcription, sr: int = 22050, pans: dict[str, float] | None = None,
           vibrato: dict[str, float] | None = None) -> Audio:
    """Render a transcription to stereo audio. `pans`: part name -> -1 (L) .. +1 (R)."""
    n = int((tr.duration + 1.0) * sr)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for i, part in enumerate(tr.parts):
        y = render_part(part, sr, n, vibrato_cents=(vibrato or {}).get(part.name, 0.0), seed=i)
        p = (pans or {}).get(part.name, 0.0)
        theta = (p + 1) * np.pi / 4  # constant-power pan
        left += np.cos(theta) * y * np.sqrt(2)
        right += np.sin(theta) * y * np.sqrt(2)
    stereo = np.stack([left, right])
    peak = np.abs(stereo).max()
    if peak > 0.99:
        stereo /= peak / 0.99
    return Audio(stereo, sr)
