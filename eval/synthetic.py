"""Synthetic songs with exact ground truth.

A short pop-style arrangement: lead vocal (with vibrato), a diatonic third-above
harmony and a fifth-below harmony panned left/right, bass line, drum groove and
a chord pad. Rendered with `mp3_to_midi.synth`, so every note is known.
"""

from __future__ import annotations

import numpy as np

from mp3_to_midi.analysis.key import Key
from mp3_to_midi.synth import render
from mp3_to_midi.types import Audio, NoteEvent, Part, Transcription

# I-V-vi-IV in the given key, as scale degrees of the chord roots.
PROGRESSION = [0, 4, 5, 3]


def _diatonic(key: Key, base: int, degree: int) -> int:
    """MIDI pitch of scale degree `degree` (may be negative / >6) above tonic octave `base`."""
    octave, d = divmod(degree, 7)
    return base + 12 * octave + (key.scale[d] - key.tonic) % 12


def make_song(seed: int = 0, key_name: str = "C major", bpm: float = 100.0, bars: int = 8,
              sr: int = 22050) -> tuple[Audio, Transcription, dict[str, Audio]]:
    """Return (mix, ground truth, isolated stems)."""
    rng = np.random.default_rng(seed)
    key = Key.parse(key_name)
    tonic4 = 60 + key.tonic if key.tonic < 6 else 48 + key.tonic  # tonic around C4
    beat = 60.0 / bpm
    t0 = 0.5
    lead, h_up, h_down, bass, pad, drums = [], [], [], [], [], []

    for bar in range(bars):
        bar_t = t0 + bar * 4 * beat
        root_deg = PROGRESSION[bar % 4]
        # pad: triad, whole bar
        for d in (0, 2, 4):
            pad.append(NoteEvent(_diatonic(key, tonic4 - 12, root_deg + d), bar_t, bar_t + 4 * beat - 0.05, 50))
        # bass: root on beats 1 and 3, fifth on beat 4 'and'
        b_root = _diatonic(key, tonic4 - 24, root_deg)
        for st, dur, p in ((0, 1.9, b_root), (2, 1.4, b_root), (3.5, 0.45, _diatonic(key, tonic4 - 24, root_deg + 4))):
            bass.append(NoteEvent(p, bar_t + st * beat, bar_t + (st + dur) * beat, 95))
        # drums: kick 1 & 3, snare 2 & 4, hi-hat eighths
        for i in range(8):
            t = bar_t + i * beat / 2
            drums.append(NoteEvent(42, t, t + 0.05, 75))
            if i in (0, 4):
                drums.append(NoteEvent(36, t, t + 0.05, 110))
            if i in (2, 6):
                drums.append(NoteEvent(38, t, t + 0.05, 100))
        # lead: chord tones / passing tones, quarter and half notes; harmonies follow
        pos = 0.0
        while pos < 4:
            dur = 2.0 if rng.random() < 0.25 and pos <= 2 else 1.0
            deg = root_deg + int(rng.choice([0, 2, 4, 1, 3])) - (7 if root_deg >= 4 else 0)
            on, off = bar_t + pos * beat, bar_t + (pos + dur) * beat - 0.06
            lead.append(NoteEvent(_diatonic(key, tonic4, deg), on, off, 100))
            if bar >= bars // 2:  # harmonies only in the second half ("chorus")
                h_up.append(NoteEvent(_diatonic(key, tonic4, deg + 2), on, off, 80))
                h_down.append(NoteEvent(_diatonic(key, tonic4, deg - 3), on, off, 75))
            pos += dur

    gt = Transcription(
        parts=[
            Part("Lead Vocal", "lead_vocal", lead, 53),
            Part("Harmony 3rd above", "harmony", h_up, 52),
            Part("Harmony 4th below", "harmony", h_down, 52),
            Part("Bass", "bass", bass, 33),
            Part("Drums", "drums", drums, is_drum=True),
            Part("Other", "other", pad, 0),
        ],
        duration=t0 + bars * 4 * beat,
        beats=[t0 + i * beat for i in range(bars * 4)],
        downbeats=[t0 + i * 4 * beat for i in range(bars)],
        tempo_bpm=bpm,
        key=key.name,
    )
    pans = {"Harmony 3rd above": -0.7, "Harmony 4th below": 0.7}
    vibrato = {"Lead Vocal": 35.0, "Harmony 3rd above": 25.0, "Harmony 4th below": 25.0}

    def stem(names: list[str]) -> Audio:
        sub = Transcription([p for p in gt.parts if p.name in names], duration=gt.duration)
        return render(sub, sr, pans, vibrato)

    stems = {
        "vocals": stem(["Lead Vocal", "Harmony 3rd above", "Harmony 4th below"]),
        "bass": stem(["Bass"]),
        "drums": stem(["Drums"]),
        "other": stem(["Other"]),
    }
    n = max(s.samples.shape[1] for s in stems.values())
    mix = np.zeros((2, n), dtype=np.float32)
    for s in stems.values():
        mix[:, : s.samples.shape[1]] += s.samples
    mix /= max(1.0, np.abs(mix).max() / 0.98)
    return Audio(mix, sr), gt, stems
