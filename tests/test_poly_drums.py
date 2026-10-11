import numpy as np
from metrics import drum_f1, note_f1

from mp3_to_midi.synth import render
from mp3_to_midi.transcription import poly_notes, transcribe_drums
from mp3_to_midi.types import NoteEvent, Part, Transcription


def test_salience_three_voice_harmony():
    parts, t = [], 0.2
    for name, shift in (("lead", 0), ("h1", 4), ("h2", -5)):
        ns, t = [], 0.2
        for p in [60, 62, 64, 65, 67, 65, 64, 62]:
            ns.append(NoteEvent(p + shift, t, t + 0.45, 90))
            t += 0.5
        parts.append(Part(name, "harmony", ns))
    audio = render(Transcription(parts, duration=t), 22050)
    ref = [n for p in parts for n in p.notes]
    est = poly_notes(audio, "salience")
    assert note_f1(ref, est)["f"] >= 0.9


def test_drum_groove():
    beat, ns = 60 / 110, []
    for bar in range(4):
        for i in range(8):
            t = 0.3 + bar * 4 * beat + i * beat / 2
            ns.append(NoteEvent(42, t, t + 0.05, 75))
            if i in (0, 5):
                ns.append(NoteEvent(36, t, t + 0.05, 110))
            if i in (2, 6):
                ns.append(NoteEvent(38, t, t + 0.05, 100))
    audio = render(Transcription([Part("D", "drums", ns, is_drum=True)], duration=0.3 + 16 * beat), 22050)
    scores = drum_f1(ns, transcribe_drums(audio))
    assert all(v["f"] >= 0.9 for v in scores.values()), scores


def test_silence_gives_no_notes():
    from mp3_to_midi.types import Audio

    silent = Audio(np.zeros((1, 22050)), 22050)
    assert transcribe_drums(silent) == []
    assert poly_notes(silent, "salience") == []
