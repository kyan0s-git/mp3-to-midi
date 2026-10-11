from metrics import note_f1

from mp3_to_midi.pitch import f0_to_notes, track
from mp3_to_midi.pitch.melody import hybrid, predominant_f0
from mp3_to_midi.synth import render
from mp3_to_midi.types import NoteEvent, Part, Transcription


def _melody():
    notes, t = [], 0.2
    for i, p in enumerate([60, 62, 64, 64, 65, 67, 69, 67, 65, 64, 62, 60]):
        d = 0.45 if i % 3 else 0.3
        notes.append(NoteEvent(p, t, t + d - 0.05, 90))
        t += d
    return notes, t


def test_sung_melody_with_vibrato():
    ref, dur = _melody()
    audio = render(Transcription([Part("Lead", "lead_vocal", ref)], duration=dur), 22050, vibrato={"Lead": 40})
    est = f0_to_notes(track(audio, "pyin", "voice"))
    assert note_f1(ref, est)["f"] == 1.0  # includes the repeated E4 (energy split)
    assert note_f1(ref, est, with_offset=True)["f"] >= 0.9
    assert all(n.bend for n in est)  # vibrato kept as pitch bend
    cents = [c for n in est for _, c in n.bend]
    assert max(abs(c) for c in cents) > 20


def test_lead_survives_harmony_voices(song):
    _, gt, stems = song
    lead = gt.part("Lead Vocal").notes
    v = stems["vocals"]
    est = f0_to_notes(hybrid(track(v, "pyin", "voice"), predominant_f0(v)))
    assert note_f1(lead, est)["f"] >= 0.9
