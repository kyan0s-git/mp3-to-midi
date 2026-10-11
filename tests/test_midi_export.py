import mido
import numpy as np

from mp3_to_midi.midi_export import DRUM_CHANNEL, TempoMap, read_notes, write_midi
from mp3_to_midi.types import NoteEvent, Part, Transcription


def _tr():
    # Slightly accelerating beats, first downbeat on the 2nd beat (one-beat pickup).
    beats = list(0.7 + np.cumsum(np.r_[0, np.full(15, 0.5) + np.linspace(0, 0.05, 15)]))
    lead = Part("Lead Vocal", "lead_vocal", [
        NoteEvent(60, 0.71, 1.2, 90, bend=[(0.8, -30.0), (1.0, 20.0)]),
        NoteEvent(64, 1.25, 2.0, 80),
    ], program=53)
    drums = Part("Drums", "drums", [NoteEvent(36, 0.7, 0.75, 100)], is_drum=True)
    return Transcription([lead, drums], beats=beats, downbeats=beats[1::4], key="A minor")


def test_roundtrip_timing(tmp_path):
    tr = _tr()
    out = read_notes(write_midi(tr, tmp_path / "a.mid"))
    for part in tr.parts:
        got = out[part.name]
        assert len(got) == len(part.notes)
        for g, n in zip(got, part.notes):
            assert g.pitch == n.pitch
            assert abs(g.onset - n.onset) < 0.005
            assert abs(g.offset - n.offset) < 0.005


def test_first_downbeat_on_bar_line():
    tr = _tr()
    tm = TempoMap(tr.beats, tr.downbeats, 4)
    assert tm.to_ticks(tr.downbeats[0]) % (4 * 480) == 0
    assert tm.to_ticks(tr.beats[0]) > 0  # pickup beat sits before the bar line, not at negative time


def test_tracks_channels_and_meta(tmp_path):
    mid = mido.MidiFile(str(write_midi(_tr(), tmp_path / "a.mid")))
    assert mid.type == 1 and len(mid.tracks) == 3
    meta = {m.type: m for m in mid.tracks[0]}
    assert meta["key_signature"].key == "Am"
    assert meta["time_signature"].numerator == 4
    lead_msgs = list(mid.tracks[1])
    assert any(m.type == "program_change" and m.program == 53 for m in lead_msgs)
    # RPN 0 bend range set before any bend
    ccs = [(m.control, m.value) for m in lead_msgs if m.type == "control_change"]
    assert ccs[:4] == [(101, 0), (100, 0), (6, 2), (38, 0)]
    bends = [m.pitch for m in lead_msgs if m.type == "pitchwheel"]
    assert min(bends) < 0 < max(bends)
    drum_notes = [m for m in mid.tracks[2] if m.type == "note_on"]
    assert drum_notes and all(m.channel == DRUM_CHANNEL for m in drum_notes)


def test_overlapping_same_pitch_is_truncated(tmp_path):
    p = Part("P", "other", [NoteEvent(60, 0.0, 1.0), NoteEvent(60, 0.5, 1.5)])
    tr = Transcription([p], beats=[0.0, 0.5, 1.0, 1.5, 2.0])
    got = read_notes(write_midi(tr, tmp_path / "a.mid"))["P"]
    assert len(got) == 2
    assert got[0].offset <= got[1].onset + 1e-6
    assert p.notes[0].offset == 1.0  # the transcription itself is not mutated


def test_no_beats_falls_back_to_constant_tempo(tmp_path):
    tr = Transcription([Part("P", "other", [NoteEvent(62, 1.0, 2.0)])])
    got = read_notes(write_midi(tr, tmp_path / "a.mid"))["P"][0]
    assert abs(got.onset - 1.0) < 0.005 and abs(got.offset - 2.0) < 0.005
