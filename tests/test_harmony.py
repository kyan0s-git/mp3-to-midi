from mp3_to_midi.analysis.key import Key
from mp3_to_midi.harmony import (
    analyze_voice,
    assign_voices,
    assign_voices_to_lead,
    interval_label,
    lead_doubling_ratio,
    scale_steps,
    split_lead_doubles,
    sync_to_lead,
)
from mp3_to_midi.types import NoteEvent

C = Key.parse("C major")
LEAD = [NoteEvent(p, i * 0.5, i * 0.5 + 0.45) for i, p in enumerate([60, 62, 64, 65, 67, 65, 64, 62])]
THIRD_UP = {60: 64, 62: 65, 64: 67, 65: 69, 67: 71}


def test_scale_steps_are_diatonic():
    assert scale_steps(64, 60, C) == 2  # C->E major third = 3rd
    assert scale_steps(65, 62, C) == 2  # D->F minor third is also a 3rd
    assert scale_steps(55, 60, C) == -3  # G below C = 4th below
    assert scale_steps(72, 60, C) == 7
    assert scale_steps(63, 60, None) == 2  # chromatic fallback
    assert interval_label(2) == "3rd above"
    assert interval_label(-4) == "5th below"
    assert interval_label(-7) == "octave below"
    assert interval_label(9) == "3rd above (+1 oct)"


def test_parallel_third_and_octave_and_doubles():
    h1 = [NoteEvent(THIRD_UP[n.pitch], n.onset, n.offset) for n in LEAD]
    h2 = [NoteEvent(n.pitch - 12, n.onset, n.offset) for n in LEAD]
    dbl = [NoteEvent(n.pitch, n.onset + 0.02, n.offset) for n in LEAD[:4]]
    harmony, doubles = split_lead_doubles(h1 + h2 + dbl, LEAD)
    assert len(doubles) == 4 and len(harmony) == 16
    labels = sorted(analyze_voice(v, LEAD, C).label for v in assign_voices_to_lead(harmony, LEAD, C))
    assert labels == ["3rd above", "octave below"]
    assert 0.4 < lead_doubling_ratio(LEAD, doubles) < 0.55


def test_assignment_survives_missing_notes():
    # The upper harmony has gaps; pitch-continuity alone would hop between lines.
    h1 = [NoteEvent(THIRD_UP[n.pitch], n.onset, n.offset) for k, n in enumerate(LEAD) if k % 3]
    h2 = [NoteEvent(n.pitch - 5 if n.pitch - 5 in (55, 57, 59, 60, 62) else n.pitch - 4, n.onset, n.offset)
          for n in LEAD]
    voices = assign_voices_to_lead(h1 + h2, LEAD, C)
    assert len(voices) == 2
    assert {len(v) for v in voices} == {len(h1), len(h2)}


def test_ad_lib_and_pad_labels():
    adlib = [NoteEvent(70, 5.0 + i * 0.3, 5.25 + i * 0.3) for i in range(4)]
    assert analyze_voice(adlib, LEAD, C).label == "response / ad-lib"
    pad = [NoteEvent(55, 0.0, 2.0), NoteEvent(57, 2.0, 4.0)]
    a = analyze_voice(pad, LEAD, C)
    assert a.label.startswith("pad") and a.role == "backing_vocal"


def test_sync_to_lead_splits_merged_repeats():
    lead = [NoteEvent(64, 0.0, 0.45), NoteEvent(64, 0.5, 0.95), NoteEvent(65, 1.0, 1.45)]
    merged = [NoteEvent(67, 0.0, 0.95), NoteEvent(69, 1.0, 1.45)]
    out = sync_to_lead(merged, lead)
    assert [(n.pitch, n.onset) for n in out] == [(67, 0.0), (67, 0.5), (69, 1.0)]
    assert out[0].offset <= 0.5
    pad = [NoteEvent(60, 0.2, 1.45)]  # not aligned with lead onsets: left alone
    assert sync_to_lead(pad, lead) == pad


def test_voice_leading_assignment_orders_voices():
    notes = [NoteEvent(72, 0, 1), NoteEvent(60, 0, 1), NoteEvent(71, 1, 2), NoteEvent(62, 1, 2)]
    voices = assign_voices(notes)
    assert [[n.pitch for n in v] for v in voices] == [[72, 71], [60, 62]]
