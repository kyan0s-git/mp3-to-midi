import pytest

from mp3_to_midi import registry
from mp3_to_midi.fusion import consensus, fit_range, merge_duplicates, remove_cross_stem_bleed
from mp3_to_midi.types import NoteEvent, Part


def test_consensus_needs_votes_or_confidence():
    a = [NoteEvent(60, 0.00, 1.0, confidence=0.5), NoteEvent(64, 2.0, 3.0, confidence=0.5),
         NoteEvent(67, 4.0, 5.0, confidence=0.95)]
    b = [NoteEvent(60, 0.03, 0.9, confidence=0.6), NoteEvent(65, 2.0, 3.0, confidence=0.5)]
    fused = consensus([a, b], min_votes=2, keep_confidence=0.9)
    assert [(n.pitch, round(n.onset, 3)) for n in fused] == [(60, 0.015), (67, 4.0)]


def test_merge_and_range():
    merged = merge_duplicates([NoteEvent(60, 0, 1), NoteEvent(60, 0.9, 2), NoteEvent(62, 0, 1)])
    assert [(n.pitch, n.onset, n.offset) for n in merged] == [(60, 0, 2), (62, 0, 1)]
    folded = fit_range([NoteEvent(20, 0, 1), NoteEvent(90, 0, 1), NoteEvent(40, 0, 1)], 28, 67)
    assert [n.pitch for n in folded] == [32, 66, 40]


def test_bleed_removed_from_quieter_stem():
    bass = Part("Bass", "bass", [NoteEvent(40, 1.0, 2.0, velocity=110)])
    other = Part("Other", "other", [NoteEvent(40, 1.01, 2.0, velocity=40), NoteEvent(64, 1.0, 2.0, velocity=40)])
    assert remove_cross_stem_bleed([bass, other]) == 1
    assert [n.pitch for n in other.notes] == [64] and len(bass.notes) == 1


def test_registry_fallback_and_commercial_filter():
    assert registry.choose("separation", ["definitely-not-a-model", "dsp"]).name == "dsp"
    for stage in ("separation", "karaoke", "f0", "poly", "drums", "beats"):
        c = registry.choose(stage, [], commercial_safe=True)
        assert c.commercial_ok and c.available()
    with pytest.raises(RuntimeError):
        registry.choose("no-such-stage", [])
