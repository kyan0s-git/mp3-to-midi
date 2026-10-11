import json

import numpy as np
from metrics import evaluate

from mp3_to_midi.analysis import estimate_key, track_beats
from mp3_to_midi.audio import load, save
from mp3_to_midi.cli import main
from mp3_to_midi.midi_export import read_notes
from mp3_to_midi.pipeline import transcribe


def test_key_and_tempo(song):
    mix, gt, stems = song
    assert estimate_key(stems["other"]).name == "C major"
    grid = track_beats(stems["drums"])
    assert abs(grid.tempo - 100) < 3
    assert grid.beats_per_bar == 4


def test_oracle_stems_end_to_end(song, tmp_path):
    mix, gt, stems = song
    tr = transcribe(mix, tmp_path, preset="fast", stems=stems)
    scores = evaluate(gt, tr)
    assert scores["lead_vocal"]["onset"]["f"] >= 0.9
    assert scores["bass"]["onset"]["f"] >= 0.85
    assert all(v["f"] >= 0.9 for v in scores["drums"].values())
    labels = {m["label"] for m in scores["harmony"]["matched"]}
    assert "3rd above" in labels

    files = {p.name for p in (tmp_path / "parts").iterdir()}
    assert "lead_vocal.mid" in files and "drums.mid" in files and "melody.mid" in files
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["key"]["name"] == "C major"
    assert report["components"]["separation"] == "provided"
    song_tracks = read_notes(tmp_path / "song.mid")
    assert "Lead Vocal" in song_tracks and any(k.startswith("Harmony") for k in song_tracks)


def test_full_mix_from_file_via_cli(song, tmp_path, capsys):
    mix, _, _ = song
    wav = save(mix, tmp_path / "song.wav")
    assert load(wav).sr == 44100
    out = tmp_path / "out"
    assert main([str(wav), "-o", str(out), "--preset", "fast", "-q"]) == 0
    printed = capsys.readouterr().out
    assert "song.mid" in printed and "Lead Vocal" in printed
    report = json.loads((out / "report.json").read_text())
    assert report["components"]["separation"] == "dsp"
    assert any("DSP separation" in w for w in report["warnings"])
    assert (out / "song.mid").stat().st_size > 0


def test_cli_lists_components(capsys):
    assert main(["--list-components"]) == 0
    out = capsys.readouterr().out
    assert "demucs" in out and "salience" in out


def test_instrumental_track_has_no_vocal_parts(song, tmp_path):
    mix, _, stems = song
    no_vox = {k: v for k, v in stems.items() if k != "vocals"}
    tr = transcribe(mix, None, preset="fast", stems=no_vox)
    assert not tr.parts_by_role("lead_vocal")
    assert tr.part("Melody") is not None  # falls back to the instruments' top line
    assert np.isfinite(tr.tempo_bpm)
