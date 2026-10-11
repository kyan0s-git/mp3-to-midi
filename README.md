# mp3-to-midi

This project turns a song into a multi-track MIDI file. The output has separate parts for:

- the lead vocal (with pitch bends)
- each harmony voice, labelled by its interval to the lead ("3rd above", "4th below", "octave below", …)
- a melody track
- bass
- drums
- the remaining instruments

The design follows the research in [`research/`](research/README.md):
1. Separate the song into stems.
2. Transcribe each stem with a specialist.
3. Fuse the results.
4. Export with a tempo map so the notes line up with bars.

> **Status: v0.1, first working pipeline.** Every stage runs end to end and is
> covered by tests. Each stage has a built-in baseline that needs only
> numpy/scipy/librosa, plus adapters for the neural models from the research.
> The adapters activate automatically when those models are installed. Results
> on real recordings depend heavily on having a neural separator installed (see
> [Results](#results-so-far)).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # core + tests (numpy, scipy, librosa, soundfile, mido, mir_eval)
```

Optional components. Each stage uses the best one that is installed:

| Extra | What it adds | License |
|-------|--------------|---------|
| `pip install demucs torch` | HTDemucs 4/6-stem separation (the most important upgrade) | MIT |
| `pip install audio-separator` | BS/Mel-RoFormer vocals and a **karaoke lead/backing split** | varies per model |
| `pip install basic-pitch` | Spotify Basic Pitch for instruments (piano, guitar, other) | Apache-2.0 |
| `pip install torchcrepe torch` | CREPE f0 for the lead vocal | MIT |
| `pip install beat-this torch` | Beat This! beat and downbeat tracker | MIT |

On Python 3.13, Basic Pitch's TensorFlow dependency has no wheel. Use its bundled ONNX model instead:
`pip install --no-deps basic-pitch && pip install onnxruntime pretty_midi resampy`.

`ffmpeg` on your PATH is recommended for decoding MP3/M4A. Without it, `soundfile` is used.

## Usage

```bash
mp3-to-midi song.mp3                       # -> song_midi/
mp3-to-midi song.mp3 -o out/ --preset max --stems
mp3-to-midi --list-components              # what is installed, and each license
mp3-to-midi song.mp3 --commercial-safe     # refuse non-commercial weights
```

Output folder:

```
song.mid            all parts in one Type-1 MIDI file (tempo map, time/key signature)
parts/*.mid         one file per part (lead_vocal.mid, harmony_1_3rd_above.mid, bass.mid, ...)
report.json         components used + fallbacks, key, tempo, harmony analysis, per-part confidence
stems/*.wav         separated stems (with --stems)
```

From Python:

```python
from mp3_to_midi import transcribe
tr = transcribe("song.mp3", "out/", preset="balanced")
print(tr.key, tr.tempo_bpm, [p.name for p in tr.parts])
print(tr.report["harmony"])   # e.g. {"lead_doubled_ratio": 0.4, "voices": [{"label": "3rd above", ...}]}
```

### Presets

Each stage falls back to the next installed option, and `report.json` records what was used.

| Stage | `fast` | `balanced` (default) | `max` |
|-------|--------|----------------------|-------|
| separation | DSP | Demucs → DSP | RoFormer → Demucs-6 → Demucs → DSP |
| lead/backing | mid/side | mid/side | RoFormer karaoke → mid/side |
| lead f0 | pYIN | pYIN | CREPE → pYIN |
| instruments | salience | Basic Pitch → salience | Basic Pitch → salience |
| beats | librosa | Beat This! → librosa | Beat This! → librosa |

The lead f0 is always fused with a predominant-melody tracker. Backing vocals always use the salience
multi-pitch tracker, which beat Basic Pitch at separating harmony lines on our benchmark.

## How it works

| Module | Role |
|--------|------|
| `audio.py` | ffmpeg/soundfile decoding, resampling |
| `separation/` | DSP fallback (HPSS + stereo panning mask), Demucs and python-audio-separator adapters, lead/backing split |
| `pitch/f0.py`, `pitch/melody.py` | pYIN/CREPE f0. Predominant-melody tracker (harmonic-sum salience + Viterbi), fused with pYIN so harmonies can't steal the lead |
| `pitch/segment.py` | f0 curve → notes: smoothed hysteresis, raw-pitch boundary refinement, energy/onset splits, tail trimming, **pitch-bend curves** |
| `transcription/poly.py` | multi-pitch: CQT harmonic sum with iterative cancellation (no model), or Basic Pitch |
| `transcription/drums.py` | kick/snare/hi-hat: band-normalised spectral flux + NNLS templates (handles simultaneous hits) |
| `harmony/` | remove lead doubles/bleed → group notes into voices **by their scale-step interval to the lead** → re-articulate on lead syllables → label each voice, doubling ratio, per-section summary |
| `analysis/` | beats/downbeats/meter, onset detection, key (Krumhansl–Kessler) |
| `fusion.py` | multi-model consensus, duplicate merging, octave folding, cross-stem bleed removal |
| `midi_export.py` | Type-1 SMF: per-beat tempo map with the first downbeat on a bar line, GM programs, drums on channel 10, RPN bend range, markers |
| `pipeline.py` | orchestration, presets, fallbacks, report |
| `synth.py` | small additive synth, used to build test songs with exact ground truth |

## Results so far

[`eval/`](eval) builds synthetic pop arrangements with exact ground truth:
- lead with vibrato
- a diatonic third-above and a fourth-below harmony, panned left and right
- bass
- a drum groove
- a chord pad

It scores the output with `mir_eval`. "Oracle" feeds the true stems, which measures transcription alone.
"Mix" runs the whole system, including separation.

Mean over 3 songs, note onset F1 (onset+offset F1 in brackets):

| Part | Oracle stems, `fast` | Oracle stems, `balanced` | Full mix, DSP separation |
|------|----------------------|--------------------------|--------------------------|
| Lead vocal | **0.99** (0.96) | **0.99** (0.96) | 0.57 (0.52) |
| Harmony voices (matched) | 0.74 — labels correct (3rd above / 4th below) | same | ≈0 |
| Bass | **0.96** (0.89) | **0.96** (0.89) | 0.34 (0.06) |
| Drums (kick/snare/hi-hat) | **1.00 / 1.00 / 1.00** | same | 0.65 / 0.71 / 0.98 |
| Pad / other | 0.47 (0.37) | 0.62 (0.42) | ≈0 |
| Key / tempo | 3/3 correct | 3/3 correct | 3/3 correct |

What these numbers say:
- **Transcription works well when stems are clean.** The full-mix column is the model-free DSP fallback, which
  leaks every part into every other. That is exactly why the research recommends neural separation.
- **No neural model has been run yet.** The sandbox this was built in could not download weights from Hugging
  Face, GitHub releases or Meta. The Demucs and RoFormer adapters follow the libraries' documented APIs but
  are **untested**. Running `python eval/run_eval.py --preset balanced` with `demucs` installed is the first
  thing to do on a normal machine.
- **These are synthetic songs.** Real recordings bring reverb, distortion, timbre variety and production
  effects. Building a real-recording test set is next on the roadmap.

Reproduce with:
```bash
python eval/run_eval.py --songs 3 --preset fast
pytest -q
```

## Next steps

These come from [`research/09-roadmap-and-risks.md`](research/09-roadmap-and-risks.md):
1. Verify the Demucs / audio-separator adapters on a machine with model downloads, and re-run the mix benchmark.
2. Assemble a real-recording evaluation set (multitrack sources plus a hand-annotated pop-harmony set).
3. Add a neural drum transcriber (ADTOF / Noise-to-Notes) and MuScriptor/YourMT3+ for "other", behind license flags.
4. Add analysis-by-synthesis verification: render each part, compare it with its stem, and re-decode mismatches.
5. Add a rhythm quantization export (score-like MIDI) and MusicXML.
