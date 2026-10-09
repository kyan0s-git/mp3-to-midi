# Audio → MIDI Transcription: Research Notes

Research for a program that turns an arbitrary song (MP3, WAV, FLAC, …) into an
accurate, multi-track MIDI file. It should separate and transcribe lead vocals,
backing vocals and harmonies, the melody, the bass line, drums, and whatever other
instruments are present.

*Research date: October 2026. Paper numbers are as reported by their authors unless
noted otherwise. Check licenses and current checkpoints before building on any of them.*

## Contents

| # | File | What it covers |
|---|------|----------------|
| 1 | [01-problem-overview.md](01-problem-overview.md) | Why audio→MIDI is hard, what "accurate" can realistically mean, the two broad strategies |
| 2 | [02-source-separation.md](02-source-separation.md) | Stem separation: Demucs, BS-/Mel-RoFormer, SCNet, ensembles, drum and karaoke models, prompt-based separation |
| 3 | [03-instrument-transcription.md](03-instrument-transcription.md) | Multi-instrument AMT (MT3, YourMT3+, MuScriptor, Basic Pitch) and specialists for piano, guitar, bass and drums |
| 4 | [04-vocals-melody-harmony.md](04-vocals-melody-harmony.md) | Vocal f0, note segmentation, lead/backing split, multi-singer separation, **harmony detection** |
| 5 | [05-rhythm-key-chords-quantization.md](05-rhythm-key-chords-quantization.md) | Beats, downbeats, tempo map, key, chords, rhythm quantization, MIDI/score output |
| 6 | [06-proposed-architecture.md](06-proposed-architecture.md) | **The recommended pipeline**, stage by stage, with fusion and self-correction loops |
| 7 | [07-evaluation-and-datasets.md](07-evaluation-and-datasets.md) | Metrics (mir_eval, COnPOff, SDR), datasets, how to build our own test set |
| 8 | [08-tools-licensing-commercial.md](08-tools-licensing-commercial.md) | Libraries, license matrix (much of the field is non-commercial), commercial competitors |
| 9 | [09-roadmap-and-risks.md](09-roadmap-and-risks.md) | Phased build plan, open problems, risks |
| — | [references.md](references.md) | All sources with links |

## Summary

1. **No single model does all of this well.** The best open multi-instrument
   transcribers still reach only about **0.60 note F1** on unseen, custom-composed music
   (winner of the 2025 AMT Challenge; the MT3 baseline scored about 0.39), and accuracy
   falls quickly as more instruments play at once. Running one "do-everything" model
   on a full mix will not meet the goal.

2. **Separate first, then transcribe each part with a specialist.** Source
   separation is a much more mature field than multi-instrument transcription.
   RoFormer-family models (BS-RoFormer, Mel-Band RoFormer) and SCNet produce clean
   vocal, bass and drum stems. Specialist transcribers on clean stems beat
   general-purpose transcribers on mixes. For piano, specialist onset F1 is about 98%
   on MAESTRO.

3. **Vocals and harmonies take three stages.**
   - (a) Separate all vocals from the mix.
   - (b) Split lead from backing with a *karaoke* (lead/back) model.
   - (c) Transcribe the lead as a monophonic line: f0 tracker (RMVPE, FCPE or PESTO)
     plus note segmentation (ROSVOT or SOME), with pitch bends kept.
   - (d) Transcribe the backing stem as a polyphonic vocal ensemble: multi-f0
     salience, then voice-assignment/tracking into separate harmony parts.
   - (e) Classify each harmony part against the lead (unison double, octave, third
     above, fifth below, pad, and so on).

   Multi-singer separation (MedleyVox-style) is the research frontier for splitting
   stacked harmonies further.

4. **The rhythm grid is its own subsystem.** Beat and downbeat tracking (Beat This!,
   All-In-One) produces a tempo map so the MIDI lines up with bars. A beat-aware
   transformer quantizer turns "performance MIDI" into score-like MIDI.

5. **Fuse and self-correct.** Use several models per stem where possible, take
   consensus, and constrain pitches with key and chord context. Then do
   *analysis-by-synthesis*: render the MIDI, compare it with the stem spectrogram, and
   re-examine regions where they disagree.

6. **Licensing will shape the product.** Many of the strongest checkpoints are
   CC BY-NC: MuScriptor weights, ADTOF, LarsNet, madmom models, MoisesDB, and many
   community separation weights. A commercial product will need permissively licensed
   models or its own trained weights. See file 08.

7. **Set expectations honestly.** With a good pipeline, expect very good results on
   lead vocals, bass and drums in typical pop/rock. Expect good but imperfect results
   on piano and guitar. Dense harmonies, unison doubles, distorted guitars, synth pads
   and heavy reverb will remain hard. Keeping a human-editing step (or confidence
   flags in the output) is worth planning for.

## Recommended v1 stack (open-source, research use)

| Part | Separation | Transcription |
|------|------------|---------------|
| Vocals (all) | Mel-/BS-RoFormer vocal model (or an ensemble with SCNet XL) | — |
| Lead vs. backing | Mel-RoFormer *karaoke* model on the vocal stem | Lead: RMVPE/FCPE f0 → ROSVOT/SOME notes + pitch bend |
| Backing / harmonies | (optional) multi-singer separator | Multi-f0 (vocal-ensemble CNN or Basic Pitch) → voice tracking → harmony labelling |
| Bass | 4/6-stem RoFormer or HTDemucs | Monophonic f0 + onset model; Basic Pitch as a cross-check |
| Drums | Drum stem → per-piece drum separator (DrumSep/LarsNet) | ADTOF-style ADT (or Noise-to-Notes when available) |
| Piano | 6-stem model piano stem | Transkun / hFT-Transformer |
| Guitar | 6-stem model guitar stem | Guitar-specialised AMT, or the YourMT3+/MuScriptor guitar track |
| Everything else | "other" residual | MuScriptor / YourMT3+ (instrument-aware) with Basic Pitch fallback |
| Grid | — | Beat This! / All-In-One → tempo map, time signature, sections; key and chord estimation |
