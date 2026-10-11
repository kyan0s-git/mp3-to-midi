# 9. Roadmap, open problems, and risks

> **Build status (v0.1):** the Phase 0 evaluation harness (synthetic songs + mir_eval) is in place.
> The Phase 1 baseline cascade is done, with model-free fallbacks for every stage plus
> Demucs/RoFormer/Basic Pitch/CREPE/Beat This! adapters. The core of Phase 3 is done: harmony voices
> grouped by interval to the lead, harmony labels, doubling detection, and a melody track. The
> Demucs/RoFormer adapters still need to be run against real downloaded weights. See the top-level
> README for current numbers.

## 9.1 Phased build plan

### Phase 0: Evaluation first (1–2 weeks)
- Build the evaluation harness (file 07): mir_eval wrappers, dataset loaders, report generator.
- Assemble a **dev set**: MIR-ST500 subset, Slakh subset, MUSDB18-HQ (separation), a few
  MedleyDB multitracks, plus 10 hand-picked real songs.
- Start the hand-annotated **pop-harmony set**. It takes the longest to build.

### Phase 1: Baseline cascade (2–3 weeks)
- ffmpeg decode → HTDemucs 6-stem → Basic Pitch per stem → PESTO/RMVPE lead vocal →
  Beat This! tempo map → Type-1 MIDI.
- Uses only permissive licenses. This becomes the **commercial-safe baseline** to beat.
- Measure per-stem metrics and runtime.

### Phase 2: Strong separation + specialists (3–4 weeks)
- RoFormer/SCNet ensembles through MSST. Mel-RoFormer karaoke lead/back split.
- Lead SVT: RMVPE → ROSVOT / SOME. Choose by COnPOff on the dev set.
- Bass: low-f0 tracker + onset model + octave correction.
- Drums: drum-piece separation + ADT + stem-based velocity.
- Piano: Transkun. Guitar: specialist or MuScriptor(guitar).
- "Other": MuScriptor / YourMT3+ with tagger hints.

### Phase 3: Harmony and melody (3–5 weeks), the differentiator
- Vocal-ensemble multi-f0 + Basic Pitch salience fusion on the backing stem.
- Voice tracking (Viterbi / min-cost flow), lead-leak removal, doubling detection.
- Stereo L/R/M/S analysis.
- Harmony classification in scale-degree space. Section summaries.
- Melody track assembly from the predominant-melody contour.
- Synthetic harmony dataset generator → fine-tune the multi-f0 model.

### Phase 4: Fusion and self-correction (2–3 weeks)
- Multi-model note consensus with per-instrument learned weights.
- Cross-stem de-duplication, key/chord priors.
- Analysis-by-synthesis loop (render → residual → re-decode).
- Confidence calibration (reliability diagrams on the dev set).

### Phase 5: Output quality and UX (2–3 weeks)
- Quantized export (grid snapping → learned quantizer later), MusicXML.
- Report JSON + simple web/desktop viewer: piano roll per part with confidence shading,
  solo/mute stems against the MIDI render for A/B checks.
- Quality presets (fast CPU / balanced / max GPU).

### Phase 6: Own models (ongoing)
- Fine-tune transcribers on **separated stems with realistic bleed** (closes the
  clean→real gap).
- Train commercial-safe replacements for NC checkpoints (drums, multi-instrument).
- Consider a **foundation-model encoder** (MusicFM / MERT-style) shared across
  transcription heads. This is what the 2025 AMT Challenge winner did.

## 9.2 Open research problems (where accuracy will be lost)

| Problem | Why hard | Mitigation |
|---------|----------|------------|
| Stacked/unison backing vocals | spectra overlap, deliberate blending | stereo analysis, doubling detection, report "doubled" instead of guessing |
| Dense distorted guitars | intermodulation, wide-band noise | guitar-specific models, chord priors, voicing constraints |
| Synth pads / heavy FX | no stable timbre, long reverbs | Basic Pitch + offset capping, generic labelling |
| Piano/keys in pop mixes | weak separation, processed sounds | fine-tune on separated stems, cross-check with chord model |
| Cymbals in ADT | annotator disagreement, washy spectra | accept lower precision, velocity from stem energy |
| Instrument leakage in multi-instrument AMT | timbre confusion | tagger-gated instrument lists, consensus |
| Rubato / live tempo | grid estimation errors | tempo map per beat, performance-timing export |
| Non-Western tuning / microtonality | 12-TET MIDI assumption | keep pitch-bend curves, optional MPE / MIDI 2.0 |
| Long-tail instruments | little training data | SAM Audio prompt separation + generic AMT |
| Evaluation of harmonies | no standard dataset or metric | build our own (file 07) |

## 9.3 Risks

- **Licensing:** the best checkpoints are non-commercial. *Mitigation:* model registry with
  license flags, a commercial-safe recipe, and a budget for training our own models.
- **Compute cost:** max-quality ensembles take minutes of GPU time per song.
  *Mitigation:* presets, caching stems, ONNX/fp16, running only the stems that the tagger
  says are present.
- **Fast-moving field:** new SOTA every few months (e.g. MuScriptor in July 2026, VocalParse
  in May 2026). *Mitigation:* pluggable interfaces plus the evaluation harness, so swapping
  in a new model is a config change followed by an eval run.
- **Expectation gap:** "any song, no matter how complex" will not be perfect.
  *Mitigation:* confidence-annotated output, an editing UI, and honest per-part quality
  indicators.
- **Copyright:** see file 08 §8.4.

## 9.4 Next concrete steps

1. Create a repo skeleton: `mp3_to_midi/{io,separation,transcription,f0,harmony,grid,fusion,export}/`,
   `eval/`, `recipes/*.yaml`.
2. Implement the Phase 1 baseline end to end and produce a MIDI for 5 test songs.
3. Stand up the eval harness and record baseline numbers.
4. Start the pop-harmony annotation set in parallel.
