# 1. Problem overview

## 1.1 What we are actually asking for

"Transcribe a song to MIDI" is several separate MIR (music information retrieval) tasks
stacked together:

| Sub-task | Output | Field name |
|----------|--------|------------|
| Which instruments/voices are present, and when | instrument activity | instrument recognition / tagging |
| Isolate each part's audio | stems | music source separation (MSS) |
| Which pitches sound, when they start and stop, and how loud | note events (pitch, onset, offset, velocity) | automatic music transcription (AMT) |
| Continuous pitch for voice, bends, slides and vibrato | f0 curve → MIDI pitch-bend | f0 / pitch estimation |
| Which notes belong to which voice or line | voice/part assignment | voice separation / streaming |
| Where the beats and bars are | tempo map, time signature | beat/downbeat tracking |
| Key, chords | key signature, chord labels | key / chord recognition |
| Clean, readable rhythm | quantized note values | rhythm quantization |

Each of these is an active research area. A product that "transcribes any song" has to
chain them together and handle their combined error.

## 1.2 Why it is hard

1. **Overlapping harmonics.** Notes in consonant intervals (octaves, fifths, thirds),
   which are the backbone of harmony, share partials. Two singers a third apart look
   like one complex sound in the spectrum. Octave errors are the classic failure.
2. **Polyphony degrades accuracy steeply.** The 2025 AMT Challenge winner went from
   about 0.72 F1 on single-instrument excerpts to about 0.44–0.46 on three-instrument
   excerpts.
3. **Timbre overlap and "instrument leakage."** Multi-instrument models often give
   the right pitch to the wrong instrument, or invent instruments that aren't there.
   MR-MT3 was written specifically to reduce this.
4. **Production effects.** Reverb smears offsets. Distortion adds inharmonic energy.
   Heavy compression and sidechain pumping confuse velocity. Auto-tune produces
   unnatural pitch steps. Stereo widening and chorus create pseudo-unison.
5. **Vocal specifics.** Singing has continuous pitch (scoops, vibrato, portamento), so
   "which MIDI note is this?" is a *segmentation and quantization* decision, not just
   a measurement. Lyrics with consonants break pitch continuity. Doubled lead vocals
   and stacked harmonies are often mixed to blend on purpose.
6. **Data scarcity.** Aligned audio+MIDI for real multi-instrument pop songs barely
   exists. Most training data is synthesized (Slakh), piano-only (MAESTRO), or small.
   The 2025 AMT Challenge organizers name data scarcity as the main bottleneck.
7. **No single ground truth.** Cymbal hits, ghost notes and backing "oohs" are things
   human annotators also disagree on. The ADTOF error analysis found cymbals are the
   main source of annotator disagreement in drum transcription.

## 1.3 Two broad strategies

### A. End-to-end multi-instrument transcription on the mix
One model (MT3, YourMT3+, MuScriptor) takes the mixture and emits MIDI tokens for every
instrument.

- **Pros:** one model, no separation artifacts, learns instrument interactions.
- **Cons:** accuracy collapses with polyphony. Vocals are weak or missing in many of
  these models (MT3 trained largely without vocals; MuScriptor's README doesn't list
  vocals; YourMT3+ added singing). Instrument leakage. Weak at expressive pitch
  (bends, vibrato).

### B. Separate-then-transcribe ("cascade")
Split into stems with a strong separator, then run a specialist on each stem.

- **Pros:** uses the most mature components (separation SDR on vocals/bass/drums is
  high). Each specialist solves an easier, often monophonic problem. Modular, so any
  component can be upgraded.
- **Cons:** separation artifacts (bleed, "watery" residue) can create phantom notes.
  Errors cascade. Compute cost is higher.

### C. Hybrid (recommended)
Use **B as the backbone** and **A as a cross-check and a catch-all**:
- Cascade for vocals, bass, drums, piano and guitar, where stems and specialists are strong.
- Multi-instrument model on the "other" residual, and as a second opinion on every stem.
- Key, chord and beat context plus analysis-by-synthesis to settle disagreements.

[06-proposed-architecture.md](06-proposed-architecture.md) gives the full design.

## 1.4 What "accurate" can mean (target definitions)

These targets should be set explicitly, because "accurate MIDI" means different things:

| Level | Meaning | Metric (see file 07) |
|-------|---------|-----------------------|
| L1 – pitch/onset | right notes at the right times | note F1, onset only (50 ms, ±50 cents) |
| L2 – + durations | right note lengths | note F1 with offsets |
| L3 – + dynamics | sensible velocities | note F1 with offset + velocity |
| L4 – + expression | pitch bends, vibrato, slides for voice/guitar/bass | f0 RPA on bends, COnPOff for singing |
| L5 – + structure | correct tempo map, bars, key, part assignment | beat F1, MV2H, voice-assignment accuracy |
| L6 – score-ready | quantized, readable, notatable | MUSTER / MV2H, human edit distance |

A realistic v1 should aim for strong L1–L3 on vocals, bass and drums, L4 on lead vocals,
and L5 for the grid. L6 is a separate, later product.
