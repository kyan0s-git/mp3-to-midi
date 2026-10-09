# 5. Rhythm grid, key, chords, and quantization

Without a correct tempo map, MIDI that sounds right *looks* wrong: notes don't line
up with bars in a DAW and notation is unreadable. The grid deserves its own subsystem.

## 5.1 Beat and downbeat tracking

| System | Notes |
|--------|-------|
| madmom (RNN + DBN) | long-time standard. **Code BSD, but pretrained models are CC BY-NC-SA.** DBN post-processing |
| **Beat This!** (Foscarin, Schlüter, Widmer, ISMIR 2024) | conv + transformer, shift-tolerant loss, **no DBN needed**. Beats prior SOTA in F1. Weaker on continuity metrics and on rare genres. Reported MIT-licensed (verify) |
| All-In-One (mir-aidj) | joint beats, downbeats, **section boundaries and labels** (intro/verse/chorus), using demixed inputs. Good for section-level harmony summaries |
| Beat tracking as object detection (2025) | newer formulation. Confirms the two-head (beat/downbeat) design is now consensus |

**Recommendation:** Beat This! as primary and All-In-One for structure. Run them on the
**drums+bass stems** as well as the mix. Rhythmic stems give cleaner beats, and the two
estimates can be checked against each other.

## 5.2 Tempo map → MIDI

- Convert beat times to a **tempo map**: one `set_tempo` meta-event per beat (or per bar
  when tempo is stable). MIDI ticks then line up with musical beats even for live,
  drifting performances.
- Time signature: infer from downbeat spacing (beats per bar) with `time_signature` meta-events.
- Default PPQ 480 or 960.
- **Two export modes:**
  1. *Performance* — original timing, tempo map follows the performer. This is what DAW
     users want.
  2. *Constant tempo* — one global BPM. Notes keep their real-time positions, so they
     drift off the grid. Useful for some workflows.

## 5.3 Key and scale

- Global key plus **local key changes** (modulations), e.g. with key-profile correlation
  on chroma from the harmonic stems, or a CNN key estimator.
- Uses: key signature meta-event; **diatonic interval classification for harmonies**
  (file 04); pitch snapping for ambiguous vocal notes; a prior for octave and
  semitone error correction in bass and guitar.

## 5.4 Chord recognition

- Large-vocabulary chord recognition (maj/min/7/maj7/min7/sus/dim/aug/inversions).
  Recent work: **ChordFormer** (2025, conformer, +2% frame-wise / +6% class-wise accuracy
  over prior SOTA), plus 2025 analysis showing rare chords are still poorly recognized
  and pitch augmentation helps. Older open options: BTC (bi-directional transformer),
  Chordino (NNLS chroma).
- Uses in our pipeline:
  - **Prior for transcription fusion:** notes consistent with the current chord get a
    confidence boost; notes far outside it are re-checked.
  - **Bass octave correction:** prefer the octave of the chord root.
  - **Chord track output:** an optional MIDI track with block chords and chord-name text
    events, which is useful in itself.
- A simple, robust alternative: derive chords *from our own transcribed notes* (pitch-class
  histogram per beat), then reconcile with an audio chord model.

## 5.5 Rhythm quantization (performance MIDI → score-like MIDI)

| Work | Notes |
|------|-------|
| PM2S (Liu et al., ISMIR 2022) | CRNN for beats + quantization. Beats commercial software on MV2H. Open toolbox |
| Beyer & Dai (ISMIR 2024) | end-to-end piano performance-MIDI → score transformer |
| **Beat-based transformer quantization** (Klangio/KIT, 2025–2026) | T5 with custom MIDI tokenizer + beat-based pre-quantization. **ASAP: onset F1 97.3%, note-value accuracy 83.3%.** Proposes a confusion-based metric. Trained on piano and guitar |

**Approach for us:**
1. Use the beat grid from 5.1 (do not re-estimate it).
2. First pass: snap onsets to the nearest grid subdivision from {1/4, 1/8, 1/16, triplets}.
   Choose the subdivision per beat by minimum total deviation with a complexity penalty
   (prefer simpler rhythms).
3. Optional learned quantizer (beat-based transformer approach) for score-quality output.
4. Always export **unquantized and quantized** versions. Quantization is lossy, and
   users disagree on how much they want.

## 5.6 MIDI file format details

- **Type 1 SMF**, one track per part, track names set (`Lead Vocal`, `Harmony 1 (3rd
  above)`, `Bass`, `Drums`, `Piano`, `Guitar`, …).
- **General MIDI program changes** per track, drums on **channel 10**.
- **Pitch bend:** set the range per channel with RPN 0. Monophonic tracks only (bend is
  per-channel). For polyphonic expressive parts use **MPE** (one channel per note) or
  drop bends.
- **Velocity** from note energy relative to the stem's loudness distribution, mapped to
  roughly 20–120.
- **Meta-events:** tempo map, time signature, key signature, markers (sections:
  verse/chorus), text (harmony labels), optionally lyrics.
- Also consider **MIDI 2.0 / MusicXML** export for higher-resolution pitch or notation
  (later).
- Libraries: `pretty_midi`, `mido`, `miditoolkit` (Python). `music21` for MusicXML.
