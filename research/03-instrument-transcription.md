# 3. Instrument transcription (AMT)

## 3.1 General-purpose / multi-instrument models

| Model | Year | Architecture | Instruments | License | Notes |
|-------|------|--------------|-------------|---------|-------|
| **Basic Pitch** (Spotify) | 2022 | tiny CNN with harmonic stacking; outputs onsets, notes, **pitch bends** | instrument-agnostic (no instrument labels) | Apache-2.0 | Very fast (up to ~10× real time on CPU). Works best on one instrument at a time. Beats a 2020 multi-instrument model, loses to piano/guitar specialists. **Excellent per-stem fallback and cross-checker** |
| Omnizart | 2021 | collection of models (piano, multi-instrument, vocal, drum, chord, beat) | various | MIT | Older, but a convenient all-in-one toolkit |
| **MT3** (Google Magenta) | 2022 | T5 encoder-decoder, spectrogram → MIDI-like tokens | multi-instrument (program numbers) | Apache-2.0 | Trained jointly on six datasets. Introduced the "multi-instrument F1". Big gains on low-resource instruments. Suffers from instrument leakage. Baseline F1 ≈ 0.39 on the 2025 AMT Challenge test set |
| MR-MT3 | 2024 | MT3 + memory retention | multi | — | Targets instrument leakage |
| **YourMT3+** | 2024 | hierarchical time-frequency attention + mixture of experts; cross-dataset *stem augmentation* | multi-instrument **including singing** | open (see repo) | Strong on Slakh. Authors note weaker results on real-world recordings. Basis of several 2025 challenge entries |
| MIROS system (2025 AMT Challenge winner) | 2025 | YourMT3+ framework with **MusicFM** foundation-model encoder; parallel per-instrument-group decoding via a recurrent adapter | multi | research | Challenge F1 **0.60** vs MT3 0.39. ~370 M params. Slakh F1 0.83. Domain-pretrained encoders help generalization |
| **MuScriptor** (Kyutai + Mirelo) | Jul 2026 | decoder-only Transformer; mel input + **list of target instruments**; RL post-training | multi (guitar, bass, drum kit, …; tab output for fretted instruments) | code MIT, **weights CC BY-NC 4.0** | Trained on 170k recordings / 11k hours of real aligned data plus synthetic data. 103 M / 307 M / 1.4 B variants. Self-reported Multi-F1 48.2 vs 21.9 for YourMT3+ on their 372-track test set. `uvx muscriptor transcribe song.wav`. Worse on rubato. Vocals not listed |

### Lessons from the 2025 AMT Challenge (arXiv 2603.27528)
- Only 2 of 8 valid submissions beat MT3. MT3 is still a strong baseline.
- The winner used a **pretrained music foundation model (MusicFM) as encoder**. This
  generalized better than YourMT3+, which appears to overfit Slakh.
- Accuracy falls sharply with the number of simultaneous instruments: about 0.72 F1 for
  one instrument, about 0.44 for three.
- Persistent errors: **instrument leakage / hallucinated instruments**, and confusion
  between instruments with overlapping range and timbre.
- **Data is the bottleneck.** This supports MuScriptor's thesis that large-scale real
  aligned data matters more than architecture.

**Implication for us:** use the multi-instrument models (a) on the residual "other" stem,
(b) on *individual* stems with the target instrument specified (MuScriptor accepts an
instrument list), and (c) as a second opinion. Do not rely on them for the full mix.

## 3.2 Piano (and keys)

Piano is the most mature AMT sub-field because of MAESTRO (about 200 h of aligned
piano performances).

| Model | Onset F1 (MAESTRO) | Onset+offset F1 | Notes |
|-------|--------------------|-----------------|-------|
| Onsets & Frames (2018) | ~94.8 | — | classic baseline |
| ByteDance high-res (Kong et al., 2021) | 96.72 | — | regresses precise onset/offset times. **Pedal transcription** (pedal onset F1 91.86) |
| hFT-Transformer (Sony, 2023) | ~97.4 | ~90.3–90.5 | SOTA on frame/note/offset/velocity at publication |
| **Transkun** (2023–24) | 98.32 | **93.48** | semi-CRF; strong offsets (pedal-aware) |
| SFT-CRNN (optimal transport, 2026) | 98.36 | 90.78 | new onset SOTA; no pedal detection yet |
| Aria-AMT (EleutherAI) | — | — | piano AMT trained for robustness on real-world recordings; we did not verify its numbers |

Caveat: these numbers are on clean solo piano. On a separated piano stem from a pop mix,
expect a large drop because of bleed and processed "keys" sounds (Rhodes, synth
keys). Fine-tuning on separated stems or on stems with artificially added bleed is
advisable.

## 3.3 Guitar

- **Data:** GuitarSet (acoustic, hexaphonic pickup annotations), plus synthetic sets such
  as SynthTab (rendered from DadaGP tabs with commercial plugins) and procedurally
  generated audio (AIMC 2025).
- **Findings:** pre-training on synthetic tab data greatly reduces cross-dataset
  overfitting, and fine-tuning with a little real data improves it further.
  Real-tone/effects augmentation (Pedroza et al.) improves robustness to electric guitar
  tones. HeavyCNN (2025) handles drop tunings by conditioning on the tuning.
- **String/fret assignment** is a separate symbolic problem (Fretting-Transformer, 2025).
  This matters only if we output tabs. For MIDI, pitch is enough.
- **Distorted electric guitar power chords** are still hard. Distortion creates
  intermodulation partials that look like extra notes. Remedies: transcribe with a
  guitar-specialized model, constrain chords to plausible guitar voicings, and use
  chord-recognition context.
- MuScriptor explicitly supports guitar (with tab output). YourMT3+ covers guitar.
  Basic Pitch is reasonable on clean guitar.

## 3.4 Bass

Bass is mostly monophonic, which makes it one of the easier parts once separated.
- **Approach:** monophonic f0 tracking (CREPE, PESTO, RMVPE or FCPE configured for
  low frequencies), plus an onset detector, plus note segmentation (HMM/Viterbi over
  semitone states, the pYIN-style approach). Basic Pitch on the bass stem is a strong
  cross-check.
- **Classic issues:** **octave errors** (the fundamental is weak relative to the 2nd
  harmonic on small speakers and on synth bass), slides, ghost notes, and kick-drum
  bleed causing false onsets.
- **Fixes:** harmonic-sum salience; enforce bass range (E1–G3 for 4-string, B0 for
  5-string, lower for synth); prefer the octave consistent with chord roots; suppress
  onsets that coincide with kick hits and have no pitch change.
- **Pitch bends:** keep slides as MIDI pitch-bend, or optionally as "glide" notes.

## 3.5 Drums (automatic drum transcription, ADT)

| System | Notes |
|--------|-------|
| **ADTOF** (Zehren et al.) | Trained on large crowdsourced rhythm-game charts. 5 classes (kick, snare, toms, hi-hat, cymbals). Main open baseline. **CC BY-NC-SA 4.0**. PyTorch port available. ISMIR 2024 analysis: toms largely solved, **cymbals remain weakest** |
| ADTOF + drum separation (Riley & Dixon 2024) | 7 classes plus **velocity estimated from separated stems** |
| **Noise-to-Notes (N2N)** (Sony, Sept 2025) | ADT as conditional **diffusion**. Generates onsets *and velocities*. Reports new SOTA across ADT benchmarks. Music-foundation-model features improve robustness. Code/weight availability unverified |
| STAR Drums (TISMIR 2025) | dataset method: separate drums, auto-annotate, resynthesize with samples. Useful for making training data |
| Omnizart drum, MT3/YourMT3+/MuScriptor drum tracks | usable cross-checks |

**MIDI output:** channel 10, General MIDI drum map (36 kick, 38 snare, 42/46 closed/open
hi-hat, 49/57 crash, 51 ride, 45/47/48/50 toms). Velocity is computed from per-piece
stem energy at the onset.

## 3.6 Strings, brass, winds, synths, and the long tail

- Covered (with decreasing reliability) by MT3, YourMT3+ and MuScriptor through program
  numbers.
- **Synth pads, arps and FX** have no stable timbre. Transcribe pitch with Basic Pitch
  and label the track generically ("Synth Pad", GM program 89–96). The pitch content is
  usually more valuable than the instrument label.
- **Instrument identification** decides what to label and which specialist to call.
  Options: an audio tagger (AST/PANNs or a MERT/MusicFM-based classifier fine-tuned on
  instrument tags), or MuScriptor/YourMT3+ token statistics.

## 3.7 Recommendations

1. Per stem, run **one specialist + one generalist** (e.g. Transkun + MuScriptor on
   piano, or bass-f0 + Basic Pitch on bass), and fuse (file 06).
2. Run **MuScriptor or YourMT3+ on the "other" residual** with instrument hints from a
   tagger.
3. Use **Basic Pitch as a universal fallback** and pitch-bend source. It is Apache-2.0,
   fast and robust.
4. Budget time to **fine-tune on separated stems** (with realistic bleed) rather than
   clean recordings. This closes much of the clean→real gap.
5. Watch the licenses: MuScriptor weights and ADTOF are non-commercial.
