# 4. Vocals, melody, and harmony detection

This is the hardest *and* most valuable part of the request: separate the vocals, transcribe
the lead accurately, **detect and transcribe harmonies**, and produce a "melody" track.

## 4.1 Sub-problems

1. Get a clean **vocal stem** (all voices). See file 02.
2. Split it into **lead** and **backing** stems with a karaoke model. See file 02.
3. **Lead vocal → MIDI** (monophonic, expressive).
4. **Backing vocals → N harmony parts** (polyphonic vocal ensemble).
5. **Classify** each harmony part relative to the lead (double, octave, third above, …).
6. Build a **melody** track: the lead vocal, plus the lead instrument where nobody is
   singing.

## 4.2 Lead vocal: f0 estimation

Singing needs a *continuous* f0 curve first. Notes come second.

| Tracker | Type | Notes |
|---------|------|-------|
| pYIN | DSP + HMM | classic, no training, decent on clean stems; fallback |
| CREPE | supervised CNN | well known, robust; heavier |
| **RMVPE** | supervised, designed for vocals in polyphonic music | de facto standard in the singing-voice community (RVC, ROSVOT and STARS all use it). ~90 M params |
| **FCPE** (2025) | fast context-based | 96.79% RPA on MIR-1K, comparable to RMVPE, ~5.3× faster, 10.6 M params (self-reported) |
| **PESTO** (TISMIR 2025) | self-supervised, transposition-equivariant | tiny, <10 ms latency. Competitive with supervised methods on MIR-1K / MDB-stem-synth |

**Recommendation:** RMVPE as primary, FCPE or PESTO as a fast second opinion. Agreement
between them doubles as a confidence score. Run them on the **separated lead stem**.
RMVPE also tolerates some accompaniment, which helps when separation leaves bleed.

## 4.3 Lead vocal: f0 → notes (singing voice transcription, SVT)

Turning a wobbly f0 curve into notes is the core SVT problem: where notes start and end,
and which MIDI pitch a scooped, vibrato-laden note "is".

| System | Year | Notes |
|--------|------|-------|
| VOCANO | 2021 | note transcription of singing voice *in polyphonic music*. Trained on DALI (mostly Western pop). Common baseline |
| MIR-ST500 models (Wang & Jang) | 2021 | 500 pop songs annotated with vocal notes. Standard pop SVT benchmark |
| **ROSVOT** | ACL 2024 | multi-scale model with RMVPE f0 input. Robust to noise and can transcribe **raw songs with accompaniment**. +17% relative over the previous best on pitch, +23% under noise |
| **SOME** (openvpi) | 2023– | Singing-Oriented MIDI Extractor. Faster than real time on CPU/GPU. Optional RMVPE. Outputs **non-integer MIDI pitches** (good for preserving intonation). Trainable on ~3 h of data |
| SongTrans | 2024 | joint lyrics + notes for singing in polyphonic music. On par with ROSVOT/VOCANO at note level |
| **STARS** | ACL Findings 2025 | unified transcription, alignment and *style* annotation (techniques such as vibrato and breathy) |
| **VocalParse** | May 2026 | built on Qwen3-ASR (an audio LLM). Decodes lyrics first, then interleaved lyric–note pairs. Reports SOTA SVT on several datasets. **Trained mainly on Mandarin**; pre-segment audio over 30 s |

**Recommended lead-vocal recipe:**
1. RMVPE f0 + voicing on the lead stem.
2. Note segmentation with ROSVOT or SOME (evaluate both on MIR-ST500 and our own test set).
3. **Pitch quantization with expression kept:** each note gets the *median* f0 over
   its stable middle portion, rounded to the nearest semitone. Optionally snap to the
   detected key when it lies within ±40 cents of a scale tone. The residual deviation
   (scoops, vibrato, falls) is written as **MIDI pitch-bend**, with the bend range set by
   RPN 0 to ±2 semitones (or ±12 for big slides).
4. Two outputs: an **"expressive" lead track** (with bends) and a **"clean" lead track**
   (quantized notes only) for notation and MIDI editing.
5. Heuristics: merge notes separated by <30–50 ms at the same pitch (consonant breaks).
   Drop notes shorter than about 60 ms unless they have a clear onset. Treat melisma
   (several pitches on one syllable) as separate notes.

## 4.4 Harmony / backing vocals: the core design

There is **no off-the-shelf model that takes a pop song and outputs "harmony 1, harmony
2, harmony 3" as MIDI.** The search for 2024–2026 work found strong single-voice SVT,
choir multi-f0 work and multi-singer separation datasets, but no end-to-end
pop-harmony transcriber. We have to compose one.

### 4.4.1 Prior work to build on
- **Cuesta et al., "A Framework for Multi-f0 Modeling in SATB Choir Recordings" (SMC
  2019)** and **"Multiple F0 Estimation in Vocal Ensembles using CNNs" (ISMIR 2020).**
  Deep-salience-style CNN multi-f0 for vocal ensembles. One singer per part is estimated
  "to a relatively high degree of accuracy". **Unison (several singers per part) is
  much harder.** They model each part's f0 *and its dispersion*. Code is public
  (`multif0-estimation-polyvocals`).
- **Schramm & Benetos, "Automatic Transcription of Polyphonic Vocal Music" (2017).**
  PLCA with a music language model that detects concurrent pitches *and assigns them to
  voice types*, evaluated on Bach chorales and barbershop quartets. Older, but the
  voice-assignment idea transfers directly.
- **MedleyVox (2023):** audio-domain separation of multiple singers (duet, unison,
  main-vs-rest). Useful to split two-part harmony stems before monophonic SVT.
- **Choir/ensemble datasets:** Choral Singing Dataset, Dagstuhl ChoirSet, Cantoría, the
  ESMUC choir set. These are useful for training/fine-tuning a vocal-ensemble multi-f0
  model.

### 4.4.2 Proposed harmony pipeline

```
backing_vocal_stem
   │
   ├─(A) multi-f0 salience  ── vocal-ensemble CNN (Cuesta-style) + Basic Pitch (pitch posteriorgram)
   │                           → frame × pitch salience map (20-cent resolution)
   │
   ├─(B) peak picking per frame → candidate f0s (limit to ≤4–5 voices, vocal range ~E2–C6)
   │
   ├─(C) voice tracking: link candidates over time into continuous "voices"
   │      • cost = |Δpitch| + crossing penalty + salience reward + gap penalty
   │      • solve with Viterbi / min-cost-flow / Hungarian per frame + HMM smoothing
   │      • prefer non-crossing voices (Soprano > Alto > …) but allow crossings
   │
   ├─(D) per-voice note segmentation  (same SVT logic as the lead: onset/offset + quantize)
   │
   ├─(E) remove lead leakage: drop backing notes that match the lead's notes in time and
   │      pitch AND whose energy correlates with lead energy (bleed, not a real double)
   │
   └─(F) harmony classification vs. lead and key (see 4.4.3)
```

Optional (B′): if the tracker finds exactly two well-separated voices for long stretches,
run a **MedleyVox-style duet separator**, then monophonic SVT on each output. This is
usually more precise than multi-f0.

### 4.4.3 Harmony classification (the "detect harmonization" requirement)

For each backing note `b` that overlaps a lead note `l` in time:

| Interval (b − l), in semitones | Label |
|-------------------------------|-------|
| 0 (±30 cents), energy-correlated with lead | **double / unison** — usually *not* a separate MIDI part; flag as "doubled" |
| ±12 / ±24 | **octave double** |
| +3/+4 (diatonic) | **third above** (most common pop harmony) |
| −3/−4 | **third below** |
| +7 / −5 | **fifth above / fourth below** |
| −7 / +5 | **fifth below / fourth above** |
| varies, sustained across lead notes, on vowels | **pad / "oohs" / block-chord backing** |
| no concurrent lead note | **response / call-and-answer / ad-lib** |

Diatonic intervals depend on key: a "third above" is 3 *or* 4 semitones depending on the
scale degree. So classify in **scale-degree space** using the detected key (file 05).
Then summarize per *section*, for example "Chorus: lead + diatonic 3rd above + 5th
below; verse 2: octave double".

Output in MIDI:
- Separate tracks named `Lead Vocal`, `Harmony 1 (3rd above)`, `Harmony 2 (5th below)`,
  `Backing Pad`, `Ad-libs`.
- Optional **marker/text meta-events** with the harmony labels per section.
- A JSON sidecar with confidence per note and the harmony analysis.

### 4.4.4 Hard cases and mitigations
- **Unison doubles / tracked leads** (the same singer recorded 2–4 times). Spectrally
  these look like one voice with chorus. Detect them with pitch agreement within about
  30 cents, high envelope correlation, and slight timing offsets (10–40 ms). Report
  "lead is doubled" instead of creating duplicate notes.
- **Stacked harmonies panned wide:** use **stereo information**. Run multi-f0 on L, R,
  Mid and Side channels separately. Harmonies are often panned off-center while the lead
  sits in the center, which makes Side-channel analysis very informative.
- **Gang vocals / crowd / choir sections:** many singers per part. Use the dispersion
  modeling from Cuesta et al. and output one note per part, not one per singer.
- **Vocoder / auto-tune / talkbox:** pitch is quantized and buzzy. f0 trackers still work.
  Label the part.
- **Heavy reverb:** tails create phantom sustained notes. Cap offsets using an energy
  decay threshold relative to the onset energy.

## 4.5 The "melody" track

"Melody" is not always the lead vocal. Instrumental intros, solos and hooks also carry it.

1. Run **predominant-melody estimation on the full mix** (deep-salience melody model,
   Melodia-style, or the vocal f0 itself when vocals are active).
2. Per segment, choose the **stem whose transcription best matches the predominant melody
   contour**: lead vocal when singing, otherwise the instrument (guitar solo, synth lead,
   sax, …).
3. Write a dedicated **`Melody` track** that concatenates these segments. This is often
   what users actually want ("the tune").

## 4.6 Lyrics (optional, later)

Aligned lyrics can be added as MIDI lyric meta-events (karaoke-style). This needs ASR on
the lead stem (Whisper-family with word timestamps, or SongTrans/VocalParse joint
models). It is a separate feature. Note that song lyrics are copyrighted. If this is
added, show lyrics only for the user's own file and do not store or redistribute them.

## 4.7 Recommendations

1. Vocal stem → Mel-RoFormer karaoke → lead + backing.
2. Lead: RMVPE (plus FCPE cross-check) → ROSVOT or SOME → semitone notes + pitch-bend.
3. Backing: vocal-ensemble multi-f0 + Basic Pitch → voice tracking → per-voice notes →
   lead-leak removal → harmony labels in scale-degree space.
4. Use stereo (L/R/M/S) analysis for harmonies.
5. Build a **small hand-annotated pop harmony test set** (20–50 song excerpts). Nothing
   public covers this well, and it is our differentiator.
6. Research track: fine-tune a multi-f0 model on synthetic pop backing vocals. Create
   them by pitch-shifting real lead stems to diatonic harmonies with formant-preserving
   shifting, or with singing-voice synthesis (DiffSinger-style), then mixing with a lead.
   This gives exact ground truth for harmony parts at scale.
