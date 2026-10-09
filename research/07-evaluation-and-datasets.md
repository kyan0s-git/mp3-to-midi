# 7. Evaluation and datasets

## 7.1 Metrics

### Note-level transcription (`mir_eval.transcription`)
- **Onset F1:** pitch within ±50 cents, onset within ±50 ms.
- **Onset+offset F1:** plus offset within max(50 ms, 20% of note duration).
- **Onset+offset+velocity F1:** plus velocity tolerance (as in the MAESTRO papers).
- **Frame-level F1:** per-frame pitch activity.
- **Multi-instrument F1** (MT3): the note must also carry the right instrument
  (program family). This is the right metric for us, because it penalizes leakage.

### Singing-voice transcription (Molina et al., 2014 framework)
- **COn** (correct onset), **COnP** (correct onset + pitch), **COnPOff** (+ offset).
  Standard on MIR-ST500 and in ROSVOT/VOCANO papers.

### f0 / melody (`mir_eval.melody`)
- **RPA** (raw pitch accuracy, within 50 cents), **RCA** (chroma accuracy, octave-agnostic,
  which exposes octave errors), **VR/VFA** (voicing recall / false alarm), **OA**
  (overall accuracy).

### Multi-f0 / harmony
- `mir_eval.multipitch` (frame-level precision/recall over f0 sets).
- **Our own harmony metrics** (nothing standard exists):
  - *Voice-count accuracy* per frame (did we find the right number of harmony voices?).
  - *Per-voice COnP* after optimal voice matching (Hungarian on note overlap).
  - *Harmony-label accuracy* (interval class vs. lead) per section.
  - *Doubling detection* precision/recall.

### Separation
- **SDR** (cSDR chunk-median, or uSDR), via `museval`. Report per stem. More important
  for us: a **bleed** measure, i.e. energy of other sources in a stem, since bleed causes
  phantom notes.

### Drums
- Per-class onset F1 (kick/snare/hi-hat/toms/cymbals), 50 ms tolerance (some papers use 30 ms).

### Rhythm / structure / score
- Beat F1, CMLt/AMLt (continuity), downbeat F1.
- **MV2H** (multi-pitch, voice, meter, value, harmony) and **MUSTER** for score-level output.

### End-to-end and perceptual
- **Re-synthesis similarity:** render our MIDI and compare with the original stems
  (spectral distance, chroma similarity).
- **Human A/B listening and edit-distance:** how many edits does a musician need to fix
  the MIDI? This is the metric users actually feel.

## 7.2 Public datasets

| Dataset | Content | Use |
|---------|---------|-----|
| **MUSDB18 / MUSDB18-HQ** | 150 tracks, 4 stems | separation benchmark |
| **MoisesDB** | 240 tracks, 45–47 artists, 12 genres, hierarchical stems (beyond 4) | separation beyond 4 stems. CC BY-NC-SA |
| **Slakh2100** | 2,100 synthetic multi-track songs rendered from MIDI | multi-instrument AMT train/eval |
| **MAESTRO v3** | ~200 h aligned piano | piano AMT |
| **GuitarSet** | acoustic guitar, hexaphonic annotations | guitar AMT |
| SynthTab, DadaGP | synthetic tab-aligned guitar / GuitarPro tabs | guitar pre-training |
| **MedleyDB v1/v2** | multitracks with melody (f0) annotations | melody, multi-f0, separation |
| **URMP** | chamber ensembles with per-instrument notes | multi-instrument AMT |
| MusicNet | classical, aligned notes | multi-instrument AMT |
| **MIR-ST500** | 500 pop songs, vocal note annotations | **pop singing transcription** |
| MIR-1K, MDB-stem-synth, PTDB | vocal / speech f0 | f0 trackers |
| DALI | vocal notes + lyrics from karaoke data (noisy) | SVT pre-training |
| GTSinger | multi-technique singing corpus with realistic scores | SVT, style |
| **Choral Singing Dataset, Dagstuhl ChoirSet, Cantoría, ESMUC** | multi-part vocal ensembles with per-part f0 | **harmony / vocal multi-f0** |
| **MedleyVox** | 1.1 h multi-singer evaluation (unison, duet, main-vs-rest, N-singing) | multi-singer separation |
| ADTOF datasets, E-GMD, Groove MIDI | drums | ADT |
| StemGMD | isolated synthetic drum-piece stems | drum separation |
| Harmonix Set, SALAMI | beats, downbeats, sections | grid / structure |
| ASAP | aligned performance↔score piano | quantization |
| 2025 AMT Challenge test set | custom-composed multi-instrument pieces | unbiased multi-instrument eval |

## 7.3 Our own evaluation set (recommended)

Public data doesn't cover "real pop/rock/hip-hop/EDM mixes with harmonies," so build one:

1. **Multitrack-sourced set (best ground truth):** songs with released stems or
   multitracks (licensed remix packs, MedleyDB, MoisesDB, Cambridge-MT-style mixing
   multitracks; check each license). Transcribe the *isolated* stems semi-automatically
   (monophonic, easy), then have humans correct them. This gives near-exact per-part
   ground truth, including harmony parts.
2. **Synthetic harmony set:** take real lead vocal stems, generate diatonic harmony
   parts (formant-preserving pitch shift following the key, or a singing-voice
   synthesizer), and mix them with known intervals, panning and doubling. Exact labels
   at scale.
3. **Stress set:** 30–50 hard real songs (dense metal, a cappella with 5+ voices,
   gospel choir, jazz with walking bass, EDM with sidechained pads, lo-fi with tape
   wow, live recordings with rubato). Evaluate qualitatively and by edit distance.
4. **Genre coverage matrix:** make sure each release is scored per genre, not only on average.

Lock this set early. Every model swap or tuning change gets a regression report with a
per-stem and per-metric table.

## 7.4 Evaluation harness

```
eval/
  datasets/      loaders → (audio, reference parts as Transcription)
  metrics.py     wrappers around mir_eval + custom harmony metrics
  run_eval.py    pipeline recipe × dataset → results.parquet
  report.py      per-stem / per-genre tables, regressions vs. baseline
```
