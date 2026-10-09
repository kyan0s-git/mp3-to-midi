# 6. Proposed architecture

A hybrid **separate → specialize → fuse → verify** pipeline. Every stage is a swappable
module behind a small interface, because the state of the art in each sub-field moves
every few months.

## 6.1 High-level flow

```
                       ┌──────────────────────────────────────────────┐
 input.mp3/wav/flac ──►│ 0. Decode & normalize (ffmpeg → 44.1k stereo)│
                       └───────────────┬──────────────────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 1. Global analysis (on the mix)   │
                     │  • instrument tagging (timeline)  │
                     │  • beats/downbeats/sections       │
                     │  • key (global + local)           │
                     │  • predominant melody contour     │
                     └───────────────┬───────────────────┘
                                     ▼
          ┌──────────────────────────────────────────────────────────┐
          │ 2. Hierarchical source separation (ensembles)            │
          │   vocals ─► lead / backing                               │
          │   instrumental ─► drums ─► kick/snare/toms/hh/ride/crash │
          │                ─► bass                                   │
          │                ─► guitar, piano (if tagged present)      │
          │                ─► other (residual) ─► prompt-sep (opt.)  │
          └───────────────┬──────────────────────────────────────────┘
                          ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ 3. Per-stem transcription (≥2 models per stem when possible)        │
   │  lead vox : RMVPE/FCPE f0 → ROSVOT/SOME notes + bends               │
   │  backing  : vocal multi-f0 + Basic Pitch → voice tracking → notes   │
   │  bass     : low-f0 tracker + onsets → notes;  Basic Pitch x-check   │
   │  drums    : ADT on drum stem + per-piece onsets/velocity            │
   │  piano    : Transkun / hFT;  MuScriptor x-check                     │
   │  guitar   : guitar AMT / MuScriptor(guitar);  Basic Pitch x-check   │
   │  other    : MuScriptor / YourMT3+ (instrument list from tagger)     │
   └───────────────┬────────────────────────────────────────────────────┘
                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 4. Fusion & musical post-processing                           │
   │  • note-level consensus between models (weighted voting)      │
   │  • cross-stem de-duplication (bleed removal)                  │
   │  • key/chord priors, range constraints, octave correction     │
   │  • harmony classification vs. lead; melody track assembly     │
   └───────────────┬───────────────────────────────────────────────┘
                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 5. Verification by analysis-by-synthesis (iterative)          │
   │  render MIDI per stem → compare with stem spectrogram         │
   │  → find missed / spurious notes → re-decode those regions     │
   └───────────────┬───────────────────────────────────────────────┘
                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 6. Grid & export                                              │
   │  tempo map, time sig, key sig, markers; optional quantization │
   │  → song.mid (Type 1), per-part .mid, stems/, report.json      │
   └───────────────────────────────────────────────────────────────┘
```

## 6.2 Stage details

### Stage 0: Decode
- `ffmpeg` decodes any format to float32 stereo PCM at 44.1 kHz. Models resample
  internally as needed: Basic Pitch 22.05 kHz, RMVPE 16 kHz, separators 44.1 kHz.
- Keep stereo. The Mid/Side analysis for harmonies needs it.
- MP3 encoder delay/padding: use a decoder that honors the gapless (LAME/Xing) info so
  timestamps are exact.

### Stage 1: Global analysis
- **Instrument timeline:** a tagger over 1–3 s windows gives a "what is playing when"
  activity matrix. This decides which separators and transcribers to run, and suppresses
  hallucinated instruments in later stages (a fix for instrument leakage).
- **Beats, downbeats, sections:** Beat This! + All-In-One. Sections are reused for
  harmony summaries and markers.
- **Key:** global and windowed.
- **Predominant melody contour:** for the Melody track (file 04 §4.5).

### Stage 2: Separation
- Ensembles of RoFormer/SCNet models, applied hierarchically (file 02).
- Each stem gets a **quality estimate**. Options: an MVSEP-style quality checker, a
  residual-energy check (sum of stems vs. mix), or energy in out-of-range frequencies.
  Low quality lowers downstream confidence.
- Chunked inference with overlap-add (models typically run on 8–11 s chunks).

### Stage 3: Transcription
- Each transcriber outputs `NoteEvent(pitch, onset, offset, velocity, confidence,
  bend_curve?, source_model, stem)`.
- **Two or more models per stem** where available. Disagreement is the main
  confidence signal.
- Conditioning: pass instrument hints from the tagger to MuScriptor-style models.

### Stage 4: Fusion and post-processing
1. **Consensus voting:** match notes across models (onset within 50 ms, same pitch).
   Keep notes found by at least 2 models, or by one high-confidence specialist. Weight
   models per instrument using our own validation scores.
2. **Cross-stem de-duplication:** if the same pitch+onset appears in two stems (e.g. bass
   note also in "other" due to bleed), keep it in the stem with the higher local energy
   ratio for that pitch.
3. **Musical priors:**
   - Instrument range limits (bass E1–G4, voice ~E2–C6, etc.).
   - **Octave correction** with harmonic-sum checks and chord roots.
   - Key prior: out-of-key notes with low confidence are re-checked, *not* auto-snapped
     (chromaticism is real).
   - Minimum note length per instrument; merge fragmented sustains.
4. **Harmony analysis and melody track** (file 04).
5. **Velocity normalization** per track.

### Stage 5: Analysis-by-synthesis verification (the self-correction loop)
The key idea for accuracy beyond the single-model ceiling:
1. Render each stem's MIDI with a neutral synth. Better: a **timbre-matched** renderer, such
   as a differentiable or neural synth (DDSP-style) fitted to the stem, or a sample
   library for drums/bass/piano.
2. Compare the CQT/log-mel of the rendering with that of the stem: compute a frame×pitch
   **residual** (stem − render) and **excess** (render − stem).
3. Large positive residual at pitch p = **missed note**. Re-run the transcriber locally
   with a lower threshold, or add the note if multi-f0 salience supports it.
   Large excess = **spurious note**: remove it, or lower its velocity.
4. Iterate 1–3 times. This is like the "residual" refinement of iterative separation
   methods and like how humans check a transcription by playing it back.

Cost: rendering is cheap. Re-decoding only the flagged regions keeps it tractable.

### Stage 6: Export
- `song.mid` (all tracks), `parts/*.mid` (one per part), `stems/*.wav`, and `report.json`
  (confidence per note/track, detected key, tempo, sections, harmony analysis,
  model versions).
- Export modes: performance timing / quantized (file 05).

## 6.3 Data model (sketch)

```python
@dataclass
class NoteEvent:
    pitch: int                # MIDI note number
    onset: float              # seconds
    offset: float             # seconds
    velocity: int             # 1..127
    confidence: float         # 0..1 after fusion
    bend: list[tuple[float, float]] | None  # (time, cents) curve for expressive parts
    sources: list[str]        # models that agreed

@dataclass
class Part:
    name: str                 # "Lead Vocal", "Harmony 1 (3rd above)", "Bass", ...
    role: str                 # lead_vocal | harmony | melody | bass | drums | keys | guitar | other
    gm_program: int | None    # None for drums
    is_drum: bool
    notes: list[NoteEvent]
    stem_path: str | None
    meta: dict                # harmony interval labels, doubling flags, etc.

@dataclass
class Transcription:
    parts: list[Part]
    beats: list[float]; downbeats: list[float]
    tempo_map: list[tuple[float, float]]           # (time, bpm)
    time_signatures: list[tuple[float, int, int]]
    key_changes: list[tuple[float, str]]
    sections: list[tuple[float, float, str]]
    chords: list[tuple[float, float, str]]
```

## 6.4 Module interfaces (pluggable)

```python
class Separator(Protocol):
    def separate(self, audio: Audio, targets: list[str]) -> dict[str, Audio]: ...

class Transcriber(Protocol):
    instruments: set[str]
    def transcribe(self, audio: Audio, hints: Hints) -> list[NoteEvent]: ...

class F0Tracker(Protocol):
    def track(self, audio: Audio) -> F0Curve: ...   # times, hz, voicing, confidence
```

A YAML "recipe" chooses models per stem, so quality presets (fast / balanced / max) and
license presets (commercial-safe / research) are config changes, not code changes.

## 6.5 Compute budget (rough, for a 4-minute song on one modern GPU)

| Stage | Rough cost |
|-------|-----------|
| Vocal separation ensemble (2–3 RoFormers) | 30–90 s |
| Instrument separation (4/6 stems) + karaoke + drum split | 30–90 s |
| Lead SVT + backing multi-f0 | 5–20 s |
| Bass/drums/piano/guitar transcription | 10–40 s |
| MuScriptor-large on residual (1.4 B) | 20–60 s |
| Fusion + 2 analysis-by-synthesis passes | 10–30 s |
| **Total** | **~2–5 min (max-quality preset)** |

A CPU-only "fast" preset (HTDemucs + Basic Pitch + PESTO + small models) is feasible at
several minutes per song. These are estimates. Measure them in Phase 1.

## 6.6 Why this design can approach "most songs, no matter how complex"

- Complexity in a mix becomes **several simpler problems** through separation.
- Each sub-problem uses the **best specialist** available, and specialists can be upgraded.
- **Multiple-model consensus and musical priors** catch individual model errors.
- **Analysis-by-synthesis** turns "did we get it right?" into a measurable signal at
  inference time.
- **Confidence everywhere** lets the UI show users where to look, instead of claiming
  certainty.
