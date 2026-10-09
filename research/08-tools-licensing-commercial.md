# 8. Tools, licensing, and the commercial landscape

## 8.1 License matrix (verify before shipping, because licenses change)

| Component | Code license | Weights / data license | Commercial use? |
|-----------|--------------|------------------------|-----------------|
| Demucs / HTDemucs | MIT | MIT (released checkpoints) | ✅ |
| ZFTurbo MSST framework | MIT | **varies per checkpoint** (many community weights carry no explicit license) | ⚠️ check each |
| BS-RoFormer / Mel-RoFormer implementations (e.g. lucidrains) | MIT | depends on who trained the weights | ⚠️ |
| SCNet | see repo | see repo | ⚠️ |
| UVR / audio-separator models | MIT (tool) | mixed | ⚠️ |
| SAM Audio | see Meta release | see Meta release (Meta research licenses often restrict use) | ⚠️ |
| Basic Pitch | Apache-2.0 | Apache-2.0 | ✅ |
| MT3 | Apache-2.0 | Apache-2.0 (checkpoints) | ✅ (verify) |
| YourMT3 / YourMT3+ | see repo | see repo | ⚠️ |
| **MuScriptor** | MIT | **CC BY-NC 4.0** | ❌ (weights) |
| Transkun | see repo | see repo | ⚠️ |
| hFT-Transformer (Sony) | see repo | see repo | ⚠️ |
| ADTOF | — | **CC BY-NC-SA 4.0** | ❌ |
| LarsNet / LARS | open | **CC BY-NC 4.0** | ❌ |
| madmom | BSD (code) | **CC BY-NC-SA (models)** | ❌ (models) |
| Beat This! | reported MIT | reported MIT | ✅ (verify) |
| CREPE | MIT | MIT | ✅ |
| PESTO | see repo (LGPL reported for early versions) | — | ⚠️ |
| RMVPE, FCPE, ROSVOT, SOME | see repos | see repos | ⚠️ |
| mir_eval, pretty_midi, mido, librosa | MIT/ISC | — | ✅ |
| MoisesDB | — | CC BY-NC-SA 4.0 (non-commercial research) | ❌ for training commercial models (check) |
| Slakh2100 | — | CC BY 4.0 | ✅ |
| MAESTRO | — | CC BY-NC-SA 4.0 | ❌ for commercial training (check) |
| MUSDB18 | — | research/non-commercial | ❌ |

**Implications:**
- A **personal / research tool** can use the strongest models today.
- A **commercial product** needs: (a) only permissively licensed weights (HTDemucs, Basic
  Pitch, MT3, CREPE, Beat This!), and/or (b) **our own trained checkpoints** on
  commercially licensable data, and/or (c) a license deal (e.g. Kyutai/Mirelo for
  MuScriptor, or a paid separation API such as AudioShake or Music.AI).
- Keep a `LICENSES.md` / model registry with the license of every checkpoint the
  pipeline can load, and a "commercial-safe" recipe that refuses to load NC weights.

## 8.2 Core open-source building blocks

| Purpose | Library |
|---------|---------|
| Decode/encode | ffmpeg, `soundfile`, `torchaudio` |
| DSP / features | `librosa`, `nnAudio` (GPU CQT), `torchaudio` |
| Separation | `demucs`, ZFTurbo MSST, `audio-separator` |
| Transcription | `basic-pitch`, `muscriptor`, MT3/YourMT3 repos, Transkun, Omnizart |
| f0 | `torchcrepe`, RMVPE, FCPE (`torchfcpe`), `pesto-pitch`, pYIN (librosa) |
| Beats/structure | `beat_this`, `allin1`, `madmom` (non-commercial models) |
| MIDI I/O | `pretty_midi`, `mido`, `miditoolkit`, `symusic` (fast) |
| Notation | `music21`, MuseScore CLI (MusicXML → PDF) |
| Rendering (for A-by-S) | FluidSynth + SoundFonts, `pretty_midi.fluidsynth`, DDSP |
| Evaluation | `mir_eval`, `museval` |
| Deployment | ONNX Runtime / TorchScript, Triton for GPU serving |

## 8.3 Commercial landscape (competitors / benchmarks for "good enough")

| Product | What it does | Notes |
|---------|--------------|-------|
| **Celemony Melodyne 5** | "DNA" polyphonic note detection, editing, **audio-to-MIDI export** | Gold standard for vocal/monophonic note detection and editing. Works on a single recorded track, not a full mix |
| **RipX DAW** (Hit'n'Mix) | separates a song into layers and exposes them as editable notes, MIDI export | Closest existing product to the "whole song → notes" goal |
| **Klangio** (Piano2Notes, Guitar2Tabs, Sing2Notes, Drum2Notes, Transcription Studio) | AI transcription to MIDI/sheet music/tabs per instrument | The company behind the beat-based quantization research (file 05) |
| AnthemScore | piano-focused audio → sheet music | older CNN approach |
| Samplab | audio-to-MIDI and polyphonic editing for samples | |
| Logic Pro / Ableton Live / Cubase | built-in audio-to-MIDI (melody, harmony, drums) and stem separation | Ableton native separation is 4 stems |
| NeuralNote (open-source plugin) | wraps Basic Pitch as a VST/AU | good reference UX |
| Moises / Music.AI, LALAL.AI, AudioShake, MVSEP | stem separation services (AudioShake and MVSEP offer **lead/backing vocal** split) | possible commercial APIs for separation |
| ScoreCloud | melody → notation | |

**Gap we can fill:** none of these produce, in one step, a *complete multi-track MIDI of a
full mix* with *lead, labelled harmony parts, melody, bass and drums* plus confidence
information. RipX comes closest but doesn't do structured harmony analysis.

## 8.4 Ethics and copyright

- Transcribing a copyrighted recording creates a derivative representation of the
  composition. That is fine for personal study and for the user's own files, but
  redistribution can infringe. The product should make users responsible for their
  inputs and should not host or share transcriptions of commercial songs publicly by
  default.
- If lyrics are ever extracted (file 04 §4.6), treat them as the user's private data. Do
  not store or display them beyond the user's own session.
- Training-data provenance matters for commercial models. Prefer licensed, synthetic or
  user-consented data.
