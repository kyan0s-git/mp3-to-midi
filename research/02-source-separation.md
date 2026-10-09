# 2. Music source separation (MSS)

Separation is the backbone of the pipeline. Transcription quality per part is limited
by stem quality: bleed from other instruments becomes phantom notes.

## 2.1 Landscape (as of late 2026)

| Model | Year | Notes | Reported quality |
|-------|------|-------|------------------|
| Spleeter (Deezer) | 2019 | 2/4/5 stems, fast, dated | superseded |
| Open-Unmix | 2019 | reference baseline | superseded |
| **HTDemucs / Demucs v4** (Meta) | 2022 | hybrid waveform+spectrogram transformer; 4-stem and 6-stem (`htdemucs_6s` adds guitar, piano); MIT license | ~9.0–9.2 dB avg SDR on MUSDB18-HQ (9.20 with extra data + fine-tuning) |
| MDX23C / MDX-Net | 2023 | Sound Demixing Challenge models; many community checkpoints (UVR) | e.g. MDX23C vocals SDR ≈ 10.2 (ZFTurbo table) |
| **BS-RoFormer** (ByteDance) | 2023 | band-split + rotary-position transformer; SDX'23 MSS winner | 9.80 dB avg SDR (small, no extra data); higher with extra data |
| **Mel-Band RoFormer** | 2023 | mel-scale band split; the base for many of the best community vocal and karaoke models | 11.99 dB avg SDR reported with MUSDB18-HQ + 500 extra songs |
| **SCNet / SCNet XL** | 2024 | sparse-compression network; comparable to BS-RoFormer with ~7× fewer parameters | SOTA-competitive on MUSDB18-HQ |
| Diffusion/consistency refinement on BS-RoFormer | 2024–25 | post-refines a separator's output | reports new SOTA on MUSDB18 over BS-RoFormer |
| Band-SCNet | 2025 | causal, real-time (92 ms latency, 2.59 M params) | 7.79 dB SDR — for real-time use only |
| Moises-Light | 2025 | resource-efficient band-split U-Net | efficiency-focused |
| **SAM Audio** (Meta) | Dec 2025 | prompt-based "separate anything": text, visual click, or time-span prompt | flexible, but Meta's paper says text-prompted *instrument* separation lags domain-specific models such as Demucs |
| Music Source Restoration line | 2026 | separation + GAN restoration (BS-RoFormer + HiFi++) | aims at clean, de-mastered stems |

Takeaways:
- **The RoFormer family plus SCNet is the current workhorse.** The best results in
  practice come from *ensembles* of them. MVSEP's ensemble combines BS-RoFormer,
  Mel-RoFormer and SCNet XL vocal models. Adding SCNet XL raised vocals SDR from
  11.54 to 11.61 on their multisong benchmark.
- **HTDemucs is still the easy, permissively licensed default.** It is also the only
  well-known open model with **guitar and piano stems** (`htdemucs_6s`), although
  piano quality is weak.
- **Benchmarks are not directly comparable.** Results differ by whether extra training
  data was used, MUSDB18 vs MUSDB18-HQ, cSDR vs uSDR, and community "multisong" sets.

## 2.2 Hierarchical / cascaded separation (the key trick)

No single model produces every stem we need. Chain specialized models instead:

```
mix
 ├─ vocals ──────────────► karaoke model ─┬─ lead vocal
 │                                         └─ backing vocals ─► (multi-singer sep.) ─► harmony parts
 └─ instrumental
     ├─ drums ───────────► drum separator ─► kick / snare / toms / hi-hat / ride / crash
     ├─ bass
     ├─ (guitar)  ◄─ 6-stem model or a dedicated guitar model
     ├─ (piano)   ◄─ 6-stem model or a dedicated piano/keys model
     └─ other (residual) ─► multi-instrument transcriber / prompt-based separator
```

Practical findings from the community (UVR and MVSEP users):
- **Run the vocal model first, then the instrument models on the instrumental.** This
  cuts vocal bleed in the bass and "other" stems.
- **Run karaoke (lead/back) models on the *vocal stem*, not on the full mix.** MVSEP
  notes that running MDX-B Karaoke on a pre-split vocal stem gives higher quality and
  cleanly pulls backing vocals into their own track.
- **Best lead/back models as of late 2025 are Mel-RoFormer karaoke models.** MVSEP
  added four such models in September 2025. The best reports **lead-vocal SDR ≈ 10.1**
  on their lead/back benchmark (self-reported). Older options: UVR's `UVR_MDXNET_KARA_2`
  and MDX-B Karaoke.
- **Ensemble by averaging** (waveform or spectrogram mask) across 2–4 strong models.
  This consistently gives a few tenths of a dB SDR. For transcription, the bigger win
  is less *bleed*, which ensembles also reduce.

## 2.3 Drum separation (drum stem → individual pieces)

| Model | Stems | License | Notes |
|-------|-------|---------|-------|
| LarsNet (Politecnico di Milano, 2024) | kick, snare, toms, hi-hat, cymbals | weights CC BY-NC 4.0 | bank of U-Nets trained on StemGMD (synthetic). Faster than real time. LARS plug-in wraps it |
| DrumSep (MDX23C-based, community) | kick, snare, hi-hat, toms, ride, crash | check checkpoint | anecdotally better than LarsNet (MVSEP users); packaged as an Ableton extension |

Riley & Dixon (ISMIR 2024 LBD) combined ADTOF transcription with open drum-separation
models. This expanded the output from 5 to 7 drum classes and **estimated MIDI velocities
from the separated stems**. We should do the same.

## 2.4 Lead/backing vocals and multi-singer separation

This matters for the "detect harmonization" requirement:

- **Lead vs. backing (2 outputs):** solved well enough by the karaoke models above.
  Commercial: AudioShake, MVSEP, LALAL.AI.
- **N singers → N stems:** a research problem. **MedleyVox** (Jeon et al., ICASSP 2023)
  defines the tasks (unison, duet, main-vs-rest, N-singing) and a 1.1 h evaluation set
  from MedleyDB. It builds training mixtures from single-singer datasets and adds an
  improved super-resolution network (iSRNet) on top of Conv-TasNet. On duet and unison it
  approaches ideal-mask performance.
- **Two-stage singer separation for karaoke** (Wave-U-Net front end + DPRNN/DPTNet) is a
  more recent research direction.
- **Takeaway:** we can reliably get *lead* and *all backing*. Splitting the backing stem
  into individual harmony voices is better done **in the pitch domain** (multi-f0 +
  voice tracking, file 04) than in the audio domain. Audio-domain multi-singer
  separation is an optional refinement.

## 2.5 Prompt-based / open-vocabulary separation

For "whatever it is thrown" (sitar, accordion, brass section, theremin):
- **SAM Audio** (Meta, arXiv 2512.18099) separates by text prompt ("trumpet"), by time-span
  prompt ("the sound here"), or by visual prompt. It was trained on about 1M hours of
  audio-video. Meta reports it lags specialized models on standard instrument stems, and
  third parties report heavy hardware requirements. **Use it only for long-tail
  instruments that the fixed-stem models put into "other",** not for vocals, bass or drums.
- Text-queried separation (AudioSep, LASS-style) is an older, weaker alternative.

## 2.6 Tooling

- **ZFTurbo `Music-Source-Separation-Training` (MSST):** MIT-licensed training and
  inference framework. Supports BS-RoFormer, Mel-RoFormer, SCNet, MDX23C, HTDemucs,
  BandIt and more, with a published table of pretrained checkpoints and SDRs. A
  framework paper was published in 2026 (arXiv 2607.23395). It is maintained by the
  MVSEP author. **This is the most practical base for our separation stage.**
- **`audio-separator` (python-audio-separator):** pip-installable wrapper around UVR
  models (MDX, VR, Demucs, RoFormer). Lists models with `--list_models` and downloads
  them automatically.
- **Demucs** (`adefossez/demucs`; the original facebookresearch repo is archived): simple
  CLI and Python API, MIT.
- **UVR5 GUI:** good for manual experiments and A/B listening.

## 2.7 Recommendations

1. **Vocals:** an ensemble of a Mel-RoFormer vocal model and a BS-RoFormer vocal model
   (optionally plus SCNet XL). Fall back to HTDemucs if licensing requires.
2. **Lead/back:** Mel-RoFormer karaoke model on the vocal stem.
3. **Drums, bass:** take them from the 4-stem RoFormer/SCNet run on the instrumental.
   Cross-check bass against HTDemucs, since bass bleed into "other" is common.
4. **Guitar/piano:** `htdemucs_6s`, or dedicated community guitar/piano RoFormer
   checkpoints from MSST/MVSEP. Accept that piano and keys separation is still weak.
5. **Drums → pieces:** DrumSep-style MDX23C model, with LarsNet as a fallback.
6. **Long tail:** run an instrument tagger on the "other" stem. If it finds a confident
   unusual instrument, try SAM Audio with that label.
7. **Keep stems as intermediate artifacts.** They are needed for analysis-by-synthesis
   (file 06) and are a useful product feature in their own right.
