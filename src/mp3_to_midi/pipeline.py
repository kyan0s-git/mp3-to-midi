"""End-to-end pipeline: audio file -> stems -> per-part transcription -> MIDI.

Stages (see research/06-proposed-architecture.md):
  1. decode
  2. separate (vocals / drums / bass / other [/ guitar / piano])
  3. global analysis: beats, downbeats, key
  4. vocals: lead/backing split -> lead notes (f0 + segmentation, with pitch
     bends) -> backing notes (polyphonic) -> harmony voices + labels
  5. bass, drums, remaining instruments
  6. fusion: cross-stem bleed removal, melody track
  7. export: song.mid, parts/*.mid, report.json, optional stems
Every stage picks the best *available* component for the preset and records
what it used (and any fallback) in the report.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import registry
from .analysis import Key, detect_onsets, estimate_key, track_beats
from .audio import load, save
from .fusion import fit_range, merge_duplicates, remove_cross_stem_bleed
from .harmony import (
    analyze_voice,
    assign_voices_to_lead,
    lead_doubling_ratio,
    section_summary,
    split_lead_doubles,
    sync_to_lead,
)
from .midi_export import write_midi
from .pitch import f0_to_notes, track
from .pitch.melody import hybrid as hybrid_f0
from .pitch.melody import predominant_f0
from .separation import separate, split_lead_backing
from .transcription import poly_notes, transcribe_drums
from .types import Audio, NoteEvent, Part, Transcription

log = logging.getLogger("mp3_to_midi")

PRESETS: dict[str, dict[str, list[str]]] = {
    "fast": {
        "separation": ["dsp"], "karaoke": ["midside"], "f0": ["pyin"], "poly": ["salience"],
        "drums": ["drums-dsp"], "beats": ["librosa-beats"],
    },
    "balanced": {
        "separation": ["demucs", "dsp"], "karaoke": ["midside"], "f0": ["pyin"],
        "poly": ["basic-pitch", "salience"], "drums": ["drums-dsp"], "beats": ["beat-this", "librosa-beats"],
    },
    "max": {
        "separation": ["roformer", "demucs6", "demucs", "dsp"], "karaoke": ["roformer-karaoke", "midside"],
        "f0": ["crepe", "pyin"], "poly": ["basic-pitch", "salience"], "drums": ["drums-dsp"],
        "beats": ["beat-this", "librosa-beats"],
    },
}

# General MIDI programs (0-based) used for playback of each role.
PROGRAMS = {"lead_vocal": 53, "harmony": 52, "backing_vocal": 52, "melody": 73, "bass": 33,
            "piano": 0, "guitar": 27, "other": 0}
STEM_ROLES = {"piano": ("Piano", "piano"), "guitar": ("Guitar", "guitar"), "other": ("Other", "other")}
SILENT_DB = -40.0  # stems quieter than this (relative to the mix) are treated as absent
# Polyphonic back-end for backing vocals. On the synthetic harmony benchmark
# (eval/) the salience tracker separates parallel harmony lines better than
# Basic Pitch (mean matched-voice F1 0.82 vs 0.75); revisit on real recordings.
HARMONY_POLY = "salience"


def _db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-9))


class _Run:
    def __init__(self, preset: str, commercial_safe: bool, overrides: dict[str, str] | None,
                 progress: Callable[[str], None]):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        self.prefs = {k: list(v) for k, v in PRESETS[preset].items()}
        for stage, name in (overrides or {}).items():
            self.prefs[stage] = [name, *self.prefs.get(stage, [])]
        self.commercial_safe = commercial_safe
        self.progress = progress
        self.used: dict[str, str] = {}
        self.warnings: list[str] = []
        self.timings: dict[str, float] = {}

    def component(self, stage: str) -> str:
        c = registry.choose(stage, self.prefs[stage], self.commercial_safe)
        wanted = self.prefs[stage][0]
        if c.name != wanted:
            self.warnings.append(f"{stage}: '{wanted}' not available, using '{c.name}'")
        self.used[stage] = c.name
        return c.name

    def stage(self, name: str):
        run = self

        class _Timer:
            def __enter__(self):
                run.progress(f"[{name}]")
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                run.timings[name] = round(time.perf_counter() - self.t0, 2)
                return False

        return _Timer()

    def with_fallback(self, stage: str, fn: Callable[[str], object], fallback: str):
        """Run `fn(component)`; on failure (missing weights, no network...) use `fallback`."""
        name = self.component(stage)
        try:
            return fn(name)
        except Exception as exc:
            if name == fallback:
                raise
            self.warnings.append(f"{stage}: '{name}' failed ({exc.__class__.__name__}: {exc}); using '{fallback}'")
            self.used[stage] = fallback
            return fn(fallback)


def transcribe(
    source: str | Path | Audio,
    out_dir: str | Path | None = None,
    preset: str = "balanced",
    commercial_safe: bool = False,
    overrides: dict[str, str] | None = None,
    stems: dict[str, Audio] | None = None,
    save_stems: bool = False,
    max_harmony_voices: int = 3,
    progress: Callable[[str], None] | None = None,
) -> Transcription:
    """Transcribe a song.

    `stems` lets callers supply pre-separated stems (e.g. from another tool, or
    ground truth for evaluation); separation is then skipped.
    """
    run = _Run(preset, commercial_safe, overrides, progress or (lambda m: log.info(m)))
    work = Path(out_dir) if out_dir else None

    with run.stage("decode"):
        audio = source if isinstance(source, Audio) else load(source)
    mix_db = _db(audio.rms())

    with run.stage("separation"):
        if stems is None:
            stems = run.with_fallback("separation", lambda m: separate(audio, m, work / "_cache" if work else None),
                                      "dsp")
        else:
            run.used["separation"] = "provided"
    present = {k: v for k, v in stems.items() if _db(v.rms()) - mix_db > SILENT_DB}
    if run.used.get("separation") == "dsp":
        run.warnings.append("DSP separation is a rough fallback; expect bleed between parts "
                            "(install demucs or audio-separator for real stems)")

    with run.stage("analysis"):
        rhythm = _sum_stems([stems[k] for k in ("drums", "bass") if k in present])
        beat_method = run.component("beats")
        grid = track_beats(audio, beat_method, rhythm)
        run.warnings.extend(grid.notes)
        key = estimate_key(_sum_stems([v for k, v in stems.items() if k != "drums"]) or audio)

    tr = Transcription(duration=audio.duration, beats=grid.beats.tolist(), downbeats=grid.downbeats.tolist(),
                       beats_per_bar=grid.beats_per_bar, tempo_bpm=grid.tempo, key=key.name)
    harmony_report: dict = {}

    if "vocals" in present:
        with run.stage("vocals"):
            parts, harmony_report = _vocals(run, stems["vocals"], key, grid.downbeats.tolist(),
                                            grid.beats_per_bar, max_harmony_voices, work)
            tr.parts.extend(parts)

    if "bass" in present:
        with run.stage("bass"):
            tr.parts.append(Part("Bass", "bass", _bass(run, stems["bass"]), PROGRAMS["bass"]))

    if "drums" in present:
        with run.stage("drums"):
            run.component("drums")
            tr.parts.append(Part("Drums", "drums", transcribe_drums(stems["drums"]), is_drum=True))

    poly_method = run.component("poly")
    for stem_name, (part_name, role) in STEM_ROLES.items():
        if stem_name in present:
            with run.stage(stem_name):
                notes = poly_notes(stems[stem_name], poly_method, fmin_midi=28, fmax_midi=100, max_voices=6)
                tr.parts.append(Part(part_name, role, merge_duplicates(notes), PROGRAMS[role]))

    with run.stage("fusion"):
        instruments = [p for p in tr.parts if p.role in ("bass", "piano", "guitar", "other")]
        removed = remove_cross_stem_bleed(instruments)
        melody = _melody(tr)
        if melody.notes:
            tr.parts.insert(0, melody)

    tr.report = {
        "input": str(source) if not isinstance(source, Audio) else "<audio>",
        "preset": preset,
        "commercial_safe": commercial_safe,
        "components": run.used,
        "warnings": run.warnings,
        "timings_s": run.timings,
        "duration_s": round(audio.duration, 3),
        "tempo_bpm": round(grid.tempo, 2),
        "beats_per_bar": grid.beats_per_bar,
        "key": {"name": key.name, "confidence": round(key.confidence, 3)},
        "stems_present": sorted(present),
        "bleed_notes_removed": removed,
        "harmony": harmony_report,
        "parts": [
            {"name": p.name, "role": p.role, "notes": len(p.notes),
             "mean_confidence": round(float(np.mean([n.confidence for n in p.notes])), 3) if p.notes else None,
             **({"meta": p.meta} if p.meta else {})}
            for p in tr.parts
        ],
    }

    if work:
        _write_outputs(tr, work, stems if save_stems else None)
    return tr


def _sum_stems(items: list[Audio]) -> Audio | None:
    if not items:
        return None
    sr = items[0].sr
    n = max(a.samples.shape[1] for a in items)
    ch = max(a.channels for a in items)
    acc = np.zeros((ch, n), dtype=np.float32)
    for a in items:
        acc[:, : a.samples.shape[1]] += a.samples if a.channels == ch else np.repeat(a.samples, ch, axis=0)
    return Audio(acc, sr)


def _vocals(run: _Run, vocals: Audio, key: Key, downbeats: list[float], beats_per_bar: int,
            max_voices: int, work: Path | None) -> tuple[list[Part], dict]:
    lb = run.with_fallback("karaoke", lambda m: split_lead_backing(vocals, m, work / "_cache" if work else None),
                           "midside")
    f0_method = run.component("f0")
    try:
        curve = track(lb["lead"], f0_method, "voice")
    except Exception as exc:  # noqa: BLE001
        run.warnings.append(f"f0: '{f0_method}' failed ({exc}); using 'pyin'")
        run.used["f0"] = "pyin"
        curve = track(lb["lead"], "pyin", "voice")
    # Follow the most salient continuous voice so harmonies cannot steal the lead.
    curve = hybrid_f0(curve, predominant_f0(lb["lead"], "voice"))
    lead = f0_to_notes(curve, key_pitch_classes=set(key.scale), source=run.used["f0"])
    parts = [Part("Lead Vocal", "lead_vocal", lead, PROGRAMS["lead_vocal"])]

    # Backing vocals, analysed in the pitch domain. With a real karaoke model the
    # backing stem is clean; with the mid/side fallback, analyse the whole vocal
    # stem and remove the lead's notes (centre-panned harmonies stay findable).
    src = lb["backing"] if run.used["karaoke"] != "midside" else vocals
    backing: list[NoteEvent] = []
    if _db(src.rms()) - _db(vocals.rms()) > -30:
        backing = merge_duplicates(poly_notes(src, HARMONY_POLY, fmin_midi=40, fmax_midi=86,
                                              max_voices=max_voices + 1))
    harmony_notes, doubles = split_lead_doubles(backing, lead)
    voices = []
    for v in assign_voices_to_lead(harmony_notes, lead, key, max_voices=max_voices):
        v = sync_to_lead(v, lead)
        if len(v) < 3 or sum(n.duration for n in v) < 1.0:
            continue
        if analyze_voice(v, lead, key).steps == 0:  # sits on the lead's notes: bleed or a double
            doubles.extend(v)
            continue
        voices.append(v)

    report = {"lead_doubled_ratio": round(lead_doubling_ratio(lead, doubles), 3), "poly": HARMONY_POLY,
              "voices": []}
    for i, v in enumerate(voices, start=1):
        a = analyze_voice(v, lead, key)
        name = f"Harmony {i} ({a.label})" if a.role == "harmony" else f"Backing {i} ({a.label})"
        meta = {"label": a.label, "consistency": round(a.consistency, 3), "lead_overlap": round(a.lead_overlap, 3),
                "intervals": a.histogram}
        parts.append(Part(name, a.role, v, PROGRAMS[a.role], meta=meta))
        report["voices"].append({"part": name, **meta})
    if downbeats and voices:
        starts = downbeats[::8] or [0.0]  # summarise per 8 bars
        sections = [(s, starts[k + 1] if k + 1 < len(starts) else float("inf"), f"bars {k * 8 + 1}-{k * 8 + 8}")
                    for k, s in enumerate(starts)]
        report["by_section"] = [s for s in section_summary(lead, voices, key, sections) if s["harmonies"]]
    return parts, report


def _bass(run: _Run, bass: Audio) -> list[NoteEvent]:
    # Bass is (nearly) monophonic: f0 tracking + onset-based re-pluck splits.
    # On the synthetic benchmark this beats Basic Pitch and a consensus of the
    # two (onset+offset F1 0.87 vs 0.82), so no polyphonic cross-check here.
    run.used["bass"] = "pyin+onsets"
    curve = track(bass, "pyin", "bass")
    notes = f0_to_notes(curve, min_note=0.07, energy_split_db=5.0, onsets=detect_onsets(bass), source="pyin")
    return fit_range(notes, 23, 67)


def _melody(tr: Transcription, min_gap: float = 2.0) -> Part:
    """Lead vocal where someone sings; elsewhere the top line of the pitched instruments."""
    lead = next((p for p in tr.parts if p.role == "lead_vocal"), None)
    notes = [NoteEvent(n.pitch, n.onset, n.offset, n.velocity, n.confidence, n.bend, list(n.sources))
             for n in (lead.notes if lead else [])]
    others = [n for p in tr.parts if p.role in ("other", "guitar", "piano") for n in p.notes]
    if others:
        sung = sorted((n.onset, n.offset) for n in notes)
        gaps, cursor = [], 0.0
        for on, off in sung + [(tr.duration, tr.duration)]:
            if on - cursor >= min_gap:
                gaps.append((cursor, on))
            cursor = max(cursor, off)
        for g0, g1 in gaps:
            inside = sorted((n for n in others if g0 <= n.onset < g1), key=lambda n: (n.onset, -n.pitch))
            last_end = -1.0
            for n in inside:  # skyline: highest note starting at each onset, no overlaps
                if n.onset >= last_end - 0.02:
                    notes.append(NoteEvent(n.pitch, n.onset, min(n.offset, g1), n.velocity, n.confidence * 0.7,
                                           None, [*n.sources, "skyline"]))
                    last_end = n.offset
    notes.sort(key=lambda n: n.onset)
    return Part("Melody", "melody", notes, PROGRAMS["melody"])


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def _write_outputs(tr: Transcription, out: Path, stems: dict[str, Audio] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_midi(tr, out / "song.mid")
    for p in tr.parts:
        write_midi(tr, out / "parts" / f"{_slug(p.name)}.mid", parts=[p])
    if stems:
        for name, a in stems.items():
            save(a, out / "stems" / f"{name}.wav")
    (out / "report.json").write_text(json.dumps(tr.report, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)
