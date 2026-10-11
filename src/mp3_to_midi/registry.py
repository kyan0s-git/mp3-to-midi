"""Component registry: what each stage can use, its license, and whether it is installed.

Every model the pipeline can load is listed here with its license so that a
"commercial-safe" run can refuse non-commercial weights (see research/08).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    stage: str  # separation | karaoke | f0 | poly | drums | beats
    description: str
    license: str
    commercial_ok: bool
    requires: tuple[str, ...] = ()  # importable modules
    quality: int = 0  # rough rank used for "best available"

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.requires)


COMPONENTS: dict[str, Component] = {
    c.name: c
    for c in [
        # --- separation ---------------------------------------------------
        Component("dsp", "separation", "HPSS + band/stereo heuristics (no model, low quality)", "MIT", True, (), 0),
        Component("demucs", "separation", "HTDemucs v4 4-stem (Meta)", "MIT", True, ("demucs", "torch"), 2),
        Component("demucs6", "separation", "HTDemucs v4 6-stem incl. guitar/piano", "MIT", True, ("demucs", "torch"), 2),
        Component("roformer", "separation", "BS/Mel-RoFormer via python-audio-separator", "model-dependent", False, ("audio_separator",), 3),
        # --- lead/backing vocal split --------------------------------------
        Component("midside", "karaoke", "lead = centre of vocal stem, backing = sides (heuristic)", "MIT", True, (), 0),
        Component("roformer-karaoke", "karaoke", "Mel-RoFormer karaoke model via python-audio-separator", "model-dependent", False, ("audio_separator",), 3),
        # --- monophonic f0 -------------------------------------------------
        Component("pyin", "f0", "probabilistic YIN (librosa)", "ISC", True, ("librosa",), 1),
        Component("crepe", "f0", "CREPE via torchcrepe", "MIT", True, ("torchcrepe", "torch"), 2),
        # --- polyphonic note transcription --------------------------------
        Component("salience", "poly", "harmonic-sum multi-f0 with iterative cancellation (no model)", "MIT", True, ("librosa",), 0),
        Component("basic-pitch", "poly", "Spotify Basic Pitch (ONNX/TF)", "Apache-2.0", True, ("basic_pitch",), 2),
        # --- drums ---------------------------------------------------------
        Component("drums-dsp", "drums", "band-split onset detection (kick/snare/hi-hat)", "MIT", True, ("librosa",), 0),
        # --- beats ---------------------------------------------------------
        Component("librosa-beats", "beats", "dynamic-programming beat tracker (librosa)", "ISC", True, ("librosa",), 1),
        Component("beat-this", "beats", "Beat This! (ISMIR 2024)", "MIT", True, ("beat_this", "torch"), 3),
    ]
}


def choose(stage: str, preferences: list[str], commercial_safe: bool = False) -> Component:
    """Return the first available component from `preferences` for `stage`.

    Falls back to the best available component of that stage so the pipeline
    always runs; the caller records which one was used.
    """
    def ok(c: Component) -> bool:
        return c.stage == stage and c.available() and (c.commercial_ok or not commercial_safe)

    for name in preferences:
        c = COMPONENTS.get(name)
        if c is not None and ok(c):
            return c
    candidates = sorted((c for c in COMPONENTS.values() if ok(c)), key=lambda c: -c.quality)
    if not candidates:
        raise RuntimeError(f"no available component for stage {stage!r}")
    return candidates[0]


def status() -> list[dict]:
    return [
        {
            "name": c.name,
            "stage": c.stage,
            "available": c.available(),
            "license": c.license,
            "commercial_ok": c.commercial_ok,
            "description": c.description,
        }
        for c in COMPONENTS.values()
    ]
