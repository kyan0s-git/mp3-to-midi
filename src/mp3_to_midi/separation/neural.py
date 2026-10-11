"""Adapters for neural separators. Imported lazily; each needs optional deps.

* Demucs (MIT): `pip install demucs` - downloads weights on first use.
* python-audio-separator: RoFormer / MDX / karaoke models
  (`pip install audio-separator`) - model licenses vary, check each one.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np

from ..audio import load, resample, save
from ..types import Audio


def demucs_separate(audio: Audio, model_name: str = "htdemucs", shifts: int = 1) -> dict[str, Audio]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(model_name)
    model.eval()
    a = resample(audio, model.samplerate)
    x = a.samples if a.channels == 2 else np.repeat(a.samples[:1], 2, axis=0)
    wav = torch.from_numpy(np.ascontiguousarray(x))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=shifts,
                              split=True, overlap=0.25, progress=False)[0]
    sources = sources * std + mean
    return {name: Audio(src.cpu().numpy(), model.samplerate) for name, src in zip(model.sources, sources)}


def audio_separator_run(audio: Audio, model_filename: str, work_dir: Path | None = None) -> dict[str, Audio]:
    """Run one python-audio-separator model; returns {stem name (lower case): Audio}."""
    from audio_separator.separator import Separator

    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory()
        work_dir = Path(tmp.name)
    work_dir = Path(work_dir) / Path(model_filename).stem
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = save(audio, work_dir / "input.wav")
        sep = Separator(output_dir=str(work_dir), output_format="WAV")
        sep.load_model(model_filename=model_filename)
        stems: dict[str, Audio] = {}
        for f in sep.separate(str(src)):
            path = Path(f) if Path(f).is_absolute() else work_dir / f
            m = re.search(r"_\(([^)]+)\)", path.name)
            if m:
                stems[m.group(1).lower()] = load(path, sr=audio.sr)
        return stems
    finally:
        if tmp is not None:
            tmp.cleanup()


def roformer_separate(audio: Audio, vocal_model: str, work_dir: Path | None = None) -> dict[str, Audio]:
    """RoFormer for vocals vs. instrumental, then Demucs (or DSP) on the instrumental."""
    stems = audio_separator_run(audio, vocal_model, work_dir)
    vocals = stems.get("vocals")
    instrumental = stems.get("instrumental")
    if vocals is None or instrumental is None:
        raise RuntimeError(f"unexpected vocal model outputs: {sorted(stems)}")
    try:
        inst = demucs_separate(instrumental, "htdemucs_6s")
    except ImportError:
        from .dsp import separate as dsp_separate
        inst = dsp_separate(instrumental)
    inst.pop("vocals", None)  # residual vocal bleed; the RoFormer stem is better
    return {"vocals": vocals, **inst}
