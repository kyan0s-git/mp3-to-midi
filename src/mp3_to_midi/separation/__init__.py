"""Source separation: mix -> stems, and vocal stem -> lead / backing.

Stem names used throughout the pipeline: vocals, drums, bass, other, and
optionally guitar, piano. Lead/backing split returns: lead, backing.
"""

from __future__ import annotations

from pathlib import Path

from ..types import Audio
from . import dsp

# python-audio-separator model files. Check `audio-separator --list_models` for
# the current catalogue and each model's license before relying on these names.
ROFORMER_VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
ROFORMER_KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"


def separate(audio: Audio, method: str, work_dir: Path | None = None) -> dict[str, Audio]:
    if method == "demucs":
        from .neural import demucs_separate
        return demucs_separate(audio, "htdemucs")
    if method == "demucs6":
        from .neural import demucs_separate
        return demucs_separate(audio, "htdemucs_6s")
    if method == "roformer":
        from .neural import roformer_separate
        return roformer_separate(audio, ROFORMER_VOCAL_MODEL, work_dir)
    return dsp.separate(audio)


def split_lead_backing(vocals: Audio, method: str, work_dir: Path | None = None) -> dict[str, Audio]:
    if method == "roformer-karaoke":
        from .neural import audio_separator_run
        stems = audio_separator_run(vocals, ROFORMER_KARAOKE_MODEL, work_dir)
        # Karaoke models call the lead "vocals" and everything else "instrumental".
        lead = stems.get("vocals")
        backing = stems.get("instrumental") or stems.get("other")
        if lead is not None and backing is not None:
            return {"lead": lead, "backing": backing}
        raise RuntimeError(f"unexpected karaoke model outputs: {sorted(stems)}")
    return dsp.split_mid_side(vocals)


__all__ = ["separate", "split_lead_backing"]
