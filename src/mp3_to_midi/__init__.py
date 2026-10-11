"""mp3-to-midi: transcribe full songs into multi-track MIDI."""

from .types import Audio, NoteEvent, Part, Transcription

__version__ = "0.1.0"
__all__ = ["Audio", "NoteEvent", "Part", "Transcription", "transcribe"]


def transcribe(*args, **kwargs):
    """Run the full pipeline. See `mp3_to_midi.pipeline.transcribe`."""
    from .pipeline import transcribe as _transcribe

    return _transcribe(*args, **kwargs)
