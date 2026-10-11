"""Write a Transcription as a Type-1 Standard MIDI File.

* Conductor track: tempo map following the detected beats (so bars line up in
  a DAW even when the performance drifts), time signature, key signature,
  section/harmony markers.
* One track per part, named, with a General MIDI program; drums on channel 10.
* Expressive parts (vocals, bass) carry pitch-bend, with the bend range set via
  RPN 0 on their channel.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mido
import numpy as np

from .types import NoteEvent, Part, Transcription

TICKS_PER_BEAT = 480
DRUM_CHANNEL = 9
MIDO_KEYS_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MIDO_KEYS_MINOR = ["Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"]


class TempoMap:
    """Maps seconds <-> ticks with one tempo segment per beat interval.

    The first downbeat is placed on a bar line, so the MIDI file's bar grid
    matches the music; anything before it becomes a pickup.
    """

    def __init__(self, beats: list[float], downbeats: list[float] | None = None, beats_per_bar: int = 4,
                 fallback_bpm: float = 120.0):
        b = np.asarray(sorted(beats), dtype=float)
        if len(b) < 2:
            spb = 60.0 / fallback_bpm
            b = np.array([0.0, spb])
        self.beats = b
        self.bar_ticks = beats_per_bar * TICKS_PER_BEAT
        first_interval = b[1] - b[0]
        pre_ticks = b[0] / first_interval * TICKS_PER_BEAT
        j = 0
        if downbeats:
            j = int(np.argmin(np.abs(b - downbeats[0])))
        m = int(np.ceil((j * TICKS_PER_BEAT + pre_ticks - 1e-6) / self.bar_ticks))
        self.first_beat_tick = m * self.bar_ticks - j * TICKS_PER_BEAT
        # anchor points: (seconds, ticks)
        secs = [0.0] + b.tolist() if b[0] > 0 else b.tolist()
        ticks = ([0] if b[0] > 0 else []) + [self.first_beat_tick + i * TICKS_PER_BEAT for i in range(len(b))]
        self.secs = np.array(secs)
        self.ticks = np.array(ticks, dtype=float)
        self.last_spb = float(b[-1] - b[-2])

    def to_ticks(self, t: float) -> int:
        if t <= self.secs[-1]:
            return int(round(float(np.interp(t, self.secs, self.ticks))))
        return int(round(self.ticks[-1] + (t - self.secs[-1]) / self.last_spb * TICKS_PER_BEAT))

    def tempo_events(self) -> list[tuple[int, int]]:
        """(tick, microseconds per quarter) for each segment."""
        out = []
        for (s0, s1), (k0, k1) in zip(zip(self.secs[:-1], self.secs[1:]), zip(self.ticks[:-1], self.ticks[1:])):
            if k1 > k0:
                out.append((int(round(k0)), int(round((s1 - s0) / ((k1 - k0) / TICKS_PER_BEAT) * 1e6))))
        out.append((int(round(self.ticks[-1])), int(round(self.last_spb * 1e6))))
        # drop consecutive duplicates
        dedup = []
        for tick, us in out:
            if not dedup or dedup[-1][1] != us:
                dedup.append((tick, max(1, min(us, 0xFFFFFF))))
        return dedup


def _key_signature(key: str | None) -> str | None:
    if not key:
        return None
    try:
        from .analysis.key import Key
        k = Key.parse(key)
    except (ValueError, IndexError):
        return None
    return (MIDO_KEYS_MAJOR if k.mode == "major" else MIDO_KEYS_MINOR)[k.tonic]


def _bend_value(cents: float, range_semitones: int) -> int:
    return int(np.clip(round(cents / (range_semitones * 100.0) * 8192), -8192, 8191))


def _part_events(part: Part, channel: int, tmap: TempoMap, bend_range: int) -> list[tuple[int, int, mido.Message]]:
    """(tick, order, message); order sorts note-offs before note-ons at the same tick."""
    ev: list[tuple[int, int, mido.Message]] = []
    notes = sorted((replace(n) for n in part.notes), key=lambda n: (n.onset, n.pitch))
    # MIDI cannot overlap the same pitch on one channel: end the earlier note first.
    last_end: dict[int, NoteEvent] = {}
    for n in notes:
        prev = last_end.get(n.pitch)
        if prev is not None and prev.offset > n.onset:
            prev.offset = max(prev.onset + 1e-3, n.onset)
        last_end[n.pitch] = n
    has_bend = any(n.bend for n in notes) and not part.is_drum
    for n in notes:
        on, off = tmap.to_ticks(n.onset), tmap.to_ticks(n.offset)
        off = max(off, on + 1)
        vel = int(np.clip(n.velocity, 1, 127))
        if has_bend:
            ev.append((on, 1, mido.Message("pitchwheel", channel=channel, pitch=0)))
            if n.bend:
                for t, cents in n.bend:
                    tick = tmap.to_ticks(t)
                    if on <= tick < off:
                        ev.append((tick, 3, mido.Message("pitchwheel", channel=channel,
                                                         pitch=_bend_value(cents, bend_range))))
        ev.append((on, 2, mido.Message("note_on", channel=channel, note=n.pitch, velocity=vel)))
        ev.append((off, 0, mido.Message("note_off", channel=channel, note=n.pitch, velocity=0)))
    return ev


def _to_track(name: str, timed: list[tuple[int, int, mido.Message]], head: list[mido.Message]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    for m in head:
        track.append(m.copy(time=0))
    now = 0
    for tick, _, msg in sorted(timed, key=lambda x: (x[0], x[1])):
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


def to_midi(tr: Transcription, parts: list[Part] | None = None, bend_range: int = 2,
            markers: list[tuple[float, str]] | None = None) -> mido.MidiFile:
    parts = tr.parts if parts is None else parts
    tmap = TempoMap(tr.beats, tr.downbeats, tr.beats_per_bar, tr.tempo_bpm)
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

    conductor: list[tuple[int, int, mido.Message]] = []
    for tick, us in tmap.tempo_events():
        conductor.append((tick, 0, mido.MetaMessage("set_tempo", tempo=us)))
    head = [mido.MetaMessage("time_signature", numerator=tr.beats_per_bar, denominator=4)]
    ks = _key_signature(tr.key)
    if ks:
        head.append(mido.MetaMessage("key_signature", key=ks))
    for start, _, name in tr.sections:
        conductor.append((tmap.to_ticks(start), 1, mido.MetaMessage("marker", text=name)))
    for t, text in markers or []:
        conductor.append((tmap.to_ticks(t), 1, mido.MetaMessage("text", text=text)))
    mid.tracks.append(_to_track("Conductor", conductor, head))

    channels = iter(c for c in range(16) if c != DRUM_CHANNEL)
    for part in parts:
        if part.is_drum:
            ch = DRUM_CHANNEL
        else:
            ch = next(channels, None)
            if ch is None:  # more than 15 pitched parts: share the last channel
                ch = 15
        head = []
        if not part.is_drum:
            head.append(mido.Message("program_change", channel=ch, program=int(part.program) % 128))
            if any(n.bend for n in part.notes):
                head += [
                    mido.Message("control_change", channel=ch, control=101, value=0),
                    mido.Message("control_change", channel=ch, control=100, value=0),
                    mido.Message("control_change", channel=ch, control=6, value=bend_range),
                    mido.Message("control_change", channel=ch, control=38, value=0),
                    mido.Message("control_change", channel=ch, control=101, value=127),
                    mido.Message("control_change", channel=ch, control=100, value=127),
                ]
        mid.tracks.append(_to_track(part.name, _part_events(part, ch, tmap, bend_range), head))
    return mid


def write_midi(tr: Transcription, path: str | Path, parts: list[Part] | None = None, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_midi(tr, parts, **kw).save(str(path))
    return path


def read_notes(path: str | Path) -> dict[str, list[NoteEvent]]:
    """Read back a MIDI file as {track name: notes in seconds} (used by tests/eval)."""
    mid = mido.MidiFile(str(path))
    tempo_changes: list[tuple[int, int]] = []
    for msg_track in mid.tracks:
        tick = 0
        for msg in msg_track:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((tick, msg.tempo))
    tempo_changes.sort()
    if not tempo_changes or tempo_changes[0][0] != 0:
        tempo_changes.insert(0, (0, 500000))

    def tick_to_sec(t: int) -> float:
        sec, last_tick, tempo = 0.0, 0, tempo_changes[0][1]
        for ct, ctempo in tempo_changes[1:]:
            if ct > t:
                break
            sec += (ct - last_tick) * tempo / 1e6 / mid.ticks_per_beat
            last_tick, tempo = ct, ctempo
        return sec + (t - last_tick) * tempo / 1e6 / mid.ticks_per_beat

    out: dict[str, list[NoteEvent]] = {}
    for track in mid.tracks:
        name, tick, active, notes = "", 0, {}, []
        for msg in track:
            tick += msg.time
            if msg.type == "track_name":
                name = msg.name
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (tick, msg.velocity)
            elif msg.type in ("note_off", "note_on"):
                start = active.pop((msg.channel, msg.note), None)
                if start:
                    notes.append(NoteEvent(msg.note, tick_to_sec(start[0]), tick_to_sec(tick), start[1]))
        if notes:
            out[name] = notes
    return out
