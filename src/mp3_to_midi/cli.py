"""Command line interface: `mp3-to-midi song.mp3 -o out/`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import registry
from .pipeline import PRESETS, transcribe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mp3-to-midi",
        description="Transcribe a song (vocals, harmonies, melody, bass, drums, other) into multi-track MIDI.",
    )
    ap.add_argument("input", nargs="?", help="audio file (mp3, wav, flac, m4a, ...)")
    ap.add_argument("-o", "--out", help="output directory (default: <input name>_midi/)")
    ap.add_argument("-p", "--preset", default="balanced", choices=sorted(PRESETS),
                    help="quality preset; each stage falls back to what is installed (default: balanced)")
    ap.add_argument("--commercial-safe", action="store_true",
                    help="refuse components whose weights are not licensed for commercial use")
    ap.add_argument("--stems", action="store_true", help="also save the separated stems as WAV")
    ap.add_argument("--harmony-voices", type=int, default=3, help="maximum number of harmony voices (default 3)")
    for stage in ("separation", "karaoke", "f0", "poly", "beats"):
        ap.add_argument(f"--{stage}", help=f"force the {stage} component (see --list-components)")
    ap.add_argument("--list-components", action="store_true", help="show components, licenses and availability")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list_components:
        rows = registry.status()
        w = max(len(r["name"]) for r in rows)
        for r in rows:
            mark = "yes" if r["available"] else " - "
            lic = r["license"] + ("" if r["commercial_ok"] else " (non-commercial/varies)")
            print(f"{r['stage']:<11} {r['name']:<{w}}  installed: {mark}  {lic:<34} {r['description']}")
        return 0
    if not args.input:
        ap.error("input file is required")

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    src = Path(args.input)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_midi")
    overrides = {s: getattr(args, s) for s in ("separation", "karaoke", "f0", "poly", "beats") if getattr(args, s)}

    tr = transcribe(src, out, preset=args.preset, commercial_safe=args.commercial_safe, overrides=overrides,
                    save_stems=args.stems, max_harmony_voices=args.harmony_voices,
                    progress=None if args.quiet else (lambda m: print(m, file=sys.stderr)))

    r = tr.report
    print(f"\nWrote {out / 'song.mid'}  ({len(tr.parts)} parts)")
    print(f"key {r['key']['name']}, {r['tempo_bpm']} BPM, {r['beats_per_bar']}/4")
    for p in r["parts"]:
        print(f"  {p['name']:<36} {p['notes']:>5} notes")
    if r["harmony"].get("voices"):
        print(f"  lead doubled {r['harmony']['lead_doubled_ratio']:.0%} of the time")
    for w in r["warnings"]:
        print(f"  ! {w}")
    print(f"components: {json.dumps(r['components'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
