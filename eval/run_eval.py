"""Run the pipeline on synthetic songs and print per-part scores.

    python eval/run_eval.py                 # oracle stems + full mix, preset "fast"
    python eval/run_eval.py --preset balanced --songs 3

"oracle" feeds the ground-truth stems (measures transcription alone);
"mix" runs separation too (measures the whole system).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import evaluate
from synthetic import make_song

from mp3_to_midi.pipeline import transcribe

KEYS = ["C major", "G major", "A minor", "Eb major", "E minor"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--songs", type=int, default=2)
    ap.add_argument("--modes", default="oracle,mix")
    ap.add_argument("--json", help="write results to this file")
    args = ap.parse_args()

    results = []
    for i in range(args.songs):
        key = KEYS[i % len(KEYS)]
        mix, gt, stems = make_song(seed=i, key_name=key, bpm=90 + 10 * i)
        for mode in args.modes.split(","):
            est = transcribe(mix, preset=args.preset, stems=stems if mode == "oracle" else None,
                             progress=lambda m: None)
            scores = evaluate(gt, est)
            row = {"song": i, "key": key, "mode": mode, "est_key": est.key, "tempo": est.report["tempo_bpm"],
                   "scores": scores, "warnings": est.report["warnings"]}
            results.append(row)
            print(_summary(row))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


def _summary(row: dict) -> str:
    s = row["scores"]
    parts = [f"song {row['song']} [{row['mode']:<6}] key {row['key']} -> {row['est_key']}, tempo {row['tempo']}"]
    for k in ("lead_vocal", "bass", "other"):
        if k in s:
            parts.append(f"  {k:<11} onset F1 {s[k]['onset']['f']:.2f}   onset+offset F1 {s[k]['onset_offset']['f']:.2f}")
    if "drums" in s:
        parts.append("  drums       " + "  ".join(f"{k} {v['f']:.2f}" for k, v in s["drums"].items()))
    if "harmony" in s:
        h = s["harmony"]
        parts.append(f"  harmony     voices {h['voices_est']}/{h['voices_ref']}, all-notes onset F1 "
                     f"{h['all_notes_onset']['f']:.2f}")
        for m in h["matched"]:
            parts.append(f"      {m['ref']:<20} <- {m['est']:<40} F1 {m['f']:.2f}")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
