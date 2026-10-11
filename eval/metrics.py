"""Evaluation metrics (mir_eval wrappers + harmony-voice matching)."""

from __future__ import annotations

import numpy as np

from mp3_to_midi.types import NoteEvent, Part, Transcription


def _arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes]).reshape(-1, 2)
    hz = 440.0 * 2 ** ((np.array([n.pitch for n in notes], dtype=float) - 69) / 12)
    return iv, hz


def note_f1(ref: list[NoteEvent], est: list[NoteEvent], with_offset: bool = False) -> dict:
    import mir_eval

    if not ref and not est:
        return {"p": 1.0, "r": 1.0, "f": 1.0}
    if not ref or not est:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, offset_ratio=0.2 if with_offset else None)
    return {"p": round(p, 3), "r": round(r, 3), "f": round(f, 3)}


def drum_f1(ref: list[NoteEvent], est: list[NoteEvent], window: float = 0.05) -> dict:
    import mir_eval

    out = {}
    for pitch, name in ((36, "kick"), (38, "snare"), (42, "hihat")):
        r = np.array([n.onset for n in ref if n.pitch == pitch])
        e = np.array([n.onset for n in est if n.pitch == pitch])
        if len(r) == 0 and len(e) == 0:
            continue
        f, p, rc = mir_eval.onset.f_measure(r, e, window) if len(r) and len(e) else (0.0, 0.0, 0.0)
        out[name] = {"p": round(p, 3), "r": round(rc, 3), "f": round(f, 3)}
    return out


def evaluate(gt: Transcription, est: Transcription) -> dict:
    res: dict = {}
    pairs = [("lead_vocal", "lead_vocal"), ("bass", "bass"), ("other", "other")]
    for gt_role, est_role in pairs:
        g = [n for p in gt.parts_by_role(gt_role) for n in p.notes]
        e = [n for p in est.parts_by_role(est_role) for n in p.notes]
        if g:
            res[gt_role] = {"onset": note_f1(g, e), "onset_offset": note_f1(g, e, True)}
    g_dr = [n for p in gt.parts if p.is_drum for n in p.notes]
    e_dr = [n for p in est.parts if p.is_drum for n in p.notes]
    if g_dr:
        res["drums"] = drum_f1(g_dr, e_dr)

    # Harmony: optimal one-to-one matching of estimated voices to reference voices.
    g_h = gt.parts_by_role("harmony")
    e_h = [p for p in est.parts if p.role in ("harmony", "backing_vocal")]
    if g_h:
        res["harmony"] = _match_voices(g_h, e_h)
        all_g = [n for p in g_h for n in p.notes]
        all_e = [n for p in e_h for n in p.notes]
        res["harmony"]["all_notes_onset"] = note_f1(all_g, all_e)
    return res


def _match_voices(ref: list[Part], est: list[Part]) -> dict:
    from scipy.optimize import linear_sum_assignment

    if not est:
        return {"voices_ref": len(ref), "voices_est": 0, "matched": []}
    F = np.array([[note_f1(r.notes, e.notes)["f"] for e in est] for r in ref])
    rows, cols = linear_sum_assignment(-F)
    matched = [{"ref": ref[i].name, "est": est[j].name, "f": float(F[i, j]),
                "label": est[j].meta.get("label")} for i, j in zip(rows, cols)]
    return {"voices_ref": len(ref), "voices_est": len(est), "matched": matched}
