"""Fixed-denominator cross-video CT metrics and registered controls.

This module deliberately separates GT eligibility from prediction-dependent
state births.  It accepts aligned rows in chronological order and a matching
decision sequence; no model or threshold is consulted while constructing the
denominator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def _ival(row: Mapping, key: str, default: int = -1) -> int:
    try: return int(float(row.get(key, default)))
    except (TypeError, ValueError): return default


def _action(row: Mapping) -> str:
    x = str(row.get("sem_action", row.get("action", ""))).lower()
    return {"known_category": "known", "new_novel": "new", "existing_novel": "existing"}.get(x, x)


def _sid(row: Mapping):
    x = row.get("sem_sid", row.get("semantic_id"))
    try: return int(x)
    except (TypeError, ValueError): return None


def fixed_eligibility(aligned: Sequence[Mapping], selected_categories: Iterable[int]) -> list[int]:
    """Return one immutable GT-only denominator in chronological row order."""
    selected = {int(x) for x in selected_categories}
    history: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    eligible = []
    order = sorted(range(len(aligned)), key=lambda i: (_ival(aligned[i], "video_id"),
              _ival(aligned[i], "frame_id"), _ival(aligned[i], "proposal_local_id"), i))
    for i in order:
        row = aligned[i]; role = str(row.get("gt_role", "")); cat = _ival(row, "gt_category_id")
        if role == "novel" and cat in selected:
            vid, tr = _ival(row, "video_id"), _ival(row, "gt_track_id")
            # Another physical track and video must already have appeared.
            if any(pv != vid and (pv, pt) != (vid, tr) for pv, pt, _ in history[cat]):
                eligible.append(i)
            history[cat].append((vid, tr, i))
        elif role == "novel":
            history[cat].append((_ival(row, "video_id"), _ival(row, "gt_track_id"), i))
    return eligible


def _earlier_legal_births(aligned, decisions, selected):
    births: dict[int, tuple[int, int, int, int]] = {}
    # semantic state -> (category, video, physical track, row index), legal only
    order = sorted(range(len(aligned)), key=lambda i: (_ival(aligned[i], "video_id"), _ival(aligned[i], "frame_id"), _ival(aligned[i], "proposal_local_id"), i))
    for i in order:
        a = _action(decisions[i]); k = _sid(decisions[i]); r = aligned[i]
        if a == "new" and k is not None and str(r.get("gt_role", "")) == "novel":
            c = _ival(r, "gt_category_id")
            if c in selected and k not in births:
                births[k] = (c, _ival(r, "video_id"), _ival(r, "track_id", _ival(r, "gt_track_id")), i)
    return births


def fixed_ct_metrics(aligned: Sequence[Mapping], decisions: Sequence[Mapping], selected_categories: Iterable[int], *, eligible_indices: Sequence[int] | None = None) -> dict:
    if len(aligned) != len(decisions): raise ValueError("aligned/decision length mismatch")
    selected = {int(x) for x in selected_categories}
    eligible = list(eligible_indices) if eligible_indices is not None else fixed_eligibility(aligned, selected)
    # Defensive contract: the supplied denominator must itself be GT-only.
    canonical = fixed_eligibility(aligned, selected)
    if list(eligible) != canonical: raise ValueError("eligible_indices differ from canonical GT-only denominator")
    births = _earlier_legal_births(aligned, decisions, selected)
    correct = []
    for i in eligible:
        r, d = aligned[i], decisions[i]; k = _sid(d)
        ok = _action(d) == "existing" and k in births and births[k][0] == _ival(r, "gt_category_id") and births[k][1] != _ival(r, "video_id")
        correct.append(bool(ok))
    by_cat = defaultdict(list); by_vid = defaultdict(list)
    for i, ok in zip(eligible, correct):
        by_cat[str(_ival(aligned[i], "gt_category_id"))].append(int(ok)); by_vid[str(_ival(aligned[i], "video_id"))].append(int(ok))
    predicted_existing = [i for i, d in enumerate(decisions) if _action(d) == "existing" and str(aligned[i].get("gt_role", "")) == "novel"]
    precision = sum(i in set(eligible) and bool(correct[eligible.index(i)]) for i in predicted_existing) / max(len(predicted_existing), 1)
    cat_macro = sum(sum(v) / len(v) for v in by_cat.values()) / max(len(by_cat), 1)
    vid_macro = sum(sum(v) / len(v) for v in by_vid.values()) / max(len(by_vid), 1)
    return {
        "eligible": len(eligible), "correct": int(sum(correct)), "recall": sum(correct) / max(len(eligible), 1),
        "eligible_indices_sha": __import__("hashlib").sha256(json.dumps(eligible).encode()).hexdigest(),
        "category_macro_recall": cat_macro, "video_macro_recall": vid_macro,
        "category_coverage": sum(bool(sum(v)) for v in by_cat.values()) / max(len(by_cat), 1),
        "video_coverage": sum(bool(sum(v)) for v in by_vid.values()) / max(len(by_vid), 1),
        "by_category": {k: {"correct": sum(v), "eligible": len(v), "recall": sum(v) / len(v)} for k, v in sorted(by_cat.items())},
        "by_video": {k: {"correct": sum(v), "eligible": len(v), "recall": sum(v) / len(v)} for k, v in sorted(by_vid.items())},
        "predicted_existing_novel": len(predicted_existing), "predicted_existing_precision": precision,
        "birth_states": len(births), "eligible_indices": eligible,
        "legacy_prediction_conditioned_ct": None
    }


def _oracle(aligned, selected):
    # Oracle creates one state per category at first selected novel occurrence;
    # later rows reuse it. Known rows are labelled known. This control is
    # evaluator-facing and uses labels only by design.
    out = []; states = {}; nxt = 100000
    for r in aligned:
        role, c = str(r.get("gt_role", "")), _ival(r, "gt_category_id")
        if role == "supported_known": out.append({"sem_action": "known", "sem_sid": c})
        elif role == "novel" and c in selected:
            if c not in states: states[c] = nxt; nxt += 1; out.append({"sem_action": "new", "sem_sid": states[c]})
            else: out.append({"sem_action": "existing", "sem_sid": states[c]})
        else: out.append({"sem_action": "new", "sem_sid": nxt}); nxt += 1
    return out


def controls(aligned, selected):
    oracle = _oracle(aligned, selected)
    # Keep each NEW birth legal but deliberately point later EXISTING actions
    # at a state born from another true category.  This tests the semantic
    # correspondence condition rather than merely changing a known class ID.
    cat_birth_sid = {}
    for d, r in zip(oracle, aligned):
        if _action(d) == "new" and str(r.get("gt_role", "")) == "novel":
            cat_birth_sid.setdefault(_ival(r, "gt_category_id"), _sid(d))
    cats = sorted(cat_birth_sid)
    wrong = []
    for d, r in zip(oracle, aligned):
        q = dict(d); c = _ival(r, "gt_category_id")
        if _action(q) == "known":
            q["sem_sid"] = c + 1
        elif _action(q) == "existing" and c in cat_birth_sid and len(cats) > 1:
            other = cats[(cats.index(c) + 1) % len(cats)]
            q["sem_sid"] = cat_birth_sid[other]
        wrong.append(q)
    all_new = [{"sem_action": "new", "sem_sid": 100000 + i} for i in range(len(aligned))]
    one = []; sid = 100000; born = False
    for r in aligned:
        if str(r.get("gt_role", "")) == "novel":
            if not born: one.append({"sem_action": "new", "sem_sid": sid}); born = True
            else: one.append({"sem_action": "existing", "sem_sid": sid})
        else: one.append({"sem_action": "known", "sem_sid": _ival(r, "gt_category_id")})
    return {name: fixed_ct_metrics(aligned, ds, selected) for name, ds in [("correct_label_oracle", oracle), ("wrong_label", wrong), ("all_new", all_new), ("all_one_state", one)]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--aligned", default="data/iclr27_phase15s/sources/proposals_aligned.csv"); ap.add_argument("--decisions", required=True); ap.add_argument("--categories", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); aligned = list(csv.DictReader((ROOT / args.aligned).open())); decisions = list(csv.DictReader((ROOT / args.decisions).open())); selected = {int(x) for x in json.loads((ROOT / args.categories).read_text())}
    value = fixed_ct_metrics(aligned, decisions, selected); Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True); tmp = ROOT / (args.out + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); os.replace(tmp, ROOT / args.out); print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
