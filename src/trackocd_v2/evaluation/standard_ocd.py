"""Standard anonymous-state OCD metrics with one global Hungarian mapping."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _state(row: Mapping) -> str:
    decision = row.get("decision", row)
    kind = str(decision.get("kind", "DEFER"))
    token = decision.get("token")
    if kind == "DEFER" or token is None:
        return "DEFER"
    return str(token)


def _h_score(old: float, new: float) -> float:
    return 0.0 if old + new <= 0 else 2.0 * old * new / (old + new)


def evaluate_standard(
    rows: Sequence[Mapping],
    decisions: Sequence[Mapping] | None = None,
    *,
    known_ids: Iterable[int],
    novel_ids: Iterable[int],
    distractor_ids: Iterable[int] = (),
) -> dict:
    """Evaluate Old/New/All using a single state-to-category matching.

    ``rows`` are evaluator-side rows.  Methods receive only the corresponding
    ``TrackSample.model_view`` and therefore cannot access the GT fields used
    here.
    """

    known, novel, distractor = set(map(int, known_ids)), set(map(int, novel_ids)), set(map(int, distractor_ids))
    if decisions is None:
        decisions = rows
    if len(rows) != len(decisions):
        raise ValueError("rows and decisions must have identical length")
    usable = []
    for row, decision in zip(rows, decisions):
        category = int(row["gt_category_id"])
        if category in distractor:
            continue
        if category not in known and category not in novel:
            raise ValueError(f"category outside locked roles: {category}")
        usable.append((row, decision, category, "old" if category in known else "new", _state(decision)))
    states = sorted({state for _, _, _, _, state in usable if state != "DEFER"})
    categories = sorted({category for _, _, category, _, state in usable if state != "DEFER"})
    contingency = np.zeros((len(states), len(categories)), dtype=np.int64)
    s_index, c_index = {s: i for i, s in enumerate(states)}, {c: i for i, c in enumerate(categories)}
    for _, _, category, _, state in usable:
        if state != "DEFER":
            contingency[s_index[state], c_index[category]] += 1
    mapping: dict[str, int] = {}
    if contingency.size:
        rr, cc = linear_sum_assignment(-contingency)
        mapping = {states[int(r)]: int(categories[int(c)]) for r, c in zip(rr, cc) if contingency[r, c] > 0}
    correct = Counter()
    totals = Counter()
    mapped_rows = []
    for row, decision, category, split, state in usable:
        predicted = mapping.get(state)
        hit = predicted == category
        correct[split] += int(hit)
        totals[split] += 1
        mapped_rows.append({"sample_key": row.get("sample_key"), "gt_category_id": category, "gt_split": split, "state": state, "mapped_category_id": predicted, "correct": hit})
    old = correct["old"] / totals["old"] if totals["old"] else 0.0
    new = correct["new"] / totals["new"] if totals["new"] else 0.0
    all_acc = (correct["old"] + correct["new"]) / max(totals["old"] + totals["new"], 1)
    return {
        "old_acc": old,
        "new_acc": new,
        "h_score": _h_score(old, new),
        "all_acc": all_acc,
        "old_correct": correct["old"],
        "old_total": totals["old"],
        "new_correct": correct["new"],
        "new_total": totals["new"],
        "state_count": len(states),
        "hungarian_mapping": mapping,
        "row_diagnostics": mapped_rows,
    }
