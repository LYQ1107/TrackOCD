"""Auxiliary stream-level open-world discovery metrics for Phase88.

These metrics are reporting-only.  They never participate in H2 checkpoint
selection, which remains the preregistered Phase19R TRAIN selection score.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _global_assignment(rows: list[Mapping[str, object]]) -> dict[str, int]:
    pred = sorted({str(r["predicted_token"]) for r in rows})
    truth = sorted({str(r["target_category"]) for r in rows})
    if not pred or not truth:
        return {}
    matrix = np.zeros((len(pred), len(truth)), dtype=np.int64)
    pi = {x: i for i, x in enumerate(pred)}
    ti = {x: i for i, x in enumerate(truth)}
    for r in rows:
        matrix[pi[str(r["predicted_token"])], ti[str(r["target_category"])]] += 1
    rr, cc = linear_sum_assignment(-matrix)
    return {pred[i]: truth[j] for i, j in zip(rr, cc)}


def stream_discovery_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Compute one global mapping per fold stream, then OLD/NEW/ALL scores.

    ``stream_id`` is only a grouping key; anonymous tokens from different
    streams are intentionally not matched across streams.  The caller should
    concatenate rows from one fold stream at a time when a global mapping is
    desired.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        return {"rows": 0, "old_rows": 0, "new_rows": 0,
                "all_correct": 0, "old_correct": 0, "new_correct": 0,
                "all_acc": 0.0, "old_acc": 0.0, "new_acc": 0.0,
                "pseudo_novel_acc": 0.0, "h_score": 0.0, "nmi": 0.0, "ari": 0.0}
    mapping = _global_assignment(rows)
    correct = [mapping.get(str(r["predicted_token"])) == str(r["target_category"]) for r in rows]
    old = [i for i, r in enumerate(rows) if str(r.get("split", "new")).lower() == "old"]
    new = [i for i, r in enumerate(rows) if str(r.get("split", "new")).lower() == "new"]
    old_acc = float(np.mean([correct[i] for i in old])) if old else 0.0
    new_acc = float(np.mean([correct[i] for i in new])) if new else 0.0
    all_acc = float(np.mean(correct))
    h = 2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12)
    y = [str(r["target_category"]) for r in rows]
    p = [str(r["predicted_token"]) for r in rows]
    return {
        "rows": len(rows), "old_rows": len(old), "new_rows": len(new),
        "all_correct": int(sum(correct)), "old_correct": int(sum(correct[i] for i in old)),
        "new_correct": int(sum(correct[i] for i in new)),
        "all_acc": all_acc, "old_acc": old_acc, "new_acc": new_acc,
        "pseudo_novel_acc": new_acc, "h_score": float(h),
        "nmi": float(normalized_mutual_info_score(y, p)),
        "ari": float(adjusted_rand_score(y, p)),
        "global_pred_to_category": mapping,
        "old_rows": len(old), "new_rows": len(new),
    }
