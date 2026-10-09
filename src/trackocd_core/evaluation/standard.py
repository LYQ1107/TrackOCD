"""Coverage-aware OCD: exact Known IDs; one global anonymous/Novel mapping."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import EvaluationJoin, ratio


def evaluate_standard(join: EvaluationJoin) -> dict:
    targets = [t for t in join.targets if t.role != "distractor"]
    gt_to_pred = {gt: physical for physical, gt in join.matches if gt is not None}
    committed = {c.event.physical_key: c.event for c in join.replay.commits}
    active_tokens = set()
    for target in targets:
        event = committed.get(gt_to_pred.get(target.key))
        if target.role == "novel" and event is not None and event.kind in {"NEW", "EXISTING"}:
            active_tokens.add(event.token)
    # Zero-support rows cannot create a correct assignment. Omit them from
    # the dense matrix, but NEVER from the reported full cluster count.
    tokens = [token for token in join.replay.created_tokens if token in active_tokens]
    novel_categories = sorted({t.category_id for t in targets if t.role == "novel"})
    token_index = {token: i for i, token in enumerate(tokens)}
    category_index = {category: i for i, category in enumerate(novel_categories)}
    contingency = np.zeros((len(tokens), len(novel_categories)), dtype=np.int64)
    for target in targets:
        event = committed.get(gt_to_pred.get(target.key))
        if target.role == "novel" and event is not None and event.kind in {"NEW", "EXISTING"}:
            contingency[token_index[event.token], category_index[target.category_id]] += 1
    mapping = {}
    if contingency.size:
        rows, columns = linear_sum_assignment(-contingency)
        mapping = {tokens[r]: novel_categories[c] for r, c in zip(rows, columns)
                   if contingency[r, c] > 0}
    totals, correct = Counter(), Counter()
    matched = non_wait = missing = wait = 0
    for target in targets:
        totals[target.role] += 1
        physical = gt_to_pred.get(target.key)
        event = committed.get(physical)
        matched += physical is not None
        missing += physical is None
        wait += physical is not None and event is None
        non_wait += event is not None
        hit = False
        if event is not None:
            if target.role == "known":
                hit = event.kind == "KNOWN" and event.known_category_id == target.category_id
            else:
                hit = event.kind in {"NEW", "EXISTING"} and mapping.get(event.token) == target.category_id
        correct[target.role] += bool(hit)
    old, new = ratio(correct["known"], totals["known"]), ratio(correct["novel"], totals["novel"])
    h = None if old is None or new is None else (2 * old * new / (old + new) if old + new else 0.0)
    denominator = len(targets)
    return {
        "schema_version": "trackocd.core.standard_ocd.v1",
        "old_acc": old, "new_acc": new, "h_score": h,
        "all_acc": ratio(sum(correct.values()), denominator),
        "old_correct": correct["known"], "old_denominator": totals["known"],
        "new_correct": correct["novel"], "new_denominator": totals["novel"],
        "all_denominator": denominator,
        "novel_cluster_count": len(join.replay.created_tokens),
        "anonymous_tokens_with_novel_support_in_mapping": len(tokens),
        "zero_support_tokens_omitted_only_from_dense_matrix": True,
        "cluster_count_includes_unmatched_and_known_misassigned_creations": True,
        "unresolved_count": missing + wait,
        "unresolved_coverage": ratio(missing + wait, denominator),
        "missing_predicted_target_count": missing, "matched_wait_or_no_decision_count": wait,
        "valid_predicted_target_count": matched,
        "non_wait_target_count": non_wait,
        "prediction_coverage": ratio(matched, denominator),
        "non_wait_coverage": ratio(non_wait, denominator),
        "global_anonymous_hungarian_mapping_evaluator_only": mapping,
        "hungarian_invocations": int(bool(contingency.size)),
        "known_ids_remapped": False,
        "zero_denominator_convention": "null/NOT_APPLICABLE, never manufactured perfect accuracy",
    }
