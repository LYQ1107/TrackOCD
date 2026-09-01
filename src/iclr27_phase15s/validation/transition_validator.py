"""Independent Phase15S transition and prefix validator."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

VALID_ACTIONS = {"known", "new", "existing"}


def action(row: Mapping) -> str | None:
    x = row.get("sem_action", row.get("action"))
    if x in (None, ""):
        return None
    return {"known_category": "known", "new_novel": "new", "new_novel_state": "new",
            "existing_novel": "existing"}.get(str(x).lower(), str(x).lower())


def sid(row: Mapping) -> int | None:
    x = row.get("sem_sid", row.get("semantic_id"))
    if x in (None, "", "None"):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def validate_transitions(rows: Sequence[Mapping], known_ids: Iterable[int], *,
                         internal_state_count: int | None = None,
                         metadata: Mapping | None = None) -> dict:
    known = {int(x) for x in known_ids}; errors = []; births = {}; uses = defaultdict(int)
    physical = defaultdict(set); counts = {a: 0 for a in sorted(VALID_ACTIONS)}
    for pos, row in enumerate(rows):
        a, k = action(row), sid(row)
        if a not in VALID_ACTIONS:
            errors.append({"row": pos, "type": "missing_or_invalid_action", "value": a}); continue
        counts[a] += 1
        if k is None: errors.append({"row": pos, "type": "missing_semantic_id", "action": a})
        if a == "known" and k not in known: errors.append({"row": pos, "type": "unsupported_known_id", "sid": k})
        if a == "new" and k is not None:
            if k in births: errors.append({"row": pos, "type": "repeated_new", "sid": k, "first_row": births[k]})
            else: births[k] = pos
            uses[k] += 1
        if a == "existing" and k is not None:
            if k not in births: errors.append({"row": pos, "type": "existing_before_new", "sid": k})
            uses[k] += 1
        if a != "known" and k is not None:
            physical[(row.get("video_id"), row.get("track_id", row.get("physical_id")))].add(k)
        for f in ("future_frames_used", "q1_label_used", "devplus_label_used",
                  "physical_id_used_as_feature", "private_gt_used_for_decision"):
            if bool(row.get(f, False)): errors.append({"row": pos, "type": "forbidden_decision_input", "field": f})
    if counts["new"] != len(births): errors.append({"type": "new_rows_vs_unique_births_mismatch", "new_rows": counts["new"], "unique_births": len(births)})
    if internal_state_count is not None and int(internal_state_count) != len(births):
        errors.append({"type": "internal_vs_global_birth_mismatch", "internal_state_count": int(internal_state_count), "unique_births": len(births)})
    return {"valid": not errors, "rows": len(rows), "actions": counts,
            "new_action_count": counts["new"], "unique_new_state_count": len(births),
            "existing_state_uses": {str(k): int(v) for k, v in sorted(uses.items())},
            "birth_rows": {str(k): int(v) for k, v in sorted(births.items())},
            "physical_semantic_pairs": int(sum(len(v) for v in physical.values())),
            "errors": errors, "metadata": dict(metadata or {})}


def audit_prefix_invariance(rows: Sequence[Mapping], replay_fn: Callable,
                            cuts: Iterable[int] = (1, 2, 4, 8, 16)) -> dict:
    full = list(replay_fn(rows)); errors = []
    if len(full) != len(rows): return {"valid": False, "errors": [{"type": "full_length_mismatch"}]}
    for cut in sorted({int(x) for x in cuts if int(x) > 0}):
        short = list(replay_fn(rows[:cut]))
        if len(short) != cut: errors.append({"cut": cut, "type": "prefix_length_mismatch"}); continue
        for i, (a, b) in enumerate(zip(short, full[:cut])):
            if (action(a), sid(a)) != (action(b), sid(b)):
                errors.append({"cut": cut, "row": i, "prefix": (action(a), sid(a)), "full": (action(b), sid(b)), "type": "future_dependent_output"}); break
    return {"valid": not errors, "cuts": sorted({int(x) for x in cuts if int(x) > 0}), "errors": errors}
