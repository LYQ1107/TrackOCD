"""Independent evaluator-facing transition validator for Phase 15R.

The validator deliberately knows nothing about a model or feature cache.  It
checks only the immutable row contract and can therefore detect a mismatch
between a model's internal state counter and the strict global birth count.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence


VALID_ACTIONS = {"known", "new", "existing"}


def _sid(row: Mapping) -> int | None:
    value = row.get("sem_sid", row.get("semantic_id"))
    if value in (None, "", "None"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _action(row: Mapping) -> str | None:
    value = row.get("sem_action", row.get("action"))
    if value in (None, ""):
        return None
    value = str(value).lower()
    # Historical CSV uses these spellings; normalise only the fixed aliases.
    return {"known_category": "known", "new_novel": "new",
            "new_novel_state": "new", "existing_novel": "existing"}.get(value, value)


def validate_transitions(rows: Sequence[Mapping], known_ids: Iterable[int],
                         *, internal_state_count: int | None = None,
                         metadata: Mapping | None = None) -> dict:
    """Return a strict transition audit, raising no exception.

    ``errors`` is intentionally explicit so a caller can save the failed
    audit instead of accidentally treating a partial replay as valid.
    """
    known = {int(x) for x in known_ids}
    errors: list[dict] = []
    births: dict[int, int] = {}
    uses: defaultdict[int, int] = defaultdict(int)
    physical_to_semantic: defaultdict[tuple, set[int]] = defaultdict(set)
    actions = {a: 0 for a in sorted(VALID_ACTIONS)}
    for pos, row in enumerate(rows):
        action = _action(row)
        sid = _sid(row)
        key = (row.get("video_id"), row.get("track_id", row.get("physical_id")))
        if action not in VALID_ACTIONS:
            errors.append({"row": pos, "type": "missing_or_invalid_action", "value": action})
            continue
        actions[action] += 1
        if sid is None:
            errors.append({"row": pos, "type": "missing_semantic_id", "action": action})
        if action == "known":
            if sid not in known:
                errors.append({"row": pos, "type": "unsupported_known_id", "sid": sid})
        elif action == "new":
            if sid is None:
                continue
            if sid in births:
                errors.append({"row": pos, "type": "repeated_new", "sid": sid,
                               "first_row": births[sid]})
            else:
                births[sid] = pos
                uses[sid] += 1
        elif action == "existing":
            if sid is None:
                continue
            if sid not in births:
                errors.append({"row": pos, "type": "existing_before_new", "sid": sid})
            uses[sid] += 1
        if sid is not None and action != "known":
            physical_to_semantic[key].add(sid)

        # Callers may pass an explicit audit marker from a decision function.
        for forbidden in ("future_frames_used", "q1_label_used", "devplus_label_used",
                          "physical_id_used_as_feature", "private_gt_used_for_decision"):
            if bool(row.get(forbidden, False)):
                errors.append({"row": pos, "type": "forbidden_decision_input", "field": forbidden})

    # The strict global birth count is a unique semantic-state count, not a
    # count of every row labelled NEW.  The equality is the key regression.
    unique_births = len(births)
    new_rows = actions["new"]
    if new_rows != unique_births:
        errors.append({"type": "new_rows_vs_unique_births_mismatch",
                       "new_rows": new_rows, "unique_births": unique_births})
    if internal_state_count is not None and int(internal_state_count) != unique_births:
        errors.append({"type": "internal_vs_global_birth_mismatch",
                       "internal_state_count": int(internal_state_count),
                       "unique_births": unique_births})

    return {
        "valid": not errors,
        "rows": int(len(rows)),
        "actions": actions,
        "new_action_count": int(new_rows),
        "unique_new_state_count": int(unique_births),
        "existing_state_uses": {str(k): int(v) for k, v in sorted(uses.items())},
        "birth_rows": {str(k): int(v) for k, v in sorted(births.items())},
        "physical_semantic_pairs": int(sum(len(v) for v in physical_to_semantic.values())),
        "errors": errors,
        "metadata": dict(metadata or {}),
    }


def audit_prefix_invariance(rows: Sequence[Mapping], replay_fn: Callable[[Sequence[Mapping]], Sequence[Mapping]],
                           cut_points: Iterable[int] = (1, 2, 4, 8, 16)) -> dict:
    """Check that a causal replay's prior outputs do not depend on future rows."""
    full = list(replay_fn(rows))
    if len(full) != len(rows):
        return {"valid": False, "errors": [{"type": "full_length_mismatch"}]}
    errors = []
    for cut in sorted({int(c) for c in cut_points if int(c) > 0}):
        prefix_rows = list(rows[:cut])
        prefix_out = list(replay_fn(prefix_rows))
        if len(prefix_out) != len(prefix_rows):
            errors.append({"cut": cut, "type": "prefix_length_mismatch"})
            continue
        for i, (a, b) in enumerate(zip(prefix_out, full[:cut])):
            aa = (_action(a), _sid(a))
            bb = (_action(b), _sid(b))
            if aa != bb:
                errors.append({"cut": cut, "row": i, "prefix": aa, "full": bb,
                               "type": "future_dependent_output"})
                break
    return {"valid": not errors, "cuts": sorted({int(c) for c in cut_points if int(c) > 0}),
            "errors": errors}
