"""Full-stream persistent semantic memory metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _decision(row: Mapping) -> tuple[str, str | None]:
    decision = row.get("decision", row)
    return str(decision.get("kind", "DEFER")), decision.get("token")


def evaluate_persistent(
    rows: Sequence[Mapping],
    decisions: Sequence[Mapping] | None = None,
    *,
    known_ids: Iterable[int],
    novel_ids: Iterable[int],
    distractor_ids: Iterable[int] = (),
) -> dict:
    """Evaluate causal cross-video reuse without resetting memory.

    The history below is evaluator-side only.  It is used to score anonymous
    tokens; no ground-truth category is passed to a method during replay.
    """

    known, novel, distractor = set(map(int, known_ids)), set(map(int, novel_ids)), set(map(int, distractor_ids))
    if decisions is None:
        decisions = rows
    if len(rows) != len(decisions):
        raise ValueError("rows and decisions must have identical length")
    category_tracks: defaultdict[int, set[str]] = defaultdict(set)
    category_videos: defaultdict[int, set[int]] = defaultdict(set)
    for row in rows:
        category = int(row["gt_category_id"])
        if category in novel and category not in distractor:
            category_tracks[category].add(str(row["physical_track_id"]))
            category_videos[category].add(int(row["video_id"]))
    persistent_categories = {c for c in category_tracks if len(category_tracks[c]) >= 2 and len(category_videos[c]) >= 2}

    token_history: defaultdict[str, list[tuple[int, int, str]]] = defaultdict(list)
    # This history is evaluator-only.  It defines when a later row is a
    # cross-video reuse opportunity without exposing the category to a method.
    gt_history: list[tuple[int, int]] = []
    eligible = correct = false_assignment = unresolved = 0
    for row, decision in zip(rows, decisions):
        category = int(row["gt_category_id"])
        if category in distractor or category not in known and category not in novel:
            continue
        video_id = int(row["video_id"])
        kind, token = _decision(decision)
        token = None if token is None else str(token)
        prior = token_history[token] if token is not None else []
        prior_same_category_other_video = any(c == category and v != video_id for c, v, _ in prior)
        earlier_same = any(c == category and v != video_id for c, v in gt_history)
        if category in persistent_categories and earlier_same:
            eligible += 1
            if token is None or kind == "DEFER":
                unresolved += 1
            elif prior_same_category_other_video:
                correct += 1
            else:
                false_assignment += 1
        gt_history.append((category, video_id))
        if token is not None and kind != "DEFER":
            token_history[token].append((category, video_id, kind))

    denominator = max(eligible, 1)
    # The headline false-assignment denominator is the same fixed causal
    # target population as Commit-CT: every GT_REUSE_ELIGIBLE novel target.
    false_denominator = eligible
    return {
        "commit_ct": correct / denominator if eligible else 0.0,
        "commit_ct_correct": correct,
        "commit_ct_denominator": eligible,
        "false_assignment_rate": false_assignment / false_denominator if false_denominator else 0.0,
        "false_assignment_count": false_assignment,
        "false_assignment_denominator": false_denominator,
        "unresolved_count": unresolved,
        "persistent_category_count": len(persistent_categories),
        "persistent_categories": sorted(persistent_categories),
    }
