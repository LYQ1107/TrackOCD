"""Posthoc physical cross-video support, NEVER semantic correctness/memory."""
from collections import defaultdict
from .contracts import Target, TrackKey, ratio
from .persistent import fixed_cross_video_targets


def cross_video_support(rows, order, prefixes=(1, 2, 4, 8, 16)):
    if tuple(prefixes) != (1, 2, 4, 8, 16):
        raise ValueError("All five registered prefixes required; no favorable-prefix selection")
    targets = [Target(TrackKey(r["video_id"], str(r["gt_local_id"])), r["category_id"], r["role"]) for r in rows]
    keys = [t.key for t in targets]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate GT namespace in physical support audit")
    eligible = fixed_cross_video_targets(targets, order)
    eligible_keys = {t.key for t in eligible}
    by_video = defaultdict(list)
    for row, target in zip(rows, targets):
        count = row["reliable_predicted_observations"]
        if type(count) is not int or count < 0 or row["role"] not in {"known", "novel"}:
            raise ValueError("Invalid geometry-only support row")
        by_video[target.key.video_id].append((row, target))
    past_best = defaultdict(int)
    counts = {p: 0 for p in prefixes}; categories = {p: set() for p in prefixes}
    current = 0
    for video in order:
        # Score before adding ANY member from this video; same-video tracks
        # cannot provide an earlier cross-video source to each other.
        for row, target in by_video[video]:
            if target.key not in eligible_keys:
                continue
            n = row["reliable_predicted_observations"]
            current += n > 0
            for p in prefixes:
                if n >= p and past_best[target.category_id] >= p:
                    counts[p] += 1; categories[p].add(target.category_id)
        for row, target in by_video[video]:
            if target.role == "novel":
                past_best[target.category_id] = max(past_best[target.category_id], row["reliable_predicted_observations"])
    denominator = len(eligible)
    return {"fixed_gt_reuse_opportunities": denominator,
            "gt_categories_with_cross_video_reuse": len({t.category_id for t in eligible}),
            "reliable_current_opportunities": current,
            "missing_current_physical_opportunities": denominator - current,
            "current_reliable_but_no_earlier_reliable_same_category": current - counts[1],
            "prefix_support": {str(p): {"both_earlier_and_current_have_at_least_p": counts[p],
                                       "fixed_gt_opportunity_denominator": denominator,
                                       "fraction_of_all_fixed_gt_opportunities": ratio(counts[p], denominator),
                                       "categories_with_supported_pairs": len(categories[p])} for p in prefixes},
            "semantic_decisions_or_correct_reuse_computed": False,
            "short_tracks_removed_from_denominator": False,
            "unknown_unmatched_rows_assumed_background_or_pure": False}
