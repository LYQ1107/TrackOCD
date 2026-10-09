"""Fixed-GT-opportunity persistent reuse, with absorbing impurity history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .contracts import EvaluationJoin, ratio


@dataclass(slots=True)
class _History:
    first_category: int | None = None
    mixed: bool = False
    unknown: bool = False
    first_video: int | None = None
    multiple_videos: bool = False
    members: int = 0

    def add(self, target, video_id: int) -> None:
        self.members += 1
        if self.first_video is None:
            self.first_video = video_id
        elif self.first_video != video_id:
            self.multiple_videos = True
        if target is None:
            self.unknown = True
        elif self.first_category is None:
            self.first_category = target.category_id
        elif self.first_category != target.category_id:
            self.mixed = True


def evaluate_persistent(join: EvaluationJoin) -> dict:
    ranks = {v: i for i, v in enumerate(join.replay.video_order)}
    targets_by_key = {t.key: t for t in join.targets}
    predicted_to_gt = dict(join.matches)
    gt_to_predicted = {gt: pred for pred, gt in join.matches if gt is not None}
    commits = {c.event.physical_key: c for c in join.replay.commits}
    first_rank = {}
    novel = [t for t in join.targets if t.role == "novel"]
    for target in novel:
        first_rank[target.category_id] = min(first_rank.get(target.category_id, ranks[target.key.video_id]),
                                              ranks[target.key.video_id])
    eligible = [t for t in novel if ranks[t.key.video_id] > first_rank[t.category_id]]
    categories_with_reuse = {t.category_id for t in eligible}
    outcomes = Counter()
    matched = non_wait = 0
    events_by_key = {e.physical_key for e in join.replay.events}
    eligible_keys = {t.key for t in eligible}
    for target in eligible:
        physical = gt_to_predicted.get(target.key)
        commit = commits.get(physical)
        matched += physical is not None
        if physical is None:
            outcomes["missed_predicted_opportunity"] += 1
            continue
        if commit is None:
            outcomes["wait"] += 1
            if physical not in events_by_key:
                outcomes["wait_with_no_decision_record"] += 1
            continue
    # One evaluator-only chronological pass. Store absorbing summaries, not
    # a copied ever-growing token-member list at every commitment (O(n^2)).
    histories: dict[str, _History] = {}
    for commit in join.replay.commits:
        event = commit.event
        target = targets_by_key.get(predicted_to_gt.get(event.physical_key))
        if event.kind == "NEW":
            if event.token in histories:
                raise ValueError("Sealed ledger illegally recycles NEW")
            history = histories[event.token] = _History()
        elif event.kind == "EXISTING":
            history = histories.get(event.token)
            if history is None or history.members != commit.prior_token_member_count:
                raise ValueError("Sealed membership order/count mismatch")
        else:
            history = None
        if target is not None and target.key in eligible_keys:
            non_wait += 1
            if event.kind == "KNOWN":
                outcomes["wrong_known_assignment"] += 1
            elif event.kind == "NEW":
                outcomes["false_split_new"] += 1
            elif history.mixed:
                outcomes["contaminated_token_reuse"] += 1
            elif history.first_category is not None and history.first_category != target.category_id:
                outcomes["wrong_category_merge"] += 1
            elif history.unknown:
                # Unknown/background membership is not positively proven
                # pollution, but cannot certify pure reuse either.
                outcomes["unverified_token_reuse"] += 1
            elif (history.first_category == target.category_id
                  and (history.multiple_videos or history.first_video != target.key.video_id)):
                outcomes["pure_correct_reuse"] += 1
            else:
                outcomes["same_video_only_unsupported_existing"] += 1
        if history is not None:
            # Add only AFTER scoring this commitment; never future members.
            history.add(target, event.physical_key.video_id)
    denominator = len(eligible)
    primary = ("pure_correct_reuse", "false_split_new", "wrong_known_assignment", "contaminated_token_reuse",
               "unverified_token_reuse", "wrong_category_merge", "same_video_only_unsupported_existing",
               "missed_predicted_opportunity", "wait")
    if sum(outcomes[name] for name in primary) != denominator:
        raise AssertionError("Persistent outcomes must partition the fixed GT opportunity universe")
    correct = outcomes["pure_correct_reuse"]
    false_merge = outcomes["wrong_category_merge"] + outcomes["contaminated_token_reuse"]
    unresolved = outcomes["missed_predicted_opportunity"] + outcomes["wait"]
    return {
        "schema_version": "trackocd.core.persistent_ocd.v1",
        "all_eligible_novel_gt_tracks": len(novel),
        "fixed_gt_cross_video_reuse_opportunities": denominator,
        "valid_predicted_opportunity_count": matched,
        "non_wait_opportunity_count": non_wait,
        "persistent_category_count": len(categories_with_reuse),
        "commit_ct_correct": correct, "commit_ct_denominator": denominator,
        "correct_commit_ct": ratio(correct, denominator),
        "false_merge_count": false_merge, "false_merge_rate": ratio(false_merge, denominator),
        "false_split_new_count": outcomes["false_split_new"],
        "false_split_new_rate": ratio(outcomes["false_split_new"], denominator),
        "wrong_known_assignment_count": outcomes["wrong_known_assignment"],
        "wrong_known_assignment_rate": ratio(outcomes["wrong_known_assignment"], denominator),
        "unverified_token_reuse_rate": ratio(outcomes["unverified_token_reuse"], denominator),
        "unsupported_same_video_existing_rate": ratio(outcomes["same_video_only_unsupported_existing"], denominator),
        "wait_unresolved_count": unresolved, "wait_unresolved_rate": ratio(unresolved, denominator),
        "effective_commit_coverage": ratio(non_wait, denominator),
        "coverage_counts_incorrect_commits_and_is_not_accuracy": True,
        "matched_only_commit_ct_diagnostic": ratio(correct, matched),
        "outcome_counts": {name: outcomes[name] for name in (*primary, "wait_with_no_decision_record")},
        "unknown_members_cannot_certify_purity_but_are_not_proven_false_merges": True,
        "zero_denominator_convention": "null/NOT_APPLICABLE",
        "posthoc_hungarian_used": False,
    }
