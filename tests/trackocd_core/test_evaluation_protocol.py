"""M3 anti-cheating fixtures; entirely synthetic, no Val/Test data access."""

import pytest

from src.trackocd_core.evaluation import (
    DecisionEvent, Target, TrackKey, evaluate_persistent, evaluate_standard,
    join_evaluation, seal_decisions,
)


def gt(video, local, category, role="novel"):
    return Target(TrackKey(video, str(local)), category, role)


def event(sequence, target, kind, token=None, known=None, prefix=1):
    return DecisionEvent(sequence, target.key, prefix, kind, known, token)


def joined(targets, events, matches=None, order=None, prefix=1):
    if order is None:
        order = sorted({t.key.video_id for t in targets} | {e.physical_key.video_id for e in events})
    replay = seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=(1, 2))
    if matches is None:
        gt_keys = {t.key for t in targets}
        matches = {e.physical_key: e.physical_key if e.physical_key in gt_keys else None for e in events}
    return join_evaluation(replay, targets, matches)


def test_one_novel_category_across_three_videos_pure_reuse():
    targets = [gt(v, 7, 10) for v in (1, 2, 3)]
    events = [event(0, targets[0], "NEW", "x"), event(1, targets[1], "EXISTING", "x"),
              event(2, targets[2], "EXISTING", "x")]
    j = joined(targets, events)
    persistent = evaluate_persistent(j)
    assert persistent["commit_ct_denominator"] == 2
    assert persistent["correct_commit_ct"] == 1
    assert persistent["effective_commit_coverage"] == 1
    assert evaluate_standard(j)["new_acc"] == 1
    assert len(j.replay.commits) == 3  # same local ID is distinct in each video


def test_two_individuals_of_same_category_are_not_two_categories():
    a, b, c = gt(1, "individual-a", 10), gt(1, "individual-b", 10), gt(2, "individual-c", 10)
    j = joined([a, b, c], [event(0, a, "NEW", "x"), event(1, b, "EXISTING", "x"), event(2, c, "EXISTING", "x")])
    assert evaluate_standard(j)["novel_cluster_count"] == 1
    assert evaluate_persistent(j)["commit_ct_denominator"] == 1
    assert evaluate_persistent(j)["correct_commit_ct"] == 1


def test_wait_then_existing_counts_target_once_and_only_observed_prefix():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    j = joined([a, b], [event(0, a, "NEW", "x"), event(1, b, "WAIT"),
                        event(2, b, "EXISTING", "x", prefix=2)], prefix=2)
    assert evaluate_persistent(j)["commit_ct_denominator"] == 1
    assert evaluate_persistent(j)["commit_ct_correct"] == 1
    assert len(j.replay.commits) == 2


def test_wrong_new_is_a_split_not_a_false_merge():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    p = evaluate_persistent(joined([a, b], [event(0, a, "NEW", "x"), event(1, b, "NEW", "y")]))
    assert p["false_split_new_count"] == 1
    assert p["false_merge_count"] == 0
    assert p["correct_commit_ct"] == 0
    assert p["effective_commit_coverage"] == 1  # commitment is not correctness


def test_similar_novel_features_do_not_justify_wrong_merge_and_poison_is_absorbing():
    # Identical model descriptors could produce these confused events; the
    # evaluator neither observes features nor repairs them with GT mapping.
    a, b, c, d = gt(1, "a", 10), gt(1, "b", 11), gt(2, "c", 11), gt(3, "d", 10)
    j = joined([a, b, c, d], [event(0, a, "NEW", "x"), event(1, b, "NEW", "y"),
                             event(2, c, "EXISTING", "x"), event(3, d, "EXISTING", "x")])
    p = evaluate_persistent(j)
    assert p["commit_ct_denominator"] == 2
    assert p["outcome_counts"]["wrong_category_merge"] == 1
    assert p["outcome_counts"]["contaminated_token_reuse"] == 1
    assert p["false_merge_count"] == 2
    assert p["correct_commit_ct"] == 0


def test_wrong_known_assignment_cannot_be_hungarian_remapped_to_correct():
    a, b, c = gt(1, "a", 1, "known"), gt(1, "b", 2, "known"), gt(1, "c", 10)
    j = joined([a, b, c], [event(0, a, "KNOWN", known=2), event(1, b, "KNOWN", known=1),
                          event(2, c, "KNOWN", known=1)])
    s = evaluate_standard(j)
    assert s["old_acc"] == 0 and s["new_acc"] == 0
    assert s["global_anonymous_hungarian_mapping_evaluator_only"] == {}
    assert not s["known_ids_remapped"]


def test_novel_reuse_wrong_known_reported_separately():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    p = evaluate_persistent(joined([a, b], [event(0, a, "NEW", "x"), event(1, b, "KNOWN", known=1)]))
    assert p["wrong_known_assignment_count"] == 1
    assert p["false_merge_count"] == p["false_split_new_count"] == 0


def test_missing_predicted_track_retains_gt_opportunity_denominator():
    a, b, c = gt(1, 7, 10), gt(2, 7, 10), gt(3, 7, 10)
    j = joined([a, b, c], [event(0, a, "NEW", "x"), event(1, c, "EXISTING", "x")])
    p, s = evaluate_persistent(j), evaluate_standard(j)
    assert p["commit_ct_denominator"] == 2
    assert p["correct_commit_ct"] == 0.5
    assert p["matched_only_commit_ct_diagnostic"] == 1
    assert p["outcome_counts"]["missed_predicted_opportunity"] == 1
    assert s["new_denominator"] == 3 and s["new_acc"] == pytest.approx(2 / 3)


def test_missing_source_cannot_remove_later_eligible_opportunity():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    p = evaluate_persistent(joined([a, b], [event(0, b, "NEW", "x")]))
    assert p["commit_ct_denominator"] == 1
    assert p["false_split_new_count"] == 1
    assert p["correct_commit_ct"] == 0


def test_all_wait_never_gets_a_manufactured_high_score():
    known, a, b = gt(1, "known", 1, "known"), gt(1, 7, 10), gt(2, 7, 10)
    j = joined([known, a, b], [event(i, t, "WAIT") for i, t in enumerate([known, a, b])])
    s, p = evaluate_standard(j), evaluate_persistent(j)
    assert s["old_acc"] == s["new_acc"] == s["h_score"] == s["all_acc"] == 0
    assert s["unresolved_coverage"] == 1
    assert p["correct_commit_ct"] == p["effective_commit_coverage"] == 0
    assert p["wait_unresolved_rate"] == 1


def test_global_hungarian_cannot_repair_wrong_online_existing():
    a, b, c, d = gt(1, "a", 10), gt(1, "b", 11), gt(2, "c", 11), gt(3, "d", 11)
    j = joined([a, b, c, d], [event(0, a, "NEW", "x"), event(1, b, "NEW", "y"),
                             event(2, c, "EXISTING", "x"), event(3, d, "EXISTING", "x")])
    s, p = evaluate_standard(j), evaluate_persistent(j)
    assert s["global_anonymous_hungarian_mapping_evaluator_only"]["x"] == 11
    assert s["hungarian_invocations"] == 1
    assert p["correct_commit_ct"] == 0  # first wrong merge, then contaminated
    assert not p["posthoc_hungarian_used"]


def test_unknown_unmatched_member_cannot_certify_purity_or_be_declared_proven_merge():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    unmatched = gt(1, "background-physical", 99)
    j = joined([a, b], [event(0, unmatched, "NEW", "x"), event(1, b, "EXISTING", "x")])
    p = evaluate_persistent(j)
    assert p["correct_commit_ct"] == 0
    assert p["outcome_counts"]["unverified_token_reuse"] == 1
    assert p["false_merge_count"] == 0
    assert evaluate_standard(j)["novel_cluster_count"] == 1


def test_same_video_only_existing_is_not_cross_video_commit():
    a, b, c = gt(1, 7, 10), gt(2, 7, 10), gt(2, 8, 10)
    p = evaluate_persistent(joined([a, b, c], [event(0, b, "NEW", "x"), event(1, c, "EXISTING", "x")]))
    assert p["commit_ct_denominator"] == 2
    assert p["outcome_counts"]["same_video_only_unsupported_existing"] == 1
    assert p["false_split_new_count"] == 1 and p["false_merge_count"] == 0


def test_future_members_do_not_retroactively_pollute_an_earlier_correct_commit():
    a, b, c, d = gt(1, 7, 10), gt(2, 7, 10), gt(3, 7, 11), gt(4, 7, 10)
    p = evaluate_persistent(joined([a, b, c, d], [event(0, a, "NEW", "x"), event(1, b, "EXISTING", "x"),
                                                event(2, c, "EXISTING", "x"), event(3, d, "EXISTING", "x")]))
    assert p["commit_ct_denominator"] == 2
    assert p["commit_ct_correct"] == 1
    assert p["outcome_counts"]["contaminated_token_reuse"] == 1


@pytest.mark.parametrize("bad", ["future_token", "reuse_new", "hide_commit", "future_prefix", "past_video", "reverse_sequence"])
def test_illegal_prediction_ledgers_are_not_silently_repaired(bad):
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    events = [event(0, a, "NEW", "x"), event(1, b, "EXISTING", "x")]
    if bad == "future_token": events[0] = event(0, a, "EXISTING", "future")
    if bad == "reuse_new": events[1] = event(1, b, "NEW", "x")
    if bad == "hide_commit": events.insert(1, event(1, a, "WAIT"))
    if bad == "future_prefix": events[1] = event(1, b, "EXISTING", "x", prefix=2)
    if bad == "past_video": events = [event(0, b, "NEW", "x"), event(1, a, "EXISTING", "x")]
    if bad == "reverse_sequence": events[1] = event(0, b, "EXISTING", "x")
    with pytest.raises(ValueError): joined([a, b], events)


def test_duplicate_gt_join_is_not_double_counted():
    a, p, q = gt(1, "gt", 10), gt(1, "pred-p", 10), gt(1, "pred-q", 10)
    with pytest.raises(ValueError, match="one-to-one"):
        joined([a], [event(0, p, "NEW", "x"), event(1, q, "EXISTING", "x")],
               matches={p.key: a.key, q.key: a.key})


def test_unmatched_created_tokens_are_not_hidden_from_cluster_count():
    a, unmatched = gt(1, 7, 10), gt(1, "unmatched", 99)
    s = evaluate_standard(joined([a], [event(0, a, "NEW", "x"), event(1, unmatched, "NEW", "spurious")]))
    assert s["novel_cluster_count"] == 2
    assert s["anonymous_tokens_with_novel_support_in_mapping"] == 1
    assert s["new_acc"] == 1


def test_zero_opportunities_is_not_applicable_not_perfect_accuracy():
    a = gt(1, 7, 10)
    j = joined([a], [event(0, a, "NEW", "x")])
    assert evaluate_persistent(j)["correct_commit_ct"] is None
    assert evaluate_standard(j)["old_acc"] is None
    assert evaluate_standard(j)["h_score"] is None


@pytest.mark.parametrize("order", [(1, 2, 3, 4), (3, 1, 4, 2), (4, 3, 2, 1), (2, 4, 1, 3)])
@pytest.mark.parametrize("prefix", [1, 2, 4, 8, 16])
def test_four_toy_video_orders_and_five_prefixes_use_independent_ledgers(order, prefix):
    # Preconstructed correct events are a protocol fixture, not results of
    # any learned method. Each of twenty cases starts with a fresh token.
    targets = [gt(video, 7, 10) for video in (1, 2, 3, 4)]
    by_video = {target.key.video_id: target for target in targets}
    events = [event(i, by_video[video], "NEW" if i == 0 else "EXISTING", "x", prefix=prefix)
              for i, video in enumerate(order)]
    j = joined(targets, events, order=order, prefix=prefix)
    p = evaluate_persistent(j)
    assert p["commit_ct_denominator"] == 3
    assert p["correct_commit_ct"] == 1
    assert j.replay.created_tokens == ("x",)
    assert j.replay.commits[0].prior_token_member_count == 0


def test_matched_but_no_decision_is_unresolved_not_undetected_or_silent_drop():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    j = joined([a, b], [event(0, a, "NEW", "x")], matches={a.key: a.key, b.key: b.key})
    p = evaluate_persistent(j)
    assert p["valid_predicted_opportunity_count"] == 1
    assert p["outcome_counts"]["wait_with_no_decision_record"] == 1
    assert p["outcome_counts"]["missed_predicted_opportunity"] == 0
    assert evaluate_standard(j)["matched_wait_or_no_decision_count"] == 1


def test_distractor_or_known_member_does_not_disappear_from_purity_history():
    a, distractor, b = gt(1, 7, 10), gt(1, "distractor", 20, "distractor"), gt(2, 7, 10)
    j = joined([a, distractor, b], [event(0, a, "NEW", "x"), event(1, distractor, "EXISTING", "x"),
                                  event(2, b, "EXISTING", "x")])
    assert evaluate_standard(j)["all_denominator"] == 2
    assert evaluate_persistent(j)["outcome_counts"]["contaminated_token_reuse"] == 1


def test_repeated_identical_confirmation_is_not_a_second_membership_commit():
    a, b = gt(1, 7, 10), gt(2, 7, 10)
    j = joined([a, b], [event(0, a, "NEW", "x"), event(1, a, "NEW", "x", prefix=2),
                        event(2, b, "EXISTING", "x", prefix=2)], prefix=2)
    assert len(j.replay.commits) == 2
    assert j.replay.commits[1].prior_token_member_count == 1
    assert evaluate_persistent(j)["correct_commit_ct"] == 1


def test_long_token_ledger_stores_counts_not_quadratically_copied_history():
    targets = [gt(1, i, 10) for i in range(2000)]
    events = [event(i, target, "NEW" if i == 0 else "EXISTING", "x") for i, target in enumerate(targets)]
    j = joined(targets, events)
    assert len(j.replay.commits) == 2000
    assert j.replay.commits[-1].prior_token_member_count == 1999
    assert not hasattr(j.replay.commits[-1], "prior_token_members")
    assert evaluate_standard(j)["new_acc"] == 1
    assert evaluate_persistent(j)["commit_ct_denominator"] == 0


def test_float_or_boolean_ids_and_fractional_prefixes_do_not_alias_frozen_schema():
    with pytest.raises(ValueError): TrackKey(True, "7")
    with pytest.raises(ValueError): Target(TrackKey(1, "7"), 10.0, "novel")
    with pytest.raises(ValueError): DecisionEvent(0, TrackKey(1, "7"), 1.5, "WAIT")
    with pytest.raises(ValueError): DecisionEvent(0, TrackKey(1, "7"), 1, "KNOWN", known_category_id=True)
