from src.trackocd_cvpr.existing_gate_structural_probe import (
    known_gate_bypassed_by_own_prior_support, known_output_retains_old_anonymous_support,
    already_Known_without_anonymous_birth)


def test_actual_synthetic_atomic_controller_self_continuation_can_override_positive_Known_gate():
    r = known_gate_bypassed_by_own_prior_support()
    assert r['frames'][0]['actual_action'] == 'NEW'
    assert not r['frames'][0]['frozen_named_membership_gate_positive']
    for frame in r['frames'][1:]:
        assert frame['frozen_named_membership_gate_positive'] and frame['actual_action'] == 'EXISTING'
        assert frame['max_Existing_cosine'] >= frame['max_Known_cosine']
    assert r['frames'][-1]['max_Known_cosine'] > .99
    assert r['anonymous_tokens'] == 1 and r['actual_commits'] == 9 and r['sole_member_support_groups'] == [1]
    assert not r['GT_labels_weights_data_or_dev_predictions_read']
    assert not r['registered_real_experiment_rule_or_history_changed'] and not r['actual_dev_attribution_or_Goal_accepted']


def test_actual_forced_Known_transaction_retains_old_support_and_immutable_history_not_GT_repair():
    r = known_output_retains_old_anonymous_support()
    assert r['first_actual_action'] == 'NEW' and r['diagnostic_forced_second_action'] == 'KNOWN'
    assert r['prior_commits_unchanged_and_retained'] and r['prior_anonymous_numeric_support_retained']
    assert not r['GT_history_repair_or_actual_dev_attribution']


def test_negative_case_no_anonymous_birth_no_inevitable_absorption_claim():
    r = already_Known_without_anonymous_birth()
    assert r['actions'] == ['KNOWN']*3 and r['anonymous_tokens'] == r['anonymous_commits'] == 0
    assert not r['self_absorption_claimed_inevitable_for_all_sequences']
