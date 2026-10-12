import torch
from src.trackocd_cvpr.candidate_score_contract import CandidateBinding
from src.trackocd_cvpr.candidate_training_loss import (
    ExternalPredictionHistory, prediction_owned_loss_plan, candidate_supervised_loss,
)
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl
from tests.trackocd_cvpr.test_atomic_member_control import obs, new_policy, reuse_policy


def named(i): return (CandidateBinding('KNOWN', i),)
def state(s): return (CandidateBinding('EXISTING', s),)


def test_known_targets_do_not_label_unknown_candidates_negative_and_missing_target_masked():
    plan = prediction_owned_loss_plan([named(0), state('S:0')], [(1, 'q')],
        {(1, 'q'): ('Known', 0)}, {'S:0': (((2, 'past'), None),)})
    assert plan.valid == (True, False, True, True) and plan.positive == (True, False, False, False)
    scores = torch.tensor([0., 100., 0., 0.], requires_grad=True)
    loss = candidate_supervised_loss(scores, plan); loss.backward()
    assert scores.grad[1] == 0 and scores.grad[0] < 0
    absent = prediction_owned_loss_plan([named(1)], [(1, 'q')], {(1, 'q'): ('Known', 0)}, {})
    assert candidate_supervised_loss(torch.zeros(3), absent) is None


def test_pseudo_new_not_justified_by_unknown_state_and_pure_existing_requires_real_past():
    keys = [(1, 'q')]; labels = {(1, 'q'): ('policy_pseudo', 0)}
    empty = prediction_owned_loss_plan([named(0)], keys, labels, {})
    assert empty.positive == (False, True, False)
    unknown = prediction_owned_loss_plan([state('S:9')], keys, labels, {})
    assert not any(unknown.positive)
    same = prediction_owned_loss_plan([state('S:0')], keys, labels,
        {'S:0': (((1, 'q'), ('policy_pseudo', 0)),)})
    assert same.positive[0] and not same.positive_has_other_physical_member
    different = prediction_owned_loss_plan([state('S:0')], keys, labels,
        {'S:0': (((2, 'p'), ('policy_pseudo', 0)),)})
    assert different.positive_has_other_physical_member and different.positive_has_other_video_member


def test_real_wrong_commits_retained_and_GT_loss_cannot_change_model_owned_bank():
    calls = []
    def scripted_numeric_policy(features, context):
        calls.append(1)
        return (reuse_policy if len(calls) <= 2 else new_policy)(features, context)
    history = ExternalPredictionHistory(); control = AtomicMemberControl({}, scripted_numeric_policy)
    labels = {(1, 'a'): ('policy_pseudo', 0), (2, 'b'): ('policy_pseudo', 1)}
    first = control.ingest(0, [obs('a')]); history.observe_frame(1, first, labels)
    old = history.records(); control.start_next_video()
    prior = history.snapshot(); second = control.ingest(0, [obs('b')])
    assert second.events[0].token == 'S:0'  # Actual wrong reuse, not a GT reset.
    plan = prediction_owned_loss_plan(second.events[0].candidate_bindings_for_external_loss,
        [(2, 'b')], labels, prior)
    assert plan.positive[-2] and not plan.positive[0]
    bank_before = control.state_snapshot()[0].commit_history()
    candidate_supervised_loss(torch.zeros(len(plan.valid), requires_grad=True), plan).backward()
    assert control.state_snapshot()[0].commit_history() == bank_before
    history.observe_frame(2, second, labels)
    assert history.records()[:len(old)] == old and len(history.snapshot()['S:0']) == 2
    # Moving b later to its own predicted token cannot erase pollution in S:0.
    third = control.ingest(1, [obs('b')]); history.observe_frame(2, third, labels)
    assert third.events[0].token == 'S:1'
    assert len(history.snapshot()['S:0']) == 2
    assert len(history.snapshot()['S:1']) == 1


def test_exact_candidate_binding_ambiguity_waits_without_oracle_ID_choice():
    plan = prediction_owned_loss_plan([(CandidateBinding('KNOWN', 1), CandidateBinding('KNOWN', 2))],
        [(1, 'q')], {(1, 'q'): ('Known', 1)}, {})
    assert plan.reason == 'WAIT_FOR_UNRESOLVABLE_BINDING_AMBIGUITY' and plan.positive == (False, False, True)


def test_unknown_and_conflicting_query_aliases_not_novel_or_background():
    unknown = prediction_owned_loss_plan([], [(1, 'unknown')], {}, {})
    conflict = prediction_owned_loss_plan([], [(1, 'a'), (1, 'b')],
        {(1, 'a'): ('Known', 0), (1, 'b'): ('policy_pseudo', 0)}, {})
    assert not any(unknown.positive) and not any(conflict.positive)


def test_candidate_permutation_and_multi_positive_loss_value_and_gradient_equivariance():
    bindings = [named(1), state('S:0'), state('S:1')]
    prior = {s: (((2, s), ('policy_pseudo', 0)),) for s in ('S:0', 'S:1')}
    labels = {(1, 'q'): ('policy_pseudo', 0)}; order = [2, 0, 1]
    a = prediction_owned_loss_plan(bindings, [(1, 'q')], labels, prior)
    b = prediction_owned_loss_plan([bindings[i] for i in order], [(1, 'q')], labels, prior)
    x = torch.tensor([.3, .7, -.2, .1, -.5], requires_grad=True)
    index = order + [3, 4]; y = x.detach()[index].requires_grad_(True)
    u = candidate_supervised_loss(x, a); v = candidate_supervised_loss(y, b)
    u.backward(); v.backward()
    assert torch.equal(u, v) and torch.equal(x.grad[index], y.grad)
