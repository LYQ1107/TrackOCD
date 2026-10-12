import pytest
import torch
from src.trackocd_cvpr.reliability_training_loss import (
    prediction_owned_support_targets, support_agreement_plan, support_agreement_losses)
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs


def test_soft_other_group_targets_no_oracle_majority_or_positive_singleton():
    labels = (('Known', 0), ('Known', 0), ('policy_pseudo', 1))
    plan = support_agreement_plan(labels)
    assert plan.targets == (.5, .5, 0.)
    logits = torch.zeros(3, requires_grad=True)
    terms = support_agreement_losses(logits, plan)
    assert len(terms) == 3
    sum(loss for _, loss in terms).backward()
    assert logits.grad[2] > 0 and logits.grad[0] == logits.grad[1] == 0
    one = support_agreement_plan(labels[:1])
    assert one.targets == (None,) and support_agreement_losses(logits[:1], one) == ()


def test_unknown_peer_masks_whole_support_not_background_or_negative():
    plan = support_agreement_plan((('Known', 0), None, ('Known', 0)))
    assert plan.targets == (None,) * 3
    assert support_agreement_losses(torch.zeros(3), plan) == ()
    for label in (('Novel', 17), ('background', 0), ('Known', -1)):
        with pytest.raises(ValueError): support_agreement_plan((label, ('Known', 0)))


def test_actual_wrong_predicted_commits_used_without_GT_repair_alias_conflict_masked():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a'), obs('alias'), obs('wrong', 1)]), ['S:0', 'S:0'])
    saved = bank.commit_history()
    targets = {(77, 'a'): ('Known', 0), (77, 'alias'): ('Known', 0), (77, 'wrong'): ('policy_pseudo', 1)}
    labels = prediction_owned_support_targets(bank, 'S:0', targets, {0: 77})
    assert set(labels) == {('Known', 0), ('policy_pseudo', 1)}
    assert support_agreement_plan(labels).targets == (0., 0.)
    targets[(77, 'alias')] = ('Known', 2)
    labels = prediction_owned_support_targets(bank, 'S:0', targets, {0: 77})
    assert None in labels and support_agreement_plan(labels).targets == (None, None)
    assert bank.commit_history() == saved


def test_current_support_targets_do_not_claim_to_clean_strict_historical_lineage():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a'), obs('b', 1)]), ['S:0', 'S:0'])
    original = bank.commit_history()
    bank.apply_frame(router.ingest(1, [obs('a', 2)]), ['S:1'])
    targets = {(3, 'a'): ('Known', 0), (3, 'b'): ('policy_pseudo', 1)}
    assert prediction_owned_support_targets(bank, 'S:0', targets, {0: 3}) == (('policy_pseudo', 1),)
    assert bank.commit_history()[:2] == original and len(bank.commit_history()) == 3
    assert support_agreement_plan((('policy_pseudo', 1),)).targets == (None,)


def test_identical_payload_different_video_conflicting_truth_is_conservatively_masked():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    bank.start_next_video(); router = CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    targets = {(3, 'a'): ('Known', 0), (4, 'a'): ('policy_pseudo', 1)}
    assert prediction_owned_support_targets(bank, 'S:0', targets, {0: 3, 1: 4}) == (None, None)
