from types import SimpleNamespace as Binding
import pytest
import torch
from src.trackocd_cvpr.candidate_training_loss import CandidateLossPlan
from src.trackocd_cvpr.complementary_known_loss import complementary_known_rejection, complementary_candidate_loss


def bindings():
    return ((Binding(kind='KNOWN', identifier=0),), (Binding(kind='EXISTING', identifier='S:0'),))


def masked(family):
    return CandidateLossPlan((False,) * 4, (False,) * 4, 'MASKED_UNCERTAIN_PSEUDO_OR_ABSENT_KNOWN_CANDIDATE', family)


def test_uncertain_policy_query_gets_only_Known_rejection_not_claimed_NEW_or_state_positive():
    value = torch.tensor([2., 0., -.1, -.2], requires_grad=True)
    loss = complementary_candidate_loss(value, masked('policy_pseudo'), bindings())
    loss.backward()
    assert torch.isfinite(loss) and loss > 0 and value.grad[0] > 0
    assert torch.all(value.grad[1:] < 0)  # collective alternatives, no selected state/action teacher
    expected = .25 * (torch.logsumexp(value, 0) - torch.logsumexp(value[1:], 0))
    assert loss.item() == pytest.approx(expected.item())


def test_unknown_and_absent_Known_queries_stay_masked_no_background_or_novel_pseudolabel():
    value = torch.zeros(4, requires_grad=True)
    for family in (None, 'Known'):
        assert complementary_candidate_loss(value, masked(family), bindings()) is None
    with pytest.raises(ValueError, match='authorized'):
        complementary_known_rejection(value, bindings(), 'declared_pseudoNovel')  # dev label prohibited


def test_actual_correct_known_CE_unmodified_and_no_Known_exposure_not_fake_zero_update():
    value = torch.tensor([1., .2, -.1, -.2], requires_grad=True)
    plan = CandidateLossPlan((True,) * 4, (True, False, False, False), 'EXPOSED_RESOLVABLE_CORRECT_CANDIDATE', 'Known')
    loss = complementary_candidate_loss(value, plan, bindings())
    assert loss.item() == pytest.approx((torch.logsumexp(value, 0) - value[0]).item())
    assert complementary_known_rejection(torch.zeros(3), (bindings()[1],), 'policy_pseudo') is None


def test_row_permutation_and_binding_ID_renames_do_not_change_loss_or_gradients():
    original = torch.tensor([2., .4, -.1, -.2], requires_grad=True)
    altered = original.detach()[torch.tensor([1, 0, 2, 3])].clone().requires_grad_(True)
    left = complementary_known_rejection(original, bindings(), 'policy_pseudo')
    renamed = ((Binding(kind='EXISTING', identifier='opaque-new'),), (Binding(kind='KNOWN', identifier=999),))
    right = complementary_known_rejection(altered, renamed, 'policy_pseudo')
    left.backward(); right.backward()
    assert left.item() == right.item()
    assert torch.equal(original.grad, altered.grad[torch.tensor([1, 0, 2, 3])])


def test_nonfinite_misaligned_or_mixed_family_row_never_guesses_supervision():
    with pytest.raises(ValueError): complementary_known_rejection(torch.tensor([float('nan')] * 4), bindings(), 'policy_pseudo')
    with pytest.raises(ValueError): complementary_known_rejection(torch.zeros(3), bindings(), 'policy_pseudo')
    with pytest.raises(ValueError): complementary_known_rejection(torch.zeros(3), (bindings()[0] + bindings()[1],), 'policy_pseudo')
    with pytest.raises(ValueError): complementary_candidate_loss(torch.zeros(4), masked('policy_pseudo'), bindings(), coefficient=1.)
