import pytest
import torch
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix, reencode_supports
from src.trackocd_cvpr.exact_support_reduction import reencode_exact_supports


def prefix(seed, count=3, gradients=False):
    visual = torch.randn(count, 768, generator=torch.Generator().manual_seed(seed)).requires_grad_(gradients)
    return VisiblePrefix(visual, torch.tensor([[0., 0., 1., 1.]]).repeat(count, 1),
                         torch.ones(count), torch.arange(count, dtype=torch.int64))


def test_complete_equal_member_values_and_gradients_match_unpartitioned_reference():
    torch.manual_seed(1701)
    original, streamed = RawPreservingEvidence(), RawPreservingEvidence()
    with torch.no_grad(): original.residual[-1].weight.normal_(0, 0.005)
    streamed.load_state_dict(original.state_dict())
    a = [prefix(s, gradients=True) for s in (1, 2, 3, 4)]
    b = [prefix(s, gradients=True) for s in (1, 2, 3, 4)]
    reference = reencode_supports(original, [a[:3], a[3:]], torch.tensor([1., 0.]), torch.ones(2))
    actual = reencode_exact_supports(streamed, [b[:3], b[3:]], torch.tensor([1., 0.]), torch.ones(2), chunk_observations=3)
    for first, second in zip(reference, actual[:3]): torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)
    target = torch.arange(768).float() / 768
    (reference[1] * target).sum().backward(); (actual[1] * target).sum().backward()
    for x, y in zip(a, b):
        assert x.visual.grad.norm() > 0 and y.visual.grad.norm() > 0
        torch.testing.assert_close(x.visual.grad, y.visual.grad, atol=1e-6, rtol=1e-5)
    for x, y in zip(original.parameters(), streamed.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    assert actual[3]['whole_member_chunks'] == 4


def test_complete_call_can_exceed_legacy_bound_without_drop_or_zero_support_gradients():
    model = RawPreservingEvidence()
    a, b = prefix(7, 8200, True), prefix(8, 8200, True)
    with pytest.raises(ValueError, match='bound'):
        reencode_supports(model, [[a, b]], torch.ones(1), torch.ones(1))
    raw, adapted, stats, proof = reencode_exact_supports(model, [[a, b]], torch.ones(1), torch.ones(1))
    assert proof['observations_reencoded'] == 16400 and proof['groups_reencoded'] == 2
    assert proof['whole_member_chunks'] == 2 and stats[0, 2] == 2
    torch.testing.assert_close(raw, adapted, atol=0, rtol=0)
    adapted[:, 0].sum().backward()
    assert a.visual.grad.norm() > 0 and b.visual.grad.norm() > 0
    assert model.residual[-1].weight.grad.norm() > 0


def test_no_member_split_silent_drop_overbudget_or_semantic_argument():
    model = RawPreservingEvidence()
    with pytest.raises(ValueError, match='Whole temporal'):
        reencode_exact_supports(model, [[prefix(1)]], torch.ones(1), torch.ones(1), chunk_observations=2)
    a = prefix(1, 100)
    with pytest.raises(ValueError, match='Whole-call'):
        reencode_exact_supports(model, [[a] * 656], torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match='Nonempty'):
        reencode_exact_supports(model, [[]], torch.ones(1), torch.ones(1))
    with pytest.raises(ValueError, match='Numeric visible'):
        reencode_exact_supports(model, [[{'category_id': 99}]], torch.ones(1), torch.ones(1))


def test_empty_candidates_and_fresh_weight_binding_not_adapted_cache():
    model = RawPreservingEvidence(); a = prefix(1)
    empty = reencode_exact_supports(model, [], torch.empty(0), torch.empty(0))
    assert empty[0].shape == (0, 768) and empty[3]['observations_reencoded'] == 0
    first = reencode_exact_supports(model, [[a]], torch.ones(1), torch.ones(1))
    with torch.no_grad(): model.residual[-1].bias[0] = 0.2
    second = reencode_exact_supports(model, [[a]], torch.ones(1), torch.ones(1))
    assert torch.equal(first[0], second[0]) and not torch.equal(first[1], second[1])
