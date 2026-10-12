import pytest
import torch
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix, reencode_supports
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.differentiable_candidate_features import candidate_tensors
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP


def prefix(seed, count=3):
    visual = torch.randn(count, 768, generator=torch.Generator().manual_seed(seed))
    return VisiblePrefix(visual, torch.tensor([[0., 0., 1., 1.]]).repeat(count, 1),
                         torch.ones(count), torch.arange(count, dtype=torch.int64))


def test_identity_start_group_means_and_visible_stats_no_GT():
    model = RawPreservingEvidence()
    groups = [[prefix(1), prefix(2)], [prefix(3)]]
    raw, adapted, stats = reencode_supports(model, groups, torch.tensor([1., 0.]), torch.ones(2))
    torch.testing.assert_close(raw, adapted, rtol=0, atol=0)
    assert stats.shape == (2, 5) and stats[:, 2].tolist() == [2., 1.]
    assert stats[:, 3].tolist() == [1., 1.]


def test_actual_joint_R_C_backward_through_query_and_reencoded_candidate_support():
    model, scorer = RawPreservingEvidence(), CandidateScoreMLP(feature_count=14)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    scorer_before = {k: v.clone() for k, v in scorer.state_dict().items()}
    a, b, q = prefix(1), prefix(2), prefix(3)
    # Input leaf gradients expose both branches, not just shared parameter
    # gradients that could originate entirely from the query branch.
    a.visual.requires_grad_(); b.visual.requires_grad_(); q.visual.requires_grad_()
    raw, adapted, stats = reencode_supports(model, [[a], [b]], torch.tensor([1., 0.]), torch.ones(2))
    adapted.retain_grad()
    query = model(q.visual, q.boxes, q.quality, q.elapsed)
    query.embedding.retain_grad()
    rows, context = candidate_tensors(query, raw, adapted, stats)
    logits = scorer(rows, context)
    loss = torch.nn.functional.cross_entropy(logits.reshape(1, -1), torch.tensor([0]))
    loss.backward()
    assert all(p.visual.grad is not None and p.visual.grad.norm() > 0 for p in (a, b, q))
    assert model.residual[-1].weight.grad.norm() > 0
    assert adapted.grad.norm() > 0 and query.embedding.grad.norm() > 0
    assert scorer.rows[0].weight.grad.norm() > 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, scorer.state_dict()[k]) for k, v in scorer_before.items())


def test_reencoding_uses_current_weights_not_stale_adapted_prototype():
    model = RawPreservingEvidence()
    groups = [[prefix(1)]]
    first = reencode_supports(model, groups, torch.ones(1), torch.ones(1))
    with torch.no_grad():
        model.residual[-1].bias[0] = 0.2  # Synthetic weight mutation, not an optimizer/fit.
    second = reencode_supports(model, groups, torch.ones(1), torch.ones(1))
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])


def test_snapshot_mutation_after_forward_does_not_corrupt_saved_backward_inputs():
    model, a = RawPreservingEvidence(), prefix(1)
    _, adapted, _ = reencode_supports(model, [[a]], torch.ones(1), torch.ones(1))
    with torch.no_grad():
        a.visual.mul_(-1)
    adapted.sum().backward()
    assert model.residual[-1].weight.grad is not None


def test_empty_candidates_and_unsupported_empty_member_group():
    model = RawPreservingEvidence()
    raw, adapted, stats = reencode_supports(model, [], torch.empty(0), torch.empty(0))
    assert raw.shape == adapted.shape == (0, 768) and stats.shape == (0, 5)
    with pytest.raises(ValueError):
        reencode_supports(model, [[]], torch.ones(1), torch.ones(1))


def test_extra_semantic_fields_not_a_numeric_visible_prefix():
    with pytest.raises(ValueError):
        reencode_supports(RawPreservingEvidence(), [[{'category_id': 1}]], torch.ones(1), torch.ones(1))


def test_budget_refuses_whole_call_without_silent_support_or_observation_drop():
    a = prefix(1, count=1)
    oversized = VisiblePrefix(a.visual, a.boxes, torch.ones(16385), a.elapsed)
    with pytest.raises(ValueError, match='bound'):
        reencode_supports(RawPreservingEvidence(), [[oversized]], torch.ones(1), torch.ones(1))
