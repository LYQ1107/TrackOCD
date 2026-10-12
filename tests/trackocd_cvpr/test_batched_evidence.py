import pytest
import torch
from src.trackocd_cvpr.batched_evidence import pack_visible, batched_evidence, TimelessStaticEvidence, reencode_batched_supports
from src.trackocd_cvpr.exact_support_reduction import reencode_exact_supports
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix


def prefix(seed, count, grad=False):
    rng = torch.Generator().manual_seed(seed)
    v = torch.randn(count, 768, generator=rng).requires_grad_(grad)
    lower = torch.rand(count, 2, generator=rng)
    b = torch.cat((lower, lower + torch.rand(count, 2, generator=rng) + 0.1), 1)
    q = torch.rand(count, generator=rng)
    return VisiblePrefix(v, b, q, torch.arange(count, dtype=torch.int64) * 3)


@pytest.mark.parametrize('temporal', [True, False])
def test_ragged_values_and_parameter_input_gradients_match_original_per_member(temporal):
    torch.manual_seed(33)
    ref = RawPreservingEvidence(temporal=temporal); batch = RawPreservingEvidence(temporal=temporal)
    with torch.no_grad():
        ref.residual[-1].weight.normal_(0, 0.005); ref.gates[-1].weight.normal_(0, 0.01)
    batch.load_state_dict(ref.state_dict())
    a = [prefix(i, n, True) for i, n in enumerate((1, 2, 7, 19))]
    b = [prefix(i, n, True) for i, n in enumerate((1, 2, 7, 19))]
    expected = [ref(p.visual, p.boxes, p.quality, p.elapsed) for p in a]
    actual = batched_evidence(batch, pack_visible(b))
    for i, row in enumerate(expected):
        for field in row.__dataclass_fields__:
            torch.testing.assert_close(getattr(row, field), getattr(actual.member(i), field), atol=2e-6, rtol=2e-5)
    target = torch.arange(768).float() / 768
    sum((r.embedding * target).sum() for r in expected).backward()
    (actual.embedding * target).sum().backward()
    for x, y in zip(a, b): torch.testing.assert_close(x.visual.grad, y.visual.grad, atol=2e-6, rtol=2e-5)
    for x, y in zip(ref.parameters(), batch.parameters()): torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)


def test_zero_quality_singleton_and_underflow_fallback():
    model = RawPreservingEvidence(); rows = [prefix(1, 1), prefix(2, 3)]
    rows[0].quality.zero_(); rows[1].quality[:] = torch.tensor([0.1, 0., 0.9])
    with torch.no_grad(): model.gates[-1].bias.fill_(-1e38)
    actual = batched_evidence(model, pack_visible(rows))
    for i, p in enumerate(rows):
        ref = model(p.visual, p.boxes, p.quality, p.elapsed)
        torch.testing.assert_close(actual.member(i).visible_weights, ref.visible_weights, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(actual.member(i).embedding, ref.embedding, atol=1e-6, rtol=1e-5)


def test_true_timeless_same_parameters_payload_permutation_and_elapsed_invariance():
    model = TimelessStaticEvidence(); p = prefix(4, 7)
    assert sum(t.numel() for t in model.parameters()) == 50113
    with torch.no_grad():
        model.residual[-1].weight.normal_(0, 0.01); model.gates[-1].weight.normal_(0, 0.05)
    permutation = torch.tensor([6, 1, 3, 0, 2, 5, 4])
    first = model(p.visual, p.boxes, p.quality, p.elapsed)
    second = model(p.visual[permutation], p.boxes[permutation], p.quality[permutation], p.elapsed * 99)
    assert torch.equal(first.embedding, second.embedding) and torch.equal(first.raw, second.raw)
    assert first.log_elapsed == second.log_elapsed == 0
    # Canonical summation differs from noncanonical torch.sum by ~1e-8.
    torch.testing.assert_close(first.visible_weights, p.quality / p.quality.sum(), atol=2e-7, rtol=2e-6)
    assert torch.equal(second.visible_weights, first.visible_weights[permutation])


def test_no_semantic_fields_padding_or_future_chronology_repair():
    with pytest.raises(ValueError): pack_visible([{'category_id': 1}])
    p = prefix(1, 3); p.elapsed[-1] = 0
    with pytest.raises(ValueError): batched_evidence(RawPreservingEvidence(), pack_visible([p]))


def test_complete_batched_candidate_prototypes_gradients_and_member_teacher_match_reference():
    torch.manual_seed(14)
    reference = RawPreservingEvidence(); batch = RawPreservingEvidence()
    with torch.no_grad(): reference.residual[-1].weight.normal_(0, 0.005)
    batch.load_state_dict(reference.state_dict())
    a = [[prefix(1, 3, True), prefix(2, 19, True)], [prefix(3, 1, True)]]
    b = [[prefix(1, 3, True), prefix(2, 19, True)], [prefix(3, 1, True)]]
    expected = reencode_exact_supports(reference, a, torch.tensor([1., 0.]), torch.ones(2))
    actual = reencode_batched_supports(batch, b, torch.tensor([1., 0.]), torch.ones(2))
    for x, y in zip(expected[:3], actual[:3]): torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
    expected[1][:, 0].sum().backward(); actual[1][:, 0].sum().backward()
    for ga, gb in zip(a, b):
        for x, y in zip(ga, gb): torch.testing.assert_close(x.visual.grad, y.visual.grad, atol=2e-6, rtol=2e-5)
    assert actual[3].embedding.shape == (3, 768) and actual[3].embedding.requires_grad
