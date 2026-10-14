import numpy as np
import pytest
import torch
from src.trackocd_cvpr.member_reliability import (
    MemberReliabilityMLP, member_features, weighted_support)
from src.trackocd_cvpr.reliability_support_bridge import committed_reliability
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter


def inputs():
    raw = torch.zeros(3, 768); raw[:2, 0] = 1; raw[2, 1] = 1
    return raw, torch.tensor([.8, .8, .2]), torch.tensor([2., 2., 1.])


def observation(alias, axis):
    raw = np.zeros(768, np.float32); raw[axis] = 1
    return CurrentObservation(alias, raw, np.array([0., 0., 1., 1.]), .8)


def test_leave_self_out_and_singleton_absence_not_self_certainty():
    raw, quality, count = inputs()
    f, _ = member_features(raw, quality, count)
    torch.testing.assert_close(f[:, 0], torch.tensor([2 ** -.5, 2 ** -.5, 0.]))
    torch.testing.assert_close(f[:, 1], torch.tensor([.5, .5, 0.]))
    one, _ = member_features(raw[:1], quality[:1], count[:1])
    assert one[0, 0] == 0 and one[0, 6] == 1 and one[0, 7] == 0


def test_cancelled_peers_are_distinguished_from_no_peers():
    raw = torch.zeros(3, 768); raw[0, 1] = 1; raw[1, 0] = 1; raw[2, 0] = -1
    f, _ = member_features(raw, torch.ones(3), torch.ones(3))
    assert f[0, 0] == 0 and f[0, 6] == 0 and f[0, 7] == 1 and f[0, 2] == 1


def test_matched_parameter_count_equal_mean_start_and_pooled_control():
    torch.manual_seed(1027)
    model = MemberReliabilityMLP()
    pooled = MemberReliabilityMLP(pooled_control=True)
    pooled.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == 161
    raw, q, n = inputs()
    a, b = weighted_support(model, raw, q, n), weighted_support(pooled, raw, q, n)
    assert torch.equal(a.logits, torch.zeros(3))
    torch.testing.assert_close(a.prototype, raw.sum(0) / raw.sum(0).norm())
    torch.testing.assert_close(a.prototype, b.prototype, atol=0, rtol=0)
    with torch.no_grad():
        model.rows[-1].weight.fill_(.1); pooled.load_state_dict(model.state_dict())
    assert torch.unique(model(a.features)).numel() > 1
    assert torch.unique(pooled(a.features)).numel() == 1


def test_gradient_synthetic_only_no_fit_and_snapshot_input_mutation_safe():
    model = MemberReliabilityMLP(); raw, q, n = inputs()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    result = weighted_support(model, raw, q, n)
    raw.zero_(); q.zero_(); n.zero_()
    torch.nn.functional.binary_cross_entropy_with_logits(
        result.logits, torch.tensor([1., 1., 0.])).backward()
    assert model.rows[-1].weight.grad.norm() > 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())


def test_complete_large_offline_support_group_is_not_pruned():
    generator = torch.Generator().manual_seed(1027)
    raw = torch.randn(867, 768, generator=generator)
    quality = torch.ones(867)
    counts = torch.full((867,), 22., dtype=torch.float32)
    features, vectors = member_features(raw, quality, counts)
    assert features.shape == (867, 8)
    assert vectors.shape == (867, 768)
    assert MemberReliabilityMLP()(features).shape == (867,)


def test_permutation_equivariance_and_zero_resultant_not_fabricated():
    model = MemberReliabilityMLP(); raw, q, n = inputs(); order = torch.tensor([2, 0, 1])
    with torch.no_grad(): model.rows[-1].weight.fill_(.1)
    a, b = weighted_support(model, raw, q, n), weighted_support(model, raw[order], q[order], n[order])
    torch.testing.assert_close(b.features, a.features[order])
    torch.testing.assert_close(b.logits, a.logits[order])
    torch.testing.assert_close(a.prototype, b.prototype)
    pair = torch.zeros(2, 768); pair[:, 0] = torch.tensor([1., -1.])
    result = weighted_support(MemberReliabilityMLP(), pair, torch.ones(2), torch.ones(2))
    assert result.prototype is None and torch.all(result.weights > 0)


def test_actual_bank_alias_dedup_rename_latest_replacement_and_old_snapshot():
    model = MemberReliabilityMLP(); bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [observation('a', 0), observation('alias', 0), observation('b', 1)]), ['S:0', 'S:0'])
    saved = bank.raw_support_snapshot()
    first = committed_reliability(model, saved, 'S:0')
    assert first.features.shape == (2, 8)
    other, other_router = PrefixMemberBank(), CurrentObservationRouter()
    other.apply_frame(other_router.ingest(0, [observation('999', 1), observation('renamed', 0), observation('copy', 0)]), ['S:28', 'S:28'])
    renamed = committed_reliability(model, other.raw_support_snapshot(), 'S:28')
    torch.testing.assert_close(first.features, renamed.features, rtol=0, atol=0)
    bank.apply_frame(router.ingest(1, [observation('a', 2), observation('alias', 2)]), ['S:0'])
    latest = committed_reliability(model, bank.raw_support_snapshot(), 'S:0')
    assert latest.features.shape == (2, 8) and not torch.equal(latest.features, first.features)
    torch.testing.assert_close(first.features, committed_reliability(model, saved, 'S:0').features, rtol=0, atol=0)


@pytest.mark.parametrize('case', ['zero', 'nan', 'quality', 'count', 'bound'])
def test_invalid_numeric_input_refuses_whole_call(case):
    raw, q, n = inputs()
    if case == 'zero': raw[0].zero_()
    if case == 'nan': raw[0, 0] = float('nan')
    if case == 'quality': q[0] = 1.1
    if case == 'count': n[0] = 1.5
    if case == 'bound': n[0] = 65537
    with pytest.raises(ValueError): weighted_support(MemberReliabilityMLP(), raw, q, n)


def test_missing_empty_duplicate_snapshot_bindings_are_not_silently_filtered():
    from src.trackocd_cvpr.prefix_member_bank import CommittedRawSupport
    model = MemberReliabilityMLP()
    for snapshot, token in (((), 'S:0'), ((CommittedRawSupport('S:0', ()),), 'S:0'),
                            ((CommittedRawSupport('S:0', ()),) * 2, 'S:0')):
        with pytest.raises(ValueError): committed_reliability(model, snapshot, token)
