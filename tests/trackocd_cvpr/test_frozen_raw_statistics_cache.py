import numpy as np
import pytest
import torch
from src.trackocd_core.features import prefix_view
from src.trackocd_cvpr.causal_observation_router import visible_bytes
from src.trackocd_cvpr.prefix_member_bank import CommittedRawSupport
from src.trackocd_cvpr.reliability_support_bridge import committed_reliability_inputs
from src.trackocd_cvpr.frozen_raw_statistics_cache import FrozenRawStatisticsCache


def payload(seed=3, quality=1., length=4):
    rng = np.random.default_rng(seed); x = rng.normal(size=(length, 768)).astype(np.float32)
    return visible_bytes(prefix_view(x, np.tile(np.array([[0., 0., 1., 1.]], np.float32), (length, 1)),
                                    np.full(length, quality, np.float32), np.arange(length, dtype=np.int64), length))


@pytest.mark.parametrize('q', [1., 0., .35])
def test_exact_original_CPU_bytes_and_tensor_values_with_repeated_scope_rows(q):
    groups = (payload(3, q), payload(4, q), payload(3, q))
    old = committed_reliability_inputs((CommittedRawSupport('S:0', groups),), 'S:0')
    cache = FrozenRawStatisticsCache(); first = cache.inputs(groups); cached = cache.inputs(groups)
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(old, first, cached))
    assert len(first[0]) == 3  # identical payload in different scopes is not collapsed
    first[0].zero_(); first[1].zero_()
    assert all(torch.equal(a, b) for a, b in zip(old, cache.inputs(groups)))


def test_derived_cache_eviction_or_oversize_is_exact_recompute_not_member_drop():
    groups = (payload(3), payload(4))
    old = committed_reliability_inputs((CommittedRawSupport('S:0', groups),), 'S:0')
    cache = FrozenRawStatisticsCache(max_bytes=1)
    assert all(torch.equal(a, b) for a, b in zip(old, cache.inputs(groups)))
    assert cache.retained_bytes == 0


def test_bad_raw_or_whole_support_bound_is_refused_not_best_subset():
    cache = FrozenRawStatisticsCache()
    with pytest.raises(ValueError): cache.inputs((payload(),) * 513)
    with pytest.raises(ValueError): cache.get(b'bad')
    with pytest.raises(ValueError): FrozenRawStatisticsCache(max_bytes=0)
