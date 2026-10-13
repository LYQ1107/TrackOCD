"""Synthetic exactness/runtime-boundary tests, not actual policy qualification."""
import hashlib
import numpy as np
import pytest
import torch
from src.trackocd_cvpr import member_reliability as member
from src.trackocd_cvpr.frozen_member_feature_cache_scope import (
    FrozenMemberFeaturesCache, RefreshSnapshotIndex, frozen_member_feature_cache_scope, ORIGINAL_FEATURES)
from src.trackocd_cvpr.prefix_member_bank import CommittedRawSupport, PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.versioned_reliability_memory import VersionedReliabilityMemory
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.frame_atomic_state_ema import StateEMAMemberControl
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse


@pytest.mark.parametrize('n', [1, 2, 8, 512])
def test_complete_raw_quality_count_key_exact_miss_hit_and_fresh_outputs(n):
    generator = torch.Generator().manual_seed(1027)
    raw = torch.randn(n, 768, generator=generator)
    quality, counts = torch.linspace(0, 1, n), torch.arange(1, n + 1).float()
    if n == 512: counts = torch.ones(n)
    cache = FrozenMemberFeaturesCache()
    with torch.inference_mode():
        expected = ORIGINAL_FEATURES(raw, quality, counts)
        first = cache.features(raw, quality, counts)
        first[0].fill_(123); first[1].zero_()
        second = cache.features(raw, quality, counts)
        assert all(torch.equal(a, b) for a, b in zip(expected, second))
        assert cache.hits == cache.misses == 1
        for changed in ((raw + .01, quality, counts), (raw, quality * .5, counts), (raw, quality, counts + 1)):
            actual = cache.features(*changed)
            assert all(torch.equal(a, b) for a, b in zip(ORIGINAL_FEATURES(*changed), actual))
    assert cache.retained_bytes <= cache.limit


def test_eviction_and_uncacheable_complete_calls_recompute_not_prune():
    raw, quality, counts = torch.eye(2, 768), torch.ones(2), torch.ones(2)
    with torch.no_grad():
        cache = FrozenMemberFeaturesCache(max_bytes=13000)
        for value in (raw, raw + .1, raw):
            actual = cache.features(value, quality, counts)
            assert all(torch.equal(a, b) for a, b in zip(actual, ORIGINAL_FEATURES(value, quality, counts)))
            assert cache.retained_bytes <= cache.limit
        tiny = FrozenMemberFeaturesCache(max_bytes=1)
        tiny.features(raw, quality, counts); tiny.features(raw, quality, counts)
        assert tiny.retained_bytes == tiny.hits == 0 and tiny.misses == 2


def test_gradient_and_noncontiguous_paths_keep_original_graph_no_cache():
    cache = FrozenMemberFeaturesCache()
    raw = torch.randn(2, 768, requires_grad=True)
    quality, counts = torch.ones(2), torch.ones(2)
    features, vectors = cache.features(raw, quality, counts)
    (features.sum() + vectors.sum()).backward()
    assert raw.grad is not None and cache.hits == cache.misses == 0
    with torch.no_grad():
        view = torch.randn(768, 2).T
        assert not view.is_contiguous()
        actual = cache.features(view, quality, counts)
        assert all(torch.equal(a, b) for a, b in zip(actual, ORIGINAL_FEATURES(view, quality, counts)))
    assert cache.fallbacks == 2 and cache.retained_bytes == 0


def test_invalid_numeric_inputs_cannot_be_cached_or_silently_repaired():
    cache = FrozenMemberFeaturesCache()
    with torch.no_grad():
        for raw, q, n in ((torch.zeros(1, 768), torch.ones(1), torch.ones(1)),
                (torch.ones(1, 768), torch.ones(1) * 2, torch.ones(1)),
                (torch.ones(1, 768), torch.ones(1), torch.ones(1) * .5),
                (torch.full((1, 768), float('nan')), torch.ones(1), torch.ones(1))):
            with pytest.raises(ValueError): cache.features(raw, q, n)
    assert cache.retained_bytes == cache.hits == 0


def test_snapshot_index_is_immutable_current_refresh_only_not_ID_key_for_model():
    index = RefreshSnapshotIndex()
    snapshot = (CommittedRawSupport('S:0', (b'first',)), CommittedRawSupport('S:1', (b'second',)))
    with index.transaction():
        assert index.select(snapshot, 'S:0') == (b'first',)
        assert index.select(snapshot, 'S:1') == (b'second',)
        assert index.hits == index.misses == 1
        # Fresh snapshot is not trusted by a stale address, even with same token.
        changed = (CommittedRawSupport('S:0', (b'changed',)),)
        assert index.select(changed, 'S:0') == (b'changed',)
        with pytest.raises(ValueError): index.select(changed, 'missing')
    assert index.snapshot is None and index.retained_bytes == 0 and not index.mapping
    assert index.select(snapshot, 'S:0') is None
    tiny = RefreshSnapshotIndex(max_bytes=1)
    with tiny.transaction(): assert tiny.select(snapshot, 'S:0') is None
    with index.transaction():
        with pytest.raises(ValueError): index.select((snapshot[0], snapshot[0]), 'S:0')


@pytest.mark.parametrize('family', ['Mean', 'EMA', 'relative', 'pooled', 'quarantine'])
def test_whole_synthetic_memory_events_commits_versions_alias_transfer_exact(family):
    torch.manual_seed(1027)
    model = member.MemberReliabilityMLP(pooled_control=family == 'pooled')
    with torch.no_grad(): model.rows[-1].weight.fill_(.03)
    def run():
        if family in ('Mean', 'EMA'):
            control = StateEMAMemberControl({}, reuse, momentum=0. if family == 'Mean' else .9)
        else: control = ReliabilityMemberControl({}, reuse, model, quarantine=family == 'quarantine')
        rows = []
        for video in range(2):
            if video: control.start_next_video()
            for frame, inputs in ((0, [obs('a'), obs('alias'), obs('independent', 1)]), (1, [obs('a', 1)]), (2, [])):
                result = control.ingest(frame, inputs)
                rows.append((result, control.memory_versions(), control.chosen_version_bindings(),
                    control.state_snapshot()[0].commit_history()))
        return repr(rows)
    with torch.inference_mode():
        old = run()
        with frozen_member_feature_cache_scope() as (raw, sha, features, index):
            assert run() == old
            assert raw.retained_bytes <= 33554432 and sha.retained_bytes <= 16252928
            assert features.retained_bytes <= 16777216 and index.peak_bytes <= 524288
            assert index.snapshot is None and index.retained_bytes == 0
            assert sha.hits > 0 and index.hits > 0
            if family not in ('Mean', 'EMA'): assert features.hits > 0


def test_feature_hit_never_reuses_logits_after_model_parameter_change():
    bank, router, model = PrefixMemberBank(), CurrentObservationRouter(), member.MemberReliabilityMLP()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    memory = VersionedReliabilityMemory()
    with torch.inference_mode(), frozen_member_feature_cache_scope() as (_, _, features, index):
        _, first = memory.refresh(bank, model)
        prior = first['S:0'].logits.clone()
        model.rows[-1].bias.fill_(1.)
        _, second = memory.refresh(bank, model)
        assert features.hits > 0 and not torch.equal(prior, second['S:0'].logits)
        assert memory.versions()[0].numeric_sha256 != memory.versions()[1].numeric_sha256
        assert index.snapshot is None


def test_own_scope_restores_all_aliases_and_refresh_methods_on_exception():
    from src.trackocd_cvpr import frame_atomic_state_ema as ema, versioned_reliability_memory as reliability
    from src.trackocd_cvpr.reliability_support_bridge import committed_reliability_inputs
    original_M, original_EMA = VersionedReliabilityMemory.refresh, ema.FrameAtomicStateEMA.refresh
    with pytest.raises(RuntimeError):
        with frozen_member_feature_cache_scope():
            assert hashlib.sha256 is not None and member.member_features is not ORIGINAL_FEATURES
            with pytest.raises(ValueError):
                with frozen_member_feature_cache_scope(): pass
            raise RuntimeError('restore all owned bindings')
    assert member.member_features is ORIGINAL_FEATURES
    assert VersionedReliabilityMemory.refresh is original_M and ema.FrameAtomicStateEMA.refresh is original_EMA
    assert ema.hashlib is reliability.hashlib is hashlib
    assert ema.committed_reliability_inputs is reliability.committed_reliability_inputs is committed_reliability_inputs
