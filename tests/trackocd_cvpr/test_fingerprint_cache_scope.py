import hashlib
import pytest
from src.trackocd_cvpr.qualified_fingerprint_cache_scope import ExactSegmentSHA256Cache, _SHA256Proxy, qualified_fingerprint_cache_scope
from src.trackocd_cvpr.qualified_raw_cache_scope import qualified_raw_cache_scope
from src.trackocd_cvpr.frame_atomic_state_ema import StateEMAMemberControl
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse


def test_exact_complete_segment_keys_misses_hits_empty_and_distinct_framing():
    cache = ExactSegmentSHA256Cache(max_bytes=1024)
    for pieces in ((), (b'a', b'bc'), (b'ab', b'c'), (b'', b'abc')):
        expected = hashlib.sha256(b''.join(pieces)).digest()
        assert cache.digest(pieces) == cache.digest(pieces) == expected
    assert cache.hits == cache.misses == 4  # same concatenation need not share cache key
    assert cache.retained_bytes <= 1024
    with pytest.raises(ValueError): cache.digest([bytearray(b'mutable')])


def test_eviction_and_oversize_recompute_without_missing_data_or_digest_change():
    cache = ExactSegmentSHA256Cache(max_bytes=64)
    for pieces in ((b'a',), (b'b',), (b'a',), (b'x' * 1024,), (b'x' * 1024,)):
        assert cache.digest(pieces).hex() == hashlib.sha256(b''.join(pieces)).hexdigest()
    assert cache.misses == 5 and cache.retained_bytes <= 64


def test_deferred_hash_repeated_digest_update_copy_and_initialized_numeric_fallback():
    cache = ExactSegmentSHA256Cache(); proxy = _SHA256Proxy(cache)
    value = proxy.sha256(); value.update(b'a'); copied = value.copy()
    assert value.hexdigest() == value.hexdigest() == hashlib.sha256(b'a').hexdigest()
    value.update(b'b')
    assert value.digest() == hashlib.sha256(b'ab').digest()
    assert copied.digest() == hashlib.sha256(b'a').digest()
    assert type(proxy.sha256(b'numeric-latent')) is type(hashlib.sha256())


@pytest.mark.parametrize('family', ['Mean', 'EMA', 'relative', 'pooled', 'quarantine'])
def test_complete_synthetic_controller_events_versions_and_original_history_exact(family):
    def run():
        if family in ('Mean', 'EMA'):
            control = StateEMAMemberControl({}, reuse, momentum=0. if family == 'Mean' else .9)
        else:
            control = ReliabilityMemberControl({}, reuse, MemberReliabilityMLP(pooled_control=family == 'pooled'),
                quarantine=family == 'quarantine')
        rows = []
        for video in range(2):
            if video: control.start_next_video()
            for frame, obs_rows in ((0, [obs('a'), obs('alias')]), (1, [obs('a', 1)]), (2, [])):
                result = control.ingest(frame, obs_rows)
                rows.append((result.events, control.memory_versions(), control.chosen_version_bindings(),
                    control.state_snapshot()[0].commit_history()))
        return rows
    old = run()
    with qualified_fingerprint_cache_scope() as (raw, fingerprints):
        assert run() == old
        assert raw.retained_bytes + fingerprints.retained_bytes <= 67108864
        if family in ('Mean', 'EMA'): assert fingerprints.hits > 0


def test_exception_restoration_nested_refusal_and_global_hashlib_unchanged():
    from src.trackocd_cvpr import frame_atomic_state_ema as ema
    from src.trackocd_cvpr.reliability_support_bridge import committed_reliability_inputs
    original_hash = hashlib.sha256
    with pytest.raises(RuntimeError):
        with qualified_fingerprint_cache_scope():
            assert hashlib.sha256 is original_hash and ema.hashlib is not hashlib
            with pytest.raises(ValueError):
                with qualified_fingerprint_cache_scope(): pass
            with pytest.raises(ValueError):
                with qualified_raw_cache_scope(): pass
            raise RuntimeError('test exception')
    assert ema.hashlib is hashlib and ema.committed_reliability_inputs is committed_reliability_inputs
    with qualified_raw_cache_scope():
        with pytest.raises(ValueError):
            with qualified_fingerprint_cache_scope(): pass


@pytest.mark.parametrize('momentum', [0., .9])
def test_transfer_tombstone_and_revival_preserve_entire_dirty_numeric_ledger(momentum):
    from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
    from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
    from src.trackocd_cvpr.frame_atomic_state_ema import FrameAtomicStateEMA
    def run():
        bank, router, memory = PrefixMemberBank(), CurrentObservationRouter(), FrameAtomicStateEMA(momentum=momentum)
        batch = router.ingest(0, [obs('a'), obs('b', 1)])
        bank.apply_frame(batch, ['S:0' if view.visual[0, 0] == 1 else 'S:1' for view in batch.model_views])
        first = memory.refresh(bank)
        bank.apply_frame(router.ingest(1, [obs('a', 2)]), ['S:1'])
        moved = memory.refresh(bank)
        assert moved['S:0'].prototype() is None
        assert ('S:0', 0) in memory.versions()[-1].parents
        bank.apply_frame(router.ingest(2, [obs('a', 3)]), ['S:0'])
        revived = memory.refresh(bank)
        return first, moved, revived, memory.versions(), bank.commit_history()
    old = run()
    with qualified_fingerprint_cache_scope(): assert run() == old
