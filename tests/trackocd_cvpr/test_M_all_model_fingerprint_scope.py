import hashlib
import json
from pathlib import Path
import pytest
import torch
from src.trackocd_cvpr.qualified_M_fingerprint_cache_scope import qualified_M_fingerprint_cache_scope, _SupportOnlySHA256Proxy
from src.trackocd_cvpr.qualified_fingerprint_cache_scope import ExactSegmentSHA256Cache, qualified_fingerprint_cache_scope
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.versioned_reliability_memory import VersionedReliabilityMemory
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.frame_atomic_state_ema import StateEMAMemberControl
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse
from scripts.trackocd_cvpr.shadow_M_all_model_fingerprints import CONFIG, verify_profile_configuration

ROOT = Path(__file__).resolve().parents[2]


def test_only_omitted_constructor_is_deferred_all_initialized_numeric_hashes_original():
    cache = ExactSegmentSHA256Cache(); proxy = _SupportOnlySHA256Proxy(cache)
    for data in (b'', b'learned-numeric', bytearray(b'mutable-numeric')):
        value = proxy.sha256(data)
        assert type(value) is type(hashlib.sha256())
        assert value.digest() == hashlib.sha256(data).digest()
    assert type(proxy.sha256(usedforsecurity=False)) is type(hashlib.sha256())
    assert cache.hits == cache.misses == 0
    value = proxy.sha256(); value.update(b'Raw'); value.update(b'support')
    assert value.hexdigest() == value.hexdigest() == hashlib.sha256(b'Rawsupport').hexdigest()
    assert cache.hits == cache.misses == 1


@pytest.mark.parametrize('family', ['Mean', 'EMA', 'relative', 'pooled', 'quarantine'])
def test_whole_synthetic_events_dirty_ledger_and_commit_history_exact_all_families(family):
    torch.manual_seed(1027); model = MemberReliabilityMLP(pooled_control=family == 'pooled')
    with torch.no_grad(): model.rows[-1].weight.fill_(.03)
    def run():
        if family in ('Mean', 'EMA'):
            control = StateEMAMemberControl({}, reuse, momentum=0. if family == 'Mean' else .9)
        else:
            control = ReliabilityMemberControl({}, reuse, model, quarantine=family == 'quarantine')
        rows = []
        for video in range(2):
            if video: control.start_next_video()
            for frame, observed in ((0, [obs('a'), obs('alias')]), (1, [obs('a', 1)]), (2, [])):
                result = control.ingest(frame, observed)
                rows.append((result.events, control.memory_versions(), control.chosen_version_bindings(),
                    control.state_snapshot()[0].commit_history()))
        return rows
    old = run()
    with qualified_M_fingerprint_cache_scope() as (raw, fingerprints):
        assert run() == old and fingerprints.hits > 0
        assert raw.retained_bytes <= 33554432 and fingerprints.retained_bytes <= 33554432


@pytest.mark.parametrize('quarantine', [False, True])
def test_transfer_empty_tombstone_revival_and_parent_history_all_exact(quarantine):
    torch.manual_seed(1028); model = MemberReliabilityMLP()
    with torch.no_grad(): model.rows[-1].bias.fill_(-.4)
    def run():
        bank, router = PrefixMemberBank(), CurrentObservationRouter()
        memory = VersionedReliabilityMemory(quarantine=quarantine)
        batch = router.ingest(0, [obs('a'), obs('b', 1)])
        bank.apply_frame(batch, ['S:0' if view.visual[0, 0] == 1 else 'S:1' for view in batch.model_views])
        memory.refresh(bank, model)
        saved = memory.versions()
        bank.apply_frame(router.ingest(1, [obs('a', 2)]), ['S:1']); memory.refresh(bank, model)
        assert {v.token:v for v in memory.versions()}['S:0'].retained_groups == 0
        bank.apply_frame(router.ingest(2, [obs('a', 3)]), ['S:0']); memory.refresh(bank, model)
        assert memory.versions()[:len(saved)] == saved
        return memory.versions(), bank.commit_history()
    old = run()
    with qualified_M_fingerprint_cache_scope(): assert run() == old


def test_raw_fingerprint_hit_never_reuses_model_outputs_after_parameter_change():
    bank, router, model = PrefixMemberBank(), CurrentObservationRouter(), MemberReliabilityMLP()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    memory = VersionedReliabilityMemory()
    with qualified_M_fingerprint_cache_scope() as (_, fingerprints):
        _, first = memory.refresh(bank, model); old = first['S:0'].logits.detach().clone()
        with torch.no_grad(): model.rows[-1].bias.fill_(1.)
        _, second = memory.refresh(bank, model)
        assert fingerprints.hits > 0 and not torch.equal(old, second['S:0'].logits)
        assert memory.versions()[0].support_sha256 == memory.versions()[1].support_sha256
        assert memory.versions()[0].numeric_sha256 != memory.versions()[1].numeric_sha256


def test_own_aliases_restored_on_exception_global_hash_and_nested_scopes_refused():
    from src.trackocd_cvpr import versioned_reliability_memory as reliability, frame_atomic_state_ema as ema
    original = hashlib.sha256
    with pytest.raises(RuntimeError):
        with qualified_M_fingerprint_cache_scope():
            assert hashlib.sha256 is original and reliability.hashlib is not hashlib and ema.hashlib is not hashlib
            with pytest.raises(ValueError):
                with qualified_M_fingerprint_cache_scope(): pass
            with pytest.raises(ValueError):
                with qualified_fingerprint_cache_scope(): pass
            raise RuntimeError('exception restoration')
    assert reliability.hashlib is hashlib and ema.hashlib is hashlib
    with qualified_fingerprint_cache_scope():
        with pytest.raises(ValueError):
            with qualified_M_fingerprint_cache_scope(): pass


def test_registered_entire_all11_profile_requires_actual_completed_M_profile():
    cfg = json.loads((ROOT / CONFIG).read_text())
    assert verify_profile_configuration(ROOT, cfg)['actual_entire_ALL11_fingerprint_shadow_profile_checked']
    for change in ({'wall_seconds_ceiling':6000}, {'training_seeds':[1027]}, {'order_seed':1028},
                   {'output_directory':'outputs/old'}, {'profile_evidence_sha256':'0'*64}):
        with pytest.raises(ValueError): verify_profile_configuration(ROOT, {**cfg, **change})
