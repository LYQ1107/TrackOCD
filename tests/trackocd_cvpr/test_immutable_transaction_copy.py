from copy import deepcopy
import numpy as np
import torch
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.versioned_reliability_memory import VersionedReliabilityMemory
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.frame_atomic_state_ema import FrameAtomicStateEMA
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from tests.trackocd_cvpr.test_versioned_reliability_memory import obs, reuse


def recursive_old_copy(self, memo):
    # Exact prior generic object deepcopy semantics, not a different policy.
    other = type(self).__new__(type(self)); memo[id(self)] = other
    other.__dict__.update(deepcopy(self.__dict__, memo))
    return other


class RecursiveBank(PrefixMemberBank):
    __deepcopy__ = recursive_old_copy


class RecursiveMemory(VersionedReliabilityMemory):
    __deepcopy__ = recursive_old_copy


def test_bank_mutable_containers_isolated_old_support_and_commit_records_immutable():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    clone = deepcopy(bank)
    assert clone._members is not bank._members and clone._tokens is not bank._tokens
    assert clone.commit_history() is bank.commit_history()
    assert next(iter(clone._members.values())) is next(iter(bank._members.values()))
    clone.start_next_video(); clone.apply_frame(CurrentObservationRouter().ingest(0, [obs('b', 1)]), ['S:1'])
    assert len(clone.commit_history()) == 2 and len(bank.commit_history()) == 1
    assert len(bank.state_evidence()) == 1 and len(clone.state_evidence()) == 2


def test_version_and_ema_copy_changes_never_mutate_old_lineage_or_bindings():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [obs('a')]), ['S:0'])
    model = MemberReliabilityMLP()
    for memory in (VersionedReliabilityMemory(), FrameAtomicStateEMA()):
        memory.refresh(bank, model) if isinstance(memory, VersionedReliabilityMemory) else memory.refresh(bank)
        saved = memory.versions(); clone = deepcopy(memory)
        assert clone.versions() is memory.versions() and clone._current is not memory._current
        changed_bank = deepcopy(bank); changed_bank.start_next_video()
        changed_bank.apply_frame(CurrentObservationRouter().ingest(0, [obs('b', 1)]), ['S:0'])
        clone.refresh(changed_bank, model) if isinstance(clone, VersionedReliabilityMemory) else clone.refresh(changed_bank)
        assert memory.versions() == saved and len(clone.versions()) > len(saved)
        assert clone.versions()[:len(saved)] == saved


def test_actual_Raw_rollout_exact_recursive_copy_parity_full_updates_and_moves():
    fast, slow = AtomicMemberControl({}, reuse), AtomicMemberControl({}, reuse)
    slow._bank = RecursiveBank()
    for video in range(3):
        if video: fast.start_next_video(); slow.start_next_video()
        for frame in range(24):
            rows = [obs('a', frame % 4), obs('alias', frame % 4), obs('b', (frame + 1) % 4)]
            assert fast.ingest(frame, rows) == slow.ingest(frame, rows)
            a, na = fast.state_snapshot(); b, nb = slow.state_snapshot()
            assert na == nb and a.commit_history() == b.commit_history()
            assert a.state_evidence() == b.state_evidence() and a.raw_support_snapshot() == b.raw_support_snapshot()


def test_actual_M_rollout_exact_recursive_bank_and_version_copy_parity():
    torch.manual_seed(1027); model = MemberReliabilityMLP()
    fast = ReliabilityMemberControl({}, reuse, model, quarantine=True)
    slow = ReliabilityMemberControl({}, reuse, model, quarantine=True)
    slow._bank, slow._memory = RecursiveBank(), RecursiveMemory(quarantine=True)
    for video in range(2):
        if video: fast.start_next_video(); slow.start_next_video()
        for frame in range(12):
            rows = [obs('a', frame % 3), obs('b', (frame + 1) % 3)]
            assert fast.ingest(frame, rows) == slow.ingest(frame, rows)
            assert fast.memory_versions() == slow.memory_versions()
            assert fast.chosen_version_bindings() == slow.chosen_version_bindings()
            for token in fast.preframe_training_evidence():
                a = fast.preframe_training_evidence()[token]; b = slow.preframe_training_evidence()[token]
                torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
                torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
