import numpy as np
import pytest
import torch
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.reliability_member_control import ReliabilityMemberControl
from src.trackocd_cvpr.versioned_reliability_memory import VersionedReliabilityMemory
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter


def obs(route, axis=0):
    vector = np.zeros(768, np.float32); vector[axis] = 1
    return CurrentObservation(route, vector, np.array([0., 0., 1., 1.]), .8)


def reuse(features, context):
    scores = np.full(len(features) + 2, -2., np.float32)
    scores[0 if len(features) else -2] = 1
    return scores


def synthetic_model(bias):
    model = MemberReliabilityMLP()
    with torch.no_grad(): model.rows[-1].bias.fill_(bias)
    return model


def test_preframe_visibility_and_postframe_versions_do_not_rewrite_predictions():
    calls = []
    def scorer(f, c): calls.append(len(f)); return reuse(f, c)
    control = ReliabilityMemberControl({}, scorer, MemberReliabilityMLP())
    first = control.ingest(0, [obs('x'), obs('y', 1)])
    assert calls == [0, 0] and all(e.action == 'NEW' for e in first.events)
    saved = control.memory_versions()
    assert len(saved) == 2 and all(v.lifecycle == 'Tentative' for v in saved)
    assert control.chosen_version_bindings() == (None, None)
    control.ingest(1, [obs('x', 2)])
    assert control.memory_versions()[:2] == saved
    assert first.events[0].action == 'NEW'


def test_quarantine_is_numeric_only_recovery_is_prospective_and_support_retained():
    model = synthetic_model(-10)
    control = ReliabilityMemberControl({}, reuse, model, quarantine=True)
    first = control.ingest(0, [obs('a')]); original = control.memory_versions()
    assert original[0].lifecycle == 'Quarantined' and original[0].retained_groups == 1
    second = control.ingest(1, [obs('b')])
    assert second.events[0].action == 'NEW'  # old quarantined token unavailable
    bank, _ = control.state_snapshot()
    assert len(bank.commit_history()) == 2 and len(bank.raw_support_snapshot()) == 2
    with torch.no_grad(): model.rows[-1].bias.fill_(10)
    # Same-direction separate tokens remain ambiguous, not ID-selected recovery.
    third = control.ingest(2, [obs('query')])
    assert third.events[0].action == 'WAIT'
    assert control.memory_versions()[:len(original)] == original
    assert first.events[0].action == 'NEW'


def test_versions_keep_transfer_parent_and_empty_tombstone_history():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [obs('a'), obs('b', 1)])
    bank.apply_frame(batch, ['S:0' if v.visual[0, 0] == 1 else 'S:1' for v in batch.model_views])
    memory = VersionedReliabilityMemory(); memory.refresh(bank, MemberReliabilityMLP())
    saved = memory.versions()
    bank.apply_frame(router.ingest(1, [obs('a', 2)]), ['S:1'])
    memory.refresh(bank, MemberReliabilityMLP())
    latest = {v.token: v for v in memory.versions()}
    assert latest['S:0'].retained_groups == 0 and latest['S:0'].lifecycle == 'Quarantined'
    assert ('S:0', 0) in latest['S:1'].parents and ('S:1', 0) in latest['S:1'].parents
    assert memory.versions()[:len(saved)] == saved and len(bank.commit_history()) == 3


def test_confirmed_requires_multiple_support_groups_not_route_alias_votes():
    control = ReliabilityMemberControl({}, reuse, synthetic_model(10))
    control.ingest(0, [obs('a'), obs('alias')])
    assert control.memory_versions()[-1].retained_groups == 1
    assert control.memory_versions()[-1].lifecycle == 'Tentative'
    control.start_next_video(); result = control.ingest(0, [obs('other-video')])
    assert result.events[0].action == 'EXISTING'
    assert control.memory_versions()[-1].retained_groups == 2
    assert control.memory_versions()[-1].lifecycle == 'Confirmed'
    assert control.chosen_version_bindings()[0] == ('S:0', 0)


def test_numeric_ID_rename_and_reverse_delivery_preserve_events_and_versions():
    left = ReliabilityMemberControl({}, reuse, MemberReliabilityMLP())
    right = ReliabilityMemberControl({}, reuse, MemberReliabilityMLP())
    for frame, a, b in ((0, [obs('a'), obs('b', 1)], [obs('999', 1), obs('renamed')]),
                        (1, [obs('a', 2)], [obs('renamed', 2)])):
        assert left.ingest(frame, a).events == right.ingest(frame, b).events
        assert left.memory_versions() == right.memory_versions()


def test_version_bound_failure_is_transactional_for_entire_frame():
    control = ReliabilityMemberControl({}, reuse, MemberReliabilityMLP(), max_versions=1)
    before = control.state_snapshot()
    with pytest.raises(ValueError, match='Version bound'):
        control.ingest(0, [obs('a'), obs('b', 1)])
    assert control.state_snapshot()[0].commit_history() == before[0].commit_history()
    assert control.state_snapshot()[1] == 0 and control.memory_versions() == ()
    control.ingest(0, [obs('a')])
    assert len(control.memory_versions()) == 1


def test_actual_preframe_memory_logits_have_graph_without_optimizer_or_GT_input():
    model = MemberReliabilityMLP(); control = ReliabilityMemberControl({}, reuse, model)
    control.ingest(0, [obs('a')]); control.start_next_video(); control.ingest(0, [obs('b')])
    control.ingest(1, [])
    value = control.preframe_training_evidence()['S:0']
    assert value.logits.shape == (2,) and value.logits.requires_grad
    value.logits.sum().backward()
    assert model.rows[-1].weight.grad.norm() > 0
    assert len(control.memory_versions()) == 2  # empty frame no invented revision
