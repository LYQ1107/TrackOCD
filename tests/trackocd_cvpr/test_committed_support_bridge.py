from dataclasses import replace
import struct
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.committed_support_bridge import reencode_committed_supports
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.differentiable_candidate_features import candidate_tensors
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP


def observation(identity, axis):
    v = np.zeros(768, dtype=np.float32); v[axis] = 1
    return CurrentObservation(identity, v, np.array([0., 0., 1., 1.]), .8)


def bank_pair():
    bank, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [observation('physical-a', 0), observation('alias-a', 0),
                              observation('physical-b', 1)])
    # Router orders complete visible bytes, not the supplied route order.
    bank.apply_frame(batch, ['S:0' if view.visual[0, 0] == 1 else 'S:1'
                             for view in batch.model_views])
    return bank, router


def test_actual_bank_alias_group_and_numeric_R_C_backward_without_optimizer():
    bank, _ = bank_pair()
    snapshot = bank.raw_support_snapshot()
    assert [len(row.payloads) for row in snapshot] == [1, 1]
    model, scorer = RawPreservingEvidence(), CandidateScoreMLP(feature_count=14)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    raw, adapted, stats = reencode_committed_supports(model, snapshot, ['S:0', 'S:1'])
    adapted.retain_grad()
    query_visual = torch.zeros(1, 768); query_visual[0, :2] = 1
    query = model(query_visual, torch.tensor([[0., 0., 1., 1.]]), torch.ones(1), torch.zeros(1, dtype=torch.int64))
    rows, context = candidate_tensors(query, raw, adapted, stats)
    loss = torch.nn.functional.cross_entropy(scorer(rows, context)[None], torch.tensor([0]))
    loss.backward()
    assert adapted.grad.norm() > 0 and model.residual[-1].weight.grad.norm() > 0
    assert scorer.rows[0].weight.grad.norm() > 0
    assert stats[:, 0].tolist() == [0., 0.] and stats[:, 2].tolist() == [1., 1.]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_future_commits_do_not_mutate_snapshot_latest_prefix_replaces_old_vote():
    bank, router = bank_pair()
    model = RawPreservingEvidence()
    saved = bank.raw_support_snapshot()
    old = reencode_committed_supports(model, saved, ['S:0'])[0].clone()
    batch = router.ingest(1, [observation('physical-a', 2), observation('alias-a', 2)])
    bank.apply_frame(batch, ['S:0'])
    new = bank.raw_support_snapshot()
    assert len(new[0].payloads) == 1 and saved != new
    assert torch.equal(old, reencode_committed_supports(model, saved, ['S:0'])[0])
    assert not torch.equal(old, reencode_committed_supports(model, new, ['S:0'])[0])


def test_same_raw_other_video_is_separate_scope_not_globally_deduplicated():
    bank, _ = bank_pair()
    bank.start_next_video()
    router = CurrentObservationRouter()
    bank.apply_frame(router.ingest(0, [observation('new-route', 0)]), ['S:0'])
    _, _, stats = reencode_committed_supports(RawPreservingEvidence(), bank.raw_support_snapshot(), ['S:0'])
    assert stats[0, 2] == 2


def test_snapshot_survives_reassignment_tombstone_cannot_be_encoded():
    bank, router = bank_pair()
    saved = bank.raw_support_snapshot()
    bank.apply_frame(router.ingest(1, [observation('physical-a', 2), observation('alias-a', 2)]), ['S:1'])
    assert not bank.raw_support_snapshot()[0].payloads
    with pytest.raises(ValueError, match='tombstone'):
        reencode_committed_supports(RawPreservingEvidence(), bank.raw_support_snapshot(), ['S:0'])
    assert reencode_committed_supports(RawPreservingEvidence(), saved, ['S:0'])[0].shape == (1, 768)


def test_binding_order_is_external_and_unknown_or_duplicate_binding_refuses():
    bank, _ = bank_pair(); model = RawPreservingEvidence(); snapshot = bank.raw_support_snapshot()
    left = reencode_committed_supports(model, snapshot, ['S:0', 'S:1'])
    right = reencode_committed_supports(model, snapshot, ['S:1', 'S:0'])
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b.flip(0), rtol=0, atol=0)
    for tokens in (['S:999'], ['S:0', 'S:0']):
        with pytest.raises(ValueError):
            reencode_committed_supports(model, snapshot, tokens)


def test_actual_route_and_token_renaming_preserve_numeric_model_inputs():
    left, _ = bank_pair()
    right, router = PrefixMemberBank(), CurrentObservationRouter()
    batch = router.ingest(0, [observation('9999', 1), observation('id-z', 0), observation('id-a', 0)])
    right.apply_frame(batch, ['S:27' if view.visual[0, 0] == 1 else 'S:3' for view in batch.model_views])
    model = RawPreservingEvidence()
    a = reencode_committed_supports(model, left.raw_support_snapshot(), ['S:0', 'S:1'])
    b = reencode_committed_supports(model, right.raw_support_snapshot(), ['S:27', 'S:3'])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_corrupt_payload_and_whole_call_group_limit_refuse_without_bank_mutation():
    bank, _ = bank_pair(); model = RawPreservingEvidence(); snapshot = bank.raw_support_snapshot()
    first = snapshot[0]
    for payloads in ((b'bad',), (first.payloads[0] + b'garbage',),
                     (struct.pack('<Q', 16385),), first.payloads * 4097):
        with pytest.raises(ValueError):
            reencode_committed_supports(model, (replace(first, payloads=payloads),), [first.token])
    assert bank.raw_support_snapshot() == snapshot


def test_empty_selection_and_empty_bank_return_empty_numeric_contract():
    raw, adapted, stats = reencode_committed_supports(RawPreservingEvidence(), (), ())
    assert raw.shape == adapted.shape == (0, 768) and stats.shape == (0, 5)
