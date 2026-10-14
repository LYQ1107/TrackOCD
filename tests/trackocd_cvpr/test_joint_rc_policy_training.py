import numpy as np
import torch

from src.trackocd_cvpr.atomic_member_control import MemberFrameResult
from src.trackocd_cvpr.causal_observation_router import CurrentObservation
from src.trackocd_cvpr.candidate_training_loss import candidate_supervised_loss, prediction_owned_loss_plan
from src.trackocd_cvpr.joint_rc_policy_training import (
    JointAtomicMemberControl,
    prefix_view_to_tensor,
    visible_payload_to_prefix,
)
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_core.features import prefix_view
from src.trackocd_cvpr.causal_observation_router import visible_bytes


def prefix(seed=0, count=2):
    generator = torch.Generator().manual_seed(seed)
    visual = torch.randn(count, 768, generator=generator)
    boxes = torch.tensor([[0.1, 0.1, 0.6, 0.7]], dtype=torch.float32).repeat(count, 1)
    quality = torch.ones(count)
    elapsed = torch.arange(count, dtype=torch.int64)
    return VisiblePrefix(visual, boxes, quality, elapsed)


def test_bank_payload_roundtrip_keeps_only_numeric_visible_prefix():
    source = prefix_view(np.ones((2, 768), dtype=np.float32),
                         np.array([[.1, .1, .5, .5], [.1, .1, .5, .5]], dtype=np.float32),
                         np.ones(2, dtype=np.float32), np.array([3, 8], dtype=np.int64), 2)
    decoded = visible_payload_to_prefix(visible_bytes(source))
    expected = prefix_view_to_tensor(source)
    for left, right in zip((decoded.visual, decoded.boxes, decoded.quality, decoded.elapsed),
                            (expected.visual, expected.boxes, expected.quality, expected.elapsed)):
        assert torch.equal(left, right)


def test_joint_controller_loss_reaches_R_and_C_after_actual_atomic_decision():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1027)
        evidence = RawPreservingEvidence()
        scorer = CandidateScoreMLP(feature_count=14)
        support = prefix(3, 2)
        with torch.inference_mode():
            known = evidence(support.visual, support.boxes, support.quality, support.elapsed).raw.numpy()
        controller = JointAtomicMemberControl(evidence, scorer, {0: known}, ((support,),), K=8)
        observation = CurrentObservation('opaque-route', support.visual[0].numpy(),
                                         support.boxes[0].numpy(), 1.)
        frame = controller.ingest(0, (observation,))
        assert isinstance(frame.result, MemberFrameResult)
        assert len(frame.result.events) == len(frame.logits) == 1
        event = frame.result.events[0]
        plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,
            (('video', 'opaque-route'),), {('video', 'opaque-route'): ('Known', 0)}, {})
        loss = candidate_supervised_loss(frame.logits[0], plan)
        assert loss is not None
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in evidence.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters())
