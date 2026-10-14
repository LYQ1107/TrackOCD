import torch

from src.trackocd_cvpr.atomic_member_control import MemberControlEvent
from src.trackocd_cvpr.candidate_training_loss import candidate_supervised_loss, prediction_owned_loss_plan
from src.trackocd_cvpr.causal_observation_router import CurrentObservation
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix
from src.trackocd_cvpr.joint_rcm_policy_training import JointRCMAtomicMemberControl
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence


def prefix(seed=0, count=2):
    generator = torch.Generator().manual_seed(seed)
    visual = torch.randn(count, 768, generator=generator)
    boxes = torch.tensor([[.1, .1, .6, .7]], dtype=torch.float32).repeat(count, 1)
    return VisiblePrefix(visual, boxes, torch.ones(count), torch.arange(count, dtype=torch.int64))


def test_joint_rcm_candidate_loss_reaches_all_three_modules_after_real_atomic_decision():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1027)
        evidence = RawPreservingEvidence()
        scorer = CandidateScoreMLP(feature_count=14)
        member = MemberReliabilityMLP()
        support = prefix(3)
        with torch.inference_mode():
            known = evidence(support.visual, support.boxes, support.quality, support.elapsed).raw.numpy()
        control = JointRCMAtomicMemberControl(evidence, scorer, member, {0: known}, ((support,),), K=8)
        row = CurrentObservation('opaque-route', support.visual[0].numpy(), support.boxes[0].numpy(), 1.)
        result, logits = control.ingest(0, (row,))
        event = result.events[0]
        plan = prediction_owned_loss_plan(event.candidate_bindings_for_external_loss,
            (('video', 'opaque-route'),), {('video', 'opaque-route'): ('Known', 0)}, {})
        loss = candidate_supervised_loss(logits[0], plan)
        assert loss is not None
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in evidence.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in member.parameters())

