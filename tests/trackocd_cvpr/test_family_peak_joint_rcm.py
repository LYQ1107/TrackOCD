import json

import torch

from scripts.trackocd_cvpr.pilot_family_peak_joint_rcm import (
    FamilyPeakResidualScorer,
    ROOT,
)


def test_family_peak_residual_keeps_fixed_action_and_real_scorer_gradient():
    scorer = FamilyPeakResidualScorer()
    features = torch.tensor([
        [0.80, 0.0, 0.0, 0.1, 0.7, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.1],
        [0.20, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.01, 0.1],
    ], dtype=torch.float32)
    context = torch.zeros(6, dtype=torch.float32)
    logits = scorer(features, context)
    assert logits.shape == (4,)
    assert int(torch.argmax(logits)) == 0
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in scorer.parameters())
    assert sum(p.numel() for p in scorer.parameters()) == 804


def test_family_peak_joint_config_is_frozen_and_not_acceptance():
    path = ROOT / "configs/trackocd_cvpr/family_peak_joint_rcm_training_pilot_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["pilot_policy_relative_ordinals"] == [0, 1]
    assert config["updates_per_episode"] == 2
    assert config["boundaries"]["predicted_track_training"] is False
    assert config["boundaries"]["goal_acceptance"] is False
