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
    assert config["require_joint_loss_terms"] is True
    assert config["boundaries"]["predicted_track_training"] is False
    assert config["boundaries"]["goal_acceptance"] is False


def test_family_peak_joint_receipt_has_real_c_and_m_updates_without_acceptance():
    path = ROOT / "outputs/trackocd_cvpr/cvpr5/family_peak_joint_rcm_training_pilot_v4/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    summary = result["summary"]
    assert result["status"] == "COMPLETED_FAMILY_PEAK_RESIDUAL_JOINT_RCM_PILOT_NOT_DEV_PRED_OR_GOAL_ACCEPTANCE"
    assert summary["optimizer_steps"] == 2
    assert summary["joint_graph_reached_R_C_and_M"] is True
    assert summary["prediction_owned_memory_only"] is True
    assert summary["episodes"][0]["metadata_frames"] == 79
    assert summary["episodes"][0]["physical_observations"] == 215
    assert all(row["C_loss"] is not None and row["M_loss"] is not None
               for row in summary["loss_curve"])
    assert all(min(row[key] for key in ("R_gradient_norm_sum", "C_gradient_norm_sum",
                                        "M_gradient_norm_sum")) > 0
               for row in summary["loss_curve"])
    assert result["private_endpoint"]["reopened_exact"] is True
    assert result["private_endpoint"]["content_uploaded"] is False
    assert result["boundaries"]["predicted_track_training"] is False
    assert result["boundaries"]["goal_acceptance"] is False
