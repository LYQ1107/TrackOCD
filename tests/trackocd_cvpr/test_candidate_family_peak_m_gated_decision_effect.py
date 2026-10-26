import json

from scripts.trackocd_cvpr import candidate_family_peak_m_gated_decision_effect as effect


def test_m_gated_decision_effect_scope_is_frozen():
    config = json.loads((effect.ROOT / "configs/trackocd_cvpr/candidate_family_peak_m_gated_decision_effect_v1.json").read_text(encoding="utf-8"))
    assert config["frames_per_video"] == 8
    assert config["online_optimizer_steps"] == 1
    assert config["scorer"]["feature_indices"] == {"adapted_cos": 0, "adapted_minus_raw": 12}
    assert config["scorer"]["coefficient_sweep"] is False
    assert config["boundaries"]["predicted_track_training"] is False


def test_m_adjusted_scorer_is_numeric_and_shape_bounded():
    import torch

    scorer = effect.MAdjustedFamilyPeakScorer()
    features = torch.zeros((2, 14), dtype=torch.float32)
    features[:, 1] = torch.tensor([0.0, 1.0])
    features[:, 0] = torch.tensor([0.6, 0.5])
    features[:, 12] = torch.tensor([0.1, 0.0])
    scores = scorer(features, torch.zeros(6, dtype=torch.float32))
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
