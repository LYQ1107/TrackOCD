import json

from scripts.trackocd_cvpr import candidate_family_peak_m_training_smoke as smoke


def test_family_peak_m_full_clock_is_fixed_and_untrained_as_pred():
    config_path = smoke.ROOT / "configs/trackocd_cvpr/candidate_family_peak_m_full_clock_v1.json"
    output_path = smoke.ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_m_full_clock_v1/result.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert config["frames_per_video"] is None
    assert config["max_optimizer_steps"] == 2
    assert result["status"] == "PASS_FAMILY_PEAK_M_BRANCH_TRAINING_SMOKE"
    assert result["scope"]["frames"] == 79
    assert result["scope"]["positive_other_video_groups"] >= 1
    assert result["training"]["optimizer_steps"] == 2
    assert result["training"]["parameters_changed"] is True
    assert result["training"]["predicted_track_training"] is False
