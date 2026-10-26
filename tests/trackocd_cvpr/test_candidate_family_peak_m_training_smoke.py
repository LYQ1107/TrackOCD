import json

from scripts.trackocd_cvpr import candidate_family_peak_m_training_smoke as smoke


def test_family_peak_m_branch_training_smoke_has_one_real_update():
    config = json.loads(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(smoke.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["max_optimizer_steps"] == 1
    assert result["status"] == "PASS_FAMILY_PEAK_M_BRANCH_TRAINING_SMOKE"
    assert result["scope"]["frames"] == 4
    assert result["scope"]["positive_other_video_groups"] >= 1
    assert result["training"]["optimizer_steps"] == 1
    assert result["training"]["parameters_changed"] is True
    assert result["training"]["predicted_track_training"] is False
    assert result["boundaries"]["labels_or_gt_in_numeric_forward"] is False
