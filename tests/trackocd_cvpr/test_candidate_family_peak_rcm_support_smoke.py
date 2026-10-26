import json

from scripts.trackocd_cvpr import candidate_family_peak_rcm_support_smoke as smoke


def test_family_peak_rcm_support_smoke_is_bounded_and_positive_without_training():
    config = json.loads(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(smoke.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["limits"]["optimizer_steps"] == 0
    assert config["boundaries"]["labels_or_gt_in_numeric_forward"] is False
    assert config["boundaries"]["harness_forced_cross_video_reuse"] is False
    assert result["status"] == "PASS_FAMILY_PEAK_RCM_M_SUPPORT_SMOKE"
    assert result["lifecycle"]["cross_video_token_count"] >= 1
    assert result["m_support"]["m_support_steps"] >= 1
    assert result["m_support"]["positive_other_video_groups"] >= 1
    assert result["resources"]["optimizer_steps"] == 0
    assert result["mechanism"]["score_sweep"] is False
