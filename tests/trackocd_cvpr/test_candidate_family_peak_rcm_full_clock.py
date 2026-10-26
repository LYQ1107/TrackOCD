import json

from scripts.trackocd_cvpr import candidate_family_peak_rcm_support_smoke as smoke


def test_family_peak_full_clock_receipt_is_causal_and_untrained():
    config_path = smoke.ROOT / "configs/trackocd_cvpr/candidate_family_peak_rcm_full_clock_v1.json"
    output_path = smoke.ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_rcm_full_clock_v1/result.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert config["frames_per_video"] is None
    assert result["status"] == "PASS_FAMILY_PEAK_RCM_M_SUPPORT_SMOKE"
    assert result["lifecycle"]["videos"] == 2
    assert result["lifecycle"]["frames"] == 79
    assert result["lifecycle"]["cross_video_token_count"] >= 1
    assert result["m_support"]["m_support_steps"] >= 1
    assert result["resources"]["optimizer_steps"] == 0
    assert result["boundaries"]["val_or_test_accessed"] is False
