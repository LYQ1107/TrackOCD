import json

from scripts.trackocd_cvpr import candidate_aware_rcm_support_smoke as smoke


def test_candidate_aware_rcm_support_smoke_preserves_zero_support_boundary():
    config = json.loads(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(smoke.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["limits"]["optimizer_steps"] == 0
    assert config["boundaries"]["labels_or_gt_in_numeric_forward"] is False
    assert result["status"] == "FAIL_CANDIDATE_AWARE_RCM_M_SUPPORT_ZERO"
    assert result["lifecycle"]["cross_video_token_count"] == 0
    assert result["m_support"]["m_support_steps"] == 0
    assert result["boundaries"]["harness_forced_cross_video_reuse"] is False
