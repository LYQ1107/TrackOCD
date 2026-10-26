import json

from scripts.trackocd_cvpr import candidate_aware_lifecycle_smoke as smoke


def test_candidate_aware_lifecycle_smoke_is_fixed_and_not_training():
    config = json.loads(smoke.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(smoke.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["fixed_similarity_reuse_threshold"] == 0.55
    assert config["boundaries"]["harness_forced_cross_video_reuse"] is False
    assert result["status"] == "PASS_UNTRAINED_CANDIDATE_AWARE_LIFECYCLE_SMOKE"
    assert result["lifecycle"]["cross_video_token_count"] == 1
    assert result["lifecycle"]["optimizer_steps"] == 0
    assert result["boundaries"]["labels_or_gt_read"] is False
