import json

from scripts.trackocd_cvpr import audit_rcm_support_availability as audit


def test_rcm_support_audit_is_read_only_and_finds_static_support():
    config = json.loads(audit.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(audit.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["boundaries"]["read_only"] is True
    assert config["boundaries"]["optimizer_started"] is False
    assert result["status"] == "PASS_LAWFUL_CROSS_VIDEO_KNOWN_SUPPORT_AVAILABLE"
    assert result["support"]["cross_video_known_support_pair_count"] > 0
    assert result["preserved_rcm_coverage"]["m_loss_steps"] == 0
    assert result["preserved_rcm_coverage"]["positive_other_video_groups"] == 0
    assert result["boundaries"]["semantic_ids_written_to_receipt"] is False
