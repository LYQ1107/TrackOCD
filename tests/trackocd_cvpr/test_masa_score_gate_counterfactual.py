import json

from scripts.trackocd_cvpr import audit_masa_score_gate_counterfactual as audit


def test_masa_score_gate_is_frozen_bounded_and_not_selection():
    config = json.loads(audit.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["thresholds"] == [0.25, 0.5, 0.75, 0.9]
    assert config["boundaries"]["existing_sealed_prediction_only"] is True
    assert config["boundaries"]["model_selection"] is False
    assert config["boundaries"]["test_accessed"] is False


def test_masa_score_gate_receipt_has_no_mapping_or_selection():
    result = json.loads(audit.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_MASA_SCORE_GATE_COUNTERFACTUAL_DIAGNOSTIC_NOT_SELECTION"
    assert len(result["thresholds"]) == 4
    assert "selected_threshold" not in result
    assert result["boundaries"]["prediction_to_gt_track_mapping_persisted"] is False
