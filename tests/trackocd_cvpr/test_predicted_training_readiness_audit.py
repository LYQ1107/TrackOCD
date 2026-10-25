import json

from scripts.trackocd_cvpr import audit_predicted_training_readiness as audit


def test_predicted_training_readiness_is_fail_closed():
    result = json.loads(audit.OUTPUT.read_text(encoding="utf-8"))
    assert result["status"] == "BLOCKED_PREDICTED_TRAINING_INPUT_NOT_QUALIFIED"
    assert result["full_train_geometry"]["reliable_predicted_tracks"] == 0
    assert result["full_train_geometry"]["reliable_temporal_pairs"] == 0
    assert result["full_train_continuity"]["source_frame_gap_one_pairs"] == 0
    assert result["policy_train_geometry"]["reliable_predicted_tracks"] == 0
    assert result["interface_and_optimizer_state"]["policy_input_optimizer_steps"] == 0
    assert result["boundary"]["training_started"] is False
    assert result["boundary"]["tao_test_read"] is False
