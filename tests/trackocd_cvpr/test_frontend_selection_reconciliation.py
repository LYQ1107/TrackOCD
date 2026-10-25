import json

from scripts.trackocd_cvpr import audit_frontend_selection_reconciliation as audit


def test_historical_and_current_frontend_streams_are_not_silently_merged():
    result = json.loads(audit.OUTPUT.read_text(encoding="utf-8"))
    assert result["status"] == (
        "HISTORICAL_SELECTION_NOT_REPROMOTED_CURRENT_STREAM_IDENTITY_MISMATCH"
    )
    assert result["historical_selection"]["selected_frontend"] == "COVTrack-native"
    assert result["identity_comparison"]["physical_stream_sha256_match"] is False
    assert result["identity_comparison"]["native_evaluator_rows_sha256_match"] is False
    assert result["identity_comparison"]["same_input_stream_proven"] is False
    assert result["boundary"]["frontend_repromoted"] is False
    assert result["boundary"]["model_selection_performed"] is False
