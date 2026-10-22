from pathlib import Path

from scripts.trackocd_v2.audit_current_model_contract import waiting_for_selection


def test_missing_selection_is_structured_and_fail_closed(tmp_path):
    selection = tmp_path / "missing_selection.json"
    result = waiting_for_selection(selection)

    assert result["status"] == "WAITING_MISSING_SELECTION_ARTIFACT"
    assert result["decision"] == "WAITING_MISSING_SELECTION_ARTIFACT"
    assert result["selection_artifact_exists"] is False
    assert result["folds"] == []
    assert result["test_semantic_accessed"] is False
