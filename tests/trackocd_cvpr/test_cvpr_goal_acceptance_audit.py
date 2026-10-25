from pathlib import Path

from scripts.trackocd_cvpr.audit_cvpr_goal_acceptance import build_audit


def test_cvpr_goal_acceptance_audit_is_fail_closed_and_receipt_only():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["schema_version"] == "trackocd.cvpr.goal-acceptance-audit.v1"
    assert audit["scope"] == {
        "receipt_only": True,
        "large_streams_opened": False,
        "data_or_weights_read": False,
        "tao_test_read": False,
        "external_processes_touched": False,
        "model_selection_performed": False,
    }
    assert audit["goal_status"]["audit_result"] == "INCOMPLETE_OPEN_REQUIREMENTS"
    assert audit["goal_status"]["missing_evidence"] == []
    assert audit["goal_status"]["goal_a"] == "PENDING"
    assert audit["goal_status"]["goal_b"] == "PENDING"
    assert audit["goal_status"]["goal_c"] == "PENDING"

    rows = {item["id"]: item for item in audit["requirements"]}
    assert rows["literature_and_source_review"]["facts"]["actual_core_reviews"] == 12
    assert rows["lawful_strong_frontend"]["status"] == "OPEN_FRONTEND_NOT_QUALIFIED"
    assert rows["predicted_track_training"]["facts"]["policy_input_optimizer_steps"] == 0
    assert rows["independent_final_and_goals"]["status"] == "OPEN_GOAL_A_B_C_PENDING"
