from pathlib import Path

from scripts.trackocd_cvpr.audit_mechanism_coverage import build_audit


def test_mechanism_coverage_is_receipt_only_and_has_three_families():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["status"] == "PASS_THREE_ROOT_CAUSE_MECHANISM_FAMILIES_RECORDED_NOT_SCIENTIFIC_ACCEPTANCE"
    assert audit["scope"] == {
        "receipt_only": True,
        "large_streams_opened": False,
        "data_or_weights_read": False,
        "tao_test_read": False,
        "external_processes_touched": False,
        "model_selection_performed": False,
        "formal_metrics_created": False,
        "goal_acceptance_claimed": False,
    }
    assert audit["mechanism_family_count"] == 3
    assert audit["mechanism_variant_count"] == 4
    assert all(item["boundary_safe"] for item in audit["mechanisms"])
    assert {item["root_cause_family"] for item in audit["mechanisms"]} == {
        "semantic_association_or_feature_fusion",
        "score_contract_and_normalization",
        "physical_temporal_association",
    }
    assert all(
        item["disposition"] != "SCIENTIFIC_ACCEPTANCE"
        for item in audit["mechanisms"]
    )
