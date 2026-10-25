from pathlib import Path

import pytest

from scripts.trackocd_cvpr.audit_val_nearest_baseline import build_audit


def test_sealed_val_nearest_baseline_receipt_is_compact_and_fail_closed():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["status"] == "PASS_SEALED_VAL_NEAREST_OCD_BASELINE"
    assert audit["scope"] == {
        "val_only": True,
        "sealed_existing_result": True,
        "new_inference_started": False,
        "training_started": False,
        "tao_test_read": False,
        "model_selection_performed": False,
        "external_processes_touched": False,
    }
    assert audit["protocol"]["fixed_gt_denominator"] == 5232
    assert audit["protocol"]["post_decision_join_rows"] == 1206
    assert audit["protocol"]["unmatched_gt_deferred"] == 4026
    p1 = audit["prefix_metrics_fraction"]["p1"]
    p16 = audit["prefix_metrics_fraction"]["p16"]
    assert p1["h_score"] == pytest.approx(0.08521459987909867)
    assert p16["h_score"] == pytest.approx(0.10735585987893223)
    assert audit["qualification_boundary"]["is_formal_frontend_selection"] is False
    assert audit["qualification_boundary"]["is_new_rerun"] is False
