from pathlib import Path

import pytest

from scripts.trackocd_cvpr.audit_covtrack_native_val_metrics_recheck import build_audit


def test_covtrack_native_val_recheck_is_real_but_not_frontend_acceptance():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["status"] == (
        "PASS_VAL_CATEGORY_FREE_METRICS_RECHECK_TETA_PENDING_NOT_FRONTEND_ACCEPTANCE"
    )
    assert audit["scope"] == {
        "val_only": True,
        "large_prediction_stream_reopened": False,
        "data_or_weights_read": False,
        "tao_test_read": False,
        "model_selection_performed": False,
        "training_started": False,
        "external_processes_touched": False,
    }
    values = audit["metric_audit"]["tracking_values_percent"]
    assert values["HOTA"] == pytest.approx(15.402)
    assert values["AssA"] == pytest.approx(44.535)
    assert values["DetRe"] == pytest.approx(54.418)
    assert audit["metric_audit"]["novel_track_observability"]["numerator"] == 189
    assert audit["metric_audit"]["novel_track_observability"]["denominator"] == 819
    assert audit["metric_audit"]["persistent_observability"]["numerator"] == 84
    assert audit["metric_audit"]["persistent_observability"]["denominator"] == 527
    assert audit["qualification_boundary"]["teta_reference_complete"] is False
    assert audit["qualification_boundary"]["frontend_registered_as_bakeoff_complete"] is False
