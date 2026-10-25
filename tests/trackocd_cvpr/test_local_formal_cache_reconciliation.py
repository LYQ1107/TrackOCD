from pathlib import Path

from scripts.trackocd_cvpr.audit_local_formal_cache_reconciliation import build_audit


def test_local_formal_cache_checkpoint_is_partial_and_not_promoted():
    repo = Path(__file__).resolve().parents[2]
    audit = build_audit(repo)

    assert audit["status"] == (
        "PASS_LOCAL_PARTIAL_CACHE_RECONCILED_HISTORICAL_COMPLETE_CACHE_UNREACHABLE_NOT_FORMAL_READY"
    )
    assert audit["scope"] == {
        "small_metadata_receipts_only": True,
        "feature_arrays_opened": False,
        "large_streams_opened": False,
        "data_or_weights_read": False,
        "tao_test_read": False,
        "external_processes_touched": False,
        "model_selection_performed": False,
    }
    assert audit["local_cache"]["required_shards"] == 723
    assert audit["local_cache"]["completed_shards"] == 384
    assert audit["local_cache"]["missing_shards"] == 339
    assert audit["local_cache"]["formal_cache_complete"] is False
    assert audit["historical_cache"]["reported_completed_shards"] == 723
    assert audit["historical_cache"]["historical_pass_is_current_asset"] is False
    assert audit["binding_checks"]["historical_cache_reachable"] is False
