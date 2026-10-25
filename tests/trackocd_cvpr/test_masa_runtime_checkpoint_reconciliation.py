from scripts.trackocd_cvpr.audit_masa_runtime_checkpoint_reconciliation import build_receipt


def test_masa_runtime_reconciliation_keeps_provenance_gate_open():
    receipt = build_receipt()

    assert receipt["status"] == "PASS_RUNTIME_CHECKPOINT_BINDING_RECONCILED_PROVENANCE_STAGE_OPEN"
    assert all(receipt["checks"].values())
    assert receipt["qualification_boundary"]["runtime_checkpoint_compatibility_reconciled"] is True
    assert receipt["qualification_boundary"]["exact_release_stage_training_binding_verified"] is False
    assert receipt["qualification_boundary"]["dense_full_train_or_val_frontend_qualified"] is False
    assert receipt["boundary"] == {
        "new_model_constructed_or_run": False,
        "new_inference_or_training": False,
        "new_data_or_annotation_read": False,
        "tao_test_read": False,
        "external_processes_touched": False,
        "weights_downloaded": False,
    }
