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
    assert rows["root_cause_supported_mechanisms"]["status"] == (
        "COMPLETE_MECHANISM_COVERAGE_NOT_SCIENTIFIC_ACCEPTANCE"
    )
    assert rows["root_cause_supported_mechanisms"]["facts"]["mechanism_family_count"] == 3
    assert rows["root_cause_supported_mechanisms"]["facts"]["all_boundaries_safe"] is True
    assert rows["legacy_research_preservation"]["status"] == (
        "COMPLETE_LEGACY_RESEARCH_PRESERVATION_VERIFIED"
    )
    assert rows["legacy_research_preservation"]["facts"]["current_head_is_descendant"] is True
    assert rows["legacy_research_preservation"]["facts"]["protected_refs_match"] is True
    assert rows["lawful_strong_frontend"]["status"] == "OPEN_FRONTEND_NOT_QUALIFIED"
    assert rows["lawful_strong_frontend"]["facts"]["runtime_checkpoint_binding_reconciled"] is True
    assert rows["lawful_strong_frontend"]["facts"]["exact_release_stage_training_binding_verified"] is False
    assert rows["lawful_strong_frontend"]["facts"]["official_release_route_reproduced"] is False
    assert rows["lawful_strong_frontend"]["facts"]["current_candidate_is_official_tao_route"] is False
    assert rows["lawful_strong_frontend"]["facts"]["formal_v2_ready"] is False
    assert rows["lawful_strong_frontend"]["facts"]["local_cache_required_shards"] == 723
    assert rows["lawful_strong_frontend"]["facts"]["local_cache_completed_shards"] == 384
    assert rows["lawful_strong_frontend"]["facts"]["local_cache_missing_shards"] == 339
    assert rows["lawful_strong_frontend"]["facts"]["local_cache_formal_complete"] is False
    assert rows["lawful_strong_frontend"]["facts"]["historical_complete_cache_reachable"] is False
    assert (
        rows["lawful_strong_frontend"]["facts"]["formal_v2_admission_status"]
        == "BLOCKED_MISSING_OR_INVALID_FORMAL_V2_ASSETS"
    )
    assert rows["lawful_strong_frontend"]["facts"]["formal_v2_missing_or_invalid_asset_count"] > 0
    assert rows["lawful_strong_frontend"]["facts"]["masa_dense_train_pilot_status"] == (
        "PASS_BOUNDED_MASA_DENSE_FRAME_ONLINE_TRAIN_SMOKE_NO_GT"
    )
    assert rows["lawful_strong_frontend"]["facts"]["masa_dense_train_pilot_frames"] == 32
    assert rows["lawful_strong_frontend"]["facts"]["masa_dense_train_pilot_train_images_only"] is True
    assert rows["lawful_strong_frontend"]["facts"]["masa_dense_train_pilot_all_finite"] is True
    assert rows["lawful_strong_frontend"]["facts"]["masa_dense_train_pilot_max_retained_ids"] == 50
    assert rows["lawful_strong_frontend"]["facts"]["hybrid_geometry_smoke_status"] == (
        "COMPLETE_CATEGORY_FREE_HYBRID_GEOMETRY_SMOKE_NOT_QUALIFICATION"
    )
    assert rows["lawful_strong_frontend"]["facts"]["hybrid_geometry_smoke_precision"] == 0.03956639566395664
    assert rows["lawful_strong_frontend"]["facts"]["hybrid_geometry_smoke_recall"] == 0.8021978021978022
    assert rows["lawful_strong_frontend"]["facts"]["hybrid_geometry_smoke_conditional_switch_rate"] == 0.2926829268292683
    assert rows["lawful_strong_frontend"]["facts"]["hybrid_geometry_smoke_is_frontend_qualification"] is False
    assert rows["lawful_strong_frontend"]["facts"]["nas_recovery_probe_status"] == (
        "BLOCKED_REVERSE_TUNNEL_UNAVAILABLE"
    )
    assert rows["lawful_strong_frontend"]["facts"]["nas_reverse_tunnel_listening"] is False
    assert rows["lawful_strong_frontend"]["facts"]["historical_current_frontend_stream_identity_status"] == (
        "HISTORICAL_SELECTION_NOT_REPROMOTED_CURRENT_STREAM_IDENTITY_MISMATCH"
    )
    assert rows["lawful_strong_frontend"]["facts"]["historical_current_same_input_stream_proven"] is False
    assert rows["candidate_aware_decision"]["facts"]["gt_controlled_matrix_variant_count"] == 3
    assert rows["candidate_aware_decision"]["facts"]["gt_controlled_matrix_trial_count"] == 9
    assert rows["candidate_aware_decision"]["facts"]["gt_controlled_matrix_orders_per_trial"] == 3
    assert rows["candidate_aware_decision"]["facts"]["gt_controlled_matrix_pred_or_goal_accepted"] is False
    assert rows["predicted_track_training"]["facts"]["predicted_training_readiness_status"] == (
        "BLOCKED_PREDICTED_TRAINING_INPUT_NOT_QUALIFIED"
    )
    assert rows["predicted_track_training"]["facts"]["predicted_training_reliable_predicted_tracks"] == 0
    assert rows["predicted_track_training"]["facts"]["predicted_training_reliable_temporal_pairs"] == 0
    assert rows["predicted_track_training"]["facts"]["predicted_training_source_frame_gap_one_pairs"] == 0
    assert rows["predicted_track_training"]["facts"]["policy_input_optimizer_steps"] == 0
    assert rows["predicted_track_training"]["facts"]["predicted_category_free_feature_pilot_videos"] == 2
    assert rows["predicted_track_training"]["facts"]["predicted_feature_video0275_status"] == (
        "PASS_FIXED_SINGLE_VIDEO_DINO_FEATURE_PILOT_NO_TRAINING"
    )
    assert rows["predicted_track_training"]["facts"]["predicted_cross_video_lifecycle_token_count"] == 3
    assert rows["predicted_track_training"]["facts"]["predicted_cross_video_lifecycle_optimizer_steps"] == 0
    assert rows["predicted_track_training"]["facts"]["rcm_m_support_coverage_status"] == (
        "FAIL_RCM_PILOT_M_SUPPORT_COVERAGE_ZERO"
    )
    assert rows["predicted_track_training"]["facts"]["rcm_m_loss_steps"] == 0
    assert rows["predicted_track_training"]["facts"]["rcm_positive_other_video_groups"] == 0
    assert rows["self_correcting_persistent_memory"]["facts"][
        "rcm_m_branch_scientifically_exercised"
    ] is False
    assert rows["self_correcting_persistent_memory"]["facts"][
        "rcm_pilot_is_goal_acceptance"
    ] is False
    assert rows["self_correcting_persistent_memory"]["facts"][
        "predicted_cross_video_lifecycle_is_supervision"
    ] is False
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_val_recheck_status"
    ] == "PASS_VAL_CATEGORY_FREE_METRICS_RECHECK_TETA_PENDING_NOT_FRONTEND_ACCEPTANCE"
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_hota_percent"
    ] == 15.402
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_assa_percent"
    ] == 44.535
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_detre_percent"
    ] == 54.418
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_novel_observability"
    ] == {"numerator": 189, "denominator": 819, "value": 0.23076923076923078}
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_persistent_observability"
    ] == {"numerator": 84, "denominator": 527, "value": 0.15939278937381404}
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_teta_reference_status"
    ] == "NATIVE_TETA_REFERENCE_EVAL_FAILED"
    assert rows["formal_causal_online_metrics"]["facts"][
        "covtrack_native_recheck_is_frontend_acceptance"
    ] is False
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_baseline_status"
    ] == "PASS_SEALED_VAL_NEAREST_OCD_BASELINE"
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_p1_h_score"
    ] == 0.08521459987909867
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_p16_h_score"
    ] == 0.10735585987893223
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_fixed_gt_denominator"
    ] == 5232
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_model_input_gt_free"
    ] is True
    assert rows["formal_causal_online_metrics"]["facts"][
        "val_nearest_is_new_rerun"
    ] is False
    assert rows["independent_final_and_goals"]["status"] == "OPEN_GOAL_A_B_C_PENDING"
