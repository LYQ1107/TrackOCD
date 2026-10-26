#!/usr/bin/env python3
"""Build a fail-closed, receipt-only audit of the TrackOCD CVPR goal.

This audit is deliberately not a scientific evaluator.  It reads only compact
JSON receipts and the public research status file, records their hashes, and
maps the requested end-state requirements to authoritative evidence.  Open
requirements remain open; a completed substage is never promoted to goal
acceptance by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.goal-acceptance-audit.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(repo: Path, relative_path: str) -> tuple[Path, dict[str, Any]]:
    path = repo / relative_path
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative_path}")
    return path, value


def evidence(repo: Path, relative_path: str, *, reported_status: str | None = None) -> dict[str, Any]:
    path = repo / relative_path
    item: dict[str, Any] = {
        "path": relative_path,
        "exists": path.is_file(),
    }
    if path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
        if reported_status is not None:
            item["reported_status"] = reported_status
    return item


def nested(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return value if value is not None else default


def count_or_length(value: Any) -> int | None:
    """Accept both count-style and list-style receipt fields."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, list):
        return len(value)
    return None


def git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_audit(repo: Path) -> dict[str, Any]:
    status_path, status = load_json(repo, "outputs/trackocd_cvpr/status.json")
    root_path, root = load_json(
        repo, "outputs/trackocd_cvpr/cvpr0/root_cause_synthesis_20261022.json"
    )
    review_path, review = load_json(
        repo, "outputs/trackocd_cvpr/cvpr1/review_coverage_index.json"
    )
    review_verify_path, review_verify = load_json(
        repo, "outputs/trackocd_cvpr/cvpr1/review_coverage_verification.json"
    )
    cache_path, cache = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json",
    )
    geometry_path, geometry = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json"
    )
    pred_protocol_path, pred_protocol = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/joint_causal_pred_dense_protocol_v1/result.json"
    )
    policy_input_path, policy_input = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json"
    )
    rcm_path, rcm = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json"
    )
    rcm_coverage_path, rcm_coverage = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/rcm_pilot_coverage_audit_20261026.json",
    )
    rcm_support_path, rcm_support = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/rcm_support_availability_audit_20261026.json",
    )
    predicted_feature_video0275_path, predicted_feature_video0275 = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_video0275_v1/result.json",
    )
    cross_video_smoke_path, cross_video_smoke = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/predicted_train_rcm_cross_video_smoke_v1/result.json",
    )
    masa_path, masa = load_json(
        repo, "outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json"
    )
    asset_path, asset = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/image_only_frontend_asset_recheck_v1.json"
    )
    runtime_path, runtime = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json"
    )
    release_path, release = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/masa_release_source_closure_20261026.json"
    )
    admission_path, admission = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/v2_recovery_asset_admission_20261026.json"
    )
    dense_masa_path, dense_masa = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_smoke_v1/result.json"
    )
    dense_masa_pilot_path, dense_masa_pilot = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_pilot_bdd32_v1/result.json",
    )
    masa_score_gate_path, masa_score_gate = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/masa_score_gate_counterfactual_v1/result.json",
    )
    hybrid_path, hybrid = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr2/masa_covtrack_hybrid_geometry_smoke_v1/result.json",
    )
    mechanism_path, mechanism = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/mechanism_coverage_audit_20261026.json"
    )
    preservation_path, preservation = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/legacy_preservation_audit_20261026.json"
    )
    local_cache_path, local_cache = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/local_formal_cache_reconciliation_20261026.json"
    )
    val_path, val = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json"
    )
    covtrack_recheck_path, covtrack_recheck = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/covtrack_native_val_metrics_recheck_20261026.json",
    )
    nearest_path, nearest = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/val_nearest_ocd_metrics_receipt_20261026.json",
    )
    candidate_matrix_path, candidate_matrix = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json",
    )
    nas_probe_path, nas_probe = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/nas_recovery_probe_20261026.json",
    )
    frontend_reconciliation_path, frontend_reconciliation = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/frontend_selection_reconciliation_20261026.json",
    )
    predicted_readiness_path, predicted_readiness = load_json(
        repo,
        "outputs/trackocd_cvpr/cvpr5/predicted_training_readiness_audit_20261026.json",
    )

    core_reviews = review.get("scoped_actual_core_reviews", [])
    extra_reviews = review.get("licensed_extra_inspection_reviews", [])
    full_cache_totals = cache.get("totals", {})
    geometry_totals = geometry.get("aggregate_geometry", {})
    pred_boundaries = pred_protocol.get("boundaries", {})
    root_boundaries = root.get("boundaries", {})
    covtrack_recheck_metrics = covtrack_recheck.get("metric_audit", {})
    covtrack_recheck_values = covtrack_recheck_metrics.get("tracking_values_percent", {})
    covtrack_recheck_observability = covtrack_recheck_metrics.get(
        "novel_track_observability", {}
    )
    covtrack_recheck_persistent = covtrack_recheck_metrics.get(
        "persistent_observability", {}
    )
    hybrid_metrics = {
        str(item.get("name")): item
        for item in hybrid.get("metrics", [])
        if isinstance(item, dict) and item.get("name")
    }
    hybrid_union = hybrid_metrics.get("covtrack_first_union_dedup", {})
    candidate_aggregate = candidate_matrix.get("aggregate", {})
    candidate_models = candidate_aggregate.get("models", {})
    candidate_trials = candidate_matrix.get("all_original_trials", {})

    requirements: list[dict[str, Any]] = [
        {
            "id": "cvpr0_root_cause_diagnostics",
            "status": "COMPLETE_DIAGNOSTIC_NOT_SCIENTIFIC_ACCEPTANCE",
            "requirement": "Complete FULL=0, Top-K/action upper-bound, first-error propagation, and representation-geometry diagnostics.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr0/root_cause_synthesis_20261022.json", reported_status=root.get("status")),
                evidence(repo, "docs/trackocd_cvpr/CVPR0_ROOT_CAUSE_SYNTHESIS_20261022.md"),
            ],
            "facts": {
                "root_boundaries": root_boundaries,
                "interpretation": "diagnostic evidence only; no new-method acceptance",
            },
            "open_gate": "No additional CVPR-0 claim is promoted to end-to-end scientific success.",
        },
        {
            "id": "literature_and_source_review",
            "status": "MINIMUM_REVIEW_COUNT_MET_STAGE_ACCEPTANCE_OPEN",
            "requirement": "Audit at least 12 real papers and actual model/loss/train/eval source.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr1/review_coverage_index.json", reported_status=review.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr1/review_coverage_verification.json", reported_status=review_verify.get("status")),
                evidence(repo, "docs/trackocd_cvpr/CVPR1_LITERATURE_AND_GAP_ANALYSIS.md"),
            ],
            "facts": {
                "actual_core_reviews": count_or_length(core_reviews),
                "licensed_extra_reviews": count_or_length(extra_reviews),
                "minimum_required_actual_core_reviews": review.get("minimum_required_actual_core_reviews"),
                "scientific_goal_complete": review.get("scientific_goal_complete"),
            },
            "open_gate": "Source lineage, fresh causal folds, and mechanism/Goal acceptance remain separate gates.",
        },
        {
            "id": "anonymous_train_dense_cache",
            "status": "COMPLETE_CACHE_PROVENANCE_NOT_FRONTEND_QUALIFICATION",
            "requirement": "Preserve and reconcile the lawful anonymous Train physical cache for later predicted-track work.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json", reported_status=cache.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json", reported_status=geometry.get("status")),
            ],
            "facts": {
                "totals": full_cache_totals,
                "geometry": {
                    "observation_precision": geometry_totals.get("observation_precision_on_annotated_images"),
                    "observation_recall": geometry_totals.get("observation_recall_against_annotated_gt"),
                    "reliable_temporal_pairs": geometry_totals.get("reliable_temporal_pairs"),
                },
            },
            "open_gate": "The cache is reusable provenance, but its current geometry/continuity does not qualify a strong frontend.",
        },
        {
            "id": "root_cause_supported_mechanisms",
            "status": (
                "COMPLETE_MECHANISM_COVERAGE_NOT_SCIENTIFIC_ACCEPTANCE"
                if mechanism.get("status", "").startswith("PASS_")
                else "OPEN_MECHANISM_COVERAGE_INCOMPLETE"
            ),
            "requirement": "Explore at least three distinct root-cause-supported mechanisms with finite paired evidence.",
            "evidence": [
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/mechanism_coverage_audit_20261026.json",
                    reported_status=mechanism.get("status"),
                ),
            ],
            "facts": {
                "mechanism_family_count": mechanism.get("mechanism_family_count"),
                "mechanism_variant_count": mechanism.get("mechanism_variant_count"),
                "all_boundaries_safe": all(
                    item.get("boundary_safe") is True for item in mechanism.get("mechanisms", [])
                ),
                "scientific_acceptance_claimed": False,
            },
            "open_gate": "Mechanism coverage is recorded, but only a qualified predicted frontend and locked causal evaluation can accept a method.",
        },
        {
            "id": "legacy_research_preservation",
            "status": (
                "COMPLETE_LEGACY_RESEARCH_PRESERVATION_VERIFIED"
                if preservation.get("status", "").startswith("PASS_")
                else "OPEN_LEGACY_RESEARCH_PRESERVATION_UNVERIFIED"
            ),
            "requirement": "Preserve the verified 7546527 lineage and later legal research history without overwriting old branches or failed evidence.",
            "evidence": [
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/legacy_preservation_audit_20261026.json",
                    reported_status=preservation.get("status"),
                ),
            ],
            "facts": {
                "historical_parent": nested(preservation, "repository", "historical_parent"),
                "current_head_is_descendant": nested(
                    preservation, "repository", "current_head_is_descendant_of_historical_parent"
                ),
                "protected_refs_match": all(
                    item.get("matches_expected") is True
                    for item in preservation.get("protected_refs", {}).values()
                ),
                "preserved_failed_attempt_count": nested(
                    preservation, "manifest_binding", "preserved_failed_attempt_count"
                ),
                "refs_moved": nested(preservation, "scope", "refs_moved"),
            },
            "open_gate": "Lineage preservation is verified; this is provenance evidence and does not accept scientific goals or metrics.",
        },
        {
            "id": "lawful_strong_frontend",
            "status": "OPEN_FRONTEND_NOT_QUALIFIED",
            "requirement": "Qualify a lawful, dense, causally valid frontend before formal predicted-track training and evaluation.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json", reported_status=geometry.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json", reported_status=masa.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_smoke_v1/result.json",
                    reported_status=dense_masa.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_pilot_bdd32_v1/result.json",
                    reported_status=dense_masa_pilot.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr2/masa_score_gate_counterfactual_v1/result.json",
                    reported_status=masa_score_gate.get("status"),
                ),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/image_only_frontend_asset_recheck_v1.json", reported_status=asset.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json", reported_status=runtime.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/masa_release_source_closure_20261026.json", reported_status=release.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/v2_recovery_asset_admission_20261026.json", reported_status=admission.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr2/masa_covtrack_hybrid_geometry_smoke_v1/result.json",
                    reported_status=hybrid.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/nas_recovery_probe_20261026.json",
                    reported_status=nas_probe.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/frontend_selection_reconciliation_20261026.json",
                    reported_status=frontend_reconciliation.get("status"),
                ),
            ],
            "facts": {
                "covtrack_reliable_temporal_pairs": geometry_totals.get("reliable_temporal_pairs"),
                "masa_admissible_for_formal_v2": masa.get("admissible_for_formal_v2"),
                "masa_adjacent_dense_smoke_status": dense_masa.get("status"),
                "masa_adjacent_dense_smoke_frames": count_or_length(
                    nested(dense_masa, "worker", "frames", default=[])
                ),
                "masa_adjacent_dense_smoke_train_images_only": nested(
                    dense_masa, "boundaries", "train_images_only"
                ),
                "masa_adjacent_dense_smoke_max_retained_ids": (
                    max(
                        (
                            int(row.get("retained_track_ids_from_previous_frame", 0))
                            for row in nested(dense_masa, "worker", "frames", default=[])
                        ),
                        default=None,
                    )
                    if isinstance(nested(dense_masa, "worker", "frames", default=[]), list)
                    else None
                ),
                "masa_dense_train_pilot_status": dense_masa_pilot.get("status"),
                "masa_dense_train_pilot_frames": count_or_length(
                    nested(dense_masa_pilot, "worker", "frames", default=[])
                ),
                "masa_dense_train_pilot_train_images_only": nested(
                    dense_masa_pilot, "boundaries", "train_images_only"
                ),
                "masa_dense_train_pilot_all_finite": (
                    all(
                        row.get("finite_boxes") is True
                        for row in nested(dense_masa_pilot, "worker", "frames", default=[])
                    )
                    if isinstance(nested(dense_masa_pilot, "worker", "frames", default=[]), list)
                    else False
                ),
                "masa_dense_train_pilot_max_retained_ids": (
                    max(
                        (
                            int(row.get("retained_track_ids_from_previous_frame", 0))
                            for row in nested(dense_masa_pilot, "worker", "frames", default=[])
                        ),
                        default=None,
                    )
                    if isinstance(nested(dense_masa_pilot, "worker", "frames", default=[]), list)
                    else None
                ),
                "masa_score_gate_status": masa_score_gate.get("status"),
                "masa_score_gate_fixed_thresholds": [
                    {
                        "threshold": row.get("threshold"),
                        "precision": row.get("observation_precision"),
                        "recall": row.get("observation_recall"),
                        "conditional_switch_rate": row.get("conditional_switch_rate"),
                    }
                    for row in masa_score_gate.get("thresholds", [])
                    if isinstance(row, dict)
                ],
                "masa_score_gate_selected_threshold": None,
                "strong_legal_primary_accepted": asset.get("strong_legal_primary_Pred_or_scientific_Goal_accepted"),
                "runtime_checkpoint_binding_reconciled": nested(runtime, "qualification_boundary", "runtime_checkpoint_compatibility_reconciled"),
                "exact_release_stage_training_binding_verified": nested(runtime, "qualification_boundary", "exact_release_stage_training_binding_verified"),
                "official_release_route_reproduced": nested(release, "qualification_boundary", "official_release_route_reproduced"),
                "current_candidate_is_official_tao_route": nested(release, "qualification_boundary", "current_image_only_candidate_is_official_tao_route"),
                "formal_v2_admission_status": admission.get("status"),
                "formal_v2_ready": admission.get("formal_ready"),
                "formal_v2_missing_or_invalid_asset_count": len(admission.get("problems", [])),
                "local_cache_reconciliation_status": local_cache.get("status"),
                "local_cache_required_shards": nested(local_cache, "local_cache", "required_shards"),
                "local_cache_completed_shards": nested(local_cache, "local_cache", "completed_shards"),
                "local_cache_missing_shards": nested(local_cache, "local_cache", "missing_shards"),
                "local_cache_formal_complete": nested(local_cache, "local_cache", "formal_cache_complete"),
                "historical_complete_cache_reachable": nested(
                    local_cache, "binding_checks", "historical_cache_reachable"
                ),
                "hybrid_geometry_smoke_status": hybrid.get("status"),
                "hybrid_geometry_smoke_precision": hybrid_union.get("observation_precision"),
                "hybrid_geometry_smoke_recall": hybrid_union.get("observation_recall"),
                "hybrid_geometry_smoke_conditional_switch_rate": nested(
                    hybrid_union, "temporal", "conditional_switch_rate"
                ),
                "hybrid_geometry_smoke_is_frontend_qualification": False,
                "nas_recovery_probe_status": nas_probe.get("status"),
                "nas_reverse_tunnel_listening": nested(
                    nas_probe, "managed_mount", "reverse_endpoint_listening_at_probe"
                ),
                "historical_current_frontend_stream_identity_status": frontend_reconciliation.get(
                    "status"
                ),
                "historical_current_same_input_stream_proven": nested(
                    frontend_reconciliation,
                    "identity_comparison",
                    "same_input_stream_proven",
                ),
            },
            "open_gate": "The bounded MASA dense pilot and fixed score-gate counterfactual strengthen engineering evidence, but frontend admission still needs exact release provenance, full lawful coverage, and shared metric validation; the reachable local formal cache is only 384/723 shards and the historical 723/723 path is currently unreachable.",
        },
        {
            "id": "predicted_track_training",
            "status": "OPEN_NOT_STARTED_AS_QUALIFIED_TRAINING",
            "requirement": "Train and evaluate the policy on frozen predicted trajectories under the causal Train-development protocol.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/joint_causal_pred_dense_protocol_v1/result.json", reported_status=pred_protocol.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json", reported_status=policy_input.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json", reported_status=rcm.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_v1/result.json", reported_status="PASS_FIXED_SINGLE_VIDEO_DINO_FEATURE_PILOT_NO_TRAINING"),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_video0275_v1/result.json", reported_status=predicted_feature_video0275.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/predicted_train_rcm_cross_video_smoke_v1/result.json", reported_status=cross_video_smoke.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/predicted_training_readiness_audit_20261026.json",
                    reported_status=predicted_readiness.get("status"),
                ),
            ],
            "facts": {
                "predicted_cache_boundary": pred_boundaries,
                "policy_input_optimizer_steps": policy_input.get("optimizer_steps"),
                "rcm_pilot_optimizer_steps": nested(rcm, "summary", "optimizer_steps"),
                "predicted_category_free_feature_pilot_videos": 2,
                "predicted_feature_video0275_status": predicted_feature_video0275.get("status"),
                "predicted_cross_video_lifecycle_smoke_status": cross_video_smoke.get("status"),
                "predicted_cross_video_lifecycle_token_count": nested(
                    cross_video_smoke, "lifecycle", "cross_video_token_count"
                ),
                "predicted_cross_video_lifecycle_optimizer_steps": nested(
                    cross_video_smoke, "lifecycle", "optimizer_steps"
                ),
                "rcm_m_support_coverage_status": rcm_coverage.get("status"),
                "rcm_m_loss_steps": nested(rcm_coverage, "coverage", "m_loss_steps"),
                "rcm_positive_other_video_groups": nested(
                    rcm_coverage, "coverage", "positive_other_video_groups"
                ),
                "predicted_training_readiness_status": predicted_readiness.get("status"),
                "predicted_training_full_observation_precision": nested(
                    predicted_readiness, "full_train_geometry", "observation_precision"
                ),
                "predicted_training_full_observation_recall": nested(
                    predicted_readiness, "full_train_geometry", "observation_recall"
                ),
                "predicted_training_reliable_predicted_tracks": nested(
                    predicted_readiness, "full_train_geometry", "reliable_predicted_tracks"
                ),
                "predicted_training_reliable_temporal_pairs": nested(
                    predicted_readiness, "full_train_geometry", "reliable_temporal_pairs"
                ),
                "predicted_training_source_frame_gap_one_pairs": nested(
                    predicted_readiness, "full_train_continuity", "source_frame_gap_one_pairs"
                ),
                "predicted_training_dominant_source_gaps": nested(
                    predicted_readiness, "full_train_continuity", "dominant_source_gaps"
                ),
            },
            "open_gate": "The protocol and engineering pilot exist, but no qualified predicted-track training or formal metric result exists.",
        },
        {
            "id": "candidate_aware_decision",
            "status": "OPEN_PILOT_AND_DIAGNOSTIC_ONLY",
            "requirement": "Deliver a candidate-aware decision mechanism with fair causal comparison and no outcome leakage.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr0/root_cause_synthesis_20261022.json", reported_status=root.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json", reported_status=rcm.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json",
                    reported_status=candidate_matrix.get("status"),
                ),
            ],
            "facts": {
                "old_diagnostic_candidate_ceiling": "measured as an upper bound, not a deployed method",
                "pilot_goal_acceptance": nested(rcm, "model_selection_or_goal_acceptance"),
                "gt_controlled_matrix_status": candidate_matrix.get("status"),
                "gt_controlled_matrix_variant_count": len(candidate_models),
                "gt_controlled_matrix_trial_count": len(candidate_trials),
                "gt_controlled_matrix_orders_per_trial": min(
                    (len(value) for value in candidate_trials.values()),
                    default=0,
                ),
                "gt_controlled_matrix_selected_model_or_seed": candidate_aggregate.get(
                    "selected_model_seed_order_or_prefix"
                ),
                "gt_controlled_matrix_pred_or_goal_accepted": candidate_aggregate.get(
                    "official_CT_full_semantic_truth_Pred_or_Goal_accepted"
                ),
            },
            "open_gate": "Need a trained/predicted implementation, independent folds, multi-seed/order/coverage controls, and locked evaluation.",
        },
        {
            "id": "self_correcting_persistent_memory",
            "status": "OPEN_PILOT_ONLY",
            "requirement": "Demonstrate self-correcting persistent category memory on actual causal predicted histories.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json", reported_status=rcm.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/rcm_pilot_coverage_audit_20261026.json",
                    reported_status=rcm_coverage.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/rcm_support_availability_audit_20261026.json",
                    reported_status=rcm_support.get("status"),
                ),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr2/predicted_train_rcm_cross_video_smoke_v1/result.json",
                    reported_status=cross_video_smoke.get("status"),
                ),
                evidence(repo, "docs/trackocd_cvpr/CVPR2_ROOT_SUPPORTED_MECHANISMS.md"),
            ],
            "facts": {
                "pilot_scope": rcm.get("pilot_scope"),
                "prediction_owned_memory_only": nested(rcm, "summary", "prediction_owned_memory_only"),
                "pilot_is_goal_acceptance": nested(rcm, "summary", "development_or_goal_acceptance"),
                "rcm_m_support_coverage_status": rcm_coverage.get("status"),
                "rcm_m_loss_steps": nested(rcm_coverage, "coverage", "m_loss_steps"),
                "rcm_positive_other_video_groups": nested(
                    rcm_coverage, "coverage", "positive_other_video_groups"
                ),
                "rcm_static_support_status": rcm_support.get("status"),
                "rcm_static_support_pair_count": nested(
                    rcm_support, "support", "cross_video_known_support_pair_count"
                ),
                "rcm_static_physical_track_count": nested(
                    rcm_support, "support", "physical_track_count"
                ),
                "rcm_static_masked_track_count": nested(
                    rcm_support, "support", "masked_target_tracks"
                ),
                "rcm_m_branch_scientifically_exercised": nested(
                    rcm_coverage, "boundary", "m_branch_scientifically_exercised"
                ),
                "rcm_pilot_is_goal_acceptance": nested(
                    rcm_coverage, "boundary", "goal_a_b_c_accepted"
                ),
                "predicted_cross_video_lifecycle_smoke_status": cross_video_smoke.get("status"),
                "predicted_cross_video_lifecycle_token_count": nested(
                    cross_video_smoke, "lifecycle", "cross_video_token_count"
                ),
                "predicted_cross_video_lifecycle_is_supervision": False,
            },
            "open_gate": "The static policy fold contains lawful cross-video Known support, but the preserved pilot produced zero prediction-owned cross-video groups; repair and freeze the anonymous lifecycle coverage before claiming or selecting self-correcting memory.",
        },
        {
            "id": "formal_causal_online_metrics",
            "status": "PARTIAL_SEALED_VAL_EVIDENCE_FULL_ACCEPTANCE_OPEN",
            "requirement": "Produce paper-grade causal online OCD/MOT metrics under the frozen protocol.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json", reported_status=val.get("status")),
                evidence(
                    repo,
                    "outputs/trackocd_cvpr/cvpr5/covtrack_native_val_metrics_recheck_20261026.json",
                    reported_status=covtrack_recheck.get("status"),
                ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/frontend_selection_reconciliation_20261026.json",
                reported_status=frontend_reconciliation.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/predicted_training_readiness_audit_20261026.json",
                reported_status=predicted_readiness.get("status"),
            ),
            ],
            "facts": {
                "sealed_layers": {
                    "category_free_mot": nested(val, "category_free_mot", "status"),
                    "nearest": nested(val, "ocd_nearest", "status"),
                    "dpmeans": nested(val, "ocd_dpmeans", "status"),
                },
                "covtrack_native_val_recheck_status": covtrack_recheck.get("status"),
                "covtrack_native_hota_percent": covtrack_recheck_values.get("HOTA"),
                "covtrack_native_assa_percent": covtrack_recheck_values.get("AssA"),
                "covtrack_native_detre_percent": covtrack_recheck_values.get("DetRe"),
                "covtrack_native_novel_observability": {
                    "numerator": covtrack_recheck_observability.get("numerator"),
                    "denominator": covtrack_recheck_observability.get("denominator"),
                    "value": covtrack_recheck_observability.get("value"),
                },
                "covtrack_native_persistent_observability": {
                    "numerator": covtrack_recheck_persistent.get("numerator"),
                    "denominator": covtrack_recheck_persistent.get("denominator"),
                    "value": covtrack_recheck_persistent.get("value"),
                },
                "covtrack_native_teta_reference_status": covtrack_recheck_metrics.get(
                    "teta_reference_status"
                ),
                "covtrack_native_recheck_is_frontend_acceptance": covtrack_recheck.get(
                    "qualification_boundary", {}
                ).get("frontend_registered_as_bakeoff_complete"),
                "val_nearest_baseline_status": nearest.get("status"),
                "val_nearest_p1_h_score": nested(
                    nearest, "prefix_metrics_fraction", "p1", "h_score"
                ),
                "val_nearest_p16_h_score": nested(
                    nearest, "prefix_metrics_fraction", "p16", "h_score"
                ),
                "val_nearest_fixed_gt_denominator": nested(
                    nearest, "protocol", "fixed_gt_denominator"
                ),
                "val_nearest_model_input_gt_free": (
                    nested(nearest, "protocol", "model_input_contract", "gt_category_id") is False
                    and nested(nearest, "protocol", "model_input_contract", "gt_split") is False
                ),
                "val_nearest_is_new_rerun": nested(
                    nearest, "qualification_boundary", "is_new_rerun"
                ),
                "formal_v2_status": "not complete; DP-means and full causal Goal evaluation remain open",
            },
            "open_gate": "The COVTrack-native Val category-free MOT/observability recheck is real sealed evidence, but TETA reference evaluation is pending and no frontend, OCD, predicted-track, or Goal acceptance is promoted.",
        },
        {
            "id": "independent_final_and_goals",
            "status": "OPEN_GOAL_A_B_C_PENDING",
            "requirement": "Complete fair independent final evaluation and satisfy Goal A, Goal B, and Goal C.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/status.json", reported_status=status.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/independent_final_GT_video_cache_v1/result.json"),
            ],
            "facts": {
                "goal_a": status.get("Goal_A"),
                "goal_b": status.get("Goal_B"),
                "goal_c": status.get("Goal_C"),
                "scientific_success_claimed": status.get("scientific_success_claimed"),
                "independent_final_cache_is_only_control": True,
            },
            "open_gate": "No Goal A/B/C item is accepted; the final evaluation must follow the qualified predicted frontend and locked protocol.",
        },
    ]

    missing = [
        item["id"]
        for item in requirements
        if not all(entry.get("exists", False) for entry in item["evidence"])
    ]
    open_requirements = [
        item["id"] for item in requirements if item["status"].startswith(("OPEN", "PARTIAL"))
    ]
    goal_complete = not missing and not open_requirements and all(
        status.get(key) == "ACCEPTED" for key in ("Goal_A", "Goal_B", "Goal_C")
    )

    clean_before = git_value(repo, "status", "--porcelain") == ""
    audit = {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "branch": git_value(repo, "branch", "--show-current"),
            "head": git_value(repo, "rev-parse", "HEAD"),
            "clean_before_audit": clean_before,
        },
        "scope": {
            "receipt_only": True,
            "large_streams_opened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "external_processes_touched": False,
            "model_selection_performed": False,
        },
        "goal_status": {
            "audit_result": "INCOMPLETE_OPEN_REQUIREMENTS" if not goal_complete else "COMPLETE",
            "missing_evidence": missing,
            "open_requirements": open_requirements,
            "goal_a": status.get("Goal_A"),
            "goal_b": status.get("Goal_B"),
            "goal_c": status.get("Goal_C"),
            "scientific_success_claimed": status.get("scientific_success_claimed"),
        },
        "requirements": requirements,
        "source_receipts": [
            evidence(repo, "outputs/trackocd_cvpr/status.json", reported_status=status.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr0/root_cause_synthesis_20261022.json", reported_status=root.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr1/review_coverage_verification.json", reported_status=review_verify.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json", reported_status=cache.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr5/mechanism_coverage_audit_20261026.json", reported_status=mechanism.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr5/legacy_preservation_audit_20261026.json", reported_status=preservation.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr5/local_formal_cache_reconciliation_20261026.json", reported_status=local_cache.get("status")),
            evidence(repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json", reported_status=val.get("status")),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/covtrack_native_val_metrics_recheck_20261026.json",
                reported_status=covtrack_recheck.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/val_nearest_ocd_metrics_receipt_20261026.json",
                reported_status=nearest.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/rcm_pilot_coverage_audit_20261026.json",
                reported_status=rcm_coverage.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/rcm_support_availability_audit_20261026.json",
                reported_status=rcm_support.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr2/predicted_train_rcm_cross_video_smoke_v1/result.json",
                reported_status=cross_video_smoke.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_smoke_v1/result.json",
                reported_status=dense_masa.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_pilot_bdd32_v1/result.json",
                reported_status=dense_masa_pilot.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr2/masa_score_gate_counterfactual_v1/result.json",
                reported_status=masa_score_gate.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr2/masa_covtrack_hybrid_geometry_smoke_v1/result.json",
                reported_status=hybrid.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr3/C_development_all9_matrix_v1/result.json",
                reported_status=candidate_matrix.get("status"),
            ),
            evidence(
                repo,
                "outputs/trackocd_cvpr/cvpr5/nas_recovery_probe_20261026.json",
                reported_status=nas_probe.get("status"),
            ),
        ],
    }
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/cvpr_goal_acceptance_audit_20261026.json"),
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "audit_result": result["goal_status"]["audit_result"],
        "open_requirements": result["goal_status"]["open_requirements"],
        "missing_evidence": result["goal_status"]["missing_evidence"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if not result["goal_status"]["missing_evidence"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
