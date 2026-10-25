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
    masa_path, masa = load_json(
        repo, "outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json"
    )
    asset_path, asset = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/image_only_frontend_asset_recheck_v1.json"
    )
    runtime_path, runtime = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json"
    )
    val_path, val = load_json(
        repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json"
    )

    core_reviews = review.get("scoped_actual_core_reviews", [])
    extra_reviews = review.get("licensed_extra_inspection_reviews", [])
    full_cache_totals = cache.get("totals", {})
    geometry_totals = geometry.get("aggregate_geometry", {})
    pred_boundaries = pred_protocol.get("boundaries", {})
    root_boundaries = root.get("boundaries", {})

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
            "id": "lawful_strong_frontend",
            "status": "OPEN_FRONTEND_NOT_QUALIFIED",
            "requirement": "Qualify a lawful, dense, causally valid frontend before formal predicted-track training and evaluation.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_train_geometry_audit_v1/result.json", reported_status=geometry.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/v2/masa_frontend_migration_eligibility_v1.json", reported_status=masa.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/image_only_frontend_asset_recheck_v1.json", reported_status=asset.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json", reported_status=runtime.get("status")),
            ],
            "facts": {
                "covtrack_reliable_temporal_pairs": geometry_totals.get("reliable_temporal_pairs"),
                "masa_admissible_for_formal_v2": masa.get("admissible_for_formal_v2"),
                "strong_legal_primary_accepted": asset.get("strong_legal_primary_Pred_or_scientific_Goal_accepted"),
                "runtime_checkpoint_binding_reconciled": nested(runtime, "qualification_boundary", "runtime_checkpoint_compatibility_reconciled"),
                "exact_release_stage_training_binding_verified": nested(runtime, "qualification_boundary", "exact_release_stage_training_binding_verified"),
            },
            "open_gate": "Need an admitted frontend/cache with exact provenance, dense causal contract, and shared metric validation.",
        },
        {
            "id": "predicted_track_training",
            "status": "OPEN_NOT_STARTED_AS_QUALIFIED_TRAINING",
            "requirement": "Train and evaluate the policy on frozen predicted trajectories under the causal Train-development protocol.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/joint_causal_pred_dense_protocol_v1/result.json", reported_status=pred_protocol.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1/result.json", reported_status=policy_input.get("status")),
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json", reported_status=rcm.get("status")),
            ],
            "facts": {
                "predicted_cache_boundary": pred_boundaries,
                "policy_input_optimizer_steps": policy_input.get("optimizer_steps"),
                "rcm_pilot_optimizer_steps": nested(rcm, "summary", "optimizer_steps"),
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
            ],
            "facts": {
                "old_diagnostic_candidate_ceiling": "measured as an upper bound, not a deployed method",
                "pilot_goal_acceptance": nested(rcm, "model_selection_or_goal_acceptance"),
            },
            "open_gate": "Need a trained/predicted implementation, independent folds, multi-seed/order/coverage controls, and locked evaluation.",
        },
        {
            "id": "self_correcting_persistent_memory",
            "status": "OPEN_PILOT_ONLY",
            "requirement": "Demonstrate self-correcting persistent category memory on actual causal predicted histories.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json", reported_status=rcm.get("status")),
                evidence(repo, "docs/trackocd_cvpr/CVPR2_ROOT_SUPPORTED_MECHANISMS.md"),
            ],
            "facts": {
                "pilot_scope": rcm.get("pilot_scope"),
                "prediction_owned_memory_only": nested(rcm, "summary", "prediction_owned_memory_only"),
                "pilot_is_goal_acceptance": nested(rcm, "summary", "development_or_goal_acceptance"),
            },
            "open_gate": "The bounded R/C/M pilot is not a trained, generalizable, or Goal-accepted memory result.",
        },
        {
            "id": "formal_causal_online_metrics",
            "status": "PARTIAL_SEALED_VAL_EVIDENCE_FULL_ACCEPTANCE_OPEN",
            "requirement": "Produce paper-grade causal online OCD/MOT metrics under the frozen protocol.",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json", reported_status=val.get("status")),
            ],
            "facts": {
                "sealed_layers": {
                    "category_free_mot": nested(val, "category_free_mot", "status"),
                    "nearest": nested(val, "ocd_nearest", "status"),
                    "dpmeans": nested(val, "ocd_dpmeans", "status"),
                },
                "formal_v2_status": "not complete; DP-means and full causal Goal evaluation remain open",
            },
            "open_gate": "Existing Val numbers are sealed interim evidence, not end-to-end CVPR/Goal acceptance.",
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
            evidence(repo, "outputs/trackocd_cvpr/cvpr5/val_metric_truth_reaudit_20261025.json", reported_status=val.get("status")),
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
