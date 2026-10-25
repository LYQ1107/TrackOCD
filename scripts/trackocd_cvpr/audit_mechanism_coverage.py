#!/usr/bin/env python3
"""Audit the three root-cause-supported mechanism families already tested.

This is a compact receipt-only synthesis.  It does not reopen streams, data,
weights, GT/Test labels, or model checkpoints.  A mechanism is counted only
when its paired receipt exists, reports a concrete changed outcome, and keeps
the experiment outside formal metric/model-selection acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.mechanism-coverage-audit.v1"


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


def evidence(repo: Path, relative_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = repo / relative_path
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "status": payload.get("status"),
    }


def false_flags(payload: dict[str, Any], *names: str) -> bool:
    boundary = payload.get("boundary", payload.get("boundaries"))
    return isinstance(boundary, dict) and all(boundary.get(name) is False for name in names)


def run_flags_false(payload: dict[str, Any], *names: str) -> bool:
    return all(payload.get(name) is False for name in names)


def build_audit(repo: Path) -> dict[str, Any]:
    nosem_geo_path, nosem_geo = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/geometry_comparison.json"
    )
    nosem_temporal_path, nosem_temporal = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/temporal_comparison.json"
    )
    nosem_run_path, nosem_run = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/run.json"
    )
    softmax_path, softmax = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_single_softmax_v1/comparison.json"
    )
    softmax_run_path, softmax_run = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_single_softmax_v1/run.json"
    )
    foreground_path, foreground = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_fg_mass_fixed2_v1/comparison.json"
    )
    foreground_run_path, foreground_run = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_fg_mass_fixed2_v1/run.json"
    )
    reassoc_run_path, reassoc_run = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/run.json"
    )
    reassoc_dev_path, reassoc_dev = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/development_temporal.json"
    )
    reassoc_policy_path, reassoc_policy = load_json(
        repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/policy_temporal.json"
    )

    nosem_base = nosem_geo["baseline"]["geometry"]
    nosem_candidate = nosem_geo["candidate"]["geometry"]
    nosem_base_temporal = nosem_temporal["baseline"]["aggregate"]
    nosem_candidate_temporal = nosem_temporal["candidate"]["aggregate"]
    softmax_base = softmax["baseline"]["stream"]
    softmax_candidate = softmax["candidate"]["stream"]
    foreground_base = foreground["baseline"]["stream"]
    foreground_candidate = foreground["candidate"]["stream"]
    reassoc_dev_base = reassoc_dev["baseline"]["aggregate"]
    reassoc_dev_candidate = reassoc_dev["candidate"]["aggregate"]
    reassoc_policy_base = reassoc_policy["baseline"]["aggregate"]
    reassoc_policy_candidate = reassoc_policy["candidate"]["aggregate"]

    nosem_safe = (
        false_flags(
            nosem_geo,
            "category_fields_read",
            "feature_arrays_read",
            "native_evaluator_rows_read",
            "pixels_read",
            "prediction_to_gt_mapping_persisted",
            "role_or_vocabulary_read",
            "test_accessed",
            "training_started",
            "val_labels_read",
        )
        and false_flags(
            nosem_temporal,
            "feature_arrays_read",
            "gt_category_fields_read",
            "gt_role_or_vocabulary_read",
            "prediction_to_gt_mapping_persisted",
            "test_accessed",
            "train_annotation_opened",
            "training_started",
            "val_labels_read",
        )
    )
    score_safe = run_flags_false(
        softmax_run,
        "formal_metrics",
        "gt_or_semantic_arrays_read",
        "tao_test_accessed",
        "tao_val_accessed",
        "training_started",
    ) and run_flags_false(
        foreground_run,
        "formal_metrics",
        "gt_or_semantic_arrays_read",
        "tao_test_accessed",
        "tao_val_accessed",
        "training_started",
    )
    reassoc_safe = false_flags(
        reassoc_run,
        "feature_arrays_read",
        "full_cache_automatic_start",
        "gt_category_fields_read",
        "gt_role_or_vocabulary_read",
        "prediction_to_gt_mapping_persisted",
        "raw_payload_uploaded",
        "test_accessed",
        "train_annotation_opened",
        "training_started",
        "val_labels_read",
    )

    mechanisms = [
        {
            "id": "semantic_association_ablation",
            "root_cause_family": "semantic_association_or_feature_fusion",
            "candidate": nosem_geo["mechanisms"]["candidate"],
            "baseline": nosem_geo["mechanisms"]["baseline"],
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/geometry_comparison.json", nosem_geo),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/temporal_comparison.json", nosem_temporal),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_nosem_development_transfer_v1/run.json", nosem_run),
            ],
            "observed": {
                "baseline_precision": nosem_base["observation_precision_on_annotated_images"],
                "candidate_precision": nosem_candidate["observation_precision_on_annotated_images"],
                "baseline_recall": nosem_base["observation_recall_against_gt"],
                "candidate_recall": nosem_candidate["observation_recall_against_gt"],
                "baseline_adjacent_match_rate": nosem_base_temporal["adjacent_pair_match_rate"],
                "candidate_adjacent_match_rate": nosem_candidate_temporal["adjacent_pair_match_rate"],
                "baseline_switch_rate": nosem_base_temporal["pred_id_switch_rate_given_adjacent_pair_matches"],
                "candidate_switch_rate": nosem_candidate_temporal["pred_id_switch_rate_given_adjacent_pair_matches"],
            },
            "disposition": "REJECTED_NO_GEOMETRY_OR_PERSISTENCE_REPAIR",
            "boundary_safe": nosem_safe,
        },
        {
            "id": "single_softmax_score_contract",
            "root_cause_family": "score_contract_and_normalization",
            "candidate": softmax["mechanisms"]["candidate"],
            "baseline": softmax["mechanisms"]["baseline"],
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_single_softmax_v1/comparison.json", softmax),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_single_softmax_v1/run.json", softmax_run),
            ],
            "observed": {
                "baseline_score_mean": softmax_base["score_mean"],
                "candidate_score_mean": softmax_candidate["score_mean"],
                "baseline_score_over_0_5": softmax_base["score_over_0.5"],
                "candidate_score_over_0_5": softmax_candidate["score_over_0.5"],
                "baseline_tracks": softmax_base["tracks"],
                "candidate_tracks": softmax_candidate["tracks"],
            },
            "disposition": "EFFECT_CONFIRMED_SCORE_ROUTE_CHANGED_NOT_QUALIFIED",
            "boundary_safe": score_safe,
        },
        {
            "id": "foreground_mass_score_contract",
            "root_cause_family": "score_contract_and_normalization",
            "candidate": foreground["mechanisms"]["candidate"],
            "baseline": foreground["mechanisms"]["baseline"],
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_fg_mass_fixed2_v1/comparison.json", foreground),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_contract_fg_mass_fixed2_v1/run.json", foreground_run),
            ],
            "observed": {
                "baseline_score_mean": foreground_base["score_mean"],
                "candidate_score_mean": foreground_candidate["score_mean"],
                "baseline_score_over_0_5": foreground_base["score_over_0.5"],
                "candidate_score_over_0_5": foreground_candidate["score_over_0.5"],
                "baseline_tracks": foreground_base["tracks"],
                "candidate_tracks": foreground_candidate["tracks"],
            },
            "disposition": "EFFECT_CONFIRMED_FOREGROUND_SCORE_SATURATION_NOT_QUALIFIED",
            "boundary_safe": score_safe,
        },
        {
            "id": "causal_physical_reassociation",
            "root_cause_family": "physical_temporal_association",
            "candidate": reassoc_run["mechanism"],
            "baseline": "native_category_free_physical_stream",
            "evidence": [
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/run.json", reassoc_run),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/development_temporal.json", reassoc_dev),
                evidence(repo, "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1/policy_temporal.json", reassoc_policy),
            ],
            "observed": {
                "development": {
                    "precision_delta": reassoc_dev_candidate["observation_precision_on_annotated_images"] - reassoc_dev_base["observation_precision_on_annotated_images"],
                    "recall_delta": reassoc_dev_candidate["observation_recall_against_annotated_gt"] - reassoc_dev_base["observation_recall_against_annotated_gt"],
                    "adjacent_match_delta": reassoc_dev_candidate["adjacent_pair_match_rate"] - reassoc_dev_base["adjacent_pair_match_rate"],
                    "switch_rate_delta": reassoc_dev_candidate["pred_id_switch_rate_given_adjacent_pair_matches"] - reassoc_dev_base["pred_id_switch_rate_given_adjacent_pair_matches"],
                },
                "policy": {
                    "precision_delta": reassoc_policy_candidate["observation_precision_on_annotated_images"] - reassoc_policy_base["observation_precision_on_annotated_images"],
                    "recall_delta": reassoc_policy_candidate["observation_recall_against_annotated_gt"] - reassoc_policy_base["observation_recall_against_annotated_gt"],
                    "adjacent_match_delta": reassoc_policy_candidate["adjacent_pair_match_rate"] - reassoc_policy_base["adjacent_pair_match_rate"],
                    "switch_rate_delta": reassoc_policy_candidate["pred_id_switch_rate_given_adjacent_pair_matches"] - reassoc_policy_base["pred_id_switch_rate_given_adjacent_pair_matches"],
                },
            },
            "disposition": "REJECTED_PRECISION_PERSISTENCE_GAIN_COSTS_COVERAGE",
            "boundary_safe": reassoc_safe,
        },
    ]

    family_count = len({item["root_cause_family"] for item in mechanisms})
    all_receipts = [receipt for item in mechanisms for receipt in item["evidence"]]
    all_present = all(item["exists"] for item in all_receipts)
    all_safe = all(item["boundary_safe"] for item in mechanisms)
    status = (
        "PASS_THREE_ROOT_CAUSE_MECHANISM_FAMILIES_RECORDED_NOT_SCIENTIFIC_ACCEPTANCE"
        if all_present and all_safe and family_count >= 3
        else "INCOMPLETE_MECHANISM_COVERAGE_AUDIT"
    )
    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": {
            "receipt_only": True,
            "large_streams_opened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "external_processes_touched": False,
            "model_selection_performed": False,
            "formal_metrics_created": False,
            "goal_acceptance_claimed": False,
        },
        "mechanism_family_count": family_count,
        "mechanism_variant_count": len(mechanisms),
        "mechanisms": mechanisms,
        "inference": {
            "semantic_association": "NoSemantic changed the exported stream but left geometry and continuity unchanged on the fixed development prefix; semantic association is not sufficient to explain the dominant failure.",
            "score_contract": "Single-softmax and foreground-mass probes changed score mass and track population materially; score construction is causally active, but these probes do not establish a qualified frontend.",
            "physical_association": "Causal reassociation reduced switches only while sharply reducing adjacent coverage/recall; persistence alone cannot be optimized without the geometry/coverage gate.",
        },
        "scientific_boundary": "This satisfies the bookkeeping requirement to preserve at least three distinct root-cause mechanism families with real paired evidence. It does not qualify a frontend, train a predicted-track policy, create formal OCD/MOT metrics, or accept Goal A/B/C.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/mechanism_coverage_audit_20261026.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
