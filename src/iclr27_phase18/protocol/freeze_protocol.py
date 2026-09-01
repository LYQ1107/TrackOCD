"""Freeze Phase18 protocol artifacts before learned model optimization."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase18"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    manifests = OUT / "manifests"
    config_path = ROOT / "configs/iclr27_phase18/dstm.json"
    protocol_path = ROOT / "docs/iclr27_phase18/PROTOCOL.md"
    method_path = ROOT / "docs/iclr27_phase18/METHOD.md"
    audit_path = ROOT / "docs/iclr27_phase18/OFFICIAL_METHOD_AUDIT.md"
    fold_path = manifests / "fold_manifest.json"
    denom_path = manifests / "identifiable_ct_denominators.json"
    alignment_path = manifests / "feature_alignment.json"
    census_path = manifests / "eligible_category_census.json"
    oracle_path = OUT / "eval/oracle_contracts.json"
    diagnosis_path = OUT / "eval/phase17r_terminal_diagnosis_reproduction.json"
    b1_path = OUT / "eval/b1_prereg_baseline.json"
    required = [config_path, protocol_path, method_path, audit_path, fold_path,
                denom_path, alignment_path, census_path, oracle_path,
                diagnosis_path, b1_path]
    for path in required:
        assert path.is_file() and path.stat().st_size > 0, path
    cfg = json.loads(config_path.read_text())
    folds = json.loads(fold_path.read_text())
    denom = json.loads(denom_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    census = json.loads(census_path.read_text())
    oracle = json.loads(oracle_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    b1 = json.loads(b1_path.read_text())
    assert diagnosis["all_required_reproductions_match"] is True
    assert alignment["rows"] == 43423 and alignment["dinov2_set_match"] and alignment["dinov3_set_match"]
    assert census["eligible_category_count"] == 11 and census["eligible_reliable_rows"] == 221
    assert denom["positive_event_count"] == denom["negative_event_count"] == 41
    assert oracle["identifiability_passed"]
    assert b1["metrics"]["commit_ct"]["correct"] == 9

    task_contract = {
        "protocol": "trackocd_iclr27_phase18_task_contract",
        "frozen_before_learned_phase18_training": True,
        "physical_stream": "Phase17R corrected DSCT proposal/physical tracks, immutable",
        "population_rows": 43423,
        "population_row_keys_sha256": alignment["source_row_keys_ordered_sha256"],
        "feature_alignment": {
            "dinov2_set_exact": True, "dinov2_reindexed_by_immutable_row_key": True,
            "dinov3_order_exact": True, "deployed_main": "DINOv2 CLS+ROI",
        },
        "identities": {
            "physical_track": "index-only local buffer key",
            "supported_known": "declared 48-category namespace",
            "novel_semantic_state": "opaque online IDs >=100000; no GT/physical numeric value",
        },
        "actions": ["KNOWN(c)", "DEFER(local_track)", "NEW_NOVEL(k)", "EXISTING_NOVEL(k)"],
        "reliability_label_gt_only": "assigned == 1 and exact current row_iou >= 0.5",
        "robustness_iou_thresholds": [0.3, 0.7],
        "primary_positive_events": 41,
        "matched_negative_events": 41,
        "eligible_categories": census["eligible_categories"],
        "fold_manifest_sha256": folds["fold_sha256"],
        "denominator_sha256": denom["denominator_sha256"],
        "fit_roles": ["known_bank", "novel_correspondence_train"],
        "held_category_in_loss": False,
        "held_video_in_loss": False,
        "nested_calibration_category_in_loss": False,
        "checkpoint_selection_uses_held_prediction": False,
        "primary_metric": "Commit-CT at first semantic commitment on/after exact reliable target prefix",
        "secondary_metrics": [
            "post-prefix CT row recall", "existing precision", "matched-negative false merge",
            "time to correct commit", "pre-prefix defer/premature commit", "coverage-risk",
            "fragmentation/duplicates/merges/unresolved", "known micro/macro",
            "readiness AUROC/AUPRC/locked precision/recall", "category/video/quality/length/prefix strata",
            "category/video clustered uncertainty",
        ],
        "legacy_immediate_ct": "unchanged stress diagnostic only",
        "past_actions_immutable": True,
        "merge_scope": "current and future local mapping only",
        "defer_global_update": False,
        "max_states": cfg["model"]["max_deployed_novel_states"],
        "max_anchors_per_state": cfg["model"]["state_anchor_top_k"],
        "devplus_accessed": False, "q1_accessed": False,
    }
    atomic_json(manifests / "task_contract.json", task_contract)

    prereg = {
        "protocol": "trackocd_iclr27_phase18_preregistration",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "historical_phase17r_observed": True,
        "phase18_learned_held_fold_predictions_observed": False,
        "diagnosis_reproduced_before_preregistration": True,
        "no_training_B1_observed_before_numeric_gate_lock": {
            "commit_ct_correct": 9, "commit_ct_eligible": 41,
            "correct_categories": 3, "correct_target_videos": 6,
            "existing_precision": b1["metrics"]["existing_precision"],
            "negative_false_merge_rate": b1["metrics"]["negative_false_merge_rate"],
        },
        "architecture_and_training": cfg,
        "fold_manifest_sha256": folds["fold_sha256"],
        "denominator_sha256": denom["denominator_sha256"],
        "oracles_frozen": ["O0", "O1", "O2", "O3", "O4", "O5"],
        "baseline_order": ["B0 historical replay", "B1 frozen prototype", "B2 trained pair scorer", "B3 same-capacity no-DEFER/no-merge"],
        "main_order": ["DSTM seed1801 full cross-fit", "exactly one repair if required", "three-seed final if gate/repair improvement", "essential one-seed ablations"],
        "minimum_per_fold": {"updates": 20000, "unique_fit_row_passes": 10, "rule": "whichever requires greater coverage"},
        "checkpoint_selection": cfg["calibration_selection"],
        "public_gates": cfg["public_gates"],
        "repair_policy": cfg["repair_policy"],
        "external_policy": "DEV+ only after locked public DEV+ gate; Q1 only after DEV+ pass; measurement-only",
        "forbidden": [
            "future frames", "Q1/DEV+ label tuning", "held-category loss", "held-video fitting",
            "physical ID as semantic value", "GT/IoU/category as deployed input", "denominator changes after predictions",
            "audit checkpoint selection", "more than one Phase18 repair family",
        ],
    }
    atomic_json(manifests / "preregistration.json", prereg)

    locked_paths = required + [manifests / "task_contract.json", manifests / "preregistration.json"]
    hashes = {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in locked_paths}
    atomic_json(manifests / "pretraining_artifact_hashes.json", hashes)
    public_lock = {
        "protocol": "trackocd_iclr27_phase18_public_lock_pretraining",
        "freeze_timestamp_utc": prereg["freeze_timestamp_utc"],
        "population": {"rows": 43423, "row_keys_sha256": alignment["source_row_keys_ordered_sha256"]},
        "eligible": {"categories": 11, "reliable_rows": 221, "positive_events": 41, "negative_events": 41},
        "fold_sha256": folds["fold_sha256"],
        "denominator_sha256": denom["denominator_sha256"],
        "pretraining_artifact_hash_manifest": str((manifests / "pretraining_artifact_hashes.json").resolve()),
        "pretraining_artifact_hash_manifest_sha256": sha(manifests / "pretraining_artifact_hashes.json"),
        "phase17r_already_observed": True,
        "phase18_learned_prediction_existed_at_freeze": False,
        "public_is_blind_external_test": False,
        "locked_candidate": None,
        "devplus_authorized": False,
        "q1_authorized": False,
    }
    atomic_json(manifests / "public_lock.json", public_lock)
    external = {
        "protocol": "trackocd_iclr27_phase18_external_evaluation_lock",
        "devplus_accessed": False, "devplus_authorized": False,
        "q1_accessed": False, "q1_authorized": False,
        "reason": "awaiting fully locked public cross-fit candidate and preregistered gate",
        "phase17r_external_boundary_preserved": True,
    }
    atomic_json(manifests / "external_evaluation_lock.json", external)
    print(json.dumps({
        "frozen": True, "fold_sha256": folds["fold_sha256"],
        "denominator_sha256": denom["denominator_sha256"],
        "hash_manifest_sha256": public_lock["pretraining_artifact_hash_manifest_sha256"],
        "phase18_learned_prediction_existed_at_freeze": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
