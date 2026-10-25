#!/usr/bin/env python3
"""Reconcile existing MASA runtime/checkpoint evidence without rerunning it.

The receipt is intentionally narrower than frontend qualification: it proves
that the pinned checkpoint was strictly loaded by the pinned image-only native
wrapper in an already-sealed bounded smoke, while keeping release-stage
provenance, dense causal coverage, and formal metric admission open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c"
SOURCE_COMMIT = "c5472b9c7615f35abdf1188cb1a0c5408fe50d66"


def read_json(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {relative}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_receipt() -> dict[str, Any]:
    checkpoint = read_json("outputs/trackocd_core/audit/masa_checkpoint_recovery.json")
    physical = read_json("outputs/trackocd_core/audit/masa_physical_prediction_summary.json")
    smoke = read_json("outputs/trackocd_core/audit/masa_native_smoke.json")
    route = read_json("outputs/trackocd_core/audit/masa_training_route.json")
    manifest = read_json("configs/trackocd_core/masa_native_source_manifest.json")
    plan = read_json("configs/trackocd_core/masa_sam_candidate_smoke.json")

    frozen = physical.get("frozen_model", {})
    smoke_frozen = smoke.get("frozen_model", {})
    checkpoint_spec = plan.get("weights", [{}])[0]
    checks = {
        "checkpoint_recovery_hash_matches_plan": (
            checkpoint.get("sha256") == CHECKPOINT_SHA256
            and checkpoint.get("sha256") == checkpoint_spec.get("sha256_from_repository_lfs_metadata")
            and checkpoint.get("size_and_sha256_match") is True
        ),
        "checkpoint_required_prefixes_present": (
            checkpoint.get("tensor_structure", {}).get("all_required_component_prefixes_present") is True
        ),
        "historical_physical_strict_runtime_load": (
            frozen.get("checkpoint_sha256") == CHECKPOINT_SHA256
            and frozen.get("strict_state_tensor_keys") == 419
            and frozen.get("missing_keys") == []
            and frozen.get("unexpected_keys") == []
            and frozen.get("shape_mismatches") == []
            and frozen.get("all_parameters_frozen") is True
            and frozen.get("second_pretrain_weight_used") is False
        ),
        "historical_native_smoke_contract": (
            smoke.get("status") == "PASS_BOUNDED_FROZEN_NATIVE_SMOKE_NOT_M1_QUALIFICATION"
            and smoke_frozen.get("strict_state_tensor_keys") == 419
            and smoke_frozen.get("missing_keys") == []
            and smoke_frozen.get("unexpected_keys") == []
            and smoke_frozen.get("shape_mismatches") == []
            and smoke.get("prefix_invariance", {}).get("future_pixels_actually_changed") is True
            and all(
                row.get("exact_detection_and_track_arrays_equal") is True
                for row in smoke.get("prefix_invariance", {}).get("comparisons", [])
            )
        ),
        "pinned_source_lineage_matches_route": (
            manifest.get("commit") == SOURCE_COMMIT
            and route.get("source_commit") == SOURCE_COMMIT
        ),
        "release_stage_binding_remains_open": (
            route.get("exact_checkpoint_stage_config_supervision_binding_verified") is False
        ),
    }
    status = (
        "PASS_RUNTIME_CHECKPOINT_BINDING_RECONCILED_PROVENANCE_STAGE_OPEN"
        if all(checks.values())
        else "FAIL_RUNTIME_CHECKPOINT_RECONCILIATION"
    )
    return {
        "schema_version": "trackocd.cvpr.masa-runtime-checkpoint-reconciliation.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_audit": git_head(),
        "status": status,
        "checks": checks,
        "runtime_evidence": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "strict_state_tensor_keys": frozen.get("strict_state_tensor_keys"),
            "real_image_forwards_in_sealed_physical_receipt": physical.get("real_image_forwards_started"),
            "native_smoke_real_image_forwards": smoke.get("real_image_forwards_started"),
            "native_smoke_prefix_scope": smoke.get("prefix_invariance", {}).get("proof_scope"),
        },
        "qualification_boundary": {
            "runtime_checkpoint_compatibility_reconciled": all(
                checks[name]
                for name in (
                    "checkpoint_recovery_hash_matches_plan",
                    "checkpoint_required_prefixes_present",
                    "historical_physical_strict_runtime_load",
                    "historical_native_smoke_contract",
                )
            ),
            "exact_release_stage_training_binding_verified": False,
            "dense_full_train_or_val_frontend_qualified": False,
            "formal_v2_cache_admitted": False,
            "formal_ocd_or_mot_metrics_created": False,
            "model_selection_performed": False,
        },
        "boundary": {
            "new_model_constructed_or_run": False,
            "new_inference_or_training": False,
            "new_data_or_annotation_read": False,
            "tao_test_read": False,
            "external_processes_touched": False,
            "weights_downloaded": False,
        },
        "next_valid_gate": (
            "Recover or establish exact release-stage/source provenance and a dense causal v2 input contract; "
            "then register a separate qualification run before any formal cache or Goal evaluation."
        ),
        "source_receipts": [
            receipt("outputs/trackocd_core/audit/masa_checkpoint_recovery.json"),
            receipt("outputs/trackocd_core/audit/masa_physical_prediction_summary.json"),
            receipt("outputs/trackocd_core/audit/masa_native_smoke.json"),
            receipt("outputs/trackocd_core/audit/masa_training_route.json"),
            receipt("configs/trackocd_core/masa_native_source_manifest.json"),
            receipt("configs/trackocd_core/masa_sam_candidate_smoke.json"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json"),
    )
    args = parser.parse_args()
    result = build_receipt()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
