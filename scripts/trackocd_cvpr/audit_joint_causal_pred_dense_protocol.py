#!/usr/bin/env python3
"""Audit the frozen joint causal protocol without reading private GT arrays."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/trackocd_cvpr/joint_causal_pred_dense_protocol_v1.json"
DEFAULT_SMOKE = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_smoke_v1/run.json"
DEFAULT_IMAGE_PLAN = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v12_images/private_exact_joint_proof_or_assignment.json"
DEFAULT_RECONCILIATION = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_causal_pred_dense_protocol_v1/result.json"

PENDING_STATUS = "FROZEN_CAUSAL_BOUNDARIES_PRED_CACHE_PENDING"
COMPLETE_STATUS = "FROZEN_CAUSAL_BOUNDARIES_FULL_PRED_CACHE_COMPLETE_FRONTEND_QUALIFICATION_PENDING"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    config_path: Path,
    smoke_path: Path,
    image_plan_path: Path,
    output_path: Path,
    reconciliation_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    image_plan = json.loads(image_plan_path.read_text(encoding="utf-8"))
    protocol_status = config.get("status")
    if protocol_status not in {PENDING_STATUS, COMPLETE_STATUS}:
        raise ValueError("unexpected protocol status")
    cache_complete = protocol_status == COMPLETE_STATUS
    if config.get("data_protocol", {}).get("video_scope") != 407:
        raise ValueError("protocol is not bound to 407 approved Train videos")
    shards = config.get("predicted_dense_cache", {}).get("shards", [])
    if len(shards) != 4:
        raise ValueError("expected exactly four fixed dense shards")
    plan_ids = sorted({int(row["video_id"]) for row in image_plan.get("image_metadata", [])})
    if len(plan_ids) != 407:
        raise ValueError(f"approved image plan has {len(plan_ids)} video IDs, expected 407")
    plan_id_order_sha256 = hashlib.sha256(json.dumps(plan_ids, separators=(",", ":")).encode()).hexdigest()
    data_protocol = config.get("data_protocol", {})
    if data_protocol.get("approved_image_plan_sha256") != _sha256(image_plan_path):
        raise ValueError("approved image plan hash does not match the frozen protocol")
    if data_protocol.get("approved_video_id_order_sha256") != plan_id_order_sha256:
        raise ValueError("approved video-ID order hash does not match the frozen protocol")
    expected_start = 0
    for shard in shards:
        ordinal_min = int(shard["ordinal_min"])
        ordinal_max = int(shard["ordinal_max"])
        if ordinal_min != expected_start or ordinal_max < ordinal_min:
            raise ValueError("dense shard ordinal ranges are not contiguous")
        count = ordinal_max - ordinal_min + 1
        if int(shard["video_count"]) != count:
            raise ValueError("dense shard count mismatch")
        actual_ids = plan_ids[ordinal_min : ordinal_max + 1]
        if actual_ids[0] != int(shard["video_id_first"]) or actual_ids[-1] != int(shard["video_id_last"]):
            raise ValueError("dense shard endpoint does not match approved video-ID order")
        actual_hash = hashlib.sha256(json.dumps(actual_ids, separators=(",", ":")).encode()).hexdigest()
        if actual_hash != shard["video_id_order_sha256"]:
            raise ValueError("dense shard video-ID hash does not match approved order")
        expected_start = ordinal_max + 1
    if expected_start != len(plan_ids):
        raise ValueError("dense shard ordinals do not cover exactly the approved 407 IDs")
    if config.get("predicted_dense_cache", {}).get("full_cache_automatic_start") is not False:
        raise ValueError("full predicted cache is not fail-closed")
    reconciliation = None
    if cache_complete:
        if not reconciliation_path.is_file():
            raise ValueError("completed protocol state requires the reconciliation receipt")
        expected_hash = config.get("predicted_dense_cache", {}).get("reconciliation_receipt_sha256")
        actual_hash = _sha256(reconciliation_path)
        if expected_hash != actual_hash:
            raise ValueError("reconciliation receipt hash does not match the protocol")
        reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
        if reconciliation.get("status") != "PASS_ALL_407_FIXED_ANONYMOUS_TRAIN_DENSE_SUBSHARDS":
            raise ValueError("reconciliation receipt is not complete")
        if reconciliation.get("totals", {}).get("videos") != 407:
            raise ValueError("reconciliation receipt does not cover 407 videos")
        if not all(reconciliation.get(key) is True for key in (
            "all_receipts_complete",
            "all_physical_streams_present",
            "all_evaluator_streams_present",
            "all_stream_audit_evidence_pass",
            "all_subshard_input_union_exact",
        )):
            raise ValueError("reconciliation receipt has incomplete evidence")
    if smoke.get("status") != "COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SMOKE":
        raise ValueError("the completed smoke receipt is not successful")
    if smoke.get("split") != "train_predicted" or smoke.get("formal_metrics") is not False:
        raise ValueError("smoke receipt violates predicted-only/no-metrics boundary")
    if smoke.get("tao_val_accessed") is not False or smoke.get("tao_test_accessed") is not False:
        raise ValueError("smoke receipt violates Val/Test boundary")
    if smoke.get("training_started") is not False or smoke.get("gt_or_semantic_arrays_read") is not False:
        raise ValueError("smoke receipt violates training/GT boundary")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    result = {
        "schema_version": "trackocd.cvpr.joint-causal-pred-dense-protocol-audit.v1",
        "status": (
            "PASS_FROZEN_JOINT_CAUSAL_BOUNDARIES_FULL_PRED_CACHE_COMPLETE"
            if cache_complete
            else "PASS_FROZEN_JOINT_CAUSAL_BOUNDARIES_PRED_CACHE_PENDING"
        ),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": str(config_path.resolve()),
        "protocol_sha256": _sha256(config_path),
        "smoke_receipt": str(smoke_path.resolve()),
        "smoke_receipt_sha256": _sha256(smoke_path),
        "approved_image_plan": str(image_plan_path.resolve()),
        "approved_image_plan_sha256": _sha256(image_plan_path),
        "approved_video_id_order_sha256": plan_id_order_sha256,
        "protocol_state": protocol_status,
        "full_cache_reconciliation": (
            {
                "path": str(reconciliation_path.resolve()),
                "sha256": _sha256(reconciliation_path),
                "status": reconciliation.get("status"),
                "totals": reconciliation.get("totals", {}),
            }
            if reconciliation is not None
            else None
        ),
        "repository_head_at_audit": head,
        "fixed_dense_shards": shards,
        "smoke_counts": smoke.get("counts", {}),
        "boundaries": {
            "full_predicted_cache_started": cache_complete,
            "full_predicted_cache_complete": cache_complete,
            "training_started": False,
            "formal_metrics_created": False,
            "tao_val_or_test_labels_used": False,
            "novel_vocabulary_used": False,
            "private_gt_arrays_read_by_audit": False,
            "model_selection_performed": False,
        },
        "interpretation": (
            "Protocol and reconciled-cache boundary audit only; formal OCD/MOT, frontend qualification, and Goal A/B/C remain open."
            if cache_complete
            else "Protocol boundary audit only; Goal A/B/C, full predicted cache, formal OCD/MOT, and frontend qualification remain open."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--image-plan", type=Path, default=DEFAULT_IMAGE_PLAN)
    parser.add_argument("--reconciliation", type=Path, default=DEFAULT_RECONCILIATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.config.resolve(),
        args.smoke.resolve(),
        args.image_plan.resolve(),
        args.output.resolve(),
        args.reconciliation.resolve(),
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
