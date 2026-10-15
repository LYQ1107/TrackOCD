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
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_causal_pred_dense_protocol_v1/result.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(config_path: Path, smoke_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_CAUSAL_BOUNDARIES_PRED_CACHE_PENDING":
        raise ValueError("unexpected protocol status")
    if config.get("data_protocol", {}).get("video_scope") != 407:
        raise ValueError("protocol is not bound to 407 approved Train videos")
    shards = config.get("predicted_dense_cache", {}).get("shards", [])
    if len(shards) != 4:
        raise ValueError("expected exactly four fixed dense shards")
    expected_start = 0
    for shard in shards:
        if int(shard["video_id_min"]) != expected_start:
            raise ValueError("dense shard ranges are not contiguous")
        if int(shard["video_id_max"]) < int(shard["video_id_min"]):
            raise ValueError("invalid dense shard range")
        count = int(shard["video_id_max"]) - int(shard["video_id_min"]) + 1
        if int(shard["video_count"]) != count:
            raise ValueError("dense shard count mismatch")
        expected_start = int(shard["video_id_max"]) + 1
    if expected_start != 407:
        raise ValueError("dense shard ranges do not cover exactly 0..406")
    if config.get("predicted_dense_cache", {}).get("full_cache_automatic_start") is not False:
        raise ValueError("full predicted cache is not fail-closed")
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
        "status": "PASS_FROZEN_JOINT_CAUSAL_BOUNDARIES_PRED_CACHE_PENDING",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": str(config_path.resolve()),
        "protocol_sha256": _sha256(config_path),
        "smoke_receipt": str(smoke_path.resolve()),
        "smoke_receipt_sha256": _sha256(smoke_path),
        "repository_head_at_audit": head,
        "fixed_dense_shards": shards,
        "smoke_counts": smoke.get("counts", {}),
        "boundaries": {
            "full_predicted_cache_started": False,
            "training_started": False,
            "formal_metrics_created": False,
            "tao_val_or_test_labels_used": False,
            "novel_vocabulary_used": False,
            "private_gt_arrays_read_by_audit": False,
            "model_selection_performed": False,
        },
        "interpretation": "Protocol boundary audit only; Goal A/B/C, full predicted cache, formal OCD/MOT, and frontend qualification remain open.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.config.resolve(), args.smoke.resolve(), args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
