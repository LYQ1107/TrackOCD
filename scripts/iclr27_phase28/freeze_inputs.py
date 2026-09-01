#!/usr/bin/env python3
"""Record read-only Phase26/Phase19R inputs for the compatibility diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase28"


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    phase26_decision = ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"
    phase26_report = ROOT / "docs/iclr27_phase26/PHASE26_PROPOSAL_SOURCE_CANDIDATE_COVERAGE_COMPLETE_REPORT.md"
    old_manifest = ROOT / "outputs/iclr27_phase19r/manifests/fold_manifest.json"
    positive = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
    negative = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
    csv_path = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
    feat_path = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
    decision = json.loads(phase26_decision.read_text())
    required = [phase26_decision, phase26_report, old_manifest, positive, negative, csv_path, feat_path]
    old_ckpts = [ROOT / "outputs/iclr27_phase19r/checkpoints" / f"fold{f}_best_internal.pt" for f in range(4)]
    source_ckpts = [ROOT / "outputs/iclr27_phase26/checkpoints" / f"source_f{f}_best.pt" for f in range(4)]
    required += old_ckpts + source_ckpts
    frozen = {
        "protocol": "trackocd_iclr27_phase28_frozen_representation_compatibility",
        "phase26_decision": decision.get("decision_code"),
        "phase26_gate_p2": decision.get("gate_p2"),
        "phase26_source_prefix16": 41,
        "raw_prefix16": 25,
        "positive_event_denominator": 76,
        "old_controller": "Phase19R RC-MS-OCD checkpoint fold*_best_internal.pt; StateMemory/threshold/action semantics unchanged",
        "representation": "original normalized fused DINOv2 CLS/ROI used by Phase19R",
        "evaluator": "src/iclr27_phase19r/evaluation/internal.py read-only",
        "fold_manifest": str(old_manifest),
        "fold_manifest_sha256": sha(old_manifest),
        "positive_events_sha256": sha(positive),
        "negative_events_sha256": sha(negative),
        "csv_sha256": sha(csv_path),
        "feature_sha256": sha(feat_path),
        "old_controller_checkpoints": {str(p): sha(p) for p in old_ckpts},
        "phase26_source_checkpoints": {str(p): sha(p) for p in source_ckpts},
        "required_paths": {str(p): p.exists() for p in required},
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs as features", "semantic text", "held GT as model input"],
        "training": False,
        "threshold_sweep": False,
        "proposal_changed": False,
        "public_evaluation": False,
    }
    atomic(OUT / "audit/frozen_inputs.json", frozen)
    (OUT / "completion/stage0.done").write_text(json.dumps({"stage": "phase28_freeze", "source_prefix16": 41, "positive_denominator": 76}, sort_keys=True) + "\n")
    print(json.dumps({"phase26_gate": frozen["phase26_gate_p2"], "source_prefix16": 41, "positive_denominator": 76, "required_ok": all(frozen["required_paths"].values())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
