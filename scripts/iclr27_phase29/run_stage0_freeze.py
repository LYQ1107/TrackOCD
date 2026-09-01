#!/usr/bin/env python3
"""Freeze Phase29 inputs and record the single-route boundary."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase29"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p26_dec = ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"
    p28_dec = ROOT / "outputs/iclr27_phase28/audit/phase28_decision.json"
    p28_diag = ROOT / "outputs/iclr27_phase28/audit/compatibility_confusion_diagnostic.json"
    manifest = OUT / "manifests/fold_manifest.json"
    pos = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
    csv_path = ROOT / "data/InterMOT/InterMOT_train.csv"
    # Resolve actual frozen feature/CSV locations through the Phase26 facade;
    # no files are copied into Phase29.
    from src.iclr27_phase29.protocol import CSV_PATH, FEAT_PATH
    required = [p26_dec, p28_dec, p28_diag, manifest, pos, CSV_PATH, FEAT_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    p26 = json.loads(p26_dec.read_text())
    p28 = json.loads(p28_dec.read_text())
    events = [json.loads(x) for x in pos.read_text().splitlines() if x.strip()]
    if len(events) != 76:
        raise RuntimeError(f"positive event denominator changed: {len(events)}")
    if p26.get("decision_code") != "P26_GATE_P2_PASS_AUTHORIZE_CORRESPONDENCE":
        raise RuntimeError("Phase26 proposal is not the frozen PASS comparator")
    if p28.get("decision_code") != "P28_GATE_C_FAIL_STOP_BEFORE_NEW_REPRESENTATION":
        raise RuntimeError("Phase28 frozen compatibility decision is not the expected FAIL")
    try:
        nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    except Exception as exc:
        nvidia = f"nvidia-smi unavailable: {exc!r}"
    payload = {
        "protocol": "trackocd_iclr27_phase29_stage0_frozen_domain_alignment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_frozen": True,
        "controller_frozen": True,
        "phase26_decision": str(p26_dec),
        "phase26_decision_sha256": sha(p26_dec),
        "phase28_decision": str(p28_dec),
        "phase28_decision_sha256": sha(p28_dec),
        "phase28_diagnostic_sha256": sha(p28_diag),
        "source_checkpoint_links": {str(p): str(p.resolve()) for p in sorted((OUT / "checkpoints").glob("source_f*_best.pt"))},
        "source_checkpoint_hashes": {str(p.resolve()): sha(p) for p in sorted((OUT / "checkpoints").glob("source_f*_best.pt"))},
        "csv_path": str(CSV_PATH), "csv_sha256": sha(CSV_PATH),
        "feature_path": str(FEAT_PATH), "feature_sha256": sha(FEAT_PATH),
        "fold_manifest": str(manifest), "fold_manifest_sha256": sha(manifest),
        "positive_event_manifest": str(pos), "positive_event_manifest_sha256": sha(pos),
        "positive_event_denominator": len(events),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "semantic text", "physical/semantic IDs as inputs", "held/test GT as input"],
        "representation_route": "DomainAlignedResidualEncoder; causal mean/last/abs-delta; zero-initialized residual scale 0.10; no GRU/backbone",
        "training_protocol": {"fit_source": "public TRAIN rows/GT metadata only", "folds": 4, "video_disjoint": True, "category_disjoint": True, "devices": [4,5,6,7], "batch_size": 32, "steps": 2000, "checkpoint_every": 500, "amp": "bf16_if_finite_else_fp32", "seed_base": 20262901},
        "phase28_finding": "3/76 correct commits are fold3/category81/source video575 only; folds0-2 zero; no controller safety-preserving broad gain",
        "nvidia_smi_preflight": nvidia,
        "public_evaluation_started": False,
        "sealed": True,
    }
    atomic(OUT / "audit/frozen_inputs.json", payload)
    (OUT / "completion/stage0.done").write_text(json.dumps({"positive_event_denominator": 76, "proposal_frozen": True, "controller_frozen": True}, sort_keys=True) + "\n")
    print(json.dumps({"stage0": "done", "output": str(OUT / "audit/frozen_inputs.json"), "positive_event_denominator": 76}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
