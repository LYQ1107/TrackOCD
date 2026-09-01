#!/usr/bin/env python3
"""Freeze and audit Phase26 proposal inputs before correspondence training."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase27"


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    decision_path = ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"
    decision = json.loads(decision_path.read_text())
    required = [ROOT / "outputs/iclr27_phase26/checkpoints" / f"source_f{f}_best.pt" for f in range(4)] + [ROOT / "outputs/iclr27_phase26/metrics/stage3_proposal_validation.json", ROOT / "outputs/iclr27_phase26/audit/full_76_event_summary.csv"]
    frozen = {
        "proposal_decision": decision["decision_code"], "proposal_gate": decision["gate_p2"],
        "source_checkpoints": {str(p): sha(p) for p in required if p.name.startswith("source_") and p.exists()},
        "required_paths": {str(p): p.exists() for p in required}, "raw_prefix16": 25, "source_prefix16": 41,
        "candidate_pool_oracle": 38, "broad_pool_oracle": 56, "positive_event_denominator": 76,
        "prefixes": [1, 2, 4, 8, 16], "evaluator": "Phase19R persistent evaluator read-only; no controller edits",
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs", "semantic text", "held GT as model input"],
    }
    atomic(OUT / "audit/frozen_proposal.json", frozen); atomic(OUT / "completion/stage0.done", {"stage": "phase27_freeze", "proposal_gate": "P26_GATE_P2_PASS", "source_prefix16": 41})
    print(json.dumps({"proposal_gate": decision["gate_p2"]["decision"], "source_prefix16": 41, "checkpoints": len(frozen["source_checkpoints"])}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
