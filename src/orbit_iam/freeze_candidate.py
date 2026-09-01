"""Freeze an ORBIT-IAM official candidate BEFORE any official evaluation.

Writes outputs/orbit_iam/frozen_candidates/candidate_<x>.json with SHA256
of the checkpoint, the exact thresholds, and the train-side meta-dev
evidence used for selection. Candidate B is refused while the
multi-prototype gate is NOT_JUSTIFIED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate", choices=["A", "B"], required=True)
    ap.add_argument("--compat_threshold", type=float, required=True)
    ap.add_argument("--compat_margin", type=float, required=True)
    ap.add_argument("--selection_evidence", default="")
    args = ap.parse_args()

    if args.candidate == "B":
        gate = (ROOT / "docs/iclr27_phase4e/MULTI_PROTOTYPE_JUSTIFICATION.md")
        if "MULTI_PROTOTYPE_NOT_JUSTIFIED" in gate.read_text():
            raise SystemExit("Candidate B refused: multi-prototype gate NOT_JUSTIFIED")

    ckpt = ROOT / args.checkpoint
    assert ckpt.exists(), ckpt
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    out_dir = ROOT / "outputs/orbit_iam/frozen_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "candidate": args.candidate,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha,
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
        "gate_threshold": 0.5,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_evidence": args.selection_evidence,
        "official_validation": "NOT_YET_RUN",
    }
    out = out_dir / f"candidate_{args.candidate.lower()}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
