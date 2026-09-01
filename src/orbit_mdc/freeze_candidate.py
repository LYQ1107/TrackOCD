"""Freeze an ORBIT-MDC candidate before official validation (SHA-256)."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["A", "B"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--birth_threshold", type=float, default=0.5)
    ap.add_argument("--policy", choices=["auto", "compat", "birth"],
                    default="auto")
    ap.add_argument("--quarantine_mode", type=int, default=0)
    ap.add_argument("--quarantine_support_thr", type=int, default=3)
    ap.add_argument("--quarantine_dispersion_thr", type=float, default=0.3)
    ap.add_argument("--quarantine_coef", type=float, default=1.0)
    ap.add_argument("--selection_evidence", required=True)
    args = ap.parse_args()
    ck_path = ROOT / args.checkpoint
    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    doc = {
        "candidate": args.candidate,
        "checkpoint": str(ck_path),
        "checkpoint_sha256": sha,
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
        "birth_threshold": args.birth_threshold,
        "policy": args.policy,
        "quarantine_mode": args.quarantine_mode,
        "quarantine_support_thr": args.quarantine_support_thr,
        "quarantine_dispersion_thr": args.quarantine_dispersion_thr,
        "quarantine_coef": args.quarantine_coef,
        "selection_evidence": args.selection_evidence,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "official_validation": "PENDING",
    }
    out = ROOT / "outputs/orbit_mdc/frozen_candidates" / f"candidate_{args.candidate.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
