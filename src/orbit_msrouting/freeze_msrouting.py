"""Freeze an ORBIT-MSRouting candidate before official validation (SHA-256).

Records the exact checkpoint, gate mode, state features, compatibility
policy, and selection evidence.  The official run may only use frozen
candidates; the freeze file must predate the official result.
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
    ap.add_argument("--candidate", required=True, choices=["A", "B"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_mode", required=True, choices=["G0", "G1", "G2"])
    ap.add_argument("--state_feats", default="")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--selection_evidence", required=True)
    args = ap.parse_args()

    ck_path = ROOT / args.checkpoint
    if not ck_path.exists():
        raise SystemExit(f"checkpoint not found: {ck_path}")
    sha = hashlib.sha256(ck_path.read_bytes()).hexdigest()
    doc = {
        "candidate": args.candidate,
        "checkpoint": str(ck_path),
        "checkpoint_sha256": sha,
        "gate_mode": args.gate_mode,
        "state_feats": [f.strip() for f in args.state_feats.split(",")
                        if f.strip()],
        "gate_threshold": args.gate_threshold,
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
        "state_window": args.window,
        "selection_evidence": args.selection_evidence,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "official_validation": "PENDING",
    }
    out = ROOT / "outputs/orbit_msrouting/frozen_candidates" / \
        f"candidate_{args.candidate.lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
