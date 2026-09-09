#!/usr/bin/env python3
"""Create isolated Phase89 output namespace with read-only Phase88 manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase89"
SOURCE = ROOT / "outputs/iclr27_phase88/manifests"


def main() -> None:
    for name in ("audit", "checkpoints", "completion", "logs", "metrics", "validation", "diagnostic"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    link = OUT / "manifests"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != SOURCE.resolve():
            raise RuntimeError(f"existing Phase89 manifests path is not the approved read-only link: {link}")
    else:
        link.symlink_to(SOURCE.resolve(), target_is_directory=True)
    payload = {
        "phase": 89,
        "status": "PREPARED",
        "read_only_manifest_source": str(SOURCE.resolve()),
        "read_only_manifest_target": str(link.resolve()),
        "architecture": "hierarchical_open_world_router",
        "physical_stream_frozen": True,
        "support_mode": False,
        "public_dev_q1_sealed_accessed": False,
    }
    (OUT / "audit/phase89_namespace.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
