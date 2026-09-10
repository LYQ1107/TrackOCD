#!/usr/bin/env python3
"""Prepare an isolated Phase90 namespace without copying shared features."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase90"
TARGET = Path("/data2/usr_for_deadline/trackocd_phase90/project_outputs")
SOURCE_MANIFESTS = (ROOT / "outputs/iclr27_phase88/manifests").resolve()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and not OUT.is_symlink():
        raise RuntimeError(f"Phase90 output path exists and is not a symlink: {OUT}")
    if not OUT.exists():
        OUT.symlink_to(TARGET, target_is_directory=True)
    for name in ("audit", "checkpoints", "completion", "logs", "metrics", "validation", "diagnostic"):
        (TARGET / name).mkdir(parents=True, exist_ok=True)
    link = TARGET / "manifests"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != SOURCE_MANIFESTS:
            raise RuntimeError(f"Phase90 manifests link is not the approved read-only source: {link}")
    else:
        link.symlink_to(SOURCE_MANIFESTS, target_is_directory=True)
    payload = {
        "phase": 90,
        "status": "PREPARED",
        "output_target": str(TARGET),
        "output_link": str(OUT),
        "read_only_manifest_source": str(SOURCE_MANIFESTS),
        "read_only_manifest_target": str(link.resolve()),
        "phase89_frozen": True,
        "physical_stream_frozen": True,
        "support_mode": False,
        "public_dev_q1_sealed_accessed": False,
    }
    atomic_json(TARGET / "audit/phase90_namespace.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
