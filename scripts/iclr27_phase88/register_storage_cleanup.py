#!/usr/bin/env python3
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
ARCH = Path("/data2/usr_for_deadline/trackocd_archive/20260909/project_cleanup")

def size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def main() -> None:
    entries = []
    for rel in ("outputs/iclr27_phase54", "outputs/iclr27_phase69", "outputs/iclr27_phase70", "data/caches/features"):
        p = ROOT / rel
        target = Path(p.resolve())
        entries.append({
            "original_path": str(p),
            "archive_target": str(target),
            "symlink": str(p.is_symlink()),
            "archived_bytes": size(target),
            "status": "ARCHIVED_RECOVERABLE",
            "reason": "historical or regenerable project-local data not used by active Phase88 worker",
        })
    artifact = {
        "schema_version": "trackocd.phase88.storage_cleanup.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "filesystem_before": "data1 nearly full; cleanup was requested by user",
        "archive_root": str(ARCH),
        "entries": entries,
        "active_worker_preserved": "Phase88 H1 worker/checkpoints/manifests/logs were not removed",
        "recovery": "restore by replacing the symlink with the archived directory if needed",
        "external_processes_touched": False,
    }
    out = OUT / "audit/storage_cleanup_20260909.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)

if __name__ == "__main__":
    main()
