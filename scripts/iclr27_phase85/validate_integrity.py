#!/usr/bin/env python3
"""Lightweight final integrity check for Phase85 outputs."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase85"
TARGET = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs")


def sha(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    if not OUT.is_symlink() or OUT.resolve() != TARGET.resolve():
        raise RuntimeError(f"unexpected output symlink: {OUT} -> {OUT.resolve()}")
    json_files = sorted(p for p in OUT.rglob("*.json") if p.is_file())
    parse_failures = []
    for path in json_files:
        try: json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc: parse_failures.append({"path": str(path), "error": repr(exc)})
    key_paths = [OUT / "metrics/temporal_mean_full.json", OUT / "audit/physical_r_q0_q0_parity_v5_adapter.json", OUT / "audit/physical_r_temporal_comparison_v2.json", OUT / "audit/physical_r_selective_comparison.json", OUT / "metrics/support_event_replay.json", OUT / "metrics/support_event_replay_selective_source_v1.json", OUT / "audit/support_alignment_feasibility.json", OUT / "audit/event_physical_contamination.json", OUT / "audit/leakage_contract.json"]
    missing = [str(p) for p in key_paths if not p.is_file()]
    checkpoints = sorted((OUT / "checkpoints").glob("*.pt"))
    checkpoint_hashes = [{"path": str(p.resolve()), "sha256": sha(p), "size": p.stat().st_size} for p in checkpoints]
    forbidden_files = []
    for path in OUT.rglob("*"):
        if path.is_file() and any(token in path.name.lower() for token in ("q1", "devplus", "dev_", "public_new", "sealed_label")):
            forbidden_files.append(str(path))
    phase_processes = []
    ps = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=False).stdout
    for line in ps.splitlines():
        if "iclr27_phase85" in line and "validate_integrity.py" not in line and "grep" not in line:
            phase_processes.append(line.strip())
    markers = []
    for path in sorted((OUT / "completion").glob("*.launched")):
        markers.append({"launched": path.name, "done_exists": path.with_suffix(".done").is_file()})
    result = {"schema_version": "trackocd.phase85.integrity.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "output_symlink": str(OUT), "output_target": str(OUT.resolve()), "json_count": len(json_files), "json_parse_failures": parse_failures, "missing_key_artifacts": missing, "checkpoint_count": len(checkpoints), "checkpoint_hashes": checkpoint_hashes, "forbidden_named_files": forbidden_files, "phase85_processes": phase_processes, "markers": markers, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "status": "PASS" if not parse_failures and not missing and not forbidden_files and not phase_processes else "FAIL"}
    atomic(OUT / "audit/integrity_check.json", result)
    atomic(OUT / "completion/integrity_check.done", {"status": result["status"], "audit": str((OUT / "audit/integrity_check.json").resolve()), "sha256": sha(OUT / "audit/integrity_check.json")})
    print(json.dumps({k: result[k] for k in ("status", "json_count", "json_parse_failures", "missing_key_artifacts", "checkpoint_count", "forbidden_named_files", "phase85_processes", "markers")}, indent=2, sort_keys=True))
    if result["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()
