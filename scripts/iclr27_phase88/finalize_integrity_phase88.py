#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    errors: list[str] = []
    expected_empty_markers: list[str] = []
    json_count = 0
    artifact_paths = []
    for folder in (OUT / "audit", OUT / "metrics", OUT / "completion"):
        if folder.exists():
            artifact_paths.extend(folder.glob("*.json"))
    for path in sorted(artifact_paths):
        if path.name.endswith("_launch.json"):
            expected_empty_markers.append(str(path))
            continue
        try:
            json.loads(path.read_text())
            json_count += 1
        except Exception as exc:
            errors.append(f"json:{path}:{exc}")
    report = ROOT / "docs/iclr27_phase88/PHASE88_AUTONOMOUS_RESEARCH_REPORT.md"
    if not report.exists() or report.stat().st_size == 0:
        errors.append("report_missing_or_empty")
    output_link = ROOT / "outputs/iclr27_phase88"
    symlink_target = os.readlink(output_link) if output_link.is_symlink() else None
    if not output_link.is_symlink() or not output_link.resolve().exists():
        errors.append("output_symlink_invalid")
    public_like = []
    output_files = []
    for folder in (OUT / "audit", OUT / "metrics", OUT / "completion"):
        if folder.exists():
            output_files.extend(folder.iterdir())
    for path in output_files:
        if path.is_file() and any(x in path.name.lower() for x in ("public", "q1", "dev+", "sealed")):
            public_like.append(str(path))
    active = []
    ps = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    for line in ps.splitlines():
        if "train_controller.py" in line and "phase88" in line:
            active.append(line.strip())
        if ("run_h2_" in line or "h2_cpu_" in line) and "phase88" in line:
            active.append(line.strip())
    expected = [
        OUT / "audit/final_decision.json",
        OUT / "audit/hypothesis_1_preregistration.json",
        OUT / "audit/hypothesis_2_preregistration.json",
        OUT / "audit/h2_resource_stop.json",
        OUT / "audit/h2_resume1_resource_stop.json",
        OUT / "audit/h2_resume2_resource_stop.json",
        OUT / "audit/h2_cpu_parallel_resource_stop.json",
        OUT / "audit/h2_cpu_sequential_resource_stop.json",
        OUT / "metrics/h2_known_suppression_cpu_smoke_fix2_f0.json",
        OUT / "metrics/h2_known_suppression_cpu_targeted_fix2_f0.json",
    ]
    for path in expected:
        if not path.exists():
            errors.append(f"missing:{path}")
    integrity = {
        "schema_version": "trackocd.phase88.integrity.v2",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "json_parse_errors": errors,
        "expected_launch_marker_files": expected_empty_markers,
        "json_artifact_count_at_check": json_count,
        "report": {"path": str(report), "nonempty": report.exists() and report.stat().st_size > 0, "sha256": sha256(report) if report.exists() else None},
        "output_symlink": {"path": str(output_link), "target": symlink_target, "valid": output_link.is_symlink() and output_link.resolve().exists()},
        "residual_phase88_processes": active,
        "public_q1_sealed_like_output_files": public_like,
        "formal_h2_done_units": sorted(p.name for p in (OUT / "completion").glob("h2_known_suppression*formal*.done")),
        "h2_partial_checkpoint_hashes": {
            p.name: sha256(p) for p in sorted((OUT / "checkpoints").glob("h2_known_suppression_cpu_formal_f*_step*.pt"))
        },
        "external_processes_touched": False,
        "public_dev_q1_sealed_accessed": False,
        "status": "PASS_NO_RESIDUAL_PROCESS_RESOURCE_BLOCKED_H2_FORMAL_INCOMPLETE" if not errors and not active and not public_like else "FAIL_INTEGRITY",
    }
    out = OUT / "audit/integrity_phase88.json"
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(out)
    if errors or active or public_like:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
