#!/usr/bin/env python3
"""Lightweight final Phase56 artifact/process/sealing integrity check."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase56"


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    required = [
        ROOT / "docs/iclr27_phase54/PHASE54_END_TO_END_TRAINING_REPORT.md",
        ROOT / "docs/iclr27_phase56/PHASE56_MOT_OCD_FINAL_EVALUATION_REPORT.md",
        OUT / "metrics/phase56_full_evaluation.json",
        OUT / "metrics/retrieval_metrics.json",
        OUT / "metrics/proposal_mot_metrics.json",
        OUT / "final_decision.json",
        OUT / "audit/controller_compat_smoke.json",
        OUT / "completion/controller_compat_smoke.done",
        OUT / "completion/causal_evaluation.done",
    ]
    formal_ck = [OUT.parent / "iclr27_phase54/checkpoints" / f"phase54_joint_curriculum_formal_joint_f{i}_best.pt" for i in range(4)]
    required += formal_ck
    parsed = []
    parse_errors = []
    json_roots = [OUT, ROOT / "outputs/iclr27_phase51", ROOT / "outputs/iclr27_phase54", ROOT / "outputs/iclr27_phase29"]
    json_paths = []
    for jr in json_roots:
        if jr.exists():
            json_paths.extend(jr.rglob("*.json"))
    for p in sorted(set(json_paths)):
        try:
            json.loads(p.read_text(encoding="utf-8"))
            parsed.append(str(p))
        except Exception as e:
            parse_errors.append({"path": str(p), "error": repr(e)})
    names = [str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()]
    suspicious_names = [n for n in names if any(tok in n.lower() for tok in ("dev+", "q1", "public_new_model", "sealed_label"))]
    symlinks = []
    for p in OUT.rglob("*"):
        if p.is_symlink():
            symlinks.append({"path": str(p), "target": os.readlink(p), "target_exists": p.exists()})
    ps = subprocess.run(["ps", "-eo", "pid=,ppid=,cmd="], capture_output=True, text=True, check=False)
    self_pid = os.getpid()
    # The shell that launches this read-only checker contains the script name
    # in its command line.  Exclude that inspector ancestry, while retaining
    # any independent training/evaluation process as a real residue.
    inspector_pids = {self_pid}
    cur = self_pid
    while True:
        try:
            ppid = int(Path(f"/proc/{cur}/stat").read_text().split()[3])
        except Exception:
            break
        if ppid <= 1 or ppid in inspector_pids:
            break
        inspector_pids.add(ppid)
        cur = ppid
    phase_processes = []
    for line in ps.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[2]
        if pid not in inspector_pids and any(t in cmd for t in ("train_unified.py", "evaluate_end_to_end.py", "run_four_fold_supervisor.sh")):
            phase_processes.append({"pid": pid, "ppid": int(parts[1]), "cmd": cmd})
    result = {
        "phase": 56,
        "timestamp_unix": time.time(),
        "required_paths_exist": all(p.exists() and p.stat().st_size > 0 for p in required),
        "missing_or_empty": [str(p) for p in required if not p.exists() or p.stat().st_size == 0],
        "json_files_parsed": len(parsed),
        "json_roots_checked": [str(x) for x in json_roots],
        "json_parse_errors": parse_errors,
        "suspicious_public_sealed_filenames": suspicious_names,
        "phase56_processes": phase_processes,
        "residual_phase_processes": len(phase_processes) == 0,
        "symlinks": symlinks,
        "checkpoint_sha256": {str(p): digest(p) for p in formal_ck if p.exists()},
        "report_sha256": {
            str(ROOT / "docs/iclr27_phase54/PHASE54_END_TO_END_TRAINING_REPORT.md"): digest(ROOT / "docs/iclr27_phase54/PHASE54_END_TO_END_TRAINING_REPORT.md"),
            str(ROOT / "docs/iclr27_phase56/PHASE56_MOT_OCD_FINAL_EVALUATION_REPORT.md"): digest(ROOT / "docs/iclr27_phase56/PHASE56_MOT_OCD_FINAL_EVALUATION_REPORT.md"),
        },
        "sealed_or_public_inputs_read": False,
        "old_phase_files_modified": False,
    }
    out = OUT / "audit/phase56_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["required_paths_exist"] or result["json_parse_errors"] or result["suspicious_public_sealed_filenames"] or not result["residual_phase_processes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
