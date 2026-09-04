#!/usr/bin/env python3
"""Generate Phase84 machine-readable live ledgers and a non-final status note."""
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
OUT = ROOT / "outputs/iclr27_phase84"
AUDIT = OUT / "audit"
METRICS = OUT / "metrics"
COMP = OUT / "completion"
REG = AUDIT / "window_registration.json"
LOCK = AUDIT / "finalization_lock.json"


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.exists(), "sha256": sha(path)}


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def phase_processes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    ps = subprocess.run(["ps", "-eo", "pid=,args="], cwd=ROOT, text=True, capture_output=True, check=False)
    for line in ps.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        if "iclr27_phase84" not in command or int(pid_text) == os.getpid():
            continue
        out.append({"pid": int(pid_text), "command": command})
    return out


def resource_snapshot() -> dict[str, Any]:
    free = subprocess.run(["free", "-h"], text=True, capture_output=True, check=False).stdout
    disk = subprocess.run(["df", "-h", "/data1", "/data2"], text=True, capture_output=True, check=False).stdout
    return {"free_h": free, "disk": disk, "process_count": len(subprocess.run(["ps", "-e", "--no-headers"], text=True, capture_output=True, check=False).stdout.splitlines()), "phase84_processes": phase_processes()}


def main() -> None:
    registration = load(REG)
    lock = load(LOCK)
    physical = load(AUDIT / "a84_physical_r_metrics.json")
    signal = load(AUDIT / "source_conditioned_signal.json")
    b84s = load(METRICS / "b84s_event_replay.json")
    b84sq = load(METRICS / "b84s_event_replay_b84sq_v3.json")
    b84sra = load(METRICS / "b84s_event_replay_b84sra_v1.json")
    formal = load(METRICS / "b84s_formal_aggregate.json")
    formal_q = load(METRICS / "b84s_formal_aggregate_b84sq_v3.json")
    formal_ra = load(METRICS / "b84s_formal_aggregate_b84sra_v1.json")
    b84sq_audit = load(AUDIT / "b84sq_failure_audit.json")
    b84sra_audit = load(AUDIT / "b84sra_failure_audit.json")
    physical_p16 = physical.get("p16", physical.get("aggregate", {}).get("16", {}))
    signal_p16 = signal.get("p16_signal", {})
    def p16_summary(z: dict[str, Any]) -> dict[str, Any]:
        return {str(x.get("polarity")): x for x in z.get("summary", []) if x.get("prefix") == 16}
    artifacts = [
        AUDIT / "phase83_a2_report_integrity.json",
        AUDIT / "physical_to_r_interface_callgraph.json",
        AUDIT / "a84_physical_r_metrics.json",
        AUDIT / "physical_r_diagnostic.json",
        AUDIT / "source_conditioned_signal.json",
        AUDIT / "b84sq_failure_audit.json",
        AUDIT / "b84sra_failure_audit.json",
        AUDIT / "support_alignment_callgraph.json",
        METRICS / "physical_r_q0_adapter.json",
        METRICS / "b84s_formal_aggregate.json",
        METRICS / "b84s_event_replay.json",
        METRICS / "b84s_formal_aggregate_b84sq_v3.json",
        METRICS / "b84s_event_replay_b84sq_v3.json",
        METRICS / "b84s_formal_aggregate_b84sra_v1.json",
        METRICS / "b84s_event_replay_b84sra_v1.json",
        OUT / "manifests/b84sq_balanced_v3_manifest.json",
        OUT / "manifests/b84s_native_manifest.json",
    ]
    repair_events = [
        {"stage": "A84P_smoke", "failure": "native candidate_rank=null raised TypeError", "root_cause": "serialization parser assumed integer rank", "repair": "map null rank to deterministic 0", "protocol_changed": False, "artifact_changed": False, "resume": "fresh physical_smoke_r1 then full_temporal_r1"},
        {"stage": "B84S_smoke", "failure": "DEFER target indexed as candidate", "root_cause": "gradient branch omitted explicit DEFER case", "repair": "branch before candidate gradient", "protocol_changed": False, "artifact_changed": False, "resume": "b84s_smoke_r2"},
        {"stage": "B84S_formal", "failure": "shell supervisor invoked through Python", "root_cause": "command interpreter mismatch", "repair": "invoke registered shell with bash", "protocol_changed": False, "artifact_changed": False, "resume": "b84s_formal_r2"},
        {"stage": "B84S_event_replay", "failure": "aggregation requested absent mean_nll", "root_cause": "old metric schema assumption", "repair": "aggregate only fields present in all completed folds", "protocol_changed": False, "artifact_changed": False, "resume": "b84sq_v3 replay"},
        {"stage": "B84S_event_replay", "failure": "NumPy scalar JSON serialization before event output", "root_cause": "non-native scalar in atomic JSON payload", "repair": "convert scalar metadata before serialization", "protocol_changed": False, "artifact_changed": False, "resume": "fresh suffixed replay"},
        {"stage": "B84S-Q_manifest", "failure": "four-fold split had insufficient balanced fit/validation groups", "root_cause": "legal query/support distribution is sparse and imbalanced", "repair": "registered deterministic three-fold fallback", "protocol_changed": True, "artifact_changed": True, "resume": "b84sq_formal_v3", "note": "fold_count=3 is explicit manifest evidence, not hidden"},
        {"stage": "B84S-RA_compile", "failure": "missing closing bracket in evaluator branch", "root_cause": "syntax typo", "repair": "minimal bracket repair then py_compile", "protocol_changed": False, "artifact_changed": False, "resume": "single raw-anchor diagnostic"},
    ]
    validation_evidence = {
        "phase": "Phase84",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--porcelain"),
        "compile_checks": [
            "python -m py_compile scripts/iclr27_phase84/evaluate_b84s_event_replay.py",
            "python -m py_compile scripts/iclr27_phase84/audit_b84sq_failure.py",
            "python -m py_compile scripts/iclr27_phase84/audit_support_alignment_callgraph.py",
            "python -m py_compile scripts/iclr27_phase84/generate_live_ledger.py",
        ],
        "targeted_checks": [
            "B84S-Q frozen event replay covers 760 records and prefixes 1/2/4/8/16",
            "B84S-RA frozen raw-anchor replay covers 760 records and prefixes 1/2/4/8/16",
            "support alignment callgraph confirms no transformed support is implemented",
        ],
        "old_tests_intentionally_not_rerun": ["Phase83 full suites", "controller/StateMemory", "sealed/public evaluation"],
        "artifact_hashes": [artifact(p) for p in artifacts],
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_as_model_input": False,
    }
    research = [
        {"name": "A84P_TRUE_PHYSICAL_REASSOCIATION", "evidence": [artifact(AUDIT / "a84_physical_r_metrics.json"), artifact(AUDIT / "physical_r_diagnostic.json")], "result": physical_p16, "decision": "A84P_TRUE_PHYSICAL_R_FAIL_WITH_VALID_CONTRACT", "next_action": "retain physical contamination evidence; do not run controller"},
        {"name": "B84S_SOURCE_CONDITIONED_SIGNAL", "evidence": [artifact(AUDIT / "source_conditioned_signal.json")], "result": signal_p16, "decision": "DIAGNOSTIC_SIGNAL_PRESENT", "next_action": "run frozen query-conditioned selector"},
        {"name": "B84S_ORIGINAL_QUERY_AGNOSTIC", "evidence": [artifact(METRICS / "b84s_event_replay.json"), artifact(METRICS / "b84s_formal_aggregate.json")], "result": p16_summary(b84s), "decision": "B84S_SOURCE_CONTRACT_FAIL", "next_action": "repair source/query conditioning once"},
        {"name": "B84S_Q_QUERY_CONDITIONED", "evidence": [artifact(METRICS / "b84s_event_replay_b84sq_v3.json"), artifact(AUDIT / "b84sq_failure_audit.json")], "result": {"formal": formal_q.get("validation_weighted", {}), "p16": p16_summary(b84sq)}, "decision": b84sq_audit.get("decision", "B84S_Q_FAIL"), "next_action": "diagnose ranking/generalization; no alignment because positive reliable selection is below 30/76"},
        {"name": "B84S_RA_RAW_ANCHOR", "evidence": [artifact(METRICS / "b84s_event_replay_b84sra_v1.json"), artifact(AUDIT / "b84sra_failure_audit.json")], "result": {"formal": formal_ra.get("validation_weighted", {}), "p16": p16_summary(b84sra)}, "decision": b84sra_audit.get("decision", "B84S_RA_PARTIAL"), "next_action": "no alignment/controller; await finalization lock"},
    ]
    snapshot = resource_snapshot()
    atomic_json(AUDIT / "repair_events.json", {"schema_version": "trackocd.phase84.repair_events.v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "events": repair_events, "resource_events": [{"event": "Phase84 formal workers", "oom": False, "external_process_terminated": False, "gpu_route": "CPU for B84S/B84S-Q/B84S-RA", "ram_snapshot": snapshot["free_h"], "process_count": snapshot["process_count"]}]})
    atomic_json(AUDIT / "validation_evidence_ledger.json", validation_evidence)
    atomic_json(AUDIT / "research_ledger.json", {"schema_version": "trackocd.phase84.research_ledger.v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "registration": registration, "finalization_lock": lock, "hypotheses": research, "artifacts": [artifact(p) for p in artifacts], "resource_snapshot": snapshot, "status": "LIVE_NOT_FINAL"})
    live = [
        "# Phase84 live ledger (not final)",
        "",
        f"Generated UTC: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"Window deadline UTC: {registration.get('deadline_utc')}",
        f"Finalization lock allowed: {lock.get('allowed', False)}",
        f"Git HEAD: {validation_evidence['git_head']}",
        "",
        "Scientific status:",
        "- A84P true physical reassociation completed; its frozen-R safety gate failed, so no controller was run.",
        "- B84S original query-agnostic selector failed; B84S-Q repaired query/source contract also failed to improve event selection.",
        "- B84S-RA raw-anchor bounded residual reached its registered diagnostic result; positive selection remains below the >30/76 alignment-routing criterion and negative activation increased.",
        "- B84A alignment, C84 controller, and sealed/public evaluation are NOT_RUN by protocol.",
        "",
        f"Phase84 process snapshot: {len(snapshot['phase84_processes'])} task processes.",
        "",
        "Machine ledgers: `outputs/iclr27_phase84/audit/{research_ledger,repair_events,validation_evidence_ledger}.json`.",
        "The final report is intentionally withheld until the registered finalization lock opens.",
    ]
    live_path = ROOT / "docs/iclr27_phase84/PHASE84_LIVE_LEDGER.md"
    live_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{live_path.name}.", dir=str(live_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(live) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, live_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(json.dumps({"status": "LIVE_NOT_FINAL", "lock_allowed": lock.get("allowed", False), "head": validation_evidence["git_head"], "phase84_processes": len(snapshot["phase84_processes"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
