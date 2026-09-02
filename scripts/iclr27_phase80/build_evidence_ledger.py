#!/usr/bin/env python3
"""Build the machine-readable Phase80 validation/decision ledger."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def main() -> None:
    start_utc = dt.datetime.fromisoformat("2026-09-02T18:27:32+00:00")
    deadline_utc = dt.datetime.fromisoformat("2026-09-03T04:27:32+00:00")
    now_utc = dt.datetime.now(dt.timezone.utc)
    a = read_json(ROOT / "outputs/iclr27_phase80a/audit/phase80a_decision.json")
    b = read_json(ROOT / "outputs/iclr27_phase80b/audit/phase80b_decision.json")
    c = read_json(ROOT / "outputs/iclr27_phase80c/audit/observability_quality_audit.json")
    d = read_json(ROOT / "outputs/iclr27_phase80d/audit/tract_route_audit.json")
    modern = read_json(ROOT / "outputs/iclr27_phase80d/audit/modern_trajectory_audit.json")
    input_trace = Path(c["input"])
    if not input_trace.exists():
        raise FileNotFoundError(input_trace)
    ledger = {
        "phase": "Phase80+",
        "created_utc": now_utc.isoformat(),
        "research_start_utc": start_utc.isoformat().replace("+00:00", "Z"),
        "research_deadline_utc": deadline_utc.isoformat().replace("+00:00", "Z"),
        "actual_runtime_hours": round((now_utc - start_utc).total_seconds() / 3600.0, 4),
        "head": git("rev-parse", "HEAD"),
        "origin_main": git("ls-remote", "origin", "refs/heads/main"),
        "code_commits": ["eac6654", "9ac7bd9", "64972ba", "8c7a94e", "bb32667", "a985aaf", "676bb9a", "374cb00", "48b80ad", "491044b"],
        "phase80a": {"decision": a.get("routing_criterion"), "raw_parity": a.get("raw_parity"), "aggregate_p16": a.get("aggregate", {}).get("16")},
        "phase80b": {"decision": b.get("decision"), "aggregate": b.get("aggregate"), "fold_deltas": b.get("fold_deltas"), "formal_completion": sorted(str(x) for x in (ROOT / "outputs/iclr27_phase80b/completion").glob("phase80b_formal_f*.done"))},
        "phase80c": {"decision": "PHASE80C_AUDIT_ASSIGNMENT_HEADROOM_NO_NEW_PHASE80_MODEL", "input_sha256": c.get("input_sha256"), "p16": c.get("summary", {}).get("by_prefix", {}).get("16"), "quality": c.get("summary", {}).get("p16_quality_audit")},
        "phase80d": {"decision": d.get("decision"), "repo": d.get("repo"), "commit": d.get("commit"), "license": d.get("license"), "executed": d.get("executed"), "downloaded": d.get("downloaded"), "modern_search": modern.get("methods"), "modern_selection": modern.get("selection")},
        "artifacts": {
            "final_report": {"path": str(ROOT / "docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md"), "sha256": sha(ROOT / "docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md") if (ROOT / "docs/AUTONOMOUS_TRACKOCD_10H_RESEARCH_REPORT.md").exists() else None},
            "phase80a_decision": {"path": str(ROOT / "outputs/iclr27_phase80a/audit/phase80a_decision.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80a/audit/phase80a_decision.json")},
            "phase80b_decision": {"path": str(ROOT / "outputs/iclr27_phase80b/audit/phase80b_decision.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80b/audit/phase80b_decision.json")},
            "phase80b_exact": {"path": str(ROOT / "outputs/iclr27_phase80b/metrics/exact_memory_replay.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80b/metrics/exact_memory_replay.json")},
            "phase80c_audit": {"path": str(ROOT / "outputs/iclr27_phase80c/audit/observability_quality_audit.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80c/audit/observability_quality_audit.json")},
            "phase80d_audit": {"path": str(ROOT / "outputs/iclr27_phase80d/audit/tract_route_audit.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80d/audit/tract_route_audit.json")},
            "phase80d_modern_audit": {"path": str(ROOT / "outputs/iclr27_phase80d/audit/modern_trajectory_audit.json"), "sha256": sha(ROOT / "outputs/iclr27_phase80d/audit/modern_trajectory_audit.json")},
            "phase75b_input": {"path": str(input_trace), "sha256": c.get("input_sha256")},
        },
        "protocol": {"held_dev_q1_public_new_sealed_accessed": False, "future_rows_or_tracks": False, "physical_ids_as_model_input": False, "category_text_as_model_input": False, "controller_run": False, "commit_ct_run": False, "denominator": "Phase30 TRAIN-disjoint diagnostics for A/B; original 76 positive event audit for C; no held outcome selection"},
        "resource_events": [{"type": "duplicate_audit_process", "pids": [3170, 3167], "action": "explicit SIGTERM task-owned duplicate; original 2322 retained", "external_processes_touched": False}, {"type": "oom", "occurred": False}],
        "symlink_ledger": [{"path": str(ROOT / "outputs/iclr27_phase80a/cache"), "target": "/data2/usr_for_deadline/trackocd_phase80a/dense_cache"}],
        "decision": "PHASE80_FAMILY_A_DENSE_FAIL_FAMILY_B_MEMORY_FAIL_FAMILY_C_ASSIGNMENT_HEADROOM_FAMILY_D_INELIGIBLE",
        "status": "AUTONOMOUS_EARLY_STOP_NO_COMPLIANT_ROUTE",
        "stop_basis": "The registered A/B hypotheses failed and the only remaining C/D references lack a compliant, reproducible training/inference interface under the frozen causal protocol (C requires a new physical-association implementation; D requires forbidden/external dependencies or unavailable resources). No unregistered long training was started.",
        "next_action": "Do not run controller or sealed/public evaluation under Phase80. A future authorized route must repair causal physical association/observability or provide a legal visual support contract; do not repeat frozen-feature memory/ranker variants.",
    }
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "validation").mkdir(parents=True, exist_ok=True)
    for path, value in ((OUT / "validation_evidence_ledger.json", ledger), (OUT / "audit/final_decision.json", ledger)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        tmp.replace(path)
    print(json.dumps({"phase": "Phase80+", "head": ledger["head"], "decision": ledger["decision"], "ledger": str(OUT / "validation_evidence_ledger.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
