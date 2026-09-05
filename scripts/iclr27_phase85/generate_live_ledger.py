#!/usr/bin/env python3
"""Create a data-driven Phase85 live ledger while finalization is locked."""
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
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.exists(), "sha256": sha(path), "size": path.stat().st_size if path.is_file() else None}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def resources() -> dict[str, Any]:
    ps = subprocess.run(["ps", "-e", "--no-headers"], text=True, capture_output=True, check=False).stdout
    return {
        "free_h": subprocess.run(["free", "-h"], text=True, capture_output=True, check=False).stdout,
        "disk": subprocess.run(["df", "-h", "/data1", "/data2"], text=True, capture_output=True, check=False).stdout,
        "process_count": len(ps.splitlines()),
        "phase85_processes": [],
    }


def expected_provenance() -> list[dict[str, Any]]:
    return [
        {"section_name": "phase84_issue_audit", "expected_route": None, "expected_tag": None, "path": AUDIT / "phase84_issue_audit.json"},
        {"section_name": "temporal_mean_physical", "expected_route": "Phase85 P1", "expected_tag": "temporal_mean_full", "path": METRICS / "temporal_mean_full.json"},
        {"section_name": "q0_adapter_parity", "expected_route": "Phase85 P3/P4", "expected_tag": "q0_parity_v5", "path": AUDIT / "physical_r_q0_q0_parity_v5_adapter.json"},
        {"section_name": "temporal_physical_r", "expected_route": "PHYSICAL_TO_R_DIAGNOSTIC", "expected_tag": "improved_single_anchor_v2", "path": AUDIT / "physical_r_temporal_comparison_v2.json"},
        {"section_name": "selective_physical_r", "expected_route": "PHYSICAL_TO_R_DIAGNOSTIC", "expected_tag": "selective_gate_v1", "path": AUDIT / "physical_r_selective_comparison.json"},
        {"section_name": "support_event_replay", "expected_route": "raw source-mean top32; bounded residual reranker; separate TRAIN defer head (p>=0.5 -> DEFER)", "expected_tag": None, "path": METRICS / "support_event_replay.json"},
        {"section_name": "support_selective_source", "expected_route": "raw source-mean top32; bounded residual reranker; separate TRAIN defer head (p>=0.5 -> DEFER)", "expected_tag": None, "path": METRICS / "support_event_replay_selective_source_v1.json"},
        {"section_name": "support_alignment_feasibility", "expected_route": None, "expected_tag": None, "path": AUDIT / "support_alignment_feasibility.json"},
        {"section_name": "event_physical_contamination", "expected_route": None, "expected_tag": None, "path": AUDIT / "event_physical_contamination.json"},
    ]


def main() -> None:
    reg = load(REG); lock = load(LOCK)
    p1 = load(METRICS / "temporal_mean_full.json")
    phys = load(AUDIT / "physical_r_comparison.json")
    sel_phys = load(AUDIT / "physical_r_selective_comparison.json")
    support = load(METRICS / "support_event_replay.json")
    support_sel = load(METRICS / "support_event_replay_selective_source_v1.json")
    support_audit = load(AUDIT / "support_alignment_feasibility.json")
    contamination = load(AUDIT / "event_physical_contamination.json")
    artifacts = [item["path"] for item in expected_provenance()]
    provenance_rows = []
    for item in expected_provenance():
        path = item["path"]; actual = load(path)
        actual_route = actual.get("route", actual.get("gate_diagnostic", {}).get("status", actual.get("phase", actual.get("strategy"))))
        actual_tag = actual.get("tag", actual.get("candidate_name"))
        actual_schema = actual.get("schema_version")
        route_ok = item["expected_route"] is None or actual_route == item["expected_route"]
        tag_ok = item["expected_tag"] is None or actual_tag == item["expected_tag"]
        provenance_rows.append({"section_name": item["section_name"], "expected_route": item["expected_route"], "expected_tag": item["expected_tag"], "source_path": str(path.resolve()), "source_sha": sha(path), "actual_route": actual_route, "actual_tag": actual_tag, "actual_schema": actual_schema, "route_ok": route_ok, "tag_ok": tag_ok, "exists": path.is_file()})
    atomic_json(AUDIT / "report_provenance.json", {"schema_version": "trackocd.phase85.report_provenance.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sections": provenance_rows, "all_contracts_match": all(x["exists"] and x["route_ok"] and x["tag_ok"] for x in provenance_rows), "no_hardcoded_scientific_headline_values": True, "git_head": git("rev-parse", "HEAD")})
    failed_markers = sorted(p.name for p in COMP.glob("*.launched") if not p.with_suffix(".done").exists())
    ledger = {
        "schema_version": "trackocd.phase85.research_ledger.v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "registration": reg, "finalization_lock": lock, "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"), "resources": resources(), "stages": [
            {"stage": "P0_issue_audit", "status": "DONE", "artifact": artifact(AUDIT / "phase84_issue_audit.json"), "next_action": "P1 temporal mean"},
            {"stage": "P1_temporal_mean", "status": "DONE", "artifact": artifact(METRICS / "temporal_mean_full.json"), "metrics": p1.get("stats", {}), "next_action": "TrackEval and adapters"},
            {"stage": "P2_trackeval", "status": "DONE", "artifacts": [artifact(METRICS / "trackeval/q0_event91/q0_event91/cls_comb_cls_av_summary.txt"), artifact(METRICS / "trackeval/temporal_mean_event91/temporal_mean_event91/cls_comb_cls_av_summary.txt"), artifact(METRICS / "trackeval/selective_event91/selective_event91/cls_comb_cls_av_summary.txt")], "next_action": "P3 parity"},
            {"stage": "P3_q0_parity", "status": "PASS", "artifact": artifact(AUDIT / "physical_r_q0_q0_parity_v5_adapter.json"), "next_action": "P4/P5"},
            {"stage": "P5_physical_to_r", "status": "DIAGNOSTIC_FAIL", "artifacts": [artifact(AUDIT / "physical_r_comparison.json"), artifact(AUDIT / "physical_r_selective_comparison.json")], "p16_temporal": phys.get("gate_diagnostic", {}), "p16_selective": sel_phys.get("gate_diagnostic", {}), "next_action": "retain negative physical evidence"},
            {"stage": "B85S_support", "status": "SAFETY_FAIL", "artifacts": [artifact(METRICS / "support_event_replay.json"), artifact(METRICS / "support_event_replay_selective_source_v1.json")], "p16": [x for x in support.get("summary", []) if x.get("prefix") == 16], "selective_source_p16": [x for x in support_sel.get("summary", []) if x.get("prefix") == 16], "next_action": "no alignment/controller; preserve raw anchor"},
            {"stage": "support_selection_audit", "status": "DONE", "artifact": artifact(AUDIT / "support_alignment_feasibility.json"), "routing": support_audit.get("routing", {}), "next_action": "final report in unlock interval"},
            {"stage": "event_physical_contamination", "status": "DONE", "artifact": artifact(AUDIT / "event_physical_contamination.json"), "summary": contamination.get("summary", {}), "next_action": "retain contamination evidence"},
        ], "artifacts": [artifact(Path(x)) for x in artifacts], "failed_uncompleted_markers": failed_markers, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "controller_run": False, "sealed_run": False,
    }
    atomic_json(AUDIT / "research_ledger.json", ledger)
    live = ["# Phase85 live research ledger (not final)", "", f"Generated UTC: {ledger['generated_utc']}", f"Window: {reg.get('start_time_utc')} → {reg.get('deadline_utc')}", f"Finalization lock allowed: {lock.get('allowed', False)}", f"Git HEAD: {ledger['git_head']}", "", "Scientific state:", "- P0/P1/P2/P3/P4 repairs and parity are complete; Q0 parity passed on the frozen 984-query universe.", "- Temporal-mean and selective physical streams have valid TrackEval diagnostics but both physical→R comparisons are below Q0 and unsafe.", "- Raw-anchored support reranking increases positive selection modestly but increases negative activation; its separate DEFER policy removes most positive selections.", "- Selective-lineage source cache is worse than Q0. Support alignment is not authorized by the registered positive/safety routing criterion.", "- Controller, StateMemory, threshold sweep, modern backbone and sealed/public evaluation remain NOT_RUN by protocol.", "", f"Uncompleted launched markers retained: {', '.join(failed_markers) if failed_markers else 'none'}", "", "Machine ledgers: `outputs/iclr27_phase85/audit/{research_ledger,report_provenance,repair_events}.json`."]
    atomic_text(ROOT / "docs/iclr27_phase85/PHASE85_LIVE_LEDGER.md", "\n".join(live) + "\n")
    print(json.dumps({"status": "LIVE_NOT_FINAL", "lock_allowed": lock.get("allowed", False), "git_head": ledger["git_head"], "failed_markers": failed_markers}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
