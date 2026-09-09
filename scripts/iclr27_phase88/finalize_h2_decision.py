#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(name: str):
    return json.loads((OUT / "audit" / name).read_text())


def main() -> None:
    path = OUT / "audit/final_decision.json"
    decision = json.loads(path.read_text())
    decision.update({
        "status": "PHASE88_H2_FORMAL_RESOURCE_BLOCKED_NO_CONTROLLER",
        "decision_code": "P88_H2_FORMAL_RESOURCE_BLOCKED_NO_CONTROLLER",
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report": "docs/iclr27_phase88/PHASE88_AUTONOMOUS_RESEARCH_REPORT.md",
    })
    decision.setdefault("gates", {}).update({
        "c0_c1_fair_train_selection": "COMPLETE_C0_SELECTED",
        "h1_train_selection": "REJECTED_TRAIN_ONLY",
        "h2_smoke": "PASS",
        "h2_targeted": "PASS",
        "h2_formal_30k": "INCOMPLETE_RESOURCE_BLOCKED",
        "h2_train_selection": "NOT_EVALUATED",
        "h2_held_76_plus_76": "NOT_RUN",
        "controller_compatibility": "NOT_RUN",
        "sealed_evaluation": "SEALED_NOT_ACCESSED",
    })
    decision["train_only_routes"] = {
        "c0_mean_selection": 0.21347198814065957,
        "c1_mean_selection": 0.20046518210566075,
        "h1_mean_selection": 0.21099278906817454,
        "h1_decision": "H1_REJECTED_TRAIN_ONLY",
        "h2_registered_change": "known_suppression_softplus_margin_weight_1.5",
        "h2_formal": "INCOMPLETE_RESOURCE_BLOCKED",
        "h2_preregistration": "outputs/iclr27_phase88/audit/hypothesis_2_preregistration.json",
    }
    decision["h2_partial_checkpoints"] = []
    for fold, name, step in [
        (0, "h2_known_suppression_cpu_formal_f0_step024000.pt", 24000),
        (1, "h2_known_suppression_cpu_formal_f1_step022000.pt", 22000),
        (2, "h2_known_suppression_cpu_formal_f2_step020000.pt", 20000),
        (3, "h2_known_suppression_cpu_formal_f3_step022000.pt", 22000),
    ]:
        p = OUT / "checkpoints" / name
        decision["h2_partial_checkpoints"].append({
            "fold": fold,
            "step": step,
            "path": str(p.resolve()),
            "sha256": sha256(p),
            "formal_selected": False,
        })
    existing_events = decision.get("resource_events", [])
    for name in [
        "h2_resource_stop.json",
        "h2_resume1_resource_stop.json",
        "h2_resume2_resource_stop.json",
        "h2_cpu_parallel_resource_stop.json",
        "h2_cpu_sequential_resource_stop.json",
    ]:
        event = read_json(name)
        existing_events.append({
            "event": event["route"],
            "pids": event["task_owned_pids"],
            "reason": event["reason"],
            "status": event["status"],
            "external_processes_touched": False,
            "artifact": f"outputs/iclr27_phase88/audit/{name}",
        })
    # Keep one copy if a finalizer is rerun.
    dedup = {}
    for event in existing_events:
        dedup[(event.get("event"), tuple(event.get("pids", [])))] = event
    decision["resource_events"] = list(dedup.values())
    decision["cpu_repair"] = {
        "root_cause": "legacy CPU load_state_dict writes through read-only memmap-backed active_known_mask",
        "repair": "clone known buffers and explicit CPU tensor copies; CUDA path unchanged",
        "smoke": "h2_known_suppression_cpu_smoke_fix2_f0",
        "targeted": "h2_known_suppression_cpu_targeted_fix2_f0",
        "smoke_rss_bytes": 456019968,
        "targeted_rss_bytes": 452603904,
        "segfault_failures_preserved": [
            "h2_known_suppression_cpu_smoke_f0",
            "h2_known_suppression_cpu_smoke_fix1_f0",
        ],
    }
    decision["protocol"]["public_dev_q1_sealed_accessed"] = False
    decision["next_action"] = "Obtain an isolated memory-safe host/window or measured out-of-core implementation; then resume H2 from retained partial checkpoints with one worker, run frozen TRAIN validation, and only if selected perform one held diagnostic. No controller/backbone/threshold/public/sealed access is authorized from partial H2."
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    print(path)


if __name__ == "__main__":
    main()
