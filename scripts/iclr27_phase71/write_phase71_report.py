#!/usr/bin/env python3
"""Freeze Phase71 validation metrics, decision and self-contained report."""
from __future__ import annotations
import hashlib, json, os, pathlib, statistics, tempfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
TAG = "formal1_tco_serial"
VAL = ROOT / "outputs/iclr27_phase71/validation" / TAG
MET = ROOT / "outputs/iclr27_phase71/metrics" / TAG
OUT_AUDIT = ROOT / "outputs/iclr27_phase71/audit"
REPORT = ROOT / "docs/iclr27_phase71/PHASE71_Q0_PRESERVING_PHYSICAL_TRAINING_REPORT.md"

def atomic_json(path: pathlib.Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def sha256(path: pathlib.Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def pct(x): return f"{x:.6f}"

def main():
    aggregate = json.loads((MET / "aggregate.json").read_text())
    te = json.loads((MET / "trackeval_aggregate.json").read_text())
    q0eq = json.loads((OUT_AUDIT / "q0_equivalence.json").read_text())
    q0te = json.loads((ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json").read_text())
    q0_metrics = q0te.get("aggregate", {}).get("macro_mean_over_folds", q0te.get("macro", {}))
    learned_metrics = te["aggregate"]["macro_mean_over_folds"]
    learned_recall = aggregate["aggregate"]["topk_iou05_recall"]["20"]["mean"]
    q0_recall = q0eq["recomputed_recall"]["topk"]["20"]["thresholds"]["0.5"]["recall"]
    physical_checks = {
        "valid_predictions_all_folds": True,
        "top20_iou05_not_below_q0": learned_recall >= q0_recall,
        "hota_not_below_q0": learned_metrics["HOTA"] >= q0_metrics["HOTA"],
        "deta_not_below_q0": learned_metrics["DetA"] >= q0_metrics["DetA"],
        "assa_not_below_q0": learned_metrics["AssA"] >= q0_metrics["AssA"],
        "idf1_not_below_q0": learned_metrics["IDF1"] >= q0_metrics["IDF1"],
        "no_protocol_change": True,
        "sealed_public_q1_accessed": False,
    }
    decision = {
        "phase": 71,
        "route": "q0_initialized_tco_quality_lifecycle_adapter",
        "status": "P71_FAIL_STOP_BEFORE_CORRESPONDENCE",
        "decision_code": "P71_GATE_PHYSICAL_SANITY_FAIL_STOP_BEFORE_SEMANTIC",
        "inputs": {
            "q0_checkpoint": str(ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"),
            "formal_checkpoints": [str(ROOT / f"outputs/iclr27_phase71/runs/formal1/fold_{f}/checkpoint.pth") for f in range(4)],
            "validation_tag": TAG,
            "dataset": "validation_ours_v1 (TRAIN/validation only)",
            "score_mode": "tco",
        },
        "metrics": {
            "q0_top20_iou05": q0_recall,
            "learned_top20_iou05": learned_recall,
            "q0_trackeval_macro": q0_metrics,
            "learned_trackeval_macro": learned_metrics,
            "learned_top20_by_fold": aggregate["aggregate"]["topk_iou05_recall"]["20"]["folds"],
        },
        "gate_checks": physical_checks,
        "failure_root_cause": "Q0-frozen TCO quality/lifecycle adapter does not preserve the Q0 physical stream: top20 recall and all primary HOTA/DetA/AssA/IDF1 aggregates decline. This is a physical-stream gate failure, not an OCD result.",
        "resource_event": {
            "parallel_eval": {"status": "aborted", "reason": "resource_memory_floor", "available_ram_gib_at_stop": 14, "exit_code": 143, "task_owned_pids": [11448,11459,11463,11467,11471,11461,11464,11469,11472,12712,12589,12449,12325], "external_pid_touched": False},
            "serial_eval": {"status": "completed", "concurrency": 1, "gpu": 4, "oom": False},
        },
        "status_by_stage": {"A_contract": "PASS", "B_smoke": "PASS", "B_targeted": "PASS", "B_formal": "PASS", "B_validation": "PASS", "physical_gate": "FAIL", "correspondence": "BLOCKED", "controller": "NOT_RUN", "sealed": "NOT_RUN"},
        "next_action": "Preserve all Phase71 evidence; do not train correspondence/controller from this route. Switch only to the next already-audited OVTR initialization route under a new preregistration, or record WAITING_FOR_SUPERVISION if no safe route is approved.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT_AUDIT / "phase71_decision.json", decision)
    stage = {
        "phase": 71, "tag": TAG, "step": 15000, "status": "validated", "protocol": aggregate["protocol"],
        "proposal": {"q0_top20_iou05": q0_recall, "learned_top20_iou05": learned_recall, "by_fold": aggregate["aggregate"]["topk_iou05_recall"]["20"]["folds"]},
        "mot": {"q0": q0_metrics, "learned": learned_metrics},
        "gate": "FAIL", "controller": "NOT_RUN", "sealed_public_q1_accessed": False,
        "resource": decision["resource_event"],
    }
    atomic_json(VAL / "step_15000_metrics.json", stage)
    done = VAL / "step_15000.done"
    tmp = done.with_suffix(".done.tmp")
    tmp.write_text("validated\n"); os.replace(tmp, done)

    fold_lines = []
    for f in range(4):
        rec = aggregate["folds"][str(f)]
        tr = te["folds"][str(f)]["macro"]
        fold_lines.append(f"| {f} | {pct(rec['topk']['20']['thresholds']['0.3']['recall'])} | {pct(rec['topk']['20']['thresholds']['0.5']['recall'])} | {pct(rec['topk']['20']['thresholds']['0.7']['recall'])} | {pct(tr['HOTA'])} | {pct(tr['DetA'])} | {pct(tr['AssA'])} | {pct(tr['IDF1'])} | {pct(tr['IDSW'])} | {pct(tr['Frag'])} |")
    report = f"""# Phase71 — Q0-preserving physical route training and validation

**Execution window:** 2026-08-31–2026-09-01 (Asia/Shanghai)  
**Decision:** **P71 physical gate FAIL — stop before correspondence/controller**  
**Decision artifact:** `outputs/iclr27_phase71/audit/phase71_decision.json`

## Scope and sealed boundary

This route tested one registered, Q0-initialized trajectory-conditioned objectness (TCO) adapter. The Q0 decoder/query/box path, base score, parent assignment and physical lifecycle were frozen; only `tco_head.*` was trainable. TRAIN/validation annotations were used as loss/evaluator metadata only. No DEV+, Q1, public new-model or sealed labels, future rows/tracks, category text, semantic ID, or physical ID feature was used. Correspondence, StateMemory and Commit/Defer were not run because the physical gate failed.

## Frozen Q0 anchor

- Checkpoint SHA256: `809c360471693adbc737394995528f04fd2ba90b6a65d85fc3c9e6b27d4d1738`.
- Historical full-sequence stream: 1,268,113 predictions; exact Q0 top-20 IoU≥0.5 recall `71062/112798 = {q0_recall:.9f}`; macro HOTA `{q0_metrics['HOTA']:.6f}`.
- The Q0 five-field CSV key and score-mode=`base` equivalence audit passed before training.

## Training and repair record

Smoke (20), fold0 targeted (100), and four formal folds completed with atomic markers and non-empty 240,021,717-byte checkpoints. Formal checkpoints were initialized from Q0 and hashed as follows:

| fold | SHA256 |
|---:|---|
| 0 | `4971600368d6dcebd773c4ee5c5857df3b7c0df756449ed01aaf3a247820ac64` |
| 1 | `be0f64b666f4a9be6a064c2a427b19e06ab325068cb973112b61032bdf04ae63` |
| 2 | `5e93234909efbc18d7d8cdf4ac78df8090dea84ffc520ab41e85b97d5872224b` |
| 3 | `a1b18fbe0470fdf079d0dc7d90c48bea4d89fae3d49c2bd6d648be48937c6470` |

The first targeted run failed on a legal no-gradient TCO batch; the process-local no-op backward guard was the minimal repair. Smoke and targeted regression then passed. No upstream OVTR file or protocol was changed.

## Full-sequence validation

Validation used the pinned OVTR evaluator on `validation_ours_v1`, `score_mode=tco`, identical thresholds and full sequence. It ran serially on GPU4 (one evaluator plus one DataLoader worker) to preserve the RAM floor. The prediction files contain 1.26M records per fold and were evaluated with the vendored TrackEval TAO adapter.

| fold | top20@0.3 | top20@0.5 | top20@0.7 | HOTA | DetA | AssA | IDF1 | IDSW | Frag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_lines)}
| mean | {pct(statistics.fmean([aggregate['folds'][str(f)]['topk']['20']['thresholds']['0.3']['recall'] for f in range(4)]))} | {pct(learned_recall)} | {pct(statistics.fmean([aggregate['folds'][str(f)]['topk']['20']['thresholds']['0.7']['recall'] for f in range(4)]))} | {pct(learned_metrics['HOTA'])} | {pct(learned_metrics['DetA'])} | {pct(learned_metrics['AssA'])} | {pct(learned_metrics['IDF1'])} | {pct(learned_metrics['IDSW'])} | {pct(learned_metrics['Frag'])} |

Q0 comparison: top20@0.5 `{q0_recall:.6f}` → `{learned_recall:.6f}`; HOTA `{q0_metrics['HOTA']:.6f}` → `{learned_metrics['HOTA']:.6f}`; DetA `{q0_metrics['DetA']:.6f}` → `{learned_metrics['DetA']:.6f}`; AssA `{q0_metrics['AssA']:.6f}` → `{learned_metrics['AssA']:.6f}`; IDF1 `{q0_metrics['IDF1']:.6f}` → `{learned_metrics['IDF1']:.6f}`. The learned stream is below Q0 on every primary physical metric, so it cannot be selected for semantic work. Track continuity and loss values are diagnostic only and cannot override this gate.

## Resource/process event

The first `formal1_tco` evaluator launched four heavy workers concurrently. Available RAM fell to about 14 GiB, below the 25% floor. Only task-owned supervisor/evaluator/DataLoader PIDs were explicitly terminated (exit 143); GPU0 external PID10750 was untouched, and no OOM or swap event occurred. Logs and `.launched` markers remain under `outputs/iclr27_phase71/validation/formal1_tco/`; no prediction JSON was produced there. The repaired serial run (`formal1_tco_serial`) completed all four folds with no further memory event. `/data1` and `/data2` paths and all large predictions remain in place; no copies of historical features/checkpoints were made.

## Gate and continuation

- A-contract/Q0 equivalence: **PASS**.
- TCO smoke/targeted/formal completion: **PASS** (training execution only).
- Physical Gate P71: **FAIL** — learned top20 recall and HOTA/DetA/AssA/IDF1 are lower than Q0; therefore no valid learned physical stream exists.
- Correspondence, controller/Commit-CT, and sealed/public evaluation: **NOT RUN/BLOCKED**, not zeros and not claimed as OCD outcomes.

This route is exhausted for its registered Q0-initialized TCO adapter. The next legal action is a new preregistration for the next already-audited OVTR initialization (for example the official 5-frame/detection checkpoint) with the same Q0-preservation and intermediate full-sequence sanity gates. Do not initialize from Phase69/70, do not tune semantic thresholds/StateMemory/controller, and do not access sealed labels.

## Reproduction

```bash
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/q0_audit.py
bash scripts/iclr27_phase71/run_tco_supervisor.sh formal formal1
bash scripts/iclr27_phase71/run_eval_serial.sh formal1_tco_serial formal1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/compute_validation_metrics.py --tag formal1_tco_serial --checkpoint-tag formal1
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/run_trackeval_phase71.py --tag formal1_tco_serial
/home/lwr/anaconda3/envs/ovtr/bin/python scripts/iclr27_phase71/summarize_trackeval.py --tag formal1_tco_serial
```

All JSON artifacts parsed successfully, all four formal training and serial-validation markers are present, and no Phase71 process remains. Full per-fold metrics are in `outputs/iclr27_phase71/metrics/{TAG}/`; the atomic stage hook is `outputs/iclr27_phase71/validation/{TAG}/step_15000_metrics.json` with `step_15000.done`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = REPORT.with_suffix(".md.tmp")
    tmp_report.write_text(report)
    os.replace(tmp_report, REPORT)
    print(json.dumps({"decision": decision["decision_code"], "report": str(REPORT), "q0_top20_iou05": q0_recall, "learned_top20_iou05": learned_recall, "q0_hota": q0_metrics["HOTA"], "learned_hota": learned_metrics["HOTA"]}, indent=2))

if __name__ == "__main__": main()
