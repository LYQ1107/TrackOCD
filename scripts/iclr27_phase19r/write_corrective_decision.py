#!/usr/bin/env python
"""Write the machine-readable Phase19R stop/decision artifact."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase19r"
REPORT = OUT / "audit/phase19r_corrective_decision.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_row(m: dict, source: str, fold: int, step: int | None = None) -> dict:
    return {
        "source": source, "fold": int(fold), "step": step,
        "commit_ct_correct": int(m["commit_ct"]["correct"]),
        "commit_ct_eligible": int(m["commit_ct"]["eligible"]),
        "category_coverage": int(m.get("category_coverage", 0)),
        "video_coverage": int(m.get("video_coverage", 0)),
        "existing_precision": float(m.get("existing_precision", 0.0)),
        "existing_recall": float(m.get("existing_recall", 0.0)),
        "negative_false_merge": float(m.get("negative_false_merge_rate", m.get("false_merge_rate_macro", 0.0))),
        "known_micro": float(m.get("known_micro", 0.0)),
        "known_macro": float(m.get("known_macro", 0.0)),
        "unresolved": float(m.get("unresolved_rate", 0.0)),
        "duplicate_births": int(m.get("duplicate_births", 0)),
    }


def main() -> None:
    mixed_rows = []
    for fold in range(4):
        d = load(OUT / "metrics" / f"fold{fold}_training.json")
        e = next(x for x in d["logs"] if int(x["step"]) == 8000)
        mixed_rows.append(metric_row(e["persistent_event_validation"]["metrics"], "mixed_baseline", fold, 8000))
    event_rows = []
    repair_rows = []
    checkpoint_hashes = {}
    for fold in range(4):
        d = load(OUT / "metrics" / f"event_aligned_f{fold}_4000.json")
        e = d["logs"][-1]; event_rows.append(metric_row(e["validation"]["persistent_event_metrics"], "event_aligned", fold, int(e["step"])))
        d2 = load(OUT / "metrics" / f"event_repair_f{fold}_4000.json")
        e2 = d2["logs"][-1]; repair_rows.append(metric_row(e2["validation"]["persistent_event_metrics"], "event_repair", fold, int(e2["step"])))
        ck = OUT / "checkpoints" / f"event_repair_f{fold}_best.pt"
        checkpoint_hashes[str(fold)] = {"path": str(ck), "exists": ck.exists(),
                                       "sha256": hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None,
                                       "prototype_hash": d2.get("prototype_hash_before_after")}

    def aggregate(rows: list[dict], source: str) -> dict:
        return {
            "source": source, "folds": rows,
            "commit_ct": {"correct": sum(x["commit_ct_correct"] for x in rows),
                          "eligible": sum(x["commit_ct_eligible"] for x in rows)},
            "category_coverage_sum": sum(x["category_coverage"] for x in rows),
            "video_coverage_sum": sum(x["video_coverage"] for x in rows),
            "existing_precision_mean": sum(x["existing_precision"] for x in rows) / len(rows),
            "existing_recall_mean": sum(x["existing_recall"] for x in rows) / len(rows),
            "negative_false_merge_mean": sum(x["negative_false_merge"] for x in rows) / len(rows),
            "known_micro_mean": sum(x["known_micro"] for x in rows) / len(rows),
            "known_macro_mean": sum(x["known_macro"] for x in rows) / len(rows),
            "unresolved_mean": sum(x["unresolved"] for x in rows) / len(rows),
            "duplicate_births": sum(x["duplicate_births"] for x in rows),
        }

    # Anchored to the measured benchmark artifacts; event-repair per-fold
    # throughput is read from its completed summaries.
    bench = load(OUT / "metrics/acceleration_benchmark_fast.json")
    old_bench = load(OUT / "metrics/acceleration_benchmark.json")
    first_speed = [load(OUT / "metrics" / f"event_aligned_f{f}_4000.json")["updates_per_second"] for f in range(4)]
    repair_speed = [load(OUT / "metrics" / f"event_repair_f{f}_4000.json")["updates_per_second"] for f in range(4)]
    public_paths = [OUT / "completion/public_predictions.frozen", OUT / "metrics/public_after_freeze.json",
                    OUT / "metrics/public_gate.json", OUT / "manifests/prediction_freeze.json"]
    # Anchored process check: only a command whose argv starts with the AVI
    # interpreter and the Phase19R train module is task training.
    training_pids = []
    try:
        import subprocess
        text = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        for line in text.splitlines():
            if re.search(r"^\s*\d+\s+/home/lwr/anaconda3/envs/AVI/bin/python -m src\.iclr27_phase19r\.training\.train_controller(?:\s|$)", line):
                training_pids.append(int(line.split()[0]))
    except Exception:
        training_pids = ["check_failed"]
    artifact = {
        "protocol": "trackocd_iclr27_phase19r_corrective_decision_v1",
        "decision_code": "P19R_EVENT_ALIGNED_INTERNAL_GATE_FAILS_STOP_BEFORE_FINAL",
        "stop_training": True, "final_50k_started_after_stop": False,
        "public_status": "sealed_internal_gate_failed_no_public_new_model_labels_read",
        "authoritative_comparison": {
            "mixed_baseline": aggregate(mixed_rows, "mixed_baseline"),
            "event_aligned": aggregate(event_rows, "event_aligned"),
            "event_repair": aggregate(repair_rows, "event_repair"),
        },
        "speed": {
            "old_steady_updates_per_second": old_bench.get("historical_old_fold_updates_per_second"),
            "event_aligned_updates_per_second": first_speed,
            "event_repair_updates_per_second": repair_speed,
            "hard_pair_cache_indexed_prebenchmark_speedup": old_bench.get("speedup_updates_per_second"),
            "fast_path_speedup_vs_old": bench.get("speedup_vs_old"),
            "strict_two_x_target_met": False,
            "interpretation": "hard-pair cache and feature-free episode indices remove generation overhead; fold-parallel four-GPU scheduling improves wall time, but rollout/state-machine CPU work remains. event_commit_margin adds loss work and lowers throughput.",
        },
        "error_evidence": {
            "synthetic_proxy": "existing precision approximately 1.0 with existing recall approximately 0.35-0.42",
            "persistent_mixed": "existing precision and recall are zero at the comparable 2/76 baseline events",
            "event_aligned": "2/76 and does not exceed mixed baseline",
            "event_repair": "0/76; the fold3/category81 successes disappear",
            "audited_counts": load(OUT / "audit/event_mismatch.json").get("aggregate_error_counts", {}),
            "inference_only": ["frozen DINOv2 representation plus the current controller may lack stable cross-instance semantic separability", "synthetic rollout and persistent state evolution may still be non-equivalent"],
        },
        "checkpoints": checkpoint_hashes,
        "integrity": {
            "training_processes_clean_exit": training_pids == [],
            "training_pids": training_pids,
            "event_repair_done_markers": {str(f): (OUT / "completion" / f"event_repair_4k_f{f}.done").exists() for f in range(4)},
            "event_aligned_done_markers": {str(f): (OUT / "completion" / f"event_aligned_4k_f{f}.done").exists() for f in range(4)},
            "json_inputs_parse": True,
            "public_artifacts_present": {str(p.relative_to(ROOT)): p.exists() for p in public_paths},
        },
        "next_direction": "Establish a separately verifiable cross-instance semantic-correspondence/representation-learning baseline first, then design an online state controller; do not continue threshold or memory tuning on this branch.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_name(REPORT.name + ".tmp"); tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n"); tmp.replace(REPORT)
    print(json.dumps({"path": str(REPORT), "decision_code": artifact["decision_code"], "training_pids": training_pids}, indent=2), flush=True)


if __name__ == "__main__": main()

