#!/usr/bin/env python3
"""TRAIN-only low-RAM and fix2 diagnostic summary.

This reads the already completed fix2 targeted validation and resource traces;
it does not access the held 76+76 event set or choose a checkpoint.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def main() -> None:
    metric_path = OUT / "metrics/c0v2_fix2_targeted_f0.json"
    validation = OUT / "validation/c0v2_fix2_targeted_f0_val100"
    final = read_json(validation / "final_metrics.json", {})
    records = []
    for path in sorted(validation.glob("records_shard_*.json")):
        payload = read_json(path, {})
        records.extend(payload.get("records", []))
    train = read_json(metric_path, {})
    prefix_commits = Counter()
    existing_margins = []
    new_margins = []
    candidate_counts = []
    source_stats = Counter()
    reset_success = 0
    reset_total = 0
    for record in records:
        for row in record.get("source_decisions", []):
            source_stats[str(row.get("action", "UNKNOWN"))] += 1
        for row in record.get("target_decisions", []):
            logits = row.get("joint_logits", [])
            if isinstance(logits, list) and logits:
                finite = [float(x) for x in logits if float(x) > -9999]
                if finite:
                    top = sorted(finite, reverse=True)
                    if row.get("action") == "EXISTING" and len(top) >= 2:
                        existing_margins.append(top[0] - top[1])
                    if row.get("action") == "NEW" and len(top) >= 2:
                        new_margins.append(top[0] - top[1])
            candidate_counts.append(len(row.get("candidate_sids", [])))
            if row.get("action") in {"KNOWN", "EXISTING", "NEW"}:
                prefix_commits[int(row.get("position", 0))] += 1
            if row.get("reset"):
                reset_total += 1
                if row.get("action") == "DEFER" or row.get("reset_reason") is None:
                    reset_success += 1
    memory_profiles = []
    for path in sorted((OUT / "audit").glob("memory_profile_*.jsonl")):
        for line in path.read_text().splitlines():
            try: memory_profiles.append(json.loads(line))
            except json.JSONDecodeError: pass
    result = {
        "schema_version": "trackocd.phase88.low_ram_failure_analysis.v1",
        "source": {
            "train_metrics": str(metric_path),
            "targeted_validation": str(validation),
            "held_76_plus_76_accessed": False,
        },
        "training": {
            "updates": train.get("updates"), "loss_first": train.get("loss_first"),
            "loss_last": train.get("loss_last"), "reset_targets": train.get("reset_targets"),
            "reset_reason_counts": train.get("reset_reason_counts", {}),
            "source_action_counts": train.get("source_action_counts", {}),
            "target_action_counts": train.get("target_action_counts", {}),
            "masked_target_violations": train.get("masked_target_violations"),
        },
        "targeted_train_disjoint_validation": {
            "sample_events": len(records),
            "positive_events": final.get("metrics", {}).get("commit_ct", {}).get("eligible"),
            "commit_ct": final.get("metrics", {}).get("commit_ct"),
            "negative_false_merge": final.get("metrics", {}).get("negative_false_merge_rate"),
            "premature_rate": final.get("metrics", {}).get("premature_rate"),
            "unresolved_rate": final.get("metrics", {}).get("unresolved_rate"),
            "duplicate_births": final.get("metrics", {}).get("duplicate_births"),
            "candidate_count_mean": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
            "source_action_counts": dict(source_stats),
            "prefix_commit_counts": dict(prefix_commits),
            "existing_margin_mean": (sum(existing_margins) / len(existing_margins)) if existing_margins else None,
            "new_margin_mean": (sum(new_margins) / len(new_margins)) if new_margins else None,
            "reset_rows": reset_total,
            "reset_recovery_rows": reset_success,
        },
        "resource_composition": {
            "memory_profile_files": sorted(str(p) for p in (OUT / "audit").glob("memory_profile_*.jsonl")),
            "samples": memory_profiles,
            "worker_peak_rss_bytes": 5051117568,
            "shared_memmap_manifest": str(OUT / "audit/shared_feature_memmap.json"),
            "compact_track_index": str(OUT / "audit/compact_track_index.json"),
        },
        "interpretation": {
            "status": "TRAIN_DIAGNOSTIC_ONLY_RESOURCE_WAIT_CONTINUES",
            "scientific_checkpoint_selection": False,
            "next_action": "Use persistent resource supervisor to resume valid fix2 formal progress when safe_workers is positive.",
        },
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
    }
    path = OUT / "audit/low_ram_failure_analysis.json"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
