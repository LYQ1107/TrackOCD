#!/usr/bin/env python3
"""Actual bounded Train Known GT replay, NOT formal TAO Val M4/M9."""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_core.baselines import anonymous_memory_size, decide_prefix, make_baseline
from src.trackocd_core.evaluation import DecisionEvent, Target, TrackKey, evaluate_persistent, evaluate_standard, join_evaluation, seal_decisions
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES
from src.trackocd_v2.io import atomic_json, atomic_write_text, sha256_file

BASELINES = ("B0_frame_snapshot_vote", "B1_track_nearest", "B2_track_dpmeans")


def registered_orders(video_ids: list[int]) -> dict[str, list[int]]:
    orders = {"main": sorted(video_ids)}
    for seed in (1027, 1028, 1029):
        videos = sorted(video_ids)
        random.Random(seed).shuffle(videos)
        orders[f"seed{seed}"] = videos
    return orders


def replay(cache, routes: list[dict], prototypes: dict, thresholds: dict,
           name: str, order: list[int], prefix: int, represent=None) -> tuple:
    """No target categories, semantic roles, matches or Hungarian input."""
    model = make_baseline(name, prototypes, thresholds)
    by_video = defaultdict(list)
    for row in routes:
        by_video[row["video_id"]].append(row)
    events, memory_counts = [], []
    inference_seconds = 0.
    for video in order:
        # Completion time is observed; future full-track duration is not used.
        for row in sorted(by_video[video], key=lambda r: (r["frame_ids"][prefix - 1], r["physical_track_id"])):
            view = cache.get_prefix(row["key"], prefix)
            started = time.perf_counter()
            action = decide_prefix(model, view) if represent is None else model.step(represent(view))
            inference_seconds += time.perf_counter() - started
            kind = action["kind"]
            events.append(DecisionEvent(len(events), TrackKey(video, str(row["physical_track_id"])), len(view.visual), kind,
                                        known_category_id=int(action["category_id"]) if kind == "KNOWN" else None,
                                        token=action["token"] if kind in {"NEW", "EXISTING"} else None))
            memory_counts.append(anonymous_memory_size(model))
    sealed = seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=tuple(prototypes))
    return sealed, {"policy_inference_seconds": inference_seconds,
                    "policy_seconds_per_track": inference_seconds / len(routes),
                    "anonymous_memory_final": anonymous_memory_size(model),
                    "anonymous_memory_counts": memory_counts}


def main() -> int:
    started = time.monotonic()
    import pyarrow as pa
    import pyarrow.parquet as pq
    config_path = ROOT / "configs/trackocd_core/gt_feasibility_pilot.json"
    config = json.loads(config_path.read_text())
    mem = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] * 1024 - config["ram_plan_bytes"] < mem["MemTotal"] * 1024 * .25:
        raise RuntimeError("Resource wait: leave 25% system headroom")
    cache_root = ROOT / config["cache_directory"]
    cache = CompactGTFeasibilityCache(cache_root)
    if cache.manifest["pilot_config"]["sha256"] != sha256_file(config_path):
        raise ValueError("Pilot differs from pre-feature registration")
    output = ROOT / "outputs/trackocd_core/pilots/gt_train_known/baselines"
    if output.exists():
        raise RuntimeError("Preserve completed pilot results; no silent overwrite")
    labels = pq.read_table(cache_root / "train_labels.parquet").to_pylist()
    labels_by_key = {r["key"]: r for r in labels}
    routes = pq.read_table(cache_root / "index.parquet").to_pylist()
    prototypes = {int(label["category_id"]): cache.get_prefix(label["key"], 16).weighted_mean()
                  for label in labels if label["purpose"] == "prototype"}
    if len(prototypes) != config["known_categories"]:
        raise ValueError("Incomplete registered Known prototype supervision")
    cases, ledger_rows, order_records = [], [], {}
    for partition in ("policy_train", "heldout_selection"):
        stream = [row for row in routes if labels_by_key[row["key"]]["partition"] == partition]
        orders = registered_orders(sorted({r["video_id"] for r in stream}))
        order_records[partition] = orders
        for name in BASELINES:
            for order_name, order in orders.items():
                for prefix in PREFIXES:
                    sealed, runtime = replay(cache, stream, prototypes, config["baseline_thresholds_fixed_before_feature_results"],
                                             name, order, prefix)
                    # Evaluation ONLY after sealing. Identity join is explicit GT
                    # feasibility, not a predicted/geometry adapter claim.
                    targets = [Target(TrackKey(r["video_id"], str(r["physical_track_id"])),
                                      int(labels_by_key[r["key"]]["category_id"]),
                                      "known" if labels_by_key[r["key"]]["simulation_role"] == "known" else "novel")
                               for r in stream]
                    joined = join_evaluation(sealed, targets, {t.key: t.key for t in targets})
                    standard, persistent = evaluate_standard(joined), evaluate_persistent(joined)
                    standard.pop("global_anonymous_hungarian_mapping_evaluator_only")
                    cases.append({"partition": partition, "method": name, "order": order_name, "prefix": prefix,
                                  "standard": standard, "persistent": persistent, "runtime": runtime})
                    for event in sealed.events:
                        ledger_rows.append({"partition": partition, "method": name, "order": order_name, "prefix": prefix,
                                            "sequence": event.sequence, "video_id": event.physical_key.video_id,
                                            "physical_track_id": event.physical_key.local_track_id, "kind": event.kind,
                                            "known_category_id": event.known_category_id, "token": event.token})
    metrics = {"old_acc": "standard", "new_acc": "standard", "h_score": "standard", "all_acc": "standard",
               "correct_commit_ct": "persistent", "false_merge_rate": "persistent", "false_split_new_rate": "persistent",
               "wrong_known_assignment_rate": "persistent", "wait_unresolved_rate": "persistent", "effective_commit_coverage": "persistent"}
    aggregate = []
    for partition in ("policy_train", "heldout_selection"):
        for method in BASELINES:
            for prefix in PREFIXES:
                rows = [c for c in cases if (c["partition"], c["method"], c["prefix"]) == (partition, method, prefix)]
                result = {"scope": "SMALL_TRAIN_KNOWN_GT_FEASIBILITY_ONLY", "partition": partition,
                          "method": method, "prefix": prefix, "orders": len(rows), "known_gt": 12, "pseudo_novel_gt": 12,
                          "fixed_gt_reuse_opportunities": 8}
                for metric, section in metrics.items():
                    values = [r[section][metric] for r in rows]
                    result[metric + "_mean"] = float(np.mean(values))
                    result[metric + "_std_ddof0"] = float(np.std(values))
                aggregate.append(result)
    output.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(ledger_rows), output / "sealed_predictions.parquet")
    receipt = {"schema_version": "trackocd.core.gt-pilot-baselines.v1", "status": "ACTUAL_SMALL_TRAIN_GT_BASELINES_NOT_MAIN",
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "scope": config["source_bias"],
               "config_sha256": sha256_file(config_path), "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
               "same_feature_protocol_and_four_known_prototypes_for_all_methods": True,
               "prototype_labels_legal_train_known_only": True, "pseudo_novel_labels_in_policy_state": False,
               "heldout_selection_not_an_independent_final_test": True, "threshold_search_performed": False,
               "causality": "Video-sequential observation-prefix replay, not full frame-online verification",
               "B3_PHE": {"status": "INCOMPARABLE", "reason": "Compatible checkpoint/source/supervision absent; no score"},
               "cases": cases, "aggregate": aggregate, "video_orders": order_records,
               "all_methods_trainable_parameter_count": 0, "trainable_model_training_started": False,
               "baseline_algorithm_scientific_superiority_claimed": False,
               "formal_M4_complete": False, "formal_M9_complete": False, "qualified_frontend_frozen": False,
               "val_or_test_access": False, "external_process_interference": False,
               "source_sha256": {name: sha256_file(ROOT / name) for name in
                                 ("scripts/trackocd_core/run_gt_pilot_baselines.py", "src/trackocd_core/baselines.py",
                                  "src/trackocd_v2/methods/nearest_prototype.py", "src/trackocd_v2/methods/dpmeans.py")},
               "private_sealed_predictions": {"bytes": (output / "sealed_predictions.parquet").stat().st_size,
                                              "sha256": sha256_file(output / "sealed_predictions.parquet")},
               "resources": {"wall_seconds": time.monotonic() - started, "worker_count": 1, "gpu_used": False,
                             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    atomic_json(output / "results.json", receipt)
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_pilot_baselines.json", receipt)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(aggregate[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(aggregate)
    atomic_write_text(ROOT / "outputs/trackocd_core/audit/GT_PILOT_BASELINE_COMPARISON.csv", csv_buffer.getvalue())
    print(json.dumps({"status": receipt["status"], "actual_replays": len(cases), "aggregate_rows": len(aggregate),
                      "resources": receipt["resources"], "heldout_p16": [r for r in aggregate if r["partition"] == "heldout_selection" and r["prefix"] == 16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
