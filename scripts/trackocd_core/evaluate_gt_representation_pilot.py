#!/usr/bin/env python3
"""Freeze pilot fits, then evaluate heldout Train categories; not main PASS."""
from __future__ import annotations

import datetime as dt
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders, replay
from src.trackocd_core.evaluation import Target, TrackKey, evaluate_persistent, evaluate_standard, join_evaluation
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES
from src.trackocd_v2.io import atomic_json, sha256_file


def main() -> int:
    started = time.monotonic()
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.trackocd_core.representation import CategoryEvidence
    torch.set_num_threads(1)
    config_path = ROOT / "configs/trackocd_core/gt_representation_pilot.json"
    config = json.loads(config_path.read_text())
    data = json.loads((ROOT / config["data_plan"]).read_text())
    cache_root = ROOT / data["cache_directory"]
    cache = CompactGTFeasibilityCache(cache_root)
    fit_receipt = ROOT / config["output_directory"] / "training_receipt.json"
    training = json.loads(fit_receipt.read_text())
    if (training["config_sha256"] != sha256_file(config_path)
            or training["cache_manifest_sha256"] != sha256_file(cache_root / "manifest.json")):
        raise ValueError("Frozen training/data lineage changed")
    receipt_path = ROOT / "outputs/trackocd_core/audit/gt_representation_evaluation.json"
    if receipt_path.exists():
        raise RuntimeError("Preserve actual prior evaluation; no hidden re-selection")
    labels = pq.read_table(cache_root / "train_labels.parquet").to_pylist()
    by_key = {r["key"]: r for r in labels}
    routes = pq.read_table(cache_root / "index.parquet").to_pylist()
    prototype_labels = [r for r in labels if r["purpose"] == "prototype"]
    cases, modules, seed_results, predictions = [], [], [], []
    baseline = json.loads((ROOT / "outputs/trackocd_core/audit/gt_pilot_baselines.json").read_text())
    variants = [("A0_frozen_raw_mean", None, None)] + [(fit["model"], fit["seed"], fit) for fit in training["fits"]]
    for name, seed, fit in variants:
        if fit is None:
            represent = lambda view: view.weighted_mean()
            parameters = 0
        else:
            checkpoint_path = ROOT / fit["checkpoint"]["path"]
            if sha256_file(checkpoint_path) != fit["checkpoint"]["sha256"]:
                raise ValueError("Frozen fit checkpoint changed")
            saved = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            model = CategoryEvidence(name == "A2_semantic_adapter_evidence").eval().requires_grad_(False)
            model.load_state_dict(saved["state_dict"], strict=True)
            parameters = sum(p.numel() for p in model.parameters())

            def represent(view):
                with torch.inference_mode():
                    result = model(torch.tensor(view.visual[None]), torch.tensor(view.quality[None]),
                                   torch.tensor(view.elapsed_frames[None], dtype=torch.float32))
                return result["embedding"][0].numpy()

        modules.append({"model": name, "seed": seed, "parameters": parameters})
        prototypes = {int(r["category_id"]): represent(cache.get_prefix(r["key"], 16)) for r in prototype_labels}
        for partition in ("policy_train", "heldout_selection"):
            stream = [r for r in routes if by_key[r["key"]]["partition"] == partition]
            orders = registered_orders(sorted({r["video_id"] for r in stream}))
            for order_name, order in orders.items():
                for prefix in PREFIXES:
                    sealed, runtime = replay(cache, stream, prototypes, data["baseline_thresholds_fixed_before_feature_results"],
                                             "B1_track_nearest", order, prefix, represent=represent)
                    # Targets/identity joins are created only after the prediction ledger.
                    targets = [Target(TrackKey(r["video_id"], str(r["physical_track_id"])), int(by_key[r["key"]]["category_id"]),
                                      "known" if by_key[r["key"]]["simulation_role"] == "known" else "novel") for r in stream]
                    joined = join_evaluation(sealed, targets, {t.key: t.key for t in targets})
                    standard, persistent = evaluate_standard(joined), evaluate_persistent(joined)
                    standard.pop("global_anonymous_hungarian_mapping_evaluator_only")
                    if name == "A0_frozen_raw_mean":
                        previous = next(c for c in baseline["cases"] if (c['method'], c['partition'], c['order'], c['prefix']) ==
                                        ('B1_track_nearest', partition, order_name, prefix))
                        if standard != previous['standard'] or persistent != previous['persistent']:
                            raise AssertionError("A0 must reproduce the original frozen B1 scores exactly")
                    cases.append({"model": name, "seed": seed, "partition": partition, "order": order_name, "prefix": prefix,
                                  "standard": standard, "persistent": persistent, "runtime": runtime})
                    for event in sealed.events:
                        predictions.append({"model": name, "seed": seed, "partition": partition, "order": order_name,
                                            "prefix": prefix, "sequence": event.sequence,
                                            "video_id": event.physical_key.video_id, "physical_track_id": event.physical_key.local_track_id,
                                            "kind": event.kind, "token": event.token, "known_category_id": event.known_category_id})
    metrics = {"old_acc": "standard", "new_acc": "standard", "h_score": "standard", "all_acc": "standard",
               "correct_commit_ct": "persistent", "false_merge_rate": "persistent", "false_split_new_rate": "persistent",
               "wrong_known_assignment_rate": "persistent", "wait_unresolved_rate": "persistent", "effective_commit_coverage": "persistent"}
    for module in modules:
        for partition in ("policy_train", "heldout_selection"):
            for prefix in PREFIXES:
                rows = [c for c in cases if (c['model'], c['seed'], c['partition'], c['prefix']) ==
                        (module['model'], module['seed'], partition, prefix)]
                result = {**module, "partition": partition, "prefix": prefix, "orders": len(rows),
                          "known_gt": 12, "pseudo_novel_gt": 12, "fixed_gt_reuse_opportunities": 8}
                for metric, section in metrics.items():
                    result[metric + "_order_mean"] = float(np.mean([r[section][metric] for r in rows]))
                    result[metric + "_order_std_ddof0"] = float(np.std([r[section][metric] for r in rows]))
                result["policy_seconds_per_track_order_mean"] = float(np.mean([r["runtime"]["policy_seconds_per_track"] for r in rows]))
                seed_results.append(result)
    paired = []
    for partition in ("policy_train", "heldout_selection"):
        for prefix in PREFIXES:
            for seed in config["training_seeds"]:
                a1, a2 = [next(r for r in seed_results if (r['model'], r['seed'], r['partition'], r['prefix']) ==
                              (model_name, seed, partition, prefix)) for model_name in ("A1_semantic_adapter_mean", "A2_semantic_adapter_evidence")]
                paired.append({"partition": partition, "prefix": prefix, "seed": seed,
                               **{metric + "_A2_minus_A1": a2[metric + "_order_mean"] - a1[metric + "_order_mean"] for metric in metrics}})
    prediction_path = ROOT / config["output_directory"] / "sealed_representation_predictions.parquet"
    pq.write_table(pa.Table.from_pylist(predictions), prediction_path)
    receipt = {"schema_version": "trackocd.core.gt-representation-evaluation.v1", "status": "ACTUAL_BOUNDED_GT_HOLDOUT_NOT_MAIN_PASS",
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config_sha256": sha256_file(config_path),
               "training_receipt_sha256": sha256_file(fit_receipt), "cases": cases,
               "per_seed_order_aggregates": seed_results, "paired_A2_minus_A1": paired,
               "fixed_GT_denominators": {"known": 12, "pseudo_novel": 12, "reuse_opportunities": 8},
               "A0_matches_original_frozen_B1_scores_exactly": True,
               "private_sealed_predictions": {"rows": len(predictions), "bytes": prediction_path.stat().st_size,
                                               "sha256": sha256_file(prediction_path)},
               "all_seeds_orders_prefixes_kept": True, "trainable_models_frozen_during_evaluation": True,
               "same_prototype_supervision_and_recovered_B1_arbitration": True,
               "threshold_calibration_matched_between_representations": False,
               "A2_architecture_vs_auxiliary_loss_contribution_isolated": False,
               "selection_categories_not_seen_in_fit": True, "GT_or_mapping_in_policy_state": False,
               "val_or_test_access": False, "external_process_interference": False,
               "primary_frontend_frozen": False, "formal_M5_M8_M9_complete": False, "M11_authorized": False,
               "root_cause_correction_rounds_used": 0,
               "source_sha256": {name: sha256_file(ROOT / name) for name in
                                 ("scripts/trackocd_core/evaluate_gt_representation_pilot.py", "scripts/trackocd_core/run_gt_pilot_baselines.py",
                                  "src/trackocd_core/representation.py", "src/trackocd_v2/methods/nearest_prototype.py")},
               "resources": {"wall_seconds": time.monotonic() - started, "worker_count": 1, "gpu_used": False,
                             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    atomic_json(ROOT / config["output_directory"] / "evaluation.json", receipt)
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "actual_replays": len(cases), "resources": receipt["resources"],
                      "heldout_p16": [r for r in seed_results if r['partition'] == 'heldout_selection' and r['prefix'] == 16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
