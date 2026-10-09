#!/usr/bin/env python3
"""Policy-Train-only frozen score separation; no fit/tuning/heldout/Val/Test."""
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
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES
from src.trackocd_v2.io import atomic_json, sha256_file


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "median": float(np.median(values))}


def main() -> int:
    started = time.monotonic()
    import torch
    import pyarrow.parquet as pq
    from sklearn.metrics import roc_auc_score
    from src.trackocd_core.representation import CategoryEvidence
    torch.set_num_threads(1)
    config_path = ROOT / "configs/trackocd_core/gt_representation_pilot.json"
    config = json.loads(config_path.read_text())
    data = json.loads((ROOT / config["data_plan"]).read_text())
    cache_root = ROOT / data["cache_directory"]
    cache = CompactGTFeasibilityCache(cache_root)
    training_path = ROOT / config["output_directory"] / "training_receipt.json"
    training = json.loads(training_path.read_text())
    labels = pq.read_table(cache_root / "train_labels.parquet").to_pylist()
    # Only these keys are ever requested from the descriptor mmap. Merely routing
    # the sidecar does not expose other-partition descriptors to a model.
    stream = [r for r in labels if r["partition"] == "policy_train"]
    prototypes = [r for r in labels if r["purpose"] == "prototype"]
    cases = []
    for name, seed, fit in [("A0_frozen_raw_mean", None, None)] + [(f["model"], f["seed"], f) for f in training["fits"]]:
        if fit is None:
            represent = lambda view: view.weighted_mean()
        else:
            checkpoint_path = ROOT / fit["checkpoint"]["path"]
            if sha256_file(checkpoint_path) != fit["checkpoint"]["sha256"]:
                raise ValueError("Frozen checkpoint changed")
            model = CategoryEvidence(name == "A2_semantic_adapter_evidence").eval().requires_grad_(False)
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location="cpu")["state_dict"], strict=True)

            def represent(view):
                with torch.inference_mode():
                    result = model(torch.tensor(view.visual[None]), torch.tensor(view.quality[None]),
                                   torch.tensor(view.elapsed_frames[None], dtype=torch.float32))
                return result['embedding'][0].numpy()

        known_matrix = np.stack([represent(cache.get_prefix(r['key'], 16)) for r in prototypes])
        known_ids = np.array([r['category_id'] for r in prototypes])
        for prefix in PREFIXES:
            features = np.stack([represent(cache.get_prefix(r['key'], prefix)) for r in stream])
            # Attach legal Train simulation supervision AFTER forward, for diagnostics only.
            categories = np.array([r['category_id'] for r in stream])
            known_mask = np.array([r['simulation_role'] == 'known' for r in stream])
            scores = features @ known_matrix.T
            top_index = scores.argmax(axis=1)
            top_score = scores.max(axis=1)
            unknown_features = features[~known_mask]
            unknown_categories = categories[~known_mask]
            similarities = unknown_features @ unknown_features.T
            triangle = np.triu(np.ones_like(similarities, dtype=bool), k=1)
            same = unknown_categories[:, None] == unknown_categories[None, :]
            cases.append({"model": name, "seed": seed, "prefix": prefix,
                          "known_max_prototype_cosine": distribution(top_score[known_mask]),
                          "pseudo_novel_max_prototype_cosine": distribution(top_score[~known_mask]),
                          "known_vs_pseudo_novel_prototype_score_AUROC_diagnostic_only": float(roc_auc_score(known_mask, top_score)),
                          "known_top1_without_rejection_count": int((known_ids[top_index[known_mask]] == categories[known_mask]).sum()),
                          "known_pass_fixed_gate_count": int((top_score[known_mask] >= .65).sum()),
                          "pseudo_novel_pass_fixed_known_gate_count": int((top_score[~known_mask] >= .65).sum()),
                          "pseudo_novel_same_class_cross_video_cosine": distribution(similarities[triangle & same]),
                          "pseudo_novel_different_class_cosine": distribution(similarities[triangle & ~same]),
                          "fixed_gt_counts": {"known": 12, "pseudo_novel": 12, "same_class_pairs": 12, "different_class_pairs": 54}})
    receipt = {"schema_version": "trackocd.core.gt-pilot-score-diagnosis.v1", "status": "POLICY_TRAIN_ONLY_DIAGNOSTIC_NOT_CORRECTION",
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cases": cases,
               "config_sha256": sha256_file(config_path), "training_receipt_sha256": sha256_file(training_path),
               "data_partitions_read_for_descriptors": ["separate Known prototypes", "policy_train"],
               "heldout_descriptor_or_evaluation_accessed": False, "val_or_test_accessed": False,
               "new_model_fitting_or_threshold_search": False, "policy_memory_or_GT_mapping_used": False,
               "fixed_gate_counts_ignore_anonymous_arbitration_are_not_sequential_metrics": True,
               "root_cause_correction_rounds_used": 0,
               "source_sha256": {name: sha256_file(ROOT / name) for name in
                                 ("scripts/trackocd_core/diagnose_gt_pilot_scores.py", "src/trackocd_core/representation.py")},
               "resources": {"wall_seconds": time.monotonic() - started, "worker_count": 1, "gpu_used": False,
                             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_pilot_score_diagnosis.json", receipt)
    print(json.dumps({"status": receipt['status'], "actual_cases": len(cases), "resources": receipt['resources'],
                      "policy_train_p16": [c for c in cases if c['prefix']==16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
