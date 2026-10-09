#!/usr/bin/env python3
"""Check actual tiny GT cache hashes/offsets/prefix means without model inference."""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES
from src.trackocd_v2.io import atomic_json


def main():
    root = ROOT / "outputs/trackocd_core/features/gt_train_known_smoke"
    cache = CompactGTFeasibilityCache(root)
    stored = np.load(root / "prefix_features.npy", mmap_mode="r", allow_pickle=False)
    known = set(json.loads((ROOT / "configs/trackocd_core/roles.json").read_text())["known_ids"])
    labels = pq.read_table(root / "train_labels.parquet").to_pylist()
    if {r["key"] for r in labels} != set(cache.keys()) or any(r["category_id"] not in known for r in labels):
        raise ValueError("Train supervision table violates Known/key allowlist")
    deltas = []
    for i, key in enumerate(cache.keys()):
        for j, p in enumerate(PREFIXES):
            view = cache.get_prefix(key, p)
            computed = view.weighted_mean().astype(np.float16)
            np.testing.assert_array_equal(computed, stored[i, j])
            deltas.append(float(np.max(np.abs(computed.astype(np.float32) - stored[i, j].astype(np.float32)))))
    receipt = {"schema_version": "trackocd.core.gt-cache-verification.v1", "status": "PASS_ENGINEERING_GT_ONLY",
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "payload_hashes_verified": len(cache.manifest["payloads"]), "tracks": len(cache.keys()),
               "observations": cache.manifest["observations"], "prefix_views_verified": len(deltas),
               "fp16_prefix_mean_max_abs_delta": max(deltas), "all_training_labels_in_known": True,
               "model_view_has_gt_ids_or_labels_or_total_length": False,
               "model_inference_performed": False, "training_started": False, "test_data_accessed": False,
               "predicted_main_result": False, "old_nas_shards_validated_by_this_check": False}
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_cache_verification.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
