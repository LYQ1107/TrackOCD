"""Merge and validate the four atomic full DINOv3 feature shards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv")
    ap.add_argument("--shards", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--done", type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.rows.open())); n = len(rows)
    feat = np.zeros((n, 6, 768), dtype=np.float16)
    teacher = np.zeros(n, dtype=np.uint8); seen = np.zeros(n, dtype=np.uint8)
    shard_meta = []
    for path in args.shards:
        z = np.load(path, allow_pickle=False); idx = z["global_index"].astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= n) or seen[idx].any():
            raise RuntimeError("invalid/duplicate shard global indices: " + str(path))
        keys = z["row_keys"].astype(str)
        expected = np.asarray([rows[int(i)]["row_key"] for i in idx])
        if not np.array_equal(keys, expected):
            raise RuntimeError("row-key mismatch: " + str(path))
        feat[idx] = z["features"].astype(np.float16); teacher[idx] = z["teacher_mask"].astype(np.uint8); seen[idx] = 1
        mp = path.with_suffix(".json")
        if mp.exists(): shard_meta.append(json.loads(mp.read_text()))
    if not seen.all():
        raise RuntimeError("missing merged rows: " + str(np.where(~seen.astype(bool))[0][:20].tolist()))
    raw = feat[:, 0].astype(np.float32); smooth = feat[:, 3].astype(np.float32)
    temporal_diff = np.linalg.norm(raw - smooth, axis=1)
    norms = np.linalg.norm(feat[:, :4].astype(np.float32), axis=-1)
    if not np.isfinite(feat).all() or float(norms.min()) < .95 or float(norms.max()) > 1.05:
        raise RuntimeError("merged numerical validation failed")
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    np.savez(tmp, features=feat, row_keys=np.asarray([r["row_key"] for r in rows]), teacher_mask=teacher)
    generated = Path(str(tmp) + ".npz") if not tmp.exists() else tmp
    os.replace(generated, args.out)
    value = {
        "protocol": "trackocd_iclr27_phase17r_full_dinov3_features",
        "rows": n, "views": ["PROPOSAL_RAW", "PROPOSAL_CTX10", "PROPOSAL_CTX25", "PROPOSAL_CAUSAL_SMOOTHED", "GT_TIGHT", "GT_CTX10"],
        "shape": list(feat.shape), "dtype": str(feat.dtype), "teacher_rows": int(teacher.sum()),
        "proposal_norm_min": float(norms.min()), "proposal_norm_max": float(norms.max()),
        "later_temporal_feature_difference_rows": int((temporal_diff > 1e-4).sum()),
        "raw_temporal_identical_rows": int((temporal_diff <= 1e-4).sum()),
        "row_key_sha256": hashlib.sha256(json.dumps([r["row_key"] for r in rows]).encode()).hexdigest(),
        "shards": [str(p.resolve()) for p in args.shards], "shard_meta": shard_meta,
        "gt_teacher_only": True, "future_frames_used": False, "physical_id_semantic_feature": False
    }
    mt = args.meta.with_suffix(args.meta.suffix + ".tmp"); mt.write_text(json.dumps(value, indent=2, sort_keys=True)); os.replace(mt, args.meta)
    args.done.write_text(json.dumps({"out": str(args.out), "rows": n}))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
