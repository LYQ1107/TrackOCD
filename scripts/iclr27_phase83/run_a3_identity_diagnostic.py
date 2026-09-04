#!/usr/bin/env python3
"""Phase83 A3: identity-good/semantic-bad diagnostic after A2 failure.

This is a frozen, post-hoc diagnostic.  It compares the Q0 track vectors with
the full-coverage native-Q0 track vectors on the same causal prefix (16).  No
model is trained and no controller/threshold is touched.  GT categories are
used only to form descriptive positive/hard-negative retrieval statistics.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import load_frozen_tracks, order_key

OUT = ROOT / "outputs/iclr27_phase83"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
PREFIX = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32); return v / max(float(np.linalg.norm(v)), 1e-8)


def qstats(vectors: list[np.ndarray]) -> tuple[float, float]:
    if not vectors: return 0.0, 0.0
    arr = np.asarray(vectors, np.float32); mean = norm(arr.mean(axis=0)); variance = float(np.mean(1.0 - arr @ mean)); return variance, float(np.mean(arr @ mean))


def main() -> None:
    if not NATIVE.exists() or not FEATURES.exists(): raise FileNotFoundError("A2 native lineage/features are required")
    table = load_frozen_tracks(); feats = np.load(FEATURES, allow_pickle=False)["features"].astype(np.float32)
    native: list[dict[str, Any]] = []; by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    with NATIVE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line); native.append(row)
            if row.get("bbox_xyxy") is not None: by_image[(int(row["video_id"]), int(row["image_id"]))].append(len(native) - 1)
    if len(native) != len(feats): raise RuntimeError(f"native/features mismatch {len(native)} vs {len(feats)}")
    mapped: dict[str, list[tuple[int, int, float]]] = {}; records: list[dict[str, Any]] = []
    for key, seq in table.sequences.items():
        inds = list(seq.row_indices[:PREFIX]); hits = []
        for i in inds:
            r = table.rows[i]
            try: pb = [float(x) for x in json.loads(r["bbox_xyxy"])]
            except Exception: continue
            cands = by_image.get((int(r["video_id"]), int(r["image_id"])), [])
            if not cands: continue
            score, ni = max((box_iou(pb, native[j].get("bbox_xyxy")), j) for j in cands)
            if score >= .5: hits.append((i, ni, float(score)))
        if not hits: continue
        mapped[key] = hits
        native_vecs = [norm(feats[j]) for _, j, _ in hits]; native_mean = norm(np.asarray(native_vecs).mean(axis=0)); q0_vecs = [norm(table.features[i]) for i in inds]; q0_mean = norm(np.asarray(q0_vecs).mean(axis=0))
        native_ids = [int(native[j].get("physical_track_id", -1)) for _, j, _ in hits]; segments = 1 + sum(a != b for a, b in zip(native_ids, native_ids[1:]))
        seg_values: list[np.ndarray] = []
        for sid in dict.fromkeys(native_ids): seg_values.append(norm(np.asarray([v for v, x in zip(native_vecs, native_ids) if x == sid]).mean(axis=0)))
        adjacent = [float(a @ b) for a, b in zip(seg_values, seg_values[1:])]
        var_n, mean_cos_n = qstats(native_vecs); var_q, mean_cos_q = qstats(q0_vecs)
        records.append({"track_key": key, "video": int(table.metadata[key]["video"]), "category": int(table.metadata[key]["category"]), "causal_prefix": PREFIX, "public_rows": len(inds), "mapped_rows": len(hits), "native_mapping_fraction": len(hits) / max(1, len(inds)), "native_physical_ids": native_ids, "reconnected_segments": segments, "native_appearance_variance": var_n, "q0_appearance_variance": var_q, "native_mean_self_cosine": mean_cos_n, "q0_mean_self_cosine": mean_cos_q, "segment_adjacent_cosine_mean": float(np.mean(adjacent)) if adjacent else None, "segment_adjacent_cosine_count": len(adjacent), "raw_mean_shift_1_minus_cosine": float(1.0 - native_mean @ q0_mean), "native_vector": native_mean.tolist(), "q0_vector": q0_mean.tolist()})
    rec_by_key = {r["track_key"]: r for r in records}; keys = sorted(rec_by_key); n = len(keys)
    native_mat = np.asarray([rec_by_key[k]["native_vector"] for k in keys], np.float32); q0_mat = np.asarray([rec_by_key[k]["q0_vector"] for k in keys], np.float32); cats = np.asarray([rec_by_key[k]["category"] for k in keys]); vids = np.asarray([rec_by_key[k]["video"] for k in keys])
    native_sim = native_mat @ native_mat.T if n else np.zeros((0, 0), np.float32); q0_sim = q0_mat @ q0_mat.T if n else np.zeros((0, 0), np.float32)
    retrieval: list[dict[str, Any]] = []
    for i, k in enumerate(keys):
        pos = (cats == cats[i]) & (vids != vids[i]); neg = (cats != cats[i]) & (vids != vids[i]); pos[i] = False; neg[i] = False
        if not pos.any() or not neg.any(): continue
        retrieval.append({"track_key": k, "category": int(cats[i]), "video": int(vids[i]), "native_best_positive": float(native_sim[i, pos].max()), "native_hard_negative": float(native_sim[i, neg].max()), "native_gap": float(native_sim[i, pos].max() - native_sim[i, neg].max()), "q0_best_positive": float(q0_sim[i, pos].max()), "q0_hard_negative": float(q0_sim[i, neg].max()), "q0_gap": float(q0_sim[i, pos].max() - q0_sim[i, neg].max())})
    def avg(name: str, rs: list[dict[str, Any]]) -> float | None:
        vals = [float(r[name]) for r in rs if r.get(name) is not None]; return float(np.mean(vals)) if vals else None
    summary = {"mapped_tracks": len(records), "total_frozen_tracks": len(table.sequences), "mapped_track_fraction": len(records) / max(1, len(table.sequences)), "mean_mapped_rows": avg("mapped_rows", records), "mean_reconnected_segments": avg("reconnected_segments", records), "mean_native_appearance_variance": avg("native_appearance_variance", records), "mean_q0_appearance_variance": avg("q0_appearance_variance", records), "mean_native_self_cosine": avg("native_mean_self_cosine", records), "mean_q0_self_cosine": avg("q0_mean_self_cosine", records), "mean_segment_adjacent_cosine": avg("segment_adjacent_cosine_mean", records), "mean_raw_mean_shift": avg("raw_mean_shift_1_minus_cosine", records), "retrieval_queries": len(retrieval), "native_query_positive": avg("native_best_positive", retrieval), "native_query_hard_negative": avg("native_hard_negative", retrieval), "native_query_gap": avg("native_gap", retrieval), "q0_query_positive": avg("q0_best_positive", retrieval), "q0_query_hard_negative": avg("q0_hard_negative", retrieval), "q0_query_gap": avg("q0_gap", retrieval), "diagnostic_conclusion": "A3 compares identity/appearance statistics only; it does not select a controller or held checkpoint."}
    output = {"schema_version": "trackocd.phase83.a3.identity_diagnostic.v1", "phase": "Phase83 A3", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "prefix": PREFIX, "summary": summary, "track_records": records, "retrieval_records": retrieval, "native_lineage": str(NATIVE.resolve()), "native_sha256": sha(NATIVE), "native_features": str(FEATURES.resolve()), "native_features_sha256": sha(FEATURES), "q0_feature_sha256": table.feature_sha256, "posthoc_category_labels": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "audit/a3_identity_diagnostic.json", output); atomic_json(OUT / "status.json", {"phase": "Phase83", "route": "A3_IDENTITY_DIAGNOSTIC", "status": "COMPLETE", "summary": summary, "public_dev_q1_sealed_accessed": False}); atomic_json(OUT / "completion/a3_identity_diagnostic.done", {"status": "DONE", "metrics": str((OUT / "audit/a3_identity_diagnostic.json").resolve())})
    print(json.dumps({"status": "COMPLETE", "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
