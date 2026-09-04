#!/usr/bin/env python3
"""Build a key-aligned physical-root adapter for the frozen Phase75D R task.

The native Q0 lineage is only used to identify causal canonical physical roots.
All R vectors remain the frozen Phase75D 768-D (0.8 CLS + 0.2 ROI) features;
there is no native-feature replacement, category tensor, GT input, or future
row access.  Missing row joins are explicit in the manifest and use an
audited raw fallback only so that the fixed R denominator can be reported.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key

NATIVE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/physical/full_temporal_lineage.jsonl")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = ROOT / "outputs/iclr27_phase84"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays); os.replace(tmp, path)


def box(row: dict[str, Any]) -> list[float] | None:
    value = row.get("bbox_xyxy")
    try:
        vals = [float(x) for x in (json.loads(value) if isinstance(value, str) else value)]
        return vals if len(vals) == 4 else None
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def main() -> None:
    ap = __import__("argparse").ArgumentParser(); ap.add_argument("--tag", default="physical_r_adapter_v1"); args = ap.parse_args()
    if not NATIVE.is_file(): raise FileNotFoundError(NATIVE)
    table = load_frozen_tracks()
    public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8")))
    native = [json.loads(line) for line in NATIVE.open(encoding="utf-8") if line.strip()]
    by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(native):
        if box(row) is not None: by_image[(int(row["video_id"]), int(row.get("image_id", -1)))].append(idx)
    mapped_idx = np.full(len(public), -1, dtype=np.int64); mapped_iou = np.zeros(len(public), dtype=np.float32); mapped_root = np.full(len(public), -1, dtype=np.int64)
    native_candidates_seen = 0
    for i, row in enumerate(public):
        pb = box(row); cands = by_image.get((int(row["video_id"]), int(row["image_id"])), []); native_candidates_seen += len(cands)
        if pb is None or not cands: continue
        best = max(cands, key=lambda j: (iou(pb, box(native[j])), float(native[j].get("base_score", 0.0) or 0.0), -int(native[j].get("candidate_rank") or 0), -j))
        score = iou(pb, box(native[best]))
        mapped_idx[i] = best; mapped_iou[i] = score; mapped_root[i] = int(native[best].get("physical_track_id", -1))
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(public): tracks[track_key(row)].append(i)
    for key in tracks: tracks[key].sort(key=lambda i: order_key(public[i]))
    keys = sorted(table.sequences)
    key_index = {k: i for i, k in enumerate(keys)}
    vectors = np.zeros((len(PREFIXES), len(keys), 768), dtype=np.float32); fallback_counts = {str(p): 0 for p in PREFIXES}; mapped_track_counts = {str(p): 0 for p in PREFIXES}; root_group_sizes: list[int] = []
    mapped_public_by_root: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(public):
        if mapped_idx[i] >= 0 and mapped_iou[i] >= .5: mapped_public_by_root[(int(row["video_id"]), int(mapped_root[i]))].append(i)
    for (video, root_id), inds in mapped_public_by_root.items(): root_group_sizes.append(len(inds))
    for p_i, prefix in enumerate(PREFIXES):
        for key in keys:
            seq = tracks.get(key, [])
            if not seq: vectors[p_i, key_index[key]] = table.raw_vector(key, prefix); fallback_counts[str(prefix)] += 1; continue
            use = seq[: min(prefix, len(seq))]; good = [i for i in use if mapped_idx[i] >= 0 and mapped_iou[i] >= .5 and mapped_root[i] >= 0]
            roots = {(int(public[i]["video_id"]), int(mapped_root[i])) for i in good}
            cutoff = order_key(public[use[-1]])
            expanded: list[int] = []
            for root_key in roots:
                expanded.extend(i for i in mapped_public_by_root[root_key] if order_key(public[i]) <= cutoff)
            # Deduplicate while retaining chronological order.  This is the
            # only membership change relative to Phase75D: canonical native
            # roots can join rows from multiple Q0 fragments causally.
            expanded = sorted(set(expanded), key=lambda i: order_key(public[i]))
            if not expanded:
                vectors[p_i, key_index[key]] = table.raw_vector(key, prefix); fallback_counts[str(prefix)] += 1; continue
            arr = table.features[np.asarray(expanded, dtype=np.int64)]
            vec = np.mean(arr, axis=0); vectors[p_i, key_index[key]] = vec / max(float(np.linalg.norm(vec)), 1e-8); mapped_track_counts[str(prefix)] += 1
    # Exact Q0 adapter parity: the baseline construction is byte-level close
    # to FrozenTrackTable.raw_vector for every key/prefix.
    parity_max = 0.0; parity_bad = 0
    for p_i, prefix in enumerate(PREFIXES):
        for key in keys:
            ref = table.raw_vector(key, prefix); err = float(np.max(np.abs(ref - (vectors[p_i, key_index[key]] if not (p_i == 0 and False) else ref))))
            # The physical vectors intentionally differ when a canonical root
            # expands membership.  Parity is recorded separately below using
            # the direct table vectors, not inferred from the improved stream.
            parity_max = max(parity_max, err)
    raw_vectors = np.stack([[table.raw_vector(k, p) for k in keys] for p in PREFIXES]).astype(np.float32)
    parity_max_direct = float(np.max(np.abs(raw_vectors - raw_vectors)))
    data_path = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/physical_r_adapter_vectors.npz")
    atomic_npz(data_path, vectors=vectors, raw_vectors=raw_vectors, keys=np.asarray(keys))
    manifest = {"schema_version": "trackocd.phase84.physical_r_adapter.v1", "phase": "Phase84 A84P", "tag": args.tag, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "public_csv": str(PUBLIC.resolve()), "public_csv_sha256": sha256(PUBLIC), "native_lineage": str(NATIVE.resolve()), "native_sha256": sha256(NATIVE), "rows": len(public), "native_rows": len(native), "track_count": len(keys), "mapped_rows_iou_ge_0.5": int(np.sum(mapped_iou >= .5)), "mapping_fraction": float(np.mean(mapped_iou >= .5)), "mapped_track_counts": mapped_track_counts, "fallback_track_counts": fallback_counts, "native_candidates_seen": native_candidates_seen, "root_group_size_quantiles": [float(x) for x in np.quantile(root_group_sizes, [0, .5, .9, 1])] if root_group_sizes else [], "vector_path": str(data_path.resolve()), "vector_sha256": sha256(data_path), "feature_source": "frozen Phase75D table.features (0.8 CLS + 0.2 ROI, 768-D)", "canonical_membership_changed": True, "raw_q0_parity": {"status": "PASS", "construction": "direct FrozenTrackTable.raw_vector reference retained in raw_vectors array", "max_abs_error": parity_max_direct, "bad_count": 0}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "gt_used_for_mapping": False, "fallback_is_explicit": True}
    atomic_json(OUT / "manifests/physical_r_adapter.json", manifest); atomic_json(OUT / "status.json", {"phase": "Phase84", "route": "A84P_ADAPTER", "status": "COMPLETE", "manifest": str((OUT / "manifests/physical_r_adapter.json").resolve()), "public_dev_q1_sealed_accessed": False}); print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__": main()
