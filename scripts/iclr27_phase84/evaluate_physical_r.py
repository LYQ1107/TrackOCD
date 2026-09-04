#!/usr/bin/env python3
"""Evaluate Q0 and canonical-root vectors on the frozen Phase75D R universe."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records

OUT = ROOT / "outputs/iclr27_phase84"; EPISODES = ROOT / "outputs/iclr27_phase30/manifests"; ADAPTER = OUT / "manifests/physical_r_adapter.json"; DATA = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/physical_r_adapter_vectors.npz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def main() -> None:
    table = load_frozen_tracks(); manifest = json.loads(ADAPTER.read_text(encoding="utf-8")); z = np.load(DATA, allow_pickle=False); keys = [str(x) for x in z["keys"]]; key_idx = {k: i for i, k in enumerate(keys)}; physical = z["vectors"].astype(np.float32); raw = z["raw_vectors"].astype(np.float32)
    if tuple(physical.shape) != (len(PREFIXES), len(keys), 768): raise RuntimeError(f"bad physical vectors {physical.shape}")
    fold_rows: list[dict[str, Any]] = []; prefix_rows: list[dict[str, Any]] = []
    for fold in range(4):
        ep = json.loads((EPISODES / f"episode_manifest_f{fold}.json").read_text(encoding="utf-8")); val = sorted({str(r["query_track_key"]) for r in ep["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
        vids = np.asarray([table.metadata[k]["video"] for k in val], dtype=np.int64); cats = np.asarray([table.metadata[k]["category"] for k in val], dtype=np.int64); idx = np.arange(len(val), dtype=np.int64)
        for p_i, prefix in enumerate(PREFIXES):
            records: list[dict[str, Any]] = []
            for i, q in enumerate(val):
                ci = idx[(idx != i) & (vids != vids[i])]; cand = [val[int(j)] for j in ci]; pos = [val[int(j)] for j in ci if cats[int(j)] == cats[i]]; neg = [val[int(j)] for j in ci if cats[int(j)] != cats[i]]
                qv = physical[p_i, key_idx[q]]; rv = raw[p_i, key_idx[q]]; records.append({"query_key": q, "category": int(cats[i]), "video": int(vids[i]), "candidates": cand, "positives": pos, "negatives": neg, "scores": [float(qv @ physical[p_i, key_idx[c]]) for c in cand], "raw_scores": [float(rv @ raw[p_i, key_idx[c]]) for c in cand]})
            mm = score_records(records); compact = {k: mm[k] for k in ("queries", "r1", "r5", "map", "raw_r1", "raw_r5", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "category_macro_r1", "video_macro_r1", "unsafe_flip_count", "unsafe_flip_micro_rate", "top1_change_count", "top1_change_rate")}; row = {"fold": fold, "prefix": prefix, "metrics": compact, "candidate_rule": "all validation tracks except self and same video", "manifest": str((EPISODES / f"episode_manifest_f{fold}.json").resolve()), "manifest_sha256": sha256(EPISODES / f"episode_manifest_f{fold}.json")}; fold_rows.append(row); prefix_rows.append({"fold": fold, "prefix": prefix, **compact})
    aggregate = {}
    for prefix in PREFIXES: aggregate[str(prefix)] = aggregate_fold_metrics([x["metrics"] for x in fold_rows if x["prefix"] == prefix])
    p16 = aggregate["16"]; gate = {"p16": {"r1": p16["r1"], "raw_r1": p16["raw_r1"], "map": p16["map"], "raw_map": p16["raw_map"], "hard_negative_gap": p16["hard_negative_gap"], "raw_hard_negative_gap": p16["raw_hard_negative_gap"], "unsafe_flip_count": p16["unsafe_flip_count"]}, "folds_non_decreasing_both": sum(int(x["metrics"]["r1"] >= x["metrics"]["raw_r1"] and x["metrics"]["map"] >= x["metrics"]["raw_map"]) for x in fold_rows if x["prefix"] == 16), "safe_r_signal": bool(p16["unsafe_flip_count"] == 0 and p16["r1"] >= p16["raw_r1"] + .01 and p16["map"] >= p16["raw_map"] + .005)}
    result = {"schema_version": "trackocd.phase84.physical_r_adapter_metrics.v1", "phase": "Phase84 A84P", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "adapter_manifest": str(ADAPTER.resolve()), "adapter_manifest_sha256": sha256(ADAPTER), "vector_path": str(DATA.resolve()), "vector_sha256": sha256(DATA), "prefix": aggregate, "folds": fold_rows, "prefix_rows": prefix_rows, "gate_diagnostic": gate, "same_984_query_universe": True, "same_candidate_order": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "controller_run": False, "sealed_run": False}
    atomic_json(OUT / "metrics/physical_r_q0_adapter.json", result); atomic_json(OUT / "audit/a84_physical_r_metrics.json", result); atomic_json(OUT / "completion/physical_r_adapter.done", {"status": "DONE", "metrics": str((OUT / "metrics/physical_r_q0_adapter.json").resolve())}); print(json.dumps({"p16": p16, "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
