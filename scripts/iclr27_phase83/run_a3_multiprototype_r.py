#!/usr/bin/env python3
"""Phase83 A3: fixed three-prototype causal track representation diagnostic."""
from __future__ import annotations

import datetime as dt
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

OUT = ROOT / "outputs/iclr27_phase83"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
M = 3


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32); return x / max(float(np.linalg.norm(x)), 1e-8)


def prototypes(table: Any, key: str, prefix: int) -> np.ndarray:
    seq = table.get_frame_sequence(key, prefix)
    if len(seq) == 0: return np.zeros((0, table.features.shape[1]), np.float32)
    # Fixed causal bins: each prototype is a contiguous prefix chunk.  No
    # future suffix is consulted, and M=3 is not swept.
    bins = min(M, len(seq)); out = []
    for i in range(bins):
        lo = (i * len(seq)) // bins; hi = ((i + 1) * len(seq)) // bins
        out.append(norm(seq[lo:hi].mean(axis=0)))
    return np.asarray(out, np.float32)


def symmetric_max(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b): return 0.0
    sim = a @ b.T
    return float((sim.max(axis=1).mean() + sim.max(axis=0).mean()) * .5)


def main() -> None:
    table = load_frozen_tracks(); vec = {p: {k: prototypes(table, k, p) for k in table.sequences} for p in PREFIXES}
    rows = []; aggregate = {}
    for fold in range(4):
        manifest = json.loads((EPISODES / f"episode_manifest_f{fold}.json").read_text(encoding="utf-8")); keys = sorted({str(r["query_track_key"]) for r in manifest["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
        vids = np.asarray([table.metadata[k]["video"] for k in keys], np.int64); cats = np.asarray([table.metadata[k]["category"] for k in keys], np.int64)
        for p in PREFIXES:
            available = [k for k in keys if len(vec[p][k])]; vv = np.asarray([table.metadata[k]["video"] for k in available], np.int64); cc = np.asarray([table.metadata[k]["category"] for k in available], np.int64); idx = np.arange(len(available)); recs = []
            for i, q in enumerate(available):
                mask = (idx != i) & (vv != vv[i]); ci = idx[mask]; cand = [available[int(j)] for j in ci]; pos = [available[int(j)] for j in ci if cc[int(j)] == cc[i]]; neg = [available[int(j)] for j in ci if cc[int(j)] != cc[i]]
                qv = vec[p][q]; scores = [symmetric_max(qv, vec[p][available[int(j)]]) for j in ci]; raw_q = table.raw_vector(q, p); raw_scores = [float(raw_q @ table.raw_vector(available[int(j)], p)) for j in ci]
                recs.append({"query_key": q, "category": int(cc[i]), "video": int(vv[i]), "candidates": cand, "positives": pos, "negatives": neg, "scores": scores, "raw_scores": raw_scores})
            m = score_records(recs); compact = {k: m[k] for k in ("queries", "r1", "r5", "map", "raw_r1", "raw_r5", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "category_macro_r1", "video_macro_r1", "unsafe_flip_count", "unsafe_flip_micro_rate", "top1_change_count", "top1_change_rate")}; rows.append({"fold": fold, "prefix": p, "metrics": compact, "keys_total": len(keys), "keys_evaluable": len(available)}); aggregate[str(p)] = aggregate.get(str(p), []) + [compact]
    agg = {p: aggregate_fold_metrics(v) for p, v in aggregate.items()}; p16 = [x for x in rows if x["prefix"] == 16]; direction = [x["metrics"]["r1"] >= x["metrics"]["raw_r1"] and x["metrics"]["map"] >= x["metrics"]["raw_map"] for x in p16]
    output = {"schema_version": "trackocd.phase83.a3.multiprototype_r.v1", "phase": "Phase83 A3", "representation": "M=3 fixed contiguous causal prototypes; symmetric max prototype cosine", "prefixes": list(PREFIXES), "aggregate": agg, "folds": rows, "gate_diagnostic": {"p16_fold_non_decrease": direction, "p16_folds_non_decrease": int(sum(direction)), "p16_unsafe_flip_count": int(sum(x["metrics"]["unsafe_flip_count"] for x in p16)), "held_events_used_for_selection": False}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "controller_run": False}
    atomic_json(OUT / "metrics/a3_multiprototype_r.json", output); atomic_json(OUT / "status.json", {"phase": "Phase83", "route": "A3_MULTIPROTOTYPE_R", "status": "COMPLETE", "p16": agg["16"], "gate_diagnostic": output["gate_diagnostic"], "public_dev_q1_sealed_accessed": False}); atomic_json(OUT / "completion/a3_multiprototype_r.done", {"status": "DONE", "metrics": str((OUT / "metrics/a3_multiprototype_r.json").resolve())})
    print(json.dumps({"status": "COMPLETE", "p16": agg["16"], "gate_diagnostic": output["gate_diagnostic"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
