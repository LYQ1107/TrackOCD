#!/usr/bin/env python3
"""Run Phase80A frozen global and dense-set correspondence diagnostics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80a"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def load_dense(cache_root: Path, expected_keys: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shards = sorted(cache_root.glob("dense_shard_*.npz"))
    if len(shards) != 4:
        raise RuntimeError(f"expected 4 dense shards, found {len(shards)}")
    cls_parts, patch_parts, keys = [], [], []
    for path in shards:
        z = np.load(path, allow_pickle=False)
        cls_parts.append(z["cls"])
        patch_parts.append(z["patch"])
        keys.extend(z["row_keys"].astype(str).tolist())
        if not np.all(z["valid"] == 1):
            raise RuntimeError(f"invalid rows in {path}")
    if keys != expected_keys:
        raise RuntimeError("dense row-key order does not exactly match corrected CSV")
    cls = np.concatenate(cls_parts, axis=0).astype(np.float32)
    patch = np.concatenate(patch_parts, axis=0).astype(np.float32)
    if cls.shape[1] != 768 or patch.shape[1:] != (32, 768):
        raise RuntimeError(f"unexpected dense arrays {cls.shape} {patch.shape}")
    return cls, patch, {"shards": [str(p.resolve()) for p in shards], "rows": len(keys), "cls_shape": list(cls.shape), "patch_shape": list(patch.shape)}


def validation_keys(fold: int, table: Any) -> list[str]:
    data = json.loads((EPISODES / f"episode_manifest_f{fold}.json").read_text(encoding="utf-8"))
    return sorted({str(r["query_track_key"]) for r in data["records"] if r.get("split") == "val" and str(r["query_track_key"]) in table.metadata})


def ap_score(scores: np.ndarray, positives: set[str], candidates: list[str]) -> tuple[float, bool, str]:
    order = np.argsort(scores)[::-1]
    hits = np.asarray([int(candidates[int(i)] in positives) for i in order], dtype=np.float32)
    cumulative = np.cumsum(hits)
    ap = float(np.sum(cumulative / (np.arange(len(hits)) + 1) * hits) / max(len(positives), 1))
    top = candidates[int(order[0])] if len(order) else ""
    return ap, bool(hits[0] if len(hits) else False), top


def dense_pair_scores(query: np.ndarray, candidates: np.ndarray, device: torch.device) -> np.ndarray:
    """Symmetric mutual-nearest patch score for one query and candidate bank."""
    q = torch.as_tensor(normalize(query), dtype=torch.float16, device=device)
    c = torch.as_tensor(normalize(candidates), dtype=torch.float16, device=device)
    sim = torch.einsum("qd,nkd->nqk", q, c)
    forward = sim.max(dim=2).values.mean(dim=1)
    backward = sim.max(dim=1).values.mean(dim=1)
    return ((forward + backward) * 0.5).float().cpu().numpy()


def track_dense(cls: np.ndarray, patch: np.ndarray, indices: tuple[int, ...], prefix: int) -> tuple[np.ndarray, np.ndarray]:
    use = np.asarray(indices[: min(prefix, len(indices))], dtype=np.int64)
    if len(use) == 0:
        return np.zeros(768, np.float32), np.zeros((16, 768), np.float32)
    global_vec = normalize(cls[use].mean(axis=0))
    tokens = normalize(patch[use].mean(axis=0))
    return global_vec, tokens[::2]


def evaluate_fold(table: Any, cls: np.ndarray, patch: np.ndarray, fold: int, prefix: int, device: torch.device) -> dict[str, Any]:
    keys = validation_keys(fold, table)
    seqs = {k: table.sequences[k].row_indices for k in keys}
    videos = {k: int(table.metadata[k]["video"]) for k in keys}
    cats = {k: int(table.metadata[k]["category"]) for k in keys}
    raw_vecs = {k: table.raw_vector(k, prefix) for k in keys}
    dense_global = {}
    dense_tokens = {}
    for k in keys:
        dense_global[k], dense_tokens[k] = track_dense(cls, patch, seqs[k], prefix)
    token_bank = np.stack([dense_tokens[k] for k in keys], axis=0)
    records = []
    for q_index, query in enumerate(keys):
        candidate_indices = [j for j, candidate in enumerate(keys) if candidate != query and videos[candidate] != videos[query]]
        candidates = [keys[j] for j in candidate_indices]
        positives = {keys[j] for j in candidate_indices if cats[keys[j]] == cats[query]}
        negatives = {keys[j] for j in candidate_indices if cats[keys[j]] != cats[query]}
        raw_scores = np.asarray([float(raw_vecs[query] @ raw_vecs[c]) for c in candidates], dtype=np.float32)
        global_scores = np.asarray([float(dense_global[query] @ dense_global[c]) for c in candidates], dtype=np.float32)
        dense_scores = dense_pair_scores(dense_tokens[query], token_bank[np.asarray(candidate_indices, dtype=np.int64)], device)
        row = {"query_key": query, "category": cats[query], "video": videos[query], "candidate_count": len(candidates), "positive_count": len(positives), "negative_count": len(negatives)}
        for name, scores in (("raw", raw_scores), ("global", global_scores), ("dense", dense_scores)):
            ap, top_hit, top_key = ap_score(scores, positives, candidates)
            order = np.argsort(scores)[::-1]
            hit5 = any(candidates[int(i)] in positives for i in order[:5])
            pos = scores[np.asarray([c in positives for c in candidates], dtype=bool)]
            neg = scores[np.asarray([c in negatives for c in candidates], dtype=bool)]
            row[name] = {"r1": int(top_hit), "r5": int(hit5), "map": ap, "hard_gap": float(pos.max(initial=-1.0) - neg.max(initial=-1.0)), "top1": top_key}
        row["dense_rescue"] = bool(row["raw"]["r1"] == 0 and row["dense"]["r1"] == 1)
        row["dense_harm"] = bool(row["raw"]["r1"] == 1 and row["dense"]["r1"] == 0)
        row["global_rescue"] = bool(row["raw"]["r1"] == 0 and row["global"]["r1"] == 1)
        row["global_harm"] = bool(row["raw"]["r1"] == 1 and row["global"]["r1"] == 0)
        records.append(row)

    def mean(name: str, field: str) -> float:
        return float(np.mean([r[name][field] for r in records])) if records else 0.0

    summary = {
        "fold": fold,
        "prefix": prefix,
        "queries": len(records),
        "raw": {f: mean("raw", f) for f in ("r1", "r5", "map", "hard_gap")},
        "global": {f: mean("global", f) for f in ("r1", "r5", "map", "hard_gap")},
        "dense": {f: mean("dense", f) for f in ("r1", "r5", "map", "hard_gap")},
        "dense_rescued_raw_wrong": int(sum(r["dense_rescue"] for r in records)),
        "dense_harmed_raw_correct": int(sum(r["dense_harm"] for r in records)),
        "global_rescued_raw_wrong": int(sum(r["global_rescue"] for r in records)),
        "global_harmed_raw_correct": int(sum(r["global_harm"] for r in records)),
    }
    summary["dense_net_rescue"] = summary["dense_rescued_raw_wrong"] - summary["dense_harmed_raw_correct"]
    summary["global_net_rescue"] = summary["global_rescued_raw_wrong"] - summary["global_harmed_raw_correct"]
    return {"summary": summary, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--cache-root", default="/data2/usr_for_deadline/trackocd_phase80a/dense_cache")
    args = ap.parse_args()
    table = load_frozen_tracks()
    expected = [str(r["row_key"]) for r in table.rows]
    cls, patch, cache_info = load_dense(Path(args.cache_root), expected)
    device = torch.device(args.device)
    folds = []
    all_records = []
    for fold in range(4):
        for prefix in PREFIXES:
            evaluated = evaluate_fold(table, cls, patch, fold, prefix, device)
            folds.append(evaluated["summary"])
            all_records.extend(evaluated["records"])
            print(json.dumps(evaluated["summary"], sort_keys=True), flush=True)
    aggregate = {}
    for prefix in PREFIXES:
        rows = [x for x in folds if x["prefix"] == prefix]
        aggregate[str(prefix)] = {}
        for method in ("raw", "global", "dense"):
            aggregate[str(prefix)][method] = {f: float(np.mean([x[method][f] for x in rows])) for f in ("r1", "r5", "map", "hard_gap")}
        for method in ("dense", "global"):
            aggregate[str(prefix)][f"{method}_rescued_raw_wrong"] = int(sum(x[f"{method}_rescued_raw_wrong"] for x in rows))
            aggregate[str(prefix)][f"{method}_harmed_raw_correct"] = int(sum(x[f"{method}_harmed_raw_correct"] for x in rows))
            aggregate[str(prefix)][f"{method}_net_rescue"] = int(sum(x[f"{method}_net_rescue"] for x in rows))
    raw_expected = {"r1": 0.8932193826961726, "map": 0.8483743539237845}
    raw_actual = aggregate["16"]["raw"]
    result = {
        "phase": "Phase80A",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "Phase30 TRAIN-disjoint global validation; candidates are different physical tracks and different videos",
        "prefixes": list(PREFIXES),
        "positive_label_metadata_only": True,
        "held_dev_q1_public_new_accessed": False,
        "cache": cache_info,
        "folds": folds,
        "aggregate": aggregate,
        "raw_parity": {"expected_phase30_p16": raw_expected, "actual": {"r1": raw_actual["r1"], "map": raw_actual["map"]}, "within_1e-6": abs(raw_actual["r1"] - raw_expected["r1"]) < 1e-6 and abs(raw_actual["map"] - raw_expected["map"]) < 1e-6},
        "routing_criterion": {
            "p16_dense_net_rescue": aggregate["16"]["dense_net_rescue"],
            "folds_net_positive": sum(x["dense_net_rescue"] > 0 for x in folds if x["prefix"] == 16),
            "folds_catastrophic_drop_gt_0.02": sum((x["dense"]["r1"] - x["raw"]["r1"]) < -0.02 for x in folds if x["prefix"] == 16),
            "criterion": "dense rescued_raw_wrong > harmed_raw_correct AND at least 3 folds net_rescue > 0 AND no fold R1 drop < -0.02",
        },
        "records": all_records,
    }
    atomic_json(OUT / "metrics/phase80a_dense_diagnostic.json", result)
    atomic_json(OUT / "audit/phase80a_decision.json", {k: result[k] for k in ("phase", "created_utc", "protocol", "aggregate", "raw_parity", "routing_criterion", "held_dev_q1_public_new_accessed")})
    atomic_json(OUT / "completion/phase80a_diagnostic.done", {"phase": "Phase80A", "metrics": str((OUT / "metrics/phase80a_dense_diagnostic.json").resolve())})
    print(json.dumps(result["routing_criterion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

