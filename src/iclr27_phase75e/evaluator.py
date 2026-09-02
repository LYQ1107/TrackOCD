"""Phase75D-compatible evaluation of a frozen Phase75E adapter."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.iclr27_phase75d.protocol import FrozenTrackTable, PREFIXES
from src.iclr27_phase75d.pairwise_correspondence import fast_hungarian_score
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records

from .pairwise_adapter import pairwise_torch_score

def _load_global_builder():
    from scripts.iclr27_phase75d.run_pairwise_r import global_records as builder
    return builder


def _load_legal_builder():
    from scripts.iclr27_phase75d.run_pairwise_r import legal_records as builder
    return builder


def _global_records_light(table: FrozenTrackTable, fold: int, prefix: int, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the frozen R-global candidate universe without pre-scoring it.

    Phase75D's CLI builder computes its pairwise score immediately.  That is
    correct for the exact benchmark but would repeat millions of CPU
    assignments at every in-training checkpoint.  This equivalent builder
    keeps the same sorted validation keys, video exclusion, positives and
    negatives; the adapter scorer fills ``scores`` below.
    """
    data = __import__("json").loads((__import__("pathlib").Path(__file__).resolve().parents[2] / "outputs/iclr27_phase30/manifests" / f"episode_manifest_f{fold}.json").read_text())
    keys = sorted({str(r["query_track_key"]) for r in data["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
    if limit is not None:
        keys = keys[: int(limit)]
    videos = np.asarray([table.metadata[k]["video"] for k in keys], dtype=np.int64)
    cats = np.asarray([table.metadata[k]["category"] for k in keys], dtype=np.int64)
    records: list[dict[str, Any]] = []
    all_idx = np.arange(len(keys), dtype=np.int64)
    for i, qkey in enumerate(keys):
        cand_idx = all_idx[(all_idx != i) & (videos != videos[i])]
        candidates = [keys[int(j)] for j in cand_idx]
        positives = [keys[int(j)] for j in cand_idx if cats[int(j)] == cats[i]]
        negatives = [keys[int(j)] for j in cand_idx if cats[int(j)] != cats[i]]
        raw_q = table.raw_vector(qkey, prefix)
        raw_scores = [float(raw_q @ table.raw_vector(keys[int(j)], prefix)) for j in cand_idx]
        records.append({"query_key": qkey, "category": int(cats[i]), "video": int(videos[i]), "candidates": candidates, "positives": positives, "negatives": negatives, "scores": [0.0] * len(candidates), "raw_scores": raw_scores})
    info = {"fold": fold, "prefix": prefix, "validation_tracklets": len(keys), "candidate_rule": "all validation tracks except self and same video", "candidate_count_total": int(sum(len(r["candidates"]) for r in records)), "scope": "bounded_query_screen" if limit is not None else "exact_all_candidate"}
    return records, info


def _adapted_sequences(
    model: torch.nn.Module,
    table: FrozenTrackTable,
    keys: set[str],
    prefix: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key in sorted(keys):
            raw = table.get_frame_sequence(key, prefix).astype(np.float32, copy=False)
            if raw.shape[0] == 0:
                out[key] = raw
                continue
            x = torch.as_tensor(raw, device=device)
            out[key] = model(x).detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def _score_adapted_records(
    model: torch.nn.Module,
    table: FrozenTrackTable,
    records: list[dict[str, Any]],
    prefix: int,
    device: torch.device,
    support_prefix: int | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []
    all_keys: set[str] = set()
    for rec in records:
        all_keys.add(str(rec["query_key"]))
        all_keys.update(str(x) for x in rec["candidates"])
    # Legal support is always visible through prefix 16.  Global candidates
    # use the same prefix as the query, exactly as Phase75D registered.
    cache: dict[tuple[str, int], np.ndarray] = {}
    query_keys = {str(r["query_key"]) for r in records}
    qcache = _adapted_sequences(model, table, query_keys, prefix, device)
    for key in all_keys:
        # A legal support track can also appear as another query in the same
        # validation bank.  Cache both causal views rather than letting the
        # query-role branch shadow the support-prefix lookup.
        needed_prefixes = {prefix}
        if support_prefix is not None:
            needed_prefixes.add(support_prefix)
        for p in needed_prefixes:
            cache[(key, p)] = _adapted_sequences(model, table, {key}, p, device)[key]
    out: list[dict[str, Any]] = []
    # Inference is exact NumPy scoring: the model has already produced
    # normalized arrays, and ``fast_hungarian_score`` is algebraically the
    # Phase75D linear_sum_assignment score (with only exact small-size paths
    # for p=1/2).  This avoids millions of one-element torch allocations in a
    # checkpoint validation hook without changing candidate order or metrics.
    for rec in records:
        qkey = str(rec["query_key"])
        q = qcache[qkey]
        scores: list[float] = []
        for cand in rec["candidates"]:
            ck = str(cand)
            cp = support_prefix if support_prefix is not None else prefix
            scores.append(float(fast_hungarian_score(q, cache[(ck, cp)])))
        x = dict(rec)
        x["scores"] = scores
        out.append(x)
    return out


def evaluate_fold(
    model: torch.nn.Module,
    table: FrozenTrackTable,
    fold: int,
    device: torch.device,
    *,
    global_query_limit: int | None = None,
    legal_query_limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate one fold at all five prefixes.

    ``global_query_limit`` is used only by the bounded in-training screen; a
    final call passes ``None`` and therefore evaluates the exact all-candidate
    R-global.  Legal support is never searched or synthesized.
    """
    rows: list[dict[str, Any]] = []
    legal_builder = _load_legal_builder()
    for prefix in PREFIXES:
        global_rec, global_info = _global_records_light(table, fold, prefix, global_query_limit)
        legal_rec, unevaluable, legal_info = legal_builder(table, fold, prefix)
        if legal_query_limit is not None:
            legal_rec = legal_rec[: int(legal_query_limit)]
        raw_global = score_records(global_rec)
        raw_legal = score_records(legal_rec)
        learned_global_records = _score_adapted_records(model, table, global_rec, prefix, device)
        learned_legal_records = _score_adapted_records(model, table, legal_rec, prefix, device, support_prefix=16)
        learned_global = score_records(learned_global_records)
        learned_legal = score_records(learned_legal_records)
        rows.append({
            "prefix": prefix,
            "global": {"raw": raw_global, "learned": learned_global, "inventory": global_info, "scope": "bounded_query_screen" if global_query_limit is not None else "exact_all_candidate"},
            "legal": {"raw": raw_legal, "learned": learned_legal, "inventory": legal_info, "unevaluable": unevaluable, "scope": "bounded_query_screen" if legal_query_limit is not None else "exact_manifest_legal"},
        })
    return {"fold": fold, "prefix_rows": rows}


def aggregate_sections(fold_evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Build fold-macro aggregates while retaining query-micro counts."""
    out: dict[str, Any] = {}
    for section in ("global", "legal"):
        sec: dict[str, Any] = {}
        for prefix in PREFIXES:
            fold_rows = []
            for fe in fold_evals:
                row = next(x for x in fe["prefix_rows"] if x["prefix"] == prefix)[section]
                raw, learned = row["raw"], row["learned"]
                fold_rows.append({
                    "r1": learned["r1"], "r5": learned["r5"], "map": learned["map"],
                    "raw_r1": raw["r1"], "raw_r5": raw["r5"], "raw_map": raw["map"],
                    "hard_negative_gap": learned["hard_negative_gap"], "raw_hard_negative_gap": raw["raw_hard_negative_gap"],
                    "category_macro_r1": learned["category_macro_r1"], "video_macro_r1": learned["video_macro_r1"],
                    "queries": learned["queries"], "unsafe_flip_count": learned["unsafe_flip_count"],
                    "unsafe_flip_micro_rate": learned["unsafe_flip_micro_rate"], "top1_change_count": learned["top1_change_count"], "top1_change_rate": learned["top1_change_rate"],
                })
            agg = aggregate_fold_metrics(fold_rows)
            agg["delta_r1"] = agg["r1"] - agg["raw_r1"]
            agg["delta_map"] = agg["map"] - agg["raw_map"]
            agg["delta_hard_gap"] = agg["hard_negative_gap"] - agg["raw_hard_negative_gap"]
            sec[str(prefix)] = agg
        out[section] = sec
    return out
