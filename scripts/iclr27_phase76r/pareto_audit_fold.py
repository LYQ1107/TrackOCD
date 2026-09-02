#!/usr/bin/env python3
"""Exact p16 Pareto audit for one frozen Phase75E fold.

This script never uses held outcomes.  It evaluates every 500-step checkpoint
against the same all-candidate global and explicit manifest-legal banks used
by Phase75D, then writes one atomically-renamed fold JSON file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase75e.evaluator import _global_records_light, _load_legal_builder
from src.iclr27_phase75e.model import LowRankFeatureAdapter
from src.iclr27_phase75e.pairwise_adapter import adapter_drift
from src.iclr27_phase75d.pairwise_correspondence import fast_hungarian_score
from src.iclr27_phase75d.retrieval_metrics import score_records
from src.iclr27_phase76r.errata import checkpoint_in_safe_window

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76r/pareto"
CP = Path("/data2/usr_for_deadline/trackocd_phase75e/checkpoints")


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def checkpoint_paths(fold: int) -> list[tuple[int, Path]]:
    pat = re.compile(rf"phase75e_formal_f{fold}_step(\d+)\.pt$")
    out = []
    for p in CP.glob(f"phase75e_formal_f{fold}_step*.pt"):
        m = pat.search(p.name)
        if m: out.append((int(m.group(1)), p))
    return sorted(out)


def load_model(path: Path) -> LowRankFeatureAdapter:
    try: ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: ck = torch.load(path, map_location="cpu")
    model = LowRankFeatureAdapter().cpu(); model.load_state_dict(ck["model"]); model.eval(); return model


def seq_cache(model: torch.nn.Module, table, keys: set[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    adapted: dict[str, np.ndarray] = {}; raw: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key in sorted(keys):
            arr = table.get_frame_sequence(key, 16).astype(np.float32, copy=False)
            raw[key] = arr
            adapted[key] = model(torch.as_tensor(arr)).cpu().numpy().astype(np.float32, copy=False)
    return raw, adapted


def score_section(records: list[dict], adapted: dict[str, np.ndarray], support_prefix: bool = False) -> dict:
    out = []
    for rec in records:
        q = adapted[str(rec["query_key"])]
        scores = []
        for c in rec["candidates"]:
            scores.append(float(fast_hungarian_score(q, adapted[str(c)])))
        x = dict(rec); x["scores"] = scores; out.append(x)
    metric = score_records(out)
    metric.pop("per_query", None)
    return metric


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--tag", default="phase76r-pareto")
    args = ap.parse_args(); fold = args.fold
    completion = OUT.parent / "completion"; marker = completion / f"pareto_f{fold}.launched"; done = completion / f"pareto_f{fold}.done"
    if done.exists(): raise RuntimeError(f"already complete: {done}")
    atomic(marker, {"phase": "Phase76R", "fold": fold, "pid": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "steps": 30})
    table = load_frozen_tracks()
    global_records, global_info = _global_records_light(table, fold, 16, None)
    legal_builder = _load_legal_builder(); legal_records, unevaluable, legal_info = legal_builder(table, fold, 16)
    # Raw comparator records are immutable and reused for every checkpoint.
    raw_global = score_records(global_records); raw_global.pop("per_query", None)
    raw_legal = score_records(legal_records); raw_legal.pop("per_query", None)
    keys = {str(r["query_key"]) for r in global_records}
    keys.update(str(c) for r in global_records for c in r["candidates"])
    keys.update(str(r["query_key"]) for r in legal_records)
    keys.update(str(c) for r in legal_records for c in r["candidates"])
    raw_seq = {k: table.get_frame_sequence(k, 16).astype(np.float32, copy=False) for k in sorted(keys)}
    rows = []
    for step, path in checkpoint_paths(fold):
        model = load_model(path)
        raw, adapted = seq_cache(model, table, keys)
        gm = score_section(global_records, adapted)
        lm = score_section(legal_records, adapted)
        cos = []
        rel = []
        for k in keys:
            if not len(raw[k]): continue
            a = adapted[k]; r = raw[k]
            cos.extend(np.sum(a * r, axis=1).tolist())
            rel.extend((np.linalg.norm(a-r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-8)).tolist())
        cos_arr = np.asarray(cos, dtype=np.float32); rel_arr = np.asarray(rel, dtype=np.float32)
        row = {
            "phase": "Phase76R", "fold": fold, "step": step, "checkpoint": str(path.resolve()),
            "global_queries": gm["queries"], "legal_queries": lm["queries"],
            "global_raw_r1": float(raw_global["raw_r1"]), "global_learned_r1": float(gm["r1"]), "global_delta_r1": float(gm["r1"]-raw_global["raw_r1"]),
            "global_raw_map": float(raw_global["raw_map"]), "global_learned_map": float(gm["map"]), "global_delta_map": float(gm["map"]-raw_global["raw_map"]),
            "global_raw_hard_gap": float(raw_global["raw_hard_negative_gap"]), "global_learned_hard_gap": float(gm["hard_negative_gap"]), "global_delta_hard_gap": float(gm["hard_negative_gap"]-raw_global["raw_hard_negative_gap"]),
            "global_unsafe": int(gm["unsafe_flip_count"]), "global_top1_change": int(gm["top1_change_count"]),
            "legal_raw_r1": float(raw_legal["raw_r1"]), "legal_learned_r1": float(lm["r1"]), "legal_delta_r1": float(lm["r1"]-raw_legal["raw_r1"]),
            "legal_raw_map": float(raw_legal["raw_map"]), "legal_learned_map": float(lm["map"]), "legal_delta_map": float(lm["map"]-raw_legal["raw_map"]),
            "legal_raw_hard_gap": float(raw_legal["raw_hard_negative_gap"]), "legal_learned_hard_gap": float(lm["hard_negative_gap"]), "legal_delta_hard_gap": float(lm["hard_negative_gap"]-raw_legal["raw_hard_negative_gap"]),
            "legal_unsafe": int(lm["unsafe_flip_count"]), "legal_top1_change": int(lm["top1_change_count"]),
            "mean_raw_adapt_cosine": float(cos_arr.mean()), "p05_raw_adapt_cosine": float(np.quantile(cos_arr, .05)), "p50_raw_adapt_cosine": float(np.quantile(cos_arr, .50)), "p95_raw_adapt_cosine": float(np.quantile(cos_arr, .95)), "delta_norm_over_raw": float(rel_arr.mean()),
        }
        row["safe_window"] = checkpoint_in_safe_window(row)
        rows.append(row)
    target = OUT / f"fold{fold}.json"; atomic(target, {"phase": "Phase76R", "fold": fold, "global_inventory": global_info, "legal_inventory": legal_info, "unevaluable": unevaluable, "raw_global": raw_global, "raw_legal": raw_legal, "checkpoints": rows, "checkpoint_count": len(rows), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    atomic(done, {"phase": "Phase76R", "fold": fold, "checkpoint_count": len(rows), "output": str(target)})
    print(json.dumps({"phase": "Phase76R", "fold": fold, "checkpoint_count": len(rows), "output": str(target)}, sort_keys=True), flush=True)


if __name__ == "__main__": main()

