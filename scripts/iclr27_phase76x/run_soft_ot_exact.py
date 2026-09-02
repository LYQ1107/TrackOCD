#!/usr/bin/env python3
"""Run the one registered Phase76X soft-OT primitive on frozen banks."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records
from src.iclr27_phase76ar.data import load_stream_payload
from src.iclr27_phase76x.soft_ot import anchored_ot_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76x"
AR_OUT = ROOT / "outputs/iclr27_phase76ar"
PREFIXES = (1, 2, 4, 8, 16)


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def score_bank(table: Any, bank: Any, prefix: int) -> tuple[list[float], list[float]]:
    query = table.get_frame_sequence(bank.query_key, prefix); raw_scores: list[float] = []; learned: list[float] = []
    for candidate in bank.candidates:
        cand = table.get_frame_sequence(candidate, 16); raw = float(table.raw_vector(bank.query_key, prefix) @ table.raw_vector(candidate, 16)); raw_scores.append(raw); learned.append(anchored_ot_score(query, cand, raw, alpha=0.5))
    return raw_scores, learned


def eval_stream(table: Any, banks: list[Any]) -> dict[str, Any]:
    prefix_results = []
    for prefix in PREFIXES:
        records = []
        for bank in banks:
            raw, learned = score_bank(table, bank, prefix)
            records.append({"query_key": bank.query_key, "category": bank.category, "video": bank.video, "candidates": list(bank.candidates), "positives": list(bank.positives if hasattr(bank, "positives") else bank.positive_keys), "negatives": list(bank.negatives if hasattr(bank, "negatives") else bank.negative_keys), "scores": learned, "raw_scores": raw})
        metric = score_records(records)
        prefix_results.append({"prefix": prefix, "metric": metric, "records": records})
    return {"prefix_results": prefix_results}


def p16(result: dict[str, Any]) -> dict[str, Any]:
    metric = next(item["metric"] for item in result["prefix_results"] if item["prefix"] == 16)
    return {"r1": metric["r1"], "map": metric["map"], "raw_r1": metric["raw_r1"], "raw_map": metric["raw_map"], "hard_negative_gap": metric["hard_negative_gap"], "raw_hard_negative_gap": metric["raw_hard_negative_gap"], "delta_r1": metric["r1"] - metric["raw_r1"], "delta_map": metric["map"] - metric["raw_map"], "delta_hard_gap": metric["hard_negative_gap"] - metric["raw_hard_negative_gap"], "unsafe_flip_count": metric["unsafe_flip_count"], "queries": metric["queries"], "category_macro_r1": metric["category_macro_r1"], "video_macro_r1": metric["video_macro_r1"]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", choices=("legal_fit", "memory_mimic", "both"), default="both"); args = ap.parse_args()
    table = load_frozen_tracks(); rows_by_stream = {"legal_fit": [], "memory_mimic": []}; fold_results = {"legal_fit": [], "memory_mimic": []}
    for fold in range(4):
        streams_path = AR_OUT / "banks" / f"streams_f{fold}.json"; memory, legal = load_stream_payload(streams_path, "val"); selected = (("legal_fit", legal), ("memory_mimic", memory)) if args.stream == "both" else ((args.stream, legal if args.stream == "legal_fit" else memory),)
        for name, banks in selected:
            result = eval_stream(table, banks); row = {"fold": fold, "stream": name, "p16": p16(result), "result": result, "stream_sha256": hashlib.sha256(streams_path.read_bytes()).hexdigest()}; fold_results[name].append(row)
    aggregate = {}
    for name, rows in fold_results.items():
        if not rows: continue
        agg = aggregate_fold_metrics([{"r1": r["p16"]["r1"], "r5": 0, "map": r["p16"]["map"], "raw_r1": r["p16"]["raw_r1"], "raw_r5": 0, "raw_map": r["p16"]["raw_map"], "hard_negative_gap": r["p16"]["hard_negative_gap"], "raw_hard_negative_gap": r["p16"]["raw_hard_negative_gap"], "category_macro_r1": r["p16"]["category_macro_r1"], "video_macro_r1": r["p16"]["video_macro_r1"], "queries": r["p16"]["queries"], "unsafe_flip_count": r["p16"]["unsafe_flip_count"], "unsafe_flip_micro_rate": r["p16"]["unsafe_flip_count"] / max(r["p16"]["queries"], 1), "top1_change_count": 0, "top1_change_rate": 0} for r in rows]); agg.update({"delta_r1": float(np.mean([r["p16"]["delta_r1"] for r in rows])), "delta_map": float(np.mean([r["p16"]["delta_map"] for r in rows])), "delta_hard_gap": float(np.mean([r["p16"]["delta_hard_gap"] for r in rows])), "prefixes": {str(p): {"delta_r1": float(np.mean([next(x["metric"]["r1"] - x["metric"]["raw_r1"] for x in r["result"]["prefix_results"] if x["prefix"] == p) for r in rows])), "delta_map": float(np.mean([next(x["metric"]["map"] - x["metric"]["raw_map"] for x in r["result"]["prefix_results"] if x["prefix"] == p) for r in rows]))} for p in PREFIXES}}); aggregate[name] = agg
    gates = {}
    for name, rows in fold_results.items():
        if not rows: continue
        a = aggregate[name]; gates[name] = {"delta_r1_ge_0.02": a["delta_r1"] >= 0.02, "delta_map_ge_0.01": a["delta_map"] >= 0.01, "unsafe_zero": sum(r["p16"]["unsafe_flip_count"] for r in rows) == 0, "hard_gap_non_worse_each_fold": all(r["p16"]["delta_hard_gap"] >= -1e-7 for r in rows), "r1_three_of_four": sum(r["p16"]["delta_r1"] >= 0.02 for r in rows) >= 3, "map_three_of_four": sum(r["p16"]["delta_map"] >= 0.01 for r in rows) >= 3}
    decision = "PHASE76X_GATE_R_PASS" if gates and all(all(v for v in check.values()) for check in gates.values()) else "PHASE76X_GATE_R_FAIL_R_EXHAUSTED_UNDER_FROZEN_FEATURE_PROTOCOL"
    obj = {"phase": "Phase76X", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "primitive": "uniform-marginal soft optimal transport; temperature=0.07; iterations=50; raw anchor alpha=0.5", "decision": decision, "fold_results": fold_results, "aggregate": aggregate, "gates": gates, "controller_run": False, "state_memory_run": False, "sealed_accessed": False, "public_or_dev_accessed": False, "protocol": "frozen Phase76AR banks, prefixes 1/2/4/8/16, candidate prefix16; no training"}
    atomic(OUT / "metrics" / "phase76x_exact_retrieval.json", obj); atomic(OUT / "audit" / "phase76x_decision.json", obj); atomic(OUT / "completion" / "exact_eval.done", {"phase": "Phase76X", "decision": decision, "metrics": str(OUT / "metrics" / "phase76x_exact_retrieval.json")}); print(json.dumps({"phase": "Phase76X", "decision": decision, "aggregate": aggregate, "gates": gates}, sort_keys=True))


if __name__ == "__main__": main()
