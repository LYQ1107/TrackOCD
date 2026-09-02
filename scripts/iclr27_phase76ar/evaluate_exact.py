#!/usr/bin/env python3
"""Exact full TRAIN-disjoint validation for Phase76AR best checkpoints."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76ar.data import load_stream_payload
from src.iclr27_phase76ar.evaluator import evaluate_banks, p16
from src.iclr27_phase76ar.pair_cache import cache_hash, load_pair_cache
from src.iclr27_phase76ar.relation_model import SelectiveAnchoredRelation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76ar"


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_model(path: Path) -> SelectiveAnchoredRelation:
    try: payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: payload = torch.load(path, map_location="cpu")
    model = SelectiveAnchoredRelation().cpu(); model.load_state_dict(payload["model"]); model.eval(); return model


def summarize_stream(fold_results: list[dict]) -> dict:
    rows = [x["p16"] for x in fold_results]
    def mean(key: str) -> float: return sum(float(x[key]) for x in rows) / max(len(rows), 1)
    return {
        "r1": mean("r1"), "map": mean("map"), "hard_negative_gap": mean("hard_negative_gap"),
        "raw_r1": mean("raw_r1"), "raw_map": mean("raw_map"), "raw_hard_negative_gap": mean("raw_hard_negative_gap"),
        "delta_r1": mean("delta_r1"), "delta_map": mean("delta_map"), "delta_hard_gap": mean("delta_hard_gap"),
        "unsafe_flip_count": sum(int(x["unsafe_flip_count"]) for x in rows),
        "queries": sum(int(x["queries"]) for x in rows),
        "teacher_agreement": mean("teacher_agreement"), "teacher_use_rate": mean("teacher_use_rate"),
        "intervention_rate": mean("intervention_rate"), "mean_bank_gate": mean("mean_bank_gate"),
    }


def gate_for_stream(aggregate: dict, fold_results: list[dict], prefix_rows: list[dict]) -> dict:
    checks = {
        "p16_delta_r1_ge_0.02": aggregate["delta_r1"] >= 0.02,
        "p16_delta_map_ge_0.01": aggregate["delta_map"] >= 0.01,
        "p16_unsafe_zero": aggregate["unsafe_flip_count"] == 0,
        "all_prefix_unsafe_zero": all(int(x["unsafe_flip_count"]) == 0 for row in prefix_rows for x in [row["learned"]]),
        "hard_gap_non_worse_all_folds": all(float(x["p16"]["delta_hard_gap"]) >= -1e-7 for x in fold_results),
        "substantial_r1_3_of_4": sum(float(x["p16"]["delta_r1"]) >= 0.02 for x in fold_results) >= 3,
        "substantial_map_3_of_4": sum(float(x["p16"]["delta_map"]) >= 0.01 for x in fold_results) >= 3,
    }
    return checks


def main() -> None:
    table = load_frozen_tracks(); streams_all = []; fold_results_by_stream: dict[str, list[dict]] = {"legal_fit": [], "memory_mimic": []}
    for fold in range(4):
        streams = OUT / "banks" / f"streams_f{fold}.json"; memory_fit, legal_fit = load_stream_payload(streams, "fit"); memory_val, legal_val = load_stream_payload(streams, "val"); cache_path = OUT / "banks" / f"pair_cache_f{fold}.json"; cache = load_pair_cache(cache_path); model_path = OUT / "checkpoints" / f"ar1_formal_f{fold}_best.pt"; model = load_model(model_path)
        for name, banks in (("legal_fit", legal_val), ("memory_mimic", memory_val)):
            result = evaluate_banks(model, banks, table, cache, torch.device("cpu"), indices=list(range(len(banks))))
            fold_results_by_stream[name].append({"fold": fold, "checkpoint": str(model_path.resolve()), "stream": name, "p16": p16(result), "result": result, "cache_sha256": cache_hash(cache_path), "stream_sha256": hashlib.sha256(streams.read_bytes()).hexdigest()})
    aggregate = {}; gates = {}
    for name, rows in fold_results_by_stream.items():
        aggregate[name] = summarize_stream(rows)
        prefix_rows = []
        # Aggregate prefix safety/intervention summaries without dropping any
        # query.  Fold rows retain the complete per-query records above.
        for prefix in (1, 2, 4, 8, 16):
            metrics = [next(x for x in row["result"]["prefix_rows"] if x["prefix"] == prefix)["learned"] for row in rows]
            prefix_rows.append({"prefix": prefix, "unsafe_flip_count": sum(int(x["unsafe_flip_count"]) for x in metrics), "delta_r1": sum(float(x["r1"] - x["raw_r1"]) for x in metrics) / 4.0, "delta_map": sum(float(x["map"] - x["raw_map"]) for x in metrics) / 4.0, "delta_hard_gap": sum(float(x["hard_negative_gap"] - x["raw_hard_negative_gap"]) for x in metrics) / 4.0})
        gates[name] = gate_for_stream(aggregate[name], rows, [{"learned": {"unsafe_flip_count": x["unsafe_flip_count"]}} for x in prefix_rows])
        # The helper above expects each prefix row's learned metric; rewrite
        # the all-prefix condition explicitly to keep the artifact readable.
        gates[name]["all_prefix_unsafe_zero"] = all(x["unsafe_flip_count"] == 0 for x in prefix_rows)
        aggregate[name]["prefix_summary"] = prefix_rows
    legal_gate = all(gates["legal_fit"].values()); memory_gate = all(gates["memory_mimic"].values())
    decision = "PHASE76AR_GATE_R_PASS" if legal_gate and memory_gate else "PHASE76AR_GATE_R_FAIL_ROUTE_TO_PHASE76S_OR_PHASE76G"
    obj = {"phase":"Phase76AR","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"decision":decision,"fold_results":fold_results_by_stream,"aggregate":aggregate,"gates":gates,"raw_structural_parity":"checked by prefix raw equality in evaluator; no held/public/sealed access","controller_run":False,"state_memory_run":False,"sealed_accessed":False,"held_event_accessed_for_model":False,"public_or_dev_accessed":False,"selection":"best checkpoints selected by TRAIN-disjoint validation only; exact replay uses all validation banks","protocol":"raw-first selective relation; dual stream; <=12 prefix-union negatives + <=3 positives; prefixes 1,2,4,8,16"}
    atomic(OUT / "metrics/phase76ar_exact_retrieval.json", obj); atomic(OUT / "audit/phase76ar_decision.json", obj); atomic(OUT / "completion/exact_eval.done", {"phase":"Phase76AR","decision":decision,"metrics":str(OUT/"metrics/phase76ar_exact_retrieval.json")}); print(json.dumps({"phase":"Phase76AR","decision":decision,"aggregate":aggregate,"gates":gates}, sort_keys=True))


if __name__ == "__main__": main()
