#!/usr/bin/env python3
"""Exact TRAIN-disjoint Phase76A retrieval and gate evaluation."""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import torch

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76a.candidate_bank import banks_hash, load_banks
from src.iclr27_phase76a.evaluator import evaluate_banks
from src.iclr27_phase76a.pair_cache import cache_hash, load_pair_cache
from src.iclr27_phase76a.relation_model import AnchoredRelationReranker

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76a"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase76a/checkpoints")


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(value, h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_model(path: Path) -> AnchoredRelationReranker:
    try: ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: ck = torch.load(path, map_location="cpu")
    m = AnchoredRelationReranker().cpu(); m.load_state_dict(ck["model"]); m.eval(); return m


def p16(result: dict) -> dict:
    metric = dict(next(x for x in result["prefix_rows"] if x["prefix"] == 16)["learned"])
    metric["delta_r1"] = float(metric["r1"] - metric["raw_r1"])
    metric["delta_map"] = float(metric["map"] - metric["raw_map"])
    metric["delta_hard_gap"] = float(metric["hard_negative_gap"] - metric["raw_hard_negative_gap"])
    return metric


def main() -> None:
    table = load_frozen_tracks(); fold_results = []; memory_results = []
    for fold in range(4):
        val_banks = load_banks(OUT / "banks" / f"val_f{fold}.json"); fit_banks = load_banks(OUT / "banks" / f"fit_f{fold}.json")
        val_cache = load_pair_cache(OUT / "banks" / f"pair_cache_val_f{fold}.json"); fit_cache = load_pair_cache(OUT / "banks" / f"pair_cache_fit_f{fold}.json")
        cp_link = OUT / "checkpoints" / f"phase76a_formal1_f{fold}_best.pt"; model = load_model(cp_link)
        val = evaluate_banks(model, val_banks, table, val_cache, torch.device("cpu"), limit=None)
        memory = evaluate_banks(model, fit_banks, table, fit_cache, torch.device("cpu"), limit=min(128, len(fit_banks)))
        fold_results.append({"fold":fold,"best_checkpoint":str(cp_link.resolve()),"val":val,"p16":p16(val),"val_bank_hash":banks_hash(val_banks),"val_cache_hash":cache_hash(OUT/"banks"/f"pair_cache_val_f{fold}.json")})
        memory_results.append({"fold":fold,"scope":"bounded_fit_memory_mimic_128","p16":p16(memory),"result":memory,"fit_bank_hash":banks_hash(fit_banks),"fit_cache_hash":cache_hash(OUT/"banks"/f"pair_cache_fit_f{fold}.json")})
    p16_rows = [x["p16"] for x in fold_results]
    def mean(key: str) -> float: return sum(float(x[key]) for x in p16_rows) / max(len(p16_rows), 1)
    aggregate = {"r1":mean("r1"),"map":mean("map"),"hard_negative_gap":mean("hard_negative_gap"),"raw_r1":mean("raw_r1"),"raw_map":mean("raw_map"),"raw_hard_negative_gap":mean("raw_hard_negative_gap"),"delta_r1":mean("delta_r1"),"delta_map":mean("delta_map"),"delta_hard_gap":mean("delta_hard_gap"),"unsafe_flip_count":sum(int(x["unsafe_flip_count"]) for x in p16_rows),"queries":sum(int(x["queries"]) for x in p16_rows)}
    prefix_gate = []
    for p in (1,2,4,8,16):
        rows = []
        for f in fold_results:
            metric = dict(next(x for x in f["val"]["prefix_rows"] if x["prefix"] == p)["learned"])
            metric["delta_r1"] = float(metric["r1"] - metric["raw_r1"])
            metric["delta_map"] = float(metric["map"] - metric["raw_map"])
            rows.append(metric)
        prefix_gate.append({"prefix":p,"unsafe":sum(int(x["unsafe_flip_count"]) for x in rows),"delta_r1":sum(float(x["delta_r1"]) for x in rows)/4.0,"delta_map":sum(float(x["delta_map"]) for x in rows)/4.0})
    fold_checks = []
    for f in fold_results:
        x=f["p16"]; fold_checks.append({"fold":f["fold"],"delta_r1_pass":float(x["delta_r1"])>=0.02,"delta_map_pass":float(x["delta_map"])>=0.01,"unsafe_zero":int(x["unsafe_flip_count"])==0,"hard_gap_non_worse":float(x["delta_hard_gap"])>=-1e-12,"delta_r1":x["delta_r1"],"delta_map":x["delta_map"],"delta_hard_gap":x["delta_hard_gap"],"unsafe":x["unsafe_flip_count"]})
    checks={"global_structural_raw_parity":json.loads((OUT/"audit/raw_anchor_parity.json").read_text())["pass"],"legal_p16_delta_r1_ge_0.02":aggregate["delta_r1"]>=0.02,"legal_p16_delta_map_ge_0.01":aggregate["delta_map"]>=0.01,"legal_p16_unsafe_zero":aggregate["unsafe_flip_count"]==0,"legal_hard_gap_non_worse_all_folds":all(x["hard_gap_non_worse"] for x in fold_checks),"legal_r1_3_of_4":sum(x["delta_r1_pass"] for x in fold_checks)>=3,"legal_map_3_of_4":sum(x["delta_map_pass"] for x in fold_checks)>=3,"prefix_unsafe_zero":all(x["unsafe"]==0 for x in prefix_gate)}
    gate_pass=all(checks.values()); decision="PHASE76A_GATE_R_PASS" if gate_pass else "PHASE76A_GATE_R_FAIL_STOP_BEFORE_STATE_MEMORY"
    obj={"phase":"Phase76A","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"decision":decision,"fold_results":fold_results,"fold_checks":fold_checks,"aggregate_p16":aggregate,"prefix_gate":prefix_gate,"memory_mimic":memory_results,"checks":checks,"controller_run":False,"state_memory_run":False,"sealed_accessed":False,"held_event_accessed_for_model":False,"checkpoint_selection":"TRAIN-disjoint bounded validation only; no held/public/sealed selection","protocol":"raw global scorer + <=15-candidate local relation bank"}
    atomic(OUT/"metrics/exact_retrieval.json",obj); atomic(OUT/"audit/phase76a_decision.json",obj); atomic(OUT/"completion/exact_eval.done",{"phase":"Phase76A","decision":decision,"metrics":str(OUT/"metrics/exact_retrieval.json")}); print(json.dumps({"phase":"Phase76A","decision":decision,"aggregate_p16":aggregate,"checks":checks},sort_keys=True))


if __name__ == "__main__": main()
