#!/usr/bin/env python3
"""Exact TRAIN-disjoint replay of Family-B checkpoints."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase80b.data import frozen_table, load_memory_banks
from src.iclr27_phase80b.evaluator import evaluate_banks, p16
from src.iclr27_phase80b.model import CausalMemoryScorer

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase80b"


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_model(path: Path) -> CausalMemoryScorer:
    ck = torch.load(path, map_location="cpu", weights_only=False); m=CausalMemoryScorer(); m.load_state_dict(ck["model"]); m.eval(); return m


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--tag",default="phase80b_formal"); ap.add_argument("--run-id",default="phase80b-exact"); args=ap.parse_args()
    table=frozen_table(); folds=[]
    for fold in range(4):
        path=OUT/"checkpoints"/f"{args.tag}_f{fold}_best.pt"; model=load_model(path); val=load_memory_banks(fold,"val"); result=evaluate_banks(model,val,table,torch.device("cpu")); folds.append({"fold":fold,"checkpoint":str(path.resolve()),"p16":p16(result),"result":result})
    def mean(k): return float(np.mean([x["p16"].get(k,0.0) for x in folds]))
    aggregate={"r1":mean("r1"),"map":mean("map"),"hard_negative_gap":mean("hard_negative_gap"),"raw_r1":mean("raw_r1"),"raw_map":mean("raw_map"),"raw_hard_negative_gap":mean("raw_hard_negative_gap"),"delta_r1":mean("delta_r1"),"delta_map":mean("delta_map"),"delta_hard_gap":mean("delta_hard_gap"),"unsafe_flip_count":sum(int(x["p16"].get("unsafe_flip_count",0)) for x in folds),"queries":sum(int(x["p16"].get("queries",0)) for x in folds)}
    fold_deltas=[{"fold":x["fold"],"delta_r1":x["p16"].get("delta_r1"),"delta_map":x["p16"].get("delta_map"),"delta_hard_gap":x["p16"].get("delta_hard_gap"),"unsafe":x["p16"].get("unsafe_flip_count"),"intervention_rate":x["p16"].get("intervention_rate")} for x in folds]
    gate={"p16_delta_r1_ge_0.02":aggregate["delta_r1"]>=0.02,"p16_delta_map_ge_0.01":aggregate["delta_map"]>=0.01,"substantial_r1_3_of_4":sum(float(x["delta_r1"] or 0)>=0.02 for x in fold_deltas)>=3,"substantial_map_3_of_4":sum(float(x["delta_map"] or 0)>=0.01 for x in fold_deltas)>=3,"unsafe_zero":aggregate["unsafe_flip_count"]==0,"hard_gap_non_worse_all_folds":all(float(x["delta_hard_gap"] or 0)>=-1e-7 for x in fold_deltas)}
    decision="PHASE80B_GATE_R_PASS" if all(gate.values()) else "PHASE80B_GATE_R_FAIL_ROUTE_FAMILY_C_OR_D"
    payload={"phase":"Phase80B","run_id":args.run_id,"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"protocol":"TRAIN-disjoint memory-mimic validation, all prefixes","folds":folds,"aggregate":aggregate,"fold_deltas":fold_deltas,"gate":gate,"decision":decision,"held_event_accessed_for_model":False,"sealed_accessed":False,"public_or_dev_accessed":False}
    atomic(OUT/"metrics/exact_memory_replay.json",payload); atomic(OUT/"audit/phase80b_decision.json",payload); atomic(OUT/"completion/exact_memory_replay.done",{"phase":"Phase80B","decision":decision,"metrics":str(OUT/"metrics/exact_memory_replay.json")}); print(json.dumps({"phase":"Phase80B","decision":decision,"aggregate":aggregate,"gate":gate},sort_keys=True))


if __name__ == "__main__": main()

