#!/usr/bin/env python3
"""Audit the causal-memory supervision distribution before Family-B training."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.iclr27_phase80b.data import PREFIXES, frozen_table, load_memory_banks, materialize_bank, manifest_hash


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80b/audit"


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def stats(banks, table) -> dict:
    counts=[]; pos=[]; neg=[]; margins={str(p):[] for p in PREFIXES}; raw_correct={str(p):0 for p in PREFIXES}; total={str(p):0 for p in PREFIXES}; support_nonempty=0
    for bank in banks:
        counts.append(len(bank.candidates)); pos.append(len(bank.positives)); neg.append(len(bank.negatives)); support_nonempty += int(bool(bank.positives))
        seq=materialize_bank(bank, table); pi=[i for i,k in enumerate(bank.candidates) if k in set(bank.positives)]; ni=[i for i,k in enumerate(bank.candidates) if k in set(bank.negatives)]
        if not pi or not ni: continue
        for ti,p in enumerate(PREFIXES):
            r=seq[ti]; margins[str(p)].append(float(np.max(r[pi])-np.max(r[ni]))); raw_correct[str(p)] += int(int(np.argmax(r)) in pi); total[str(p)] += 1
    def summary(vals):
        return {"count":len(vals),"mean":float(np.mean(vals)) if vals else 0.0,"median":float(np.median(vals)) if vals else 0.0,"p10":float(np.percentile(vals,10)) if vals else 0.0,"p90":float(np.percentile(vals,90)) if vals else 0.0}
    return {"banks":len(banks),"candidate_count":{"min":min(counts) if counts else 0,"mean":float(np.mean(counts)) if counts else 0.0,"max":max(counts) if counts else 0},"positive_count":{"min":min(pos) if pos else 0,"mean":float(np.mean(pos)) if pos else 0.0,"max":max(pos) if pos else 0},"negative_count":{"min":min(neg) if neg else 0,"mean":float(np.mean(neg)) if neg else 0.0,"max":max(neg) if neg else 0},"support_nonempty_rate":float(support_nonempty/max(len(banks),1)),"raw_margin":{p:summary(v) for p,v in margins.items()},"raw_top1_positive_rate":{p:float(raw_correct[p]/max(total[p],1)) for p in margins},"evaluable_by_prefix":total}


def main() -> None:
    table=frozen_table(); folds=[]
    for fold in range(4):
        fit=load_memory_banks(fold,"fit"); val=load_memory_banks(fold,"val")
        folds.append({"fold":fold,"stream_hash":manifest_hash(fold),"fit":stats(fit,table),"val":stats(val,table)})
    payload={"phase":"Phase80B","stage":"family_b_contract_audit","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"folds":folds,"hypothesis":"Phase30 isolated episodes do not expose the same sequential candidate/evidence distribution as a persistent memory; a stateful scorer trained on prefix-union memory-mimic banks may improve persistence without changing visual source.","model_input_fields":["causal raw cosine","prefix delta","candidate rank","bank entropy","learned evidence state","causal age"],"supervision_metadata_only":["category","track_key","video","positive/negative membership"],"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"],"protocol":"Phase30 TRAIN-disjoint memory-mimic banks, prefixes 1/2/4/8/16; raw fallback exact at initialization","source_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=False).stdout.strip(),"held_dev_q1_public_new_accessed":False,"sealed_accessed":False}
    atomic(OUT/"family_b_contract_audit.json",payload); atomic(OUT/"family_b_preregistration.json",payload); print(json.dumps({"phase":"Phase80B","folds":len(folds),"audit":str(OUT/"family_b_contract_audit.json")},sort_keys=True))


if __name__ == "__main__": main()

