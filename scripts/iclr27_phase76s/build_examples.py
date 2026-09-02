#!/usr/bin/env python3
"""Materialise TRAIN/validation counterfactual router examples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase76ar.data import load_stream_payload
from src.iclr27_phase76ar.pair_cache import load_pair_cache
from src.iclr27_phase76ar.relation_model import SelectiveAnchoredRelation
from src.iclr27_phase76ar.runtime import BankFeatureLRU, score_bank
from src.iclr27_phase75d.protocol import load_frozen_tracks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76s"


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


def example_rows(model, banks, table, cache, split: str, fold: int) -> list[dict]:
    lru = BankFeatureLRU(table, cache, torch.device("cpu"), capacity=8); rows: list[dict] = []
    with torch.no_grad():
        for index, bank in enumerate(banks):
            feat = lru.get(index, bank)
            positives = set(bank.positives if hasattr(bank, "positives") else bank.positive_keys); negatives = set(bank.negatives if hasattr(bank, "negatives") else bank.negative_keys)
            for prefix in (1, 2, 4, 8, 16):
                raw = torch.stack([x["raw"] for x in feat[prefix]])
                scored = score_bank(model, feat, prefix, raw_scores=raw)
                raw_np = raw.numpy(); rel_np = scored["final"].numpy();
                order = np.argsort(rel_np)[::-1]; raw_order = np.argsort(raw_np)[::-1]
                raw_top = int(raw_order[0]); rel_top = int(order[0]); pos_idx = [i for i,k in enumerate(bank.candidates) if k in positives]
                raw_ok = raw_top in pos_idx; rel_ok = rel_top in pos_idx
                label = 0 if (rel_ok and not raw_ok) else (1 if (raw_ok and not rel_ok) else 2)
                ctx = SelectiveAnchoredRelation.bank_context(raw).numpy().tolist()
                rel_gap = float(rel_np[order[0]] - rel_np[order[1]]) if len(order) > 1 else 0.0
                x = ctx + [rel_gap, float(scored["gate"].mean()), float(scored["gate"].max()), float(np.abs(scored["delta_bounded"].numpy()).mean()), float(rel_np[order[0]]), float(raw_np[raw_order[0]])]
                rows.append({"fold":fold,"split":split,"stream":"legal_fit","episode_id":bank.episode_id,"query_key":bank.query_key,"prefix":prefix,"features":[float(v) for v in x],"label":label,"raw_correct":bool(raw_ok),"relation_correct":bool(rel_ok),"raw_scores":[float(v) for v in raw_np.tolist()],"relation_scores":[float(v) for v in rel_np.tolist()],"candidates":list(bank.candidates),"positives":list(positives),"negatives":list(negatives),"category":int(bank.category),"video":int(bank.video),"bank_gate":float(scored["bank_gate"].mean())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=None); args = ap.parse_args()
    table = load_frozen_tracks(); folds = range(4) if args.fold is None else [int(args.fold)]; summaries=[]
    for fold in folds:
        model = load_model(ROOT / "outputs/iclr27_phase76ar/checkpoints" / f"ar1_formal_f{fold}_best.pt"); cache = load_pair_cache(ROOT / "outputs/iclr27_phase76ar/banks" / f"pair_cache_f{fold}.json"); streams = ROOT / "outputs/iclr27_phase76ar/banks" / f"streams_f{fold}.json"; fit_memory, fit_legal = load_stream_payload(streams, "fit"); val_memory, val_legal = load_stream_payload(streams, "val")
        fit_rows = example_rows(model, fit_legal, table, cache, "fit", fold); val_rows = example_rows(model, val_legal, table, cache, "val", fold); payload={"phase":"Phase76S","fold":fold,"source_checkpoint":str((ROOT/"outputs/iclr27_phase76ar/checkpoints"/f"ar1_formal_f{fold}_best.pt").resolve()),"source_checkpoint_sha256":hashlib.sha256((ROOT/"outputs/iclr27_phase76ar/checkpoints"/f"ar1_formal_f{fold}_best.pt").read_bytes()).hexdigest(),"fit":fit_rows,"val":val_rows,"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}; path=OUT/"examples"/f"examples_f{fold}.json"; atomic(path,payload); summaries.append({"fold":fold,"fit_examples":len(fit_rows),"val_examples":len(val_rows),"fit_label_counts":[sum(int(r["label"]==i) for r in fit_rows) for i in range(3)],"val_label_counts":[sum(int(r["label"]==i) for r in val_rows) for i in range(3)],"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    atomic(OUT/"audit/example_summary.json",{"phase":"Phase76S","folds":summaries,"classes":["HELP","HARM","NEUTRAL"],"source":"frozen Phase76AR relation outputs; TRAIN labels only","sealed_accessed":False}); atomic(OUT/"completion/build_examples.done",{"phase":"Phase76S","folds":[x["fold"] for x in summaries]}); print(json.dumps({"phase":"Phase76S","folds":summaries},sort_keys=True))


if __name__ == "__main__": main()
