#!/usr/bin/env python3
"""Train one causal-memory-matched Family-B fold."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase80b.data import PREFIXES, frozen_table, load_memory_banks, manifest_hash, materialize_bank, source_hashes
from src.iclr27_phase80b.evaluator import evaluate_banks, p16
from src.iclr27_phase80b.losses import sequence_loss
from src.iclr27_phase80b.model import CausalMemoryScorer


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80b"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase80b/checkpoints")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as f: os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True); path.symlink_to(target.resolve())


def idx_sets(bank) -> tuple[list[int], list[int]]:
    pos, neg = set(bank.positives), set(bank.negatives)
    return ([i for i, k in enumerate(bank.candidates) if k in pos], [i for i, k in enumerate(bank.candidates) if k in neg])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--tag", default="phase80b_formal"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--targeted", action="store_true"); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu):
        raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, visible={visible}")
    torch.set_num_threads(1); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 808000 + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    steps = 100 if args.smoke else (500 if args.targeted else int(args.steps))
    table = frozen_table(); fit = load_memory_banks(args.fold, "fit"); val = load_memory_banks(args.fold, "val")
    run = f"{args.tag}_{'smoke_' if args.smoke else ('targeted_' if args.targeted else '')}f{args.fold}"
    comp = OUT / "completion"; marker = comp / f"{run}.launched"; done = comp / f"{run}.done"; failed = comp / f"{run}.failed"; metrics = OUT / "metrics" / f"{run}.json"
    if done.exists(): raise RuntimeError(f"completion exists: {run}")
    if marker.exists() and not args.resume: raise RuntimeError(f"launched marker exists: {run}; use a fresh tag or --resume")
    atomic_json(marker, {"phase":"Phase80B","run":run,"fold":args.fold,"pid":os.getpid(),"gpu":args.expected_physical_gpu,"seed":seed,"steps":steps,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    model = CausalMemoryScorer().to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); start = 0
    latest = OUT / "checkpoints" / f"{run}_latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = int(ck.get("step", 0))
    rng = np.random.default_rng(seed + 19); history: list[dict[str, Any]] = []; val_history: list[dict[str, Any]] = []; best_key = None; best_step = start

    def save(step: int, summary: dict[str, Any]) -> Path:
        target = CHECKPOINT_ROOT / f"{run}_step{step:05d}.pt"
        atomic_torch(target, {"phase":"Phase80B","model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"fold":args.fold,"seed":seed,"run":run,"validation":summary,"stream":"memory_mimic","manifest_sha256":manifest_hash(args.fold),"input_hashes":source_hashes(table,args.fold),"config":{"hidden":32,"delta_max":0.08,"lr":2e-4,"loss":"listwise+0.35 hard-negative+0.5 prefix persistence+1.5 raw safety+0.02 residual"},"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]})
        link(OUT / "checkpoints" / target.name, target); latest_target = CHECKPOINT_ROOT / f"{run}_latest.pt"; latest_target.unlink(missing_ok=True); latest_target.symlink_to(target.resolve()); link(OUT / "checkpoints" / latest_target.name, latest_target); return target

    try:
        for step in range(start + 1, steps + 1):
            model.train(); bank = fit[int(rng.integers(len(fit)))]; seq = torch.as_tensor(materialize_bank(bank, table), dtype=torch.float32, device=device)
            out = model(seq); pos, neg = idx_sets(bank); loss, parts = sequence_loss(out, pos, neg)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu()); opt.step()
            if step == 1 or step % 100 == 0 or step == steps: history.append({"step":step,"episode_id":bank.episode_id,"loss":float(loss.detach().cpu()),"grad_norm":grad,"parts":parts})
            if step % 500 == 0 or step == steps:
                model.eval(); result = evaluate_banks(model, val, table, device, limit=None); summary = p16(result); val_history.append({"step":step,"p16":summary}); cp = save(step, summary); key = (int(summary.get("unsafe_flip_count",0)), -float(summary.get("map",0.0)), -float(summary.get("hard_negative_gap",0.0)), -float(summary.get("r1",0.0)))
                if best_key is None or key < best_key:
                    best_key = key; best_step = step; best = CHECKPOINT_ROOT / f"{run}_best.pt"; best.unlink(missing_ok=True); best.symlink_to(cp.resolve()); link(OUT / "checkpoints" / best.name, best)
        model.eval(); final = evaluate_banks(model, val, table, device); obj = {"phase":"Phase80B","fold":args.fold,"run":run,"steps":steps,"seed":seed,"fit_banks":len(fit),"val_banks":len(val),"best_step":best_step,"history":history,"validation_history":val_history,"validation":p16(final),"checkpoint_best":str(OUT/"checkpoints"/f"{run}_best.pt"),"checkpoint_latest":str(OUT/"checkpoints"/f"{run}_latest.pt"),"manifest_sha256":manifest_hash(args.fold),"input_hashes":source_hashes(table,args.fold),"protocol":"causal-memory-matched memory-mimic candidate banks, prefixes 1/2/4/8/16","held_event_accessed_for_model":False,"sealed_accessed":False}
        atomic_json(metrics,obj); atomic_json(done,{"phase":"Phase80B","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt")}); print(json.dumps({"phase":"Phase80B","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"p16":p16(final)},sort_keys=True))
    except Exception as exc:
        atomic_json(failed,{"phase":"Phase80B","fold":args.fold,"run":run,"error":repr(exc),"latest_checkpoint":str(latest)}); raise


if __name__ == "__main__": main()

