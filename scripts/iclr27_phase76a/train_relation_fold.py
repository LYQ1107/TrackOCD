#!/usr/bin/env python3
"""Train one Phase76A anchored local relation reranker fold."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase75d.protocol import load_frozen_tracks
from src.iclr27_phase76a.candidate_bank import CandidateBank, banks_hash, load_banks
from src.iclr27_phase76a.evaluator import evaluate_banks
from src.iclr27_phase76a.losses import bank_loss
from src.iclr27_phase76a.pair_cache import cache_hash, load_pair_cache
from src.iclr27_phase76a.relation_model import AnchoredRelationReranker
from src.iclr27_phase76a.runtime import BankFeatureLRU, deterministic_order, score_bank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76a"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase76a/checkpoints")
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(value, h, indent=2, sort_keys=True, allow_nan=False); h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as h: os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target.resolve())


def fixed_raw(scores: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([x["raw"] for x in scores])


def one_prefix_loss(model, features: dict[int, list[dict[str, torch.Tensor]]], prefix: int, bank: CandidateBank, task_scale: float, safe_scale: float):
    outputs = []
    for item in features[prefix]: outputs.append(model(item["pair_tokens"], item["summary"], item["raw"]))
    raw = fixed_raw(features[prefix])
    pos_idx = [bank.candidates.index(k) for k in bank.positives]
    neg_idx = [bank.candidates.index(k) for k in bank.negatives]
    loss, parts = bank_loss([{"final": torch.stack([o["final"].reshape(()) for o in outputs]), "delta": torch.stack([o["delta"].reshape(()) for o in outputs]), "confidence": torch.stack([o["confidence"].reshape(()) for o in outputs])}], raw, pos_idx, neg_idx, task_scale, safe_scale)
    return loss, parts, outputs


def calibrate(model, banks: list[CandidateBank], table, pair_cache, device, order: list[int], count: int = 128) -> tuple[float, float]:
    model.eval(); task: list[float] = []; safe: list[float] = []
    lru = BankFeatureLRU(table, pair_cache, device, capacity=4)
    with torch.no_grad():
        for n in range(min(count, len(order))):
            b = banks[order[n]]; feat = lru.get(order[n], b)
            for p in PREFIXES:
                _, parts, _ = one_prefix_loss(model, feat, p, b, 1.0, 1.0)
                task.append(abs(parts["task"])); safe.append(abs(parts["safe"]))
    return max(float(np.mean(task)) if task else 1.0, 1e-3), max(float(np.mean(safe)) if safe else 1.0, 1e-3)


def compact_validation(result: dict[str, Any]) -> dict[str, Any]:
    p16 = next(x for x in result["prefix_rows"] if x["prefix"] == 16)
    m = p16["learned"]
    return {"r1": m["r1"], "map": m["map"], "hard_negative_gap": m["hard_negative_gap"], "raw_r1": m["raw_r1"], "raw_map": m["raw_map"], "raw_hard_negative_gap": m["raw_hard_negative_gap"], "delta_r1": m["r1"] - m["raw_r1"], "delta_map": m["map"] - m["raw_map"], "delta_hard_gap": m["hard_negative_gap"] - m["raw_hard_negative_gap"], "unsafe_flip_count": m["unsafe_flip_count"], "queries": m["queries"], "prefix_rows": result["prefix_rows"]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--tag", default="phase76a_formal"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--targeted", action="store_true"); ap.add_argument("--validation-limit", type=int, default=128); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); steps = 100 if args.smoke else (500 if args.targeted else int(args.steps))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, visible={visible}")
    torch.set_num_threads(1); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 760500 + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    table = load_frozen_tracks(); fit = load_banks(OUT / "banks" / f"fit_f{args.fold}.json"); val = load_banks(OUT / "banks" / f"val_f{args.fold}.json")
    fit_cache = load_pair_cache(OUT / "banks" / f"pair_cache_fit_f{args.fold}.json"); val_cache = load_pair_cache(OUT / "banks" / f"pair_cache_val_f{args.fold}.json")
    order = deterministic_order(fit, seed); val_order = deterministic_order(val, seed + 17)
    run = f"{args.tag}_{'smoke_' if args.smoke else ('targeted_' if args.targeted else '')}f{args.fold}"; comp = OUT / "completion"; marker = comp / f"{run}.launched"; done = comp / f"{run}.done"; failed = comp / f"{run}.failed"; metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists(): raise RuntimeError(f"already complete {run}")
    if marker.exists() and not args.resume: raise RuntimeError(f"launched marker exists {run}; use a new tag or --resume")
    atomic_json(marker, {"phase":"Phase76A","run":run,"fold":args.fold,"pid":os.getpid(),"gpu":args.expected_physical_gpu,"seed":seed,"steps":steps,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    model = AnchoredRelationReranker().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); warmup = 1000
    def lr_lambda(step: int) -> float:
        if step < warmup: return max(step, 1) / warmup
        t = min(max(step - warmup, 0) / max(steps - warmup, 1), 1.0); return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start = 0
    if args.resume:
        latest = OUT / "checkpoints" / f"{run}_latest.pt"
        if latest.exists():
            try: ck = torch.load(latest, map_location=device, weights_only=False)
            except TypeError: ck = torch.load(latest, map_location=device)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"]); start = int(ck.get("step", 0))
    fit_lru = BankFeatureLRU(table, fit_cache, device, capacity=8); val_lru = BankFeatureLRU(table, val_cache, device, capacity=8)
    task_scale, safe_scale = calibrate(model, fit, table, fit_cache, device, order, 128)
    history: list[dict[str, Any]] = []; val_history: list[dict[str, Any]] = []; visits = [0] * len(fit); best_key = None; best_step = 0; best_path = None
    def save(step: int, val_summary: dict[str, Any] | None) -> Path:
        target = CHECKPOINT_ROOT / f"{run}_step{step:05d}.pt"
        payload = {"phase":"Phase76A","model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step,"fold":args.fold,"seed":seed,"run":run,"task_scale":task_scale,"safe_scale":safe_scale,"lr":float(optimizer.param_groups[0]["lr"]),"visits":visits,"validation":val_summary,"protocol":"raw_anchor_plus_local_relation_only","input_hashes":{"csv":table.csv_sha256,"features":table.feature_sha256,"fit_banks":banks_hash(fit),"val_banks":banks_hash(val),"fit_pair_cache":cache_hash(OUT/"banks"/f"pair_cache_fit_f{args.fold}.json"),"val_pair_cache":cache_hash(OUT/"banks"/f"pair_cache_val_f{args.fold}.json")},"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}
        atomic_torch(target, payload); link(OUT/"checkpoints"/target.name, target); latest = CHECKPOINT_ROOT / f"{run}_latest.pt"; latest.unlink(missing_ok=True); latest.symlink_to(target.resolve()); link(OUT/"checkpoints"/latest.name, latest); return target
    try:
        for step in range(start + 1, steps + 1):
            model.train(); idx = order[(step - 1) % len(order)]; visits[idx] += 1; bank = fit[idx]; feat = fit_lru.get(idx, bank); losses=[]; parts_acc={"task":0.0,"safe":0.0,"residual":0.0,"total":0.0}
            for p in PREFIXES:
                loss, parts, _ = one_prefix_loss(model, feat, p, bank, task_scale, safe_scale); losses.append(loss)
                for k in parts_acc:
                    if k in parts: parts_acc[k] += float(parts[k])
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss step {step}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu()); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == steps: history.append({"step":step,"mode":"memory_mimic_fit" if step % 2 == 1 else "legal","loss":float(loss.detach().cpu()),"grad_norm_preclip":grad,"lr":float(optimizer.param_groups[0]["lr"]),"task":parts_acc["task"]/5.0,"safe":parts_acc["safe"]/5.0,"residual":parts_acc["residual"]/5.0,"visit_min":min(visits),"visit_max":max(visits),"visit_imbalance":max(visits)-min(visits)})
            if step % 500 == 0 or step == steps:
                model.eval(); val_obj = evaluate_banks(model, val, table, val_cache, device, limit=args.validation_limit); summary = compact_validation(val_obj); val_history.append({"step":step,"lr":float(optimizer.param_groups[0]["lr"]),"validation":summary,"selection_scope":"bounded_phase30_val_candidate_bank","memory_mimic":summary})
                cp = save(step, summary); safe = summary["unsafe_flip_count"] == 0; key = (0 if safe else 1, -summary["map"], -summary["hard_negative_gap"], -summary["r1"])
                if best_key is None or key < best_key:
                    best_key = key; best_step = step; best_path = cp; bt = CHECKPOINT_ROOT / f"{run}_best.pt"; bt.unlink(missing_ok=True); bt.symlink_to(cp.resolve()); link(OUT/"checkpoints"/bt.name, bt)
        final = {"phase":"Phase76A","fold":args.fold,"run":run,"steps":steps,"seed":seed,"fit_banks":len(fit),"val_banks":len(val),"history":history,"validation_history":val_history,"best_step":best_step,"best_checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt"),"latest_checkpoint":str(OUT/"checkpoints"/f"{run}_latest.pt"),"task_scale":task_scale,"safe_scale":safe_scale,"config":{"architecture":"pair token 1536-256-LN-GELU-128; quality 5-32-1 sigmoid; 13-summary delta/conf heads","optimizer":"AdamW","lr_initial":1e-4,"lr_final":1e-5,"warmup_steps":1000,"steps":steps,"clip":1.0,"checkpoint_every":500,"validation_every":500,"candidate_bank_max":15,"prefixes":PREFIXES},"input_hashes":{"csv":table.csv_sha256,"features":table.feature_sha256,"fit_banks":banks_hash(fit),"val_banks":banks_hash(val),"fit_pair_cache":cache_hash(OUT/"banks"/f"pair_cache_fit_f{args.fold}.json"),"val_pair_cache":cache_hash(OUT/"banks"/f"pair_cache_val_f{args.fold}.json")},"gpu":args.expected_physical_gpu,"device":str(device),"held_event_accessed_for_model":False,"sealed_accessed":False,"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}
        atomic_json(metrics_path, final); atomic_json(done, {"phase":"Phase76A","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt")}); print(json.dumps({"phase":"Phase76A","fold":args.fold,"run":run,"steps":steps,"best_step":best_step},sort_keys=True))
    except Exception as exc:
        atomic_json(failed, {"phase":"Phase76A","fold":args.fold,"run":run,"error":repr(exc),"latest_checkpoint":str(OUT/"checkpoints"/f"{run}_latest.pt")}); raise


if __name__ == "__main__": main()

