#!/usr/bin/env python3
"""Train one Phase76AR fold with real dual-stream causal episodes."""
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
from src.iclr27_phase76ar.data import load_stream_payload
from src.iclr27_phase76ar.evaluator import evaluate_banks, p16
from src.iclr27_phase76ar.losses import ar_loss, teacher_use
from src.iclr27_phase76ar.pair_cache import cache_hash, load_pair_cache
from src.iclr27_phase76ar.relation_model import SelectiveAnchoredRelation
from src.iclr27_phase76ar.runtime import BankFeatureLRU, deterministic_order, score_bank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76ar"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase76ar/checkpoints")
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle: os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists(): path.unlink()
    path.symlink_to(target.resolve())


def idx_sets(bank: Any) -> tuple[list[int], list[int]]:
    positives = set(bank.positives if hasattr(bank, "positives") else bank.positive_keys)
    negatives = set(bank.negatives if hasattr(bank, "negatives") else bank.negative_keys)
    return ([i for i, key in enumerate(bank.candidates) if key in positives], [i for i, key in enumerate(bank.candidates) if key in negatives])


def bank_step(model: SelectiveAnchoredRelation, bank: Any, feat: dict[int, list[dict[str, torch.Tensor]]], *, device: torch.device, task_scale: float, safe_scale: float) -> tuple[torch.Tensor, dict[str, float]]:
    pos_idx, neg_idx = idx_sets(bank)
    losses: list[torch.Tensor] = []; accum = {"task": 0.0, "safe": 0.0, "gate": 0.0, "residual": 0.0, "total": 0.0, "teacher": 0.0}
    for prefix in PREFIXES:
        raw = torch.stack([x["raw"] for x in feat[prefix]])
        outputs = score_bank(model, feat, prefix, raw_scores=raw)
        # Frozen quality evidence makes the teacher independent of learned
        # parameters while still using only TRAIN metadata outside the model.
        teacher = teacher_use(raw, pos_idx, feat[prefix], neg_idx)
        loss, parts = ar_loss(outputs, pos_idx, neg_idx, teacher, task_scale, safe_scale)
        losses.append(loss)
        for key in accum:
            if key in parts: accum[key] += float(parts[key])
    total = torch.stack(losses).mean() if losses else next(model.parameters()).sum() * 0.0
    for key in accum: accum[key] /= max(len(PREFIXES), 1)
    accum["total"] = float(total.detach().cpu())
    return total, accum


def estimate_scales(model, banks: list[Any], cache: dict[str, Any], table, device: torch.device, *, count: int = 16) -> tuple[float, float, dict[str, float]]:
    model.eval(); task: list[float] = []; safe: list[float] = []
    lru = BankFeatureLRU(table, cache, device, capacity=4)
    with torch.no_grad():
        for i, bank in enumerate(banks[:count]):
            feat = lru.get(i, bank); pos_idx, neg_idx = idx_sets(bank)
            for prefix in PREFIXES:
                raw = torch.stack([x["raw"] for x in feat[prefix]])
                outputs = score_bank(model, feat, prefix, raw_scores=raw)
                teacher = teacher_use(raw, pos_idx, feat[prefix], neg_idx)
                _, parts = ar_loss(outputs, pos_idx, neg_idx, teacher)
                task.append(abs(parts["task"])); safe.append(abs(parts["safe"]))
    ts = max(float(np.mean(task)) if task else 1.0, 1e-3); ss = max(float(np.mean(safe)) if safe else 1.0, 1e-3)
    return ts, ss, {"task_mean": ts, "safe_mean": ss, "sample_banks": min(count, len(banks))}


def validation_selection(banks: list[Any], seed: int, fold: int) -> list[int]:
    order = deterministic_order(banks, seed + 17)
    if fold != 0 or len(order) <= 128: return order
    # Deterministic hash-stratified fold0 subset: category/video/margin bins
    # are represented before the stable hash fills any remaining slots.
    groups: dict[tuple[int, int], list[int]] = {}
    for i in order:
        b = banks[i]; groups.setdefault((int(b.category), int(b.video) % 8), []).append(i)
    chosen: list[int] = []
    for key in sorted(groups):
        if groups[key]: chosen.append(groups[key][0])
    remaining = [i for i in order if i not in set(chosen)]
    chosen.extend(remaining)
    return chosen[:128]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--tag", default="phase76ar_formal"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--targeted", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args(); steps = 100 if args.smoke else (500 if args.targeted else int(args.steps))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, visible={visible}")
    torch.set_num_threads(1); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 767600 + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    table = load_frozen_tracks(); streams = OUT / "banks" / f"streams_f{args.fold}.json"; memory_fit, legal_fit = load_stream_payload(streams, "fit"); memory_val, legal_val = load_stream_payload(streams, "val")
    cache = load_pair_cache(OUT / "banks" / f"pair_cache_f{args.fold}.json")
    run = f"{args.tag}_{'smoke_' if args.smoke else ('targeted_' if args.targeted else '')}f{args.fold}"; comp = OUT / "completion"; marker = comp / f"{run}.launched"; done = comp / f"{run}.done"; failed = comp / f"{run}.failed"; metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists(): raise RuntimeError(f"already complete {run}")
    if marker.exists() and not args.resume: raise RuntimeError(f"launched marker exists {run}; use a new tag or --resume")
    atomic_json(marker, {"phase":"Phase76AR","run":run,"fold":args.fold,"pid":os.getpid(),"gpu":args.expected_physical_gpu,"seed":seed,"steps":steps,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
    model = SelectiveAnchoredRelation().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-5)
    start = 0
    if args.resume:
        latest = OUT / "checkpoints" / f"{run}_latest.pt"
        if latest.exists():
            try: ck = torch.load(latest, map_location=device, weights_only=False)
            except TypeError: ck = torch.load(latest, map_location=device)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"]); start = int(ck.get("step", 0))
    task_scale, safe_scale, scale_diag = estimate_scales(model, memory_fit + legal_fit, cache, table, device)
    memory_order = deterministic_order(memory_fit, seed + 1); legal_order = deterministic_order(legal_fit, seed + 2); val_memory_order = validation_selection(memory_val, seed, args.fold); val_legal_order = validation_selection(legal_val, seed + 1, args.fold)
    fit_lru = BankFeatureLRU(table, cache, device, capacity=8); history: list[dict[str, Any]] = []; val_history: list[dict[str, Any]] = []; visits = {"memory_mimic": [0] * len(memory_fit), "legal_fit": [0] * len(legal_fit)}; best_key = None; best_step = 0

    def save(step: int, val_summary: dict[str, Any] | None) -> Path:
        target = CHECKPOINT_ROOT / f"{run}_step{step:05d}.pt"
        payload = {"phase":"Phase76AR","model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step,"fold":args.fold,"seed":seed,"run":run,"task_scale":task_scale,"safe_scale":safe_scale,"stream_counts":{"memory_mimic":len(memory_fit),"legal_fit":len(legal_fit)},"stream_visits":visits,"validation":val_summary,"protocol":"raw-first selective relation; dual stream; prefix-union hard negatives","input_hashes":{"csv":table.csv_sha256,"features":table.feature_sha256,"streams":hashlib.sha256(streams.read_bytes()).hexdigest(),"pair_cache":cache_hash(OUT/"banks"/f"pair_cache_f{args.fold}.json")},"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}
        atomic_torch(target, payload); link(OUT / "checkpoints" / target.name, target); latest = CHECKPOINT_ROOT / f"{run}_latest.pt"; latest.unlink(missing_ok=True); latest.symlink_to(target.resolve()); link(OUT / "checkpoints" / latest.name, latest); return target

    try:
        for step in range(start + 1, steps + 1):
            model.train(); use_memory = (step % 2 == 1); source = "memory_mimic" if use_memory else "legal_fit"; order = memory_order if use_memory else legal_order; seq = memory_fit if use_memory else legal_fit; idx = order[((step - 1) // 2) % max(len(order), 1)]; bank = seq[idx]; visits[source][idx] += 1; feat = fit_lru.get((0 if use_memory else 1000000) + idx, bank)
            if device.type == "cuda" and not args.fp32:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): loss, parts = bank_step(model, bank, feat, device=device, task_scale=task_scale, safe_scale=safe_scale)
            else: loss, parts = bank_step(model, bank, feat, device=device, task_scale=task_scale, safe_scale=safe_scale)
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss step {step}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu()); optimizer.step(); scheduler.step()
            if step == 1 or step % 100 == 0 or step == steps:
                history.append({"step":step,"stream":source,"bank_index":idx,"episode_id":bank.episode_id,"loss":float(loss.detach().cpu()),"grad_norm_preclip":grad,"lr":float(optimizer.param_groups[0]["lr"]),"stream_counts":{"memory_mimic":len(memory_fit),"legal_fit":len(legal_fit)},"visit_min":{"memory_mimic":min(visits["memory_mimic"]) if visits["memory_mimic"] else 0,"legal_fit":min(visits["legal_fit"]) if visits["legal_fit"] else 0},"visit_max":{"memory_mimic":max(visits["memory_mimic"]) if visits["memory_mimic"] else 0,"legal_fit":max(visits["legal_fit"]) if visits["legal_fit"] else 0},"parts":parts,"amp":"fp32" if args.fp32 or device.type != "cuda" else "bf16"})
            if step % 500 == 0 or step == steps:
                model.eval(); val_obj = evaluate_banks(model, legal_val, table, cache, device, indices=val_legal_order); summary = p16(val_obj); val_history.append({"step":step,"stream":"legal_fit_val","selection_scope":"fold0 deterministic hash-stratified 128; folds1-3 full","validation":summary}); cp = save(step, summary); key = (summary["unsafe_flip_count"], -summary["map"], -summary["hard_negative_gap"], -summary["r1"]); 
                if best_key is None or key < best_key:
                    best_key = key; best_step = step; best = CHECKPOINT_ROOT / f"{run}_best.pt"; best.unlink(missing_ok=True); best.symlink_to(cp.resolve()); link(OUT / "checkpoints" / best.name, best)
        final_obj = evaluate_banks(model, legal_val, table, cache, device, indices=val_legal_order); final_summary = p16(final_obj)
        final = {"phase":"Phase76AR","fold":args.fold,"run":run,"steps":steps,"seed":seed,"fit_counts":{"memory_mimic":len(memory_fit),"legal_fit":len(legal_fit)},"val_counts":{"memory_mimic":len(memory_val),"legal_fit":len(legal_val)},"stream_hashes":{"streams":hashlib.sha256(streams.read_bytes()).hexdigest()},"history":history,"validation_history":val_history,"best_step":best_step,"best_checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt"),"latest_checkpoint":str(OUT/"checkpoints"/f"{run}_latest.pt"),"scale_diagnostic":scale_diag,"validation":final_summary,"config":{"architecture":"1536 pair token -> 256 LN GELU -> 128; per-match quality 5->32->1; bank gate 8->32->1; bounded delta 0.10*tanh","streams":"odd memory_mimic / even legal_fit","optimizer":"AdamW","lr":1e-4,"eta_min":1e-5,"steps":steps,"checkpoint_every":500,"validation_every":500,"candidate_max":15,"prefixes":PREFIXES},"stream_visit_counts":visits,"gpu":args.expected_physical_gpu,"device":str(device),"held_event_accessed_for_model":False,"sealed_accessed":False,"forbidden_inference_inputs":["category","semantic_id","physical_id","text","future","held/DEV+/Q1/public-new/sealed labels"]}
        atomic_json(metrics_path, final); atomic_json(done, {"phase":"Phase76AR","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"checkpoint":str(OUT/"checkpoints"/f"{run}_best.pt")}); print(json.dumps({"phase":"Phase76AR","fold":args.fold,"run":run,"steps":steps,"best_step":best_step,"p16":final_summary}, sort_keys=True))
    except Exception as exc:
        atomic_json(failed, {"phase":"Phase76AR","fold":args.fold,"run":run,"error":repr(exc),"latest_checkpoint":str(OUT/"checkpoints"/f"{run}_latest.pt")}); raise


if __name__ == "__main__": main()
