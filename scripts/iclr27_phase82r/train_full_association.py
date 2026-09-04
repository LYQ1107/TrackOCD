#!/usr/bin/env python3
"""Train one fold of the Phase82R full causal association route."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.iclr27_phase82r.full_association import FullAssociation, assignment_loss, contract_summary


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def as_batch(arrays: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    # Stored observations are float16 for disk efficiency; all model math is
    # explicitly float32 (AMP is intentionally not used for this small head).
    cur = torch.from_numpy(arrays["current"][idx]).to(device=device, dtype=torch.float32)
    hist = torch.from_numpy(arrays["history"][idx]).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(arrays["candidate_mask"][idx]).to(device=device, dtype=torch.bool)
    target = torch.from_numpy(arrays["target"][idx]).to(device=device, dtype=torch.long)
    return cur, hist, mask, target


@torch.no_grad()
def evaluate(model: torch.nn.Module, arrays: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval(); n = len(arrays["target"])
    totals = {"examples": 0, "correct": 0, "target_existing": 0, "pred_existing": 0, "existing_correct": 0, "new_correct": 0, "loss": 0.0, "new_pred": 0}
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(n, start + batch_size))
        cur, hist, mask, target = as_batch(arrays, idx, device)
        out = model(cur, hist, mask); loss, _ = assignment_loss(out, target, mask)
        logits = torch.cat((out["new_logit"].unsqueeze(1), out["candidate_logits"]), dim=1)
        pred = logits.argmax(dim=1); pos = target > 0; pred_pos = pred > 0
        totals["examples"] += int(target.numel()); totals["correct"] += int((pred == target).sum())
        totals["target_existing"] += int(pos.sum()); totals["pred_existing"] += int(pred_pos.sum()); totals["existing_correct"] += int((pos & (pred == target)).sum()); totals["new_correct"] += int(((~pos) & (pred == 0)).sum()); totals["new_pred"] += int((pred == 0).sum()); totals["loss"] += float(loss.cpu()) * int(target.numel())
    d = max(1, totals["examples"]); ep = max(1, totals["pred_existing"]); et = max(1, totals["target_existing"])
    return {"examples": totals["examples"], "loss": totals["loss"] / d, "accuracy": totals["correct"] / d, "target_existing_rate": totals["target_existing"] / d, "pred_existing_rate": totals["pred_existing"] / d, "existing_precision": totals["existing_correct"] / ep, "existing_recall": totals["existing_correct"] / et, "new_accuracy": totals["new_correct"] / max(1, d - totals["target_existing"]), "new_prediction_rate": totals["new_pred"] / d}


def balanced_batches(target: np.ndarray, rng: np.random.Generator, batch_size: int) -> list[np.ndarray]:
    pos = rng.permutation(np.flatnonzero(target > 0)); neg = rng.permutation(np.flatnonzero(target == 0))
    if not len(pos) or not len(neg):
        raise RuntimeError(f"both NEW and existing labels required, pos={len(pos)} neg={len(neg)}")
    half = max(1, batch_size // 2); count = max(len(pos), len(neg))
    batches = []
    for start in range(0, count, half):
        p = pos[start % len(pos): (start % len(pos)) + half]
        if len(p) < half: p = np.concatenate((p, pos[: half - len(p)]))
        q = neg[start % len(neg): (start % len(neg)) + half]
        if len(q) < half: q = np.concatenate((q, neg[: half - len(q)]))
        x = np.concatenate((p, q)); rng.shuffle(x); batches.append(x)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--data-root", type=Path, default=Path("/data2/usr_for_deadline/trackocd_phase82r/full_assoc_data")); ap.add_argument("--tag", default="full_assoc_formal_r1"); ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--max-updates", type=int, default=0); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--checkpoint-interval", type=int, default=500); ap.add_argument("--seed", type=int, default=8261); ap.add_argument("--resume", type=Path)
    args = ap.parse_args(); torch.set_num_threads(4); seed = args.seed + args.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit_z = np.load(args.data_root / f"fold{args.fold}.npz", allow_pickle=False); val_z = np.load(args.data_root / f"fold{args.fold}_val.npz", allow_pickle=False)
    fit = {k: fit_z[k] for k in ("current", "history", "candidate_mask", "target")}; val = {k: val_z[k] for k in ("current", "history", "candidate_mask", "target")}
    model = FullAssociation().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epoch = 0; updates = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); epoch = int(ck.get("epoch", 0)); updates = int(ck.get("updates", 0))
    out_root = ROOT / "outputs/iclr27_phase82r"; ckpt_dir = out_root / "checkpoints" / args.tag / f"fold{args.fold}"; metrics_dir = out_root / "metrics" / args.tag; ckpt_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); history: list[dict[str, Any]] = []; updates_per_epoch = max(1, int(np.ceil(max(len(np.flatnonzero(fit["target"] > 0)), len(np.flatnonzero(fit["target"] == 0))) / max(1, args.batch_size // 2)))); max_updates = args.max_updates if args.max_updates > 0 else args.epochs * updates_per_epoch
    while epoch < args.epochs and updates < max_updates:
        epoch += 1
        for idx in balanced_batches(fit["target"], rng, args.batch_size):
            if updates >= max_updates: break
            cur, hist, mask, target = as_batch(fit, idx, device); model.train(); optimizer.zero_grad(set_to_none=True); out = model(cur, hist, mask); loss, train_m = assignment_loss(out, target, mask); loss.backward(); grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step(); updates += 1
            if updates == 1 or updates % args.checkpoint_interval == 0 or updates >= max_updates:
                val_m = evaluate(model, val, device, args.batch_size); rec = {"fold": args.fold, "epoch": epoch, "updates": updates, "train": train_m, "val": val_m, "grad_norm": grad_norm, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}; history.append(rec)
                ck = {"schema_version": "trackocd.phase82r.full_association_checkpoint.v1", "fold": args.fold, "epoch": epoch, "updates": updates, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed, "contract": contract_summary(model), "val_metrics": val_m, "data_root": str(args.data_root)}; atomic_torch(ckpt_dir / f"step{updates:06d}.pt", ck); atomic_torch(ckpt_dir / "latest.pt", ck); atomic_json(metrics_dir / f"fold{args.fold}_step{updates:06d}.json", rec)
    final_val = evaluate(model, val, device, args.batch_size); final = {"schema_version": "trackocd.phase82r.full_association_training_metrics.v1", "fold": args.fold, "tag": args.tag, "epochs": epoch, "updates": updates, "fit_examples": len(fit["target"]), "val_examples": len(val["target"]), "fit_existing": int((fit["target"] > 0).sum()), "val_existing": int((val["target"] > 0).sum()), "history": history, "final_val": final_val, "checkpoint": str(ckpt_dir / "latest.pt"), "device": str(device), "seed": args.seed, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(metrics_dir / f"fold{args.fold}_final.json", final); print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__": main()
