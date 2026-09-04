#!/usr/bin/env python3
"""Train one Phase82R balanced two-stage residual fold."""
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
from src.iclr27_phase82r.balanced_residual import BalancedResidualGate, balanced_loss, contract_summary, predict


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


def eval_model(model: torch.nn.Module, arrays: dict[str, np.ndarray], device: torch.device, batch_size: int = 128) -> dict[str, float]:
    model.eval()
    n = len(arrays["target"])
    totals = {"examples": 0, "correct": 0, "target_reconnect": 0, "pred_reconnect": 0, "false_reconnect": 0, "correct_reconnect": 0, "rank_correct": 0, "rank_total": 0, "gate_pos_correct": 0, "gate_total": 0, "gate_tp": 0, "gate_fp": 0, "gate_fn": 0, "loss_sum": 0.0, "gate_prob_sum": 0.0}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(n, start + batch_size))
            cur = torch.from_numpy(arrays["current"][sl]).to(device)
            hist = torch.from_numpy(arrays["history"][sl]).to(device)
            mask = torch.from_numpy(arrays["candidate_mask"][sl]).to(device)
            target = torch.from_numpy(arrays["target"][sl]).to(device)
            out = model(cur, hist, mask)
            loss, _ = balanced_loss(out, target, mask)
            chosen, prob = predict(out)
            # no candidate may be selected when the causal set is empty
            chosen = torch.where(mask.any(dim=1), chosen, torch.zeros_like(chosen))
            pos = target > 0
            pred_pos = chosen > 0
            totals["examples"] += int(target.numel())
            totals["correct"] += int((chosen == target).sum())
            totals["target_reconnect"] += int(pos.sum())
            totals["pred_reconnect"] += int(pred_pos.sum())
            totals["false_reconnect"] += int(((~pos) & pred_pos).sum())
            totals["correct_reconnect"] += int((pos & (chosen == target)).sum())
            totals["gate_total"] += int(target.numel())
            gate_pos = prob >= 0.5
            totals["gate_pos_correct"] += int((gate_pos == pos).sum())
            totals["gate_tp"] += int((gate_pos & pos).sum())
            totals["gate_fp"] += int((gate_pos & ~pos).sum())
            totals["gate_fn"] += int((~gate_pos & pos).sum())
            totals["gate_prob_sum"] += float(prob.sum().cpu())
            if pos.any():
                rank_pred = out["candidate_logits"][pos].argmax(dim=1) + 1
                totals["rank_correct"] += int((rank_pred == target[pos]).sum())
                totals["rank_total"] += int(pos.sum())
            totals["loss_sum"] += float(loss.cpu()) * int(target.numel())
    d = max(1, totals["examples"])
    return {
        "examples": totals["examples"], "loss": totals["loss_sum"] / d,
        "accuracy": totals["correct"] / d, "target_reconnect_rate": totals["target_reconnect"] / d,
        "pred_reconnect_rate": totals["pred_reconnect"] / d, "gate_use_rate": totals["pred_reconnect"] / d,
        "false_reconnect_rate": totals["false_reconnect"] / d,
        "repair_precision": totals["correct_reconnect"] / max(1, totals["pred_reconnect"]),
        "repair_recall": totals["correct_reconnect"] / max(1, totals["target_reconnect"]),
        "candidate_rank_recall": totals["rank_correct"] / max(1, totals["rank_total"]),
        "gate_accuracy": totals["gate_pos_correct"] / max(1, totals["gate_total"]),
        "gate_precision": totals["gate_tp"] / max(1, totals["gate_tp"] + totals["gate_fp"]),
        "gate_recall": totals["gate_tp"] / max(1, totals["gate_tp"] + totals["gate_fn"]),
        "mean_gate_probability": totals["gate_prob_sum"] / d,
    }


def balanced_batches(fit: dict[str, np.ndarray], rng: np.random.Generator, batch_size: int) -> list[np.ndarray]:
    target = fit["target"]
    pos = np.flatnonzero(target > 0)
    neg = np.flatnonzero(target == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError(f"balanced route requires both classes, pos={len(pos)} neg={len(neg)}")
    pos = rng.permutation(pos)
    neg = rng.permutation(neg)
    half = max(1, batch_size // 2)
    batches: list[np.ndarray] = []
    for start in range(0, len(pos), half):
        p = pos[start : start + half]
        if len(p) < half:
            p = np.concatenate((p, pos[: half - len(p)]))
        n = neg[start : start + half]
        if len(n) < half:
            n = np.concatenate((n, neg[: half - len(n)]))
        x = np.concatenate((p, n)); rng.shuffle(x); batches.append(x)
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", type=Path, default=Path("/data2/usr_for_deadline/trackocd_phase82r/data"))
    ap.add_argument("--tag", default="balanced_formal")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-updates", type=int, default=0, help="bounded smoke/targeted override; 0 means epochs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=8201)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    seed = args.seed + args.fold
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit_z = np.load(args.data_root / f"fold{args.fold}.npz", allow_pickle=False)
    val_z = np.load(args.data_root / f"fold{args.fold}_val.npz", allow_pickle=False)
    fit = {k: fit_z[k] for k in ("current", "history", "candidate_mask", "target")}
    val = {k: val_z[k] for k in ("current", "history", "candidate_mask", "target")}
    model = BalancedResidualGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_epoch = 0; updates = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0)); updates = int(ck.get("updates", 0))
    out_root = ROOT / "outputs/iclr27_phase82r"
    ckpt_dir = out_root / "checkpoints" / args.tag / f"fold{args.fold}"
    metrics_dir = out_root / "metrics" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True); metrics_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    max_updates = args.max_updates if args.max_updates > 0 else args.epochs * max(1, int(np.ceil(2 * len(np.flatnonzero(fit["target"] > 0)) / args.batch_size)))
    epoch = start_epoch
    while epoch < args.epochs and updates < max_updates:
        epoch += 1
        batches = balanced_batches(fit, rng, args.batch_size)
        for idx in batches:
            if updates >= max_updates:
                break
            cur = torch.from_numpy(fit["current"][idx]).to(device)
            hist = torch.from_numpy(fit["history"][idx]).to(device)
            mask = torch.from_numpy(fit["candidate_mask"][idx]).to(device)
            target = torch.from_numpy(fit["target"][idx]).to(device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            out = model(cur, hist, mask)
            loss, train_metrics = balanced_loss(out, target, mask)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            updates += 1
            if updates == 1 or updates % args.checkpoint_interval == 0 or updates >= max_updates:
                val_metrics = eval_model(model, val, device)
                rec = {"epoch": epoch, "updates": updates, "fold": args.fold, "train": train_metrics, "val": val_metrics, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                history.append(rec)
                ck = {"schema_version": "trackocd.phase82r.balanced_residual_checkpoint.v1", "epoch": epoch, "updates": updates, "fold": args.fold, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed, "contract": contract_summary(model), "val_metrics": val_metrics}
                atomic_torch(ckpt_dir / f"step{updates:06d}.pt", ck); atomic_torch(ckpt_dir / "latest.pt", ck)
                atomic_json(metrics_dir / f"fold{args.fold}_step{updates:06d}.json", rec)
        if updates >= max_updates:
            break
    final_val = eval_model(model, val, device)
    final = {"schema_version": "trackocd.phase82r.balanced_residual_training_metrics.v1", "fold": args.fold, "tag": args.tag, "epochs": epoch, "updates": updates, "fit_examples": len(fit["target"]), "val_examples": len(val["target"]), "positive_fit": int((fit["target"] > 0).sum()), "history": history, "final_val": final_val, "checkpoint": str(ckpt_dir / "latest.pt"), "device": str(device), "seed": args.seed, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(metrics_dir / f"fold{args.fold}_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
