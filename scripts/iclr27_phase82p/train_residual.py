#!/usr/bin/env python3
"""Train one Phase82P residual fold with resumable checkpoints."""
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

# Allow direct invocation from the repository root or a scheduler shell.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.iclr27_phase82p.residual import ResidualTrajectoryEncoder, residual_loss, contract_summary

OUT = ROOT / "outputs/iclr27_phase82p"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def evaluate(model: torch.nn.Module, arrays: dict[str, np.ndarray], device: torch.device, batch_size: int = 64) -> dict[str, float]:
    model.eval()
    totals = {"count": 0, "correct": 0, "keep_target": 0, "keep_pred": 0, "false_reconnect": 0, "correct_reconnect": 0, "target_reconnect": 0, "pred_reconnect": 0, "loss_sum": 0.0}
    with torch.no_grad():
        for start in range(0, len(arrays["target"]), batch_size):
            sl = slice(start, min(len(arrays["target"]), start + batch_size))
            cur = torch.from_numpy(arrays["current"][sl]).to(device)
            hist = torch.from_numpy(arrays["history"][sl]).to(device)
            mask = torch.from_numpy(arrays["candidate_mask"][sl]).to(device)
            target = torch.from_numpy(arrays["target"][sl]).to(device)
            logits = model(cur, hist, mask)
            loss, _ = residual_loss(logits, target, mask)
            pred = logits.argmax(dim=1)
            count = int(target.numel())
            totals["count"] += count
            totals["correct"] += int((pred == target).sum())
            totals["keep_target"] += int((target == 0).sum())
            totals["keep_pred"] += int((pred == 0).sum())
            totals["false_reconnect"] += int(((target == 0) & (pred > 0)).sum())
            totals["correct_reconnect"] += int(((target > 0) & (pred == target)).sum())
            totals["target_reconnect"] += int((target > 0).sum())
            totals["pred_reconnect"] += int((pred > 0).sum())
            totals["loss_sum"] += float(loss.cpu()) * count
    denom = max(1, totals["count"])
    return {
        "examples": totals["count"], "loss": totals["loss_sum"] / denom,
        "accuracy": totals["correct"] / denom, "keep_target_rate": totals["keep_target"] / denom,
        "keep_pred_rate": totals["keep_pred"] / denom, "false_reconnect_rate": totals["false_reconnect"] / denom,
        "correct_reconnect_rate": totals["correct_reconnect"] / denom, "target_reconnect_rate": totals["target_reconnect"] / denom,
        "pred_reconnect_rate": totals["pred_reconnect"] / denom,
        "repair_precision": totals["correct_reconnect"] / max(1, totals["pred_reconnect"]),
        "repair_recall": totals["correct_reconnect"] / max(1, totals["target_reconnect"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data-root", type=Path, default=Path("/data2/usr_for_deadline/trackocd_phase82p/data"))
    ap.add_argument("--tag", default="residual")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--checkpoint-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=8201)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fit_path = args.data_root / f"fold{args.fold}.npz"
    val_path = args.data_root / f"fold{args.fold}_val.npz"
    if not fit_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"missing fold data {fit_path} or {val_path}")
    fit_z = np.load(fit_path, allow_pickle=False)
    val_z = np.load(val_path, allow_pickle=False)
    fit = {k: fit_z[k] for k in ("current", "history", "candidate_mask", "target")}
    val = {k: val_z[k] for k in ("current", "history", "candidate_mask", "target")}
    if len(fit["target"]) == 0:
        raise RuntimeError("empty fit shard")
    model = ResidualTrajectoryEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
    ckpt_dir = OUT / "checkpoints" / args.tag / f"fold{args.fold}"
    metrics_dir = OUT / "metrics" / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    order = np.arange(len(fit["target"]), dtype=np.int64)
    rng = np.random.default_rng(args.seed + args.fold)
    train_history: list[dict[str, Any]] = []
    steps_per_epoch = max(1, len(order) // args.batch_size)
    for step in range(start_step + 1, args.steps + 1):
        if (step - 1) % steps_per_epoch == 0:
            rng.shuffle(order)
        begin = ((step - 1) * args.batch_size) % len(order)
        idx = order[begin:begin + args.batch_size]
        if len(idx) < args.batch_size:
            idx = np.concatenate((idx, order[: args.batch_size - len(idx)]))
        cur = torch.from_numpy(fit["current"][idx]).to(device)
        hist = torch.from_numpy(fit["history"][idx]).to(device)
        mask = torch.from_numpy(fit["candidate_mask"][idx]).to(device)
        target = torch.from_numpy(fit["target"][idx]).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(cur, hist, mask)
        loss, train_metrics = residual_loss(logits, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % max(1, args.checkpoint_interval) == 0 or step == args.steps:
            val_metrics = evaluate(model, val, device)
            record = {"step": step, "fold": args.fold, "train": train_metrics, "val": val_metrics, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            train_history.append(record)
            checkpoint = {"schema_version": "trackocd.phase82p.residual_checkpoint.v1", "step": step, "fold": args.fold, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "seed": args.seed, "contract": contract_summary(model), "val_metrics": val_metrics}
            atomic_torch(ckpt_dir / f"step{step:06d}.pt", checkpoint)
            atomic_torch(ckpt_dir / "latest.pt", checkpoint)
            atomic_json(metrics_dir / f"fold{args.fold}_step{step:06d}.json", record)
    final = {"schema_version": "trackocd.phase82p.residual_training_metrics.v1", "fold": args.fold, "tag": args.tag, "steps": args.steps, "fit_examples": len(fit["target"]), "val_examples": len(val["target"]), "history": train_history, "final_val": evaluate(model, val, device), "checkpoint": str(ckpt_dir / "latest.pt"), "device": str(device), "seed": args.seed, "public_dev_q1_sealed_accessed": False}
    atomic_json(metrics_dir / f"fold{args.fold}_final.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
