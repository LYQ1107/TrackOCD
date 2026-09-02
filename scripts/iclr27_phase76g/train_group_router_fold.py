#!/usr/bin/env python3
"""Train one Phase76G router with rotating category-group holdouts."""
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
import torch.nn.functional as F

from src.iclr27_phase76g.router import GroupRobustRelationRouter
from src.iclr27_phase76s.evaluator import evaluate_examples, p16

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase76g"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase76g/checkpoints")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try: torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists(): path.unlink()
    path.symlink_to(target.resolve())


def weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    count = np.bincount([int(row["label"]) for row in rows], minlength=3).astype(np.float32)
    total = float(count.sum()); return torch.tensor(np.where(count > 0, total / np.maximum(3.0 * count, 1.0), 0.0), dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--tag", default="g1_formal"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--targeted", action="store_true"); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    steps = 100 if args.smoke else (500 if args.targeted else int(args.steps)); visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"GPU mapping mismatch: expected {args.expected_physical_gpu}, visible={visible}")
    torch.set_num_threads(1); device = torch.device(args.device if torch.cuda.is_available() else "cpu"); seed = 767900 + args.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    example_payload = json.loads((ROOT / "outputs/iclr27_phase76s/examples" / f"examples_f{args.fold}.json").read_text()); manifest = json.loads((OUT / "manifests" / f"meta_manifest_f{args.fold}.json").read_text())
    rows = example_payload["fit"]; val_rows = example_payload["val"]; x = torch.tensor(np.asarray([row["features"] for row in rows], dtype=np.float32), device=device); y = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long, device=device); w = weights(rows).to(device)
    group_of = torch.full((len(rows),), -1, dtype=torch.long, device=device)
    for group, indices in manifest["fit_indices_by_group"].items():
        if indices: group_of[torch.tensor(indices, dtype=torch.long, device=device)] = int(group)
    run = f"{args.tag}_{'smoke_' if args.smoke else ('targeted_' if args.targeted else '')}f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"; failed = OUT / "completion" / f"{run}.failed"; metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists(): raise RuntimeError(f"already complete {run}")
    if marker.exists() and not args.resume: raise RuntimeError(f"launched marker exists {run}; use new tag or --resume")
    atomic_json(marker, {"phase": "Phase76G", "run": run, "fold": args.fold, "pid": os.getpid(), "gpu": args.expected_physical_gpu, "seed": seed, "steps": steps, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    model = GroupRobustRelationRouter().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4); start = 0
    if args.resume:
        latest = OUT / "checkpoints" / f"{run}_latest.pt"
        if latest.exists():
            try: ck = torch.load(latest, map_location=device, weights_only=False)
            except TypeError: ck = torch.load(latest, map_location=device)
            model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); start = int(ck.get("step", 0))
    history: list[dict[str, Any]] = []; val_history: list[dict[str, Any]] = []; best_key = None; best_step = 0
    def save(step: int, summary: dict[str, Any]) -> Path:
        target = CHECKPOINT_ROOT / f"{run}_step{step:05d}.pt"; atomic_torch(target, {"phase": "Phase76G", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "fold": args.fold, "seed": seed, "validation": summary, "group_manifest": str((OUT / "manifests" / f"meta_manifest_f{args.fold}.json").resolve()), "protocol": "rotating 3-of-4 category-group meta-holdout; exact raw fallback"}); link(OUT / "checkpoints" / target.name, target); latest = CHECKPOINT_ROOT / f"{run}_latest.pt"; latest.unlink(missing_ok=True); latest.symlink_to(target.resolve()); link(OUT / "checkpoints" / latest.name, latest); return target
    try:
        for step in range(start + 1, steps + 1):
            hold = (step - 1) % 4; model.train(); logits = model(x); losses = []
            for group in range(4):
                if group == hold: continue
                idx = group_of == group
                if bool(idx.any()): losses.append(F.cross_entropy(logits[idx], y[idx], weight=w))
            if not losses: raise RuntimeError("empty rotating meta-train groups")
            loss = torch.stack(losses).mean() + 0.5 * torch.stack(losses).max(); optimizer.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu()); optimizer.step()
            if step == 1 or step % 100 == 0 or step == steps: history.append({"step": step, "loss": float(loss.detach().cpu()), "grad_norm": grad, "holdout_group": hold, "group_losses": [float(v.detach().cpu()) for v in losses]})
            if step % 500 == 0 or step == steps:
                val = evaluate_examples(model, val_rows, device); summary = p16(val); val_history.append({"step": step, "validation": summary}); cp = save(step, summary); key = (summary["unsafe_flip_count"], -summary["map"], -summary["hard_negative_gap"], -summary["r1"])
                if best_key is None or key < best_key: best_key = key; best_step = step; best = CHECKPOINT_ROOT / f"{run}_best.pt"; best.unlink(missing_ok=True); best.symlink_to(cp.resolve()); link(OUT / "checkpoints" / best.name, best)
        final = evaluate_examples(model, val_rows, device); summary = p16(final); payload = {"phase": "Phase76G", "fold": args.fold, "run": run, "steps": steps, "seed": seed, "fit_examples": len(rows), "val_examples": len(val_rows), "fit_label_counts": np.bincount([int(r["label"]) for r in rows], minlength=3).tolist(), "val_label_counts": np.bincount([int(r["label"]) for r in val_rows], minlength=3).tolist(), "history": history, "validation_history": val_history, "best_step": best_step, "best_checkpoint": str((OUT / "checkpoints" / f"{run}_best.pt")), "latest_checkpoint": str((OUT / "checkpoints" / f"{run}_latest.pt")), "validation": summary, "group_counts": {key: len(value) for key, value in manifest["fit_indices_by_group"].items()}, "config": {"architecture": "14->32 LN GELU->3", "objective": "mean group CE + 0.5 max group CE over rotating 3 groups", "steps": steps, "checkpoint_every": 500, "validation_every": 500}, "gpu": args.expected_physical_gpu, "device": str(device), "forbidden_inference_inputs": ["category", "semantic_id", "physical_id", "text", "future", "held/DEV+/Q1/public-new/sealed labels"]}; atomic_json(metrics_path, payload); atomic_json(done, {"phase": "Phase76G", "fold": args.fold, "run": run, "steps": steps, "best_step": best_step, "checkpoint": str(OUT / "checkpoints" / f"{run}_best.pt")}); print(json.dumps({"phase": "Phase76G", "fold": args.fold, "run": run, "steps": steps, "best_step": best_step, "p16": summary}, sort_keys=True))
    except Exception as exc:
        atomic_json(failed, {"phase": "Phase76G", "fold": args.fold, "run": run, "error": repr(exc)}); raise


if __name__ == "__main__": main()
