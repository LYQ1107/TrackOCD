#!/usr/bin/env python3
"""Train one Phase87 C0 fold with the shared causal rollout core."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# The rollout is intentionally single-item and GPU-bound; cap the CPU thread
# pool so one fold cannot exhaust the shared host while four folds run.
torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase87.controller import CausalPersistentOCD
from src.iclr27_phase87.data import FeatureStore
from src.iclr27_phase87.rollout import rollout_event


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def teacher_probability(step: int) -> float:
    if step <= 2000:
        return 0.80 - 0.30 * (step / 2000.0)
    if step <= 8000:
        return 0.50 * (1.0 - (step - 2000.0) / 6000.0)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=87000)
    args = parser.parse_args()
    tag = args.tag or f"c0_formal_f{args.fold}"
    marker = OUT / "completion" / f"{tag}.launched"
    done = OUT / "completion" / f"{tag}.done"
    metrics_path = OUT / "metrics" / f"{tag}.json"
    checkpoint = CHECKPOINT_ROOT / f"{tag}.pt"
    if done.exists():
        print(metrics_path.read_text())
        return
    if marker.exists():
        raise RuntimeError(f"unit already launched without completion: {marker}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    train_path = OUT / "manifests" / f"train_events_f{args.fold}.jsonl"
    val_path = OUT / "manifests" / f"val_events_f{args.fold}.jsonl"
    train_events = read_events(train_path)
    val_events = read_events(val_path)
    if not train_events:
        raise RuntimeError(f"no causal TRAIN events for fold {args.fold}")
    manifest_sha = sha(OUT / "manifests" / "causal_event_manifest.json")
    atom(marker, {"phase": 87, "route": "C0_CAUSAL_PERSISTENT_CONTROLLER", "fold": args.fold, "device": str(device), "pid": os.getpid(), "updates": args.updates, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "manifest_sha256": manifest_sha})
    store = FeatureStore()
    model = CausalPersistentOCD(max_states=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = random.Random(args.seed + args.fold)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    losses: list[float] = []
    component_sums: dict[str, float] = {}
    grad_norms: list[float] = []
    started = dt.datetime.now(dt.timezone.utc)
    model.train()
    for step in range(1, args.updates + 1):
        event = train_events[rng.randrange(len(train_events))]
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with context:
            loss, trace = rollout_event(model, store, event, train=True, teacher_probability=teacher_probability(step), rng=rng)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(grad_norm)
        for name, value in trace["losses"].items():
            component_sums[name] = component_sums.get(name, 0.0) + float(value)
        if step % 2000 == 0 or step == args.updates:
            save_checkpoint(checkpoint.with_name(f"{tag}_step{step:06d}.pt"), {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "fold": args.fold, "seed": args.seed + args.fold, "route": "C0_CAUSAL_PERSISTENT_CONTROLLER", "manifest_sha256": manifest_sha, "precision": "bf16-autocast" if use_bf16 else "fp32"})
    save_checkpoint(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.updates, "fold": args.fold, "seed": args.seed + args.fold, "route": "C0_CAUSAL_PERSISTENT_CONTROLLER", "manifest_sha256": manifest_sha, "precision": "bf16-autocast" if use_bf16 else "fp32"})
    finished = dt.datetime.now(dt.timezone.utc)
    result = {"schema_version": "trackocd.phase87.c0_train.v1", "phase": 87, "route": "C0_CAUSAL_PERSISTENT_CONTROLLER", "tag": tag, "fold": args.fold, "device": str(device), "updates": args.updates, "train_events": len(train_events), "validation_events": len(val_events), "seed": args.seed + args.fold, "loss_first": losses[0], "loss_last": losses[-1], "loss_mean_last_100": float(np.mean(losses[-100:])), "loss_components_mean": {name: value / len(losses) for name, value in component_sums.items()}, "grad_norm_mean": float(np.mean(grad_norms)), "precision": "bf16-autocast" if use_bf16 else "fp32", "started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint), "manifest_sha256": manifest_sha, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_or_text_as_model_input": False}
    atom(metrics_path, result)
    atom(done, {"status": "DONE", "metrics": str(metrics_path.resolve()), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint)})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
