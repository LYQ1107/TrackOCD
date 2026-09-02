#!/usr/bin/env python3
"""Train one registered Phase75E rank-8 feature adapter.

The worker is intentionally self-contained: it reads only a Phase30 *fit*
manifest, writes atomic markers, and performs a bounded TRAIN-disjoint screen
at each checkpoint.  No held-event, DEV+, Q1, or sealed artifact is opened.
"""
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

from src.iclr27_phase75e.data import episode_feature_cache, frozen_table, load_fit_episodes, manifest_hash
from src.iclr27_phase75e.evaluator import evaluate_fold
from src.iclr27_phase75e.losses import episode_loss
from src.iclr27_phase75e.model import LowRankFeatureAdapter
from src.iclr27_phase75e.pairwise_adapter import adapter_drift


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75e"
CHECKPOINT_ROOT = Path("/data2/usr_for_deadline/trackocd_phase75e/checkpoints")
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.path.realpath(link) == str(target.resolve()):
            return
        if link.is_file() or link.is_symlink():
            link.unlink()
        else:
            raise RuntimeError(f"refusing to replace non-file checkpoint path {link}")
    link.symlink_to(target.resolve())


def atomic_marker(path: Path, value: dict[str, Any]) -> None:
    atomic_json(path, value)


def source_hashes(table) -> dict[str, Any]:
    return {
        "csv": table.csv_sha256,
        "features": table.feature_sha256,
        "feature_alignment_permutation": table.alignment.get("permutation_sha256"),
    }


def validation_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract only the TRAIN-disjoint checkpoint-selection fields."""
    p16 = next(x for x in result["prefix_rows"] if x["prefix"] == 16)
    g, l = p16["global"], p16["legal"]
    gm, lm = g["learned"], l["learned"]
    return {
        "global": {k: gm.get(k) for k in ("r1", "map", "hard_negative_gap", "unsafe_flip_count", "queries")},
        "legal": {k: lm.get(k) for k in ("r1", "map", "hard_negative_gap", "unsafe_flip_count", "queries")},
        "selection_key": [
            int(lm.get("unsafe_flip_count", 0)),
            -float(lm.get("map", 0.0)),
            -float(lm.get("hard_negative_gap", 0.0)),
            -float(gm.get("map", 0.0)),
        ],
        "global_scope": g["scope"],
        "legal_scope": l["scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--tag", default="phase75e_formal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-physical-gpu", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--global-validation-limit", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible:
        first = visible.split(",")[0].strip()
        if first != str(args.expected_physical_gpu):
            raise RuntimeError(f"GPU mapping mismatch: expected physical {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 750500 + int(args.fold)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    table = frozen_table()
    episodes = load_fit_episodes(args.fold, set(table.sequences))
    seq_cache = episode_feature_cache(table, episodes)
    steps = 100 if args.smoke else int(args.steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"
    completion = OUT / "completion"
    marker = completion / f"{run}.launched"
    done = completion / f"{run}.done"
    failed = completion / f"{run}.failed"
    metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists():
        raise RuntimeError(f"completion already exists for {run}; refusing duplicate launch")
    if marker.exists() and not args.resume:
        raise RuntimeError(f"launched marker already exists for {run}; use a fresh tag or explicit --resume")

    atomic_marker(marker, {
        "phase": "Phase75E", "run": run, "fold": args.fold, "pid": os.getpid(),
        "gpu": args.expected_physical_gpu, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": seed, "steps": steps, "protocol": "phase30_fit_only_rank8_adapter",
    })

    model = LowRankFeatureAdapter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    rng = np.random.default_rng(seed + 17)
    start_step = 0
    if args.resume:
        latest_link = OUT / "checkpoints" / f"{run}_latest.pt"
        if latest_link.exists():
            checkpoint = torch.load(latest_link, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint.get("step", 0))

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_step = start_step
    best_checkpoint: Path | None = None

    def save_checkpoint(step: int, val: dict[str, Any] | None) -> Path:
        target = CHECKPOINT_ROOT / f"{run}_step{step:05d}.pt"
        payload = {
            "phase": "Phase75E", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "seed": seed, "fold": args.fold, "run": run,
            "protocol": "rank8_alpha16_raw_preserving_pairwise_adapter",
            "config": {"rank": 8, "alpha": 16.0, "scale": 2.0, "lr": 4e-5, "warmup_steps": 2500, "gradient_clip": 0.05, "loss": "0.5*rank+1.0*raw_reconstruction+1.0*safe"},
            "input_hashes": source_hashes(table), "manifest_sha256": manifest_hash(args.fold),
            "validation_summary": val,
        }
        atomic_torch(target, payload)
        ensure_link(OUT / "checkpoints" / target.name, target)
        latest_target = CHECKPOINT_ROOT / f"{run}_latest.pt"
        if latest_target.is_symlink() or latest_target.exists():
            latest_target.unlink()
        latest_target.symlink_to(target.resolve())
        ensure_link(OUT / "checkpoints" / latest_target.name, latest_target)
        return target

    try:
        for step in range(start_step + 1, steps + 1):
            model.train()
            episode = episodes[int(rng.integers(len(episodes)))]
            q_by_prefix = {p: torch.as_tensor(seq_cache[episode.query_key][p], device=device) for p in PREFIXES}
            positives = [{p: torch.as_tensor(seq_cache[key][p], device=device) for p in PREFIXES} for key in episode.positive_keys]
            negative = {p: torch.as_tensor(seq_cache[episode.negative_key][p], device=device) for p in PREFIXES}
            loss, parts = episode_loss(model, q_by_prefix, positives, negative)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05).detach().cpu())
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == steps:
                # A compact causal drift probe over this update's tensors.
                probe = adapter_drift(model(q_by_prefix[16]), q_by_prefix[16])
                history.append({"step": step, **parts, "grad_norm_preclip": grad_norm, **probe, "episode_id": episode.episode_id})

            due = (step % args.validation_every == 0) or step == steps
            if due:
                model.eval()
                with torch.no_grad():
                    result = evaluate_fold(model, table, args.fold, device, global_query_limit=args.global_validation_limit, legal_query_limit=None)
                summary = validation_summary(result)
                drift = adapter_drift(model(q_by_prefix[16]), q_by_prefix[16])
                val_obj = {"phase": "Phase75E", "fold": args.fold, "run": run, "step": step, "validation": result, "selection": summary, "drift_probe": drift, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "held_event_accessed_for_model": False, "sealed_accessed": False}
                atomic_json(OUT / "validation" / args.tag / f"step_{step:05d}_metrics.json", val_obj)
                atomic_marker(OUT / "validation" / args.tag / f"step_{step:05d}.done", {"step": step, "metrics": str(OUT / "validation" / args.tag / f"step_{step:05d}_metrics.json")})
                validation_history.append({"step": step, **summary, "drift": drift})
                candidate_key = tuple(float(x) for x in summary["selection_key"])
                checkpoint_path = save_checkpoint(step, summary)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key; best_step = step; best_checkpoint = checkpoint_path
                    best_target = CHECKPOINT_ROOT / f"{run}_best.pt"
                    if best_target.is_symlink() or best_target.exists():
                        best_target.unlink()
                    best_target.symlink_to(checkpoint_path.resolve())
                    ensure_link(OUT / "checkpoints" / best_target.name, best_target)

        model.eval()
        final_obj = {
            "phase": "Phase75E", "fold": args.fold, "run": run, "steps": steps, "seed": seed,
            "episodes_fit": len(episodes), "history": history, "validation_history": validation_history,
            "best_step": best_step, "checkpoint_best": str(OUT / "checkpoints" / f"{run}_best.pt"),
            "checkpoint_latest": str(OUT / "checkpoints" / f"{run}_latest.pt"),
            "config": {"rank": 8, "alpha": 16.0, "scale": 2.0, "optimizer": "Adam", "lr": 4e-5, "warmup_steps": 2500, "gradient_clip": 0.05, "checkpoint_every": args.checkpoint_every, "validation_every": args.validation_every, "loss": "0.5*L_rank + 1.0*L_raw_reconstruction + 1.0*L_safe", "assignment": "detached CPU Hungarian indices; selected torch similarities retain gradient"},
            "input_hashes": source_hashes(table), "manifest_sha256": manifest_hash(args.fold),
            "gpu": args.expected_physical_gpu, "device": str(device), "amp": "fp32", "held_event_accessed_for_model": False, "sealed_accessed": False,
            "sealed_inputs_not_read": ["DEV+", "Q1", "public-new", "sealed", "152 held events", "category/text/physical IDs as tensors", "future frames"],
        }
        atomic_json(metrics_path, final_obj)
        atomic_marker(done, {"phase": "Phase75E", "fold": args.fold, "run": run, "steps": steps, "best_step": best_step, "checkpoint": str(OUT / "checkpoints" / f"{run}_best.pt")})
        print(json.dumps({"phase": "Phase75E", "fold": args.fold, "run": run, "steps": steps, "best_step": best_step, "done": str(done)}, sort_keys=True), flush=True)
    except Exception as exc:
        atomic_marker(failed, {"phase": "Phase75E", "fold": args.fold, "run": run, "error": repr(exc), "latest_checkpoint": str(OUT / "checkpoints" / f"{run}_latest.pt")})
        raise


if __name__ == "__main__":
    main()
