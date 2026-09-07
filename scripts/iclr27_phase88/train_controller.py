#!/usr/bin/env python3
"""Train one Phase88 fold with the persistent runtime rollout."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import gc
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/iclr27_phase88"
CKPT_ROOT = OUT / "checkpoints"

from src.iclr27_phase88.controller import CausalPersistentOCD
from src.iclr27_phase88.data import FeatureStore
from src.iclr27_phase88.rollout import rollout_event, rollout_known_event
from src.iclr27_phase88.sampler import BalancedCausalEventSampler
from src.iclr27_phase88.resource_manager import process_memory_detail


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def semantic_contract_sha() -> str:
    h = hashlib.sha256()
    for name in ("rollout.py", "memory.py", "transitions.py", "data.py", "data_memmap.py"):
        path = ROOT / "src/iclr27_phase88" / name
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def teacher_probability(step: int) -> float:
    if step <= 2000:
        return 0.80 - 0.30 * (step / 2000.0)
    if step <= 8000:
        return 0.50 * (1.0 - (step - 2000.0) / 6000.0)
    return 0.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def count_jsonl(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def mem_available_kib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=88002)
    ap.add_argument("--support-mode", action="store_true")
    ap.add_argument("--init-checkpoint", default=None)
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--resume-launched", action="store_true")
    ap.add_argument("--checkpoint-interval", type=int, default=2000)
    ap.add_argument("--event-tag", default="v2")
    ap.add_argument("--memmap-root", default=None)
    args = ap.parse_args()
    marker = OUT / "completion" / f"{args.tag}.launched"
    done = OUT / "completion" / f"{args.tag}.done"
    metrics_path = OUT / "metrics" / f"{args.tag}.json"
    checkpoint = CKPT_ROOT / f"{args.tag}.pt"
    if done.exists():
        print(metrics_path.read_text())
        return
    if marker.exists() and not (args.resume_checkpoint and args.resume_launched):
        raise RuntimeError(f"unit already launched without completion: {marker}")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = int(args.seed) + int(args.fold)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.memmap_root:
        from src.iclr27_phase88.data_memmap import Phase88FoldData
        data = Phase88FoldData(args.fold, args.memmap_root)
    else:
        from src.iclr27_phase19r.data.stream import Phase19RData
        data = Phase19RData(args.fold)
    store = FeatureStore(args.fold, data)
    train_path = OUT / "manifests" / f"fit_events_v2_{args.event_tag}_f{args.fold}.jsonl"
    val_path = OUT / "manifests" / f"val_events_v2_{args.event_tag}_f{args.fold}.jsonl"
    train_events = load_jsonl(train_path)
    validation_event_count = count_jsonl(val_path)
    if not train_events:
        raise RuntimeError(f"no fit events for fold {args.fold}")
    manifest_sha = sha(OUT / "manifests" / f"causal_event_v2_{args.event_tag}_manifest.json")
    atomic_json(marker, {
        "status": "LAUNCHED", "phase": 88, "route": args.tag, "fold": args.fold,
        "device": str(device), "pid": os.getpid(), "updates": args.updates,
        "seed": seed, "support_mode": bool(args.support_mode),
        "manifest_sha256": manifest_sha, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    memory_profile_path = OUT / "audit" / f"memory_profile_{args.tag}.jsonl"
    memory_profile_path.parent.mkdir(parents=True, exist_ok=True)

    def record_memory(stage: str) -> None:
        detail = process_memory_detail(os.getpid())
        gpu_allocated = 0
        gpu_reserved = 0
        if device.type == "cuda":
            gpu_allocated = int(torch.cuda.memory_allocated(device))
            gpu_reserved = int(torch.cuda.memory_reserved(device))
        item = {
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": stage,
            "pid": os.getpid(),
            "VmRSS": int(detail.get("Rss", 0)),
            "PSS": int(detail.get("Pss", 0)),
            "Anonymous": int(detail.get("Anonymous", 0)),
            "Private_Dirty": int(detail.get("Private_Dirty", 0)),
            "system_MemAvailable": mem_available_kib(),
            "gpu_allocated": gpu_allocated,
            "gpu_reserved": gpu_reserved,
        }
        with memory_profile_path.open("a") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
        rss_samples.append({"stage": stage, "rss_bytes": int(detail.get("Rss", 0) * 1024),
                            "pss_bytes": int(detail.get("Pss", 0) * 1024),
                            "anonymous_bytes": int(detail.get("Anonymous", 0) * 1024),
                            "mem_available_kib": item["system_MemAvailable"]})

    rss_samples: list[dict] = []
    record_memory("after_data_open")
    model = CausalPersistentOCD(torch.from_numpy(np.asarray(data.known_prototypes)), torch.from_numpy(np.asarray(data.active_known_mask)), max_states=16).to(device)
    record_memory("after_model_create")
    resume_payload = None
    if args.resume_checkpoint:
        resume_payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_payload["model"], strict=False)
    elif args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(payload["model"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start_step = 0
    if resume_payload is not None:
        start_step = int(resume_payload.get("step", 0))
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
    sampler = BalancedCausalEventSampler(train_events, seed=seed)
    known_keys = list(data.known_eval_keys) if hasattr(data, "known_eval_keys") else list(__import__("src.iclr27_phase19r.evaluation.internal", fromlist=["fixed_known_keys"]).fixed_known_keys(data))
    known_mask = torch.from_numpy(np.asarray(data.active_known_mask, dtype=bool)).to(device)
    rng = random.Random(seed)
    if resume_payload is not None:
        if resume_payload.get("sampler_state"):
            sampler.load_state_dict(resume_payload["sampler_state"])
        if resume_payload.get("rollout_rng_state"):
            rng.setstate(resume_payload["rollout_rng_state"])
        if resume_payload.get("python_rng_state"):
            random.setstate(resume_payload["python_rng_state"])
        if resume_payload.get("numpy_rng_state") is not None:
            np.random.set_state(resume_payload["numpy_rng_state"])
        if resume_payload.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_payload["torch_rng_state"])
    losses: list[float] = []
    component_sums: dict[str, float] = {}
    grad_norms: list[float] = []
    reset_targets = 0; reset_predictions = 0; known_steps = 0
    reset_targets_source = 0; reset_targets_target = 0; masked_target_violations = 0
    target_action_counts = {k: 0 for k in ("EXISTING", "NEW", "DEFER", "RESET")}
    source_action_counts = {k: 0 for k in ("EXISTING", "NEW", "DEFER", "RESET")}
    reset_reason_counts: dict[str, int] = {}
    started = dt.datetime.now(dt.timezone.utc)
    pause_requested = False

    def request_pause(_signum, _frame) -> None:
        nonlocal pause_requested
        pause_requested = True

    signal.signal(signal.SIGUSR1, request_pause)

    def checkpoint_payload(step: int) -> dict:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "fold": args.fold, "seed": seed, "route": args.tag,
            "support_mode": bool(args.support_mode), "event_tag": args.event_tag,
            "manifest_sha256": manifest_sha,
            "semantic_contract_sha256": semantic_contract_sha(),
            "known_prototype_hash": hashlib.sha256(model.known_prototypes.detach().cpu().numpy().tobytes()).hexdigest(),
            "precision": "bf16-autocast" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32",
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(), "sampler_state": sampler.state_dict(),
            "rollout_rng_state": rng.getstate(),
        }

    model.train()
    for step in range(start_step + 1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        use_known = bool(known_keys) and rng.random() < 0.25
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)):
            if use_known:
                key, cat = rng.choice(known_keys)
                loss, trace = rollout_known_event(model, store, key, int(cat), known_mask, data.known_to_index.get(int(cat)))
                known_steps += 1
            else:
                event = sampler.sample()
                loss, trace = rollout_event(model, store, event, train=True,
                                            teacher_probability=teacher_probability(step), rng=rng,
                                            support_mode=args.support_mode)
                reset_targets += int(trace.get("reset_targets", 0)); reset_predictions += int(trace.get("reset_predictions", 0))
                reset_targets_source += int(trace.get("reset_targets_source", 0))
                reset_targets_target += int(trace.get("reset_targets_target", 0))
                masked_target_violations += int(trace.get("masked_target_violations", 0))
                for key, value in trace.get("target_action_counts", {}).items():
                    target_action_counts[key] = target_action_counts.get(key, 0) + int(value)
                for key, value in trace.get("source_target_counts", {}).items():
                    source_action_counts[key] = source_action_counts.get(key, 0) + int(value)
                for key, value in trace.get("reset_reason_counts", {}).items():
                    reset_reason_counts[key] = reset_reason_counts.get(key, 0) + int(value)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0).detach().cpu())
        optimizer.step()
        losses.append(float(loss.detach().cpu())); grad_norms.append(grad_norm)
        if step in {100, 500, 1000, 2000} or step % int(args.checkpoint_interval) == 0:
            record_memory(f"step_{step}")
        for name, value in trace.get("losses", {}).items():
            component_sums[name] = component_sums.get(name, 0.0) + float(value)
        if step % int(args.checkpoint_interval) == 0 or step == args.updates or pause_requested:
            save_checkpoint(CKPT_ROOT / f"{args.tag}_step{step:06d}.pt", checkpoint_payload(step))
            gc.collect(); malloc_trim()
        if pause_requested:
            pause_path = OUT / "completion" / f"{args.tag}.paused_resource.json"
            atomic_json(pause_path, {"status": "PAUSED_RESOURCE", "step": step,
                                     "checkpoint": str((CKPT_ROOT / f"{args.tag}_step{step:06d}.pt").resolve()),
                                     "manifest_sha256": manifest_sha,
                                     "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            return
    final_payload = checkpoint_payload(args.updates)
    save_checkpoint(checkpoint, final_payload)
    gc.collect(); malloc_trim()
    finished = dt.datetime.now(dt.timezone.utc)
    result = {
        "schema_version": "trackocd.phase88.train.v1", "phase": 88, "route": args.tag,
        "fold": args.fold, "device": str(device), "updates": args.updates, "start_step": start_step, "seed": seed,
        "support_mode": bool(args.support_mode), "event_tag": args.event_tag, "train_events": len(train_events),
        "validation_events": validation_event_count, "sampler": sampler.stats(), "known_steps": known_steps,
        "reset_targets": reset_targets, "reset_predictions": reset_predictions,
        "reset_targets_source": reset_targets_source, "reset_targets_target": reset_targets_target,
        "target_action_counts": target_action_counts, "source_action_counts": source_action_counts,
        "reset_reason_counts": reset_reason_counts, "masked_target_violations": masked_target_violations,
        "loss_first": losses[0], "loss_last": losses[-1], "loss_mean_last_100": float(np.mean(losses[-100:])),
        "loss_components_mean": {k: v / len(losses) for k, v in component_sums.items()},
        "grad_norm_mean": float(np.mean(grad_norms)), "precision": final_payload["precision"],
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
        "manifest_sha256": manifest_sha, "known_prototype_hash": final_payload["known_prototype_hash"],
        "semantic_contract_sha256": final_payload["semantic_contract_sha256"],
        "sampler_state_persisted": True, "rollout_rng_state_persisted": True,
        "resumed_from": args.resume_checkpoint,
        "rss_samples": rss_samples,
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
    }
    atomic_json(metrics_path, result)
    atomic_json(done, {"status": "DONE", "metrics": str(metrics_path.resolve()), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint)})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
