"""Train one registered DSTM cross-fit fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.iclr27_phase18.evaluation.dstm_runtime import evaluate_calibration
from src.iclr27_phase18.models.dstm import DSTM, parameter_counts
from src.iclr27_phase18.training.data import FoldData, ROOT


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            if value.dtype == np.float16:
                tensor = tensor.to(device, dtype=torch.float32, non_blocking=True)
            else:
                tensor = tensor.to(device, non_blocking=True)
            result[key] = tensor
        else:
            result[key] = value
    return result


def losses(model: DSTM, batch: dict[str, Any], known_weights: torch.Tensor,
           pos_weight: torch.Tensor, weights: dict[str, float], variant: str) -> tuple[torch.Tensor, dict[str, float]]:
    two_stage_r1 = variant == "repair_r1"
    allow_defer = variant not in {"b3", "repair_r1"}
    out = model(batch["query"], batch["lengths"], batch["states"], batch["state_mask"],
                batch["known_mask"], allow_defer=allow_defer, return_all=True)
    if two_stage_r1:
        identity_mask = batch["reliability"] > .5
        action = (F.cross_entropy(out["logits"][identity_mask], batch["labels"][identity_mask])
                  if identity_mask.any() else out["logits"].sum() * 0.0)
    else:
        action = F.cross_entropy(out["logits"], batch["labels"])
    known_mask = batch["known_aux"] >= 0
    known = (F.cross_entropy(out["known_aux_logits"][known_mask], batch["known_aux"][known_mask], weight=known_weights)
             if known_mask.any() else out["logits"].sum() * 0.0)
    reliability = F.binary_cross_entropy_with_logits(
        out["reliability_logit"], batch["reliability"], pos_weight=pos_weight
    )
    metric_mask = batch["metric_slot"] >= 0
    if metric_mask.any():
        q = F.normalize(out["query"][metric_mask], dim=-1)
        states = F.normalize(out["state_tokens"][metric_mask], dim=-1)
        slots = batch["metric_slot"][metric_mask]
        pos = states[torch.arange(len(slots), device=slots.device), slots]
        pos_cos = (q * pos).sum(-1)
        sim = torch.einsum("bd,bkd->bk", q, states)
        valid = batch["state_mask"][metric_mask].clone()
        valid[torch.arange(len(slots), device=slots.device), slots] = False
        neg = sim.masked_fill(~valid, -1.0).max(1).values
        metric = ((1.0 - pos_cos) + F.relu(neg - pos_cos + .2)).mean()
    else:
        metric = out["logits"].sum() * 0.0
    if two_stage_r1:
        # R1 makes readiness the only commitment gate. The identity decoder
        # cannot escape into DEFER after the calibrated gate opens.
        selective = F.binary_cross_entropy_with_logits(
            out["reliability_logit"], batch["reliability"], pos_weight=torch.tensor(2.0, device=out["logits"].device)
        )
    elif allow_defer:
        # log(P(commit)/P(defer)) is the stable two-way logit.  Using the
        # logits-domain loss is BF16 autocast-safe and exactly represents the
        # registered selective commitment objective.
        commit_logit = torch.logsumexp(out["logits"][:, :-1], dim=1) - out["logits"][:, -1]
        selective = F.binary_cross_entropy_with_logits(commit_logit, batch["reliability"])
    else:
        selective = out["logits"].sum() * 0.0
    merge_mask = batch["merge"]
    if merge_mask.any() and variant not in {"b3", "no_merge"}:
        pre_token, _ = model.encode_sequence(batch["pre"][merge_mask], batch["pre_lengths"][merge_mask])
        if two_stage_r1:
            merge = F.binary_cross_entropy_with_logits(
                model.reliability_head(pre_token).squeeze(-1), torch.zeros(len(pre_token), device=pre_token.device)
            )
        else:
            pre_states = out["state_tokens"][merge_mask]
            pre_logits, _ = model.decode(pre_token, pre_states, batch["state_mask"][merge_mask],
                                         batch["known_mask"][merge_mask], allow_defer=True)
            defer_index = pre_logits.shape[1] - 1
            merge = F.cross_entropy(pre_logits, torch.full((len(pre_logits),), defer_index, device=pre_logits.device, dtype=torch.long))
    else:
        merge = out["logits"].sum() * 0.0
    temporal_mask = batch["temporal"]
    if temporal_mask.any() and not model.no_history:
        seq = out["sequence"][temporal_mask]
        lengths = batch["lengths"][temporal_mask]
        last_idx = lengths - 1; prev_idx = lengths - 2
        last = seq[torch.arange(len(seq), device=seq.device), last_idx]
        prev = seq[torch.arange(len(seq), device=seq.device), prev_idx]
        temporal = (1.0 - F.cosine_similarity(last, prev, dim=-1)).mean()
    else:
        temporal = out["logits"].sum() * 0.0
    terms = {
        "action": action, "known": known, "reliability": reliability,
        "metric": metric, "selective": selective, "merge": merge, "temporal": temporal,
    }
    mapping = {
        "action": "set_conditioned_action_and_id_ce", "known": "balanced_supported_known_ce",
        "reliability": "reliability_bce", "metric": "category_disjoint_metric",
        "selective": "selective_defer_risk", "merge": "merge_recovery_transition",
        "temporal": "reliable_temporal_consistency",
    }
    total = sum(terms[k] * float(weights[mapping[k]]) for k in terms)
    return total, {k: float(v.detach()) for k, v in terms.items()}


def checkpoint_value(model: DSTM, optimizer: torch.optim.Optimizer, scheduler: Any,
                     step: int, data: FoldData, config: dict[str, Any], seed: int,
                     variant: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "protocol": "trackocd_iclr27_phase18_dstm_checkpoint",
        "step": step, "fold": data.fold_id, "seed": seed, "variant": variant,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "config": config,
        "known_ids": data.known_ids, "geometry_mean": data.geom_mean,
        "geometry_std": data.geom_std, "data_manifest": data.manifest_summary(),
        "calibration": calibration,
        "code_sha256": {
            "model": sha(ROOT / "src/iclr27_phase18/models/dstm.py"),
            "data": sha(ROOT / "src/iclr27_phase18/training/data.py"),
            "trainer": sha(ROOT / "src/iclr27_phase18/training/train_dstm_fold.py"),
            "runtime": sha(ROOT / "src/iclr27_phase18/evaluation/dstm_runtime.py"),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text())
    updates = args.updates if args.updates is not None else int(config["training"]["updates_per_fold"])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", args.device)
    data = FoldData(args.fold, config)
    no_history = args.variant == "no_history"
    model = DSTM(data.input_dim, int(config["model"]["hidden_dim"]), int(config["model"]["row_projection_dim"]),
                 len(data.known_ids), int(config["model"]["max_training_state_candidates"]), no_history=no_history).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    warmup = int(config["training"]["warmup_updates"]); base_lr = float(config["training"]["learning_rate"])
    def factor(step: int) -> float:
        if step < warmup:
            return max(.01, (step + 1) / warmup)
        progress = (step - warmup) / max(updates - warmup, 1)
        return .05 + .95 * .5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    known_weights = torch.from_numpy(data.known_class_weights).to(device)
    pos_weight = torch.tensor(data.reliability_pos_weight, device=device)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float32
    checkpoint_interval = args.checkpoint_interval or int(config["training"]["checkpoint_interval"])
    curves = []; accumulator: dict[str, list[float]] = defaultdict(list); action_counts = Counter()
    best_score = -float("inf"); best_step = 0; best_calibration = None
    start_time = time.time(); finite_grad_steps = 0
    model.train()
    for step in range(1, updates + 1):
        raw = data.build_batch(step, args.seed, args.variant); action_counts.update(raw["action_names"])
        batch = to_device(raw, device); optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp == "bf16", dtype=amp_dtype):
            total, terms = losses(model, batch, known_weights, pos_weight, config["loss_weights"], args.variant)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite total loss at step {step}: {float(total)}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {float(grad_norm)}")
        finite_grad_steps += 1; optimizer.step(); scheduler.step()
        accumulator["total"].append(float(total.detach())); accumulator["grad_norm"].append(float(grad_norm))
        for key, value in terms.items(): accumulator[key].append(value)
        if step % 100 == 0:
            point = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
            point.update({k: float(np.mean(v[-100:])) for k, v in accumulator.items()})
            curves.append(point)
        if step % checkpoint_interval == 0 or step == updates:
            model.eval()
            calibration = evaluate_calibration(model, data, device, args.variant)
            score = float(calibration["composite"])
            value = checkpoint_value(model, optimizer, scheduler, step, data, config, args.seed, args.variant, calibration)
            atomic_checkpoint(args.latest, value)
            if score > best_score:
                best_score = score; best_step = step; best_calibration = calibration
                atomic_checkpoint(args.best, value)
            print(json.dumps({"step": step, "fold": args.fold, "seed": args.seed, "variant": args.variant,
                              "calibration_composite": score, "best_step": best_step,
                              "train_loss_100": (curves[-1]["total"] if curves else float(np.mean(accumulator["total"][-100:]))),
                              "elapsed_seconds": time.time() - start_time}, sort_keys=True), flush=True)
            model.train()
    elapsed = time.time() - start_time
    deterministic_examples = updates * int(config["training"]["deterministic_population_examples_per_update"])
    passes = deterministic_examples / len(data.fit_indices)
    full_registered = updates >= int(config["training"]["updates_per_fold"])
    if full_registered:
        assert passes >= 10.0, passes
    summary = {
        "protocol": "trackocd_iclr27_phase18_dstm_fold_training",
        "fold": args.fold, "seed": args.seed, "variant": args.variant,
        "updates": updates, "full_registered_run": full_registered,
        "finite_gradient_steps": finite_grad_steps, "amp": args.amp,
        "elapsed_seconds": elapsed, "updates_per_second": updates / elapsed,
        "deterministic_population_examples": deterministic_examples,
        "complete_unique_fit_row_passes": passes,
        "balanced_episode_examples": updates * int(config["training"]["balanced_episode_examples_per_update"]),
        "action_example_counts": dict(action_counts), "parameters": parameter_counts(model),
        "final_loss_means": {k: float(np.mean(v[-min(100, len(v)):])) for k, v in accumulator.items()},
        "best_step": best_step, "best_calibration_composite": best_score,
        "best_calibration": best_calibration, "curves": curves,
        "data": data.manifest_summary(), "best_checkpoint": str(args.best.resolve()),
        "latest_checkpoint": str(args.latest.resolve()), "device": str(device),
    }
    atomic_json(args.summary, summary)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.done.with_name(args.done.name + ".tmp"); tmp.write_text("done\n"); os.replace(tmp, args.done)
    print(json.dumps({"complete": True, "fold": args.fold, "seed": args.seed,
                      "variant": args.variant, "best_step": best_step, "passes": passes,
                      "updates_per_second": summary["updates_per_second"]}, sort_keys=True), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", choices=["dstm", "b3", "no_merge", "no_history", "repair_r1", "repair_r2", "repair_r3", "repair_r4", "repair_r5"], required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/iclr27_phase18/dstm.json")
    p.add_argument("--updates", type=int)
    p.add_argument("--checkpoint-interval", type=int)
    p.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--best", type=Path, required=True)
    p.add_argument("--latest", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--done", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
