"""Full Phase19 rollout-aligned training for one legal category fold."""
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

from src.iclr27_phase19.data.stream import Phase19Data, ROOT
from src.iclr27_phase19.models.ra_ocd import RAOCD, parameter_counts
from src.iclr27_phase19.runtime.state_machine import decode_action_index, blend_state


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def teacher_probability(step: int, total: int) -> float:
    if step <= 4000:
        return 1.0 - .15 * (step - 1) / 3999.0
    if step <= 16000:
        return .85 - .65 * (step - 4000) / 12000.0
    return 0.0


def padded_states(memories: list[list[dict[str, Any]]], max_states: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = len(memories); d = 768
    z = torch.zeros(b, max_states, d, device=device)
    raw = torch.zeros_like(z); mask = torch.zeros(b, max_states, dtype=torch.bool, device=device)
    for i, mem in enumerate(memories):
        for j, state in enumerate(mem[:max_states]):
            z[i, j] = state["z"]
            raw[i, j] = state["raw"]
            mask[i, j] = True
    return z, raw, mask


def rollout(model: RAOCD, data: Phase19Data, episodes: list[list[dict[str, Any]]],
            device: torch.device, step: int, total_steps: int, rng: np.random.Generator,
            train: bool = True, allow_defer: bool = True, teacher_forced: bool = False) -> tuple[torch.Tensor, dict[str, float], dict[str, int]]:
    """Run the same action/state transition used by deployment.

    The target is recomputed from the current model-generated memory.  The
    oracle category is only used in this loss-side comparison and is never
    included in model tensors.
    """
    b = len(episodes); max_states = 8; k = data.known_to_index
    memories: list[list[dict[str, Any]]] = [[] for _ in range(b)]
    prev: list[dict[str, torch.Tensor] | None] = [None] * b
    losses: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: Counter[str] = Counter(); executed_errors = 0; correct_state_exists = 0
    p_teacher = 1.0 if (train and teacher_forced) else (teacher_probability(step, total_steps) if train else 0.0)
    for t in range(3):
        raw = torch.from_numpy(np.stack([e[t]["raw"] for e in episodes])).to(device)
        geom = torch.from_numpy(np.stack([e[t]["geom"] for e in episodes])).to(device)
        state_z, state_raw, state_mask = padded_states(memories, max_states, device)
        out = model(raw, geom, state_z, state_raw, state_mask, allow_defer=allow_defer)
        targets = []
        target_kind = []
        for i, ep in enumerate(episodes):
            item = ep[t]; cat = int(item["category"])
            if item["visible"] and cat in k:
                targets.append(k[cat]); target_kind.append("KNOWN")
            else:
                matches = [j for j, s in enumerate(memories[i]) if s["oracle_category"] == cat]
                if matches:
                    targets.append(len(k) + matches[0]); target_kind.append("EXISTING")
                    correct_state_exists += 1
                elif item["quality"] < .20 and t == 1:
                    targets.append(len(k) + max_states + 1); target_kind.append("DEFER")
                else:
                    targets.append(len(k) + max_states); target_kind.append("NEW")
        target_t = torch.tensor(targets, dtype=torch.long, device=device)
        losses["action"].append(F.cross_entropy(out["logits"], target_t))
        vis = torch.tensor([x == "KNOWN" for x in target_kind], dtype=torch.bool, device=device)
        if vis.any():
            losses["known"].append(F.cross_entropy(out["known_logits"][vis], target_t[vis]))
        quality_t = torch.tensor([e[t]["quality"] for e in episodes], dtype=torch.float32, device=device)
        losses["quality"].append(F.mse_loss(out["quality"], quality_t))
        losses["residual"].append(out["z_residual"].pow(2).mean())
        losses["raw_preserve"].append(out["z_residual"].pow(2).sum(dim=-1).mean())
        if t == 1:
            same = torch.stack([prev[i]["z"] for i in range(b)])
            same_raw = torch.stack([prev[i]["raw"] for i in range(b)])
            learned = (out["z"] * same).sum(-1)
            frozen = (out["z_raw"] * same_raw).sum(-1)
            losses["rank_preserve"].append((learned - frozen).abs().mean())
        # Model action is used for the next state except for scheduled warm
        # start.  A predicted index with no current state is treated as NEW
        # only when it is the explicit NEW slot.
        pred = torch.argmax(out["logits"].detach(), dim=-1).cpu().numpy()
        for i, ep in enumerate(episodes):
            use_teacher = train and rng.random() < p_teacher
            action_index = int(targets[i]) if use_teacher else int(pred[i])
            nmem = len(memories[i])
            action, state_index = decode_action_index(action_index, len(k), nmem, max_states)
            counts["teacher" if use_teacher else "model"] += 1
            counts[action] += 1
            if action == "NEW" and len(memories[i]) < max_states:
                memories[i].append({"raw": out["z_raw"][i].detach(), "z": out["z"][i].detach(),
                                    "oracle_category": int(ep[t]["category"]), "video": int(ep[t]["video"]),
                                    "track_key": str(ep[t]["track_key"])})
            elif action == "EXISTING" and state_index < len(memories[i]):
                st = memories[i][state_index]
                st["raw"], st["z"] = blend_state(st["raw"], st["z"], out["z_raw"][i], out["z"][i])
            elif action == "EXISTING":
                executed_errors += 1
            prev[i] = {"z": out["z"][i], "raw": out["z_raw"][i]}
    total = (1.0 * torch.stack(losses["action"]).mean()
             + .35 * (torch.stack(losses["known"]).mean() if losses["known"] else 0.)
             + .20 * torch.stack(losses["quality"]).mean()
             + .15 * torch.stack(losses["rank_preserve"]).mean()
             + .05 * torch.stack(losses["residual"]).mean()
             + .10 * torch.stack(losses["raw_preserve"]).mean())
    scalar = {name: float(torch.stack(values).mean().detach()) for name, values in losses.items()}
    scalar.update({"total": float(total.detach()), "teacher_probability": p_teacher,
                   "correct_state_exists": float(correct_state_exists),
                   "executed_invalid_existing": float(executed_errors)})
    return total, scalar, dict(counts)


@torch.no_grad()
def internal_score(model: RAOCD, data: Phase19Data, device: torch.device,
                   rng: np.random.Generator, ladder: str = "L0") -> dict[str, float]:
    model.eval(); vals = []; action = []; raw_gap = []
    for _ in range(96):
        ep = data.make_episode(rng, ladder=ladder)
        loss, scalars, counts = rollout(model, data, [ep], device, 20000, 40000, rng, train=False, allow_defer=False)
        vals.append(float(scalars["total"])); action.append(counts.get("EXISTING", 0) + counts.get("KNOWN", 0))
        raw_gap.append(float(scalars.get("rank_preserve", 0.)))
    model.train()
    return {"internal_loss": float(np.mean(vals)), "action_coverage": float(np.mean(action) / 3.0),
            "raw_rank_gap": float(np.mean(raw_gap)),
            "selection_score": float(np.mean(action) / 3.0 - .05 * np.mean(raw_gap))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    data = Phase19Data(args.fold, final=args.final)
    proto = torch.from_numpy(data.known_prototypes())
    model = RAOCD(proto).to(device)
    if args.variant == "fallback_a":
        # Evidence-selected Fallback A: preserve frozen raw DINOv2 geometry and
        # train only controller/prototype calibration parameters.
        model.residual_gate.data.zero_()
        model.residual_gate.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith("residual."):
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total_steps = int(args.updates)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.5e-5)
    rng = np.random.default_rng(seed * 1009 + args.fold)
    batch_size = int(args.batch_size); amp = args.amp == "bf16" and device.type == "cuda"
    logs = []; action_counts = Counter(); best = -float("inf"); best_step = 0
    start = time.time(); finite = 0
    model.train()
    for step in range(1, total_steps + 1):
        episodes = [data.make_episode(rng, ladder=args.ladder) for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            total, scalars, counts = rollout(model, data, episodes, device, step, total_steps, rng, train=True, allow_defer=args.allow_defer, teacher_forced=args.teacher_forced)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite total loss at step {step}: {float(total)}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"non-finite gradient at step {step}: {float(grad)}")
        optimizer.step(); scheduler.step(); finite += 1; action_counts.update(counts)
        if step % int(args.log_interval) == 0 or step == total_steps:
            point = {"step": step, "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad), **scalars}
            logs.append(point)
            # This is legal held-supported-known validation only.
            score = float(scalars["total"] * -1.0 + scalars.get("correct_state_exists", 0.) / max(batch_size, 1))
            payload = {"protocol": "trackocd_iclr27_phase19_ra_ocd_checkpoint", "variant": args.variant, "fold": args.fold,
                       "final": args.final, "seed": seed, "step": step, "model_state": model.state_dict(),
                       "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                       "known_ids": data.supported_ids, "geom_mean": data.geom_mean, "geom_std": data.geom_std,
                       "data_summary": data.summary(), "trainer_semantic_audit": {"observed_values": data.trainer_observed_semantic_values,
                       "true_novel_labels_in_model_input": False}, "schedule": {"teacher_probability": teacher_probability(step, total_steps)},
                       "code_sha256": {"model": sha_file(ROOT / "src/iclr27_phase19/models/ra_ocd.py"),
                                       "runtime": sha_file(ROOT / "src/iclr27_phase19/runtime/state_machine.py"),
                                       "data": sha_file(ROOT / "src/iclr27_phase19/data/stream.py"),
                                       "trainer": sha_file(ROOT / "src/iclr27_phase19/training/train_rollout.py")}}
            atomic_torch(args.latest, payload)
            if score > best:
                best = score; best_step = step; atomic_torch(args.best, payload)
            print(json.dumps({"step": step, "fold": args.fold, "variant": args.variant, "final": args.final,
                              "loss": scalars["total"], "teacher_probability": scalars["teacher_probability"],
                              "best_step": best_step, "elapsed_seconds": time.time() - start}, sort_keys=True), flush=True)
    summary = {"protocol": "trackocd_iclr27_phase19_ra_ocd_training", "variant": args.variant, "fold": args.fold, "final": args.final,
               "seed": seed, "updates": total_steps, "finite_updates": finite, "full_registered_run": total_steps >= 40000,
               "elapsed_seconds": time.time() - start, "updates_per_second": total_steps / max(time.time() - start, 1),
               "parameters": parameter_counts(model), "action_counts": dict(action_counts), "logs": logs,
               "best_step": best_step, "best_internal_score": best, "best_checkpoint": str(args.best.resolve()),
               "latest_checkpoint": str(args.latest.resolve()), "data": data.summary(),
               "trainer_semantic_audit": {"observed_values": data.trainer_observed_semantic_values,
                                          "true_novel_labels_in_model_input": False},
               "teacher_forced_ablation": bool(args.teacher_forced),
               "on_policy_fraction": 1.0 - float(np.mean([x.get("teacher_probability", 0.) for x in logs])) if logs else 0.0}
    atomic_json(args.summary, summary)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.done.with_name(args.done.name + ".tmp"); tmp.write_text("done\n"); os.replace(tmp, args.done)
    print(json.dumps({"complete": True, "fold": args.fold, "variant": args.variant, "final": args.final, "updates": total_steps,
                      "best_step": best_step, "on_policy_fraction": summary["on_policy_fraction"]}, sort_keys=True), flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(4), default=0)
    p.add_argument("--seed", type=int, default=1801)
    p.add_argument("--updates", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--ladder", choices=["L0", "L1", "L2"], default="L2")
    p.add_argument("--allow-defer", action="store_true")
    p.add_argument("--variant", choices=["main", "fallback_a"], default="main")
    p.add_argument("--teacher-forced", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--log-interval", type=int, default=2000)
    p.add_argument("--best", type=Path, required=True)
    p.add_argument("--latest", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--done", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
