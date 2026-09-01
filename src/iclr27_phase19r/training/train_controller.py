"""Full RC-MS-OCD training with persistent internal checkpoint selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from src.iclr27_phase19r.data.episodes import EpisodeFactory, MetaEpisode
from src.iclr27_phase19r.data.stream import Phase19RData, ROOT
from src.iclr27_phase19r.evaluation.internal import evaluate_model_instance, fixed_known_keys, load_events
from src.iclr27_phase19r.models.controller import RCMSOCD, parameter_counts
from src.iclr27_phase19r.training.rollout import rollout_batch


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def internal_validate(model: RCMSOCD, data: Phase19RData, episodes: list[MetaEpisode], device: torch.device,
                      ladder: str = "L2") -> dict[str, Any]:
    model.eval(); rng = np.random.default_rng(7700 + data.fold)
    counts = Counter(); by_cat: dict[int, list[int]] = defaultdict(list); nmi_vals = []; frag_vals = []
    traces_all = []
    with torch.no_grad():
        for ep in episodes:
            _, scalars, _, traces = rollout_batch(model, data, [ep], device, 0, 1, rng, train=False, ladder=ladder, allow_defer=(ladder == "L2"))
            tr = traces[0]; traces_all.append(tr)
            pred_clusters = []; gt_clusters = []
            for item, rec in zip(ep.items, tr):
                target = rec["target_kind"]
                action = rec["action"]
                correct_existing = False
                if action == "EXISTING" and item.oracle_category_for_loss_only is not None:
                    sid = rec.get("semantic_id")
                    correct_existing = any(s.get("sid") == sid and s.get("oracle_birth_category") == item.oracle_category_for_loss_only for s in rec.get("states", []))
                if target == "EXISTING":
                    counts["existing_target"] += 1; counts["existing_correct"] += int(correct_existing); by_cat[int(item.oracle_category_for_loss_only)].append(int(correct_existing))
                if target == "NEW":
                    counts["new_target"] += 1; counts["new_correct"] += int(action == "NEW")
                    if action == "EXISTING": counts["false_merge"] += 1
                if target == "KNOWN": counts["known_target"] += 1; counts["known_correct"] += int(action == "KNOWN")
                if target == "DEFER": counts["defer_target"] += 1; counts["defer_correct"] += int(action == "DEFER")
                if item.role == "pseudo_novel":
                    gt_clusters.append(int(item.oracle_category_for_loss_only) if item.oracle_category_for_loss_only is not None else -1)
                    pred_clusters.append(int(rec["semantic_id"]) if rec["action"] in {"NEW", "EXISTING"} and rec.get("semantic_id") is not None else -1)
            if len(set(gt_clusters)) > 1:
                nmi_vals.append(float(normalized_mutual_info_score(gt_clusters, pred_clusters)))
            frag_vals.append(float(max(0, len([s for s in tr[-1].get("states", []) if s.get("oracle_birth_category") is not None]) - len(set(gt_clusters)))))
    existing_p = counts["existing_correct"] / max(sum(1 for tr in traces_all for r in tr if r["target_kind"] == "EXISTING" and r["action"] == "EXISTING"), 1)
    existing_recall = counts["existing_correct"] / max(counts["existing_target"], 1)
    new_precision = counts["new_correct"] / max(sum(1 for tr in traces_all for r in tr if r["action"] == "NEW"), 1)
    new_recall = counts["new_correct"] / max(counts["new_target"], 1)
    known_macro = float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else 0.0
    existing_f1 = 2 * existing_p * existing_recall / max(existing_p + existing_recall, 1e-9)
    new_f1 = 2 * new_precision * new_recall / max(new_precision + new_recall, 1e-9)
    false_merge_rate = counts["false_merge"] / max(counts["new_target"], 1)
    positive_reuse = existing_recall
    novel_nmi = float(np.mean(nmi_vals)) if nmi_vals else 0.0
    fragmentation = float(np.mean(frag_vals)) if frag_vals else 0.0
    score = .30 * existing_f1 + .20 * new_f1 + .15 * known_macro + .15 * positive_reuse + .10 * novel_nmi - .10 * false_merge_rate - .05 * fragmentation
    result = {"protocol": "trackocd_iclr27_phase19r_internal_selection", "fold": data.fold, "ladder": ladder,
              "episodes": len(episodes), "counts": dict(counts), "existing_precision": float(existing_p),
              "existing_recall": float(existing_recall), "existing_f1": float(existing_f1),
              "new_precision": float(new_precision), "new_recall": float(new_recall), "new_f1": float(new_f1),
              "known_macro": known_macro, "positive_reuse_recall_macro": positive_reuse,
              "novel_nmi_macro": novel_nmi, "false_merge_rate_macro": float(false_merge_rate),
              "fragmentation_rate_macro": fragmentation, "selection_score": float(score),
              "known_micro": counts["known_correct"] / max(counts["known_target"], 1),
              "action_accuracy": sum(r["action"] == r["target_kind"] for tr in traces_all for r in tr) / max(sum(len(tr) for tr in traces_all), 1)}
    model.train(); return result


def persistent_selection_score(event_metrics: dict[str, Any]) -> float:
    """The exact preregistered seven-term primary score."""
    required = ("existing_f1_macro", "new_f1_macro", "known_macro",
                "positive_reuse_recall_macro", "novel_nmi_macro",
                "false_merge_rate_macro", "fragmentation_rate_macro")
    missing = [k for k in required if k not in event_metrics]
    if missing:
        raise KeyError(f"persistent evaluator missing preregistered terms: {missing}")
    return float(event_metrics["selection_score"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed = int(args.seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device); data = Phase19RData(args.fold, final=args.final)
    train_factory = EpisodeFactory(data, ladder=args.ladder, validation=False,
                                   index_path=getattr(args, "episode_index", None))
    event_ratio = float(getattr(args, "event_ratio", 0.0))
    event_factory = None
    event_rng = np.random.default_rng(seed * 1013 + args.fold + 7919)
    if event_ratio > 0.0:
        event_factory = EpisodeFactory(data, ladder=args.ladder, validation=False,
                                       index_path=getattr(args, "event_aligned_index", None))
    valid_factory = EpisodeFactory(data, ladder="L2", validation=True)
    fixed_rng = np.random.default_rng(seed * 37 + args.fold)
    fixed_validation = [valid_factory.sample(fixed_rng) for _ in range(int(args.validation_episodes))]
    # Fold training has a held-known persistent evaluator.  Final all-known
    # training has no held category left, so it uses the fixed episode diagnostic
    # only; no public truth is consulted and the final checkpoint is frozen by
    # the caller after the registered budget.
    fixed_events = [] if args.final else load_events(args.fold)
    fixed_known = fixed_known_keys(data)
    model = RCMSOCD(torch.from_numpy(data.known_prototypes), torch.from_numpy(data.active_known_mask), max_states=int(args.max_states), known_bias=torch.from_numpy(data.known_bias)).to(device)
    prototype_hash = model.prototype_hash(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.updates), eta_min=float(args.lr) * .05)
    amp = args.amp == "bf16" and device.type == "cuda"; rng = np.random.default_rng(seed * 1009 + args.fold)
    logs = []; best = -float("inf"); best_step = 0; finite = 0; action_counts = Counter(); start = time.time(); last_val = None

    def rng_snapshot() -> dict[str, Any]:
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "episode_rng": rng.bit_generator.state,
                "episode_sampler": train_factory.state_dict(),
                "event_rng": event_rng.bit_generator.state,
                "event_sampler": None if event_factory is None else event_factory.state_dict()}

    def checkpoint_payload(step: int, validation: dict[str, Any] | None) -> dict[str, Any]:
        return {"protocol": "trackocd_iclr27_phase19r_rc_ms_ocd_checkpoint", "fold": args.fold,
                "final": args.final, "seed": seed, "step": int(step), "global_update": int(step),
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": None,
                "known_ids": data.supported_ids,
                "active_supported_known_ids": [data.supported_ids[i] for i, x in enumerate(data.active_known_mask) if x],
                "prototype_hash": prototype_hash, "data_summary": data.summary(),
                "validation": validation, "config": vars(args), "logs": logs,
                "finite_updates": int(finite), "action_counts": dict(action_counts),
                "best_step": int(best_step), "best_internal_score": float(best),
                "rng_state": rng_snapshot(),
                "trainer_semantic_audit": {"true_novel_labels_in_model_input": False,
                                            "physical_id_used_as_feature": False}}

    resume_path = getattr(args, "resume", None)
    if resume_path:
        rp = Path(resume_path)
        if not rp.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {rp}")
        saved = torch.load(rp, map_location="cpu")
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        logs = list(saved.get("logs", [])); finite = int(saved.get("finite_updates", saved.get("step", 0)))
        action_counts.update(saved.get("action_counts", {})); best_step = int(saved.get("best_step", 0)); best = float(saved.get("best_internal_score", -float("inf")))
        last_val = saved.get("validation")
        rs = saved.get("rng_state", {})
        if rs.get("python") is not None: random.setstate(rs["python"])
        if rs.get("numpy") is not None: np.random.set_state(rs["numpy"])
        if rs.get("torch") is not None: torch.set_rng_state(rs["torch"])
        if torch.cuda.is_available() and rs.get("cuda") is not None: torch.cuda.set_rng_state_all(rs["cuda"])
        if rs.get("episode_rng") is not None: rng.bit_generator.state = rs["episode_rng"]
        if rs.get("event_rng") is not None: event_rng.bit_generator.state = rs["event_rng"]
        sampler = rs.get("episode_sampler") or {}
        if train_factory._index_store is not None and sampler.get("index_store"):
            train_factory._index_store.cursor = int(sampler["index_store"].get("cursor", train_factory._index_store.cursor))
        event_sampler = rs.get("event_sampler") or {}
        if event_factory is not None and event_factory._index_store is not None and event_sampler.get("index_store"):
            event_factory._index_store.cursor = int(event_sampler["index_store"].get("cursor", event_factory._index_store.cursor))
        start_step = int(saved.get("global_update", saved.get("step", 0)))
    else:
        start_step = 0

    for step in range(start_step + 1, int(args.updates) + 1):
        n_event = int(round(int(args.batch_size) * event_ratio)) if event_factory is not None else 0
        n_event = max(0, min(int(args.batch_size), n_event))
        n_mixed = int(args.batch_size) - n_event
        episodes = [train_factory.sample(rng) for _ in range(n_mixed)]
        if event_factory is not None:
            episodes.extend(event_factory.sample(event_rng) for _ in range(n_event))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            total, scalars, counts, _ = rollout_batch(model, data, episodes, device, step, int(args.updates), rng,
                                                      train=True, ladder=args.ladder,
                                                      allow_defer=args.ladder == "L2",
                                                      event_commit_weight=float(getattr(args, "event_commit_weight", 0.0)))
        if not torch.isfinite(total): raise FloatingPointError(f"non-finite loss step {step}: {float(total)}")
        total.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad): raise FloatingPointError(f"non-finite gradient step {step}: {float(grad)}")
        optimizer.step(); scheduler.step(); finite += 1; action_counts.update(counts)
        if step % int(args.log_interval) == 0 or step == int(args.updates):
            episode_val = internal_validate(model, data, fixed_validation, device, ladder="L2")
            selection_tmp = args.latest.with_name(args.latest.name + ".selection_tmp.pt")
            # Materialize the exact model state before evaluator replay.  The
            # temporary artifact is removed after the fixed event evaluation;
            # only latest_valid/best_internal survive the checkpoint policy.
            atomic_torch(selection_tmp, {"model_state": model.state_dict(), "prototype_hash": prototype_hash,
                                         "step": step, "fold": args.fold, "selection_tmp": True})
            try:
                event_val = evaluate_model_instance(model, data, device, fixed_events, fixed_known)
            finally:
                selection_tmp.unlink(missing_ok=True)
            event_metrics = event_val["metrics"]
            # Fold checkpoint selection is based on persistent causal replay;
            # final all-known training has no legal held category and therefore
            # records the fixed episode score as a diagnostic only.
            # Fold selection always uses the exact event score.  For final
            # all-known training the event set is empty, so the legal fixed
            # episode diagnostic is used solely to identify a checkpoint for
            # freezing; no fold/public truth is consulted.
            selection = (persistent_selection_score(event_metrics) if fixed_events else float(episode_val["selection_score"]))
            selection_source = ("full_persistent_held_known_event_evaluator" if fixed_events else "final_fixed_episode_diagnostic")
            val = dict(episode_val)
            val.update({"selection_score": float(selection),
                        "selection_metric_source": selection_source,
                        "episode_proxy_selection_score": float(episode_val["selection_score"]),
                        "persistent_event_metrics": event_metrics})
            entry = {"step": step, "loss": float(scalars["total"]), "grad_norm": float(grad), "lr": optimizer.param_groups[0]["lr"], "train": scalars, "actions": counts, "validation": val, "episode_validation": episode_val, "persistent_event_validation": {"events": len(fixed_events), "metrics": event_metrics, "known_metrics": event_val.get("known_metrics", {})}}
            logs.append(entry)
            last_val = val
            if val["selection_score"] > best:
                best = val["selection_score"]; best_step = step
            payload = checkpoint_payload(step, val)
            atomic_torch(args.latest, payload)
            if best_step == step:
                atomic_torch(args.best, payload)
            print(json.dumps({"step": step, "fold": args.fold, "final": args.final, "loss": scalars["total"], "selection_score": val["selection_score"], "selection_metric_source": val["selection_metric_source"], "best_step": best_step, "elapsed_seconds": time.time() - start}, sort_keys=True), flush=True)
        elif int(getattr(args, "save_interval", 1000)) > 0 and step % int(getattr(args, "save_interval", 1000)) == 0:
            # Recovery cadence is independent of the frozen 4,000-update
            # formal validation/selection cadence.
            atomic_torch(args.latest, checkpoint_payload(step, last_val))
    assert model.prototype_hash() == prototype_hash
    summary = {"protocol": "trackocd_iclr27_phase19r_rc_ms_ocd_training", "fold": args.fold, "final": args.final, "seed": seed, "updates": int(args.updates), "finite_updates": finite, "full_registered_run": int(args.updates) >= 50000, "elapsed_seconds": time.time() - start, "updates_per_second": int(args.updates) / max(time.time() - start, 1), "parameters": parameter_counts(model), "action_counts": dict(action_counts), "logs": logs, "best_step": best_step, "best_internal_score": best, "selection_metric_source": ("full_persistent_held_known_event_evaluator" if fixed_events else "final_fixed_episode_diagnostic"), "prototype_hash_before_after": [prototype_hash, model.prototype_hash()], "fixed_validation_episodes": len(fixed_validation), "fixed_persistent_events": len(fixed_events), "known_stage_frozen": True, "trainer_true_novel_labels": False, "event_aligned_ratio": event_ratio, "event_commit_weight": float(getattr(args, "event_commit_weight", 0.0)), "event_aligned_index": None if event_factory is None or event_factory._index_store is None else str(event_factory._index_store.path), "on_policy_fraction": float(np.mean([x["train"].get("on_policy_fraction", 0.) for x in logs])) if logs else 0.0}
    atomic_json(args.summary, summary); args.done.parent.mkdir(parents=True, exist_ok=True); tmp = args.done.with_name(args.done.name + ".tmp"); tmp.write_text("done\n"); os.replace(tmp, args.done)
    print(json.dumps({"complete": True, "fold": args.fold, "final": args.final, "updates": int(args.updates), "best_step": best_step, "prototype_hash": prototype_hash}, sort_keys=True)); return summary


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, choices=range(4), default=0); p.add_argument("--seed", type=int, default=1902); p.add_argument("--updates", type=int, default=50000); p.add_argument("--batch-size", type=int, default=24); p.add_argument("--device", default="cuda:0"); p.add_argument("--amp", choices=["bf16", "fp32"], default="bf16"); p.add_argument("--ladder", choices=["L0", "L1", "L2"], default="L2"); p.add_argument("--max-states", type=int, default=16); p.add_argument("--validation-episodes", type=int, default=64); p.add_argument("--log-interval", type=int, default=4000); p.add_argument("--save-interval", type=int, default=1000); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--final", action="store_true"); p.add_argument("--episode-index", type=Path, default=None); p.add_argument("--event-aligned-index", type=Path, default=None); p.add_argument("--event-ratio", type=float, default=0.0); p.add_argument("--event-commit-weight", type=float, default=0.0); p.add_argument("--resume", type=Path, default=None); p.add_argument("--best", type=Path, required=True); p.add_argument("--latest", type=Path, required=True); p.add_argument("--summary", type=Path, required=True); p.add_argument("--done", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()
