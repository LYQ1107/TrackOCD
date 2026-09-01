#!/usr/bin/env python3
"""Train one fold of the Phase25 attention-based proposal-set selector."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase25.protocol import CSV_PATH, FEAT_PATH, load_aligned_features
from src.iclr27_phase25.set_selector import ProposalSetAttentionSelector, metadata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase25"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try: torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    val = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins; m = (probs >= lo) & (probs <= hi if i == bins - 1 else probs < hi)
        if np.any(m): val += float(m.mean()) * abs(float(probs[m].mean()) - float(labels[m].mean()))
    return val


def evaluate(model: ProposalSetAttentionSelector, arr: dict[str, np.ndarray], bank: torch.Tensor, device: torch.device, batch_rows: int = 64) -> dict[str, Any]:
    model.eval(); n = len(arr["row_idx"]); q_all = np.zeros((n, arr["mask"].shape[1]), np.float32); u_all = np.zeros_like(q_all)
    with torch.no_grad():
        for st in range(0, n, batch_rows):
            sl = slice(st, min(st + batch_rows, n)); pp = arr["parent_idx"][sl]; vis = bank[np.maximum(pp, 0)]; geom = torch.from_numpy(arr["geom"][sl]).to(device, non_blocking=True); mask = torch.from_numpy(arr["mask"][sl]).to(device, non_blocking=True); q, u = model(vis, geom, mask); q_all[sl] = q.float().cpu().numpy(); u_all[sl] = u.float().cpu().numpy()
    recalls = {str(t): {k: [] for k in (5, 10, 20, 27)} for t in (.3, .5, .7)}; top_iou = {k: [] for k in (5, 10, 20, 27)}; labels_flat: list[float] = []; probs: list[float] = []; uncertainty: list[float] = []
    for i in range(n):
        ids = np.flatnonzero(arr["mask"][i].astype(bool));
        if not len(ids): continue
        order = ids[np.argsort(q_all[i, ids])[::-1]]; lab = arr["label_iou"][i, order]; ass = arr["parent_assigned"][i, order]; labels_flat.extend(arr["label_iou"][i, ids].tolist()); probs.extend((1. / (1. + np.exp(-q_all[i, ids]))).tolist()); uncertainty.extend((1. / (1. + np.exp(-u_all[i, ids]))).tolist())
        for k in recalls["0.5"]:
            kk = min(k, len(order)); top_iou[k].append(float(np.max(lab[:kk], initial=0.0)))
            for t in recalls: recalls[t][k].append(float(np.any(ass[:kk] & (lab[:kk] >= float(t)))))
    m: dict[str, Any] = {"rows": n, "candidate_slots": int(arr["mask"].sum()), "ece_quality": ece(np.asarray(probs), np.asarray(labels_flat)) if probs else 0.0, "mean_uncertainty": float(np.mean(uncertainty)) if uncertainty else 0.0}
    for t, ks in recalls.items():
        for k, v in ks.items(): m[f"candidate_recall_at_{t}_top{k}"] = float(np.mean(v)) if v else 0.0
    for k, v in top_iou.items(): m[f"top{k}_true_iou_mean"] = float(np.mean(v)) if v else 0.0; m[f"top{k}_true_iou_median"] = float(np.median(v)) if v else 0.0
    return m


def loss_fn(q: torch.Tensor, u: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    m = mask.float(); logits_target = labels.masked_fill(~mask, -1e4) / 0.10; target = torch.softmax(logits_target, dim=-1); logp = torch.log_softmax(q.masked_fill(~mask, -1e4), dim=-1); listwise = -((target * logp) * m).sum(dim=-1).div(m.sum(dim=-1).clamp_min(1.0)).mean(); quality = (((torch.sigmoid(q) - labels) ** 2) * m).sum() / m.sum().clamp_min(1.0); reliability = (labels >= 0.5).float(); unc = (F.binary_cross_entropy_with_logits(u, reliability, reduction="none") * m).sum() / m.sum().clamp_min(1.0)
    rank_terms = []
    for b in range(labels.shape[0]):
        ids = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if ids.numel() < 2: continue
        pos = ids[torch.argmax(labels[b, ids])]; neg = ids[torch.argmin(labels[b, ids])]
        if labels[b, pos] > labels[b, neg] + 0.1: rank_terms.append(F.relu(0.25 - q[b, pos] + q[b, neg]))
    hard = torch.stack(rank_terms).mean() if rank_terms else q.sum() * 0.0
    return listwise + 0.50 * quality + 0.25 * hard + 0.10 * unc, {"listwise": listwise, "quality": quality, "hard_negative": hard, "uncertainty": unc}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=None); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--checkpoint-every", type=int, default=500); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="attention"); args = ap.parse_args()
    if args.fold not in range(4): raise ValueError(args.fold)
    torch.set_num_threads(2); device = torch.device(args.device if torch.cuda.is_available() else "cpu"); visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.expected_physical_gpu is not None and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"physical GPU mapping mismatch: expected {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible!r}")
    seed = int(args.seed) + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); cls, roi, alignment = load_aligned_features(rows); bank = torch.from_numpy(np.concatenate([cls, roi], axis=1).astype(np.float32, copy=False)).to(device, non_blocking=True)
    man = json.loads((OUT / "manifests/setaware_manifest.json").read_text(encoding="utf-8")); fit_path = OUT / "manifests" / f"setaware_fit_f{args.fold}.npz"; val_path = OUT / "manifests" / f"setaware_val_f{args.fold}.npz"; fit = {k: v for k, v in np.load(fit_path, allow_pickle=False).items()}; val = {k: v for k, v in np.load(val_path, allow_pickle=False).items()}
    run = f"{args.tag}_smoke_f{args.fold}" if args.smoke else f"{args.tag}_f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"; ckdir = OUT / "checkpoints"; logp = OUT / "logs" / f"train_{run}.jsonl"; metrics_path = OUT / "metrics" / f"{run}.json"; ckdir.mkdir(parents=True, exist_ok=True); logp.parent.mkdir(parents=True, exist_ok=True)
    if done.exists() and not args.resume: print(json.dumps({"status": "already_done", "done": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"refusing relaunch with marker {marker}")
    marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device), "cuda_visible_devices": visible, "logical_cuda_index": int(torch.cuda.current_device()) if device.type == "cuda" else None, "expected_physical_gpu": args.expected_physical_gpu, "architecture": "single_self_attention_set_selector"}) + "\n", encoding="utf-8")
    model = ProposalSetAttentionSelector(); model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); start = 0; best = -1.0; best_step = 0; history: list[dict[str, Any]] = []; latest = ckdir / f"{run}_latest.pt"; steps = 2 if args.smoke else int(args.steps)
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = int(ck.get("global_step", 0)); best = float(ck.get("best_score", -1.0)); best_step = int(ck.get("best_step", 0))
    amp = torch.bfloat16 if device.type == "cuda" else None; rng = np.random.default_rng(seed + 97); n = len(fit["row_idx"]); t0 = time.time(); cats = fit.get("row_category", np.full(n, -1, np.int32)); groups: dict[int, np.ndarray] = {int(c): np.flatnonzero(cats == c) for c in np.unique(cats)}; group_keys = [c for c, ix in groups.items() if len(ix)]
    for step in range(start + 1, steps + 1):
        model.train();
        # Group-balanced row sampling keeps long-tail TRAIN categories from
        # dominating while preserving the fixed causal candidate set.
        if group_keys:
            chosen: list[int] = []
            for _ in range(min(args.batch_size, n)):
                category = group_keys[int(rng.integers(0, len(group_keys)))]
                chosen.append(int(rng.choice(groups[category])))
            bi = np.asarray(chosen, dtype=np.int64)
        else: bi = rng.integers(0, n, size=min(args.batch_size, n))
        pp = fit["parent_idx"][bi]; vis = bank[np.maximum(pp, 0)]; geom = torch.from_numpy(fit["geom"][bi]).to(device, non_blocking=True); labels = torch.from_numpy(fit["label_iou"][bi]).to(device, non_blocking=True); mask = torch.from_numpy(fit["mask"][bi]).to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp) if amp is not None else torch.autocast(device_type="cpu", enabled=False): q, u = model(vis, geom, mask); loss, pieces = loss_fn(q, u, labels, mask)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite attention loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); rec: dict[str, Any] = {"step": step, "loss": float(loss.detach().cpu()), "loss_listwise": float(pieces["listwise"].detach().cpu()), "loss_quality": float(pieces["quality"].detach().cpu()), "loss_hard_negative": float(pieces["hard_negative"].detach().cpu()), "loss_uncertainty": float(pieces["uncertainty"].detach().cpu()), "amp": "bf16" if amp is not None else "fp32"}
        if step % int(args.checkpoint_every) == 0 or step == steps:
            valm = evaluate(model, val, bank, device); rec["validation"] = valm; score = float(valm.get("candidate_recall_at_0.5_top20", 0.0) + 0.20 * valm.get("candidate_recall_at_0.5_top5", 0.0) - 0.10 * valm.get("ece_quality", 0.0)); payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "global_step": step, "best_score": best, "best_step": best_step, "fold": args.fold, "seed": seed, "metadata": metadata(model), "protocol": "trackocd_iclr27_phase25_stage2a_attention_selector", "fit_manifest": str(fit_path), "val_manifest": str(val_path), "fit_manifest_sha256": sha256(fit_path), "val_manifest_sha256": sha256(val_path), "feature_alignment": alignment, "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "amp": "bf16" if amp is not None else "fp32"}; atomic_torch(latest, payload); atomic_torch(ckdir / f"{run}_step{step:05d}.pt", payload)
            if score > best: best, best_step = score, step; payload["best_score"], payload["best_step"] = best, best_step; atomic_torch(ckdir / f"{run}_best.pt", payload)
            rec["validation_score"], rec["best_score"], rec["elapsed_s"] = score, best, time.time() - t0; history.append(rec)
            with logp.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
    final = evaluate(model, val, bank, device); result = {"protocol": "trackocd_iclr27_phase25_stage2a_attention_selector", "fold": args.fold, "tag": args.tag, "seed": seed, "steps": steps, "smoke": bool(args.smoke), "device": str(device), "amp": "bf16" if amp is not None else "fp32", "fit_rows": int(len(fit["row_idx"])), "fit_candidate_slots": int(fit["mask"].sum()), "validation_rows": int(len(val["row_idx"])), "validation_candidate_slots": int(val["mask"].sum()), "validation_metrics": final, "best_score": best, "best_step": best_step, "history": history, "checkpoint_best": str(ckdir / f"{run}_best.pt"), "checkpoint_latest": str(latest), "marker": str(marker), "done": str(done), "feature_alignment": alignment, "forbidden_inputs": metadata(model)["forbidden_inputs"], "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}; atomic_json(metrics_path, result); done.write_text(json.dumps({"fold": args.fold, "steps": steps, "checkpoint": str(ckdir / f"{run}_best.pt"), "validation": final}, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"fold": args.fold, "steps": steps, "val_top20": final.get("candidate_recall_at_0.5_top20"), "best_step": best_step, "done": str(done)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
