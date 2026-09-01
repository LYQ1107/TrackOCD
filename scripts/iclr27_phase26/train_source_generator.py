#!/usr/bin/env python3
"""Train one fold of the Phase26 causal class-agnostic source head."""
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

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, GEOM_FIELDS, load_aligned_features
from src.iclr27_phase26.source_generator import ProposalSourceGenerator, metadata

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase26"


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
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def row_geometry(rows, indices: np.ndarray) -> np.ndarray:
    return np.asarray([[float(rows[int(i)].get(k, 0.0) or 0.0) for k in GEOM_FIELDS] for i in indices], np.float32)


def iou_torch(boxes: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    g = gt.unsqueeze(1); x1 = torch.maximum(boxes[..., 0], g[..., 0]); y1 = torch.maximum(boxes[..., 1], g[..., 1]); x2 = torch.minimum(boxes[..., 2], g[..., 2]); y2 = torch.minimum(boxes[..., 3], g[..., 3]); inter = torch.relu(x2-x1) * torch.relu(y2-y1)
    aa = torch.relu(boxes[..., 2]-boxes[..., 0]) * torch.relu(boxes[..., 3]-boxes[..., 1]); ag = torch.relu(g[..., 2]-g[..., 0]) * torch.relu(g[..., 3]-g[..., 1]); return inter / torch.clamp(aa + ag - inter, min=1e-6)


def evaluate(model, idx: np.ndarray, positive: np.ndarray, gt: np.ndarray, rows, cls, roi, device, batch_size=512) -> dict[str, Any]:
    model.eval(); top = {5: [], 10: [], 20: [], 27: []}; max_i = []; probs = []; labels = []
    with torch.no_grad():
        for st in range(0, len(idx), batch_size):
            sl = slice(st, min(st + batch_size, len(idx))); ids = idx[sl]; visual = np.concatenate([cls[ids], roi[ids]], axis=1).astype(np.float32, copy=False); geom = row_geometry(rows, ids); base = geom[:, 1:5]
            v = torch.from_numpy(visual).to(device); g = torch.from_numpy(geom).to(device); b = torch.from_numpy(base).to(device); boxes, q = model(v, g, b); ious = iou_torch(boxes, torch.from_numpy(gt[sl]).to(device)).float().cpu().numpy(); qq = torch.sigmoid(q).float().cpu().numpy();
            for j in range(len(ids)):
                if positive[sl][j] > .5:
                    max_i.append(float(np.max(ious[j]))); [top[k].append(float(np.max(ious[j, :min(k, ious.shape[1])]) >= .5)) for k in top]
                probs.extend(qq[j].tolist()); labels.extend((np.full(qq.shape[1], positive[sl][j], np.float32)).tolist())
    ece = 0.0
    if probs:
        p = np.asarray(probs); y = np.asarray(labels)
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            m = (p >= lo) & (p <= hi if hi == 1 else p < hi)
            if np.any(m): ece += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return {"rows": int(len(idx)), "positive_rows": int(np.sum(positive > .5)), "negative_rows": int(np.sum(positive <= .5)), "candidate_recall_at_0.5_top5": float(np.mean(top[5])) if top[5] else 0., "candidate_recall_at_0.5_top10": float(np.mean(top[10])) if top[10] else 0., "candidate_recall_at_0.5_top20": float(np.mean(top[20])) if top[20] else 0., "candidate_recall_at_0.5_top27": float(np.mean(top[27])) if top[27] else 0., "generated_max_iou_mean": float(np.mean(max_i)) if max_i else 0., "generated_max_iou_median": float(np.median(max_i)) if max_i else 0., "quality_ece": ece}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--checkpoint-every", type=int, default=500); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="source")
    args = ap.parse_args(); torch.set_num_threads(2)
    if args.fold not in range(4): raise ValueError(args.fold)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and vis and vis.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"expected physical GPU {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={vis}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu");
    if device.type == "cuda": torch.cuda.set_device(device)
    seed = int(args.seed) + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); cls, roi, alignment = load_aligned_features(rows); manifest = json.loads((OUT / "manifests/source_manifest.json").read_text(encoding="utf-8")); mf = next(x for x in manifest["folds"] if int(x["fold"]) == args.fold)
    def load_split(name):
        z = np.load(mf[name]["path"], allow_pickle=False); return {k: z[k] for k in z.files}
    fit, val = load_split("fit"), load_split("validation"); run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"; ckdir = OUT / "checkpoints"; latest = ckdir / f"{run}_latest.pt"; best_path = ckdir / f"{run}_best.pt"; logp = OUT / "logs" / f"{run}.jsonl"
    if done.exists() and not args.resume: print(json.dumps({"status": "already_done", "done": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"refusing relaunch with marker {marker}")
    marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device), "physical_gpu": args.expected_physical_gpu}) + "\n", encoding="utf-8")
    model = ProposalSourceGenerator(); model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); start = 0; best = -1.; best_step = 0; history = []; rng = np.random.default_rng(seed + 137); steps = 2 if args.smoke else int(args.steps); amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = int(ck.get("global_step", 0)); best = float(ck.get("best_score", -1.)); best_step = int(ck.get("best_step", 0))
    fit_idx, fit_pos, fit_gt = fit["row_idx"], fit["positive"], fit["gt_box"]; n = len(fit_idx); t0 = time.time(); logp.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, steps + 1):
        model.train(); bi = rng.integers(0, n, size=min(args.batch_size, n)); ids = fit_idx[bi]; visual = np.concatenate([cls[ids], roi[ids]], axis=1).astype(np.float32, copy=False); geom = row_geometry(rows, ids); base = geom[:, 1:5]; v = torch.from_numpy(visual).to(device); g = torch.from_numpy(geom).to(device); b = torch.from_numpy(base).to(device); ygt = torch.from_numpy(fit_gt[bi]).to(device); pos = torch.from_numpy(fit_pos[bi]).to(device); opt.zero_grad(set_to_none=True)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
        with ctx:
            boxes, q = model(v, g, b); ious = iou_torch(boxes.float(), ygt.float()); q_target = ious.detach() * pos.unsqueeze(1); q_loss = F.binary_cross_entropy_with_logits(q.float(), q_target.float()); per_box = F.smooth_l1_loss(boxes.float(), ygt.float().unsqueeze(1).expand_as(boxes.float()), reduction="none").mean(-1); box_loss = (per_box.min(dim=1).values * pos).sum() / torch.clamp(pos.sum(), min=1.0); neg_loss = (torch.sigmoid(q).mean(dim=1) * (1.0-pos)).mean(); loss = box_loss + .50*q_loss + .10*neg_loss
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite source loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); rec = {"step": step, "loss": float(loss.detach().cpu()), "box_loss": float(box_loss.detach().cpu()), "quality_loss": float(q_loss.detach().cpu()), "amp": "bf16" if amp_dtype is not None else "fp32"}
        if step % int(args.checkpoint_every) == 0 or step == steps:
            valm = evaluate(model, val["row_idx"], val["positive"], val["gt_box"], rows, cls, roi, device); score = float(valm["candidate_recall_at_0.5_top27"] + .2 * valm["generated_max_iou_mean"] - .1 * valm["quality_ece"]); payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "global_step": step, "best_score": best, "best_step": best_step, "fold": args.fold, "seed": seed, "metadata": metadata(model), "protocol": "trackocd_iclr27_phase26_class_agnostic_proposal_source_head", "feature_alignment": alignment, "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "amp": "bf16" if amp_dtype is not None else "fp32", "fit_manifest": mf["fit"], "val_manifest": mf["validation"]}; atomic_torch(latest, payload); atomic_torch(ckdir / f"{run}_step{step:05d}.pt", payload)
            if score > best: best, best_step = score, step; payload["best_score"], payload["best_step"] = best, best_step; atomic_torch(best_path, payload)
            rec.update({"validation": valm, "validation_score": score, "best_score": best, "elapsed_s": time.time()-t0}); history.append(rec); logp.open("a", encoding="utf-8").write(json.dumps(rec, sort_keys=True) + "\n")
    final = evaluate(model, val["row_idx"], val["positive"], val["gt_box"], rows, cls, roi, device); result = {"protocol": "trackocd_iclr27_phase26_source_generator_training", "fold": args.fold, "tag": args.tag, "seed": seed, "steps": steps, "smoke": bool(args.smoke), "device": str(device), "physical_gpu": args.expected_physical_gpu, "amp": "bf16" if amp_dtype is not None else "fp32", "fit_rows": int(len(fit_idx)), "fit_positive_rows": int(np.sum(fit_pos > .5)), "fit_negative_rows": int(np.sum(fit_pos <= .5)), "validation_rows": int(len(val["row_idx"])), "validation_metrics": final, "best_score": best, "best_step": best_step, "history": history, "checkpoint_best": str(best_path), "checkpoint_latest": str(latest), "marker": str(marker), "done": str(done), "metadata": metadata(model), "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs", "semantic text"]}; atomic_json(OUT / "metrics" / f"{run}.json", result); done.write_text(json.dumps({"fold": args.fold, "steps": steps, "checkpoint": str(best_path), "validation": final}, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"fold": args.fold, "steps": steps, "val_top27": final["candidate_recall_at_0.5_top27"], "best_step": best_step, "checkpoint": str(best_path), "done": str(done)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
