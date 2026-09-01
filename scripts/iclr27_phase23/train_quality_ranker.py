#!/usr/bin/env python3
"""Train one fold of the Phase23 fixed-pool candidate quality ranker."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase23.protocol import (CSV_PATH, FEAT_PATH, P22_MANIFEST,
    CENTER_SHIFTS, SCALE_FACTORS, by_track, fval, load_aligned_features,
    normalized_gt, raw_box, track_positions)
from src.iclr27_phase23.ranker import CandidateQualityRanker, metadata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase23"
GEOM_FIELDS = ("score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou")
TRANSFORM_META = np.asarray([[s, dx, dy] for s in SCALE_FACTORS for dx in CENTER_SHIFTS for dy in CENTER_SHIFTS], dtype=np.float32)
TRANSFORMS = len(TRANSFORM_META)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def candidate_arrays(rows: list[dict[str, str]], idx: int, tracks: dict[str, list[int]], positions: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = f"v{int(rows[idx]['video_id'])}:p{int(rows[idx]['track_id'])}"; inds = tracks[key]; pos = positions[idx]; hist = inds[max(0, pos - 4 + 1):pos + 1]
    base = np.asarray([raw_box(rows[j]) for j in hist], dtype=np.float32); vals = []
    for tid, (scale, dx, dy) in enumerate(TRANSFORM_META):
        cx = (base[:, 0] + base[:, 2])*.5 + dx*(base[:, 2] - base[:, 0]); cy = (base[:, 1] + base[:, 3])*.5 + dy*(base[:, 3] - base[:, 1]); bw = np.maximum(0., base[:, 2] - base[:, 0])*scale; bh = np.maximum(0., base[:, 3] - base[:, 1])*scale
        vals.append(np.stack([cx-bw*.5, cy-bh*.5, cx+bw*.5, cy+bh*.5], axis=1))
    boxes = np.clip(np.concatenate(vals, axis=0), 0., 1.); parent = np.tile(np.asarray(hist, dtype=np.int32), TRANSFORMS); transform = np.repeat(np.arange(TRANSFORMS, dtype=np.int16), len(hist)); assigned = np.asarray([str(rows[j].get("assigned", "0")) == "1" for j in parent], dtype=bool)
    return boxes, parent, transform, assigned


def iou_vec(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1]); x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3]); inter = np.maximum(0., x2-x1)*np.maximum(0., y2-y1); aa = np.maximum(0., boxes[:,2]-boxes[:,0])*np.maximum(0., boxes[:,3]-boxes[:,1]); ab = max(0., gt[2]-gt[0])*max(0., gt[3]-gt[1]); return inter / np.maximum(aa+ab-inter, 1e-8)


def build_index(rows: list[dict[str, str]], indices: list[int], tracks: dict[str, list[int]], positions: dict[int, int], path: Path) -> dict[str, Any]:
    row_a: list[int] = []; parent_a: list[int] = []; box_a: list[np.ndarray] = []; meta_a: list[np.ndarray] = []; label_a: list[np.ndarray] = []; assigned_a: list[np.ndarray] = []; n_rows = 0
    for idx in indices:
        gt = normalized_gt(rows[idx])
        if gt is None: continue
        boxes, parent, transform, assigned = candidate_arrays(rows, idx, tracks, positions); labels = iou_vec(boxes, np.asarray(gt, dtype=np.float32)); row_a.append(np.full(len(boxes), idx, dtype=np.int32)); parent_a.append(parent); box_a.append(boxes); meta_a.append(TRANSFORM_META[transform]); label_a.append(labels.astype(np.float32)); assigned_a.append(assigned); n_rows += 1
    if not row_a: raise RuntimeError("empty GT index")
    arrays = {"row_idx": np.concatenate(row_a), "parent_idx": np.concatenate(parent_a), "candidate_box": np.concatenate(box_a), "transform_meta": np.concatenate(meta_a), "label_iou": np.concatenate(label_a), "parent_assigned": np.concatenate(assigned_a)}
    atomic_npz(path, **arrays)
    return {"rows_with_gt": n_rows, "candidate_samples": int(len(arrays["label_iou"])), "path": str(path), "sha256": sha256(path)}


def feature_batch(rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, idx: np.ndarray, parent: np.ndarray, boxes: np.ndarray, meta_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visual = np.concatenate([cls[parent], roi[parent]], axis=1).astype(np.float32, copy=False)
    geom = np.asarray([[fval(rows[int(j)], k) for k in GEOM_FIELDS] for j in parent], dtype=np.float32)
    return visual, np.concatenate([geom, boxes.astype(np.float32), meta_arr.astype(np.float32)], axis=1)


def evaluate(model: CandidateQualityRanker, arr: dict[str, np.ndarray], rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, device: torch.device, top_k: int = 5) -> dict[str, Any]:
    model.eval(); n = len(arr["label_iou"]); scores = np.empty(n, dtype=np.float32); bs = 8192
    with torch.no_grad():
        for st in range(0, n, bs):
            sl = slice(st, min(st+bs, n)); v, g = feature_batch(rows, cls, roi, arr["row_idx"][sl], arr["parent_idx"][sl], arr["candidate_box"][sl], arr["transform_meta"][sl]); scores[sl] = model(torch.from_numpy(v).to(device), torch.from_numpy(g).to(device)).float().cpu().numpy()
    groups = arr["row_idx"]; recalls = []; top1 = []; topk = []; brier = []; group_count = 0
    for rid in np.unique(groups):
        ix = np.flatnonzero(groups == rid); order = ix[np.argsort(scores[ix])[::-1]]; lab = arr["label_iou"][order]; ass = arr["parent_assigned"][order];
        recalls.append(float(np.any(ass[:top_k] & (lab[:top_k] >= .5)))); top1.append(float(lab[0])); topk.append(float(np.max(lab[:top_k], initial=0.))); brier.extend((1/(1+np.exp(-scores[ix])) - arr["label_iou"][ix]).tolist()); group_count += 1
    # Expected calibration error over ten fixed probability bins.
    probs = 1/(1+np.exp(-scores)); ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (probs >= lo) & (probs <= hi if hi == 1 else probs < hi)
        if np.any(m): ece += float(m.mean()) * abs(float(probs[m].mean()) - float(arr["label_iou"][m].mean()))
    return {"rows_with_gt": group_count, "candidate_samples": n, "candidate_recall_at_0.5_top1": float(np.mean([x >= .5 for x in top1])) if top1 else 0., "candidate_recall_at_0.5_top5": float(np.mean(recalls)) if recalls else 0., "top1_true_iou_mean": float(np.mean(top1)) if top1 else 0., "top5_true_iou_mean": float(np.mean(topk)) if topk else 0., "top1_true_iou_median": float(np.median(top1)) if top1 else 0., "brier_soft_iou": float(np.mean(np.square(brier))) if brier else 0., "ece_soft_iou": ece}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--checkpoint-every", type=int, default=500); ap.add_argument("--seed", type=int, default=20260828); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="ordered", help="independent artifact tag for a corrected candidate ordering"); args = ap.parse_args()
    if args.fold not in range(4): raise ValueError(args.fold)
    torch.set_num_threads(2); device = torch.device(args.device if torch.cuda.is_available() else "cpu");
    if device.type == "cuda": torch.cuda.set_device(device)
    seed = int(args.seed) + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); cls, roi, alignment = load_aligned_features(rows); manifest = json.loads(P22_MANIFEST.read_text()); fr = next(x for x in manifest["folds"] if int(x["fold"]) == args.fold); fit_v, val_v = set(map(int, fr["fit_videos"])), set(map(int, fr["validation_videos"])); fit_c, held_c = set(map(int, fr["fit_categories"])), set(map(int, fr["held_categories"])); tracks = by_track(rows); positions = track_positions(rows, tracks)
    fit_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in fit_v and int(r.get("gt_category_id_common", -1)) in fit_c and normalized_gt(r) is not None]; val_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in val_v and int(r.get("gt_category_id_common", -1)) in held_c and normalized_gt(r) is not None]
    stem = f"ranker_{args.tag}_" if str(args.tag).strip() else "ranker_"
    run = f"{stem}smoke_f{args.fold}" if args.smoke else f"{stem}f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"; ckdir = OUT / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True); logp = OUT / "logs" / f"train_{run}.jsonl"; idxdir = OUT / "manifests"; idxpath = idxdir / f"ranker_candidates_f{args.fold}.npz"
    if done.exists() and not args.resume: print(json.dumps({"status": "already_done", "done": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"refusing relaunch with marker {marker}")
    marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device)}) + "\n", encoding="utf-8")
    fit_info = build_index(rows, fit_idx, tracks, positions, idxdir / f"ranker_{args.tag}_fit_f{args.fold}.npz"); val_info = build_index(rows, val_idx, tracks, positions, idxdir / f"ranker_{args.tag}_val_f{args.fold}.npz"); fit = {k: v for k, v in np.load(fit_info["path"], allow_pickle=False).items()}; val = {k: v for k, v in np.load(val_info["path"], allow_pickle=False).items()}; model = CandidateQualityRanker(); model.to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); start = 0; best = -1.; best_step = 0; history: list[dict[str, Any]] = []; latest = ckdir / f"{run}_latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = int(ck.get("global_step", 0)); best = float(ck.get("best_score", -1.)); best_step = int(ck.get("best_step", 0))
    amp = torch.bfloat16 if device.type == "cuda" else None; rng = np.random.default_rng(seed + 91); n = len(fit["label_iou"]); t0 = time.time(); logp.parent.mkdir(parents=True, exist_ok=True); steps = 2 if args.smoke else int(args.steps)
    for step in range(start + 1, steps + 1):
        model.train(); bi = rng.integers(0, n, size=min(args.batch_size, n)); v_np, g_np = feature_batch(rows, cls, roi, fit["row_idx"][bi], fit["parent_idx"][bi], fit["candidate_box"][bi], fit["transform_meta"][bi]); v, g = torch.from_numpy(v_np).to(device), torch.from_numpy(g_np).to(device); y = torch.from_numpy(fit["label_iou"][bi]).to(device); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp) if amp is not None else torch.autocast(device_type="cpu", enabled=False): logits = model(v, g); loss_q = F.binary_cross_entropy_with_logits(logits, y) + .25 * F.smooth_l1_loss(torch.sigmoid(logits), y)
        # Hard-negative ranking is restricted to candidates from the same
        # current row, preserving candidate-set semantics.
        rank_terms = []
        for rid in np.unique(fit["row_idx"][bi]):
            loc = np.flatnonzero(fit["row_idx"][bi] == rid)
            if len(loc) < 2: continue
            pos, neg = loc[np.argmax(fit["label_iou"][bi[loc]])], loc[np.argmin(fit["label_iou"][bi[loc]])]
            if fit["label_iou"][bi[pos]] > fit["label_iou"][bi[neg]] + .1: rank_terms.append(F.relu(.25 - logits[pos] + logits[neg]))
        loss_rank = torch.stack(rank_terms).mean() if rank_terms else logits.sum() * 0.; loss = loss_q + .25 * loss_rank
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite ranker loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); rec = {"step": step, "loss": float(loss.detach().cpu()), "loss_quality": float(loss_q.detach().cpu()), "loss_rank": float(loss_rank.detach().cpu()), "amp": "bf16" if amp is not None else "fp32"}
        if step % int(args.checkpoint_every) == 0 or step == steps:
            valm = evaluate(model, val, rows, cls, roi, device); rec["validation"] = valm; score = float(valm["candidate_recall_at_0.5_top5"] + .2 * valm["top5_true_iou_mean"] - .1 * valm["ece_soft_iou"]); payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(), "global_step": step, "best_score": best, "best_step": best_step, "fold": args.fold, "seed": seed, "metadata": metadata(model), "protocol": "trackocd_iclr27_phase23_stage2a_candidate_quality_ranker", "fit_index": fit_info, "val_index": val_info, "feature_alignment": alignment, "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "amp": "bf16" if amp is not None else "fp32"}; atomic_torch(latest, payload); atomic_torch(ckdir / f"{run}_step{step:05d}.pt", payload); 
            if score > best: best, best_step = score, step; payload["best_score"], payload["best_step"] = best, best_step; atomic_torch(ckdir / f"{run}_best.pt", payload)
            rec["validation_score"], rec["best_score"], rec["elapsed_s"] = score, best, time.time() - t0; history.append(rec)
            with logp.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
    final = evaluate(model, val, rows, cls, roi, device); result = {"protocol": "trackocd_iclr27_phase23_stage2a_candidate_quality_ranker", "fold": args.fold, "tag": args.tag, "candidate_ordering": "transform-major with parent metadata transform-major", "seed": seed, "steps": steps, "smoke": bool(args.smoke), "device": str(device), "amp": "bf16" if amp is not None else "fp32", "fit_rows": fit_info["rows_with_gt"], "fit_candidates": fit_info["candidate_samples"], "validation_rows": val_info["rows_with_gt"], "validation_candidates": val_info["candidate_samples"], "validation_metrics": final, "best_score": best, "best_step": best_step, "history": history, "checkpoint_best": str(ckdir / f"{run}_best.pt"), "checkpoint_latest": str(latest), "marker": str(marker), "done": str(done), "feature_alignment": alignment, "forbidden_inputs": metadata(model)["forbidden"], "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}; atomic_json(OUT / "metrics" / f"{run}.json", result); done.write_text(json.dumps({"fold": args.fold, "steps": steps, "checkpoint": str(ckdir / f"{run}_best.pt"), "validation": final}, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps({"fold": args.fold, "steps": steps, "fit_candidates": fit_info["candidate_samples"], "val_candidates": val_info["candidate_samples"], "val_top5_recall": final["candidate_recall_at_0.5_top5"], "best_step": best_step, "done": str(done)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
