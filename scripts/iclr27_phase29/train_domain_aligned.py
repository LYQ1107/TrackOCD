#!/usr/bin/env python3
"""Train the one registered Phase29 domain-aligned residual adapter."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase29.protocol import CSV_PATH, FEAT_PATH, FOLD_MANIFEST, PHASE26_DECISION, PHASE28_DECISION, by_track, load_aligned_features, order_key
from src.iclr27_phase29.representation import DomainAlignedResidualEncoder, metadata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase29"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def pad_prefix(keys: list[str], tracks: dict[str, list[int]], feats: np.ndarray, prefix: int, max_len: int = 16) -> tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((len(keys), max_len, feats.shape[1]), dtype=np.float32)
    mask = np.zeros((len(keys), max_len), dtype=bool)
    for i, key in enumerate(keys):
        inds = tracks[key][: min(int(prefix), max_len)]
        if inds:
            arr[i, :len(inds)] = feats[np.asarray(inds, dtype=np.int64)]
            mask[i, :len(inds)] = True
    return arr, mask


@torch.no_grad()
def embed_keys(model: DomainAlignedResidualEncoder, keys: list[str], tracks: dict[str, list[int]], feats: np.ndarray, prefix: int, device: torch.device, batch_size: int = 512) -> np.ndarray:
    out: list[np.ndarray] = []
    for start in range(0, len(keys), batch_size):
        x, m = pad_prefix(keys[start:start + batch_size], tracks, feats, prefix)
        out.append(model(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, feats.shape[1]), np.float32)


def baseline_embeddings(keys: list[str], tracks: dict[str, list[int]], feats: np.ndarray, prefix: int) -> np.ndarray:
    out = np.zeros((len(keys), feats.shape[1]), dtype=np.float32)
    for i, key in enumerate(keys):
        x = feats[np.asarray(tracks[key][:min(int(prefix), 16)], dtype=np.int64)]
        v = x.mean(axis=0); out[i] = v / max(float(np.linalg.norm(v)), 1e-6)
    return out


def retrieval_from_embeddings(keys: list[str], emb: np.ndarray, cat: dict[str, int], video: dict[str, int], prefix: int) -> dict[str, Any]:
    if not keys:
        return {"prefix": int(prefix), "queries": 0, "r1": 0.0, "r5": 0.0, "map": 0.0, "hard_negative_gap": 0.0, "category_macro": 0.0, "video_macro": 0.0}
    sim = emb @ emb.T
    cats = np.asarray([cat[k] for k in keys], dtype=np.int64)
    videos = np.asarray([video[k] for k in keys], dtype=np.int64)
    indices = np.arange(len(keys), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for i, key in enumerate(keys):
        candidate_mask = (videos != videos[i]) & (indices != i)
        positive_mask = candidate_mask & (cats == cats[i])
        negative_mask = candidate_mask & (cats != cats[i])
        ci = indices[candidate_mask]; positives = indices[positive_mask]; negatives = indices[negative_mask]
        if positives.size == 0 or negatives.size == 0: continue
        order = ci[np.argsort(sim[i, ci])[::-1]]
        hit = (cats[order] == cats[i]).astype(np.float32)
        cum = np.cumsum(hit)
        rows.append({"category": int(cats[i]), "video": int(videos[i]), "r1": float(hit[:1].max(initial=0)), "r5": float(hit[:5].max(initial=0)), "map": float(np.sum(cum / (np.arange(len(hit)) + 1) * hit) / max(int(len(positives)), 1)), "hard_negative_gap": float(np.max(sim[i, positives]) - np.max(sim[i, negatives]))})
    mean = lambda field, vals: float(np.mean([r[field] for r in vals])) if vals else 0.0
    cats = defaultdict(list); vids = defaultdict(list)
    for r in rows: cats[r["category"]].append(r); vids[r["video"]].append(r)
    return {"prefix": int(prefix), "queries": len(rows), "r1": mean("r1", rows), "r5": mean("r5", rows), "map": mean("map", rows), "hard_negative_gap": mean("hard_negative_gap", rows), "category_macro": float(np.mean([mean("r1", v) for v in cats.values()])) if cats else 0.0, "video_macro": float(np.mean([mean("r1", v) for v in vids.values()])) if vids else 0.0}


def split_tracks(rows: list[dict[str, str]], tracks: dict[str, list[int]], fold: dict[str, Any]):
    cat = {k: int(rows[v[-1]].get("gt_category_id_common", -1)) for k, v in tracks.items()}
    video = {k: int(rows[v[-1]].get("video_id", -1)) for k, v in tracks.items()}
    fit_v = {int(v) for v in fold.get("fit_videos", [])}; val_v = {int(v) for v in fold.get("validation_videos", [])}
    fit_c = {int(c) for c in fold.get("fit_categories", [])}; held_c = {int(c) for c in fold.get("held_categories", [])}
    fit_by: dict[int, list[str]] = defaultdict(list); val_by: dict[int, list[str]] = defaultdict(list)
    for key in tracks:
        if cat[key] in fit_c and video[key] in fit_v and cat[key] >= 0: fit_by[cat[key]].append(key)
        if cat[key] in held_c and video[key] in val_v and cat[key] >= 0: val_by[cat[key]].append(key)
    fit_by = {c: sorted(v) for c, v in fit_by.items() if len({video[k] for k in v}) >= 2}
    return cat, video, fit_by, {c: sorted(v) for c, v in val_by.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="domain_aligned")
    args = ap.parse_args()
    torch.set_num_threads(2)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu):
        raise RuntimeError(f"expected physical GPU {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    seed = 20262901 + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with CSV_PATH.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    feats = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    tracks = by_track(rows); manifest = json.loads(FOLD_MANIFEST.read_text()); fold = next(x for x in manifest["folds"] if int(x["fold"]) == int(args.fold))
    cat, video, fit_by, val_by = split_tracks(rows, tracks, fold); fit_categories = sorted(fit_by); val_keys = sorted(k for v in val_by.values() for k in v)
    if len(fit_categories) < 2: raise RuntimeError(f"fold {args.fold} has fewer than two fit categories")
    run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"
    latest = OUT / "checkpoints" / f"{run}_latest.pt"; best_path = OUT / "checkpoints" / f"{run}_best.pt"; log_path = OUT / "logs" / f"{run}.jsonl"; metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists() and not args.resume: print(json.dumps({"status": "already_done", "done": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"refusing relaunch with existing marker {marker}")
    if not marker.exists(): marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device), "physical_gpu": args.expected_physical_gpu}, sort_keys=True) + "\n")
    model = DomainAlignedResidualEncoder().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 31); start = 0; steps = 2 if args.smoke else int(args.steps); history: list[dict[str, Any]] = []
    # Save an identity-initialized baseline checkpoint.  If training does not
    # beat the disjoint validation baseline, selection falls back to identity.
    base_val = {str(p): retrieval_from_embeddings(val_keys, baseline_embeddings(val_keys, tracks, feats, p), cat, video, p) for p in PREFIXES}
    base_score = float(base_val["16"]["r1"] + 0.2 * base_val["16"]["r5"] + 0.1 * base_val["16"]["map"])
    best_score = base_score; best_step = 0; amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    def payload(step: int, score: float, bstep: int):
        return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "global_step": step, "best_score": score, "best_step": bstep, "fold": int(args.fold), "seed": seed, "metadata": metadata(model), "protocol": "trackocd_iclr27_phase29_domain_aligned_representation", "feature_alignment": alignment, "source_csv_sha256": sha(CSV_PATH), "feature_sha256": sha(FEAT_PATH), "fold_manifest_sha256": sha(FOLD_MANIFEST), "phase26_decision_sha256": sha(PHASE26_DECISION), "phase28_decision_sha256": sha(PHASE28_DECISION), "fit_categories": fit_categories, "validation_categories": sorted(val_by), "validation_videos": sorted({video[k] for k in val_keys}), "amp": "bf16" if amp_dtype is not None else "fp32", "numpy_rng": rng.bit_generator.state}
    atomic_torch(best_path, payload(0, best_score, best_step)); atomic_torch(latest, payload(0, best_score, best_step))
    started = time.time(); log_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        anchors: list[str] = []; pos1: list[str] = []; pos2: list[str] = []; negs: list[str] = []; prefixes: list[int] = []
        for _ in range(int(args.batch_size)):
            c = int(rng.choice(fit_categories)); a = fit_by[c][int(rng.integers(len(fit_by[c])))]
            choices = [k for k in fit_by[c] if video[k] != video[a]] or fit_by[c]
            p1 = choices[int(rng.integers(len(choices)))]; p2c = [k for k in choices if k != p1] or choices; p2 = p2c[int(rng.integers(len(p2c)))]
            other = [x for x in fit_categories if x != c]; nc = int(rng.choice(other)); neg_choices = [k for k in fit_by[nc] if video[k] != video[a]] or fit_by[nc]; n = neg_choices[int(rng.integers(len(neg_choices)))]
            anchors.append(a); pos1.append(p1); pos2.append(p2); negs.append(n); prefixes.append(int(rng.choice(PREFIXES)))
        prefix = int(max(prefixes)); xa, ma = pad_prefix(anchors, tracks, feats, prefix); xp1, mp1 = pad_prefix(pos1, tracks, feats, prefix); xp2, mp2 = pad_prefix(pos2, tracks, feats, prefix); xn, mn = pad_prefix(negs, tracks, feats, prefix)
        short = min(PREFIXES, key=lambda p: abs(p - max(1, prefix // 2))); xas, mas = pad_prefix(anchors, tracks, feats, short)
        va, vp1, vp2, vn = [torch.from_numpy(x).to(device) for x in (xa, xp1, xp2, xn)]; ma_t, mp1_t, mp2_t, mn_t, mas_t = [torch.from_numpy(x).to(device) for x in (ma, mp1, mp2, mn, mas)]; vas = torch.from_numpy(xas).to(device)
        optimizer.zero_grad(set_to_none=True); ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
        with ctx:
            ea, ep1, ep2, en, eas = model(va, ma_t), model(vp1, mp1_t), model(vp2, mp2_t), model(vn, mn_t), model(vas, mas_t)
            pos = torch.stack([ep1, ep2], 1); all_candidates = torch.cat([pos, en[:, None, :]], 1); logits = torch.einsum("bd,bkd->bk", ea, all_candidates) / 0.10
            # Both positive columns are valid; the hard negative is explicit.
            info_nce = (-torch.logsumexp(logits[:, :2], 1) + torch.logsumexp(logits, 1)).mean()
            pos_sim = (ea[:, None, :] * pos).sum(-1).mean(1); neg_sim = (ea * en).sum(-1)
            rank = F.relu(0.20 - pos_sim + neg_sim).mean(); consistency = (1.0 - (ea * eas).sum(-1)).mean()
            residual_norm = model.residual.weight.pow(2).mean(); loss = info_nce + 0.25 * rank + 0.10 * consistency + 0.01 * residual_norm
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
        rec: dict[str, Any] = {"step": step, "loss": float(loss.detach().cpu()), "info_nce": float(info_nce.detach().cpu()), "rank_loss": float(rank.detach().cpu()), "prefix_consistency_loss": float(consistency.detach().cpu()), "residual_norm": float(residual_norm.detach().cpu()), "positive_similarity": float(pos_sim.detach().mean().cpu()), "negative_similarity": float(neg_sim.detach().mean().cpu()), "grad_norm": grad, "sample_prefix": prefix}
        if step % int(args.checkpoint_every) == 0 or step == steps:
            val = {str(p): retrieval_from_embeddings(val_keys, embed_keys(model, val_keys, tracks, feats, p, device), cat, video, p) for p in PREFIXES}; score = float(val["16"]["r1"] + 0.2 * val["16"]["r5"] + 0.1 * val["16"]["map"]); cp = payload(step, best_score, best_step); atomic_torch(latest, cp); atomic_torch(OUT / "checkpoints" / f"{run}_step{step:05d}.pt", cp)
            if score > best_score + 1e-8:
                best_score, best_step = score, step; atomic_torch(best_path, payload(step, best_score, best_step))
            rec.update({"validation": val, "validation_score": score, "best_score": best_score, "best_step": best_step, "elapsed_s": time.time() - started})
        history.append(rec)
        with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
    final_val = {str(p): retrieval_from_embeddings(val_keys, embed_keys(model, val_keys, tracks, feats, p, device), cat, video, p) for p in PREFIXES}
    result = {"protocol": "trackocd_iclr27_phase29_domain_aligned_training", "fold": int(args.fold), "tag": args.tag, "seed": seed, "steps": steps, "smoke": bool(args.smoke), "device": str(device), "physical_gpu": args.expected_physical_gpu, "amp": "bf16" if amp_dtype is not None else "fp32", "fit_tracklets": int(sum(len(v) for v in fit_by.values())), "fit_categories": fit_categories, "validation_tracklets": len(val_keys), "validation_categories": sorted(val_by), "validation_metrics": final_val, "baseline_validation": base_val, "baseline_score": base_score, "best_score": best_score, "best_step": best_step, "history": history, "checkpoint_best": str(best_path), "checkpoint_latest": str(latest), "marker": str(marker), "done": str(done), "metadata": metadata(model), "source_csv_sha256": sha(CSV_PATH), "feature_sha256": sha(FEAT_PATH), "fold_manifest_sha256": sha(FOLD_MANIFEST), "phase26_decision_sha256": sha(PHASE26_DECISION), "phase28_decision_sha256": sha(PHASE28_DECISION)}
    atomic_json(metrics_path, result); done.write_text(json.dumps({"fold": args.fold, "steps": steps, "metrics": str(metrics_path), "best": str(best_path)}, sort_keys=True) + "\n")
    print(json.dumps({"fold": args.fold, "best_step": best_step, "best_score": best_score, "metrics": str(metrics_path)}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
