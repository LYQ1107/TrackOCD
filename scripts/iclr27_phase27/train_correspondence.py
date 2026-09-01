#!/usr/bin/env python3
"""Train the single preregistered Phase27 causal correspondence encoder.

This script deliberately keeps all writes below ``outputs/iclr27_phase27``.
The CSV/NPZ and the fold split are read-only frozen TRAIN inputs; category and
video values are used only to construct legal positives/negatives and never
are concatenated to the model feature.
"""
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

from src.iclr27_phase27.correspondence import TrackCorrespondenceEncoder, metadata
from src.iclr27_phase27.protocol import (
    CSV_PATH,
    FEAT_PATH,
    FOLD_MANIFEST,
    PHASE26_DECISION,
    by_track,
    load_aligned_features,
    order_key,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase27"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_prefix(
    keys: list[str],
    track_rows: dict[str, list[int]],
    feats: np.ndarray,
    prefix: int,
    max_len: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Build strictly causal prefixes, zero-padding only after the endpoint."""
    arr = np.zeros((len(keys), max_len, feats.shape[1]), dtype=np.float32)
    mask = np.zeros((len(keys), max_len), dtype=bool)
    for i, key in enumerate(keys):
        inds = track_rows[key][: min(int(prefix), max_len)]
        if inds:
            x = feats[np.asarray(inds, dtype=np.int64)]
            arr[i, : len(inds)] = x
            mask[i, : len(inds)] = True
    return arr, mask


@torch.no_grad()
def embed_keys(
    model: TrackCorrespondenceEncoder,
    keys: list[str],
    track_rows: dict[str, list[int]],
    feats: np.ndarray,
    prefix: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    vals: list[np.ndarray] = []
    for start in range(0, len(keys), batch_size):
        x, m = pad_prefix(keys[start : start + batch_size], track_rows, feats, prefix)
        vals.append(model(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)).cpu().numpy())
    return np.concatenate(vals, axis=0) if vals else np.zeros((0, 768), np.float32)


def retrieval(
    model: TrackCorrespondenceEncoder,
    keys: list[str],
    track_rows: dict[str, list[int]],
    feats: np.ndarray,
    cat: dict[str, int],
    video: dict[str, int],
    device: torch.device,
    prefix: int,
) -> dict[str, Any]:
    """Cross-video track retrieval with exact query/candidate denominators."""
    if not keys:
        return {"queries": 0, "r1": 0.0, "r5": 0.0, "map": 0.0, "pairs": 0, "hard_negative_gap": 0.0}
    emb = embed_keys(model, keys, track_rows, feats, prefix, device)
    # One BLAS call avoids thousands of tiny Python/BLAS dispatches.  The
    # candidate masks below are identical to the explicit cross-video rule.
    similarity = emb @ emb.T
    r1: list[float] = []
    r5: list[float] = []
    aps: list[float] = []
    gaps: list[float] = []
    pairs = 0
    for i, key in enumerate(keys):
        candidates = [j for j, other in enumerate(keys) if j != i and video[other] != video[key]]
        positives = [j for j in candidates if cat[keys[j]] == cat[key]]
        negatives = [j for j in candidates if cat[keys[j]] != cat[key]]
        if not positives or not negatives:
            continue
        scores = similarity[i, np.asarray(candidates, dtype=np.int64)]
        order = np.asarray(candidates, dtype=np.int64)[np.argsort(scores)[::-1]]
        positive_set = set(positives)
        hits = np.asarray([int(int(j) in positive_set) for j in order], dtype=np.float32)
        r1.append(float(hits[:1].max(initial=0)))
        r5.append(float(hits[:5].max(initial=0)))
        cumulative = np.cumsum(hits)
        aps.append(float(np.sum(cumulative / (np.arange(len(hits)) + 1) * hits) / max(len(positives), 1)))
        pos_score = float(np.max(similarity[i, np.asarray(positives, dtype=np.int64)]))
        neg_score = float(np.max(similarity[i, np.asarray(negatives, dtype=np.int64)]))
        gaps.append(pos_score - neg_score)
        pairs += len(candidates)
    return {
        "queries": len(r1),
        "r1": float(np.mean(r1)) if r1 else 0.0,
        "r5": float(np.mean(r5)) if r5 else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
        "pairs": int(pairs),
        "hard_negative_gap": float(np.mean(gaps)) if gaps else 0.0,
        "prefix": int(prefix),
    }


def split_tracks(rows: list[dict[str, str]], tracks: dict[str, list[int]], fold: dict[str, Any]):
    cat = {k: int(rows[v[-1]].get("gt_category_id_common", -1)) for k, v in tracks.items()}
    video = {k: int(rows[v[-1]].get("video_id", -1)) for k, v in tracks.items()}
    fit_v = {int(v) for v in fold.get("fit_videos", [])}
    val_v = {int(v) for v in fold.get("validation_videos", [])}
    fit_c = {int(c) for c in fold.get("fit_categories", [])}
    held_c = {int(c) for c in fold.get("held_categories", [])}
    fit_by: dict[int, list[str]] = defaultdict(list)
    val_by: dict[int, list[str]] = defaultdict(list)
    for key in tracks:
        if cat[key] in fit_c and video[key] in fit_v and cat[key] >= 0:
            fit_by[cat[key]].append(key)
        if cat[key] in held_c and video[key] in val_v and cat[key] >= 0:
            val_by[cat[key]].append(key)
    # A legal cross-video positive category must have two distinct TRAIN videos.
    fit_by = {c: sorted(v) for c, v in fit_by.items() if len({video[k] for k in v}) >= 2}
    val_by = {c: sorted(v) for c, v in val_by.items()}
    return cat, video, fit_by, val_by


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-physical-gpu", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", default="correspondence")
    args = parser.parse_args()

    torch.set_num_threads(2)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu):
        raise RuntimeError(f"expected physical GPU {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seed = 20262701 + int(args.fold)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    feats = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
    tracks = by_track(rows)
    manifest = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    fold = next(x for x in manifest["folds"] if int(x["fold"]) == int(args.fold))
    cat, video, fit_by, val_by = split_tracks(rows, tracks, fold)
    fit_categories = sorted(fit_by)
    if len(fit_categories) < 2:
        raise RuntimeError(f"fold {args.fold} has fewer than two legal fit categories")
    val_keys = sorted(k for values in val_by.values() for k in values)

    run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"
    marker = OUT / "completion" / f"{run}.launched"
    done = OUT / "completion" / f"{run}.done"
    latest = OUT / "checkpoints" / f"{run}_latest.pt"
    best_path = OUT / "checkpoints" / f"{run}_best.pt"
    log_path = OUT / "logs" / f"{run}.jsonl"
    metrics_path = OUT / "metrics" / f"{run}.json"
    if done.exists() and not args.resume:
        print(json.dumps({"status": "already_done", "done": str(done)}))
        return
    if marker.exists() and not args.resume:
        raise RuntimeError(f"refusing relaunch with existing marker {marker}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device), "physical_gpu": args.expected_physical_gpu}, sort_keys=True) + "\n")

    model = TrackCorrespondenceEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start = 0
    best_score = -1.0
    best_step = 0
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 31)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    steps = 2 if args.smoke else int(args.steps)

    if args.resume and latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = int(checkpoint.get("global_step", 0))
        best_score = float(checkpoint.get("best_score", -1.0))
        best_step = int(checkpoint.get("best_step", 0))
        if checkpoint.get("numpy_rng") is not None:
            rng.bit_generator.state = checkpoint["numpy_rng"]

    def sample_track(category: int, exclude_video: int | None = None, exclude: set[str] | None = None) -> str:
        choices = [k for k in fit_by[category] if (exclude_video is None or video[k] != exclude_video) and (not exclude or k not in exclude)]
        if not choices:
            choices = [k for k in fit_by[category] if not exclude or k not in exclude]
        return choices[int(rng.integers(len(choices)))]

    # Track-level frozen DINOv2 means provide a cheap hard-negative proposal
    # distribution.  This is only used to choose TRAIN metadata negatives.
    base_emb: dict[str, np.ndarray] = {}
    for key, inds in tracks.items():
        x = feats[np.asarray(inds[:16], dtype=np.int64)]
        v = x.mean(axis=0)
        base_emb[key] = v / max(float(np.linalg.norm(v)), 1e-6)

    def hard_negative(anchor: str, anchor_cat: int) -> str:
        other_categories = [c for c in fit_categories if c != anchor_cat]
        sample_count = min(12, len(other_categories))
        chosen = rng.choice(other_categories, size=sample_count, replace=False)
        candidates: list[str] = []
        for c in np.atleast_1d(chosen):
            candidates.append(sample_track(int(c), exclude_video=video[anchor]))
        return max(candidates, key=lambda k: float(base_emb[anchor] @ base_emb[k]))

    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start + 1, steps + 1):
        anchors: list[str] = []
        pos1: list[str] = []
        pos2: list[str] = []
        negs: list[str] = []
        prefix_lengths: list[int] = []
        for _ in range(int(args.batch_size)):
            c = int(rng.choice(fit_categories))
            akey = sample_track(c)
            choices = [k for k in fit_by[c] if video[k] != video[akey]] or list(fit_by[c])
            p1 = choices[int(rng.integers(len(choices)))]
            p2_choices = [k for k in choices if k != p1] or choices
            p2 = p2_choices[int(rng.integers(len(p2_choices)))]
            anchors.append(akey)
            pos1.append(p1)
            pos2.append(p2)
            negs.append(hard_negative(akey, c))
            prefix_lengths.append(int(rng.choice(PREFIXES)))

        # A batch has a common endpoint so that it remains a compact tensor.
        prefix = int(max(prefix_lengths))
        xa, ma = pad_prefix(anchors, tracks, feats, prefix)
        xp1, mp1 = pad_prefix(pos1, tracks, feats, prefix)
        xp2, mp2 = pad_prefix(pos2, tracks, feats, prefix)
        xn, mn = pad_prefix(negs, tracks, feats, prefix)
        va = torch.from_numpy(xa).to(device)
        vp1 = torch.from_numpy(xp1).to(device)
        vp2 = torch.from_numpy(xp2).to(device)
        vn = torch.from_numpy(xn).to(device)
        tma = torch.from_numpy(ma).to(device)
        tmp1 = torch.from_numpy(mp1).to(device)
        tmp2 = torch.from_numpy(mp2).to(device)
        tmn = torch.from_numpy(mn).to(device)
        # Prefix consistency compares a shorter and longer causal view of the
        # same anchor.  It never materializes rows beyond either endpoint.
        short_prefix = min(PREFIXES, key=lambda p: abs(p - max(1, prefix // 2)))
        xa_short, ma_short = pad_prefix(anchors, tracks, feats, short_prefix)
        vas = torch.from_numpy(xa_short).to(device)
        tmas = torch.from_numpy(ma_short).to(device)

        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
        with context:
            ea = model(va, tma)
            ep1 = model(vp1, tmp1)
            ep2 = model(vp2, tmp2)
            en = model(vn, tmn)
            eas = model(vas, tmas)
            pos_sim = 0.5 * ((ea * ep1).sum(-1) + (ea * ep2).sum(-1))
            neg_sim = (ea * en).sum(-1)
            rank_loss = F.relu(0.20 - pos_sim + neg_sim).mean()
            align_loss = (1.0 - pos_sim).mean()
            consistency_loss = (1.0 - (ea * eas).sum(-1)).mean()
            loss = rank_loss + 0.10 * align_loss + 0.10 * consistency_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite correspondence loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        rec: dict[str, Any] = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "rank_loss": float(rank_loss.detach().cpu()),
            "align_loss": float(align_loss.detach().cpu()),
            "prefix_consistency_loss": float(consistency_loss.detach().cpu()),
            "pos_similarity": float(pos_sim.detach().mean().cpu()),
            "neg_similarity": float(neg_sim.detach().mean().cpu()),
            "grad_norm": grad_norm,
            "sample_prefix": prefix,
        }

        if step % int(args.checkpoint_every) == 0 or step == steps:
            validation = {str(p): retrieval(model, val_keys, tracks, feats, cat, video, device, p) for p in PREFIXES}
            v16 = validation["16"]
            score = float(v16["r1"] + 0.2 * v16["r5"] + 0.1 * v16["map"])
            payload = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "global_step": step,
                "best_score": best_score,
                "best_step": best_step,
                "fold": int(args.fold),
                "seed": seed,
                "metadata": metadata(model),
                "protocol": "trackocd_iclr27_phase27_correspondence_encoder",
                "feature_alignment": alignment,
                "source_csv_sha256": sha256(CSV_PATH),
                "feature_sha256": sha256(FEAT_PATH),
                "fold_manifest_sha256": sha256(FOLD_MANIFEST),
                "phase26_decision_sha256": sha256(PHASE26_DECISION),
                "fit_categories": fit_categories,
                "validation_categories": sorted(val_by),
                "validation_videos": sorted({video[k] for k in val_keys}),
                "amp": "bf16" if amp_dtype is not None else "fp32",
                "numpy_rng": rng.bit_generator.state,
            }
            atomic_torch(latest, payload)
            atomic_torch(OUT / "checkpoints" / f"{run}_step{step:05d}.pt", payload)
            if score > best_score:
                best_score = score
                best_step = step
                payload["best_score"] = best_score
                payload["best_step"] = best_step
                atomic_torch(best_path, payload)
            rec.update({"validation": validation, "validation_score": score, "best_score": best_score, "best_step": best_step, "elapsed_s": time.time() - started})
        history.append(rec)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    final_validation = {str(p): retrieval(model, val_keys, tracks, feats, cat, video, device, p) for p in PREFIXES}
    result = {
        "protocol": "trackocd_iclr27_phase27_correspondence_training",
        "fold": int(args.fold),
        "tag": args.tag,
        "seed": seed,
        "steps": steps,
        "smoke": bool(args.smoke),
        "device": str(device),
        "physical_gpu": args.expected_physical_gpu,
        "amp": "bf16" if amp_dtype is not None else "fp32",
        "fit_tracklets": int(sum(len(v) for v in fit_by.values())),
        "fit_categories": fit_categories,
        "validation_tracklets": len(val_keys),
        "validation_categories": sorted(val_by),
        "validation_metrics": final_validation,
        "best_score": best_score,
        "best_step": best_step,
        "history": history,
        "checkpoint_best": str(best_path),
        "checkpoint_latest": str(latest),
        "marker": str(marker),
        "done": str(done),
        "metadata": metadata(model),
        "source_csv_sha256": sha256(CSV_PATH),
        "feature_sha256": sha256(FEAT_PATH),
        "fold_manifest_sha256": sha256(FOLD_MANIFEST),
        "phase26_decision_sha256": sha256(PHASE26_DECISION),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs", "semantic text", "held GT as model input"],
    }
    atomic_json(metrics_path, result)
    done.write_text(json.dumps({"fold": int(args.fold), "steps": steps, "checkpoint": str(best_path), "validation": final_validation["16"]}, sort_keys=True) + "\n")
    print(json.dumps({"fold": int(args.fold), "steps": steps, "val_r1": final_validation["16"]["r1"], "val_r5": final_validation["16"]["r5"], "best_step": best_step, "done": str(done)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
