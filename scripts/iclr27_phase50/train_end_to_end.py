#!/usr/bin/env python3
"""TRAIN-only curriculum worker for the Phase50 causal graph.

The frozen Phase26 physical stream supplies key-aligned causal rows.  This
worker trains the semantic/state portion with labels only for loss construction;
category, track and video values never enter ``EndToEndTrackOCD.forward``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase50.end_to_end import EndToEndTrackOCD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase50"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def track_vec(key: str, meta, feats: np.ndarray, prefix: int) -> np.ndarray:
    inds = np.asarray(meta[key]["rows"][: min(prefix, 16)], dtype=np.int64)
    z = feats[inds].mean(0) if len(inds) else np.zeros(feats.shape[1], np.float32)
    return (z / max(float(np.linalg.norm(z)), 1e-8)).astype(np.float32)


def track_sequence(key: str, meta, feats: np.ndarray, prefix: int) -> np.ndarray:
    inds = np.asarray(meta[key]["rows"][: min(prefix, 16)], dtype=np.int64)
    if len(inds) == 0:
        return np.zeros((1, feats.shape[1]), np.float32)
    return feats[inds].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--tag", default="e2e_formal")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expected-physical-gpu", type=int, default=-1)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()
    torch.set_num_threads(1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu):
        raise RuntimeError(f"GPU mapping mismatch: expected physical {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 500000 + int(args.fold)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows, tracks, feats = load_tracks(); meta = track_metadata(rows, tracks)
    manifest = json.loads((ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{args.fold}.json").read_text(encoding="utf-8"))
    fit = [r for r in manifest["records"] if r.get("split") == "fit" and r.get("kind") == "multi_positive_cross_video" and r.get("query_track_key") in meta]
    if not fit:
        raise RuntimeError(f"no TRAIN fit episodes for fold {args.fold}")
    run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"
    comp = OUT / "completion"; ckdir = OUT / "checkpoints"; metdir = OUT / "metrics"; logdir = OUT / "logs"
    marker = comp / f"{run}.launched"; done = comp / f"{run}.done"; latest = ckdir / f"{run}_latest.pt"; best = ckdir / f"{run}_best.pt"; metrics = metdir / f"{run}.json"
    if done.exists():
        print(json.dumps({"fold": args.fold, "run": run, "status": "already_done", "done": str(done)})); return
    if marker.exists():
        raise RuntimeError(f"unit was already launched without completion marker: {marker}; recover explicitly")
    atomic_json(marker, {"phase": 50, "fold": args.fold, "run": run, "pid": os.getpid(), "physical_gpu": args.expected_physical_gpu, "seed": seed, "protocol": "phase50_train_only_causal_graph"})
    model = EndToEndTrackOCD().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    rng = np.random.default_rng(seed + 17)
    steps = 100 if args.smoke else int(args.steps)
    history = []
    model.train()

    for step in range(1, steps + 1):
        rec = fit[int(rng.integers(len(fit)))]
        prefix = PREFIXES[int(rng.integers(len(PREFIXES)))]
        qk = rec["query_track_key"]
        support_keys = [k for k in rec.get("support_track_keys", []) if k in meta]
        hard_key = rec.get("hard_negative_track_key")
        if not support_keys or hard_key not in meta:
            continue
        qseq_np = track_sequence(qk, meta, feats, prefix)
        support_np = np.asarray([track_vec(k, meta, feats, prefix) for k in support_keys], np.float32)
        hard_np = track_vec(hard_key, meta, feats, prefix)
        qseq = torch.from_numpy(qseq_np).unsqueeze(0).to(device)
        support = torch.from_numpy(support_np).unsqueeze(0).to(device)
        hard = torch.from_numpy(hard_np).unsqueeze(0).to(device)
        valid = torch.ones((1, support.shape[1]), dtype=torch.bool, device=device)
        out = model(qseq, support, valid)
        semantic = out["semantic"]
        pos_scores = semantic @ support.squeeze(0).T
        hard_score = (semantic * hard).sum(-1)
        corr = F.relu(0.10 - pos_scores.max(-1).values + hard_score).mean()
        # Prefix consistency uses the same query and support metadata at a
        # second causal prefix; no future frame is passed to either call.
        p2 = 16 if prefix != 16 else 8
        q2 = torch.from_numpy(track_sequence(qk, meta, feats, p2)).unsqueeze(0).to(device)
        out2 = model(q2, support, valid)
        prefix_loss = (1.0 - (semantic.detach() * out2["semantic"]).sum(-1)).abs().mean()
        hard_loss = F.relu(0.05 + hard_score - pos_scores.max(-1).values).mean()
        # Positive TRAIN correspondence is a utility target for the causal
        # action head; this is not a semantic/category input at inference.
        action_target = torch.ones((1,), dtype=torch.long, device=device)
        commit_defer = F.cross_entropy(out["action_logits"], action_target)
        state_persistence = F.relu(1.0 - out["state"][:, 1]).mean()
        raw_preserve = F.mse_loss(semantic, out["raw"].detach())
        temporal = F.mse_loss(out["track"], out["raw"].detach())
        # Proposal/association are frozen Phase26 passthrough in this run; a
        # zero tensor is recorded rather than fabricating GT detector labels.
        proposal_loss = semantic.new_zeros(())
        mot_loss = semantic.new_zeros(())
        loss = corr + 0.50 * hard_loss + 0.10 * prefix_loss + 0.10 * commit_defer + 0.03 * state_persistence + 0.05 * raw_preserve + 0.05 * temporal + proposal_loss + mot_loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "proposal": float(proposal_loss.cpu()), "mot": float(mot_loss.cpu()), "correspondence": float(corr.detach().cpu()), "hard_negative": float(hard_loss.detach().cpu()), "prefix_consistency": float(prefix_loss.detach().cpu()), "state_persistence": float(state_persistence.detach().cpu()), "commit_defer": float(commit_defer.detach().cpu()), "raw_preservation": float(raw_preserve.detach().cpu()), "prefix": prefix})
        if step % max(1, min(args.checkpoint_every, steps)) == 0 or step == steps:
            payload = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "seed": seed, "fold": args.fold, "protocol": "phase50_train_only_causal_graph", "raw_anchor_dim": 768, "proposal_physical_stream": "Phase26 frozen passthrough", "amp": "bf16_autocast_not_required_fp32"}
            atomic_torch(latest, payload); atomic_torch(best, payload)
    atomic_json(metrics, {"phase": 50, "fold": args.fold, "run": run, "steps": steps, "fit_records": len(fit), "history": history, "checkpoint_best": str(best), "checkpoint_latest": str(latest), "protocol": "phase50_train_only_causal_graph", "frozen_proposal": "Phase26", "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "held GT", "future rows/tracks", "category/text/physical/semantic IDs as model inputs"]})
    atomic_json(done, {"phase": 50, "fold": args.fold, "run": run, "steps": steps, "checkpoint": str(best)})
    print(json.dumps({"fold": args.fold, "run": run, "steps": steps, "done": str(done), "checkpoint": str(best)}))


if __name__ == "__main__":
    main()
