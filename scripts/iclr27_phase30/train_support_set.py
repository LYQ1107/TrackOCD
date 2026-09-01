#!/usr/bin/env python3
"""Train the single Phase30 support/query set correspondence encoder."""

from __future__ import annotations

import argparse
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

from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder, metadata


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase30"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def make_batch(records: list[dict[str, Any]], meta: dict[str, dict[str, Any]], feats: np.ndarray, prefix: int, device: torch.device):
    b = len(records); smax = max(1, max(len(r.get("support_track_keys", [])) for r in records))
    q = np.zeros((b, 16, feats.shape[1]), np.float32); qm = np.zeros((b, 16), bool)
    ss = np.zeros((b, smax, 16, feats.shape[1]), np.float32); sm = np.zeros((b, smax, 16), bool); setm = np.zeros((b, smax), bool)
    h = np.zeros((b, 16, feats.shape[1]), np.float32); hm = np.zeros((b, 16), bool)
    null = np.zeros((b,), np.float32); positive = np.zeros((b,), np.float32)
    for i, r in enumerate(records):
        qk = r["query_track_key"]; qi = meta[qk]["rows"][: min(prefix, 16)]; q[i, :len(qi)] = feats[np.asarray(qi)]; qm[i, :len(qi)] = True
        for j, sk in enumerate(r.get("support_track_keys", [])[:smax]):
            if sk not in meta: continue
            si = meta[sk]["rows"][: min(prefix, 16)]; ss[i, j, :len(si)] = feats[np.asarray(si)]; sm[i, j, :len(si)] = True; setm[i, j] = True
        hk = r.get("hard_negative_track_key")
        if hk in meta:
            hi = meta[hk]["rows"][: min(prefix, 16)]; h[i, :len(hi)] = feats[np.asarray(hi)]; hm[i, :len(hi)] = True
        null[i] = float(bool(r.get("null_no_match", False))); positive[i] = float(r.get("kind") == "multi_positive_cross_video")
    return (torch.from_numpy(q).to(device), torch.from_numpy(qm).to(device), torch.from_numpy(ss).to(device), torch.from_numpy(sm).to(device), torch.from_numpy(setm).to(device), torch.from_numpy(h).to(device), torch.from_numpy(hm).to(device), torch.from_numpy(null).to(device), torch.from_numpy(positive).to(device))


def loss_step(model: SupportSetCorrespondenceEncoder, batch, amp_enabled: bool) -> tuple[torch.Tensor, dict[str, float]]:
    q, qm, ss, sm, setm, h, hm, null_target, pos_target = batch
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp_enabled else torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        out = model(q, qm, ss, sm, setm)
        h_emb = model.encode_track(h, hm)
        hard_score = torch.sum(out["query_embedding"] * h_emb, dim=-1)
        valid = setm.bool(); positive_mask = valid & (pos_target[:, None] > 0.5)
        pos_scores = out["pair_scores"].masked_fill(~positive_mask, -1e4)
        all_scores = torch.cat([out["pair_scores"].masked_fill(~valid, -1e4), hard_score[:, None]], dim=1)
        has_pos = positive_mask.any(dim=1)
        denom = torch.logsumexp(all_scores, dim=1)
        pos_lse = torch.logsumexp(pos_scores, dim=1)
        contrastive = (-(pos_lse - denom)[has_pos]).mean() if bool(has_pos.any()) else torch.zeros((), device=q.device)
        max_pos = pos_scores.max(dim=1).values
        ranking = F.relu(0.15 - max_pos + hard_score)[has_pos].mean() if bool(has_pos.any()) else torch.zeros((), device=q.device)
        calibration = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
        # A second causal prefix is used only for consistency; no future rows
        # are introduced (the caller supplies a prefix <=16).
        q_short = q.clone(); qm_short = qm.clone(); short_len = max(1, int(qm.sum(1).float().median().item()) // 2); qm_short[:, short_len:] = False; q_short[:, short_len:] = 0
        short_emb = model.encode_track(q_short, qm_short)
        consistency = (1.0 - torch.sum(short_emb * out["query_embedding"], dim=-1)).mean()
        complexity = sum((p.float() ** 2).mean() for p in model.parameters())
        total = contrastive + 0.5 * ranking + 0.2 * calibration + 0.2 * consistency + 0.01 * complexity
    return total, {"loss": float(total.detach().cpu()), "contrastive": float(contrastive.detach().cpu()), "ranking": float(ranking.detach().cpu()), "calibration": float(calibration.detach().cpu()), "consistency": float(consistency.detach().cpu())}


@torch.no_grad()
def validate(model: SupportSetCorrespondenceEncoder, records: list[dict[str, Any]], meta: dict[str, dict[str, Any]], feats: np.ndarray, device: torch.device, prefix: int = 16) -> dict[str, float]:
    if not records: return {"episodes": 0, "positive_episodes": 0, "episode_accuracy": 0.0, "null_rejection": 0.0, "mean_margin": 0.0}
    margins = []; correct = []; null_ok = []; pos_count = 0
    for start in range(0, len(records), 128):
        chunk = records[start:start + 128]; q, qm, ss, sm, setm, h, hm, null_target, pos_target = make_batch(chunk, meta, feats, prefix, device)
        out = model(q, qm, ss, sm, setm); h_emb = model.encode_track(h, hm); hs = torch.sum(out["query_embedding"] * h_emb, dim=-1)
        for i, r in enumerate(chunk):
            mx = float(out["pair_scores"][i].masked_fill(~setm[i].bool(), -1e4).max().cpu()); margin = mx - float(hs[i].cpu()); margins.append(margin)
            if r.get("null_no_match", False): null_ok.append(float(torch.sigmoid(out["null_logit"][i]).cpu() > 0.5))
            else: pos_count += 1; correct.append(float(margin > 0.0))
    return {"episodes": len(records), "positive_episodes": pos_count, "episode_accuracy": float(np.mean(correct)) if correct else 0.0, "null_rejection": float(np.mean(null_ok)) if null_ok else 0.0, "mean_margin": float(np.mean(margins)) if margins else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--expected-physical-gpu", type=int, default=-1); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--checkpoint-every", type=int, default=500); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--tag", default="interface_formal"); ap.add_argument("--resume", action="store_true"); args = ap.parse_args()
    torch.set_num_threads(1)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.expected_physical_gpu >= 0 and visible and visible.split(",")[0].strip() != str(args.expected_physical_gpu): raise RuntimeError(f"expected physical GPU {args.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={visible}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu");
    if device.type == "cuda": torch.cuda.set_device(device)
    seed = 20303001 + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows, tracks, feats = load_tracks(); meta = track_metadata(rows, tracks)
    manifest = json.loads((OUT / f"manifests/episode_manifest_f{args.fold}.json").read_text())
    fit = [r for r in manifest["records"] if r["split"] == "fit"]; val = [r for r in manifest["records"] if r["split"] == "val"]
    run = f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"; marker = OUT / "completion" / f"{run}.launched"; done = OUT / "completion" / f"{run}.done"; latest = OUT / "checkpoints" / f"{run}_latest.pt"; best = OUT / "checkpoints" / f"{run}_best.pt"; metrics_path = OUT / "metrics" / f"{run}.json"; log_path = OUT / "logs" / f"{run}.jsonl"
    if done.exists() and not args.resume: print(json.dumps({"status": "already_done", "done": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"refusing relaunch with marker {marker}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists(): atomic_json(marker, {"fold": args.fold, "pid": os.getpid(), "started_utc": datetime_now(), "device": str(device), "physical_gpu": args.expected_physical_gpu})
    model = SupportSetCorrespondenceEncoder().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); rng = np.random.default_rng(seed + 17); start = 0; best_score = -1e9; history = []; steps = 100 if args.smoke else int(args.steps); checkpoint_every = max(1, min(int(args.checkpoint_every), steps // 2 if steps > 1 else 1)); amp_enabled = device.type == "cuda"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); start = int(ck.get("global_step", 0)); best_score = float(ck.get("best_score", -1e9)); rng.bit_generator.state = ck.get("numpy_rng", rng.bit_generator.state)
    for step in range(start + 1, steps + 1):
        batch_records = [fit[int(rng.integers(0, len(fit)))] for _ in range(min(args.batch_size, len(fit)))]
        optimizer.zero_grad(set_to_none=True)
        batch = make_batch(batch_records, meta, feats, int(rng.choice(PREFIXES)), device)
        loss, parts = loss_step(model, batch, amp_enabled)
        if not torch.isfinite(loss):
            if amp_enabled:
                amp_enabled = False; optimizer.zero_grad(set_to_none=True); loss, parts = loss_step(model, batch, False)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if step % checkpoint_every == 0 or step == steps:
            v = validate(model, val, meta, feats, device, 16); score = float(v["episode_accuracy"] + 0.1 * v["null_rejection"])
            rec = {"step": step, **parts, **{f"val_{k}": valv for k, valv in v.items()}, "amp": amp_enabled}; history.append(rec); payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": step, "best_score": max(best_score, score), "best_step": step if score > best_score else int((history[-2].get("step", 0) if len(history) > 1 else 0)), "numpy_rng": rng.bit_generator.state, "metadata": metadata(model), "fold": args.fold, "seed": seed, "protocol": "trackocd_iclr27_phase30_support_set_correspondence"}; atomic_torch(latest, payload); 
            if score > best_score: best_score = score; atomic_torch(best, payload)
            with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
    result = {"protocol": "trackocd_iclr27_phase30_support_set_correspondence", "fold": args.fold, "seed": seed, "steps": steps, "batch_size": args.batch_size, "device": str(device), "physical_gpu": args.expected_physical_gpu, "amp": "bf16" if amp_enabled else "fp32", "fit_episodes": len(fit), "validation_episodes": len(val), "best_score": best_score, "history": history, "checkpoint_best": str(best), "checkpoint_latest": str(latest), "metadata": metadata(model), "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "held event outcomes", "future rows/tracks", "physical/semantic IDs", "category text", "StateMemory", "controller action"], "done": True}
    atomic_json(metrics_path, result); atomic_json(done, {"fold": args.fold, "steps": steps, "best_score": best_score, "checkpoint": str(best), "completed_utc": datetime_now()})
    print(json.dumps({"fold": args.fold, "steps": steps, "best_score": best_score, "metrics": str(metrics_path), "done": str(done)}, indent=2, sort_keys=True))


def datetime_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__": main()
