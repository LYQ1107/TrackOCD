"""Small legal CLIP-space trajectory adapter prototype.

Known category labels come only from train_known_tracks.jsonl.  The second
pool is the cached public predicted-track CLIP sequence and is treated as
unlabeled video.  No Q1 labels, GT category IDs, or semantic decision state
are read here.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase11.model.clip_trajectory import ClipTrajectoryEncoder  # noqa: E402


def load_pool(path: Path, max_t: int = 8):
    xs, masks, ids = [], [], []
    for p in sorted(path.glob("*.json")):
        r = json.loads(p.read_text())
        f = np.asarray(r["frame_embeddings"], dtype=np.float32)
        f = f[:max_t]
        x = np.zeros((max_t, f.shape[1]), dtype=np.float32)
        m = np.zeros(max_t, dtype=np.uint8)
        x[:len(f)] = f
        m[:len(f)] = 1
        xs.append(x); masks.append(m); ids.append(r["sample_id"])
    if not xs:
        raise RuntimeError(f"empty feature pool: {path}")
    return np.asarray(xs), np.asarray(masks), ids


def augment(x, m, rng, noise=0.01, drop=0.15):
    y = x.copy()
    mm = m.copy()
    valid = np.argwhere(mm > 0)
    for bi, ti in valid:
        if ti > 0 and rng.rand() < drop:
            mm[bi, ti] = 0
    y += rng.normal(0, noise, size=y.shape).astype(np.float32) * mm[..., None]
    return y, mm


def atomic_torch(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase11/training/clip_trajectory_small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--batch-known", type=int, default=64)
    ap.add_argument("--batch-unlabeled", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-consistency", type=float, default=1.0)
    ap.add_argument("--w-prefix", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1111)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    out = ROOT / args.out

    known_rows = {}
    with open(ROOT / "data/trackocd_v1/pure/public/train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            known_rows[r["sample_id"]] = int(r["category_id"])
    kx, km, kids = load_pool(ROOT / "data/caches/features/clip/train_known_mean")
    if set(kids) != set(known_rows):
        raise RuntimeError(f"known cache/label mismatch {len(kids)} vs {len(known_rows)}")
    known_ids = np.asarray(sorted(set(known_rows.values())), dtype=np.int64)
    cat2idx = {int(c): i for i, c in enumerate(known_ids)}
    ky = np.asarray([cat2idx[known_rows[s]] for s in kids], dtype=np.int64)
    ux, um, _ = load_pool(ROOT / "data/caches/features/clip/pred_tracks_mean")

    model = ClipTrajectoryEncoder(in_dim=512, dim=128, hidden=128, out_dim=128).to(device)
    head = torch.nn.Linear(128, len(known_ids)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr,
                            weight_decay=1e-4)
    logs = []
    for ep in range(args.epochs):
        t0 = time.time(); acc = {"total": 0., "known": 0., "consistency": 0., "prefix": 0.}
        for _ in range(args.steps_per_epoch):
            ki = rng.choice(len(kx), size=min(args.batch_known, len(kx)), replace=False)
            ui = rng.choice(len(ux), size=min(args.batch_unlabeled, len(ux)), replace=False)
            kxt = torch.from_numpy(kx[ki]).to(device); kmt = torch.from_numpy(km[ki]).to(device)
            ux0, um0 = augment(ux[ui], um[ui], rng); ux1, um1 = augment(ux[ui], um[ui], rng)
            ux0 = torch.from_numpy(ux0).to(device); um0 = torch.from_numpy(um0).to(device)
            ux1 = torch.from_numpy(ux1).to(device); um1 = torch.from_numpy(um1).to(device)
            kh, _ = model(kxt, kmt); uh0, us0 = model(ux0, um0); uh1, _ = model(ux1, um1)
            known_loss = F.cross_entropy(head(kh), torch.from_numpy(ky[ki]).to(device))
            cons = (1.0 - F.cosine_similarity(uh0, uh1, dim=-1)).mean()
            prefix = (1.0 - F.cosine_similarity(us0[:, 1:], us0[:, :-1], dim=-1)).mean()
            loss = known_loss + args.w_consistency * cons + args.w_prefix * prefix
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            acc["total"] += float(loss.detach()); acc["known"] += float(known_loss.detach())
            acc["consistency"] += float(cons.detach()); acc["prefix"] += float(prefix.detach())
        for k in acc: acc[k] /= args.steps_per_epoch
        acc["epoch"] = ep + 1; acc["seconds"] = time.time() - t0; logs.append(acc)
        print(json.dumps(acc), flush=True)
    payload = {
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "head": {n: p.detach().cpu() for n, p in head.state_dict().items()},
        "known_ids": known_ids, "input_dim": 512, "out_dim": 128,
        "unlabeled_pool": "data/caches/features/clip/pred_tracks_mean",
        "q1_labels_used": False, "logs": logs, "args": vars(args),
    }
    atomic_torch(payload, out / "checkpoint.pth")
    (out / "train_log.json").write_text(json.dumps(logs, indent=2))
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print("saved", out / "checkpoint.pth")


if __name__ == "__main__":
    main()
