"""Train and evaluate a legal synthetic OOD correspondence probe.

Only public train-known labels are used. Hidden-val categories are excluded
from the supervised pool, then split into support/query tracks. The output is
a metric correspondence diagnostic, not a new semantic-memory architecture.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase12.model.synthetic_correspondence import SyntheticTrajectoryEncoder  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project  # noqa: E402


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def supcon(z, labels, temperature=0.12):
    if len(z) < 2:
        return torch.zeros((), device=z.device)
    sim = (z @ z.T) / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    valid = same.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=z.device)
    log_den = torch.logsumexp(sim, dim=1)
    pos_log = torch.logsumexp(sim.masked_fill(~same, -1e9), dim=1)
    return -(pos_log[valid] - log_den[valid]).mean()


def load_projected(out: Path, device):
    cache = out / "tse_tracks.npz"
    if cache.exists():
        x = np.load(cache)
        return x["features"].astype(np.float32), x["mask"].astype(np.uint8)
    tracks = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    raw = tracks["frame_feats"].astype(np.float32)
    mask = tracks["frame_mask"].astype(np.uint8)
    tse, _, _ = load_tse(device)
    z = project(device, tse, raw.reshape(-1, raw.shape[-1]))
    z = z.reshape(raw.shape[0], raw.shape[1], -1).astype(np.float32)
    atomic_npz(cache, features=z, mask=mask)
    return z, mask


def embed(model, x, mask, device, batch=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z, _ = model(
                torch.from_numpy(x[i:i + batch]).to(device),
                torch.from_numpy(mask[i:i + batch]).to(device),
            )
            out.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def correspondence_metrics(emb, support, query, labels):
    support = np.asarray(support, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    proto, cats = [], []
    for c in sorted({int(labels[i]) for i in support}):
        ids = support[labels[support] == c]
        v = e[ids].mean(axis=0)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        proto.append(v); cats.append(c)
    proto = np.asarray(proto, dtype=np.float32)
    cats = np.asarray(cats, dtype=np.int64)
    pred = cats[np.argmax(e[query] @ proto.T, axis=1)]
    qlab = labels[query]
    pair_acc = float(np.mean(pred == qlab)) if len(query) else 0.0
    # Track-level nearest support, not a learned memory; this is the direct
    # cross-instance correspondence measurement for the synthetic episode.
    sim = e[query] @ e[support].T
    nearest = labels[support[np.argmax(sim, axis=1)]]
    nn_acc = float(np.mean(nearest == qlab)) if len(query) else 0.0
    same, diff = [], []
    for qi in query:
        d = 1.0 - e[qi] @ e[support].T
        for j, si in enumerate(support):
            (same if labels[qi] == labels[si] else diff).append(float(d[j]))
    return {
        "query_tracks": int(len(query)),
        "support_tracks": int(len(support)),
        "categories": int(len(cats)),
        "prototype_accuracy": pair_acc,
        "nearest_support_accuracy": nn_acc,
        "same_category_distance": float(np.mean(same)) if same else None,
        "different_category_distance": float(np.mean(diff)) if diff else None,
        "inter_minus_intra": (float(np.mean(diff) - np.mean(same))
                               if same and diff else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="outputs/iclr27_phase12/synthetic/episodes.npz")
    ap.add_argument("--metadata", default="outputs/iclr27_phase12/synthetic/episodes.json")
    ap.add_argument("--out", default="outputs/iclr27_phase12/synthetic/training")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1213)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ep = np.load(ROOT / args.episodes)
    meta = json.loads((ROOT / args.metadata).read_text())
    x, mask = load_projected(out.parent, device)
    labels = ep["labels"].astype(np.int64)
    train_pool = np.unique(np.concatenate([ep["train_indices"], ep["visible_indices"]]))
    train_categories = sorted({int(labels[i]) for i in train_pool})
    cat_idx = {c: i for i, c in enumerate(train_categories)}
    y = np.asarray([cat_idx[int(labels[i])] for i in train_pool], dtype=np.int64)

    model = SyntheticTrajectoryEncoder(in_dim=x.shape[-1], hidden=128, out_dim=128).to(device)
    head = nn.Linear(128, len(train_categories)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-4)
    rng = np.random.RandomState(args.seed)
    logs = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_pool))
        total = ce_total = con_total = 0.0
        nstep = 0
        t0 = time.time()
        model.train(); head.train()
        for start in range(0, len(order), args.batch_size):
            sel = order[start:start + args.batch_size]
            xb = torch.from_numpy(x[train_pool[sel]]).to(device)
            mb = torch.from_numpy(mask[train_pool[sel]]).to(device)
            yb = torch.from_numpy(y[sel]).to(device)
            z, _ = model(xb, mb)
            ce = F.cross_entropy(head(z), yb)
            con = supcon(z, yb)
            loss = ce + 0.5 * con
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            total += float(loss.detach()); ce_total += float(ce.detach()); con_total += float(con.detach()); nstep += 1
        rec = {"epoch": epoch + 1, "loss": total / nstep, "ce": ce_total / nstep,
               "supcon": con_total / nstep, "seconds": time.time() - t0}
        logs.append(rec); print(json.dumps(rec), flush=True)

    model.eval()
    all_emb = embed(model, x, mask, device)
    # Frozen TSE baseline uses the same final-frame track vectors as the input
    # mean, so compare it directly to the learned trajectory output.
    raw_track = (x * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    raw_track /= np.maximum(np.linalg.norm(raw_track, axis=1, keepdims=True), 1e-12)
    support = ep["eval_support"].astype(np.int64)
    query = ep["eval_query"].astype(np.int64)
    results = {
        "learned_trajectory": correspondence_metrics(all_emb, support, query, labels),
        "frozen_tse_mean": correspondence_metrics(raw_track, support, query, labels),
        "train_categories": train_categories,
        "eval_categories": sorted({int(labels[i]) for i in query}),
        "q1_labels_used": False,
        "private_gt_used": False,
        "future_frames_used": False,
        "physical_id_used_as_feature": False,
        "metadata": meta,
        "logs": logs,
    }
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "args": vars(args), "q1_labels_used": False, "results": results,
    }
    tmp = out / "checkpoint.pth.tmp"
    torch.save(payload, tmp); os.replace(tmp, out / "checkpoint.pth")
    (out / "train_log.json").write_text(json.dumps(logs, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["learned_trajectory"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
