"""Small track-level representation training and open-world episodes.

The training asset is real TAO TRAIN.  Public category IDs are used only to
form legal TRAIN known/held-out episodes; Q1/private labels never enter this
script.  The frozen TrackOCD decision process is absent here and is evaluated
separately by ``replay_track_encoder.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase13.model.track_encoder import TrackSemanticEncoder  # noqa: E402


EVAL_CATEGORIES = [35, 118, 139, 229, 382, 429, 502, 714, 980, 1144]


def atomic_torch(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def supcon(z: torch.Tensor, labels: torch.Tensor, temperature=0.12):
    if len(z) < 2:
        return torch.zeros((), device=z.device)
    sim = (z @ z.T) / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    valid = same.any(1)
    if not valid.any():
        return torch.zeros((), device=z.device)
    den = torch.logsumexp(sim, dim=1)
    pos = torch.logsumexp(sim.masked_fill(~same, -1e9), dim=1)
    return -(pos[valid] - den[valid]).mean()


def episode_loss(z: torch.Tensor, labels: torch.Tensor, temperature=0.12):
    """Support/query classification for a 2-track-per-class mini-episode."""
    if len(z) < 2:
        return torch.zeros((), device=z.device)
    # Sampling below places one support and one query consecutively per class.
    support, query = z[0::2], z[1::2]
    sl, ql = labels[0::2], labels[1::2]
    if len(support) == 0 or len(query) == 0:
        return torch.zeros((), device=z.device)
    valid = ql[:, None].eq(sl[None, :]).any(1)
    if not valid.any():
        return torch.zeros((), device=z.device)
    logits = (query @ support.T) / temperature
    target = torch.argmax(ql[valid, None].eq(sl[None, :]).float(), dim=1)
    return F.cross_entropy(logits[valid], target)


def augment(x, m, rng, noise=0.012, drop=0.20):
    xa = x.copy()
    ma = m.copy().astype(np.uint8)
    for i in range(len(xa)):
        valid = np.flatnonzero(ma[i])
        for t in valid[1:]:
            if rng.rand() < drop:
                ma[i, t] = 0
        xa[i] += rng.normal(0.0, noise, size=xa[i].shape).astype(np.float32) * ma[i, :, None]
    return xa, ma


def balanced_episode_indices(by_class, classes, rng, n_classes=16):
    chosen = rng.choice(classes, size=min(n_classes, len(classes)), replace=False)
    idx, y = [], []
    for c in chosen:
        pool = by_class[int(c)]
        if len(pool) < 2:
            continue
        pair = rng.choice(pool, size=2, replace=False)
        idx.extend(pair.tolist())
        y.extend([int(c), int(c)])
    return np.asarray(idx, dtype=np.int64), np.asarray(y, dtype=np.int64)


def episode_indices(labels, categories, max_per_category=8):
    support, query = [], []
    for c in categories:
        ids = np.flatnonzero(labels == int(c))[:max_per_category]
        if len(ids) < 2:
            continue
        n = len(ids) // 2
        support.extend(ids[:n].tolist())
        query.extend(ids[n:].tolist())
    return np.asarray(support, dtype=np.int64), np.asarray(query, dtype=np.int64)


def correspondence_metrics(emb, labels, support, query):
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    support = np.asarray(support, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    cats = sorted(set(int(labels[i]) for i in support))
    protos = []
    for c in cats:
        v = e[support[labels[support] == c]].mean(0)
        protos.append(v / max(float(np.linalg.norm(v)), 1e-12))
    protos = np.asarray(protos, dtype=np.float32)
    cats = np.asarray(cats, dtype=np.int64)
    pred = cats[np.argmax(e[query] @ protos.T, axis=1)]
    nn = labels[support[np.argmax(e[query] @ e[support].T, axis=1)]]
    same, diff = [], []
    for qi in query:
        d = 1.0 - e[qi] @ e[support].T
        for j, si in enumerate(support):
            (same if labels[qi] == labels[si] else diff).append(float(d[j]))
    return {
        "query_tracks": int(len(query)),
        "support_tracks": int(len(support)),
        "categories": int(len(cats)),
        "prototype_accuracy": float(np.mean(pred == labels[query])) if len(query) else 0.0,
        "nearest_support_accuracy": float(np.mean(nn == labels[query])) if len(query) else 0.0,
        "same_category_distance": float(np.mean(same)) if same else None,
        "different_category_distance": float(np.mean(diff)) if diff else None,
        "inter_minus_intra": float(np.mean(diff) - np.mean(same)) if same and diff else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="outputs/iclr27_phase13/dataset/real_tao_tracks.npz")
    ap.add_argument("--metadata", default="outputs/iclr27_phase13/dataset/metadata.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["full", "no_semantic_alignment", "no_temporal", "no_episodic_unknown", "self_supervised"], default="full")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--classes-per-step", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1313)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / args.dataset)
    appearance = z["appearance"].astype(np.float32)
    motion = z["motion"].astype(np.float32)
    mask = z["mask"].astype(np.uint8)
    labels = z["labels"].astype(np.int64)
    metadata = json.loads((ROOT / args.metadata).read_text())

    eval_cats = [c for c in EVAL_CATEGORIES if int(np.sum(labels == c)) >= 2]
    train_cats = sorted(set(int(c) for c in labels) - set(eval_cats))
    by_class = defaultdict(list)
    for i, c in enumerate(labels):
        if int(c) in train_cats:
            by_class[int(c)].append(i)
    train_cats = [c for c in train_cats if len(by_class[c]) >= 2]
    train_idx = np.asarray([i for c in train_cats for i in by_class[c]], dtype=np.int64)
    support, query = episode_indices(labels, eval_cats)
    if len(support) == 0 or len(query) == 0:
        raise RuntimeError("held-out episode split is empty")
    label_map = {c: i for i, c in enumerate(train_cats)}
    model = TrackSemanticEncoder(appearance_dim=appearance.shape[-1], motion_dim=motion.shape[-1], hidden=args.hidden, out_dim=128).to(device)
    head = nn.Linear(128, len(train_cats)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)
    use_align = args.variant not in {"no_semantic_alignment", "self_supervised"}
    use_temp = args.variant != "no_temporal"
    use_episode = args.variant not in {"no_episodic_unknown", "self_supervised"}
    use_category_loss = args.variant != "self_supervised"
    logs = []
    for epoch in range(args.epochs):
        model.train(); head.train(); t0 = time.time()
        sums = {k: 0.0 for k in ("loss", "ce", "temporal", "alignment", "episode")}
        for _ in range(args.steps_per_epoch):
            if use_category_loss:
                ids, yc = balanced_episode_indices(by_class, train_cats, rng, args.classes_per_step)
            else:
                ids = rng.choice(train_idx, size=min(32, len(train_idx)), replace=False)
                yc = np.zeros(len(ids), dtype=np.int64)
            if len(ids) < 4:
                continue
            xb = torch.from_numpy(appearance[ids]).to(device)
            mb = torch.from_numpy(mask[ids]).to(device)
            motb = torch.from_numpy(motion[ids]).to(device)
            x0, m0 = augment(appearance[ids], mask[ids], rng)
            x1, m1 = augment(appearance[ids], mask[ids], rng)
            h, seq = model(xb, motb, mb)
            h0, _ = model(torch.from_numpy(x0).to(device), motb, torch.from_numpy(m0).to(device))
            h1, _ = model(torch.from_numpy(x1).to(device), motb, torch.from_numpy(m1).to(device))
            y = torch.from_numpy(np.asarray([label_map[int(c)] for c in yc], dtype=np.int64)).to(device) if use_category_loss else None
            ce = F.cross_entropy(head(h), y) if use_category_loss else h.sum() * 0.0
            temporal = (1.0 - F.cosine_similarity(h0, h1, dim=-1)).mean()
            alignment = supcon(h, torch.from_numpy(yc).to(device))
            episodic = episode_loss(h, torch.from_numpy(yc).to(device))
            if not use_temp: temporal = temporal.detach() * 0.0
            if not use_align: alignment = alignment.detach() * 0.0
            if not use_episode: episodic = episodic.detach() * 0.0
            loss = ce + 0.5 * temporal + 0.5 * alignment + 0.5 * episodic
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            for k, v in (("loss", loss), ("ce", ce), ("temporal", temporal), ("alignment", alignment), ("episode", episodic)):
                sums[k] += float(v.detach())
        den = max(1, args.steps_per_epoch)
        rec = {k: v / den for k, v in sums.items()}
        rec.update({"epoch": epoch + 1, "seconds": time.time() - t0})
        logs.append(rec); print(json.dumps(rec), flush=True)

    model.eval()
    with torch.no_grad():
        all_emb = []
        for start in range(0, len(appearance), 128):
            h, _ = model(torch.from_numpy(appearance[start:start + 128]).to(device), torch.from_numpy(motion[start:start + 128]).to(device), torch.from_numpy(mask[start:start + 128]).to(device))
            all_emb.append(h.cpu().numpy().astype(np.float32))
    all_emb = np.concatenate(all_emb, axis=0)
    raw = (appearance * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
    raw = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    results = {
        "variant": args.variant,
        "train_categories": train_cats,
        "eval_categories": eval_cats,
        "train_tracks": int(len(train_idx)),
        "eval_support_tracks": int(len(support)),
        "eval_query_tracks": int(len(query)),
        "learned": correspondence_metrics(all_emb, labels, support, query),
        "dino_v2_mean": correspondence_metrics(raw, labels, support, query),
        "q1_labels_used": False,
        "private_gt_used": False,
        "future_frames_used": False,
        "physical_id_used_as_feature": False,
        "public_train_category_labels_used": bool(use_category_loss),
        "public_train_category_labels_used_for_split": True,
        "dataset_metadata": metadata,
        "logs": logs,
    }
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "args": vars(args), "train_categories": np.asarray(train_cats),
        "eval_categories": np.asarray(eval_cats), "results": results,
    }
    atomic_torch(payload, out / "checkpoint.pth")
    (out / "train_log.json").write_text(json.dumps(logs, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["learned"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
