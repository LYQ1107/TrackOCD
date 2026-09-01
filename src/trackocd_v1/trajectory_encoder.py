#!/usr/bin/env python3
"""Candidate architecture B: lightweight trajectory-centric encoder.

Input: up to 8 frames of normalized DINOv2 (768) + CLIP (512) embeddings.
Two variants: attention pooling (B2) and 2-layer temporal transformer (B3),
plus the track-mean baseline (B1). Trained on TAO train-known tracks with
supervised contrastive + classification losses. Frozen backbones.

Outputs are consumed by the same corrected B2/OCD-v2 memories as architecture
A so the comparison isolates the trajectory representation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import hungarian_acc
from src.ocd_v2.common import load_train_known, proxy_split, build_prototypes
from src.trackocd_v1.memory import SeededMultiPrototypeMemory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import (
    STREAMS, load_gt, subset_ids, stream_orders,
)
from src.trackocd_v1.modular import simulate_ncm

MAX_FRAMES = 8
DINO_DIM = 768
CLIP_DIM = 512


def load_frame_dict(encoder, subdir):
    out = {}
    cache = PROJECT_ROOT / "data" / "caches" / "features" / encoder / subdir
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        arr = np.asarray(r["frame_embeddings"], dtype=np.float32)
        out[r["sample_id"]] = arr[:MAX_FRAMES]
    return out


class TrajectoryEncoder(nn.Module):
    def __init__(self, num_classes, variant="attn_pool", dim=256, heads=4, layers=2):
        super().__init__()
        self.variant = variant
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(DINO_DIM + CLIP_DIM, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
        )
        self.pos = nn.Parameter(torch.zeros(1, MAX_FRAMES, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if variant == "attn_pool":
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(dim)
        else:  # transformer
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                dropout=0.1, activation="gelu", batch_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.proj_head = nn.Linear(dim, 128)

    def forward(self, x, mask):
        # x: B,T,1280 ; mask: B,T bool (True = valid)
        B, T, _ = x.shape
        h = self.proj(x)
        h = h + self.pos[:, :T]
        key_padding = ~mask
        if self.variant == "attn_pool":
            q = self.query.expand(B, 1, -1)
            out, _ = self.pool(q, h, h, key_padding_mask=key_padding)
            out = self.pool_norm(out).squeeze(1)
        else:
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            kp = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), key_padding], dim=1)
            out = self.enc(h, src_key_padding_mask=kp)
            out = self.norm(out[:, 0])
        return out

    def embed(self, x, mask):
        return F.normalize(self.proj_head(self.forward(x, mask)), dim=-1)


def make_batch(feats, labels, ids):
    d = torch.zeros(len(ids), MAX_FRAMES, DINO_DIM)
    c = torch.zeros(len(ids), MAX_FRAMES, CLIP_DIM)
    mask = torch.zeros(len(ids), MAX_FRAMES, dtype=torch.bool)
    y = torch.zeros(len(ids), dtype=torch.long)
    for i, sid in enumerate(ids):
        df = feats["dino"][sid]
        cf = feats["clip"][sid]
        n = min(len(df), len(cf), MAX_FRAMES)
        d[i, :n] = torch.from_numpy(df[:n])
        c[i, :n] = torch.from_numpy(cf[:n])
        mask[i, :n] = True
        y[i] = labels[sid]
    return d, c, mask, y


def supcon(z, y, tau=0.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / tau
    n = z.shape[0]
    mask = (y.unsqueeze(0) == y.unsqueeze(1)) & ~torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim - torch.eye(n, device=z.device) * 1e9
    denom = torch.logsumexp(sim, dim=1, keepdim=True)
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * mask.float()).sum(1)
    log_prob = torch.log(pos_sum / torch.exp(denom).squeeze(1) + 1e-9)
    loss = -log_prob.mean()
    return loss


def train(args):
    torch.manual_seed(1027)
    dinos = load_frame_dict("dinov2", "train_known_mean")
    clips = load_frame_dict("clip", "train_known_mean")
    _, labels = load_train_known("dinov2")
    ids = sorted(s for s in labels if s in dinos and s in clips)
    classes = sorted(set(labels.values()))
    cid2idx = {c: i for i, c in enumerate(classes)}
    labels_idx = {s: cid2idx[labels[s]] for s in ids}
    model = TrajectoryEncoder(len(classes), variant=args.variant)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant={args.variant} params={n_params/1e6:.3f}M tracks={len(ids)}", flush=True)
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.RandomState(1027)
    for epoch in range(args.epochs):
        rng.shuffle(ids)
        model.train()
        total = 0.0
        nbatch = 0
        for i in range(0, len(ids), args.batch):
            batch_ids = ids[i:i + args.batch]
            d, c, mask, y = make_batch(
                {"dino": dinos, "clip": clips}, labels_idx, batch_ids
            )
            d, c, mask, y = d.cuda(), c.cuda(), mask.cuda(), y.cuda()
            x = torch.cat([d, c], dim=-1)
            h = model(x, mask)
            ce = F.cross_entropy(model.classifier(h), y)
            sc = supcon(model.embed(x, mask), y)
            loss = ce + 0.5 * sc
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nbatch += 1
        sched.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch} loss {total/max(nbatch,1):.4f}", flush=True)
    ckpt_dir = PROJECT_ROOT / "runs" / "trackocd_v1" / f"traj_enc_{args.variant}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": classes, "variant": args.variant,
         "params": n_params},
        ckpt_dir / "model.pth",
    )
    print("saved", ckpt_dir / "model.pth", flush=True)


def load_model(variant):
    ck = torch.load(
        PROJECT_ROOT / "runs" / "trackocd_v1" / f"traj_enc_{variant}" / "model.pth",
        map_location="cpu",
    )
    model = TrajectoryEncoder(len(ck["classes"]), variant=ck["variant"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model.cuda(), ck


def encode_all(model, feats, ids):
    outs = {}
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            batch = ids[i:i + 256]
            d = torch.zeros(len(batch), MAX_FRAMES, DINO_DIM)
            c = torch.zeros(len(batch), MAX_FRAMES, CLIP_DIM)
            mask = torch.zeros(len(batch), MAX_FRAMES, dtype=torch.bool)
            for j, sid in enumerate(batch):
                df = feats["dino"][sid]
                cf = feats["clip"][sid]
                n = min(len(df), len(cf), MAX_FRAMES)
                d[j, :n] = torch.from_numpy(df[:n])
                c[j, :n] = torch.from_numpy(cf[:n])
                mask[j, :n] = True
            h = model(torch.cat([d, c], dim=-1).cuda(), mask.cuda())
            h = F.normalize(h, dim=-1).cpu().numpy()
            for j, sid in enumerate(batch):
                outs[sid] = h[j].astype(np.float32)
    return outs


def eval_track_embeddings(args):
    model, ck = load_model(args.variant)
    tr_d = load_frame_dict("dinov2", "train_known_mean")
    tr_c = load_frame_dict("clip", "train_known_mean")
    _, labels = load_train_known("dinov2")
    tr_ids = sorted(s for s in labels if s in tr_d and s in tr_c)
    tr_emb = encode_all(model, {"dino": tr_d, "clip": tr_c}, tr_ids)
    thr = calibrate_ncm_thr(tr_emb, labels)
    print("encoder thr", thr, flush=True)
    protos = build_prototypes(tr_emb, labels, set(labels.values()))

    val_d = load_frame_dict("dinov2", "gt_tracks_mean")
    val_c = load_frame_dict("clip", "gt_tracks_mean")
    val_ids = sorted(s for s in val_d if s in val_c)
    val_emb = encode_all(model, {"dino": val_d, "clip": val_c}, val_ids)
    orders = stream_orders()
    rows_out = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in ("full", "repeated", "balanced"):
            for stream in STREAMS:
                fname = "val_gt_track_stream.jsonl" if stream == "main" else f"val_gt_track_stream_{stream[5:]}.jsonl"
                srows = []
                with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / fname) as f:
                    for line in f:
                        if line.strip():
                            srows.append(json.loads(line))
                for mem in ("b2", "ocdv2"):
                    if mem == "b2":
                        preds = simulate_ncm(srows, val_emb, protos, thr)
                    else:
                        preds = run_ocdv2(srows, val_emb, protos)
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
                    row = {
                        "architecture": f"B{args.variant}_{mem}", "protocol": proto,
                        "subset": subset, "seed": stream,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    }
                    rows_out.append(row)
                    print(proto, subset, stream, mem,
                          "all", round(row["all_track_acc"], 4),
                          "known", round(row["overall_known_acc"], 4),
                          "novel_route", round(row["route_aware_novel_acc"], 4),
                          "novel_cond", round(row["conditional_novel_acc"], 4),
                          flush=True)
    import csv
    path = PROJECT_ROOT / "outputs" / "trackocd_v1" / "metrics" / f"trajectory_architecture_{args.variant}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("saved", path)


def calibrate_ncm_thr(feats, labels):
    pk, pn = proxy_split(labels, seed=1027)
    ids = sorted(s for s, c in labels.items() if c in pn and s in feats)
    y = np.array([labels[s] for s in ids])
    protos = build_prototypes(feats, labels, pk)
    best = (0.45, -1.0)
    for thr in np.arange(0.30, 0.81, 0.025):
        preds = simulate_ncm([{"sample_id": s} for s in ids], feats, protos, float(thr))
        pv = np.array([
            (p["semantic_category_id"] if p["prediction_type"] == "known" else 100000 + p["virtual_category_id"])
            for p in preds
        ])
        uniq = sorted(set(int(v) for v in pv))
        remap = {v: i for i, v in enumerate(uniq)}
        pv = np.array([remap[int(v)] for v in pv])
        acc = hungarian_acc(y, pv)[0]
        if acc > best[1]:
            best = (float(thr), acc)
    return best[0]


def run_ocdv2(srows, feats, known_protos):
    params = {
        "attach_thr": 0.525, "create_thr": 0.375, "new_proto_thr": 0.475,
        "max_proto": 4, "ema": 0.25, "maturity_tracks": 2,
    }
    model = SeededMultiPrototypeMemory(known_protos, **params)
    preds = []
    for i, row in enumerate(srows):
        vid = model.predict_one(
            feats[row["sample_id"]], row["sample_id"], i,
            num_frames=len(row.get("frame_ids", []) or []), video_id=row["video_id"],
        )
        if vid < 200000:
            preds.append({
                "sample_id": row["sample_id"], "stream_order": i,
                "prediction_type": "known", "semantic_category_id": vid,
            })
        else:
            preds.append({
                "sample_id": row["sample_id"], "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": vid,
            })
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--variant", choices=["attn_pool", "transformer"], default="attn_pool")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    if args.mode == "train":
        train(args)
    else:
        eval_track_embeddings(args)


if __name__ == "__main__":
    main()
