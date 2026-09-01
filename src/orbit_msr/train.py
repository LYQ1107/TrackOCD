"""Train ORBIT-MSR with a 48-known + synthetic-novel episodic protocol.

The known set matches the evaluation scale (all 48 train classes), so the
best-known similarity distribution seen by the gate matches the long-stream
proxy and official evaluation.  Pseudo-novel classes are synthetic
feature-space perturbations generated a priori (train-side only).
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_fc.model import ORBITFCModel
from src.orbit_msr.losses import (
    gate_bce,
    gate_margin_loss,
    known_loss,
    novel_metric_loss,
    reuse_bce,
)
from src.orbit_msr.protocol import (
    ROOT,
    known_stats,
    novel_stats,
    stats_to_tensor,
)


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def aggregate_one(model, frames, device):
    x = _t(frames[:8], device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    return model.aggregate(x, mask)


def make_synthetic_tracks(np_rng, base_z, n_tracks=4, alpha_range=(0.35, 0.65),
                          sigma=0.08):
    alpha = float(np_rng.uniform(*alpha_range))
    w = np_rng.randn(768).astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-12)
    center = alpha * base_z + (1.0 - alpha) * w
    center = center / (np.linalg.norm(center) + 1e-12)
    tracks = []
    for _ in range(n_tracks):
        noise = np_rng.randn(768).astype(np.float32)
        z = center + sigma * noise / (np.linalg.norm(noise) + 1e-12)
        z = z / (np.linalg.norm(z) + 1e-12)
        tracks.append(np.stack([z for _ in range(8)]).astype(np.float32))
    return tracks


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_dim = 11
    reuse_dim = 11 + (2 if args.mem_scale_norm else 0)
    model = ORBITFCModel(dim=768, bottleneck=args.bottleneck,
                         gate_dim=gate_dim, reuse_dim=reuse_dim, hidden=64,
                         use_adapter=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    classes = sorted(by_class)
    np_rng = np.random.RandomState(args.seed + 1)
    rng = random.Random(args.seed)
    pad_options = [0, 50, 150, 300]

    z_cache = {}

    def refresh_z_pool():
        z_cache.clear()
        for sid in all_feats:
            out = aggregate_one(model, all_feats[sid][:8], device)
            z_cache[sid] = out["z"][0].detach().cpu().numpy().astype(np.float32)

    def build_known():
        protos = {}
        radii = {}
        for c in classes:
            zs = np.stack([z_cache[sid] for sid in by_class[c]])
            p = zs.mean(axis=0)
            p = p / (np.linalg.norm(p) + 1e-12)
            protos[c] = p
            cos = zs @ p
            radii[c] = max(float(np.percentile(1.0 - cos, 50)), 0.02)
        return protos, radii

    refresh_z_pool()
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            refresh_z_pool()
        protos, radii = build_known()
        P_known = np.stack([protos[c] for c in classes]).astype(np.float32)
        model.train()
        total = 0.0
        n_ep = 0
        for _ in range(args.episodes_per_epoch):
            # ---- synthetic novel classes ----
            n_syn = args.syn_novel_classes
            syn_pool = list(all_feats.keys())
            rng.shuffle(syn_pool)
            syn_classes = []
            syn_queries = []
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                n_first = 2 if args.balanced else 1
                tracks = make_synthetic_tracks(np_rng, base_z, n_tracks=4,
                                               alpha_range=(0.35, 0.65))
                lab = 1000000 + k
                syn_classes.append(lab)
                for j, frames in enumerate(tracks):
                    syn_queries.append({
                        "sample_id": f"syn_{sid}_{j}",
                        "label": lab, "known": False,
                        "first": j < n_first, "_frames": frames,
                    })
            # ---- known queries ----
            known_queries = []
            for c in classes:
                ids = list(by_class[c])
                rng.shuffle(ids)
                for sid in ids[:args.known_per_class]:
                    known_queries.append({"sample_id": sid, "label": c,
                                          "known": True, "first": False,
                                          "_frames": None})
            # ---- memory-scale padding ----
            n_pad = rng.choice(pad_options)
            pad_pool = [sid for sid in all_feats]
            rng.shuffle(pad_pool)
            pad_protos = [z_cache[sid] for sid in pad_pool[:n_pad]]
            pad_counts = [int(rng.choice([1, 3, 10, 30])) for _ in pad_protos]
            pad_radii = [0.3 for _ in pad_protos]

            novel_list = list(pad_protos)
            novel_counts = {i: pad_counts[i] for i in range(len(pad_protos))}
            novel_radii = {i: pad_radii[i] for i in range(len(pad_protos))}
            novel_vid_by_label = {}

            query = known_queries + syn_queries
            rng.shuffle(query)
            loss_acc = 0.0
            n_query = 0
            for q in query:
                frames = q.get("_frames") if q.get("_frames") is not None else all_feats[q["sample_id"]][:8]
                out = aggregate_one(model, frames, device)
                z = out["z"]
                z0 = out["z0"]
                rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
                length = int(out["length"][0])
                z_np = z[0].detach().cpu().numpy()
                P_novel_np = np.stack(novel_list).astype(np.float32) if novel_list else np.empty((0, 768), dtype=np.float32)
                best_n = -1.0; second_n = -1.0; margin_n = 0.0; dist_n = 1.0
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                    r_n = novel_radii.get(int(order[0]), 0.3)
                    dist_n = (1.0 - best_n) / max(r_n, 1e-6)
                gs = known_stats(z_np, P_known, radii, known_ids=classes,
                                 best_n=best_n, second_n=second_n,
                                 margin_n=margin_n, dist_n=dist_n, rel=rel,
                                 track_len=length, n_novel=len(novel_list),
                                 include_anchor=False)
                gate_logit = model.gate_forward(stats_to_tensor(gs, device))
                gate_target = torch.tensor([1.0 if q["known"] else 0.0], device=device)
                loss = (gate_margin_loss(gate_logit, gate_target, margin=1.0)
                        if args.margin else gate_bce(gate_logit, gate_target))

                known_target = None
                novel_target = None
                if q["known"]:
                    known_target = torch.tensor([classes.index(q["label"])], device=device)
                    P_known_t = torch.as_tensor(P_known, dtype=torch.float32, device=device)
                    loss = loss + args.lambda_known * known_loss(z, P_known_t, known_target)
                    loss = loss + args.lambda_geo * _geo(z, z0)
                else:
                    ks = P_known @ z_np
                    best_k = float(ks.max()) if ks.shape[0] else -1.0
                    margin_k = _margin(ks)
                    rs = novel_stats(z_np, P_novel_np, novel_counts, novel_radii,
                                     best_k=best_k, margin_k=margin_k, rel=rel,
                                     track_len=length, n_novel=len(novel_list),
                                     mem_scale_norm=args.mem_scale_norm)
                    reuse_logit = model.reuse_forward(stats_to_tensor(rs, device))
                    reuse_target = torch.tensor([0 if q["first"] else 1], device=device)
                    loss = loss + args.lambda_reuse * reuse_bce(
                        reuse_logit, reuse_target, weight_new=args.weight_new)
                    if q["first"]:
                        novel_list.append(z[0].detach().cpu().numpy().astype(np.float32))
                        vid = len(novel_list) - 1
                        novel_vid_by_label[q["label"]] = vid
                        novel_counts[vid] = 1
                        novel_radii[vid] = 0.3
                    else:
                        vid = novel_vid_by_label.get(q["label"])
                        if vid is None:
                            novel_list.append(z[0].detach().cpu().numpy().astype(np.float32))
                            vid = len(novel_list) - 1
                            novel_vid_by_label[q["label"]] = vid
                            novel_counts[vid] = 1
                            novel_radii[vid] = 0.3
                        proto_old = novel_list[vid]
                        cos_to_center = float(proto_old @ z_np)
                        new_proto = (1 - args.novel_update_rate) * proto_old + args.novel_update_rate * z_np
                        new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-12)
                        novel_list[vid] = new_proto.astype(np.float32)
                        novel_counts[vid] = novel_counts.get(vid, 1) + 1
                        if args.update_radius:
                            novel_radii[vid] = float(
                                (1 - 0.2) * novel_radii.get(vid, 0.3)
                                + 0.2 * max(1.0 - cos_to_center, 1e-3))
                        novel_target = torch.tensor([vid], device=device)
                        novel_protos_t = torch.as_tensor(novel_list, dtype=torch.float32, device=device)
                        loss = loss + args.lambda_novel * novel_metric_loss(
                            z, novel_protos_t, novel_target)
                    loss = loss + args.lambda_geo * _geo(z, z0)
                loss_acc = loss_acc + loss
                n_query += 1
            if n_query:
                opt.zero_grad()
                (loss_acc / n_query).backward()
                opt.step()
                total += float(loss_acc.item() / n_query)
                n_ep += 1
        print(f"epoch {epoch} loss {total / max(n_ep, 1):.4f}", flush=True)

    out_dir = ROOT / "runs" / "orbit_msr" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "bottleneck": args.bottleneck,
        "gate_dim": gate_dim,
        "reuse_dim": reuse_dim,
        "use_anchor": False,
        "balanced": args.balanced,
        "margin": args.margin,
        "weight_new": args.weight_new,
        "mem_scale_norm": args.mem_scale_norm,
        "update_radius": args.update_radius,
        "syn_novel_classes": args.syn_novel_classes,
        "novel_update_rate": args.novel_update_rate,
        "seed": args.seed,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def _geo(z, z0):
    from src.orbit.track_aggregator import geometry_loss
    return geometry_loss(z, z0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="KG1")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lambda_reuse", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--episodes_per_epoch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--known_per_class", type=int, default=2)
    ap.add_argument("--syn_novel_classes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--margin", action="store_true")
    ap.add_argument("--weight_new", type=float, default=1.0)
    ap.add_argument("--mem_scale_norm", action="store_true")
    ap.add_argument("--update_radius", action="store_true")
    ap.add_argument("--output_dir", default="msr_kg1")
    args = ap.parse_args()
    train(args)
