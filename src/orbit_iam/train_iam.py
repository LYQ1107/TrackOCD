"""Train ORBIT-IAM compatibility head on the ORBIT-MSR episodic protocol.

I1/I2: C1 checkpoint + frozen aggregator/gate/reuse, only the compatibility
head is trained with pair labels (same-class positive, nearest-wrong and
random-different negatives, first-occurrence all-negative).
I3: additionally fine-tunes the adapter/gate/reuse with memory-conditioned
hard negatives and the C1 loss mix.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.losses import (
    gate_margin_loss,
    known_loss,
    novel_metric_loss,
    reuse_bce,
)
from src.orbit_msr.protocol import known_stats, novel_stats, stats_to_tensor
from src.orbit_msr.train import _geo
from src.orbit_iam.compat import FEAT_ORDER, build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_iam.model import ORBITIAMModel


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def aggregate_one(model, frames, device):
    x = _t(frames[:8], device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    return model.aggregate(x, mask)


def make_synthetic_tracks(np_rng, base_z, n_tracks=4, alpha_range=(0.35, 0.65),
                          sigma=0.12):
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


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def real_pair_compat_loss(model, z_np, cls, classes, protos, by_class, z_cache,
                          feat_names, mem_size, rel, device,
                          hard_neg_k, pos_weight, lambda_compat, lambda_rank,
                          ranking_margin):
    """Pairwise compatibility loss on real train-known classes.

    Uses only train-side known classes (no GT novel labels): positive is the
    same-class prototype, negatives are the top-k nearest different-class
    prototypes.  This closes the synthetic-to-real similarity gap that
    otherwise dominates the episodic pairs.
    """
    idx = classes.index(cls)
    own = idx
    sims = np.stack([protos[c] for c in classes]) @ z_np
    order = np.argsort(sims)[::-1]
    margin_n = float(sims[order[0]] - sims[order[1]]) if len(order) >= 2 else 0.0
    neg_ids = [int(i) for i in order if int(i) != own][:hard_neg_k]
    pair_ids = neg_ids[:]
    targets = [0.0] * len(pair_ids)
    pair_ids.insert(0, own)
    targets.insert(0, 1.0)
    feat_rows = []
    for pid in pair_ids:
        c = classes[pid]
        zs_c = np.stack([z_cache[sid] for sid in by_class[c]])
        proto = protos[c]
        support = len(by_class[c])
        disp = float(np.mean(1.0 - zs_c @ proto))
        conf = (math.log1p(support) / math.log1p(20.0)
                * math.exp(-disp / 0.3))
        radius = float(np.percentile(1.0 - zs_c @ proto, 50))
        feat_rows.append(build_compat_features(
            z_np, proto, radius, support, conf, mem_size, rel, margin_n,
            feat_names))
    X = torch.as_tensor(np.asarray(feat_rows, dtype=np.float32), device=device)
    q_logits = model.compat_forward(X)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    pos_w = torch.as_tensor([pos_weight if t == 1.0 else 1.0 for t in targets],
                            dtype=torch.float32, device=device)
    loss = lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
        q_logits, y, weight=pos_w)
    if len(q_logits) >= 2:
        loss = loss + lambda_rank * torch.relu(
            ranking_margin - q_logits[0] + q_logits[1:].mean())
    return loss


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_names = [f.strip() for f in args.compat_feats.split(",") if f.strip()]
    compat_dim = len(feat_names)
    gate_dim = 11
    reuse_dim = 13 if args.mem_scale_norm else 11
    model = ORBITIAMModel(dim=768, bottleneck=args.bottleneck,
                          gate_dim=gate_dim, reuse_dim=reuse_dim,
                          hidden=64, use_adapter=True,
                          compat_dim=compat_dim).to(device)
    ck = torch.load(args.init_checkpoint, map_location="cpu")
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd:
            sd[k] = v
    model.load_state_dict(sd)
    model.train()

    trainable = []
    if args.freeze_mode == "compat":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.compat.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

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
    pair_stats = []

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
    hns_rows = []
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            refresh_z_pool()
        protos, radii = build_known()
        P_known = np.stack([protos[c] for c in classes]).astype(np.float32)
        model.train()
        total = 0.0
        n_ep = 0
        for _ in range(args.episodes_per_epoch):
            n_syn = args.syn_novel_classes
            syn_pool = list(all_feats.keys())
            rng.shuffle(syn_pool)
            syn_classes = []
            syn_queries = []
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                n_first = 2 if args.balanced else 1
                tracks = make_synthetic_tracks(np_rng, base_z, n_tracks=4,
                                               alpha_range=(0.35, 0.65),
                                               sigma=args.sigma)
                lab = 1000000 + k
                syn_classes.append(lab)
                for j, frames in enumerate(tracks):
                    syn_queries.append({
                        "sample_id": f"syn_{sid}_{j}",
                        "label": lab, "known": False,
                        "first": j < n_first, "_frames": frames,
                    })
            known_queries = []
            for c in classes:
                ids = list(by_class[c])
                rng.shuffle(ids)
                for sid in ids[:args.known_per_class]:
                    known_queries.append({"sample_id": sid, "label": c,
                                          "known": True, "first": False,
                                          "_frames": None})
            n_pad = rng.choice(pad_options)
            pad_pool = list(all_feats.keys())
            rng.shuffle(pad_pool)
            pad_protos = [z_cache[sid] for sid in pad_pool[:n_pad]]
            pad_counts = [int(rng.choice([1, 3, 10, 30])) for _ in pad_protos]
            pad_radii = [0.3 for _ in pad_protos]

            mem = IamMemory(protos, radii,
                            novel_update_rate=args.novel_update_rate)
            for z_p in pad_protos:
                vid = mem.create_novel(z_p, created_at=-1)
                mem.novel_counts[vid] = pad_counts[vid % len(pad_counts)]
                mem.novel_radii[vid] = pad_radii[vid % len(pad_radii)]
            novel_vid_by_label = {}
            # near-miss confuser prototypes: visually similar different-class
            # prototypes that the top-k negative sampler must learn to reject
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                alpha_c = float(np_rng.uniform(0.62, 0.82))
                w = np_rng.randn(768).astype(np.float32)
                w = w / (np.linalg.norm(w) + 1e-12)
                conf_center = alpha_c * base_z + (1.0 - alpha_c) * w
                conf_center = conf_center / (np.linalg.norm(conf_center) + 1e-12)
                vid = mem.create_novel(conf_center.astype(np.float32),
                                       created_at=-1)
                mem.novel_counts[vid] = int(rng.choice([1, 3, 5]))
                mem.novel_radii[vid] = 0.35
                # ensure the sampler never mistakes a confuser for a positive
                # by keeping them out of novel_vid_by_label (different label)

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
                P_novel_np = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                              .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
                best_n = -1.0; second_n = -1.0; margin_n = 0.0; dist_n = 1.0
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                    r_n = mem.novel_radii.get(int(order[0]), 0.3)
                    dist_n = (1.0 - best_n) / max(r_n, 1e-6)
                gs = known_stats(z_np, P_known, radii, known_ids=classes,
                                 best_n=best_n, second_n=second_n,
                                 margin_n=margin_n, dist_n=dist_n, rel=rel,
                                 track_len=length, n_novel=len(mem.novel),
                                 include_anchor=False)
                gate_target = torch.tensor([1.0 if q["known"] else 0.0], device=device)
                loss = (gate_margin_loss(model.gate_forward(stats_to_tensor(gs, device)),
                                         gate_target, margin=1.0)
                        if args.margin else torch.zeros((), device=device))

                if q["known"]:
                    known_target = torch.tensor([classes.index(q["label"])], device=device)
                    loss = loss + args.lambda_known * known_loss(
                        z, torch.as_tensor(P_known, device=device), known_target)
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    if args.real_pairs:
                        loss = loss + real_pair_compat_loss(
                            model, z_np, int(q["label"]), classes, protos,
                            by_class, z_cache, feat_names,
                            len(mem.novel) + len(classes), rel, device,
                            args.hard_neg_k, args.pos_weight,
                            args.lambda_compat, args.lambda_rank,
                            args.ranking_margin)
                else:
                    ks = P_known @ z_np
                    best_k = float(ks.max()) if ks.shape[0] else -1.0
                    rs = novel_stats(z_np, P_novel_np, mem.novel_counts,
                                     mem.novel_radii,
                                     novel_ids=sorted(mem.novel) if mem.novel else None,
                                     best_k=best_k, margin_k=_margin(ks),
                                     rel=rel, track_len=length,
                                     n_novel=len(mem.novel),
                                     age_norm=0.0,
                                     mem_scale_norm=args.mem_scale_norm)
                    reuse_target = torch.tensor([0 if q["first"] else 1], device=device)
                    loss = loss + args.lambda_reuse * reuse_bce(
                        model.reuse_forward(stats_to_tensor(rs, device)),
                        reuse_target, weight_new=args.weight_new)
                    # ---- identity compatibility pairs ----
                    own_vid = novel_vid_by_label.get(q["label"])
                    if P_novel_np.shape[0]:
                        order = np.argsort(ns)[::-1]
                        vids = [int(sorted(mem.novel)[o]) for o in order]
                        neg_vids = [v for v in vids if v != own_vid]
                        k_neg = args.hard_neg_k
                        if args.no_hard_negatives:
                            rest = [v for v in sorted(mem.novel) if v != own_vid]
                            rng.shuffle(rest)
                            hard_negs = rest[:k_neg]
                        else:
                            hard_negs = neg_vids[:k_neg]
                            if len(hard_negs) < k_neg:
                                rest = [v for v in sorted(mem.novel)
                                        if v not in hard_negs and v != own_vid]
                                rng.shuffle(rest)
                                hard_negs += rest[:k_neg - len(hard_negs)]
                        pair_vids = hard_negs[:]
                        targets = [0.0] * len(pair_vids)
                        if own_vid is not None:
                            pair_vids.insert(0, own_vid)
                            targets.insert(0, 1.0)
                        feat_rows = []
                        for vid in pair_vids:
                            st = mem.state(vid)
                            feat_rows.append(build_compat_features(
                                z_np, mem.novel[vid]["proto"], st["radius"],
                                st["support"], st["conf"], len(mem.novel),
                                rel, margin_n, feat_names))
                        if feat_rows:
                            X = torch.as_tensor(np.asarray(feat_rows, dtype=np.float32),
                                                device=device)
                            q_logits = model.compat_forward(X)
                            y = torch.as_tensor(targets, dtype=torch.float32,
                                                device=device)
                            pos_w = torch.as_tensor(
                                [args.pos_weight if t == 1.0 else 1.0
                                 for t in targets], dtype=torch.float32,
                                device=device)
                            loss = loss + args.lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
                                q_logits, y, weight=pos_w)
                            if own_vid is not None and len(q_logits) >= 2:
                                pos_q = q_logits[0]
                                neg_q = q_logits[1:].mean()
                                loss = loss + args.lambda_rank * torch.relu(
                                    args.ranking_margin - pos_q + neg_q)
                            if own_vid is not None:
                                pos_sim = float(np.dot(z_np, mem.novel[own_vid]["proto"]))
                            else:
                                pos_sim = float("nan")
                            pair_stats.append({
                                "epoch": epoch,
                                "first": int(q["first"]),
                                "mem_size": len(mem.novel),
                                "positive_sim": pos_sim,
                                "hard_negative_sim_mean": float(np.mean(ns[[
                                    sorted(mem.novel).index(v) for v in hard_negs]]))
                                if hard_negs else float("nan"),
                                "n_hard_neg": len(hard_negs),
                            })
                    # ---- memory mutation (same causal rule as C1) ----
                    if q["first"]:
                        vid = mem.create_novel(z[0].detach().cpu().numpy().astype(np.float32),
                                               created_at=len(mem.novel))
                        novel_vid_by_label[q["label"]] = vid
                    else:
                        vid = novel_vid_by_label.get(q["label"])
                        created_now = False
                        if vid is None:
                            vid = mem.create_novel(z[0].detach().cpu().numpy().astype(np.float32),
                                                   created_at=len(mem.novel))
                            novel_vid_by_label[q["label"]] = vid
                            created_now = True
                        cos_to_center = float(mem.novel[vid]["proto"] @ z_np)
                        mem.update_novel(vid, z_np, cos_to_center=cos_to_center,
                                         update_radius=args.update_radius,
                                         margin=margin_n)
                        if not created_now:
                            vid_pos = sorted(mem.novel).index(vid)
                            novel_target = torch.tensor([vid_pos], device=device)
                            loss = loss + args.lambda_novel * novel_metric_loss(
                                z, torch.as_tensor(P_novel_np, device=device),
                                novel_target)
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    if P_novel_np.shape[0] and not q["known"]:
                        hns_rows.append({
                            "epoch": epoch,
                            "first": int(q["first"]),
                            "mem_size": len(mem.novel),
                            "best_sim": best_n,
                            "second_sim": second_n,
                            "margin": margin_n,
                            "own_exists": int(own_vid is not None),
                            "n_hard_neg": len(hard_negs),
                            "hard_neg_sim_mean": float(np.mean(ns[[
                                sorted(mem.novel).index(v) for v in hard_negs]]))
                            if hard_negs else float("nan"),
                        })
                loss_acc = loss_acc + loss
                n_query += 1
            if n_query:
                opt.zero_grad()
                (loss_acc / n_query).backward()
                opt.step()
                total += float(loss_acc.item() / n_query)
                n_ep += 1
        print(f"epoch {epoch} loss {total / max(n_ep, 1):.4f}", flush=True)

    out_dir = ROOT / "runs" / "orbit_iam" / args.output_dir
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
        "compat_feats": args.compat_feats,
        "compat_dim": compat_dim,
        "freeze_mode": args.freeze_mode,
        "init_checkpoint": args.init_checkpoint,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")

    if hns_rows:
        out = ROOT / "outputs" / "iclr27_phase4e" / "training"
        out.mkdir(parents=True, exist_ok=True)
        keys = list(hns_rows[0].keys())
        csv_path = out / "hard_negative_statistics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(hns_rows)
        (out_dir / "hard_negative_statistics.csv").write_text(
            csv_path.read_text())
    if pair_stats:
        out = ROOT / "outputs" / "iclr27_phase4e" / "training"
        out.mkdir(parents=True, exist_ok=True)
        keys = list(pair_stats[0].keys())
        with open(out / "positive_negative_similarity.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(pair_stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="I1")
    ap.add_argument("--init_checkpoint",
                    default="runs/orbit_msr/msr_nr2/model.pth")
    ap.add_argument("--compat_feats", default="sim,margin,radius,support,mem,rel")
    ap.add_argument("--freeze_mode", choices=["compat", "full"], default="compat")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lambda_reuse", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--lambda_compat", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=2.0)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--hard_neg_k", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--real_pairs", action="store_true")
    ap.add_argument("--no_hard_negatives", action="store_true",
                    help="ablation A1: random different-prototype negatives "
                         "instead of nearest-wrong")
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
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
    ap.add_argument("--output_dir", default="iam_i1")
    args = ap.parse_args()
    train(args)
