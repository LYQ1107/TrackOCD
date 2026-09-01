"""ORBIT-CHP: counterfactual hard pseudo-novel episodic on-policy training.

Episodes hide a subset of real train-known classes and treat them as
pseudo-novel.  H1=random leave-out, H2=hard leave-out (closest to the
episode-known prototypes), H3=mixed curriculum.  The M2 architecture is
unchanged; only the training distribution changes.  Held-out meta-dev
classes are never used in episodes.
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import (
    load_frame_features,
    load_train_labels,
    meta_classes,
)
from src.orbit_msr.losses import gate_margin_loss, known_loss, reuse_bce
from src.orbit_msr.protocol import known_stats, stats_to_tensor
from src.orbit_msr.train import _geo
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_iam.model import ORBITIAMModel


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def aggregate_one(model, frames, device):
    x = _t(frames[:8], device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    return model.aggregate(x, mask)


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_mode", choices=["random", "hard", "mixed"],
                    default="random")
    ap.add_argument("--variant", default="H1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--episodes_per_epoch", type=int, default=6)
    ap.add_argument("--episode_known", type=int, default=24)
    ap.add_argument("--episode_pseudo", type=int, default=8)
    ap.add_argument("--tracks_per_pseudo_class", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--hard_neg_k", type=int, default=4)
    ap.add_argument("--lambda_compat", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=2.0)
    ap.add_argument("--lambda_birth", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--real_band_neg_k", type=int, default=2)
    ap.add_argument("--output_dir", default="chp_h1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_names = ["sim", "margin", "radius", "support", "conf", "mem", "rel"]
    model = ORBITIAMModel(dim=768, bottleneck=128, gate_dim=11, reuse_dim=13,
                          hidden=64, use_adapter=True,
                          compat_dim=len(feat_names)).to(device)
    ck = torch.load(f"{ROOT}/runs/orbit_mdc/mdc_m2/model.pth",
                    map_location="cpu")
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    model.load_state_dict(sd)
    model.train()
    trainable = []
    for name, p in model.named_parameters():
        if name.startswith("aggregator"):
            p.requires_grad_(False)
            continue
        p.requires_grad_(True)
        trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    meta_train = set(meta_classes("meta_train_classes"))
    meta_dev = set(meta_classes("meta_dev_classes"))
    train_pool = sorted(c for c in by_class if int(c) in meta_train)
    held_out = sorted(c for c in by_class if int(c) in meta_dev)
    other = sorted(c for c in by_class if int(c) not in meta_train
                   and int(c) not in meta_dev)
    # include all non-held-out classes in the training pool (frozen split)
    train_pool = sorted(c for c in by_class if int(c) not in meta_dev)
    print("train_pool", len(train_pool), "held_out", len(held_out),
          "total", len(by_class), flush=True)

    np_rng = np.random.RandomState(args.seed + 1)
    rng = random.Random(args.seed)
    z_cache = {}
    real_states = {}

    def refresh_z_pool():
        z_cache.clear()
        for sid in all_feats:
            out = aggregate_one(model, all_feats[sid][:8], device)
            z_cache[sid] = out["z"][0].detach().cpu().numpy().astype(np.float32)

    def class_proto(c):
        zs = np.stack([z_cache[sid] for sid in by_class[c]])
        p = zs.mean(axis=0)
        return p / (np.linalg.norm(p) + 1e-12)

    def class_proto_radius(c):
        zs = np.stack([z_cache[sid] for sid in by_class[c]])
        p = class_proto(c)
        cos = zs @ p
        r = max(float(np.percentile(1.0 - cos, 50)), 0.02)
        real_states[c] = {
            "radius": r,
            "support": len(by_class[c]),
            "conf": (np.log1p(len(by_class[c])) / np.log1p(20.0)
                     * np.exp(-float(np.mean(1.0 - cos)) / 0.3)),
        }
        return p, r

    refresh_z_pool()
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            refresh_z_pool()
        model.train()
        total = 0.0
        n_ep = 0
        for _ in range(args.episodes_per_epoch):
            known_classes = rng.sample(train_pool, args.episode_known)
            pseudo_pool = [c for c in train_pool if c not in known_classes]
            if args.episode_mode == "random":
                pseudo_classes = rng.sample(pseudo_pool, args.episode_pseudo)
            else:
                # hardness: mean adapted best-known sim to episode-known protos
                known_protos = {c: class_proto(c) for c in known_classes}
                Pk = np.stack([known_protos[c] for c in known_classes]).astype(np.float32)
                hard = {}
                for c in pseudo_pool:
                    zs = np.stack([z_cache[sid] for sid in by_class[c]])
                    hard[c] = float(np.mean(np.max(zs @ Pk.T, axis=1)))
                ordered = sorted(pseudo_pool, key=lambda c: -hard[c])
                if args.episode_mode == "hard":
                    pseudo_classes = ordered[:args.episode_pseudo]
                else:  # mixed: easy + medium + hard
                    n = len(ordered)
                    third = max(n // 3, 1)
                    buckets = [ordered[:third], ordered[third:2 * third],
                               ordered[2 * third:]]
                    pseudo_classes = []
                    per = max(args.episode_pseudo // 3, 1)
                    for b in buckets:
                        pseudo_classes.extend(rng.sample(b, min(per, len(b))))
                    if len(pseudo_classes) < args.episode_pseudo:
                        rest = [c for c in ordered if c not in pseudo_classes]
                        pseudo_classes.extend(
                            rng.sample(rest, args.episode_pseudo - len(pseudo_classes)))
                    pseudo_classes = pseudo_classes[:args.episode_pseudo]
            # build episode queries
            queries = []
            for c in known_classes:
                ids = list(by_class[c])
                rng.shuffle(ids)
                for sid in ids[:2]:
                    queries.append({"sample_id": sid, "label": c, "known": True,
                                    "first": False})
            pseudo_queries = []
            for c in pseudo_classes:
                ids = list(by_class[c])
                rng.shuffle(ids)
                for j, sid in enumerate(ids[:args.tracks_per_pseudo_class]):
                    pseudo_queries.append({"sample_id": sid, "label": c,
                                           "known": False, "first": j == 0})
            rng.shuffle(queries)
            rng.shuffle(pseudo_queries)
            query = queries + pseudo_queries
            rng.shuffle(query)

            # memory: pads from non-episode classes for memory-scale coverage
            known_protos = {}
            known_radii = {}
            for c in known_classes:
                known_protos[c], known_radii[c] = class_proto_radius(c)
            mem = IamMemory(known_protos, known_radii,
                            novel_update_rate=0.2)
            pad_pool = [c for c in train_pool
                        if c not in known_classes and c not in pseudo_classes]
            rng.shuffle(pad_pool)
            for c in pad_pool[:rng.choice([0, 20, 60, 120])]:
                vid = mem.create_novel(class_proto(c).astype(np.float32),
                                       created_at=-1)
                mem.novel_counts[vid] = int(rng.choice([1, 3, 10]))
                mem.novel_radii[vid] = 0.3
            own_vid_by_label = {}
            loss_acc = 0.0
            n_query = 0
            for q in query:
                out = aggregate_one(model, all_feats[q["sample_id"]][:8], device)
                z = out["z"]
                z0 = out["z0"]
                rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
                length = int(out["length"][0])
                z_np = z[0].detach().cpu().numpy()
                P_known = np.stack([mem.known[c] for c in sorted(mem.known)]
                                   ).astype(np.float32)
                known_ids = sorted(mem.known)
                P_novel = (np.stack([mem.novel[c]["proto"]
                                     for c in sorted(mem.novel)])
                           .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
                best_n = second_n = -1.0
                margin_n = 0.0
                dist_n = 1.0
                if P_novel.shape[0]:
                    ns = P_novel @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                    nid_top = int(sorted(mem.novel)[int(order[0])])
                    dist_n = (1.0 - best_n) / max(
                        mem.novel_radii.get(nid_top, 0.3), 1e-6)
                gs = known_stats(z_np, P_known,
                                 {c: known_radii[c] for c in known_ids},
                                 known_ids=known_ids,
                                 best_n=best_n, second_n=second_n,
                                 margin_n=margin_n,
                                 dist_n=dist_n,
                                 rel=rel, track_len=length,
                                 n_novel=len(mem.novel), include_anchor=False)
                gate_logit = model.gate_forward(stats_to_tensor(gs, device))
                gate_target = torch.tensor([1.0 if q["known"] else 0.0],
                                           device=device)
                loss = gate_margin_loss(gate_logit, gate_target, margin=1.0)
                gate_prob = float(torch.sigmoid(gate_logit))
                if q["known"]:
                    loss = loss + args.lambda_known * known_loss(
                        z, torch.as_tensor(P_known, device=device),
                        torch.tensor([known_ids.index(q["label"])], device=device))
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    n_query += 1
                    loss_acc = loss_acc + loss
                    continue
                # identity compat pairs (current causal memory)
                own_vid = own_vid_by_label.get(q["label"])
                pair_vids = []
                targets = []
                if P_novel.shape[0]:
                    ns = P_novel @ z_np
                    order = np.argsort(ns)[::-1]
                    vids = [int(sorted(mem.novel)[o]) for o in order]
                    negs = [v for v in vids if v != own_vid][:args.hard_neg_k]
                    if len(negs) < args.hard_neg_k:
                        rest = [v for v in sorted(mem.novel)
                                if v not in negs and v != own_vid]
                        rng.shuffle(rest)
                        negs += rest[:args.hard_neg_k - len(negs)]
                    pair_vids = negs[:]
                    targets = [0.0] * len(pair_vids)
                    if own_vid is not None:
                        pair_vids.insert(0, own_vid)
                        targets.insert(0, 1.0)
                if pair_vids:
                    X_rows = []
                    row_targets = []
                    for vid in pair_vids:
                        st = mem.state(vid)
                        X_rows.append(build_compat_features(
                            z_np, mem.novel[vid]["proto"], st["radius"],
                            st["support"], st["conf"], len(mem.novel), rel,
                            margin_n, feat_names))
                        row_targets.append(1.0 if vid == own_vid else 0.0)
                    if args.real_band_neg_k:
                        ks = P_known @ z_np
                        order = np.argsort(ks)[::-1]
                        n_add = 0
                        for o in order:
                            c = known_ids[int(o)]
                            if float(ks[o]) < 0.45:
                                continue
                            st = real_states[c]
                            X_rows.append(build_compat_features(
                                z_np, known_protos[c], st["radius"],
                                st["support"], st["conf"], len(mem.novel),
                                rel, margin_n, feat_names))
                            row_targets.append(0.0)
                            n_add += 1
                            if n_add >= args.real_band_neg_k:
                                break
                    X = torch.as_tensor(np.asarray(X_rows, dtype=np.float32),
                                        device=device)
                    q_logits = model.compat_forward(X)
                    y = torch.as_tensor(row_targets, dtype=torch.float32,
                                        device=device)
                    pos_w = torch.as_tensor(
                        [args.pos_weight if t == 1.0 else 1.0
                         for t in row_targets],
                        dtype=torch.float32, device=device)
                    loss = loss + args.lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
                        q_logits, y, weight=pos_w)
                    if own_vid is not None and len(q_logits) >= 2:
                        loss = loss + args.lambda_rank * torch.relu(
                            args.ranking_margin - q_logits[0] + q_logits[1:].mean())
                # model-generated decision
                if gate_prob >= 0.5:
                    action = "KNOWN"
                else:
                    q_best = -1.0
                    q_second = -1.0
                    nid = None
                    if P_novel.shape[0]:
                        X = torch.as_tensor(np.asarray([
                            build_compat_features(
                                z_np, mem.novel[v]["proto"],
                                mem.state(v)["radius"], mem.state(v)["support"],
                                mem.state(v)["conf"], len(mem.novel), rel,
                                margin_n, feat_names)
                            for v in sorted(mem.novel)]), dtype=torch.float32,
                            device=device)
                        q_vals = torch.sigmoid(model.compat_forward(X)).detach().cpu().numpy()
                        if q_vals.shape[0]:
                            qo = np.argsort(q_vals)[::-1]
                            q_best = float(q_vals[qo[0]])
                            q_second = float(q_vals[qo[1]]) if q_vals.shape[0] >= 2 else -1.0
                            nid = int(sorted(mem.novel)[int(qo[0])])
                    if (q_best >= 0.45 and (len(mem.novel) < 2
                                            or q_best - q_second >= 0.05)):
                        action = "EXISTING"
                    else:
                        action = "NEW"
                if action == "KNOWN":
                    pass
                elif action == "EXISTING" and nid is not None:
                    cos = float(np.dot(mem.novel[nid]["proto"], z_np))
                    mem.update_novel(nid, z_np, cos_to_center=cos,
                                     update_radius=True, margin=margin_n)
                else:
                    vid = mem.create_novel(
                        z[0].detach().cpu().numpy().astype(np.float32),
                        created_at=len(mem.novel))
                    own_vid_by_label.setdefault(q["label"], vid)
                # birth evidence
                birth_target = 1.0 if (q["first"] or own_vid is None) else 0.0
                if P_novel.shape[0]:
                    v_best = int(sorted(mem.novel)[int(np.argmax(ns))])
                    stb = mem.state(v_best)
                    feat_row = build_compat_features(
                        z_np, mem.novel[v_best]["proto"], stb["radius"],
                        stb["support"], stb["conf"], len(mem.novel), rel,
                        margin_n, feat_names)
                else:
                    feat_row = [0.0] * len(feat_names)
                q_best_t = torch.sigmoid(model.compat_forward(
                    torch.as_tensor([feat_row], dtype=torch.float32,
                                    device=device)))
                loss = loss + args.lambda_birth * torch.nn.functional.binary_cross_entropy(
                    q_best_t.clamp(1e-6, 1 - 1e-6),
                    torch.tensor([birth_target], device=device))
                loss = loss + args.lambda_geo * _geo(z, z0)
                n_query += 1
                loss_acc = loss_acc + loss
            if n_query:
                opt.zero_grad()
                (loss_acc / n_query).backward()
                opt.step()
                total += float(loss_acc.item() / n_query)
                n_ep += 1
        print(f"epoch {epoch} loss {total / max(n_ep, 1):.4f}", flush=True)

    out_dir = ROOT / "runs" / "orbit_chp" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "episode_mode": args.episode_mode,
        "seed": args.seed,
        "compat_feats": ",".join(feat_names),
        "compat_dim": len(feat_names),
        "real_band_neg_k": args.real_band_neg_k,
        "gate_thr": 0.5,
        "compat_thr": 0.45,
        "compat_margin": 0.05,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    main()
