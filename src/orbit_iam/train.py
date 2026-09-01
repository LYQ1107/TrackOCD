"""Train ORBIT-IAM: identity-aware compatibility on top of ORBIT-MSR C1.

The C1 mechanisms are preserved (48-known gate training, synthetic ambiguous
novel classes, memory-scale padding, balanced gate, factorized causal
decision, support-aware radius, memory-scale normalization).  A small
pairwise identity-compatibility head is trained with:

* same-class compatibility (BCE target 1 against the class prototype),
* different-class compatibility (BCE target 0 against wrong prototypes,
  optionally the top-k nearest wrong prototypes = memory-conditioned hard
  negatives),
* a ranking margin between the positive compatibility and the hardest
  negative.

Pair labels come exclusively from train-side pseudo-novel classes; GT
official labels never enter training.
"""
from __future__ import annotations

import argparse
import math
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


def pair_features(z_np, P_novel, states, active_n, rel, track_len, use_conf):
    """Per-candidate pair statistics for the compatibility head.

    Columns: sim, candidate-relative margin, radius norm, support norm,
    (confidence if use_conf), memory-scale norm, reliability, track len norm.
    """
    if P_novel.shape[0] == 0:
        return np.empty((0, 7 + int(use_conf)), dtype=np.float32)
    ns = P_novel @ z_np
    M = ns.shape[0]
    sims = ns
    # candidate-relative margin: sim_j - max_{k != j} sim_k
    if M >= 2:
        top2 = np.sort(ns)[::-1][:2]
        margin = np.where(ns == top2[0], top2[0] - top2[1],
                          ns - top2[0]).astype(np.float32)
    else:
        margin = np.zeros(1, dtype=np.float32)
    radius = np.clip(np.array([states[j]["radius"] for j in range(M)],
                              dtype=np.float32) / 0.5, 0.0, 1.0)
    support = np.array([math.log1p(states[j]["support"])
                        / math.log1p(300.0) for j in range(M)],
                       dtype=np.float32)
    m_scale = np.full(M, math.log1p(active_n) / math.log1p(300.0),
                      dtype=np.float32)
    rels = np.full(M, rel, dtype=np.float32)
    lens = np.full(M, float(track_len) / 40.0, dtype=np.float32)
    cols = [sims, margin, radius, support]
    if use_conf:
        conf = np.array([states[j]["confidence"] for j in range(M)],
                        dtype=np.float32)
        cols.append(conf)
    cols += [m_scale, rels, lens]
    return np.stack(cols, axis=1).astype(np.float32)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate_dim = 11
    reuse_dim = 11 + (2 if args.mem_scale_norm else 0)
    compat_dim = 7 + (1 if args.conf_feature else 0) if args.compat else 0
    model = ORBITFCModel(dim=768, bottleneck=args.bottleneck,
                         gate_dim=gate_dim, reuse_dim=reuse_dim, hidden=64,
                         use_adapter=True, compat_dim=compat_dim).to(device)
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
            known_queries = []
            for c in classes:
                ids = list(by_class[c])
                rng.shuffle(ids)
                for sid in ids[:args.known_per_class]:
                    known_queries.append({"sample_id": sid, "label": c,
                                          "known": True, "first": False,
                                          "_frames": None})
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
            proto_state = {}
            for vid in range(len(pad_protos)):
                proto_state[vid] = {
                    "created_at": 0, "support": pad_counts[vid],
                    "dispersion": 0.0, "disp_n": 0,
                    "mean_margin": 0.0, "margin_n": 0,
                    "min_margin": 1.0, "low_margin_count": 0,
                    "recent_margins": [],
                }

            query = known_queries + syn_queries
            rng.shuffle(query)
            loss_acc = 0.0
            n_query = 0
            for q in query:
                frames = (q.get("_frames") if q.get("_frames") is not None
                          else all_feats[q["sample_id"]][:8])
                out = aggregate_one(model, frames, device)
                z = out["z"]
                z0 = out["z0"]
                rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
                length = int(out["length"][0])
                z_np = z[0].detach().cpu().numpy()
                P_novel_np = (np.stack(novel_list).astype(np.float32)
                              if novel_list else np.empty((0, 768),
                                                          dtype=np.float32))
                best_n = -1.0
                second_n = -1.0
                margin_n = 0.0
                dist_n = 1.0
                ns_order = []
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    ns_order = [int(j) for j in order]
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
                gate_target = torch.tensor([1.0 if q["known"] else 0.0],
                                           device=device)
                loss = (gate_margin_loss(gate_logit, gate_target, margin=1.0)
                        if args.margin else gate_bce(gate_logit, gate_target))

                known_target = None
                novel_target = None
                if q["known"]:
                    known_target = torch.tensor([classes.index(q["label"])],
                                                device=device)
                    P_known_t = torch.as_tensor(P_known, dtype=torch.float32,
                                                device=device)
                    loss = (loss + args.lambda_known
                            * known_loss(z, P_known_t, known_target)
                            + args.lambda_geo * _geo(z, z0))
                else:
                    ks = P_known @ z_np
                    best_k = float(ks.max()) if ks.shape[0] else -1.0
                    margin_k = _margin(ks)
                    rs = novel_stats(z_np, P_novel_np, novel_counts,
                                     novel_radii,
                                     best_k=best_k, margin_k=margin_k, rel=rel,
                                     track_len=length, n_novel=len(novel_list),
                                     mem_scale_norm=args.mem_scale_norm)
                    reuse_logit = model.reuse_forward(stats_to_tensor(rs, device))
                    reuse_target = torch.tensor([0 if q["first"] else 1],
                                                device=device)
                    loss = loss + args.lambda_reuse * reuse_bce(
                        reuse_logit, reuse_target, weight_new=args.weight_new)

                    # ---- identity compatibility pairs ----
                    if args.compat and P_novel_np.shape[0]:
                        states = [{
                            "radius": novel_radii.get(j, 0.3),
                            "support": novel_counts.get(j, 1),
                            "confidence": _conf(proto_state.get(
                                j, proto_state[0])),
                        } for j in range(len(novel_list))]
                        feats = pair_features(
                            z_np, P_novel_np, states, len(novel_list), rel,
                            length, args.conf_feature)
                        q_logits = model.compat_forward(
                            torch.as_tensor(feats, dtype=torch.float32,
                                            device=device))
                        M = len(novel_list)
                        same_vid = (None if q["first"]
                                    else novel_vid_by_label.get(q["label"]))
                        wrong = [j for j in range(M) if j != same_vid]
                        pos_vids = [] if same_vid is None else [same_vid]
                        neg_vids = []
                        if wrong:
                            if args.hard_negatives:
                                k = min(args.n_neg_hard, len(wrong))
                                ranked = [j for j in ns_order if j in wrong]
                                neg_vids = ranked[:k]
                                if len(neg_vids) < 2 and len(wrong) > k:
                                    neg_vids.append(
                                        rng.choice([j for j in wrong
                                                    if j not in neg_vids]))
                            else:
                                neg_vids = rng.sample(
                                    wrong, min(3, len(wrong)))
                        ids = pos_vids + neg_vids
                        if ids:
                            targets = [1.0] * len(pos_vids) + \
                                [0.0] * len(neg_vids)
                            tq = torch.as_tensor(
                                q_logits[ids], dtype=torch.float32,
                                device=device)
                            tt = torch.as_tensor(targets,
                                                 dtype=torch.float32,
                                                 device=device)
                            loss = loss + args.lambda_id * torch.nn.functional\
                                .binary_cross_entropy_with_logits(tq, tt)
                            if pos_vids and neg_vids:
                                hard_neg = q_logits[[neg_vids]].max()
                                rank = torch.relu(
                                    args.rank_margin
                                    - q_logits[same_vid] + hard_neg)
                                loss = loss + args.lambda_rank * rank

                    if q["first"]:
                        novel_list.append(z[0].detach().cpu().numpy()
                                          .astype(np.float32))
                        vid = len(novel_list) - 1
                        novel_vid_by_label[q["label"]] = vid
                        novel_counts[vid] = 1
                        novel_radii[vid] = 0.3
                        proto_state[vid] = {
                            "created_at": 0, "support": 1,
                            "dispersion": 0.0, "disp_n": 0,
                            "mean_margin": 0.0, "margin_n": 0,
                            "min_margin": 1.0, "low_margin_count": 0,
                            "recent_margins": [],
                        }
                    else:
                        vid = novel_vid_by_label.get(q["label"])
                        if vid is None:
                            novel_list.append(z[0].detach().cpu().numpy()
                                              .astype(np.float32))
                            vid = len(novel_list) - 1
                            novel_vid_by_label[q["label"]] = vid
                            novel_counts[vid] = 1
                            novel_radii[vid] = 0.3
                            proto_state[vid] = {
                                "created_at": 0, "support": 1,
                                "dispersion": 0.0, "disp_n": 0,
                                "mean_margin": 0.0, "margin_n": 0,
                                "min_margin": 1.0, "low_margin_count": 0,
                                "recent_margins": [],
                            }
                        proto_old = novel_list[vid]
                        cos_to_center = float(proto_old @ z_np)
                        new_proto = ((1 - args.novel_update_rate) * proto_old
                                     + args.novel_update_rate * z_np)
                        new_proto = new_proto / (np.linalg.norm(new_proto)
                                                 + 1e-12)
                        novel_list[vid] = new_proto.astype(np.float32)
                        novel_counts[vid] = novel_counts.get(vid, 1) + 1
                        if args.update_radius:
                            novel_radii[vid] = float(
                                (1 - 0.2) * novel_radii.get(vid, 0.3)
                                + 0.2 * max(1.0 - cos_to_center, 1e-3))
                        st = proto_state[vid]
                        d = max(1.0 - cos_to_center, 0.0)
                        st["disp_n"] += 1
                        st["dispersion"] = ((st["disp_n"] - 1)
                                            * st["dispersion"] + d) / st["disp_n"]
                        st["support"] = novel_counts[vid]
                        st["mean_margin"] = (st["mean_margin"] * st["margin_n"]
                                             + margin_n) / (st["margin_n"] + 1)
                        st["margin_n"] += 1
                        st["min_margin"] = min(st["min_margin"], margin_n)
                        if margin_n < 0.02:
                            st["low_margin_count"] += 1
                        st["recent_margins"].append(margin_n)
                        if len(st["recent_margins"]) > 12:
                            st["recent_margins"] = \
                                st["recent_margins"][-12:]
                        novel_target = torch.tensor([vid], device=device)
                        novel_protos_t = torch.as_tensor(
                            novel_list, dtype=torch.float32, device=device)
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

    out_dir = ROOT / "runs" / "orbit_iam" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "bottleneck": args.bottleneck,
        "gate_dim": gate_dim,
        "reuse_dim": reuse_dim,
        "compat_dim": compat_dim,
        "use_anchor": False,
        "balanced": args.balanced,
        "margin": args.margin,
        "weight_new": args.weight_new,
        "mem_scale_norm": args.mem_scale_norm,
        "update_radius": args.update_radius,
        "syn_novel_classes": args.syn_novel_classes,
        "novel_update_rate": args.novel_update_rate,
        "compat": args.compat,
        "conf_feature": args.conf_feature,
        "hard_negatives": args.hard_negatives,
        "seed": args.seed,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


def _conf(st):
    """Legal prototype confidence from training bookkeeping."""
    low_rate = st["low_margin_count"] / max(st["margin_n"], 1)
    recent = st["recent_margins"]
    if len(recent) >= 2:
        stability = math.exp(-min(float(np.std(recent)), 0.2) / 0.1)
    else:
        stability = 1.0
    return (math.log1p(st["support"]) / math.log1p(20.0)
            * math.exp(-st["dispersion"] / 0.3)
            * (1.0 - low_rate) * stability)


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
    ap.add_argument("--variant", default="IAM1")
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lambda_reuse", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--lambda_id", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=0.5)
    ap.add_argument("--rank_margin", type=float, default=0.1)
    ap.add_argument("--n_neg_hard", type=int, default=3)
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
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--conf_feature", action="store_true")
    ap.add_argument("--hard_negatives", action="store_true")
    ap.add_argument("--output_dir", default="iam_kg1")
    args = ap.parse_args()
    train(args)
