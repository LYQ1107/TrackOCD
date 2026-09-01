"""ORBIT-MDC on-policy causal rollout training.

R0 (teacher-forced): Phase 4E training builds pseudo-novel memory from GT
pseudo-labels regardless of model decisions.
R1 (on-policy): the memory state is produced by the model's own gate /
compatibility / birth decisions; losses are computed against pseudo-labels
only for the current decision, never to repair history.

This script starts from the frozen Phase 4E Candidate A checkpoint and
fine-tunes gate/reuse/compat heads under model-generated memory rollouts
(aggregator frozen by default).  No oracle K, no future tracks, no
retroactive relabeling.
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

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.losses import gate_margin_loss, known_loss, reuse_bce
from src.orbit_msr.protocol import known_stats, stats_to_tensor
from src.orbit_msr.train import _geo, make_synthetic_tracks
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


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_names = [f.strip() for f in args.compat_feats.split(",") if f.strip()]
    gate_dim = 11 if args.gate_mode in ("base", "residual") else 11 + args.state_dim
    state_dim = args.state_dim if args.gate_mode == "residual" else 0
    model = ORBITIAMModel(dim=768, bottleneck=128, gate_dim=gate_dim,
                          reuse_dim=13,
                          hidden=64, use_adapter=True,
                          compat_dim=len(feat_names),
                          state_dim=state_dim).to(device)
    ck = torch.load(args.init_checkpoint, map_location="cpu")
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    model.load_state_dict(sd)
    # anchor initialization: new state inputs / residual bias start at zero
    # so the gate behaves exactly like the M2 base at initialization.
    with torch.no_grad():
        if args.gate_mode == "state":
            w = model.gate.net[0].weight
            w[:, -args.state_dim:] = 0.0
        if args.gate_mode == "residual":
            model.state_bias[-1].weight.zero_()
            model.state_bias[-1].bias.zero_()
    model.train()

    trainable = []
    for name, p in model.named_parameters():
        if args.gate_mode == "state" and not name.startswith("gate."):
            p.requires_grad_(False)
            continue
        if args.gate_mode == "residual" and not name.startswith("state_bias."):
            p.requires_grad_(False)
            continue
        if name.startswith("aggregator"):
            if not args.train_adapter:
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
    classes = sorted(by_class)
    np_rng = np.random.RandomState(args.seed + 1)
    rng = random.Random(args.seed)
    pad_options = [0, 50, 150, 300]
    z_cache = {}
    real_states = {}

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
            real_states[c] = {
                "radius": radii[c],
                "support": len(by_class[c]),
                "conf": (np.log1p(len(by_class[c])) / np.log1p(20.0)
                         * np.exp(-float(np.mean(1.0 - cos)) / 0.3)),
            }
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
            syn_queries = []
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                n_first = 2 if args.balanced else 1
                tracks = make_synthetic_tracks(np_rng, base_z, n_tracks=4,
                                               alpha_range=(0.35, 0.65),
                                               sigma=args.sigma)
                lab = 1000000 + k
                for j, frames in enumerate(tracks):
                    syn_queries.append({
                        "sample_id": f"syn_{sid}_{j}", "label": lab,
                        "known": False, "first": j < n_first,
                        "_frames": frames,
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
            mem = IamMemory(protos, radii,
                            novel_update_rate=args.novel_update_rate)
            for z_p in pad_pool[:n_pad]:
                vid = mem.create_novel(z_cache[z_p], created_at=-1)
                mem.novel_counts[vid] = int(rng.choice([1, 3, 10, 30]))
                mem.novel_radii[vid] = 0.3
            # near-miss confusers (as in Phase 4E v3)
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                alpha_c = float(np_rng.uniform(0.62, 0.82))
                w = np_rng.randn(768).astype(np.float32)
                w = w / (np.linalg.norm(w) + 1e-12)
                conf = alpha_c * base_z + (1.0 - alpha_c) * w
                conf = conf / (np.linalg.norm(conf) + 1e-12)
                vid = mem.create_novel(conf.astype(np.float32), created_at=-1)
                mem.novel_counts[vid] = int(rng.choice([1, 3, 5]))
                mem.novel_radii[vid] = 0.35
            # model-created prototype ids per pseudo-class (may be None if the
            # model failed to create it, or wrong if it merged elsewhere)
            own_vid_by_label = {}

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
                P_novel_np = (np.stack([mem.novel[c]["proto"]
                                        for c in sorted(mem.novel)])
                              .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
                best_n = -1.0; second_n = -1.0; margin_n = 0.0
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order = np.argsort(ns)[::-1]
                    second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                gs = known_stats(z_np, P_known, radii, known_ids=classes,
                                 best_n=best_n, second_n=second_n,
                                 margin_n=margin_n,
                                 dist_n=(1.0 - best_n) / max(
                                     mem.novel_radii.get(
                                         int(np.argsort(ns)[::-1][0]), 0.3), 1e-6)
                                 if P_novel_np.shape[0] else 1.0,
                                 rel=rel, track_len=length,
                                 n_novel=len(mem.novel), include_anchor=False)
                # ---- state-conditioned gate ----
                state = mem.state_summary()
                state_vec = [state["log_mem"], state["mean_support"],
                             state["low_support_ratio"],
                             state["mean_dispersion"]]
                if args.gate_mode == "state":
                    gs_ext = gs + state_vec
                    gate_logit = model.gate_forward(stats_to_tensor(gs_ext, device))
                else:
                    gate_logit = model.gate_forward(stats_to_tensor(gs, device))
                    if args.gate_mode == "residual":
                        gate_logit = model.gate_logit_with_bias(
                            gate_logit,
                            torch.as_tensor([state_vec], dtype=torch.float32,
                                            device=device))
                gate_prob = float(torch.sigmoid(gate_logit))
                gate_target = torch.tensor([1.0 if q["known"] else 0.0],
                                           device=device)
                gate_w = 1.0
                if args.mem_novel_w > 0 and not q["known"]:
                    gate_w = 1.0 + args.mem_novel_w * min(
                        state["log_mem"] / math.log1p(300.0), 1.0)
                loss = (gate_w * gate_margin_loss(gate_logit, gate_target,
                                                  margin=1.0)
                        if args.margin else torch.zeros((), device=device))
                if q["known"]:
                    loss = loss + args.lambda_known * known_loss(
                        z, torch.as_tensor(P_known, device=device),
                        torch.tensor([classes.index(q["label"])], device=device))
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    n_query += 1
                    loss_acc = loss_acc + loss
                    continue

                # ---- identity compatibility pairs against CURRENT memory ----
                train_compat = any(p.requires_grad for p in model.compat.parameters())
                own_vid = own_vid_by_label.get(q["label"])
                pair_vids = []
                targets = []
                if P_novel_np.shape[0]:
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
                if pair_vids and train_compat:
                    X_rows = []
                    row_targets = []
                    for vid in pair_vids:
                        st = mem.state(vid)
                        X_rows.append(build_compat_features(
                            z_np, mem.novel[vid]["proto"], st["radius"],
                            st["support"], st["conf"], len(mem.novel), rel,
                            margin_n, feat_names))
                        row_targets.append(1.0 if vid == own_vid else 0.0)
                    # real-band negatives: nearest real known-class prototypes
                    # (different pseudo-classes by construction) whose
                    # similarity lies in the real hard band.
                    if args.real_band_neg_k:
                        ks = P_known @ z_np
                        order = np.argsort(ks)[::-1]
                        n_add = 0
                        for o in order:
                            c = classes[int(o)]
                            if float(ks[o]) < 0.45:
                                continue
                            st = real_states[c]
                            X_rows.append(build_compat_features(
                                z_np, protos[c], st["radius"], st["support"],
                                st["conf"], len(mem.novel), rel, margin_n,
                                feat_names))
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
                        [args.pos_weight if t == 1.0 else 1.0 for t in row_targets],
                        dtype=torch.float32, device=device)
                    loss = loss + args.lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
                        q_logits, y, weight=pos_w)
                    if own_vid is not None and len(q_logits) >= 2:
                        loss = loss + args.lambda_rank * torch.relu(
                            args.ranking_margin - q_logits[0] + q_logits[1:].mean())

                # ---- model-generated decision (gate + compat) ----
                if gate_prob >= args.gate_thr:
                    action = "KNOWN"
                else:
                    q_best = -1.0
                    q_second = -1.0
                    nid = None
                    if P_novel_np.shape[0]:
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
                            qorder = np.argsort(q_vals)[::-1]
                            q_best = float(q_vals[qorder[0]])
                            q_second = float(q_vals[qorder[1]]) if q_vals.shape[0] >= 2 else -1.0
                            nid = int(sorted(mem.novel)[int(qorder[0])])
                    if (q_best >= args.compat_thr
                            and (len(mem.novel) < 2
                                 or q_best - q_second >= args.compat_margin)):
                        action = "EXISTING"
                    else:
                        action = "NEW"
                if action == "KNOWN":
                    pass  # no memory change
                elif action == "EXISTING" and nid is not None:
                    cos_to_center = float(np.dot(mem.novel[nid]["proto"], z_np))
                    mem.update_novel(nid, z_np, cos_to_center=cos_to_center,
                                     update_radius=args.update_radius,
                                     margin=margin_n)
                else:
                    vid = mem.create_novel(
                        z[0].detach().cpu().numpy().astype(np.float32),
                        created_at=len(mem.novel))
                    if own_vid is None:
                        own_vid_by_label[q["label"]] = vid
                    else:
                        # model already had a prototype for this class but the
                        # compat decision birthed a new one: keep the original
                        # mapping (semantic memory may hold both).
                        pass
                # ---- reuse/birth auxiliary loss (only when heads trainable) ----
                if train_compat:
                    if q["first"]:
                        birth_target = 1.0
                    else:
                        birth_target = 0.0 if own_vid is not None else 1.0
                    if P_novel_np.shape[0]:
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

    out_dir = ROOT / "runs" / "orbit_mdc" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "compat_feats": args.compat_feats,
        "compat_dim": len(feat_names),
        "seed": args.seed,
        "gate_thr": args.gate_thr,
        "compat_thr": args.compat_thr,
        "compat_margin": args.compat_margin,
        "train_adapter": args.train_adapter,
        "init_checkpoint": args.init_checkpoint,
        "gate_mode": args.gate_mode,
        "state_dim": state_dim,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="M1")
    ap.add_argument("--init_checkpoint",
                    default="runs/orbit_iam/iam_i2_v3/model.pth")
    ap.add_argument("--compat_feats",
                    default="sim,margin,radius,support,conf,mem,rel")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--episodes_per_epoch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--margin", action="store_true")
    ap.add_argument("--mem_scale_norm", action="store_true")
    ap.add_argument("--update_radius", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--hard_neg_k", type=int, default=4)
    ap.add_argument("--real_band_neg_k", type=int, default=0)
    ap.add_argument("--lambda_compat", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=2.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--lambda_birth", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--known_per_class", type=int, default=2)
    ap.add_argument("--syn_novel_classes", type=int, default=12)
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--gate_thr", type=float, default=0.5)
    ap.add_argument("--compat_thr", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--train_adapter", action="store_true")
    ap.add_argument("--gate_mode", choices=["base", "state", "residual"],
                    default="base")
    ap.add_argument("--state_dim", type=int, default=4)
    ap.add_argument("--mem_novel_w", type=float, default=0.0)
    ap.add_argument("--output_dir", default="mdc_m1")
    args = ap.parse_args()
    train(args)
