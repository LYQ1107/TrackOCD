"""Train ORBIT-MDC with a model-generated causal memory rollout.

Two memory regimes:
  teacher:  memory mutation follows pseudo-label (Phase 4E protocol, R0);
            known queries never mutate novel memory.
  onpolicy: memory mutation follows the model's own predicted action for
            EVERY track (KNOWN -> no novel mutation, EXISTING -> update the
            chosen prototype, NEW -> create prototype).  In particular a
            known track misrouted as novel creates/updates novel memory,
            exactly like deployment.  GT/pseudo labels are used ONLY for
            the loss of the current decision, never to repair history.

Optional components:
  --use_birth_head: learned reuse/birth decision on relative evidence.
  --real_hard_negatives: band-filtered same-looking different-class pairs
                         from real train-known classes (M2).
  --quarantine q1/q2: bounded influence for low-support/unstable prototypes.
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
from src.orbit_msr.losses import (
    gate_margin_loss,
    known_loss,
    novel_metric_loss,
)
from src.orbit_msr.protocol import known_stats, stats_to_tensor
from src.orbit_msr.train import _geo
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_mdc.model import BIRTH_FEAT_ORDER, ORBITMDCModel, build_birth_features


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


def quarantine_influence(state, mode):
    """Bounded influence multiplier for a prototype's compatibility q.

    q1: support + recent stability (low-margin rate) bounded influence.
    q2: q1 + dispersion penalty.
    Return 1.0 for mode 'none'.
    """
    if mode == "none":
        return 1.0
    support = float(state["support"])
    low_margin = float(state["low_margin_rate"])
    inf = min(support / 2.0, 1.0) * (1.0 - 0.3 * low_margin)
    if mode == "q2":
        disp = float(state["dispersion"])
        inf = inf * math.exp(-max(disp - 0.3, 0.0) / 0.3)
    return float(max(min(inf, 1.0), 0.0))


def real_band_negative_pairs(z_np, cls, classes, protos, by_class, z_cache,
                             band, hard_neg_k, rng):
    """Band-filtered same-looking different-class prototypes (train-side)."""
    sims = np.stack([protos[c] for c in classes]) @ z_np
    own = classes.index(cls)
    order = np.argsort(sims)[::-1]
    lo, hi = band
    cand = [int(i) for i in order if int(i) != own and lo <= sims[i] <= hi]
    if len(cand) < hard_neg_k:
        rest = [int(i) for i in order if int(i) != own and i not in cand]
        rng.shuffle(rest)
        cand += rest[:hard_neg_k - len(cand)]
    return cand[:hard_neg_k]


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_names = [f.strip() for f in args.compat_feats.split(",") if f.strip()]
    birth_feats = [f.strip() for f in args.birth_feats.split(",") if f.strip()]
    compat_dim = len(feat_names)
    birth_dim = len(birth_feats) if args.use_birth_head else 0
    gate_dim = 11
    reuse_dim = 13 if args.mem_scale_norm else 11
    model = ORBITMDCModel(dim=768, bottleneck=args.bottleneck,
                          gate_dim=gate_dim, reuse_dim=reuse_dim,
                          hidden=64, use_adapter=True,
                          compat_dim=compat_dim, birth_dim=birth_dim).to(device)
    ck = torch.load(args.init_checkpoint, map_location="cpu")
    sd = model.state_dict()
    for k, v in ck["state_dict"].items():
        if k in sd and v.shape == sd[k].shape:
            sd[k] = v
    model.load_state_dict(sd)
    model.train()

    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if args.freeze_mode == "compat" and name.startswith("compat."):
            p.requires_grad_(True)
            trainable.append(p)
        elif args.freeze_mode == "gate_compat" and (
                name.startswith("gate.") or name.startswith("compat.")
                or (model.birth is not None and name.startswith("birth."))):
            p.requires_grad_(True)
            trainable.append(p)
        elif args.freeze_mode == "full":
            p.requires_grad_(True)
            trainable.append(p)
    if not trainable:
        raise RuntimeError("no trainable parameters")
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
    band = (args.band_lo, args.band_hi)
    pair_stats = []
    action_trace = []  # per-query decision trace (for causality tests/audit)

    def refresh_z_pool():
        z_cache.clear()
        with torch.no_grad():
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

    def novel_path(q, z, z_np, rel, length, mem, P_known, radii,
                   novel_vid_by_label, epoch, loss):
        """Full non-known decision path: compat, birth/reuse, pair loss,
        memory mutation.  Returns (loss, action)."""
        P_novel_np = (np.stack([mem.novel[c]["proto"]
                                for c in sorted(mem.novel)])
                      .astype(np.float32)) if mem.novel else np.empty(
            (0, 768), dtype=np.float32)
        best_n = -1.0
        second_n = -1.0
        margin_n = 0.0
        ns = None
        order_n = []
        if P_novel_np.shape[0]:
            ns = P_novel_np @ z_np
            best_n = float(ns.max())
            order_n = np.argsort(ns)[::-1]
            second_n = float(ns[order_n[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
        ks = P_known @ z_np
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        own_vid = novel_vid_by_label.get(q["label"])
        if own_vid is not None and own_vid not in mem.novel:
            own_vid = None
        q_best = -1.0
        q_second = -1.0
        best_vid = None
        best_support = 0.0
        best_disp = 0.5
        states = {vid: mem.state(vid) for vid in sorted(mem.novel)}
        if P_novel_np.shape[0]:
            X_rows = []
            for vid in sorted(mem.novel):
                st = states[vid]
                X_rows.append(build_compat_features(
                    z_np, mem.novel[vid]["proto"], st["radius"],
                    st["support"], st["conf"], len(mem.novel), rel,
                    margin_n, feat_names))
            X = torch.as_tensor(np.asarray(X_rows, dtype=np.float32),
                                device=device)
            q_logits = model.compat_forward(X)
            q_probs = torch.sigmoid(q_logits.detach()).cpu().numpy()
            if args.quarantine != "none":
                infs = np.asarray(
                    [quarantine_influence(states[v], args.quarantine)
                     for v in sorted(mem.novel)], dtype=np.float32)
                q_scaled = q_probs * infs
            else:
                q_scaled = q_probs
            qorder = np.argsort(q_scaled)[::-1]
            q_best = float(q_scaled[qorder[0]])
            q_second = (float(q_scaled[qorder[1]])
                        if q_scaled.shape[0] >= 2 else -1.0)
            best_vid = int(sorted(mem.novel)[int(qorder[0])])
            best_support = states[best_vid]["support"]
            best_disp = states[best_vid]["dispersion"]
        # birth/reuse decision
        birth_logit = None
        if args.use_birth_head and len(mem.novel) > 0:
            bf = build_birth_features(
                q_best, q_second,
                math.log1p(best_support) / math.log1p(300.0),
                best_disp, rel,
                math.log1p(len(mem.novel)) / math.log1p(300.0),
                birth_feats)
            birth_logit = model.birth_forward(_t([bf], device))
            reuse = float(torch.sigmoid(birth_logit)[0].item()) >= args.birth_thr
            birth_target = torch.tensor([1.0 if own_vid is not None else 0.0],
                                        device=device)
            loss = loss + args.lambda_birth * torch.nn.functional.binary_cross_entropy_with_logits(
                birth_logit, birth_target)
        else:
            reuse = (q_best >= args.compat_thr
                     and (len(mem.novel) < 2
                          or q_best - q_second >= args.compat_margin))
        # identity compatibility pair loss (raw q logits on current memory)
        if P_novel_np.shape[0]:
            vids = [int(sorted(mem.novel)[o]) for o in order_n]
            neg_vids = [v for v in vids if v != own_vid]
            hard_negs = neg_vids[:args.hard_neg_k]
            if len(hard_negs) < args.hard_neg_k:
                rest = [v for v in sorted(mem.novel)
                        if v not in hard_negs and v != own_vid]
                rng.shuffle(rest)
                hard_negs += rest[:args.hard_neg_k - len(hard_negs)]
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
                    st["support"], st["conf"], len(mem.novel), rel,
                    margin_n, feat_names))
            if feat_rows:
                Xp = torch.as_tensor(np.asarray(feat_rows, dtype=np.float32),
                                     device=device)
                qp_logits = model.compat_forward(Xp)
                yp = torch.as_tensor(targets, dtype=torch.float32,
                                     device=device)
                pos_w = torch.as_tensor(
                    [args.pos_weight if t == 1.0 else 1.0
                     for t in targets], dtype=torch.float32, device=device)
                loss = loss + args.lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
                    qp_logits, yp, weight=pos_w)
                if own_vid is not None and len(qp_logits) >= 2:
                    loss = loss + args.lambda_rank * torch.relu(
                        args.ranking_margin - qp_logits[0]
                        + qp_logits[1:].mean())
                if own_vid is not None:
                    pos_sim = float(np.dot(z_np, mem.novel[own_vid]["proto"]))
                else:
                    pos_sim = float("nan")
                pair_stats.append({
                    "epoch": epoch,
                    "first": int(q["first"]),
                    "known": int(q["known"]),
                    "mem_size": len(mem.novel),
                    "positive_sim": pos_sim,
                    "n_hard_neg": len(hard_negs),
                })
        # memory mutation: onpolicy follows predicted action for all tracks;
        # teacher follows pseudo-label only for pseudo-novel queries.
        action = "EXISTING_NOVEL" if (reuse and best_vid is not None) else "NEW_NOVEL"
        if args.mode == "onpolicy":
            if reuse and best_vid is not None:
                cos_to_center = float(np.dot(mem.novel[best_vid]["proto"], z_np))
                mem.update_novel(best_vid, z_np, cos_to_center=cos_to_center,
                                 update_radius=args.update_radius,
                                 margin=margin_n)
                if best_vid == own_vid and P_novel_np.shape[0]:
                    novel_target = torch.tensor(
                        [sorted(mem.novel).index(own_vid)], device=device)
                    loss = loss + args.lambda_novel * novel_metric_loss(
                        z, torch.as_tensor(P_novel_np, device=device),
                        novel_target)
            else:
                vid = mem.create_novel(
                    z[0].detach().cpu().numpy().astype(np.float32),
                    created_at=len(mem.novel))
                if own_vid is None:
                    novel_vid_by_label[q["label"]] = vid
        else:  # teacher-forced
            if not q["known"]:
                if q["first"] or own_vid is None:
                    vid = mem.create_novel(
                        z[0].detach().cpu().numpy().astype(np.float32),
                        created_at=len(mem.novel))
                    novel_vid_by_label[q["label"]] = vid
                else:
                    cos_to_center = float(np.dot(mem.novel[own_vid]["proto"], z_np))
                    mem.update_novel(own_vid, z_np,
                                     cos_to_center=cos_to_center,
                                     update_radius=args.update_radius,
                                     margin=margin_n)
                    if P_novel_np.shape[0]:
                        novel_target = torch.tensor(
                            [sorted(mem.novel).index(own_vid)], device=device)
                        loss = loss + args.lambda_novel * novel_metric_loss(
                            z, torch.as_tensor(P_novel_np, device=device),
                            novel_target)
        action_trace.append({
            "epoch": epoch, "label": q["label"], "known": int(q["known"]),
            "first": int(q["first"]), "action": action,
            "gate_known": int(q.get("_gate_known", 0)),
            "q_best": q_best, "q_second": q_second,
            "mem_before": len(mem.novel) - (1 if action == "NEW_NOVEL" else 0),
        })
        return loss, action

    refresh_z_pool()
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
            refresh_z_pool()
        protos, radii = build_known()
        P_known = np.stack([protos[c] for c in classes]).astype(np.float32)
        total = 0.0
        n_ep = 0
        for _ in range(args.episodes_per_epoch):
            n_syn = args.syn_novel_classes
            syn_pool = list(all_feats.keys())
            rng.shuffle(syn_pool)
            syn_queries = []
            for k, sid in enumerate(syn_pool[:n_syn]):
                base_z = z_cache[sid]
                tracks = make_synthetic_tracks(np_rng, base_z, n_tracks=4,
                                               alpha_range=(0.35, 0.65),
                                               sigma=args.sigma)
                lab = 1000000 + k
                for j, frames in enumerate(tracks):
                    syn_queries.append({
                        "sample_id": f"syn_{sid}_{j}",
                        "label": lab, "known": False,
                        "first": j < 2, "_frames": frames,
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
            # near-miss confusers (same protocol as Phase 4E v3)
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
                              .astype(np.float32)) if mem.novel else np.empty(
                    (0, 768), dtype=np.float32)
                best_n = -1.0
                second_n = -1.0
                margin_n = 0.0
                dist_n = 1.0
                if P_novel_np.shape[0]:
                    ns = P_novel_np @ z_np
                    best_n = float(ns.max())
                    order_n = np.argsort(ns)[::-1]
                    second_n = float(ns[order_n[1]]) if ns.shape[0] >= 2 else best_n
                    margin_n = best_n - second_n
                    r_n = mem.novel_radii.get(int(order_n[0]), 0.3)
                    dist_n = (1.0 - best_n) / max(r_n, 1e-6)
                gs = known_stats(z_np, P_known, radii, known_ids=classes,
                                 best_n=best_n, second_n=second_n,
                                 margin_n=margin_n, dist_n=dist_n, rel=rel,
                                 track_len=length, n_novel=len(mem.novel),
                                 include_anchor=False)
                gate_target = torch.tensor([1.0 if q["known"] else 0.0],
                                           device=device)
                gate_logit = model.gate_forward(stats_to_tensor(gs, device))
                loss = args.lambda_gate * gate_margin_loss(
                    gate_logit, gate_target, margin=1.0)
                gate_prob = float(torch.sigmoid(gate_logit)[0].item())
                pred_known = gate_prob >= args.gate_thr
                q["_gate_known"] = int(pred_known)

                if q["known"]:
                    known_target = torch.tensor([classes.index(q["label"])],
                                                device=device)
                    loss = loss + args.lambda_known * known_loss(
                        z, torch.as_tensor(P_known, device=device), known_target)
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    if args.real_hard_negatives:
                        neg_ids = real_band_negative_pairs(
                            z_np, int(q["label"]), classes, protos, by_class,
                            z_cache, band, args.hard_neg_k, rng)
                        rows = []
                        targets = []
                        rows.append(0)
                        targets.append(0.0)
                        for pid in neg_ids:
                            c = classes[pid]
                            zs_c = np.stack([z_cache[sid] for sid in by_class[c]])
                            proto = protos[c]
                            support = len(by_class[c])
                            disp = float(np.mean(1.0 - zs_c @ proto))
                            conf = (math.log1p(support) / math.log1p(20.0)
                                    * math.exp(-disp / 0.3))
                            radius = float(np.percentile(1.0 - zs_c @ proto, 50))
                            rows.append(build_compat_features(
                                z_np, proto, radius, support, conf,
                                len(mem.novel), rel, margin_n, feat_names))
                            targets.append(0.0)
                        own_c = classes.index(q["label"])
                        zs_c = np.stack([z_cache[sid] for sid in by_class[own_c]])
                        disp = float(np.mean(1.0 - zs_c @ protos[own_c]))
                        conf = (math.log1p(len(by_class[own_c])) / math.log1p(20.0)
                                * math.exp(-disp / 0.3))
                        radius = float(np.percentile(
                            1.0 - zs_c @ protos[own_c], 50))
                        rows[0] = build_compat_features(
                            z_np, protos[own_c], radius, len(by_class[own_c]),
                            conf, len(mem.novel), rel, margin_n, feat_names)
                        targets[0] = 1.0
                        X = torch.as_tensor(np.asarray(rows, dtype=np.float32),
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
                        if len(q_logits) >= 2:
                            loss = loss + args.lambda_rank * torch.relu(
                                args.ranking_margin - q_logits[0]
                                + q_logits[1:].mean())
                    # on-policy: a known track misrouted as novel must
                    # follow the full novel decision path and mutate memory
                    # exactly like deployment.
                    if args.mode == "onpolicy" and not pred_known:
                        loss, _ = novel_path(q, z, z_np, rel, length, mem,
                                             P_known, radii,
                                             novel_vid_by_label, epoch, loss)
                else:
                    if not pred_known:
                        loss, _ = novel_path(q, z, z_np, rel, length, mem,
                                             P_known, radii,
                                             novel_vid_by_label, epoch, loss)
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

    out_dir = ROOT / "runs" / "orbit_mdc" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "variant": args.variant,
        "mode": args.mode,
        "bottleneck": args.bottleneck,
        "gate_dim": gate_dim,
        "reuse_dim": reuse_dim,
        "compat_dim": compat_dim,
        "compat_feats": args.compat_feats,
        "birth_dim": birth_dim,
        "birth_feats": args.birth_feats,
        "use_anchor": False,
        "balanced": True,
        "margin": True,
        "weight_new": 1.0,
        "mem_scale_norm": args.mem_scale_norm,
        "update_radius": args.update_radius,
        "novel_update_rate": args.novel_update_rate,
        "quarantine": args.quarantine,
        "gate_thr": args.gate_thr,
        "birth_thr": args.birth_thr,
        "compat_thr": args.compat_thr,
        "compat_margin": args.compat_margin,
        "band_lo": args.band_lo,
        "band_hi": args.band_hi,
        "seed": args.seed,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
    }, out_dir / "model.pth")
    if pair_stats:
        import csv
        keys = list(pair_stats[0].keys())
        with open(ROOT / "outputs" / "iclr27_phase4f" / "training" /
                  f"{args.output_dir}_pair_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(pair_stats)
    with open(ROOT / "outputs" / "iclr27_phase4f" / "training" /
              f"{args.output_dir}_action_trace.json", "w") as f:
        import json
        json.dump(action_trace, f, indent=1)
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_checkpoint", required=True)
    ap.add_argument("--mode", choices=["onpolicy", "teacher"], default="onpolicy")
    ap.add_argument("--freeze_mode", choices=["compat", "gate_compat", "full"],
                    default="gate_compat")
    ap.add_argument("--use_birth_head", action="store_true")
    ap.add_argument("--real_hard_negatives", action="store_true")
    ap.add_argument("--quarantine", choices=["none", "q1", "q2"], default="none")
    ap.add_argument("--compat_feats", default="sim,margin,radius,support,mem,rel")
    ap.add_argument("--birth_feats", default=",".join(BIRTH_FEAT_ORDER))
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--episodes_per_epoch", type=int, default=6)
    ap.add_argument("--syn_novel_classes", type=int, default=12)
    ap.add_argument("--known_per_class", type=int, default=2)
    ap.add_argument("--hard_neg_k", type=int, default=4)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--lambda_compat", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=2.0)
    ap.add_argument("--lambda_birth", type=float, default=1.0)
    ap.add_argument("--lambda_gate", type=float, default=1.0)
    ap.add_argument("--lambda_known", type=float, default=0.5)
    ap.add_argument("--lambda_novel", type=float, default=0.5)
    ap.add_argument("--lambda_geo", type=float, default=0.3)
    ap.add_argument("--ranking_margin", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--novel_update_rate", type=float, default=0.2)
    ap.add_argument("--mem_scale_norm", action="store_true")
    ap.add_argument("--update_radius", action="store_true")
    ap.add_argument("--gate_thr", type=float, default=0.5)
    ap.add_argument("--birth_thr", type=float, default=0.5)
    ap.add_argument("--compat_thr", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--band_lo", type=float, default=0.5)
    ap.add_argument("--band_hi", type=float, default=0.8)
    ap.add_argument("--variant", default="mdc")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    train(args)
