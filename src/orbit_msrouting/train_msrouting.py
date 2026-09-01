"""ORBIT-MSRouting on-policy causal rollout training (Phase 4G).

G0: Phase 4F M2 baseline (evidence-only gate).
G1: state-conditioned gate: gate input = known-evidence stats + selected
    current memory-state features.
G2: state-adaptive residual calibration:
        gate_logit_corrected = gate_logit(evidence) - b(S_t),
    with b a small MLP over the same memory-state features.

All variants keep the Phase 4F M2 on-policy protocol: the novel memory is
mutated by the model's own decisions (KNOWN -> no change; EXISTING ->
update chosen prototype; NEW -> create prototype), compatibility pairs and
real-band negatives are built against the CURRENT memory, history is never
repaired, and GT/pseudo labels are used only for the current query loss.
No oracle K, no future tracks, no retroactive relabeling.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.losses import gate_margin_loss, known_loss
from src.orbit_msr.protocol import known_stats, stats_to_tensor
from src.orbit_msr.train import _geo, make_synthetic_tracks
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_msrouting.model import build_msrouting_model
from src.orbit_msrouting.state_features import STATE_FEAT_ORDER, MemoryStateTracker


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def aggregate_one(model, frames, device):
    x = _t(frames[:8], device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    return model.aggregate(x, mask)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_names = [f.strip() for f in args.compat_feats.split(",") if f.strip()]
    state_names = [f.strip() for f in args.state_feats.split(",") if f.strip()]
    for n in state_names:
        assert n in STATE_FEAT_ORDER, f"bad state feat {n}"

    model, ck = build_msrouting_model(args.init_checkpoint, args.gate_mode,
                                      state_names, device)
    model.train()

    trainable = []
    for name, p in model.named_parameters():
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
        with torch.no_grad():
            for sid in all_feats:
                out = aggregate_one(model, all_feats[sid][:8], device)
                z_cache[sid] = out["z"][0].detach().cpu().numpy().astype(
                    np.float32)

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

    def novel_path(q, z, z0, z_np, rel, length, mem, P_known, radii,
                   own_vid_by_label, tracker, loss):
        """Full non-known decision path (compat pairs, model decision,
        memory mutation, birth loss).  Used for novel queries AND for known
        queries misrouted as novel, exactly like deployment."""
        P_novel_np = (np.stack([mem.novel[c]["proto"]
                                for c in sorted(mem.novel)])
                      .astype(np.float32)) if mem.novel else np.empty(
            (0, 768), dtype=np.float32)
        best_n = -1.0
        second_n = -1.0
        margin_n = 0.0
        ns = None
        if P_novel_np.shape[0]:
            ns = P_novel_np @ z_np
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
        own_vid = own_vid_by_label.get(q["label"])
        # ---- identity compatibility pairs against CURRENT memory ----
        pair_vids = []
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
            if own_vid is not None:
                pair_vids.insert(0, own_vid)
        if pair_vids:
            X_rows = []
            row_targets = []
            for vid in pair_vids:
                stv = mem.state(vid)
                X_rows.append(build_compat_features(
                    z_np, mem.novel[vid]["proto"], stv["radius"],
                    stv["support"], stv["conf"], len(mem.novel), rel,
                    margin_n, feat_names))
                row_targets.append(1.0 if vid == own_vid else 0.0)
            if args.real_band_neg_k:
                ks = P_known @ z_np
                korder = np.argsort(ks)[::-1]
                n_add = 0
                for o in korder:
                    c = classes[int(o)]
                    if float(ks[o]) < 0.45:
                        continue
                    stv = real_states[c]
                    X_rows.append(build_compat_features(
                        z_np, protos[c], stv["radius"], stv["support"],
                        stv["conf"], len(mem.novel), rel, margin_n,
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
                [args.pos_weight if t == 1.0 else 1.0
                 for t in row_targets], dtype=torch.float32, device=device)
            loss = loss + args.lambda_compat * torch.nn.functional.binary_cross_entropy_with_logits(
                q_logits, y, weight=pos_w)
            if own_vid is not None and len(q_logits) >= 2:
                loss = loss + args.lambda_rank * torch.relu(
                    args.ranking_margin - q_logits[0]
                    + q_logits[1:].mean())
        # ---- model-generated reuse/birth decision ----
        q_best = -1.0
        q_second = -1.0
        nid = None
        if P_novel_np.shape[0]:
            X = torch.as_tensor(np.asarray([
                build_compat_features(
                    z_np, mem.novel[v]["proto"], mem.state(v)["radius"],
                    mem.state(v)["support"], mem.state(v)["conf"],
                    len(mem.novel), rel, margin_n, feat_names)
                for v in sorted(mem.novel)]), dtype=torch.float32,
                device=device)
            q_vals = torch.sigmoid(model.compat_forward(X)).detach().cpu().numpy()
            if q_vals.shape[0]:
                qorder = np.argsort(q_vals)[::-1]
                q_best = float(q_vals[qorder[0]])
                q_second = (float(q_vals[qorder[1]])
                            if q_vals.shape[0] >= 2 else -1.0)
                nid = int(sorted(mem.novel)[int(qorder[0])])
        reuse = (q_best >= args.compat_thr
                 and (len(mem.novel) < 2
                      or q_best - q_second >= args.compat_margin))
        if reuse and nid is not None:
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z_np))
            mem.update_novel(nid, z_np, cos_to_center=cos_to_center,
                             update_radius=args.update_radius,
                             margin=margin_n)
            tracker.note_action("EXISTING_NOVEL", nid)
        else:
            vid = mem.create_novel(
                z[0].detach().cpu().numpy().astype(np.float32),
                created_at=len(mem.novel))
            tracker.note_action("NEW_NOVEL", vid)
            if own_vid is None:
                own_vid_by_label[q["label"]] = vid
        # ---- reuse/birth auxiliary loss ----
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
        return loss

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
            own_vid_by_label = {}
            state_tracker = MemoryStateTracker(window=args.window)

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
                state_vec = state_tracker.compute(mem, state_names)
                ev = _t([gs], device)
                st = (_t([state_vec], device)
                      if args.gate_mode in ("G1", "G2") else None)
                gate_logit = model.gate_logit(ev, st)
                gate_prob = float(torch.sigmoid(gate_logit))
                gate_target = torch.tensor([1.0 if q["known"] else 0.0],
                                           device=device)
                loss = (gate_margin_loss(gate_logit, gate_target, margin=1.0)
                        if args.margin else torch.zeros((), device=device))
                if q["known"]:
                    loss = loss + args.lambda_known * known_loss(
                        z, torch.as_tensor(P_known, device=device),
                        torch.tensor([classes.index(q["label"])], device=device))
                    loss = loss + args.lambda_geo * _geo(z, z0)
                    if gate_prob < args.gate_thr:
                        # on-policy: a known track misrouted as novel must
                        # follow the full novel path and mutate memory exactly
                        # like deployment (Phase 4F protocol).
                        loss = novel_path(
                            q, z, z0, z_np, rel, length, mem, P_known, radii,
                            own_vid_by_label, state_tracker, loss)
                    else:
                        state_tracker.note_action("KNOWN")
                    n_query += 1
                    loss_acc = loss_acc + loss
                    continue
                if gate_prob >= args.gate_thr:
                    state_tracker.note_action("KNOWN")
                else:
                    loss = novel_path(
                        q, z, z0, z_np, rel, length, mem, P_known, radii,
                        own_vid_by_label, state_tracker, loss)
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

    out_dir = ROOT / "runs" / "orbit_msrouting" / args.output_dir
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
        "state_dim": len(state_names),
        "state_feats": args.state_feats,
        "reuse_dim": 13,
    }, out_dir / "model.pth")
    print("saved", out_dir / "model.pth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="G1")
    ap.add_argument("--init_checkpoint",
                    default="runs/orbit_mdc/mdc_m2/model.pth")
    ap.add_argument("--compat_feats",
                    default="sim,margin,radius,support,conf,mem,rel")
    ap.add_argument("--gate_mode", choices=["G0", "G1", "G2"], default="G1")
    ap.add_argument("--state_feats", default=(
        "log_mem,low_support_ratio,mean_support,recent_birth_rate,"
        "high_disp_ratio"))
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--episodes_per_epoch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--margin", action="store_true")
    ap.add_argument("--update_radius", action="store_true")
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--hard_neg_k", type=int, default=4)
    ap.add_argument("--real_band_neg_k", type=int, default=2)
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
    ap.add_argument("--output_dir", default="msrouting_g1")
    args = ap.parse_args()
    train(args)
