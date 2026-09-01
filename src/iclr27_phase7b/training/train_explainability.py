"""Train the Phase 7B TOSE unified evidence-competition head.

Strict-causal proxy-OOD episodes are built from supported-known categories:
each training chunk hides a subset of known classes (episodic pseudo-novel)
plus the disjoint novel_train classes; first proxy-novel track -> NEW,
subsequent physical tracks of the same proxy category -> EXISTING with the
correct slot. Checkpoint selection uses only the legal proxy-val split
(novel_val classes hidden). No true novel GT, no Q1/heldout information, no
future information is used.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase7b.model.explainability import (
    EMAMemory,
    TOSEHead,
    TOSELinearHead,
    TrackState,
    base_feature_names,
    tose_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def load_assets():
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/train_episodes.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/val_episodes.npz").items()}
    split = json.loads(
        (ROOT / "outputs/iclr27_phase7a/assets/class_split.json").read_text())
    return tr, va, split


def load_stats():
    return dict(np.load(
        ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))


def video_ranges(ep):
    vids = ep["video_ids"]
    out = {}
    starts = {}
    for i, v in enumerate(vids):
        if int(v) not in starts:
            starts[int(v)] = i
    for v, s in starts.items():
        e = len(vids)
        for j in range(s + 1, len(vids)):
            if int(vids[j]) != v:
                e = j
                break
        out[v] = (int(s), int(e))
    return out


def replay_track_ema(ep, z_all, alpha=0.30):
    ema = {}
    h_all = np.zeros_like(z_all)
    for i in range(len(z_all)):
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        e = ema.get(key)
        z = z_all[i]
        if e is None:
            e = z.copy()
        else:
            e = (1 - alpha) * e + alpha * z
            e /= (np.linalg.norm(e) + 1e-12)
        ema[key] = e
        h_all[i] = e
    return h_all


def precompute(h_all, anchors, stats):
    """Per-row full-K anchor sims and class-conditional log-likelihoods."""
    asims = h_all @ anchors.T
    diff = h_all[:, None, :] - stats["mu"][None, :, :]
    mahal2 = np.sum(diff * diff / stats["sigma2"][None, :, :], axis=2)
    loglik = -0.5 * (mahal2 + stats["logdet"][None, :])
    return asims.astype(np.float32), loglik.astype(np.float32)


def masked_top2(scores, mask):
    s = np.where(mask, scores, -1e18)
    order = np.argsort(-s)
    k = int(mask.sum())
    if k <= 0:
        return -1.0, 0.0, -1
    i1 = int(order[0])
    v1 = float(s[i1])
    v2 = float(s[order[1]]) if k >= 2 else v1
    return v1, v2, i1


def chunk_losses(chunk_rows, z_all, h_all, asims_all, loglik_all, ep, mem,
                 policy, anchors, known_ids, visible_cls, novel_cls,
                 track_stats, device, args, rng, stats,
                 train=True, model_slot_of_class=None, class_seen=None,
                 loss_buf=None, scores=None):
    if model_slot_of_class is None:
        model_slot_of_class = {}
    if class_seen is None:
        class_seen = set()
    visible_mask = np.isin(known_ids, np.asarray(sorted(visible_cls),
                                                 dtype=np.int64))
    if not visible_mask.any():
        raise RuntimeError("empty visible known set")
    n_sup = n_known = n_first = n_attach = 0
    acc_first = acc_reuse = acc_known = 0.0
    for ri in chunk_rows:
        if int(ep["gt_role"][ri]) == 0 and rng.random() > args.fp_keep:
            continue
        key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackState()
            track_stats[key] = ts
        cat = int(ep["gt_category_id"][ri])
        is_known_row = int(ep["gt_role"][ri]) == 1
        slot_class = -1
        target = None
        slot_target = None
        if is_known_row and cat in novel_cls:
            slot_class = cat
            birth_track_ok = False
            if cat in model_slot_of_class:
                bi = model_slot_of_class[cat]
                birth_track_ok = (
                    mem.birth_key_enc[bi]
                    == int(key[0]) * 1000000 + int(key[1]))
            if cat not in class_seen:
                target = 2
                class_seen.add(cat)
                n_first += 1
            elif cat in model_slot_of_class and (
                    not args.no_cross_track or birth_track_ok):
                target = 1
                slot_target = model_slot_of_class.get(cat)
                n_attach += 1
            else:
                target = 2
                n_first += 1
        elif is_known_row and cat in visible_cls:
            target = 0
            n_known += 1
        z = z_all[ri]
        h = h_all[ri]
        res = tose_step(
            policy, z, mem, anchors, visible_mask, known_ids, stats, ts,
            int(ep["frame_ids"][ri]), key, slot_class=slot_class,
            target_slot=slot_target,
            use_dist=args.use_dist, use_traj=args.use_traj,
            known_tau=args.known_tau if args.known_tau >= 0 else None,
            asims_row=asims_all[ri], loglik_row=loglik_all[ri])
        if scores is not None:
            scores[ri] = (res["kscore"], res["decision"])
        if res["decision"] == 2 and slot_class >= 0 and \
                res["slot_idx"] is not None:
            model_slot_of_class.setdefault(slot_class, res["slot_idx"])
        logits = res["logits"]
        if target is not None:
            loss = F.cross_entropy(logits.unsqueeze(0),
                                   torch.tensor([target], device=logits.device))
            if target == 0:
                loss = args.w_known * loss
                acc_known += (res["decision"] == 0)
                n_sup += 1
            elif target == 1:
                loss = args.w_novel * loss
                correct = (res["decision"] == 1
                           and slot_target is not None
                           and res["slot_idx"] == slot_target)
                acc_reuse += correct
                n_sup += 1
                if slot_target is not None and res["slot_logits"] is not None \
                        and res["slot_logits"].numel() > 0 \
                        and res["cand_idx"] is not None:
                    pos = int(np.flatnonzero(
                        res["cand_idx"] == slot_target)[0])
                    sl = res["slot_logits"] / args.slot_temp
                    loss = loss + args.w_slot * F.cross_entropy(
                        sl.unsqueeze(0),
                        torch.tensor([pos], device=sl.device))
            else:
                loss = args.w_novel * loss
                acc_first += (res["decision"] == 2)
                n_sup += 1
            loss_buf["ce"] += float(loss.detach())
            loss_buf["total"] += loss
        else:
            if not is_known_row:
                pen_target = 1 if res["logits"][1] >= res["logits"][2] else 2
                pen = F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([pen_target], device=logits.device))
                loss_buf["total"] += args.w_unlabeled * pen
                loss_buf["pen"] += float(pen.detach())
        loss_buf["n"] += 1
        loss_buf["new"] += (res["decision"] == 2)
    return {
        "n_sup": n_sup, "n_known": n_known, "n_first": n_first,
        "n_attach": n_attach,
        "known_acc": acc_known / max(n_known, 1),
        "first_acc": acc_first / max(n_first, 1),
        "reuse_acc": max(acc_reuse, 0.0) / max(n_attach, 1),
        "model_slot_of_class": model_slot_of_class,
        "class_seen": class_seen,
    }


def run_epoch(ep, z_all, h_all, asims_all, loglik_all, policy, anchors,
              known_ids, split, device, args, rng, stats, train=True):
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    if train:
        rng.shuffle(vids)
    ranges = video_ranges(ep)
    chunks = [vids[i:i + args.chunk_videos]
              for i in range(0, len(vids), args.chunk_videos)]
    agg = defaultdict(float)
    metric = defaultdict(float)
    policy.train(train)
    scores = {}
    for ci, cvids in enumerate(chunks):
        mem = EMAMemory(dim=128)
        track_stats = {}
        model_slot_of_class = {}
        class_seen = set()
        if train:
            if args.no_proxy:
                visible = set(split["known"]) | set(split["novel_val"])
                novel = set()
            else:
                kvis = min(args.visible_known_per_chunk, len(split["known"]))
                vis_known = set(rng.sample(sorted(split["known"]), kvis))
                visible = vis_known | set(split["novel_val"])
                novel = set(split["known"]) - vis_known
                novel |= set(split["novel_train"])
        else:
            visible = set(split["known"])
            novel = set(split["novel_val"])
        if train and args.track_cap_other > 0:
            by_track = defaultdict(list)
            for v in cvids:
                s, e = ranges[v]
                for ri in range(s, e):
                    by_track[(int(ep["video_ids"][ri]),
                              int(ep["track_ids"][ri]))].append(ri)
            novel_tracks, known_tracks, fp_tracks = [], [], []
            for key, idxs in by_track.items():
                roles = ep["gt_role"][idxs]
                cats = ep["gt_category_id"][idxs]
                has_novel = any(int(rl) == 1 and int(c) in novel
                                for rl, c in zip(roles, cats))
                has_known = any(int(rl) == 1 and int(c) in visible
                                for rl, c in zip(roles, cats))
                if has_novel:
                    novel_tracks.append(key)
                elif has_known:
                    known_tracks.append(key)
                else:
                    fp_tracks.append(key)
            rng.shuffle(known_tracks)
            rng.shuffle(fp_tracks)
            pick = novel_tracks[:args.track_cap_novel]
            pick += known_tracks[:args.track_cap_other]
            pick += fp_tracks[:args.track_cap_other]
            rowset = set()
            for key in pick:
                rowset.update(by_track[key])
            rows = sorted(rowset, key=lambda ri: (
                int(ep["video_ids"][ri]), int(ep["frame_ids"][ri]),
                int(ep["track_ids"][ri])))
        else:
            rows = []
            for v in cvids:
                s, e = ranges[v]
                rows.extend(range(s, e))
        for start in range(0, len(rows), args.grad_every):
            sub = rows[start:start + args.grad_every]
            loss_buf = defaultdict(float)
            out = chunk_losses(
                sub, z_all, h_all, asims_all, loglik_all, ep, mem, policy,
                anchors, known_ids, visible, novel, track_stats, device,
                args, rng, stats, train=train,
                model_slot_of_class=model_slot_of_class,
                class_seen=class_seen, loss_buf=loss_buf, scores=scores)
            agg["ce"] += loss_buf["ce"]
            agg["pen"] += loss_buf["pen"]
            agg["n"] += loss_buf["n"]
            agg["new"] += loss_buf["new"]
            metric["known_acc"] += out["known_acc"] * out["n_known"]
            metric["first_acc"] += out["first_acc"] * out["n_first"]
            metric["reuse_acc"] += out["reuse_acc"] * out["n_attach"]
            metric["n_known"] += out["n_known"]
            metric["n_first"] += out["n_first"]
            metric["n_attach"] += out["n_attach"]
            if train:
                loss_buf["total"].backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                args.optimizer.step()
                args.scheduler.step()
                args.optimizer.zero_grad()
    n = max(agg["n"], 1)
    out = {
        "rows": int(agg["n"]),
        "n_known": int(metric["n_known"]),
        "n_first": int(metric["n_first"]),
        "n_attach": int(metric["n_attach"]),
        "ce": agg["ce"] / n,
        "pen": agg["pen"] / n,
        "new_rate": agg["new"] / n,
        "known_acc": metric["known_acc"] / max(metric["n_known"], 1),
        "first_acc": metric["first_acc"] / max(metric["n_first"], 1),
        "reuse_acc": metric["reuse_acc"] / max(metric["n_attach"], 1),
    }
    if not train and len(scores):
        out["knownness_auc"] = knownness_auc(ep, scores, split)
    return out


def knownness_auc(ep, scores, split):
    """Known explainability AUROC/AUPR on the legal proxy-val split."""
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {}
    y = []
    s = []
    for ri in sorted(scores):
        role = int(ep["gt_role"][ri])
        cat = int(ep["gt_category_id"][ri])
        if role != 1:
            continue
        y.append(1 if cat in split["known"] or cat in split["novel_train"]
                 else 0)
        s.append(scores[ri][0])
    if len(set(y)) < 2:
        return {}
    return {
        "auc_known_vs_proxy": float(roc_auc_score(y, s)),
        "aupr_known_vs_proxy": float(average_precision_score(y, s)),
        "n": len(y),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="outputs/iclr27_phase7b/training/tose_main")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk-videos", type=int, default=10)
    ap.add_argument("--visible-known-per-chunk", type=int, default=6)
    ap.add_argument("--w-known", type=float, default=2.0)
    ap.add_argument("--no-cross-track", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-slot", type=float, default=0.5)
    ap.add_argument("--w-novel", type=float, default=20.0)
    ap.add_argument("--w-unlabeled", type=float, default=0.05)
    ap.add_argument("--slot-temp", type=float, default=0.10)
    ap.add_argument("--known-tau", type=float, default=-1.0)
    ap.add_argument("--grad-every", type=int, default=2000)
    ap.add_argument("--fp-keep", type=float, default=0.25)
    ap.add_argument("--track-cap-novel", type=int, default=80)
    ap.add_argument("--track-cap-other", type=int, default=80)
    ap.add_argument("--use-dist", action="store_true", default=True)
    ap.add_argument("--use-traj", action="store_true", default=True)
    ap.add_argument("--no-dist", action="store_true")
    ap.add_argument("--frame-level", action="store_true")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--classifier-conf", action="store_true")
    ap.add_argument("--head-type", choices=["mlp", "linear"], default="mlp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.no_dist:
        args.use_dist = False
    if args.frame_level:
        args.use_traj = False
    if args.classifier_conf:
        args.use_dist = False
        args.use_traj = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tr, va, split = load_assets()
    stats = load_stats()
    if args.smoke:
        for name in ("train", "val"):
            ep = tr if name == "train" else va
            vids = np.unique(ep["video_ids"])[:2]
            m = np.isin(ep["video_ids"], vids)
            for k in ep:
                ep[k] = ep[k][m]
        args.epochs = 1
        args.chunk_videos = 2
        args.visible_known_per_chunk = 2
    model, anchors_np, known_ids = load_tse(dev)
    z_tr = project(dev, model, tr["feats"])
    z_va = project(dev, model, va["feats"])
    h_tr = replay_track_ema(tr, z_tr)
    h_va = replay_track_ema(va, z_va)
    as_tr, ll_tr = precompute(h_tr, anchors_np, stats)
    as_va, ll_va = precompute(h_va, anchors_np, stats)
    print(f"train z {z_tr.shape} h {h_tr.shape}; "
          f"val z {z_va.shape} h {h_va.shape}", flush=True)

    base_names = base_feature_names(args.use_dist, args.use_traj)
    print("base features:", base_names, flush=True)
    if args.head_type == "linear":
        policy = TOSELinearHead(
            base_dim=len(base_names), use_dist=args.use_dist,
            use_traj=args.use_traj).to(dev)
    else:
        policy = TOSEHead(base_dim=len(base_names)).to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    args.optimizer = opt
    n_tr_vids = len(np.unique(tr["video_ids"]))
    steps_per_epoch = max(1, int(np.ceil(n_tr_vids / args.chunk_videos)))
    args.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    best_val = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trm = run_epoch(tr, z_tr, h_tr, as_tr, ll_tr, policy, anchors_np,
                        known_ids, split, dev, args, rng, stats, train=True)
        vam = run_epoch(va, z_va, h_va, as_va, ll_va, policy, anchors_np,
                        known_ids, split, dev, args, rng, stats, train=False)
        score = vam["reuse_acc"] * vam["first_acc"] - 0.3 * vam["new_rate"]
        if best_val is None or score > best_val["score"]:
            best_val = dict(epoch=ep, score=score, **vam)
            torch.save({
                "policy": {k: v.detach().cpu()
                           for k, v in policy.state_dict().items()},
                "args": vars(args),
                "epoch": ep,
            }, out / "best.pth")
        print(f"epoch {ep}/{args.epochs} {time.time() - t0:.1f}s "
              f"train ce={trm['ce']:.4f} pen={trm['pen']:.4f} "
              f"new={trm['new_rate']:.4f} k={trm['known_acc']:.3f} "
              f"first={trm['first_acc']:.3f}({trm['n_first']}) "
              f"reuse={trm['reuse_acc']:.3f}({trm['n_attach']}) | "
              f"val k={vam['known_acc']:.3f} first={vam['first_acc']:.3f} "
              f"reuse={vam['reuse_acc']:.3f} new={vam['new_rate']:.4f} "
              f"auc={vam.get('knownness_auc', {}).get('auc_known_vs_proxy', -1):.3f}",
              flush=True)
        if (ep % args.save_every == 0) or (ep == args.epochs):
            torch.save({
                "policy": {k: v.detach().cpu()
                           for k, v in policy.state_dict().items()},
                "args": vars(args),
                "epoch": ep,
                "train_metrics": trm,
                "val_metrics": vam,
            }, out / f"checkpoint_{ep:03d}.pth")
    (out / "best.json").write_text(json.dumps(best_val, indent=2))
    print("BEST", best_val)


if __name__ == "__main__":
    main()
