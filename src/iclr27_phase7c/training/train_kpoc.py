"""Train Phase 7C Known-Preserving Open-World Calibration (KPOC).

Same frozen TOSE trajectory explainability + simple EMA novel memory as
Phase 7B, but trained on legal class-level hard proxy-OOD episodes with an
explicit known-preserving calibration objective:

  L = L_3way(known/existing/new) + L_kp + L_open + L_attach

where for visible-known rows
  L_kp = max(0, m_kp - (known_logit - max(attach, new)))
and for hidden proxy-OOD rows
  L_open = max(0, m_open + known_logit - max(attach, new)).

The known logit is calibrated at inference by a prior offset selected on the
legal meta-val Pareto frontier (Known vs First*Reuse), then frozen for
Q1 DEV / heldout. No true novel GT, no Q1 information, no future.
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
    TrackState,
    base_feature_names,
    tose_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def load_assets(mode="hard"):
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/train_{mode}.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/metaval_{mode}.npz").items()}
    split = json.loads(
        (ROOT / f"outputs/iclr27_phase7c/assets/class_split_{mode}.json")
        .read_text())
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
    asims = h_all @ anchors.T
    diff = h_all[:, None, :] - stats["mu"][None, :, :]
    mahal2 = np.sum(diff * diff / stats["sigma2"][None, :, :], axis=2)
    loglik = -0.5 * (mahal2 + stats["logdet"][None, :])
    return asims.astype(np.float32), loglik.astype(np.float32)


def chunk_losses(chunk_rows, z_all, h_all, asims_all, loglik_all, ep, mem,
                 policy, anchors, known_ids, visible_cls, novel_cls,
                 track_stats, device, args, rng, stats,
                 train=True, model_slot_of_class=None, class_seen=None,
                 loss_buf=None, logits_store=None):
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
        row_split = int(ep["row_split"][ri])
        if row_split < 0 and rng.random() > args.fp_keep:
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
        is_proxy = (row_split == 1) or (
            row_split == 0 and is_known_row and cat in novel_cls)
        if is_known_row and is_proxy:
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
        elif is_known_row and row_split == 0 and cat in visible_cls:
            target = 0
            n_known += 1
        z = z_all[ri]
        res = tose_step(
            policy, z, mem, anchors, visible_mask, known_ids, stats, ts,
            int(ep["frame_ids"][ri]), key, slot_class=slot_class,
            target_slot=slot_target,
            asims_row=asims_all[ri], loglik_row=loglik_all[ri])
        if logits_store is not None:
            logits_store[ri] = (
                float(res["logits"][0].item()),
                float(res["logits"][1].item()),
                float(res["logits"][2].item()),
                res["decision"], row_split, cat, key)
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
                if args.m_kp > 0:
                    open_best = torch.maximum(logits[1], logits[2])
                    kp = torch.clamp(
                        args.m_kp - (logits[0] - open_best), min=0.0)
                    loss = loss + args.w_kp * kp
                    loss_buf["kp"] += float(kp.detach())
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
                if args.m_open > 0:
                    open_best = torch.maximum(logits[1], logits[2])
                    ol = torch.clamp(
                        args.m_open + logits[0] - open_best, min=0.0)
                    loss = loss + args.w_open * ol
                    loss_buf["open"] += float(ol.detach())
            else:
                loss = args.w_novel * loss
                acc_first += (res["decision"] == 2)
                n_sup += 1
                if args.m_open > 0:
                    open_best = torch.maximum(logits[1], logits[2])
                    ol = torch.clamp(
                        args.m_open + logits[0] - open_best, min=0.0)
                    loss = loss + args.w_open * ol
                    loss_buf["open"] += float(ol.detach())
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
    logits_store = {}
    for ci, cvids in enumerate(chunks):
        mem = EMAMemory(dim=128)
        track_stats = {}
        model_slot_of_class = {}
        class_seen = set()
        if train:
            kvis = min(args.visible_known_per_chunk, len(split["train_visible"]))
            vis_known = set(rng.sample(sorted(split["train_visible"]), kvis))
            visible = vis_known
            novel = (set(split["train_visible"]) - vis_known)
            novel |= set(split["hidden_train"])
        else:
            visible = set(split["train_visible"])
            novel = set(split["hidden_val"])
        if train and args.track_cap_other > 0:
            by_track = defaultdict(list)
            for v in cvids:
                s, e = ranges[v]
                for ri in range(s, e):
                    by_track[(int(ep["video_ids"][ri]),
                              int(ep["track_ids"][ri]))].append(ri)
            novel_tracks, known_tracks, fp_tracks = [], [], []
            for key, idxs in by_track.items():
                roles = ep["row_split"][idxs]
                has_novel = any(int(rl) == 1 for rl in roles)
                has_known = any(int(rl) == 0 for rl in roles)
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
                for ri in range(s, e):
                    if int(ep["row_split"][ri]) < 0 and not train and (
                            (ri * 2654435761 + args.seed) % 1000 / 10.0
                            >= args.val_fp_keep * 100):
                        continue
                    rows.append(ri)
        for start in range(0, len(rows), args.grad_every):
            sub = rows[start:start + args.grad_every]
            loss_buf = defaultdict(float)
            out = chunk_losses(
                sub, z_all, h_all, asims_all, loglik_all, ep, mem, policy,
                anchors, known_ids, visible, novel, track_stats, device,
                args, rng, stats, train=train,
                model_slot_of_class=model_slot_of_class,
                class_seen=class_seen, loss_buf=loss_buf,
                logits_store=logits_store if not train else None)
            agg["ce"] += loss_buf["ce"]
            agg["pen"] += loss_buf["pen"]
            agg["kp"] += loss_buf["kp"]
            agg["open"] += loss_buf["open"]
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
        "kp": agg["kp"] / n,
        "open": agg["open"] / n,
        "new_rate": agg["new"] / n,
        "known_acc": metric["known_acc"] / max(metric["n_known"], 1),
        "first_acc": metric["first_acc"] / max(metric["n_first"], 1),
        "reuse_acc": metric["reuse_acc"] / max(metric["n_attach"], 1),
    }
    if not train and len(logits_store):
        out["frontier"] = frontier_from_logits(logits_store, ep, split)
    return out


def frontier_from_logits(logits_store, ep, split, offsets=None):
    """Pareto frontier on the legal meta-val by sweeping the known prior
    offset over already-recorded per-row logits (no re-replay)."""
    if offsets is None:
        offsets = np.arange(-1.0, 3.0, 0.25)
    vis = set(split["train_visible"]) | set(split["hidden_train"])
    rows = []
    for ri, (kl, al, nl, dec, row_split, cat, key) in logits_store.items():
        rows.append({
            "ri": ri, "logits": (kl, al, nl),
            "row_split": row_split, "cat": cat, "key": key,
        })
    rows.sort(key=lambda x: x["ri"])
    points = []
    for off in offsets:
        mem_slot = {}  # class -> first track key
        n_known = n_first = n_attach = 0
        acc_known = acc_first = acc_attach = 0
        for x in rows:
            kl, al, nl = x["logits"]
            vals = [kl + off, al, nl]
            act = int(np.argmax(vals))
            rs = x["row_split"]
            if rs == 0:
                n_known += 1
                acc_known += (act == 0)
            elif rs == 1:
                cat = x["cat"]
                key = tuple(x["key"])
                if cat not in mem_slot:
                    mem_slot[cat] = key
                    n_first += 1
                    acc_first += (act == 2)
                elif mem_slot[cat] != key:
                    # cross-track occurrence of the same proxy category
                    n_attach += 1
                    acc_attach += (act == 1)
        points.append({
            "offset": float(off),
            "known_acc": acc_known / max(n_known, 1),
            "first_acc": acc_first / max(n_first, 1),
            "reuse_acc": acc_attach / max(n_attach, 1),
            "novel_score": (acc_first / max(n_first, 1))
            * (acc_attach / max(n_attach, 1)),
            "n_known": n_known, "n_first": n_first, "n_attach": n_attach,
        })
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase7c/training/kpoc_main")
    ap.add_argument("--mode", choices=["hard", "random"], default="hard")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk-videos", type=int, default=10)
    ap.add_argument("--visible-known-per-chunk", type=int, default=6)
    ap.add_argument("--w-known", type=float, default=2.0)
    ap.add_argument("--w-novel", type=float, default=20.0)
    ap.add_argument("--w-slot", type=float, default=0.5)
    ap.add_argument("--w-unlabeled", type=float, default=0.05)
    ap.add_argument("--w-kp", type=float, default=10.0)
    ap.add_argument("--w-open", type=float, default=10.0)
    ap.add_argument("--m-kp", type=float, default=0.5)
    ap.add_argument("--m-open", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--slot-temp", type=float, default=0.10)
    ap.add_argument("--grad-every", type=int, default=2000)
    ap.add_argument("--fp-keep", type=float, default=0.25)
    ap.add_argument("--track-cap-novel", type=int, default=80)
    ap.add_argument("--track-cap-other", type=int, default=80)
    ap.add_argument("--val-fp-keep", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2717)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--no-cross-track", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tr, va, split = load_assets(args.mode)
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
    print(f"train z {z_tr.shape}; meta-val z {z_va.shape}", flush=True)

    base_names = base_feature_names(True, True)
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
        pts = vam.get("frontier", [])
        best_pt = max(pts, key=lambda p: p["known_acc"] * p["novel_score"]) \
            if pts else {}
        score = best_pt.get("novel_score", 0.0) * best_pt.get(
            "known_acc", 0.0) if best_pt else 0.0
        if best_val is None or score > best_val["score"]:
            best_val = dict(epoch=ep, score=score, frontier_point=best_pt,
                            **vam)
            torch.save({
                "policy": {k: v.detach().cpu()
                           for k, v in policy.state_dict().items()},
                "args": vars(args),
                "epoch": ep,
            }, out / "best.pth")
        print(f"epoch {ep}/{args.epochs} {time.time() - t0:.1f}s "
              f"train ce={trm['ce']:.3f} kp={trm['kp']:.3f} "
              f"open={trm['open']:.3f} new={trm['new_rate']:.3f} "
              f"k={trm['known_acc']:.3f} first={trm['first_acc']:.3f} "
              f"reuse={trm['reuse_acc']:.3f} | "
              f"val k={vam['known_acc']:.3f} first={vam['first_acc']:.3f} "
              f"reuse={vam['reuse_acc']:.3f} "
              f"frontier_best={json.dumps({k: round(v, 3) for k, v in best_pt.items() if isinstance(v, float)})}",
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
    print("BEST", json.dumps(best_val, indent=2, default=float))


if __name__ == "__main__":
    main()
