"""Train the RACC attach-or-create head on proxy-novel trajectory episodes.

Protocol: frozen TSE embedding + fixed reliability-aware memory; the only
learned parameters are the known/new/pair scoring heads. Training uses
supported-known labels only: a class-level split hides 15 classes as
proxy-novel (novel_train), a disjoint 15 classes are held out for validation
(novel_val), and every training chunk additionally hides a few known classes
episodically (episodic pseudo-novel). No true novel GT is used.
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
from src.iclr27_phase6c.model.tse import TSE, KnownAnchors
from src.iclr27_phase7a.model.reliability_memory import (
    MemoryState,
    RACCHead,
    TrackStats,
    online_step,
)


def load_assets():
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/train_episodes.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/val_episodes.npz").items()}
    split = json.loads(
        (ROOT / "outputs/iclr27_phase7a/assets/class_split.json").read_text())
    return tr, va, split


def load_tse(device, ckpt="outputs/iclr27_phase6c/training/tse_main/checkpoint.pth"):
    state = torch.load(ROOT / ckpt, map_location=device, weights_only=False)
    model = TSE().to(device)
    model.load_pca(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    model.load_state_dict(state["model"])
    model.eval()
    anchors = KnownAnchors(state["known_ids"]).to(device)
    anchors.load_state_dict(state["anchors"])
    known_ids = np.asarray(state["known_ids"], dtype=np.int64)
    with torch.no_grad():
        an = anchors.normalized().cpu().numpy().astype(np.float32)
    return model, an, known_ids


def project(device, model, feats, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            x = torch.from_numpy(feats[i:i + batch].astype(np.float32)).to(device)
            out.append(model.project(x).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def video_ranges(ep):
    """Returns dict video_id -> (start, end) in the packed chronological rows."""
    vids = ep["video_ids"]
    starts = {}
    out = {}
    for i, v in enumerate(vids):
        if v not in starts:
            starts[v] = i
    for v, s in starts.items():
        e = len(vids)
        for j in range(s + 1, len(vids)):
            if vids[j] != v:
                e = j
                break
        out[int(v)] = (int(s), int(e))
    return out


def chunk_losses(chunk_rows, z_all, ep, mem, policy, anchors, known_ids,
                 visible_cls, novel_cls, track_stats, device, args, rng,
                 train=True, model_slot_of_class=None, class_seen=None,
                 loss_buf=None):
    """Replay one chunk and accumulate losses."""
    if model_slot_of_class is None:
        model_slot_of_class = {}
    if class_seen is None:
        class_seen = set()
    visible_mask = np.isin(known_ids, np.asarray(sorted(visible_cls),
                                                 dtype=np.int64))
    if not visible_mask.any():
        raise RuntimeError("empty visible known set")
    known_idx = {int(c): i for i, c in enumerate(known_ids)}
    n_sup = n_known = n_first = n_reuse = n_attach = 0
    acc_first = acc_reuse = acc_known = 0.0
    for ri in chunk_rows:
        if int(ep["gt_role"][ri]) == 0 and rng.random() > args.fp_keep:
            continue
        key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackStats()
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
                    mem.birth_key[bi] == key
                    if 0 <= bi < len(mem.birth_key) else False)
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
                # class seen before but model has not yet birthed a slot:
                # keep the NEW target until a slot exists.
                target = 2
                n_first += 1
        elif is_known_row and cat in visible_cls:
            target = 0
            n_known += 1
        z = z_all[ri]
        res = online_step(
            policy, z, mem, anchors, visible_mask, known_ids, ts,
            float(ep["score"][ri]), int(ep["prior_hits"][ri]),
            ep["bbox_xyxy"][ri], int(ep["frame_ids"][ri]), key,
            slot_class=slot_class, target_slot=slot_target,
            known_tau=args.known_tau if args.known_tau >= 0 else None,
            use_rel=not args.no_rel and not args.sem_only,
            use_maturity=not args.no_maturity,
            sem_only=args.sem_only)
        if res["decision"] == 2 and slot_class >= 0 and \
                res["slot_idx"] is not None:
            model_slot_of_class.setdefault(slot_class, res["slot_idx"])
        if res.get("frozen_known"):
            continue
        logits = res["logits"]
        if target is not None:
            loss = F.cross_entropy(logits.unsqueeze(0),
                                   torch.tensor([target], device=logits.device))
            if target == 0:
                loss = args.w_known * loss
                acc_known += (res["decision"] == 0)
                n_sup += 1
            else:
                loss = args.w_novel * loss
            if target == 1:
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
                acc_first += (res["decision"] == 2)
                n_sup += 1
            loss_buf["ce"] += float(loss.detach())
            loss_buf["total"] += loss
        else:
            if not is_known_row:
                # unlabeled FP row: weak penalty against NEW
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


def run_epoch(ep, z_all, policy, anchors, known_ids, split, device, args,
              rng, train=True):
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    if train:
        rng.shuffle(vids)
    ranges = video_ranges(ep)
    chunks = [vids[i:i + args.chunk_videos]
              for i in range(0, len(vids), args.chunk_videos)]
    agg = defaultdict(float)
    metric = defaultdict(float)
    policy.train(train)
    for ci, cvids in enumerate(chunks):
        mem = MemoryState(dim=128)
        track_stats = {}
        model_slot_of_class = {}
        class_seen = set()
        if train:
            kvis = min(args.visible_known_per_chunk, len(split["known"]))
            vis_known = set(rng.sample(sorted(split["known"]), kvis))
            visible = vis_known | set(split["novel_val"])
            novel = set(split["known"]) - vis_known
            novel |= set(split["novel_train"])
        else:
            visible = set(split["known"])
            novel = set(split["novel_val"])
        if train and args.track_cap_other > 0:
            # Balanced track-level episode sampling: keep all novel-class
            # tracks (capped) plus equal numbers of known and FP tracks.
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
        step = 0
        for start in range(0, len(rows), args.grad_every):
            sub = rows[start:start + args.grad_every]
            loss_buf = defaultdict(float)
            out = chunk_losses(
                sub, z_all, ep, mem, policy, anchors, known_ids,
                visible, novel, track_stats, device, args, rng, train=train,
                model_slot_of_class=model_slot_of_class,
                class_seen=class_seen, loss_buf=loss_buf)
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
            step += 1
    n = max(agg["n"], 1)
    return {
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase7a/training/racc_main")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk-videos", type=int, default=10)
    ap.add_argument("--episodic-hide", type=int, default=8)
    ap.add_argument("--visible-known-per-chunk", type=int, default=6)
    ap.add_argument("--w-known", type=float, default=2.0)
    ap.add_argument("--no-rel", action="store_true")
    ap.add_argument("--no-maturity", action="store_true")
    ap.add_argument("--no-cross-track", action="store_true")
    ap.add_argument("--sem-only", action="store_true")
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tr, va, split = load_assets()
    if args.smoke:
        for name in ("train", "val"):
            ep = tr if name == "train" else va
            vids = np.unique(ep["video_ids"])[:2]
            m = np.isin(ep["video_ids"], vids)
            for k in ep:
                ep[k] = ep[k][m]
        args.epochs = 1
        args.chunk_videos = 2
        args.episodic_hide = 1
    model, anchors_np, known_ids = load_tse(dev)
    z_tr = project(dev, model, tr["feats"])
    z_va = project(dev, model, va["feats"])
    print(f"train z {z_tr.shape}, val z {z_va.shape}", flush=True)

    policy = RACCHead().to(dev)
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
        trm = run_epoch(tr, z_tr, policy, anchors_np, known_ids, split, dev,
                        args, rng, train=True)
        vam = run_epoch(va, z_va, policy, anchors_np, known_ids, split, dev,
                        args, rng, train=False)
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
              f"reuse={vam['reuse_acc']:.3f} new={vam['new_rate']:.4f}",
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
