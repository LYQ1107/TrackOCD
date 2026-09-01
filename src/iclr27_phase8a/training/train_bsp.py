"""Episodic training of Architecture A: causal trajectory adapter + Bayesian
semantic state process (assign-existing vs spawn-new).

The semantic state set is re-initialized per chunk from the chunk's visible
supported-known classes (legal TRAIN centroids).  The other supported-known
classes plus the hidden_train classes are simulated proxy-novel: their first
physical track must spawn a NEW state and later tracks (including different
physical tracks) must assign to that online-born state.  State statistics are
updated with detached values (teacher-forced for proxy rows), while gradients
flow into the causal trajectory adapter through the Gaussian predictive
logits.  No true novel GT and no future information are used.
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
from src.iclr27_phase8a.model.adapter import (
    CausalTrajectoryAdapter,
    TorchSemanticStateSet,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def load_assets(mode):
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/train_{mode}.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/metaval_{mode}.npz").items()}
    split = json.loads((ROOT /
        f"outputs/iclr27_phase7c/assets/class_split_{mode}.json").read_text())
    return tr, va, split


def video_ranges(ep):
    vids = ep["video_ids"]
    starts = {}
    for i, v in enumerate(vids):
        if v not in starts:
            starts[v] = i
    out = {}
    for v, s in starts.items():
        e = len(vids)
        for j in range(s + 1, len(vids)):
            if vids[j] != v:
                e = j
                break
        out[int(v)] = (int(s), int(e))
    return out


def compute_centroids(adapter, ep, z_all, device):
    """Adapter-space centroids for all supported-known classes (legal rows)."""
    st = dict(np.load(ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))
    known_ids = [int(x) for x in st["known_ids"]]
    cls_idx = {c: i for i, c in enumerate(known_ids)}
    track_state = {}
    sums = torch.zeros(len(known_ids), adapter.dim, device=device)
    cnt = torch.zeros(len(known_ids), device=device)
    with torch.no_grad():
        for i in range(len(z_all)):
            if int(ep["gt_role"][i]) != 1:
                continue
            c = int(ep["gt_category_id"][i])
            if c not in cls_idx:
                continue
            key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            z = torch.from_numpy(z_all[i]).to(device).unsqueeze(0)
            h, state = adapter(z, prev)
            track_state[key] = state.detach()
            sums[cls_idx[c]] += h[0]
            cnt[cls_idx[c]] += 1.0
    mu = F.normalize(sums / torch.clamp(cnt, min=1.0)[:, None], dim=-1)
    return mu, cnt, known_ids


def chunk_rows(ep, ranges, vids, args, rng, train):
    chunks = [vids[i:i + args.chunk_videos]
              for i in range(0, len(vids), args.chunk_videos)]
    out = []
    for cvids in chunks:
        by_track = defaultdict(list)
        for v in cvids:
            s, e = ranges[v]
            for ri in range(s, e):
                by_track[(int(ep["video_ids"][ri]),
                          int(ep["track_ids"][ri]))].append(ri)
        if train and args.track_cap_other > 0:
            novel_tracks, known_tracks, fp_tracks = [], [], []
            for key, idxs in by_track.items():
                rs = ep["row_split"][idxs]
                if any(int(r) == 1 for r in rs):
                    novel_tracks.append(key)
                elif any(int(r) == 0 for r in rs):
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
                    rs = int(ep["row_split"][ri])
                    if rs < 0 and not train and (
                            (ri * 2654435761 + args.seed) % 1000 / 10.0
                            >= args.fp_keep * 100):
                        continue
                    if train and rs < 0 and rng.random() > args.fp_keep:
                        continue
                    rows.append(ri)
        out.append(rows)
    return out


def replay_rows(adapter, states, rows, ep, z_all, device, train, args,
                known_slot_of_class, track_state, track_count,
                class_slot, class_first, opt=None, accum=None, rng=None,
                metrics=None):
    losses = []
    for ri in rows:
        rs = int(ep["row_split"][ri])
        cat = int(ep["gt_category_id"][ri])
        key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
        prev = track_state.get(key)
        if prev is None:
            prev = adapter.new_state()
        z = torch.from_numpy(z_all[ri]).to(device).unsqueeze(0)
        h, state_new = adapter(z, prev)
        h = h[0]
        if args.adapt_reg > 0:
            reg = args.adapt_reg * (
                adapter.last_raw[0] - z[0]).pow(2).sum()
            losses.append(reg)
        w = 1.0 if args.frame_level else float(track_count.get(key, 0) + 1)
        track_count[key] = int(w)
        track_state[key] = state_new.detach()

        is_known = rs == 0
        is_novel_target = rs == 1 or (
            rs == 0 and int(cat) not in known_slot_of_class)
        target = None
        target_slot = None
        force_spawn = False
        if is_known and int(cat) in known_slot_of_class:
            target = known_slot_of_class[cat]
            target_slot = target
        elif is_novel_target and not args.known_only:
            if cat not in class_first:
                class_first.add(cat)
                target = states.n  # new-state option
                force_spawn = True
            elif cat in class_slot:
                target = class_slot[cat]
                target_slot = target
            else:
                target = states.n
                force_spawn = True

        rho = adapter.rho
        logits = states.logits(h, w, rho)
        pred = int(torch.argmax(logits))
        loss = None
        max_existing = (logits[:states.n].max()
                        if states.n > 0 else rho.detach())
        if target is not None:
            if not (0 <= target < logits.numel()):
                raise RuntimeError(
                    f"bad target {target} logits {logits.numel()} "
                    f"rs {rs} cat {cat} n {states.n} "
                    f"n_classes {len(logits)}")
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target], device=device))
            if is_known:
                loss = args.w_known * loss
                margin = torch.clamp(
                    rho - logits[target] + args.m_known, min=0.0)
                loss = loss + args.w_known_margin * margin
            else:
                loss = args.w_novel * loss
                if force_spawn:
                    margin = torch.clamp(
                        max_existing - rho + args.m_birth, min=0.0)
                    loss = loss + args.w_birth_margin * margin
                else:
                    margin = torch.clamp(
                        rho - logits[target] + args.m_attach, min=0.0)
                    loss = loss + args.w_attach_margin * margin
            if metrics is not None:
                if is_known:
                    metrics["known"] += (pred == target)
                    metrics["n_known"] += 1
                elif force_spawn:
                    metrics["first"] += (pred == states.n)
                    metrics["n_first"] += 1
                else:
                    metrics["attach"] += (pred == target)
                    metrics["n_attach"] += 1
        elif rs < 0:
            # unlabeled/FP row: weak penalty against spawning a state
            if pred == states.n:
                if states.n > 0:
                    pen = torch.clamp(
                        logits[states.n] - logits[:states.n].max() +
                        args.fp_margin, min=0.0)
                    loss = args.w_fp * pen
        if loss is not None:
            losses.append(loss)

        # Online state bookkeeping (detached inside assign/spawn)
        if target_slot is not None:
            states.assign(target_slot, h, w)
        elif force_spawn:
            slot = states.spawn(h, w)
            if slot is not None:
                class_slot[cat] = slot
        elif rs < 0:
            states.decide(h, w, rho)

        if opt is not None and len(losses) >= accum:
            total = torch.stack(losses).sum()
            total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad()
            losses = []
    if opt is not None and losses:
        total = torch.stack(losses).sum()
        total.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad()


def run_epoch(adapter, ep, z_all, split, device, args, rng, train=True):
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_all, device)
    ranges = video_ranges(ep)
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    if train:
        rng.shuffle(vids)
    chunks = chunk_rows(ep, ranges, vids, args, rng, train)
    states = TorchSemanticStateSet(
        dim=args.dim, max_slots=args.max_slots, sigma2=args.sigma2,
        score_mode=args.score_mode, no_evidence=args.no_evidence,
        cosine_temp=args.cosine_temp, no_update=args.no_state_update).to(
            device)
    track_state = {}
    track_count = {}
    metrics = defaultdict(float)
    opt = None
    accum = None
    if train:
        opt = args.optimizer
        accum = args.grad_accum
    for chunk in chunks:
        states.reset()
        class_slot = {}
        class_first = set()
        known_slot_of_class = {}
        vis = set(split["train_visible"])
        if train:
            kvis = min(args.visible_known_per_chunk,
                       len(split["train_visible"]))
            vis = set(rng.sample(sorted(split["train_visible"]), kvis))
        mu_k = mu[[known_ids.index(c) for c in sorted(vis)]]
        cnt_k = cnt[[known_ids.index(c) for c in sorted(vis)]]
        states.init_known(mu_k.detach(), cnt_k.detach())
        for j, c in enumerate(sorted(vis)):
            known_slot_of_class[c] = j
        replay_rows(
            adapter, states, chunk, ep, z_all, device, train, args,
            known_slot_of_class, track_state, track_count, class_slot,
            class_first, opt=opt, accum=accum, rng=rng, metrics=metrics)
    out = {
        "n_known": int(metrics["n_known"]),
        "n_first": int(metrics["n_first"]),
        "n_attach": int(metrics["n_attach"]),
        "known_acc": metrics["known"] / max(metrics["n_known"], 1),
        "first_acc": metrics["first"] / max(metrics["n_first"], 1),
        "attach_acc": metrics["attach"] / max(metrics["n_attach"], 1),
    }
    return out


def eval_metaval(adapter, ep, z_all, split, device, args):
    """Self-driven strict-causal meta-val replay (no teacher forcing)."""
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_all, device)
    states = TorchSemanticStateSet(
        dim=args.dim, max_slots=args.max_slots, sigma2=args.sigma2,
        score_mode=args.score_mode, no_evidence=args.no_evidence,
        cosine_temp=args.cosine_temp, no_update=args.no_state_update).to(
            device)
    vis = sorted(set(split["train_visible"]))
    states.init_known(
        mu[[known_ids.index(c) for c in vis]].detach(),
        cnt[[known_ids.index(c) for c in vis]].detach())
    known_slot = {c: j for j, c in enumerate(vis)}
    ranges = video_ranges(ep)
    vids = [int(v) for v in np.unique(ep["video_ids"])]
    chunks = chunk_rows(ep, ranges, vids, args, rng=None, train=False)
    track_state = {}
    track_count = {}
    class_first = set()
    class_slot = {}
    slot_class = {}
    slot_birth = {}
    n = {"known": 0, "first": 0, "same": 0, "cross": 0}
    ok = {"known": 0, "first": 0, "same": 0, "cross": 0}
    with torch.no_grad():
        for chunk in chunks:
            for ri in chunk:
                rs = int(ep["row_split"][ri])
                if rs < 0:
                    continue
                cat = int(ep["gt_category_id"][ri])
                key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
                prev = track_state.get(key)
                if prev is None:
                    prev = adapter.new_state()
                z = torch.from_numpy(z_all[ri]).to(device).unsqueeze(0)
                h, state_new = adapter(z, prev)
                h = h[0]
                w = 1.0 if args.frame_level else float(
                    track_count.get(key, 0) + 1)
                track_count[key] = int(w)
                track_state[key] = state_new.detach()
                logits = states.logits(h, w, adapter.rho)
                pred = int(torch.argmax(logits))
                if pred == states.n:
                    slot = states.spawn(h, w)
                    if rs == 1 and slot is not None:
                        slot_class.setdefault(slot, cat)
                        slot_birth.setdefault(slot, key)
                        class_slot.setdefault(cat, slot)
                    if rs == 0:
                        n["known"] += 1
                        ok["known"] += False
                    elif rs == 1:
                        if cat not in class_first:
                            class_first.add(cat)
                            n["first"] += 1
                            ok["first"] += True
                        else:
                            ts = class_slot.get(cat)
                            if ts is not None and key == slot_birth.get(ts):
                                n["same"] += 1
                                ok["same"] += False
                            else:
                                n["cross"] += 1
                                ok["cross"] += False
                else:
                    slot = pred
                    prov = int(states.provenance[slot])
                    states.assign(slot, h, w)
                    if rs == 0:
                        n["known"] += 1
                        ok["known"] += (prov == 0 and slot == known_slot[cat])
                    elif rs == 1:
                        if cat not in class_first:
                            class_first.add(cat)
                            n["first"] += 1
                            ok["first"] += False
                        else:
                            ts = class_slot.get(cat)
                            correct = (
                                prov == 1 and ts is not None
                                and slot == ts and slot_class.get(slot) == cat)
                            if ts is not None and key == slot_birth.get(ts):
                                n["same"] += 1
                                ok["same"] += correct
                            else:
                                n["cross"] += 1
                                ok["cross"] += correct
    res = {
        "n_known": n["known"],
        "n_first": n["first"],
        "n_same": n["same"],
        "n_cross": n["cross"],
        "known_acc": ok["known"] / max(n["known"], 1),
        "first_acc": ok["first"] / max(n["first"], 1),
        "same_acc": ok["same"] / max(n["same"], 1),
        "cross_acc": ok["cross"] / max(n["cross"], 1),
        "n_states": states.n,
    }
    res["joint"] = res["known_acc"] * res["first_acc"] * res["cross_acc"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="outputs/iclr27_phase8a/training/bsp_main")
    ap.add_argument("--mode", choices=["hard", "random"], default="hard")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk-videos", type=int, default=10)
    ap.add_argument("--visible-known-per-chunk", type=int, default=6)
    ap.add_argument("--w-known", type=float, default=1.0)
    ap.add_argument("--w-novel", type=float, default=2.0)
    ap.add_argument("--w-fp", type=float, default=0.05)
    ap.add_argument("--fp-margin", type=float, default=1.0)
    ap.add_argument("--w-known-margin", type=float, default=1.0)
    ap.add_argument("--m-known", type=float, default=1.0)
    ap.add_argument("--w-birth-margin", type=float, default=2.0)
    ap.add_argument("--m-birth", type=float, default=1.0)
    ap.add_argument("--w-attach-margin", type=float, default=2.0)
    ap.add_argument("--m-attach", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--grad-accum", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max-slots", type=int, default=512)
    ap.add_argument("--sigma2", type=float, default=0.05)
    ap.add_argument("--rho-init", type=float, default=40.0)
    ap.add_argument("--score-mode", choices=["gaussian", "cosine"],
                    default="gaussian")
    ap.add_argument("--no-evidence", action="store_true")
    ap.add_argument("--cosine-temp", type=float, default=20.0)
    ap.add_argument("--frame-level", action="store_true")
    ap.add_argument("--known-only", action="store_true")
    ap.add_argument("--no-state-update", action="store_true")
    ap.add_argument("--adapt-reg", type=float, default=0.0)
    ap.add_argument("--fp-keep", type=float, default=0.25)
    ap.add_argument("--track-cap-novel", type=int, default=80)
    ap.add_argument("--track-cap-other", type=int, default=80)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=2717)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tr, va, split = load_assets(args.mode)
    if args.smoke:
        for name, ep in (("train", tr), ("val", va)):
            vids = np.unique(ep["video_ids"])[:2]
            m = np.isin(ep["video_ids"], vids)
            for k in ep:
                ep[k] = ep[k][m]
        args.epochs = 1
        args.chunk_videos = 2
        args.visible_known_per_chunk = 2
        args.track_cap_other = 0

    model, _, _ = load_tse(dev)
    z_tr = project(dev, model, tr["feats"].astype(np.float32))
    z_va = project(dev, model, va["feats"].astype(np.float32))
    adapter = CausalTrajectoryAdapter(
        dim=args.dim, rho_init=args.rho_init, sigma2=args.sigma2,
        frame_level=args.frame_level).to(dev)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    args.optimizer = opt
    n_vids = len(np.unique(tr["video_ids"]))
    steps = max(1, int(np.ceil(n_vids / args.chunk_videos)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    args.scheduler = sched
    best = None
    for ep_i in range(1, args.epochs + 1):
        t0 = time.time()
        trm = run_epoch(adapter, tr, z_tr, split, dev, args, rng, train=True)
        msg = (f"epoch {ep_i}/{args.epochs} {time.time()-t0:.0f}s "
               f"train k={trm['known_acc']:.3f} "
               f"first={trm['first_acc']:.3f} "
               f"attach={trm['attach_acc']:.3f} "
               f"rho={float(adapter.rho):.2f}")
        if ep_i % args.val_every == 0 or ep_i == args.epochs:
            vam = eval_metaval(adapter, va, z_va, split, dev, args)
            msg += (f" | val k={vam['known_acc']:.3f} "
                    f"first={vam['first_acc']:.3f} "
                    f"cross={vam['cross_acc']:.3f} "
                    f"joint={vam['joint']:.4f} states={vam['n_states']}")
            if best is None or vam["joint"] > best["val"]["joint"]:
                best = {"epoch": ep_i, "val": vam, "train": trm}
                torch.save({
                    "adapter": {k: v.detach().cpu()
                                for k, v in adapter.state_dict().items()},
                    "rho": float(adapter.rho),
                    "sigma2": args.sigma2,
                    "args": vars(args),
                    "epoch": ep_i,
                    "val_metrics": vam,
                }, out / "best.pth")
        print(msg, flush=True)
        torch.save({
            "adapter": {k: v.detach().cpu()
                        for k, v in adapter.state_dict().items()},
            "rho": float(adapter.rho),
            "sigma2": args.sigma2,
            "args": vars(args),
            "epoch": ep_i,
            "train_metrics": trm,
        }, out / f"checkpoint_{ep_i:03d}.pth")
        sched.step()
    if best:
        (out / "best.json").write_text(json.dumps(best, indent=2, default=float))
        print("BEST", json.dumps(best, indent=2, default=float))


if __name__ == "__main__":
    main()
